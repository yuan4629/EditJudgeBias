"""Combine the per-source full-version manifests into one pool + a judging subset.

Each source builder (``build_i2ebench``, ``build_imagenhub``, ``build_genaibench``,
``build_from_hf``) writes its own ``samples_*.jsonl`` / ``pairs_*.jsonl``. This
module concatenates them, enforces the cross-source invariants, and draws the
subset the judges actually run on.

Two things it does that a ``cat`` cannot:

**Normalises edit_type trust.** The three MagicBrush-derived sources have no
``edit_type`` label, so each derives one from :mod:`edit_type_rules` and falls back
to a configured default when no rule matches -- but each records that fact under a
different key (``edit_type_rule_matched`` / ``edit_type_rule_fallback`` /
``edit_type_source``). Left alone, a §9.4 breakdown would silently mix rule-derived
strata with default-guessed ones. Every combined record therefore gets a single
``metadata.edit_type_trusted`` boolean, and the judging subset can require it.

**Draws the judging subset by seeded shuffle-then-take-first-k**, stratified by
(source_dataset, edit_type) -- the same scheme ``build_i2ebench.select_pilot``
uses, and for the same reason: growing ``k`` yields a strict *superset* of the
smaller draw, so images already injected and judge calls already paid for stay
valid when the budget grows. The full pool is far too large to judge outright
(~3.8k samples x 11 conditions x 5 judges), so this subset is what bounds spend.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PIL import Image

from .io import read_jsonl, write_jsonl
from .manifest_utils import default_root
from .schema import PairRecord, SampleRecord

# Keys the individual builders use to flag "this edit_type came from a rule match
# rather than a configured default". Normalised into `edit_type_trusted`.
_TRUST_TRUE_KEYS = ("edit_type_rule_matched",)
_TRUST_FALSE_KEYS = ("edit_type_rule_fallback",)
_TRUST_SOURCE_KEY = "edit_type_source"
_TRUSTED_SOURCES = {"rules", "column", "label"}

# Editors that reach us under different spellings from different sources: the
# museum/arena manifests use CamelCase, I2EBench lowercase. `edit_model` keeps the
# source's own spelling (it is provenance -- the same architecture run by another
# team with other weights is NOT the same output), and `editor_family` carries the
# normalised name so per-model clustering can group them when it wants to.
_FAMILY_ALIASES = {
    "instructpix2pix": "instructpix2pix",
    "magicbrush": "magicbrush",
    "pix2pixzero": "pix2pixzero",
    "prompt2prompt": "prompt2prompt",
    "cycklediffusion": "cyclediffusion",
}


def editor_family(edit_model: str) -> str:
    """Case/punctuation-normalised editor name — a grouping key, not a merge."""
    key = "".join(ch for ch in (edit_model or "").lower() if ch.isalnum())
    return _FAMILY_ALIASES.get(key, key)


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def edit_type_is_trusted(sample: SampleRecord) -> bool:
    """Was this record's edit_type derived from data, or guessed by a default?

    Sources that ship an authoritative label (I2EBench) carry no flag at all, and
    are trusted -- absence of a fallback marker means nothing fell back.
    """
    extra = sample.metadata.model_extra or {}
    for key in _TRUST_TRUE_KEYS:
        if key in extra:
            return bool(extra[key])
    for key in _TRUST_FALSE_KEYS:
        if key in extra:
            return not bool(extra[key])
    if _TRUST_SOURCE_KEY in extra:
        return str(extra[_TRUST_SOURCE_KEY]) in _TRUSTED_SOURCES
    return True


@dataclass
class CombineStats:
    per_source_samples: Counter = field(default_factory=Counter)
    per_source_pairs: Counter = field(default_factory=Counter)
    edit_types: Counter = field(default_factory=Counter)
    content_categories: Counter = field(default_factory=Counter)
    edit_models: Counter = field(default_factory=Counter)
    n_trusted: int = 0
    n_human_scored: int = 0
    n_pairs_with_gt: int = 0
    duplicate_sample_ids: List[str] = field(default_factory=list)
    duplicate_pair_ids: List[str] = field(default_factory=list)
    dangling_pair_refs: List[str] = field(default_factory=list)
    subset_samples: Counter = field(default_factory=Counter)
    subset_pairs: Counter = field(default_factory=Counter)
    subset_strata_short: List[str] = field(default_factory=list)
    hblock_turns: int = 0
    hblock_samples: int = 0
    hblock_added: int = 0
    hblock_pairs_derivable: int = 0
    hblock_by_source: Counter = field(default_factory=Counter)
    subset_pair_tiers: Counter = field(default_factory=Counter)
    editor_families: Counter = field(default_factory=Counter)
    sizes_stamped: int = 0
    sizes_unreadable: List[str] = field(default_factory=list)
    image_sizes: Counter = field(default_factory=Counter)

    @property
    def n_samples(self) -> int:
        return sum(self.per_source_samples.values())

    @property
    def n_pairs(self) -> int:
        return sum(self.per_source_pairs.values())

    def summary(self) -> str:
        lines = [
            f"[full] samples={self.n_samples} pairs={self.n_pairs} "
            f"edit_type_trusted={self.n_trusted}/{self.n_samples} "
            f"({self.n_trusted / self.n_samples:.1%}) "
            f"human_scored={self.n_human_scored} pairs_with_gt={self.n_pairs_with_gt}"
            if self.n_samples
            else "[full] empty"
        ]
        lines.append(f"  samples by source: {dict(self.per_source_samples)}")
        lines.append(f"  pairs by source:   {dict(self.per_source_pairs)}")
        lines.append(f"  edit_type:         {dict(self.edit_types)}")
        lines.append(f"  content_category:  {dict(self.content_categories)}")
        lines.append(
            f"  edit_models:       {len(self.edit_models)} raw labels -> "
            f"{len(self.editor_families)} editor_family"
        )
        if self.sizes_stamped or self.sizes_unreadable:
            lines.append(
                f"  image sizes:       stamped={self.sizes_stamped} "
                f"unreadable={len(self.sizes_unreadable)} "
                f"top={dict(self.image_sizes.most_common(4))}"
            )
        if self.subset_samples:
            lines.append(
                f"  judge subset: samples={sum(self.subset_samples.values())} "
                f"pairs={sum(self.subset_pairs.values())} "
                f"by source={dict(self.subset_samples)}"
            )
        if self.hblock_samples:
            lines.append(
                f"  human-anchor blocks: turns={self.hblock_turns} "
                f"samples={self.hblock_samples} (new={self.hblock_added}) "
                f"derivable_pairs={self.hblock_pairs_derivable} "
                f"by source={dict(self.hblock_by_source)}"
            )
        if self.subset_pair_tiers:
            lines.append(
                "  subset pair tiers (0=human GT & decisive, 1=GT but close, "
                f"2=no GT): {dict(sorted(self.subset_pair_tiers.items()))}"
            )
        if self.subset_strata_short:
            lines.append(
                f"  UNDERFILLED strata ({len(self.subset_strata_short)}): "
                f"{self.subset_strata_short[:8]}"
            )
        for label, items in (
            ("duplicate sample_id", self.duplicate_sample_ids),
            ("duplicate pair_id", self.duplicate_pair_ids),
            ("pair refs missing from samples", self.dangling_pair_refs),
        ):
            if items:
                lines.append(f"  ERROR {label}: {len(items)} e.g. {items[:3]}")
        return "\n".join(lines)


def k_resolver(base: int, overrides: Optional[dict] = None):
    """stratum -> k, so one edit_type can be drawn deeper than the rest.

    Overrides are keyed either by `edit_type` (applies to every source) or by
    `"<source>/<edit_type>"` for one stratum; the specific key wins. Raising a
    single stratum's k is still a strict superset because `_stratified_take`
    shuffles each stratum independently — which is exactly why `low-level` can be
    deepened without disturbing a single already-drawn sample elsewhere.
    """
    overrides = overrides or {}

    def k_of(stratum) -> int:
        if isinstance(stratum, tuple) and len(stratum) == 2:
            source, edit_type = stratum
            for candidate in (f"{source}/{edit_type}", str(edit_type)):
                if candidate in overrides:
                    return int(overrides[candidate])
        return int(base)

    return k_of


def _stratified_take(
    items: Sequence,
    key,
    k,
    seed: int,
    *,
    ident,
    tier=None,
    stats: Optional[CombineStats] = None,
    label: str = "",
) -> List:
    """Seeded shuffle within each stratum, then take the first k.

    Superset property: the draw for k+1 contains the draw for k, because the
    per-stratum ordering is a pure function of (seed, stratum) and the take is a
    prefix. `tier` optionally splits a stratum into ordered preference tiers
    (shuffled independently, concatenated low-tier-first) so a scarce, more
    valuable kind of record -- e.g. a pair carrying a real human preference -- is
    consumed before the rest without breaking that prefix property.

    `k` is an int or a stratum -> int callable (see `k_resolver`).
    """
    k_of = k if callable(k) else (lambda _stratum: int(k))
    buckets: Dict[Tuple, List] = defaultdict(list)
    for item in items:
        buckets[key(item)].append(item)

    picked: List = []
    for stratum in sorted(buckets, key=str):
        members = buckets[stratum]
        want = k_of(stratum)
        by_tier: Dict[int, List] = defaultdict(list)
        for item in members:
            by_tier[0 if tier is None else int(tier(item))].append(item)
        ordered: List = []
        for t in sorted(by_tier):
            group = sorted(by_tier[t], key=ident)
            random.Random(f"{seed}:{stratum}:{t}").shuffle(group)
            ordered.extend(group)
        if stats is not None and len(ordered) < want:
            stats.subset_strata_short.append(f"{label}{stratum}={len(ordered)}/{want}")
        picked.extend(ordered[:want])
    return picked


def _turn_key(rec: SampleRecord) -> Tuple[str, str]:
    """A `turn` is one (original image, instruction) — the unit editors compete on."""
    return (rec.original_image_path.as_posix(), rec.instruction)


def human_score_sd(samples: Iterable[SampleRecord]) -> Dict[str, float]:
    """Per-source SD of `human_score` — the unit a pair's quality gap must be read in.

    Measured on this pool: EBench-18K 0.089, ImagenHub 0.251. The two label sets are
    simply not on the same footing, which is why a single absolute threshold cannot
    serve both (see :func:`decisive_tier`).
    """
    by_src: Dict[str, List[float]] = defaultdict(list)
    for s in samples:
        if s.human_score is not None:
            by_src[s.source_dataset].append(float(s.human_score))
    out: Dict[str, float] = {}
    for src, vals in by_src.items():
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        out[src] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return out


def decisive_tier(
    samples: Sequence[SampleRecord],
    pairs: Sequence[PairRecord],
    *,
    min_effect: float = 0.5,
):
    """pair -> 0 (human GT, decisive) / 1 (human GT, too close) / 2 (no human GT).

    Claim B needs pairs where the human label is confident enough that a judge
    disagreeing with it is a real error rather than label noise. The obvious rule --
    a fixed ``|gap| > 0.2`` -- silently assumed every source's human score spans the
    same range, and it does not: EBench-18K's MOS is a z-scored relative scale that
    lands in [0.225, 0.732] once rescaled, so its pair gaps have a median of 0.068
    and a *maximum* of 0.439. That threshold therefore called 4,464 of its 4,720
    pairs indecisive, leaving the judging subset with only 11 usable EBench pairs
    while ImagenHub's identically-meaningful gaps sailed through.

    The gap is instead measured in SDs of that source's own human score (Cohen's-d
    units, `min_effect` = 0.5 = a medium effect). Sources whose human label lives
    only on the pair -- GenAI-Bench ships arena vote margins, not per-sample scores
    -- fall back to the SD of their own gaps, which reproduces their previous count
    exactly (492 decisive).

    This orders the draw, it does not filter it: indecisive and unlabelled pairs are
    still eligible at tiers 1 and 2, so raising `k` still yields a strict superset.
    """
    sds = human_score_sd(samples)
    gap_sds: Dict[str, List[float]] = defaultdict(list)
    for p in pairs:
        if p.source_dataset not in sds and p.pair_quality_gap is not None:
            gap_sds[p.source_dataset].append(abs(float(p.pair_quality_gap)))
    for src, vals in gap_sds.items():
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            sds[src] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    def tier(pair: PairRecord) -> int:
        if pair.ground_truth_preference is None:
            return 2
        if pair.ground_truth_preference == "tie" or pair.pair_quality_gap is None:
            return 1
        sd = sds.get(pair.source_dataset)
        if not sd:
            return 1
        return 0 if abs(float(pair.pair_quality_gap)) >= min_effect * sd else 1

    return tier


def group_take(
    pool: Sequence[SampleRecord],
    *,
    source: Optional[str],
    turns_per_edit_type: int,
    models_per_turn: Optional[int],
    seed: int,
    require_human_score: bool = True,
    stats: Optional[CombineStats] = None,
) -> List[SampleRecord]:
    """Draw WHOLE turns, so several editors of the same item survive together.

    The stratified draw scatters its picks across turns, which is fine for scoring
    but ruinous for the human-agreement analysis: with one editor per turn there is
    nothing to derive a pairwise preference from. Measured on this pool the
    stratified draw yields 0.21 derivable pairs per paid sample; taking whole turns
    from a 17-editor source yields 8.0 — the same money buys ~38x the pairwise
    evidence. Hence a separate block rather than a bigger `k`.

    Superset property is preserved the same way as `_stratified_take`: turn keys are
    shuffled per edit_type with a seed derived from that edit_type, then prefixed.
    """
    turns: Dict[Tuple[str, str], List[SampleRecord]] = defaultdict(list)
    for rec in pool:
        if source and rec.source_dataset != source:
            continue
        turns[_turn_key(rec)].append(rec)

    eligible: Dict[Tuple[str, str], List[SampleRecord]] = {}
    for key, members in turns.items():
        if require_human_score:
            # Drop the *unrated editors*, not the whole turn. ImagenHub ran 12
            # editors per item but only rated 8 of them, so rejecting any turn
            # containing an unrated member rejected all 160 of its turns and the
            # block came out empty. EBench-18K rates every sample, so this is a
            # no-op there and its existing 48-turn anchor is unchanged.
            members = [m for m in members if m.human_score is not None]
        if len(members) < 2:  # a turn of one derives no pair — the whole point
            continue
        eligible[key] = members

    by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for key, members in eligible.items():
        by_type[str(getattr(members[0].edit_type, "value", members[0].edit_type))].append(key)

    picked: List[SampleRecord] = []
    n_turns = 0
    for edit_type in sorted(by_type):
        keys = sorted(by_type[edit_type])
        random.Random(f"{seed}:hblock:{edit_type}").shuffle(keys)
        if stats is not None and len(keys) < turns_per_edit_type:
            stats.subset_strata_short.append(
                f"hblock:{edit_type}={len(keys)}/{turns_per_edit_type}"
            )
        for key in keys[:turns_per_edit_type]:
            members = sorted(eligible[key], key=lambda r: r.sample_id)
            if models_per_turn and models_per_turn < len(members):
                # Shuffle per turn before capping, or every turn would keep the same
                # alphabetically-first editors (model00..model07 on EBench) and the
                # anchor would silently cover 8 of 17 editors instead of all of them.
                # Seeded by the turn, so raising models_per_turn stays a superset.
                random.Random(f"{seed}:hblock_models:{key}").shuffle(members)
                members = sorted(members[:models_per_turn], key=lambda r: r.sample_id)
            picked.extend(members)
            n_turns += 1
            if stats is not None:
                stats.hblock_pairs_derivable += len(members) * (len(members) - 1) // 2
    if stats is not None:
        # `+=`, not `=`: there is more than one anchor block (claim B needs a
        # cross-source replication, not a single-source finding).
        stats.hblock_turns += n_turns
        stats.hblock_samples += len(picked)
        if source:
            stats.hblock_by_source[source] += len(picked)
    return picked


def combine(
    config_path: Path,
    *,
    root: Optional[Path] = None,
    dry_run: bool = False,
    stamp_sizes: bool = True,
) -> CombineStats:
    cfg = load_config(config_path)
    root = Path(root) if root else default_root()
    stats = CombineStats()

    samples: List[SampleRecord] = []
    pairs: List[PairRecord] = []
    seen_samples: set[str] = set()
    seen_pairs: set[str] = set()

    for src in cfg.get("sources", []):
        name = src.get("name", "?")
        s_path = root / src["samples"]
        for rec in read_jsonl(s_path, SampleRecord):
            if rec.sample_id in seen_samples:
                stats.duplicate_sample_ids.append(rec.sample_id)
                continue
            seen_samples.add(rec.sample_id)
            trusted = edit_type_is_trusted(rec)
            # Stamp the normalised flag so downstream code has one key to read.
            rec.metadata.edit_type_trusted = trusted  # type: ignore[attr-defined]
            stats.n_trusted += int(trusted)

            family = editor_family(rec.edit_model)
            rec.metadata.editor_family = family  # type: ignore[attr-defined]
            stats.editor_families[family] += 1

            # Stamp the edited image's dimensions when the source builder did not.
            # Needed to test whether a bias's effect is partly a function of
            # resolution -- the overlay biases are sized as a fraction of the short
            # edge, and that claim has to be checkable against the real sizes.
            if stamp_sizes and not (rec.metadata.original_width and rec.metadata.original_height):
                try:
                    with Image.open(root / str(rec.edited_image_path)) as im:
                        rec.metadata.original_width, rec.metadata.original_height = im.size
                    stats.sizes_stamped += 1
                except Exception as exc:  # noqa: BLE001 — report, don't abort
                    stats.sizes_unreadable.append(f"{rec.sample_id}: {type(exc).__name__} {exc}")
            if rec.metadata.original_width and rec.metadata.original_height:
                stats.image_sizes[
                    f"{rec.metadata.original_width}x{rec.metadata.original_height}"
                ] += 1
            stats.per_source_samples[name] += 1
            stats.edit_types[str(getattr(rec.edit_type, "value", rec.edit_type))] += 1
            stats.content_categories[
                str(getattr(rec.content_category, "value", rec.content_category))
            ] += 1
            stats.edit_models[rec.edit_model] += 1
            stats.n_human_scored += int(rec.human_score is not None)
            samples.append(rec)

        p_rel = src.get("pairs")
        if not p_rel:
            continue
        for prec in read_jsonl(root / p_rel, PairRecord):
            if prec.pair_id in seen_pairs:
                stats.duplicate_pair_ids.append(prec.pair_id)
                continue
            seen_pairs.add(prec.pair_id)
            stats.per_source_pairs[name] += 1
            stats.n_pairs_with_gt += int(prec.ground_truth_preference is not None)
            pairs.append(prec)

    # Every pair must reference samples that survived the concat, or the pairwise
    # runner would fail mid-batch on a missing image path.
    for prec in pairs:
        for sid in (prec.sample_id_a, prec.sample_id_b):
            if sid not in seen_samples:
                stats.dangling_pair_refs.append(f"{prec.pair_id}->{sid}")

    sub_cfg = cfg.get("judge_subset") or {}
    subset_samples: List[SampleRecord] = []
    subset_pairs: List[PairRecord] = []
    if sub_cfg:
        pool = samples
        if sub_cfg.get("require_edit_type_trusted", True):
            pool = [s for s in pool if edit_type_is_trusted(s)]
        seed = int(sub_cfg.get("seed", 42))
        strat = lambda r: (  # noqa: E731
            r.source_dataset,
            str(getattr(r.edit_type, "value", r.edit_type)),
        )
        subset_samples = _stratified_take(
            pool,
            key=strat,
            k=k_resolver(
                int(sub_cfg.get("per_source_per_edit_type", 20)),
                sub_cfg.get("per_edit_type_overrides"),
            ),
            seed=seed,
            ident=lambda r: r.sample_id,
            stats=stats,
            label="samples:",
        )

        # The human-anchor block rides on top of the stratified draw rather than
        # replacing it: claim A needs breadth (many sources x edit_types), claim B
        # needs depth on single turns. Anything the stratified draw already picked
        # is kept as-is, so this only ever adds.
        # One block per human-labelled source. Claim B on a single source is a
        # single-source finding; a second block on an independent content pool with
        # a different label kind is what makes it a replication.
        hb_cfgs = sub_cfg.get("human_anchor_blocks")
        if hb_cfgs is None:
            hb_cfgs = [sub_cfg["human_anchor_block"]] if sub_cfg.get("human_anchor_block") else []
        # Frozen before any block runs: `subset_block` records WHICH MECHANISM drew
        # a sample, not merely which blocks contain it. A sample the stratified draw
        # already picked stays `breadth` even when an anchor block also picks it —
        # otherwise adding an anchor would quietly delete balanced samples from
        # claim A (measured: the ImagenHub block overlaps 30 of the stratified
        # draw's own picks). The overlap costs nothing: the anchor's 5 conditions
        # are a subset of breadth's 14 and share a `result_id`, so the runner's
        # resume logic never pays for either twice.
        breadth_ids = {s.sample_id for s in subset_samples}
        anchor_of: Dict[str, str] = {}  # sample_id -> which source anchored it
        for hb_cfg in hb_cfgs:
            hblock = group_take(
                pool,
                source=hb_cfg.get("source"),
                turns_per_edit_type=int(hb_cfg.get("turns_per_edit_type", 0)),
                models_per_turn=hb_cfg.get("models_per_turn"),
                seed=seed,
                require_human_score=bool(hb_cfg.get("require_human_score", True)),
                stats=stats,
            )
            already = {s.sample_id for s in subset_samples}
            new = [s for s in hblock if s.sample_id not in already]
            stats.hblock_added += len(new)
            subset_samples.extend(new)
            for s in hblock:
                anchor_of.setdefault(s.sample_id, str(hb_cfg.get("source") or "?"))
        if hb_cfgs:
            # Two independent keys, because they answer two different questions:
            #   subset_block  — how was it drawn? (partitions the subset; claim A's
            #                   cross-source analysis keeps `breadth` only, since the
            #                   anchor blocks are deliberately single-source and
            #                   deliberately unbalanced)
            #   anchor_source — which anchor may use it for claim B (may be set on a
            #                   `breadth` sample too; see `breadth_ids` above)
            for s in subset_samples:
                anchor_source = anchor_of.get(s.sample_id)
                s.metadata.subset_block = (  # type: ignore[attr-defined]
                    "breadth" if s.sample_id in breadth_ids else "anchor"
                )
                if anchor_source:
                    s.metadata.anchor_source = anchor_source  # type: ignore[attr-defined]

        for s in subset_samples:
            stats.subset_samples[s.source_dataset] += 1

        # Pairs are drawn independently of the scoring subset: PairRecord carries
        # its own image paths, so the pairwise runner never reads the samples
        # manifest. Requiring both sides to survive the scoring draw would throw
        # away almost every pair for no benefit.
        pair_pool = pairs
        if sub_cfg.get("pairs_require_ground_truth", False):
            pair_pool = [p for p in pair_pool if p.ground_truth_preference is not None]
        # Tier 0 = a real human preference that is decisive on that source's own
        # scale. Those are the pairs that let us claim bias moves the judge *away
        # from* human judgement, not merely that its verdict flipped, so spend on
        # them first. See `decisive_tier` for why "decisive" cannot be one constant.
        pair_tier = decisive_tier(
            samples, pairs, min_effect=float(sub_cfg.get("pairs_decisive_min_effect", 0.5))
        )
        subset_pairs = _stratified_take(
            pair_pool,
            key=strat,
            k=k_resolver(
                int(sub_cfg.get("pairs_per_source_per_edit_type", 20)),
                sub_cfg.get("pairs_per_edit_type_overrides"),
            ),
            seed=seed,
            ident=lambda r: r.pair_id,
            tier=pair_tier,
            stats=stats,
            label="pairs:",
        )
        for p in subset_pairs:
            stats.subset_pairs[p.source_dataset] += 1
            stats.subset_pair_tiers[pair_tier(p)] += 1

    if not dry_run:
        out = cfg.get("output", {})
        if out.get("samples"):
            write_jsonl(root / out["samples"], samples)
        if out.get("pairs"):
            write_jsonl(root / out["pairs"], pairs)
        if sub_cfg.get("samples") and subset_samples:
            write_jsonl(root / sub_cfg["samples"], subset_samples)
        if sub_cfg.get("pairs") and subset_pairs:
            write_jsonl(root / sub_cfg["pairs"], subset_pairs)

    return stats


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Combine per-source manifests into the full-version pool + judging subset."
    )
    ap.add_argument("--config", required=True, help="combine config YAML")
    ap.add_argument("--root", default=None, help="root that manifest paths resolve against")
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    ap.add_argument(
        "--no-stamp-sizes",
        action="store_true",
        help="skip reading image headers to fill missing metadata width/height",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    stats = combine(
        Path(args.config),
        root=Path(args.root) if args.root else None,
        dry_run=args.dry_run,
        stamp_sizes=not args.no_stamp_sizes,
    )
    print(stats.summary())
    if args.dry_run:
        print("(dry run — nothing written)")
    # Invariant violations are a build failure, not a warning: a dangling pair
    # reference would only surface much later as a mid-batch judge crash.
    if stats.duplicate_sample_ids or stats.duplicate_pair_ids or stats.dangling_pair_refs:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
