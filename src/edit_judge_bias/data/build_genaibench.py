"""GenAI-Bench / GenAI-Arena `image_edition` -> Sample + Pair manifests (full-version source C).

Why this source is special: every row is a **real human A/B vote** from the
GenAI-Arena leaderboard (NeurIPS 2024 D&B, arXiv:2406.04485,
`TIGER-Lab/GenAI-Bench`, CC-BY-4.0). I2EBench gives us instructions and model
outputs but no human preference; here we get
``(source_image, instruct_prompt, left_output, right_output, vote_type)``.

That human vote is the scientific payload: with it we can ask not merely "did an
injected bias move the judge's verdict" but "did it push the judge **away from
the human preference**". So the a/b orientation is load-bearing —
`left_* -> sample_id_a/edit_model_a/edited_image_a_path` and `right_* -> ..._b`,
and `ground_truth_preference` is "a" for a leftvote, "b" for a rightvote.
`tests/test_build_genaibench.py` pins that orientation.

Shape of the source (verified against the parquet, 2026-07):

- config ``image_edition`` has two shards, ``test`` (983 rows) and ``test_v1``
  (919 rows); v1 is ~99% a subset of test, so the default config reads ``test``
  only. Battle keying is content-based, so adding v1 would de-duplicate anyway.
- HF `Image` columns encode as ``struct<bytes: binary, path: string>``.
- ``vote_type`` vocabulary: ``leftvote`` / ``rightvote`` / ``tievote`` /
  ``bothbad_vote``. Only the first three are usable preferences; ``bothbad_vote``
  (~41% of rows) says *both* outputs failed, which is not an A-vs-B preference,
  so those rows are dropped (counted, never silently).
- 983 rows collapse to 560 usable **battles**: 17 battles were voted more than
  once. Repeated votes are aggregated (majority; a↔b deadlock -> "tie") and the
  signed vote margin ``(n_a - n_b) / n_votes`` is stored in
  ``PairRecord.pair_quality_gap`` as the human-derived quality gap.
- No edit-type / content-category labels ship with the data, so
  `edit_type_rules.classify_*` derives them from the instruction; unmatched rows
  fall back to ``default_edit_type`` and the fallback count is reported.

    python -m edit_judge_bias.data.build_genaibench --config configs/data/genaibench.yaml --dry-run
    python -m edit_judge_bias.data.build_genaibench --config configs/data/genaibench.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import io as _io
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.edit_type_rules import (
    classify_content_category,
    classify_edit_type,
)
from edit_judge_bias.data.manifest_utils import PathLike, default_root, slug, to_rel_posix
from edit_judge_bias.data.schema import PairRecord, SampleMetadata, SampleRecord
from edit_judge_bias.data.validate_manifest import validate_pairs, validate_samples

# GenAI-Arena vote_type -> PairRecord.ground_truth_preference.
# `bothbad_vote` maps to None: "both outputs are bad" is not an A-vs-B
# preference, so such a row carries no usable ground truth (§ dropped, counted).
DEFAULT_VOTE_MAP: Dict[str, Optional[str]] = {
    "leftvote": "a",
    "rightvote": "b",
    "tievote": "tie",
    "bothbad_vote": None,
}

DEFAULT_COLUMNS: Dict[str, str] = {
    "source_prompt": "source_prompt",
    "target_prompt": "target_prompt",
    "instruction": "instruct_prompt",
    "source_image": "source_image",
    "left_model": "left_model",
    "left_image": "left_output_image",
    "right_model": "right_model",
    "right_image": "right_output_image",
    "vote": "vote_type",
}


@dataclass
class GenAIBenchBuildStats:
    """Everything a reviewer needs to audit what was kept and what was dropped."""

    n_shards: int = 0
    n_rows: int = 0
    vote_counts: Counter = field(default_factory=Counter)
    n_unusable_vote: int = 0
    n_missing_field: int = 0
    n_battles: int = 0
    n_multi_vote_battles: int = 0
    n_conflicting_vote_battles: int = 0
    n_selected: int = 0
    n_pairs: int = 0
    n_samples: int = 0
    edit_types: Counter = field(default_factory=Counter)
    content_categories: Counter = field(default_factory=Counter)
    model_counts: Counter = field(default_factory=Counter)
    preferences: Counter = field(default_factory=Counter)
    n_rule_matched: int = 0
    n_edit_type_fallback: int = 0
    dropped_by_balance: Counter = field(default_factory=Counter)
    # Instructions no edit_type rule matched (they used `default_edit_type`).
    unclassified: List[str] = field(default_factory=list)
    images_written: int = 0
    images_reused: int = 0
    bytes_written: int = 0
    id_collisions: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def rule_coverage(self) -> float:
        n = self.n_rule_matched + self.n_edit_type_fallback
        return self.n_rule_matched / n if n else 0.0

    def summary(self) -> str:
        return (
            f"shards={self.n_shards} rows={self.n_rows} "
            f"unusable_vote={self.n_unusable_vote} missing_field={self.n_missing_field} "
            f"battles={self.n_battles} (multi_vote={self.n_multi_vote_battles}, "
            f"conflicting={self.n_conflicting_vote_battles}) selected={self.n_selected} "
            f"pairs={self.n_pairs} samples={self.n_samples} "
            f"rule_coverage={self.rule_coverage:.1%} "
            f"edit_type_fallback={self.n_edit_type_fallback} "
            f"images_written={self.images_written} images_reused={self.images_reused} "
            f"MB={self.bytes_written / 1e6:.1f} failures={len(self.failures)}"
        )


def load_config(config_path: PathLike) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _columns(cfg: dict) -> Dict[str, str]:
    cols = dict(DEFAULT_COLUMNS)
    cols.update(cfg.get("columns") or {})
    return cols


def _vote_map(cfg: dict) -> Dict[str, Optional[str]]:
    vm = dict(DEFAULT_VOTE_MAP)
    vm.update(cfg.get("vote_map") or {})
    return vm


def map_vote(vote: Any, vote_map: Optional[Dict[str, Optional[str]]] = None) -> Optional[str]:
    """`vote_type` -> "a" / "b" / "tie", or None when the vote is not a preference.

    None means *unusable* (unknown label, or an explicit "both bad" verdict) and
    the caller must drop the row rather than invent a preference.
    """
    vm = DEFAULT_VOTE_MAP if vote_map is None else vote_map
    if vote is None:
        return None
    return vm.get(str(vote).strip())


# --------------------------------------------------------------------------- #
# Shard resolution / download                                                 #
# --------------------------------------------------------------------------- #
def resolve_shards(cfg: dict) -> List[str]:
    """Which parquet files to read (explicit list wins; else prefix-filtered)."""
    hf = cfg["hf"]
    explicit = hf.get("files") or hf.get("shards") or []
    if explicit:
        return list(explicit)

    from huggingface_hub import HfApi

    files = sorted(
        f
        for f in HfApi().list_repo_files(hf["repo_id"], repo_type="dataset",
                                         revision=hf.get("revision"))
        if f.endswith(".parquet")
    )
    prefix = hf.get("shard_prefix")
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    n = hf.get("n_shards")
    return files[: int(n)] if n else files


def download_shards(cfg: dict, shards: List[str], root: Path) -> List[Path]:
    """Resolve each shard to a local parquet file, downloading only when needed.

    An entry that already names an existing file on disk is used as-is, so a
    pre-downloaded (or synthetic) shard needs no network at all.
    """
    hf = cfg["hf"]
    cache_dir = root / hf.get("cache_dir", "tmp_data/hf_cache")
    out: List[Path] = []
    for fn in shards:
        for candidate in (Path(fn), root / fn):
            if candidate.is_file():
                out.append(candidate)
                break
        else:
            from huggingface_hub import hf_hub_download

            cache_dir.mkdir(parents=True, exist_ok=True)
            out.append(
                Path(
                    hf_hub_download(
                        repo_id=hf["repo_id"],
                        repo_type="dataset",
                        filename=fn,
                        revision=hf.get("revision"),
                        cache_dir=str(cache_dir),
                    )
                )
            )
    return out


def _http_shards(cfg: dict, shards: List[str]) -> List[Any]:
    """Open shards over HTTP so a --dry-run downloads no image bytes.

    Parquet is columnar: a text-column-only read pulls kilobytes, not ~104 MB.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    repo_id = cfg["hf"]["repo_id"]
    rev = cfg["hf"].get("revision")
    repo = f"{repo_id}@{rev}" if rev else repo_id
    return [fs.open(f"datasets/{repo}/{fn}", "rb") for fn in shards]


def iter_rows(shard_paths: List[Any], columns: List[str], batch_size: int = 16) -> Iterator[dict]:
    """Stream rows as dicts from parquet shards, reading only `columns`."""
    import pyarrow.parquet as pq

    for p in shard_paths:
        pf = pq.ParquetFile(p)
        present = [c for c in columns if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=present):
            for row in batch.to_pylist():
                yield row


# --------------------------------------------------------------------------- #
# Image bytes                                                                 #
# --------------------------------------------------------------------------- #
def image_bytes(cell: Any) -> Optional[bytes]:
    """Pull raw bytes out of an HF `Image` cell (``struct<bytes, path>``)."""
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b:
            return bytes(b)
        # Some exports carry only a local path (no embedded bytes).
        p = cell.get("path")
        if p and Path(p).is_file():
            return Path(p).read_bytes()
    return None


def content_stem(raw: bytes, length: int = 12) -> str:
    """Content-addressed stem, so identical images de-duplicate on disk.

    GenAI-Arena re-embeds the same source image under a fresh uuid on every
    battle (983 rows carry only 180 distinct source images), and an output can be
    replayed against several opponents. Hashing the bytes collapses all of that.
    """
    return hashlib.sha1(raw).hexdigest()[:length]


def _write_image(
    raw: bytes, out_dir: Path, stem: str, stats: GenAIBenchBuildStats
) -> Tuple[Path, int, int]:
    """Write image bytes verbatim under their native extension; resumable."""
    from PIL import Image

    with Image.open(_io.BytesIO(raw)) as im:
        ext = (im.format or "PNG").lower()
        ext = {"jpeg": "jpg"}.get(ext, ext)
        w, h = im.size

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.{ext}"
    if path.exists() and path.stat().st_size == len(raw):
        stats.images_reused += 1
    else:
        path.write_bytes(raw)
        stats.images_written += 1
        stats.bytes_written += len(raw)
    return path, w, h


# --------------------------------------------------------------------------- #
# Battles: rows -> one aggregated human verdict per A/B match-up              #
# --------------------------------------------------------------------------- #
def battle_key(row: dict, cols: Dict[str, str]) -> Tuple[str, str, str, str]:
    """Identity of an A/B match-up: (source_prompt, instruction, left, right).

    Verified equivalent to keying on the two output-image content hashes (560
    groups either way, and no group maps to more than one image pair), but it
    needs no image bytes — so `--dry-run` planning matches the real run exactly.
    """
    return (
        str(row.get(cols["source_prompt"]) or ""),
        str(row.get(cols["instruction"]) or ""),
        str(row.get(cols["left_model"]) or ""),
        str(row.get(cols["right_model"]) or ""),
    )


def aggregate_preference(votes: List[str]) -> Tuple[str, float]:
    """Combine repeated human votes into (preference, signed margin).

    Majority of "a" vs "b"; an a-vs-b deadlock, or "tie" outnumbering both, is a
    "tie". Margin = (n_a - n_b) / n_votes in [-1, 1] — positive means A preferred.
    """
    c = Counter(votes)
    n_a, n_b, n_tie = c.get("a", 0), c.get("b", 0), c.get("tie", 0)
    total = n_a + n_b + n_tie
    if n_a > n_b:
        pref = "a"
    elif n_b > n_a:
        pref = "b"
    else:
        pref = "tie"
    if pref != "tie" and n_tie > max(n_a, n_b):
        pref = "tie"
    margin = (n_a - n_b) / total if total else 0.0
    return pref, margin


def plan_battles(
    shard_paths: List[Any], cfg: dict, stats: GenAIBenchBuildStats
) -> List[dict]:
    """Pass 1 (text columns only): usable rows -> one planned battle per match-up."""
    cols = _columns(cfg)
    vote_map = _vote_map(cfg)
    default_et = cfg.get("default_edit_type")
    text_cols = [
        cols["source_prompt"],
        cols["target_prompt"],
        cols["instruction"],
        cols["left_model"],
        cols["right_model"],
        cols["vote"],
    ]

    grouped: Dict[Tuple[str, str, str, str], dict] = {}
    order: List[Tuple[str, str, str, str]] = []
    for row in iter_rows(shard_paths, text_cols):
        stats.n_rows += 1
        raw_vote = row.get(cols["vote"])
        stats.vote_counts[str(raw_vote)] += 1

        instruction = (row.get(cols["instruction"]) or "").strip()
        if not instruction or not row.get(cols["left_model"]) or not row.get(cols["right_model"]):
            stats.n_missing_field += 1
            continue

        pref = map_vote(raw_vote, vote_map)
        if pref is None:
            stats.n_unusable_vote += 1
            continue

        key = battle_key(row, cols)
        battle = grouped.get(key)
        if battle is None:
            battle = {
                "_key": key,
                "instruction": instruction,
                "source_prompt": str(row.get(cols["source_prompt"]) or ""),
                "target_prompt": str(row.get(cols["target_prompt"]) or ""),
                "left_model": str(row[cols["left_model"]]),
                "right_model": str(row[cols["right_model"]]),
                "votes": [],
                "raw_votes": [],
            }
            grouped[key] = battle
            order.append(key)
        battle["votes"].append(pref)
        battle["raw_votes"].append(str(raw_vote))

    battles: List[dict] = []
    for key in order:
        b = grouped[key]
        if len(b["votes"]) > 1:
            stats.n_multi_vote_battles += 1
            if len(set(b["votes"])) > 1:
                stats.n_conflicting_vote_battles += 1
        pref, margin = aggregate_preference(b["votes"])
        b["preference"] = pref
        b["vote_margin"] = margin
        b["n_votes"] = len(b["votes"])

        et = classify_edit_type(b["instruction"])
        if et is None:
            stats.n_edit_type_fallback += 1
            stats.unclassified.append(b["instruction"])
            b["_edit_type_rule_matched"] = False
            et = default_et
            if et is None:  # no fallback configured -> drop, but say so
                stats.failures.append(f"unclassified edit_type: {b['instruction']!r}")
                continue
        else:
            stats.n_rule_matched += 1
            b["_edit_type_rule_matched"] = True
        b["_edit_type"] = et
        b["_content_category"] = classify_content_category(
            b["instruction"], b["source_prompt"], b["target_prompt"]
        )
        battles.append(b)

    stats.n_battles = len(battles)
    return battles


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #
def select_balanced(battles: List[dict], cfg: dict, stats: GenAIBenchBuildStats) -> List[dict]:
    """Seeded balanced subsample: up to `per_edit_type` battles per edit_type.

    Same contract as `build_from_hf.select_balanced` / `build_i2ebench.select_pilot`
    — sorted base order then a seeded shuffle — so raising `per_edit_type` yields a
    superset and already-injected/judged work stays valid. Everything dropped is
    counted per edit_type in `stats.dropped_by_balance` (never a silent truncation).
    """
    sampling = cfg.get("sampling") or {}
    per_type = sampling.get("per_edit_type")
    max_total = sampling.get("max_total")
    rng = random.Random(int(sampling.get("seed", 42)))

    by_type: Dict[str, List[dict]] = defaultdict(list)
    for b in battles:
        by_type[b["_edit_type"]].append(b)

    selected: List[dict] = []
    for et in sorted(by_type):
        rows = sorted(by_type[et], key=lambda r: r["_key"])
        rng.shuffle(rows)
        keep = rows[: int(per_type)] if per_type else rows
        if len(rows) > len(keep):
            stats.dropped_by_balance[et] += len(rows) - len(keep)
        selected.extend(keep)

    # Restore the deterministic source order for a stable manifest.
    selected.sort(key=lambda r: r["_key"])
    if max_total and len(selected) > int(max_total):
        stats.dropped_by_balance["_max_total"] += len(selected) - int(max_total)
        selected = selected[: int(max_total)]
    return selected


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
def _sample_id(prefix: str, stem: str, model: str, discriminator: str) -> str:
    return f"{prefix}_{stem}_{slug(model)}_{discriminator}"


def build(
    config_path: PathLike,
    *,
    root: PathLike | None = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_path_check: bool = False,
) -> Tuple[List[SampleRecord], List[PairRecord], GenAIBenchBuildStats]:
    cfg = load_config(config_path)
    root = Path(root) if root is not None else default_root()
    cols = _columns(cfg)
    stats = GenAIBenchBuildStats()

    shards = resolve_shards(cfg)
    stats.n_shards = len(shards)

    if dry_run and cfg["hf"].get("plan_over_http", True):
        plan_paths: List[Any] = _http_shards(cfg, shards)
    else:
        plan_paths = list(download_shards(cfg, shards, root))

    battles = plan_battles(plan_paths, cfg, stats)
    selected = select_balanced(battles, cfg, stats)
    if limit:
        selected = selected[:limit]
    stats.n_selected = len(selected)
    if dry_run:
        for b in selected:
            stats.edit_types[b["_edit_type"]] += 1
            stats.content_categories[b["_content_category"]] += 1
            stats.preferences[b["preference"]] += 1
            stats.model_counts[b["left_model"]] += 1
            stats.model_counts[b["right_model"]] += 1
        return [], [], stats

    # Pass 2 — re-read with the image columns, materializing only selected rows.
    wanted = {b["_key"]: b for b in selected}
    shard_paths = download_shards(cfg, shards, root)
    imgs = cfg["images"]
    prefix = cfg.get("id_prefix", "gb")
    orig_dir = root / imgs["original_dir"]
    edited_dir = root / imgs["edited_dir"]

    samples: Dict[str, SampleRecord] = {}
    sample_key_to_id: Dict[Tuple[str, str, str, str], str] = {}
    pairs: List[PairRecord] = []
    seen_pairs: set = set()

    all_cols = list(dict.fromkeys(list(cols.values())))
    for row in iter_rows(shard_paths, all_cols, batch_size=8):
        key = battle_key(row, cols)
        battle = wanted.get(key)
        if battle is None or key in seen_pairs:
            continue

        try:
            src_raw = image_bytes(row.get(cols["source_image"]))
            left_raw = image_bytes(row.get(cols["left_image"]))
            right_raw = image_bytes(row.get(cols["right_image"]))
            if not src_raw or not left_raw or not right_raw:
                stats.n_missing_field += 1
                stats.failures.append(f"missing image bytes: {key}")
                continue

            src_stem = content_stem(src_raw)
            src_path, w, h = _write_image(src_raw, orig_dir, src_stem, stats)
            src_rel = to_rel_posix(src_path, root)

            side_ids: List[str] = []
            side_rels: List[str] = []
            for raw, model in (
                (left_raw, battle["left_model"]),
                (right_raw, battle["right_model"]),
            ):
                stem = content_stem(raw)
                # A given output image can be replayed against several opponents;
                # the (image, model, instruction, source) tuple is its identity.
                skey = (stem, model, battle["instruction"], src_stem)
                sid = sample_key_to_id.get(skey)
                if sid is None:
                    disc = hashlib.sha1(
                        "|".join(skey).encode("utf-8")
                    ).hexdigest()[:6]
                    sid = _sample_id(prefix, stem, model, disc)
                    if sid in samples:  # pragma: no cover - sha1 collision
                        stats.id_collisions += 1
                    sample_key_to_id[skey] = sid
                    path, _, _ = _write_image(raw, edited_dir / slug(model), stem, stats)
                    samples[sid] = SampleRecord(
                        sample_id=sid,
                        source_dataset=cfg["source_dataset"],
                        edit_type=battle["_edit_type"],
                        content_category=battle["_content_category"],
                        original_image_path=src_rel,
                        instruction=battle["instruction"],
                        edit_model=model,
                        edited_image_path=to_rel_posix(path, root),
                        metadata=SampleMetadata(
                            original_width=w,
                            original_height=h,
                            edit_type_rule_matched=battle["_edit_type_rule_matched"],
                            source_prompt=battle["source_prompt"],
                            target_prompt=battle["target_prompt"],
                        ),
                    )
                    stats.edit_types[battle["_edit_type"]] += 1
                    stats.content_categories[battle["_content_category"]] += 1
                    stats.model_counts[model] += 1
                side_ids.append(sid)
                side_rels.append(samples[sid].edited_image_path.as_posix())

            pair_id = f"{prefix}p_{content_stem(left_raw)}_{content_stem(right_raw)}"
            pairs.append(
                PairRecord(
                    pair_id=pair_id,
                    sample_id_a=side_ids[0],
                    sample_id_b=side_ids[1],
                    original_image_path=src_rel,
                    instruction=battle["instruction"],
                    edited_image_a_path=side_rels[0],
                    edited_image_b_path=side_rels[1],
                    edit_model_a=battle["left_model"],
                    edit_model_b=battle["right_model"],
                    ground_truth_preference=battle["preference"],
                    pair_quality_gap=battle["vote_margin"],
                    source_dataset=cfg["source_dataset"],
                    edit_type=battle["_edit_type"],
                )
            )
            stats.preferences[battle["preference"]] += 1
            seen_pairs.add(key)
        except Exception as exc:  # noqa: BLE001 - log the item, never abort the batch
            stats.failures.append(f"{key}: {type(exc).__name__}: {exc}")

    missing_battles = set(wanted) - seen_pairs
    if missing_battles:
        stats.failures.extend(f"battle never materialized: {k}" for k in sorted(missing_battles)[:20])

    sample_list = sorted(samples.values(), key=lambda s: s.sample_id)
    stats.n_samples = len(sample_list)
    stats.n_pairs = len(pairs)

    samples_out = root / cfg["output"]["samples"]
    pairs_out = root / cfg["output"]["pairs"]
    io.write_jsonl(samples_out, sample_list)
    io.write_jsonl(pairs_out, pairs)

    if not skip_path_check:
        for out, validator, label in (
            (samples_out, validate_samples, "samples"),
            (pairs_out, validate_pairs, "pairs"),
        ):
            report = validator(out, root)
            if not report.ok:
                raise FileNotFoundError(
                    f"{label} path validation failed after writing {out}:\n{report.summary()}"
                )
    return sample_list, pairs, stats


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build GenAI-Bench sample + pair manifests (human A/B votes)."
    )
    p.add_argument("--config", type=Path, required=True, help="data config YAML")
    p.add_argument("--root", type=Path, default=None, help="project root (default: CWD)")
    p.add_argument("--dry-run", action="store_true", help="plan over HTTP, write nothing")
    p.add_argument("--limit", type=int, default=None, help="cap the number of pairs")
    p.add_argument("--skip-path-check", action="store_true")
    p.add_argument(
        "--show-unclassified",
        action="store_true",
        help="print every instruction that fell back to default_edit_type",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    samples, pairs, stats = build(
        args.config,
        root=args.root,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_path_check=args.skip_path_check,
    )
    cfg = load_config(args.config)
    print(f"[genaibench] {stats.summary()}")
    print(f"  vote_type counts: {dict(stats.vote_counts)}")
    print(f"  preference counts: {dict(stats.preferences)}")
    print(f"  edit_types: {dict(stats.edit_types)}")
    print(f"  content_categories: {dict(stats.content_categories)}")
    print(f"  models: {dict(stats.model_counts)}")
    if stats.dropped_by_balance:
        print(f"  dropped by balancing: {dict(stats.dropped_by_balance)}")
    if stats.rule_coverage and stats.rule_coverage < 0.85:
        print(
            f"  WARNING: edit_type rule coverage {stats.rule_coverage:.1%} < 85% — "
            f"{stats.n_edit_type_fallback} instructions fell back to "
            f"default_edit_type={cfg.get('default_edit_type')!r}"
        )
    if args.show_unclassified and stats.unclassified:
        print(f"  --- {len(set(stats.unclassified))} distinct unclassified instructions ---")
        for instr in sorted(set(stats.unclassified)):
            print(f"    ? {instr}")
    if stats.failures:
        print(f"  failures ({len(stats.failures)}): {stats.failures[:5]}")
    verb = "would write" if args.dry_run else "wrote"
    n_pairs = stats.n_selected if args.dry_run else len(pairs)
    print(f"{verb} pairs={n_pairs} -> {cfg['output']['pairs']}")
    print(f"{verb} samples={len(samples)} -> {cfg['output']['samples']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
