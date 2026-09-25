"""Ingest the ImagenHub Text-Guided IE museum into SampleRecord / PairRecord manifests.

Full-version source **B** (ImagenHub, ICLR 2024, arXiv:2310.01596). ImagenHub does
not ship its per-model edited images in the HF dataset; they live in the project's
museum site repo, one directory per editing model:

    Museum/ImagenHub_Text-Guided_IE/<Model>/sample_<img_id>_<turn>.jpg   # 179 files
    Museum/ImagenHub_Text-Guided_IE/GroundTruth/...                      # human reference edit
    Museum/ImagenHub_Text-Guided_IE/input/sample_<img_id>_<turn>.jpg     # original (528 files)

`git clone` is blocked on the dev machine, so files are fetched individually over
plain HTTPS from raw.githubusercontent.com. The repo file list comes from the
GitHub tree API, cached to disk (`museum.tree_cache`) because the unauthenticated
API allows only 60 requests/hour — the cache is consulted first and the API is
called at most once per machine.

Two properties make this source scientifically valuable:

1. **17 editing models over the same 179 (image, instruction) items** — a very wide
   quality range for pairwise construction.
2. **Three human raters** scored 9 of those models on (SemanticConsistency,
   PerceptualQuality) ∈ {0, 0.5, 1}²; the mean lands on `SampleRecord.human_score`
   and gives pairs a `ground_truth_preference`. That external anchor is what lets
   us later ask whether an injected bias pushes a judge *away* from human
   preference, not merely whether its score moves.

Neither ImagenHub nor its museum ships an edit-type label, so `edit_type` /
`content_category` are derived from the instruction text with the shared rules in
`edit_type_rules.py`; unclassifiable rows fall back to `default_edit_type` and the
fallback count is reported (rule coverage on this source is only ~64% — the
instructions are MagicBrush-style paraphrases such as "let the woman cry", see
`BuildStats.unclassified`).

    python -m edit_judge_bias.data.build_imagenhub --config configs/data/imagenhub.yaml --dry-run
    python -m edit_judge_bias.data.build_imagenhub --config configs/data/imagenhub.yaml
    python -m edit_judge_bias.data.build_imagenhub --config configs/data/imagenhub.yaml --no-download
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.build_pairs import build_pairs as _build_pairs_generic
from edit_judge_bias.data.edit_type_rules import (
    classify_content_category,
    classify_edit_type,
)
from edit_judge_bias.data.manifest_utils import PathLike, default_root, slug, to_rel_posix
from edit_judge_bias.data.schema import PairRecord, SampleRecord

__all__ = [
    "BuildStats",
    "RatingTable",
    "aggregate_human_score",
    "build",
    "build_pair_records",
    "build_samples",
    "load_config",
    "load_ratings",
    "main",
    "parse_rating_cell",
    "parse_uid",
]

_UID_RE = re.compile(r"^sample_(?P<img_id>\d+)_(?P<turn>\d+)(?P<ext>\.[A-Za-z0-9]+)?$")

# Rating cells that mean "not rated" rather than "rated 0".
_EMPTY_CELLS = {"", "-", "na", "n/a", "nan", "none", "null", "[]"}


# --------------------------------------------------------------------------- #
# Stats                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class BuildStats:
    n_uids: int = 0
    n_models: int = 0
    n_samples: int = 0
    n_pairs: int = 0
    n_downloaded: int = 0
    n_skipped_cached: int = 0
    bytes_downloaded: int = 0
    n_edit_type_fallback: int = 0
    n_classified: int = 0
    n_human_scored: int = 0
    n_pairs_with_preference: int = 0
    edit_types: Counter = field(default_factory=Counter)
    content_categories: Counter = field(default_factory=Counter)
    unclassified: List[str] = field(default_factory=list)
    missing_input: List[str] = field(default_factory=list)
    missing_edited: List[str] = field(default_factory=list)
    download_failed: List[str] = field(default_factory=list)

    @property
    def rule_coverage(self) -> float:
        return self.n_classified / self.n_uids if self.n_uids else 0.0

    def summary(self) -> str:
        return (
            f"uids={self.n_uids} models={self.n_models} samples={self.n_samples} "
            f"pairs={self.n_pairs} human_scored={self.n_human_scored} "
            f"pairs_with_gt={self.n_pairs_with_preference} "
            f"rule_coverage={self.rule_coverage:.1%} "
            f"edit_type_fallback={self.n_edit_type_fallback} "
            f"downloaded={self.n_downloaded} cached={self.n_skipped_cached} "
            f"dl_failed={len(self.download_failed)} "
            f"missing_input={len(self.missing_input)} "
            f"missing_edited={len(self.missing_edited)}"
        )


def load_config(config_path: PathLike) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# uid parsing                                                                  #
# --------------------------------------------------------------------------- #
def parse_uid(uid: str) -> Tuple[str, int]:
    """`sample_100081_3.jpg` -> `("100081", 3)`.

    The museum uid is also the image filename in every model dir *and* in
    `input/`, so it is the join key for instructions, ratings and pixels. Raises
    ValueError on anything that is not `sample_<digits>_<digits>[.ext]` so a
    silently-mismatched join can never reach a manifest.
    """
    m = _UID_RE.match((uid or "").strip())
    if not m:
        raise ValueError(f"not an ImagenHub uid: {uid!r}")
    return m.group("img_id"), int(m.group("turn"))


# --------------------------------------------------------------------------- #
# Human ratings                                                                #
# --------------------------------------------------------------------------- #
def parse_rating_cell(cell: object) -> Optional[Tuple[float, float]]:
    """Parse one rater TSV cell into `(semantic_consistency, perceptual_quality)`.

    Cells are 2-element lists with values in {0, 0.5, 1}, written inconsistently
    (`[0, 0.5]` and `[0,0]` both occur). Anything blank / unparsable / not a
    2-tuple of numbers returns None ("not rated") rather than raising, so one bad
    cell cannot abort the ingest.
    """
    if cell is None:
        return None
    if isinstance(cell, (list, tuple)):
        values = list(cell)
    else:
        text = str(cell).strip()
        if text.lower() in _EMPTY_CELLS:
            return None
        try:
            values = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(values, (list, tuple)):
            return None
    if len(values) != 2:
        return None
    try:
        sc, pq = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= sc <= 1.0 and 0.0 <= pq <= 1.0):
        return None
    return sc, pq


# uid -> model -> [(sc, pq) per rater]
RatingTable = Dict[str, Dict[str, List[Tuple[float, float]]]]


def load_ratings(paths: Sequence[PathLike]) -> RatingTable:
    """Read the per-rater TSVs into `{uid: {model: [(sc, pq), ...]}}`.

    Missing files are skipped (the ratings are an optional bonus, not a hard
    dependency). Models present in the TSV but absent from the museum image dirs
    (notably `Imagic`) stay in the table and are simply never looked up.
    """
    table: RatingTable = defaultdict(lambda: defaultdict(list))
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                uid = (row.get("uid") or "").strip()
                if not uid:
                    continue
                for model, cell in row.items():
                    if model == "uid" or model is None:
                        continue
                    rating = parse_rating_cell(cell)
                    if rating is not None:
                        table[uid][model].append(rating)
    return {uid: dict(models) for uid, models in table.items()}


def aggregate_human_score(
    ratings: Sequence[Tuple[float, float]],
) -> Tuple[Optional[float], dict]:
    """Collapse per-rater `(sc, pq)` votes into one scalar in [0, 1] + detail.

    The scalar is the mean over raters of the mean of the two dimensions, i.e.
    `mean(sc) / 2 + mean(pq) / 2`. The per-dimension means and the raw votes are
    returned so a later analysis can use semantic consistency alone, or
    ImagenHub's own geometric overall (also included).
    """
    votes = [r for r in ratings if r is not None]
    if not votes:
        return None, {}
    sc = sum(v[0] for v in votes) / len(votes)
    pq = sum(v[1] for v in votes) / len(votes)
    detail = {
        "semantic_consistency": round(sc, 4),
        "perceptual_quality": round(pq, 4),
        "n_raters": len(votes),
        "rater_votes": [list(v) for v in votes],
        # ImagenHub's own "overall" is the geometric mean of the two dimensions;
        # kept for comparability with the paper's leaderboard.
        "imagenhub_overall_geometric": round((sc * pq) ** 0.5, 4),
    }
    return round((sc + pq) / 2.0, 4), detail


# --------------------------------------------------------------------------- #
# Museum file listing + download                                               #
# --------------------------------------------------------------------------- #
def museum_ref(mus: dict) -> str:
    """The museum revision to read: the pinned commit (`revision`), else `branch`."""
    return str(mus.get("revision") or mus["branch"])


def load_tree(cfg: dict, root: Path, *, allow_network: bool = True) -> Dict[str, int]:
    """Return `{repo_path: size_bytes}` for the museum subtree.

    Cached to `museum.tree_cache` on first call: the unauthenticated GitHub tree
    API allows 60 requests/hour, so it must never be hit per-build.
    """
    mus = cfg["museum"]
    cache = root / mus["tree_cache"]
    if not cache.is_file():
        if not allow_network:
            raise FileNotFoundError(
                f"tree cache {cache} missing and network disabled; run without --no-download once"
            )
        url = (
            f"https://api.github.com/repos/{mus['repo']}/git/trees/"
            f"{museum_ref(mus)}?recursive=1"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "editjudgebias/1.0",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed host
            payload = resp.read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    tree = json.loads(cache.read_text(encoding="utf-8"))
    prefix = mus["subtree"].rstrip("/") + "/"
    return {
        t["path"]: int(t.get("size") or 0)
        for t in tree.get("tree", [])
        if t.get("type") == "blob" and t["path"].startswith(prefix)
    }


def _fetch(url: str, dest: Path, *, retries: int, timeout: float) -> int:
    """Download `url` to `dest` atomically; return bytes written."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "editjudgebias/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return len(data)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:  # noqa: PERF203
            last = exc
            time.sleep(min(2 ** attempt, 8) * 0.5)
    raise RuntimeError(f"failed after {retries} attempts: {url} ({last})")


def download_museum_files(
    jobs: Sequence[Tuple[str, Path, int]],
    cfg: dict,
    stats: BuildStats,
) -> None:
    """Fetch `(repo_path, dest, expected_size)` jobs, resumable and fail-soft.

    A file already on disk with the expected size is skipped, so re-running the
    builder costs no bandwidth. Failures are logged into `stats.download_failed`
    and the batch continues (the affected samples are then dropped by the
    existence check in `build_samples`, keeping the manifest valid).
    """
    dl = cfg.get("download", {}) or {}
    workers = int(dl.get("workers", 6))
    retries = int(dl.get("retries", 3))
    timeout = float(dl.get("timeout", 60))
    base = cfg["museum"].get("raw_base", "https://raw.githubusercontent.com").rstrip("/")
    repo = cfg["museum"]["repo"]
    ref = museum_ref(cfg["museum"])

    todo: List[Tuple[str, Path]] = []
    for repo_path, dest, expected in jobs:
        if dest.is_file() and (expected <= 0 or dest.stat().st_size == expected):
            stats.n_skipped_cached += 1
            continue
        todo.append((repo_path, dest))
    if not todo:
        return

    def one(job: Tuple[str, Path]) -> Tuple[str, Optional[int]]:
        repo_path, dest = job
        url = f"{base}/{repo}/{ref}/{urllib.parse.quote(repo_path)}"
        try:
            return repo_path, _fetch(url, dest, retries=retries, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - log, never abort the batch
            print(f"  ! download failed: {repo_path}: {exc}")
            return repo_path, None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for repo_path, nbytes in pool.map(one, todo):
            if nbytes is None:
                stats.download_failed.append(repo_path)
            else:
                stats.n_downloaded += 1
                stats.bytes_downloaded += nbytes


# --------------------------------------------------------------------------- #
# Sample construction                                                          #
# --------------------------------------------------------------------------- #
def _select_uids(uids: List[str], cfg: dict) -> List[str]:
    """Optionally subsample uids (seeded) — `sampling.limit: null` keeps all 179."""
    sampling = cfg.get("sampling", {}) or {}
    limit = sampling.get("limit")
    uids = sorted(uids)
    if not limit or limit >= len(uids):
        return uids
    rng = random.Random(int(sampling.get("seed", 42)))
    picked = uids[:]
    rng.shuffle(picked)
    return sorted(picked[: int(limit)])


def build_samples(
    cfg: dict,
    root: PathLike | None = None,
    *,
    download: bool = True,
    stats: Optional[BuildStats] = None,
) -> Tuple[List[SampleRecord], BuildStats]:
    """Emit one SampleRecord per (uid, edit_model), downloading pixels as needed."""
    root = Path(root) if root is not None else default_root()
    stats = stats if stats is not None else BuildStats()

    meta_dir = root / cfg["meta_dir"]
    lookup = json.loads((meta_dir / cfg["lookup_json"]).read_text(encoding="utf-8"))
    ratings = load_ratings([meta_dir / n for n in cfg.get("rater_files", [])])

    # Join-integrity guard: the uid list and the instruction lookup must agree, or
    # instructions would silently be attached to the wrong pixels.
    csv_name = cfg.get("lookup_csv")
    if csv_name and (meta_dir / csv_name).is_file():
        with (meta_dir / csv_name).open("r", encoding="utf-8", newline="") as fh:
            csv_uids = {(r["uid"] or "").strip() for r in csv.DictReader(fh)}
        missing = csv_uids - set(lookup)
        if missing:
            raise ValueError(
                f"{csv_name} lists {len(missing)} uid(s) absent from {cfg['lookup_json']}: "
                f"{sorted(missing)[:5]}"
            )

    models: List[str] = list(cfg["edit_models"])
    reference_model: Optional[str] = cfg.get("reference_model") or None
    uids = _select_uids([u for u in lookup if _UID_RE.match(u)], cfg)
    stats.n_uids = len(uids)
    stats.n_models = len(models)

    orig_dir = root / cfg["images"]["original_dir"]
    edited_root = root / cfg["images"]["edited_dir"]
    subtree = cfg["museum"]["subtree"].rstrip("/")
    input_dir_name = cfg["museum"].get("input_dir", "input")

    # ---- plan every file we need, then fetch in one polite batch -------------
    tree = load_tree(cfg, root, allow_network=download) if download else {}

    def dest_for(model: Optional[str], uid: str) -> Path:
        return (orig_dir / uid) if model is None else (edited_root / model / uid)

    if download:
        jobs: List[Tuple[str, Path, int]] = []
        wanted = models + ([reference_model] if reference_model else [])
        for uid in uids:
            for repo_dir, model in [(input_dir_name, None)] + [(m, m) for m in wanted]:
                repo_path = f"{subtree}/{repo_dir}/{uid}"
                if repo_path not in tree:
                    continue
                jobs.append((repo_path, dest_for(model, uid), tree[repo_path]))
        download_museum_files(jobs, cfg, stats)

    # ---- classify instructions (shared rules; fallback is recorded) ----------
    default_edit_type = cfg.get("default_edit_type", "replace")

    samples: List[SampleRecord] = []
    for uid in uids:
        entry = lookup[uid]
        img_id, turn = parse_uid(uid)
        instruction = (entry.get("instruction") or "").strip()
        source_caption = entry.get("source_global_caption") or ""
        target_caption = entry.get("target_global_caption") or ""
        if not instruction:
            continue

        edit_type = classify_edit_type(instruction)
        fallback = edit_type is None
        if fallback:
            stats.n_edit_type_fallback += 1
            stats.unclassified.append(instruction)
            edit_type = default_edit_type
        else:
            stats.n_classified += 1
        content_category = classify_content_category(instruction, source_caption)
        stats.edit_types[edit_type] += 1
        stats.content_categories[content_category] += 1

        original = orig_dir / uid
        if not original.is_file():
            stats.missing_input.append(uid)
            continue
        orig_rel = to_rel_posix(original, root)

        reference_rel = None
        if reference_model:
            ref = edited_root / reference_model / uid
            if ref.is_file():
                reference_rel = to_rel_posix(ref, root)

        for model in models:
            edited = edited_root / model / uid
            if not edited.is_file():
                stats.missing_edited.append(f"{model}/{uid}")
                continue
            human_score, rating_detail = aggregate_human_score(
                ratings.get(uid, {}).get(model, [])
            )
            if human_score is not None:
                stats.n_human_scored += 1
            metadata = {
                "uid": uid,
                "imagenhub_img_id": img_id,
                "turn": turn,
                "source_global_caption": source_caption,
                "target_global_caption": target_caption,
                "edit_type_rule_fallback": fallback,
                **rating_detail,
            }
            samples.append(
                SampleRecord(
                    sample_id=f"ih_{img_id}_{turn}_{slug(model)}",
                    source_dataset=cfg.get("source_dataset", "ImagenHub"),
                    edit_type=edit_type,
                    content_category=content_category,
                    original_image_path=orig_rel,
                    instruction=instruction,
                    edit_model=model,
                    edited_image_path=to_rel_posix(edited, root),
                    reference_image_path=reference_rel,
                    human_score=human_score,
                    metadata=metadata,
                )
            )

    stats.n_samples = len(samples)
    return samples, stats


# --------------------------------------------------------------------------- #
# Pair construction                                                            #
# --------------------------------------------------------------------------- #
def build_pair_records(
    samples: List[SampleRecord],
    cfg: dict,
    stats: Optional[BuildStats] = None,
) -> List[PairRecord]:
    """Pairs across edit_models within each (original, instruction) group.

    Grouping/capping/seeding is delegated to `build_pairs.build_pairs`; this
    wrapper re-ids the pairs into the `pair_ih_*` namespace (so ImagenHub pairs
    never collide with the I2EBench ones) and attaches the human ground truth:
    where BOTH sides carry a `human_score`, `ground_truth_preference` is a/b/tie
    and `pair_quality_gap` is the absolute score difference. Those are the pairs
    that can later show a bias pushing a judge *away* from human preference.
    """
    pairs_cfg = dict(cfg.get("pairs", {}) or {})
    tie_eps = float(pairs_cfg.pop("tie_epsilon", 1e-9))
    raw = _build_pairs_generic(samples, pairs_cfg)

    by_id = {s.sample_id: s for s in samples}
    out: List[PairRecord] = []
    n_with_pref = 0
    for i, pair in enumerate(raw, start=1):
        a, b = by_id[pair.sample_id_a], by_id[pair.sample_id_b]
        preference: Optional[str] = None
        gap: Optional[float] = None
        if a.human_score is not None and b.human_score is not None:
            delta = a.human_score - b.human_score
            gap = round(abs(delta), 4)
            if delta > tie_eps:
                preference = "a"
            elif delta < -tie_eps:
                preference = "b"
            else:
                preference = "tie"
            n_with_pref += 1
        out.append(
            pair.model_copy(
                update={
                    "pair_id": f"pair_ih_{i:06d}",
                    "ground_truth_preference": preference,
                    "pair_quality_gap": gap,
                }
            )
        )
    if stats is not None:
        stats.n_pairs = len(out)
        stats.n_pairs_with_preference = n_with_pref
    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def build(
    config_path: PathLike,
    *,
    root: PathLike | None = None,
    dry_run: bool = False,
    download: bool = True,
    limit: Optional[int] = None,
) -> Tuple[List[SampleRecord], List[PairRecord], BuildStats]:
    cfg = load_config(config_path)
    if limit is not None:
        cfg.setdefault("sampling", {})["limit"] = limit
    root = Path(root) if root is not None else default_root()

    stats = BuildStats()
    samples, stats = build_samples(
        cfg, root, download=download and not dry_run, stats=stats
    )
    pairs = build_pair_records(samples, cfg, stats)

    if not dry_run:
        io.write_jsonl(root / cfg["output"]["samples"], samples)
        io.write_jsonl(root / cfg["output"]["pairs"], pairs)
    return samples, pairs, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build ImagenHub sample/pair manifests.")
    p.add_argument("--config", type=Path, required=True, help="data config YAML")
    p.add_argument("--root", type=Path, default=None, help="project root (default: CWD)")
    p.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    p.add_argument(
        "--no-download",
        action="store_true",
        help="never touch the network; build from images already on disk.",
    )
    p.add_argument("--limit", type=int, default=None, help="cap the number of uids")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    samples, pairs, stats = build(
        args.config,
        root=args.root,
        dry_run=args.dry_run,
        download=not args.no_download,
        limit=args.limit,
    )
    cfg = load_config(args.config)
    print(f"[imagenhub] {stats.summary()}")
    print(f"  downloaded {stats.bytes_downloaded / 1e6:.1f} MB")
    print(f"  edit_types={dict(stats.edit_types)}")
    print(f"  content_categories={dict(stats.content_categories)}")
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} samples={len(samples)} -> {cfg['output']['samples']}")
    print(f"{verb} pairs={len(pairs)} -> {cfg['output']['pairs']}")
    if stats.rule_coverage < 0.85 and stats.n_uids:
        print(
            f"  WARNING: edit_type rule coverage {stats.rule_coverage:.1%} < 85% — "
            f"{stats.n_edit_type_fallback} instructions fell back to "
            f"'{cfg.get('default_edit_type')}'. Unclassified examples:"
        )
        for instr in stats.unclassified[:20]:
            print(f"    - {instr}")
    if stats.download_failed:
        print(f"  note: {len(stats.download_failed)} downloads failed (re-run to retry)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
