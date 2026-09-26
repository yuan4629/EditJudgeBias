"""HuggingFace parquet dataset -> SampleRecord manifest adapter (full-version M1b).

I2EBench ships as a directory tree (`build_i2ebench.py`); the other benchmark
sources we blend in ship as HF parquet shards with *embedded* images (the HF
`Image` feature encodes as ``struct<bytes: binary, path: string>``). This module
is the config-driven adapter for that shape:

    download selected shards -> read with pyarrow -> balanced seeded subsample
    -> decode image bytes onto disk -> emit a validated SampleRecord JSONL

Deliberately depends only on `huggingface_hub` + `pyarrow` + `Pillow`, not the
heavyweight `datasets` library.

Selective download matters: a source like HumanEdit is 16 GB across 34 shards but
only ~350 samples are needed, so `hf.n_shards` pulls just the first few (~0.5 GB
each). Shard choice is deterministic, never random, so a rebuild is reproducible.

Three config switches carry the awkward cases:

- `columns.sample_key` accepts a **list** of columns, joined by `key_sep`, because
  some sources have no single unique id (MagicBrush is keyed on
  ``img_id + turn_index``).
- `edit_type_from: rules` derives `edit_type` from the instruction text via
  `edit_type_rules.classify_edit_type` for sources that ship no edit-type column,
  falling back to `default_edit_type` (whose use is *counted*, never silent).
- `exclusion` drops rows that another source already covers. This is load-bearing
  for the paper's multi-source-generalization claim: ImagenHub's Text-Guided IE
  split is *derived from* MagicBrush dev, so the two manifests must be disjoint or
  "it generalizes across 3 sources" would be double-counting one source.

    python -m edit_judge_bias.data.build_from_hf --config configs/data/magicbrush_dev.yaml --dry-run
    python -m edit_judge_bias.data.build_from_hf --config configs/data/magicbrush_dev.yaml
"""

from __future__ import annotations

import argparse
import csv
import io as _io
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.edit_type_rules import (
    classify_content_category,
    classify_edit_type,
)
from edit_judge_bias.data.manifest_utils import PathLike, default_root, slug, to_rel_posix
from edit_judge_bias.data.schema import SampleMetadata, SampleRecord
from edit_judge_bias.data.validate_manifest import validate_samples


@dataclass
class HFBuildStats:
    n_shards: int = 0
    n_rows: int = 0
    n_selected: int = 0
    n_samples: int = 0
    n_excluded: int = 0
    n_exclusion_keys: int = 0
    n_edit_type_rules: int = 0
    n_edit_type_default: int = 0
    n_masks: int = 0
    skipped_unmapped_edit_type: Dict[str, int] = field(default_factory=dict)
    skipped_missing_field: List[str] = field(default_factory=list)
    unclassified_instructions: List[str] = field(default_factory=list)
    edit_types: Counter = field(default_factory=Counter)
    content_categories: Counter = field(default_factory=Counter)
    images_written: int = 0
    images_reused: int = 0

    @property
    def rule_coverage(self) -> float:
        """Fraction of rule-classified rows among those the rules were asked about."""
        asked = self.n_edit_type_rules + len(self.unclassified_instructions)
        return self.n_edit_type_rules / asked if asked else 0.0

    def summary(self) -> str:
        unmapped = sum(self.skipped_unmapped_edit_type.values())
        return (
            f"shards={self.n_shards} rows={self.n_rows} excluded={self.n_excluded} "
            f"selected={self.n_selected} samples={self.n_samples} masks={self.n_masks} "
            f"images_written={self.images_written} images_reused={self.images_reused} "
            f"rule_edit_type={self.n_edit_type_rules} "
            f"default_edit_type={self.n_edit_type_default} "
            f"rule_coverage={self.rule_coverage:.1%} "
            f"unmapped_edit_type={unmapped} "
            f"missing_field={len(self.skipped_missing_field)}"
        )


def load_config(config_path: PathLike) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Shard resolution / download                                                 #
# --------------------------------------------------------------------------- #
def resolve_shards(cfg: dict) -> List[str]:
    """Pick which parquet files in the repo to read (deterministic, no shuffle)."""
    from huggingface_hub import HfApi

    hf = cfg["hf"]
    repo_id = hf["repo_id"]
    explicit = hf.get("shards") or []
    if explicit:
        return list(explicit)

    files = sorted(
        f
        for f in HfApi().list_repo_files(repo_id, repo_type="dataset",
                                         revision=hf.get("revision"))
        if f.endswith(".parquet")
    )
    prefix = hf.get("shard_prefix")
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    n = hf.get("n_shards")
    return files[: int(n)] if n else files


def download_shards(cfg: dict, shards: List[str], root: Path) -> List[Path]:
    from huggingface_hub import hf_hub_download

    hf = cfg["hf"]
    cache_dir = root / hf.get("cache_dir", "tmp_data/hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for fn in shards:
        local = hf_hub_download(
            repo_id=hf["repo_id"],
            repo_type="dataset",
            filename=fn,
            revision=hf.get("revision"),
            cache_dir=str(cache_dir),
        )
        out.append(Path(local))
    return out


# --------------------------------------------------------------------------- #
# Composite sample keys                                                       #
# --------------------------------------------------------------------------- #
def key_columns(cfg: dict) -> List[str]:
    """Columns that together identify a row (`columns.sample_key`, str or list)."""
    sk = cfg["columns"]["sample_key"]
    return [sk] if isinstance(sk, str) else [str(c) for c in sk]


def key_sep(cfg: dict) -> str:
    return str(cfg.get("key_sep", "_"))


def row_key(row: Dict[str, Any], cols: Sequence[str], sep: str) -> Optional[str]:
    """Join the key columns of `row`; None if any part is missing/blank."""
    parts: List[str] = []
    for c in cols:
        v = row.get(c)
        if v is None or str(v).strip() == "":
            return None
        parts.append(str(v).strip())
    return sep.join(parts)


# --------------------------------------------------------------------------- #
# Exclusion list (source de-duplication)                                      #
# --------------------------------------------------------------------------- #
# ImagenHub Text-Guided IE uids look like `sample_<img_id>_<turn_index>.jpg`; the
# trailing integer is the turn, everything between the `sample_` prefix and it is
# the image id (ids are numeric today, but `.+?` keeps underscored ids working).
_IMAGENHUB_UID_RE = re.compile(r"^sample_(?P<key>.+?)_(?P<turn>\d+)$")


def _iter_exclusion_values(path: Path, column: str) -> Iterator[str]:
    """Yield the `column` value of each row; tolerates a headerless one-per-line file."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return
    header = [c.strip() for c in rows[0]]
    if column in header:
        idx = header.index(column)
        body = rows[1:]
    else:  # headerless: treat every line's first field as the value
        idx = 0
        body = rows
    for r in body:
        if len(r) > idx and r[idx].strip():
            yield r[idx].strip()


def load_exclusion_keys(cfg: dict, root: PathLike | None = None) -> Set[str]:
    """Parse `cfg["exclusion"]` into a set of composite row keys to skip.

    Formats:
      - ``imagenhub_uid`` — ``sample_<img_id>_<turn>.jpg`` -> ``<img_id><sep><turn>``
      - ``key``           — the column value is already a composite key (ext stripped)

    Raises on an unparseable uid: a silently-dropped exclusion would let a
    duplicated sample leak into a second "independent" source.
    """
    excl = cfg.get("exclusion") or {}
    if not excl or not excl.get("path"):
        return set()
    root = Path(root) if root is not None else default_root()
    path = Path(excl["path"])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"exclusion list not found: {path}")

    fmt = str(excl.get("format", "imagenhub_uid"))
    column = str(excl.get("column", "uid"))
    sep = key_sep(cfg)

    keys: Set[str] = set()
    for value in _iter_exclusion_values(path, column):
        stem = Path(value).stem if "." in Path(value).name else value
        if fmt == "imagenhub_uid":
            m = _IMAGENHUB_UID_RE.match(stem)
            if not m:
                raise ValueError(
                    f"{path}: cannot parse exclusion uid {value!r} as "
                    f"sample_<img_id>_<turn_index>"
                )
            keys.add(f"{m.group('key')}{sep}{m.group('turn')}")
        elif fmt == "key":
            keys.add(stem)
        else:
            raise ValueError(f"unknown exclusion.format: {fmt!r}")
    return keys


# --------------------------------------------------------------------------- #
# Row extraction                                                             #
# --------------------------------------------------------------------------- #
def _image_bytes(cell: Any) -> Optional[bytes]:
    """Pull raw bytes out of an HF `Image` cell (struct with bytes/path)."""
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b:
            return bytes(b)
    return None


def _dedupe(columns: Iterable[str]) -> List[str]:
    """Order-preserving de-duplication.

    Required: a duplicated name in pyarrow's ``columns=`` produces a table with
    two identically-named fields, and ``to_pylist()`` then silently keeps only
    one. Duplicates arise naturally because `columns.category_text` often *is*
    `columns.instruction`.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for c in columns:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def iter_rows(shard_paths: List[Any], columns: List[str]) -> Iterator[Dict[str, Any]]:
    """Stream rows (as dicts) from parquet shards, reading only `columns`.

    Raises if a requested column is absent from the shard: a typo'd column name
    used to be silently dropped, which surfaced much later as every row "missing
    a field".
    """
    import pyarrow.parquet as pq

    wanted = _dedupe(columns)
    for p in shard_paths:
        pf = pq.ParquetFile(p)
        names = set(pf.schema_arrow.names)
        missing = [c for c in wanted if c not in names]
        if missing:
            raise KeyError(
                f"{p}: parquet shard has no column(s) {missing}; "
                f"available: {sorted(names)}"
            )
        for batch in pf.iter_batches(batch_size=16, columns=wanted):
            for row in batch.to_pylist():
                yield row


def instruction_text(value: Any) -> str:
    """The instruction as a plain string, unwrapping a list-valued column.

    ★ Some sources ship SEVERAL paraphrases of one edit rather than a single string --
    OmniEdit's `edited_prompt_list` is `["Make it look like a cubist painting."]`. Passing that
    through `str()` would embed brackets and quotes into the instruction text, and that text is
    part of the prompt the judge is shown, so it would corrupt every downstream judgement and
    every rule-based `edit_type` classification.

    The **first** element is taken, always. Which paraphrase is used has to be deterministic:
    picking at random, or by length, would make the manifest unreproducible from the same seed
    and would silently change the judged prompt between rebuilds.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    # pyarrow list columns arrive as list/ndarray; both are sequence-like but not strings.
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        items = list(value.tolist() if hasattr(value, "tolist") else value)
        for item in items:
            text = str(item).strip()
            if text:
                return text
        return ""
    return str(value).strip()


def derive_edit_type(
    row: Dict[str, Any], cfg: dict, stats: HFBuildStats
) -> Tuple[Optional[str], str]:
    """Return ``(edit_type, source)`` where source is "rules" | "column" | "default".

    `edit_type_from: column` (default) maps a raw dataset label through
    `edit_type_map`. `edit_type_from: rules` derives it from the instruction text
    (MagicBrush/GenAI-Bench ship no label). Either way an unresolvable row falls
    back to `default_edit_type` if configured, and *that use is counted* — the
    §9.4 breakdown must be able to tell a derived label from a guessed one.
    """
    cols = cfg["columns"]
    mode = str(cfg.get("edit_type_from", "column"))

    if mode == "rules":
        instruction = instruction_text(row.get(cols["instruction"]))
        et = classify_edit_type(instruction)
        if et is not None:
            stats.n_edit_type_rules += 1
            return et, "rules"
        stats.unclassified_instructions.append(instruction)
    elif mode == "column":
        et_map = {str(k): v for k, v in (cfg.get("edit_type_map") or {}).items()}
        raw = row.get(cols["edit_type"]) if cols.get("edit_type") else None
        et = et_map.get(str(raw)) if raw is not None else None
        if et is not None:
            return et, "column"
        if cfg.get("default_edit_type") is None:
            bucket = stats.skipped_unmapped_edit_type
            bucket[str(raw)] = bucket.get(str(raw), 0) + 1
            return None, "column"
    else:
        raise ValueError(f"unknown edit_type_from: {mode!r} (want 'rules' or 'column')")

    default = cfg.get("default_edit_type")
    if default is None:
        return None, mode
    stats.n_edit_type_default += 1
    return str(default), "default"


def _content_category(row: Dict[str, Any], cfg: dict) -> str:
    """Map or infer a content_category.

    Preference order: (0) `content_category_from: rules` — the shared
    `edit_type_rules` keyword rules over instruction (+ optional caption);
    (1) an explicit column mapped through `content_category_map`; (2) per-config
    keyword rules over a caption column (`content_category_rules`) — a transparent
    heuristic, used because sources like HumanEdit ship no content-category field
    yet §9.4 needs the group key; (3) `default_content_category`.
    """
    cols = cfg["columns"]

    if str(cfg.get("content_category_from", "")) == "rules":
        texts = [
            str(row.get(c) or "")
            for c in _dedupe([cols.get("instruction"), cols.get("category_text")])
        ]
        return classify_content_category(*texts)

    col = cols.get("content_category")
    if col and row.get(col):
        mapped = (cfg.get("content_category_map") or {}).get(str(row[col]).lower())
        if mapped:
            return mapped

    rules = cfg.get("content_category_rules") or []
    text_col = cols.get("category_text") or cols.get("instruction")
    if rules and text_col and row.get(text_col):
        text = str(row[text_col]).lower()
        for rule in rules:
            for kw in rule.get("keywords", []):
                if kw.lower() in text:
                    return rule["category"]
    return cfg.get("default_content_category", "global")


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #
def select_balanced(
    candidates: List[Dict[str, Any]], cfg: dict
) -> List[Dict[str, Any]]:
    """Seeded balanced subsample: up to `per_edit_type` rows per edit_type.

    Mirrors `build_i2ebench.select_pilot`'s contract — sorted base order then a
    seeded shuffle — so growing `per_edit_type` yields a superset of the smaller
    sample and previously injected/judged work stays valid. (The optional
    `max_total` hard cap is applied last and does *not* preserve that superset
    property, so prefer tuning `per_edit_type`.)
    """
    sampling = cfg.get("sampling", {})
    per_type = sampling.get("per_edit_type")
    max_total = sampling.get("max_total")
    rng = random.Random(int(sampling.get("seed", 42)))

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_type[row["_edit_type"]].append(row)

    selected: List[Dict[str, Any]] = []
    for et in sorted(by_type):
        rows = sorted(by_type[et], key=lambda r: r["_key"])
        rng.shuffle(rows)
        selected.extend(rows[: int(per_type)] if per_type else rows)

    if max_total:
        selected = selected[: int(max_total)]
    return selected


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
def _write_image(
    raw: bytes, out_dir: Path, stem: str, stats: HFBuildStats
) -> Tuple[Path, int, int]:
    """Write image bytes verbatim under its native extension; return path + size."""
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
    return path, w, h


def build(
    config_path: PathLike,
    *,
    root: PathLike | None = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    skip_path_check: bool = False,
) -> Tuple[List[SampleRecord], HFBuildStats]:
    cfg = load_config(config_path)
    root = Path(root) if root is not None else default_root()
    cols = cfg["columns"]
    stats = HFBuildStats()

    shards = resolve_shards(cfg)
    stats.n_shards = len(shards)

    kcols = key_columns(cfg)
    ksep = key_sep(cfg)
    excluded_keys = load_exclusion_keys(cfg, root)
    stats.n_exclusion_keys = len(excluded_keys)

    # Pass 1 — cheap metadata-only scan to plan the subsample (no image bytes).
    meta_cols = _dedupe(
        [
            *kcols,
            cols.get("edit_type"),
            cols.get("instruction"),
            cols.get("content_category"),
            cols.get("category_text"),
        ]
    )
    http_handles: List[Any] = []
    if dry_run and cfg["hf"].get("plan_over_http", True):
        http_handles = _http_shards(cfg, shards)
        shard_paths: List[Any] = http_handles
    else:
        shard_paths = download_shards(cfg, shards, root)

    candidates: List[Dict[str, Any]] = []
    try:
        for row in iter_rows(shard_paths, meta_cols):
            stats.n_rows += 1
            key = row_key(row, kcols, ksep)
            instruction = instruction_text(row.get(cols["instruction"]))
            if not key or not instruction:
                stats.skipped_missing_field.append(str(key))
                continue
            if key in excluded_keys:
                stats.n_excluded += 1
                continue
            edit_type, et_source = derive_edit_type(row, cfg, stats)
            if edit_type is None:
                continue
            row["_key"] = key
            row["_edit_type"] = edit_type
            row["_edit_type_source"] = et_source
            row["_content_category"] = _content_category(row, cfg)
            candidates.append(row)
    finally:
        for fh in http_handles:
            try:
                fh.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    selected = select_balanced(candidates, cfg)
    if limit:
        selected = selected[:limit]
    stats.n_selected = len(selected)
    for r in selected:
        stats.edit_types[r["_edit_type"]] += 1
        stats.content_categories[r["_content_category"]] += 1

    if dry_run:
        return [], stats

    # Pass 2 — re-read with image columns, materializing only selected rows.
    wanted = {r["_key"]: r for r in selected}
    img_cols = _dedupe(
        [cols.get("original_image"), cols.get("edited_image"), cols.get("mask_image")]
    )
    shard_paths = download_shards(cfg, shards, root)
    imgs = cfg["images"]
    prefix = cfg.get("id_prefix") or slug(cfg["source_dataset"]).lower()
    edit_model = cfg.get("edit_model", "unknown")

    samples: List[SampleRecord] = []
    for row in iter_rows(shard_paths, meta_cols + img_cols):
        key = row_key(row, kcols, ksep)
        plan = wanted.pop(key, None) if key else None
        if plan is None:
            continue

        orig_raw = _image_bytes(row.get(cols["original_image"]))
        edit_raw = _image_bytes(row.get(cols["edited_image"]))
        if not orig_raw or not edit_raw:
            stats.skipped_missing_field.append(key)
            continue

        stem = slug(key)
        orig_path, w, h = _write_image(orig_raw, root / imgs["original_dir"], stem, stats)
        edit_path, _, _ = _write_image(edit_raw, root / imgs["edited_dir"], stem, stats)

        mask_rel = None
        mask_col = cols.get("mask_image")
        if mask_col and imgs.get("mask_dir"):
            mask_raw = _image_bytes(row.get(mask_col))
            if mask_raw:
                mask_path, _, _ = _write_image(
                    mask_raw, root / imgs["mask_dir"], stem, stats
                )
                mask_rel = to_rel_posix(mask_path, root)
                stats.n_masks += 1

        samples.append(
            SampleRecord(
                sample_id=f"{prefix}_{stem}_{slug(edit_model)}",
                source_dataset=cfg["source_dataset"],
                edit_type=plan["_edit_type"],
                content_category=plan["_content_category"],
                original_image_path=to_rel_posix(orig_path, root),
                instruction=instruction_text(row[cols["instruction"]]),
                edit_model=edit_model,
                edited_image_path=to_rel_posix(edit_path, root),
                metadata=SampleMetadata(
                    has_mask=mask_rel is not None,
                    mask_path=mask_rel,
                    original_width=w,
                    original_height=h,
                    # Provenance of the group key: "rules"/"column" is a derived
                    # label, "default" is a fallback guess. §9.4 must be able to
                    # exclude guesses, so keep it on the record.
                    edit_type_source=plan.get("_edit_type_source", "column"),
                    source_key=plan["_key"],
                ),
            )
        )

    stats.n_samples = len(samples)
    if wanted:
        stats.skipped_missing_field.extend(sorted(wanted)[:20])

    out_path = root / cfg["output"]["samples"]
    io.write_jsonl(out_path, samples)
    if not skip_path_check:
        report = validate_samples(out_path, root)
        if not report.ok:
            raise FileNotFoundError(
                f"path validation failed after writing {out_path}:\n{report.summary()}"
            )
    return samples, stats


def _http_shards(cfg: dict, shards: List[str]) -> List[Any]:
    """Open shards over HTTP (HfFileSystem) so a --dry-run downloads no images.

    Parquet is columnar, so a metadata-only column read pulls kilobytes rather
    than the ~0.5 GB shard.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    repo_id = cfg["hf"]["repo_id"]
    rev = cfg["hf"].get("revision")
    repo = f"{repo_id}@{rev}" if rev else repo_id
    return [fs.open(f"datasets/{repo}/{fn}", "rb") for fn in shards]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a samples manifest from an HF parquet dataset.")
    p.add_argument("--config", type=Path, required=True, help="data config YAML")
    p.add_argument("--root", type=Path, default=None, help="project root (default: CWD)")
    p.add_argument("--dry-run", action="store_true", help="plan over HTTP, write nothing")
    p.add_argument("--limit", type=int, default=None, help="cap the number of samples")
    p.add_argument("--skip-path-check", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    samples, stats = build(
        args.config,
        root=args.root,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_path_check=args.skip_path_check,
    )
    cfg = load_config(args.config)
    print(f"[hf:{cfg['source_dataset']}] {stats.summary()}")
    if stats.n_exclusion_keys:
        print(
            f"  de-dup: {stats.n_exclusion_keys} exclusion keys loaded, "
            f"{stats.n_excluded} rows excluded"
        )
    if stats.edit_types:
        print(f"  edit_type: {dict(sorted(stats.edit_types.items()))}")
        print(f"  content_category: {dict(sorted(stats.content_categories.items()))}")
    if stats.unclassified_instructions:
        print(
            f"  rule coverage {stats.rule_coverage:.1%}: "
            f"{len(stats.unclassified_instructions)} instructions unclassified, e.g. "
            f"{stats.unclassified_instructions[:3]}"
        )
    if stats.skipped_unmapped_edit_type:
        print(f"  unmapped edit_type values: {stats.skipped_unmapped_edit_type}")
    verb = "would write" if args.dry_run else "wrote"
    n = stats.n_selected if args.dry_run else len(samples)
    print(f"{verb} {n} samples -> {cfg['output']['samples']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
