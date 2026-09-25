"""CSV -> samples.jsonl adapter.

A flexible ingestion path for hand-curated or non-I2EBench data: author a CSV with
the required columns and convert it to a validated `SampleRecord` JSONL manifest.

Required columns:
    sample_id, source_dataset, edit_type, content_category,
    original_image_path, instruction, edit_model, edited_image_path

Optional columns (ignored if absent):
    reference_image_path, human_score, auto_score,
    has_mask, mask_path, original_width, original_height

    python -m edit_judge_bias.data.build_from_csv \
        --csv data/raw/my_samples.csv \
        --out data/manifests/samples.jsonl
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root
from edit_judge_bias.data.schema import SampleMetadata, SampleRecord
from edit_judge_bias.data.validate_manifest import validate_samples

REQUIRED_COLUMNS = (
    "sample_id",
    "source_dataset",
    "edit_type",
    "content_category",
    "original_image_path",
    "instruction",
    "edit_model",
    "edited_image_path",
)


def _opt(row: Dict[str, str], key: str):
    """Return a stripped cell value, or None if absent/blank."""
    val = row.get(key)
    if val is None:
        return None
    val = val.strip()
    return val or None


def row_to_sample(row: Dict[str, str]) -> SampleRecord:
    """Assemble one SampleRecord from a CSV row (pydantic validates the values)."""
    meta = SampleMetadata(
        has_mask=str(_opt(row, "has_mask") or "false").lower() in ("1", "true", "yes"),
        mask_path=_opt(row, "mask_path"),
        original_width=_opt(row, "original_width"),
        original_height=_opt(row, "original_height"),
    )
    return SampleRecord(
        sample_id=row["sample_id"].strip(),
        source_dataset=row["source_dataset"].strip(),
        edit_type=row["edit_type"].strip(),
        content_category=row["content_category"].strip(),
        original_image_path=row["original_image_path"].strip(),
        instruction=row["instruction"].strip(),
        edit_model=row["edit_model"].strip(),
        edited_image_path=row["edited_image_path"].strip(),
        reference_image_path=_opt(row, "reference_image_path"),
        human_score=_opt(row, "human_score"),
        auto_score=_opt(row, "auto_score"),
        metadata=meta,
    )


def rows_to_samples(rows: Iterable[Dict[str, str]]) -> List[SampleRecord]:
    return [row_to_sample(r) for r in rows]


def read_csv_rows(csv_path: PathLike) -> List[Dict[str, str]]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{csv_path}: CSV missing required columns: {missing}"
            )
        return list(reader)


def build_from_csv(
    csv_path: PathLike,
    out_path: PathLike,
    *,
    root: PathLike | None = None,
    skip_path_check: bool = False,
    dry_run: bool = False,
) -> List[SampleRecord]:
    """Convert a CSV to a samples JSONL manifest. Returns the records."""
    rows = read_csv_rows(csv_path)
    samples = rows_to_samples(rows)

    if not dry_run:
        io.write_jsonl(out_path, samples)
        if not skip_path_check:
            report = validate_samples(out_path, root if root is not None else default_root())
            if not report.ok:
                # Manifest is written (for inspection) but flag the failure loudly.
                raise FileNotFoundError(
                    f"path validation failed after writing {out_path}:\n{report.summary()}"
                )
    return samples


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build samples.jsonl from a CSV.")
    p.add_argument("--csv", type=Path, required=True, help="input CSV path")
    p.add_argument("--out", type=Path, required=True, help="output samples JSONL")
    p.add_argument("--root", type=Path, default=None, help="root for path validation")
    p.add_argument("--skip-path-check", action="store_true", help="do not verify image paths exist")
    p.add_argument("--dry-run", action="store_true", help="parse + report counts, write nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    samples = build_from_csv(
        args.csv,
        args.out,
        root=args.root,
        skip_path_check=args.skip_path_check,
        dry_run=args.dry_run,
    )
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(samples)} samples -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
