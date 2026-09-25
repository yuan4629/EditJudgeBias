"""Validate that every image path referenced by a manifest exists on disk.

Used both as a library (builders call `validate_samples` / `validate_pairs`) and
as a CLI (Milestone 1 acceptance: "all paths exist").

    python -m edit_judge_bias.data.validate_manifest \
        --samples data/manifests/samples_pilot.jsonl \
        --pairs   data/manifests/pairs_pilot.jsonl \
        --root .
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root, resolve
from edit_judge_bias.data.schema import BiasedRecord, PairRecord, SampleRecord


@dataclass
class ValidationReport:
    n_records: int = 0
    n_paths: int = 0
    # (record_id, field_name, resolved_path) for each missing file.
    missing: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    def summary(self) -> str:
        head = (
            f"records={self.n_records} paths_checked={self.n_paths} "
            f"missing={len(self.missing)}"
        )
        if self.ok:
            return f"OK: {head}"
        lines = [f"FAIL: {head}"]
        for rid, fld, path in self.missing[:20]:
            lines.append(f"  - {rid}.{fld}: {path}")
        if len(self.missing) > 20:
            lines.append(f"  ... and {len(self.missing) - 20} more")
        return "\n".join(lines)


def _check(report: ValidationReport, rid: str, field_name: str, raw_path, root: Path) -> None:
    if raw_path is None:
        return
    report.n_paths += 1
    if not resolve(raw_path, root).is_file():
        report.missing.append((rid, field_name, str(resolve(raw_path, root))))


def validate_samples(path: PathLike, root: PathLike | None = None) -> ValidationReport:
    root = Path(root) if root is not None else default_root()
    report = ValidationReport()
    for rec in io.iter_jsonl(path, SampleRecord):
        report.n_records += 1
        _check(report, rec.sample_id, "original_image_path", rec.original_image_path, root)
        _check(report, rec.sample_id, "edited_image_path", rec.edited_image_path, root)
        _check(report, rec.sample_id, "reference_image_path", rec.reference_image_path, root)
        if rec.metadata.mask_path is not None:
            _check(report, rec.sample_id, "mask_path", rec.metadata.mask_path, root)
    return report


def validate_pairs(path: PathLike, root: PathLike | None = None) -> ValidationReport:
    root = Path(root) if root is not None else default_root()
    report = ValidationReport()
    for rec in io.iter_jsonl(path, PairRecord):
        report.n_records += 1
        _check(report, rec.pair_id, "original_image_path", rec.original_image_path, root)
        _check(report, rec.pair_id, "edited_image_a_path", rec.edited_image_a_path, root)
        _check(report, rec.pair_id, "edited_image_b_path", rec.edited_image_b_path, root)
    return report


def validate_biased(path: PathLike, root: PathLike | None = None) -> ValidationReport:
    root = Path(root) if root is not None else default_root()
    report = ValidationReport()
    for rec in io.iter_jsonl(path, BiasedRecord):
        report.n_records += 1
        _check(report, rec.biased_id, "biased_image_path", rec.biased_image_path, root)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate manifest image paths exist.")
    p.add_argument("--samples", type=Path, help="samples JSONL to validate")
    p.add_argument("--pairs", type=Path, help="pairs JSONL to validate")
    p.add_argument("--biased", type=Path, help="biased samples JSONL to validate")
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root that manifest-relative paths resolve against (default: CWD).",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Exit 0 even if files are missing (just print the report).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.samples and not args.pairs and not args.biased:
        print("nothing to validate: pass --samples, --pairs and/or --biased", file=sys.stderr)
        return 2

    all_ok = True
    if args.samples:
        rep = validate_samples(args.samples, args.root)
        print(f"[samples] {rep.summary()}")
        all_ok = all_ok and rep.ok
    if args.pairs:
        rep = validate_pairs(args.pairs, args.root)
        print(f"[pairs] {rep.summary()}")
        all_ok = all_ok and rep.ok
    if args.biased:
        rep = validate_biased(args.biased, args.root)
        print(f"[biased] {rep.summary()}")
        all_ok = all_ok and rep.ok

    if args.report_only:
        return 0
    return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
