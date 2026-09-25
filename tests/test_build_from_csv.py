"""Tests for the generic CSV -> samples adapter (Milestone 1)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from conftest import touch_image

from edit_judge_bias.data import io
from edit_judge_bias.data.build_from_csv import build_from_csv, read_csv_rows
from edit_judge_bias.data.schema import ContentCategory, EditType, SampleRecord

COLUMNS = [
    "sample_id",
    "source_dataset",
    "edit_type",
    "content_category",
    "original_image_path",
    "instruction",
    "edit_model",
    "edited_image_path",
    "has_mask",
    "auto_score",
]


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(i: int, root: Path) -> dict:
    orig = touch_image(root / f"orig/{i}.png")
    edited = touch_image(root / f"edit/{i}.png")
    return {
        "sample_id": f"s{i}",
        "source_dataset": "custom",
        "edit_type": "color",
        "content_category": "object",
        "original_image_path": orig.relative_to(root).as_posix(),
        "instruction": "Make it red.",
        "edit_model": "modelX",
        "edited_image_path": edited.relative_to(root).as_posix(),
        "has_mask": "false",
        "auto_score": "0.5",
    }


def test_build_from_csv_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "in.csv"
    _write_csv(csv_path, [_row(0, tmp_path), _row(1, tmp_path)])
    out = tmp_path / "data/manifests/samples.jsonl"

    samples = build_from_csv(csv_path, out, root=tmp_path)
    assert len(samples) == 2
    assert samples[0].edit_type is EditType.COLOR
    assert samples[0].content_category is ContentCategory.OBJECT
    assert samples[0].auto_score == 0.5

    back = io.read_jsonl(out, SampleRecord)
    assert [s.sample_id for s in back] == ["s0", "s1"]


def test_missing_required_column_raises(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("sample_id,instruction\n")
        fh.write("s0,hi\n")
    with pytest.raises(ValueError, match="missing required columns"):
        read_csv_rows(csv_path)


def test_path_check_fails_on_missing_image(tmp_path: Path):
    csv_path = tmp_path / "in.csv"
    row = _row(0, tmp_path)
    row["edited_image_path"] = "edit/does_not_exist.png"
    _write_csv(csv_path, [row])
    out = tmp_path / "samples.jsonl"
    with pytest.raises(FileNotFoundError):
        build_from_csv(csv_path, out, root=tmp_path)


def test_skip_path_check_allows_missing(tmp_path: Path):
    csv_path = tmp_path / "in.csv"
    row = _row(0, tmp_path)
    row["edited_image_path"] = "edit/does_not_exist.png"
    _write_csv(csv_path, [row])
    out = tmp_path / "samples.jsonl"
    samples = build_from_csv(csv_path, out, root=tmp_path, skip_path_check=True)
    assert len(samples) == 1


def test_dry_run_writes_nothing(tmp_path: Path):
    csv_path = tmp_path / "in.csv"
    _write_csv(csv_path, [_row(0, tmp_path)])
    out = tmp_path / "samples.jsonl"
    build_from_csv(csv_path, out, root=tmp_path, dry_run=True)
    assert not out.exists()
