"""Tests for manifest path validation (Milestone 1)."""

from __future__ import annotations

from pathlib import Path

from conftest import touch_image

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import PairRecord, SampleRecord
from edit_judge_bias.data.validate_manifest import (
    main,
    validate_pairs,
    validate_samples,
)


def _sample(stem: str, root: Path, *, make_edited: bool = True) -> SampleRecord:
    touch_image(root / f"orig/{stem}.png")
    if make_edited:
        touch_image(root / f"edit/{stem}.png")
    return SampleRecord(
        sample_id=f"s_{stem}",
        source_dataset="I2EBench",
        edit_type="remove",
        content_category="human",
        original_image_path=f"orig/{stem}.png",
        instruction="Remove the man.",
        edit_model="m",
        edited_image_path=f"edit/{stem}.png",
    )


def test_validate_samples_all_present(tmp_path: Path):
    recs = [_sample("a", tmp_path), _sample("b", tmp_path)]
    out = tmp_path / "samples.jsonl"
    io.write_jsonl(out, recs)
    rep = validate_samples(out, root=tmp_path)
    assert rep.ok
    assert rep.n_records == 2
    assert rep.n_paths == 4  # 2 originals + 2 edited


def test_validate_samples_detects_missing(tmp_path: Path):
    rec = _sample("a", tmp_path, make_edited=False)  # edited file not created
    out = tmp_path / "samples.jsonl"
    io.write_jsonl(out, [rec])
    rep = validate_samples(out, root=tmp_path)
    assert not rep.ok
    assert len(rep.missing) == 1
    rid, field_name, _ = rep.missing[0]
    assert rid == "s_a"
    assert field_name == "edited_image_path"
    assert "FAIL" in rep.summary()


def test_validate_pairs(tmp_path: Path):
    a = _sample("a", tmp_path)
    b = _sample("b", tmp_path)
    pair = PairRecord(
        pair_id="pair_1",
        sample_id_a=a.sample_id,
        sample_id_b=b.sample_id,
        original_image_path=a.original_image_path,
        instruction=a.instruction,
        edited_image_a_path=a.edited_image_path,
        edited_image_b_path=b.edited_image_path,
        edit_model_a="m",
        edit_model_b="m2",
        source_dataset="I2EBench",
        edit_type="remove",
    )
    out = tmp_path / "pairs.jsonl"
    io.write_jsonl(out, [pair])
    rep = validate_pairs(out, root=tmp_path)
    assert rep.ok and rep.n_paths == 3


def test_cli_returns_nonzero_on_missing(tmp_path: Path, capsys):
    rec = _sample("a", tmp_path, make_edited=False)
    out = tmp_path / "samples.jsonl"
    io.write_jsonl(out, [rec])
    code = main(["--samples", str(out), "--root", str(tmp_path)])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_report_only_returns_zero(tmp_path: Path):
    rec = _sample("a", tmp_path, make_edited=False)
    out = tmp_path / "samples.jsonl"
    io.write_jsonl(out, [rec])
    code = main(["--samples", str(out), "--root", str(tmp_path), "--report-only"])
    assert code == 0
