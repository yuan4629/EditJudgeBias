"""JSONL io tests (Milestone 0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import (
    JudgeResult,
    QualityValidationResult,
    SampleRecord,
)


def _sample(i: int) -> SampleRecord:
    return SampleRecord(
        sample_id=f"s{i}",
        source_dataset="I2EBench",
        edit_type="remove",
        content_category="human",
        original_image_path=f"orig/{i}.png",
        instruction="Remove the man.",
        edit_model="MagicBrush",
        edited_image_path=f"edit/{i}.png",
    )


def test_write_then_read_roundtrip(tmp_path: Path):
    records = [_sample(i) for i in range(3)]
    out = tmp_path / "nested" / "samples.jsonl"  # parent dir does not exist yet
    n = io.write_jsonl(out, records)
    assert n == 3
    assert out.exists()
    back = io.read_jsonl(out, SampleRecord)
    assert back == records


def test_write_truncates_by_default(tmp_path: Path):
    out = tmp_path / "s.jsonl"
    io.write_jsonl(out, [_sample(0), _sample(1)])
    io.write_jsonl(out, [_sample(9)])  # overwrite
    back = io.read_jsonl(out, SampleRecord)
    assert [r.sample_id for r in back] == ["s9"]


def test_append_and_checkpoint(tmp_path: Path):
    out = tmp_path / "s.jsonl"
    io.write_jsonl(out, [_sample(0)])
    io.append_jsonl(out, _sample(1))
    io.write_jsonl(out, [_sample(2)], append=True)
    back = io.read_jsonl(out, SampleRecord)
    assert [r.sample_id for r in back] == ["s0", "s1", "s2"]


def test_iter_jsonl_skips_blank_lines(tmp_path: Path):
    out = tmp_path / "s.jsonl"
    io.write_jsonl(out, [_sample(0), _sample(1)])
    # Inject blank lines.
    content = out.read_text(encoding="utf-8")
    out.write_text("\n" + content + "\n\n", encoding="utf-8")
    back = list(io.iter_jsonl(out, SampleRecord))
    assert len(back) == 2


def test_iter_jsonl_is_lazy(tmp_path: Path):
    out = tmp_path / "s.jsonl"
    io.write_jsonl(out, [_sample(i) for i in range(5)])
    it = io.iter_jsonl(out, SampleRecord)
    first = next(it)
    assert first.sample_id == "s0"


def test_read_jsonl_bad_line_reports_location(tmp_path: Path):
    out = tmp_path / "s.jsonl"
    io.write_jsonl(out, [_sample(0)])
    with out.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json}\n")
    with pytest.raises(ValueError) as exc:
        io.read_jsonl(out, SampleRecord)
    assert "s.jsonl:2" in str(exc.value)


def test_unicode_instruction_roundtrip(tmp_path: Path):
    rec = _sample(0)
    rec.instruction = "把背景改成雪景"  # ensure ensure_ascii=False path works
    out = tmp_path / "s.jsonl"
    io.write_jsonl(out, [rec])
    assert "雪景" in out.read_text(encoding="utf-8")
    back = io.read_jsonl(out, SampleRecord)[0]
    assert back.instruction == "把背景改成雪景"


def test_quality_validation_jsonl_uses_pass_alias(tmp_path: Path):
    qv = QualityValidationResult(
        instruction_adherence_changed=False,
        editing_quality_changed=False,
        detail_preservation_changed=False,
        major_semantic_shift=False,
        passed=True,
        reason="ok",
    )
    out = tmp_path / "qv.jsonl"
    io.write_jsonl(out, [qv])
    assert '"pass":' in out.read_text(encoding="utf-8").replace(" ", "")
    back = io.read_jsonl(out, QualityValidationResult)[0]
    assert back.passed is True


def test_mixed_judge_results_roundtrip(tmp_path: Path):
    scoring = JudgeResult(
        result_id="score_1",
        judge_model="gpt5.5",
        task_type="scoring",
        prompt_type="vanilla_scoring",
        raw_response_path="raw/score_1.txt",
        parse_success=True,
        sample_id="s1",
        overall_score=4,
    )
    pairwise = JudgeResult(
        result_id="pair_1",
        judge_model="gpt5.5",
        task_type="pairwise",
        prompt_type="vanilla_pairwise",
        raw_response_path="raw/pair_1.txt",
        parse_success=True,
        pair_id="pair_1",
        biased_side="A",
        winner="A",
    )
    out = tmp_path / "judge.jsonl"
    io.write_jsonl(out, [scoring, pairwise])
    back = io.read_jsonl(out, JudgeResult)
    assert back == [scoring, pairwise]
