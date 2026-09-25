"""Schema tests.

Covers construction, enum validation, JSON round-trip (incl. Path + the `pass`
alias), and reading the 3 toy samples shipped in tests/data/toy_samples.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import (
    BiasAppliedTo,
    BiasedRecord,
    ContentCategory,
    EditType,
    JudgeResult,
    PairRecord,
    QualityValidationResult,
    SampleRecord,
    TaskType,
)

TOY_SAMPLES = Path(__file__).parent / "data" / "toy_samples.jsonl"


# --------------------------------------------------------------------------- #
# SampleRecord                                                                 #
# --------------------------------------------------------------------------- #
def test_sample_record_minimal_defaults():
    rec = SampleRecord(
        sample_id="s1",
        source_dataset="I2EBench",
        edit_type="remove",
        content_category="human",
        original_image_path="a/orig.png",
        instruction="Remove the man.",
        edit_model="MagicBrush",
        edited_image_path="a/edit.png",
    )
    assert rec.edit_type is EditType.REMOVE
    assert rec.content_category is ContentCategory.HUMAN
    assert isinstance(rec.original_image_path, Path)
    # Optional fields default to null-equivalents; metadata is a real submodel.
    assert rec.reference_image_path is None
    assert rec.metadata.has_mask is False


def test_sample_record_roundtrip_preserves_paths():
    rec = SampleRecord(
        sample_id="s1",
        source_dataset="I2EBench",
        edit_type="color",
        content_category="object",
        original_image_path="data/images/original/x.png",
        instruction="Make the car red.",
        edit_model="qwen",
        edited_image_path="data/images/edited/x.png",
    )
    restored = SampleRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec
    assert isinstance(restored.edited_image_path, Path)


def test_sample_record_rejects_unknown_edit_type():
    with pytest.raises(ValidationError):
        SampleRecord(
            sample_id="s1",
            source_dataset="I2EBench",
            edit_type="teleport",  # not in taxonomy
            content_category="human",
            original_image_path="a.png",
            instruction="x",
            edit_model="m",
            edited_image_path="b.png",
        )


def test_record_forbids_extra_fields():
    with pytest.raises(ValidationError):
        SampleRecord(
            sample_id="s1",
            source_dataset="I2EBench",
            edit_type="add",
            content_category="object",
            original_image_path="a.png",
            instruction="x",
            edit_model="m",
            edited_image_path="b.png",
            typo_field="oops",
        )


def test_read_three_toy_samples():
    samples = io.read_jsonl(TOY_SAMPLES, SampleRecord)
    assert len(samples) == 3
    assert [s.sample_id for s in samples] == [
        "i2e_000001_magicbrush",
        "i2e_000002_qwen",
        "i2e_000003_instructpix2pix",
    ]
    assert samples[2].metadata.has_mask is True
    assert samples[2].metadata.original_width == 768


# --------------------------------------------------------------------------- #
# PairRecord                                                                   #
# --------------------------------------------------------------------------- #
def test_pair_record_roundtrip():
    pair = PairRecord(
        pair_id="pair_1",
        sample_id_a="s_a",
        sample_id_b="s_b",
        original_image_path="orig.png",
        instruction="Remove the man.",
        edited_image_a_path="a.png",
        edited_image_b_path="b.png",
        edit_model_a="MagicBrush",
        edit_model_b="InstructPix2Pix",
        source_dataset="I2EBench",
        edit_type="remove",
    )
    restored = PairRecord.model_validate_json(pair.model_dump_json())
    assert restored == pair
    assert restored.ground_truth_preference is None


# --------------------------------------------------------------------------- #
# BiasedRecord                                                                 #
# --------------------------------------------------------------------------- #
def test_biased_record_defaults_and_quality_block():
    rec = BiasedRecord(
        biased_id="bias_1",
        base_sample_id="s1",
        bias_type="brightness",
        bias_strength=1.2,
        biased_image_path="data/images/biased/brightness/x.png",
    )
    assert rec.bias_applied_to is BiasAppliedTo.EDITED_IMAGE
    # Quality block present but unmeasured.
    assert rec.quality_preservation.ssim is None
    assert rec.quality_preservation.mllm_validation_pass is None
    restored = BiasedRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


# --------------------------------------------------------------------------- #
# JudgeResult                                                                  #
# --------------------------------------------------------------------------- #
def test_judge_result_scoring_branch():
    res = JudgeResult(
        result_id="score_1",
        judge_model="gpt5.5",
        task_type="scoring",
        prompt_type="vanilla_scoring",
        raw_response_path="results/raw/gpt5.5/score_1.txt",
        parse_success=True,
        sample_id="s1",
        overall_score=4,
        instruction_adherence=5,
        editing_quality=4,
        detail_preservation=4,
        reason="good",
    )
    assert res.task_type is TaskType.SCORING
    assert res.winner is None
    restored = JudgeResult.model_validate_json(res.model_dump_json())
    assert restored == res


def test_judge_result_parse_failure_keeps_raw_path():
    res = JudgeResult(
        result_id="score_2",
        judge_model="gemini3.1",
        task_type="scoring",
        prompt_type="vanilla_scoring",
        raw_response_path="results/raw/gemini3.1/score_2.txt",
        parse_success=False,
        parse_error="model returned prose, not JSON",
    )
    assert res.parse_success is False
    assert res.overall_score is None
    assert Path(res.raw_response_path).name == "score_2.txt"


# --------------------------------------------------------------------------- #
# QualityValidationResult — the `pass` alias                                   #
# --------------------------------------------------------------------------- #
def test_quality_validation_pass_alias_in_and_out():
    raw_from_model = {
        "instruction_adherence_changed": False,
        "editing_quality_changed": False,
        "detail_preservation_changed": False,
        "major_semantic_shift": False,
        "pass": True,
        "reason": "no change",
    }
    qv = QualityValidationResult.model_validate(raw_from_model)
    assert qv.passed is True
    # Serializes back to the wire name "pass", not "passed".
    dumped = json.loads(qv.model_dump_json(by_alias=True))
    assert dumped["pass"] is True
    assert "passed" not in dumped


def test_quality_validation_accepts_python_name():
    qv = QualityValidationResult(
        instruction_adherence_changed=False,
        editing_quality_changed=False,
        detail_preservation_changed=False,
        major_semantic_shift=False,
        passed=False,
    )
    assert qv.passed is False
