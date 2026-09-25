"""Tests for result aggregation + plotting (Milestone 5)."""

from __future__ import annotations

import csv
from pathlib import Path

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult
from edit_judge_bias.experiments.aggregate_results import aggregate
from edit_judge_bias.experiments.plot_results import make_figures


def _score(model, sid, score, bias=None):
    return JudgeResult(
        result_id=f"{model}::{sid}::{bias}", judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt", parse_success=True,
        sample_id=sid, bias_type=bias, overall_score=score,
    )


def _setup_results(root: Path):
    rj = root / "results/raw_judgments"
    bj = root / "results/biased_judgments"
    io.write_jsonl(rj / "scoring__m.jsonl", [_score("m", "s0", 3), _score("m", "s1", 3)])
    io.write_jsonl(bj / "scoring__m.jsonl", [
        _score("m", "s0", 5, "brightness"), _score("m", "s1", 4, "brightness"),
        _score("m", "s0", 2, "padding"), _score("m", "s1", 3, "padding"),
    ])


def test_aggregate_writes_csvs(tmp_path: Path):
    _setup_results(tmp_path)
    stats = aggregate(tmp_path / "results", roster=None)
    assert len(stats["scoring"]) == 2  # brightness + padding
    csv_path = tmp_path / "results/metrics/scoring_shift.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    bright = next(r for r in rows if r["bias_type"] == "brightness")
    assert float(bright["mean_shift"]) == 1.5
    assert (tmp_path / "results/metrics/pilot_summary.csv").exists()


def test_make_figures_creates_pngs(tmp_path: Path):
    _setup_results(tmp_path)
    aggregate(tmp_path / "results", roster=None)
    made = make_figures(tmp_path / "results/metrics", tmp_path / "results/figures")
    assert made
    for p in made:
        assert p.exists() and p.suffix == ".png" and p.stat().st_size > 0


def test_aggregate_handles_no_results(tmp_path: Path):
    (tmp_path / "results/raw_judgments").mkdir(parents=True)
    stats = aggregate(tmp_path / "results", roster=None)
    assert stats["scoring"] == [] and stats["pairwise"] == []


def test_quality_controlled_shift_filters_to_passed(tmp_path: Path):
    from edit_judge_bias.data.schema import QualityValidationResult
    _setup_results(tmp_path)
    # s0's brightness passed validation; s1's failed -> QC counts only s0.
    io.write_jsonl(tmp_path / "results/quality/validation__v.jsonl", [
        QualityValidationResult(instruction_adherence_changed=False,
            editing_quality_changed=False, detail_preservation_changed=False,
            major_semantic_shift=False, passed=True, biased_id="m::s0::brightness",
            parse_success=True),
        QualityValidationResult(instruction_adherence_changed=False,
            editing_quality_changed=True, detail_preservation_changed=False,
            major_semantic_shift=False, passed=False, biased_id="m::s1::brightness",
            parse_success=True),
    ])
    # biased_id on the scoring records must match the validation biased_ids.
    bj = tmp_path / "results/biased_judgments/scoring__m.jsonl"
    recs = io.read_jsonl(bj, JudgeResult)
    for r in recs:
        r.biased_id = f"m::{r.sample_id}::{r.bias_type}"
    io.write_jsonl(bj, recs)

    stats = aggregate(tmp_path / "results", roster=None)
    assert stats["qc"]
    qc_bright = next(r for r in stats["qc"] if r["bias_type"] == "brightness")
    assert qc_bright["n"] == 1  # only the passed sample (s0)
    assert (tmp_path / "results/metrics/scoring_shift_qc.csv").exists()


def _validation(biased_id, passed):
    from edit_judge_bias.data.schema import QualityValidationResult
    return QualityValidationResult(
        instruction_adherence_changed=False, editing_quality_changed=not passed,
        detail_preservation_changed=False, major_semantic_shift=False,
        passed=passed, biased_id=biased_id, parse_success=True,
    )


def _setup_two_validators(tmp_path: Path):
    _setup_results(tmp_path)
    quality = tmp_path / "results/quality"
    io.write_jsonl(quality / "validation__v1.jsonl",
                   [_validation("m::s0::brightness", True),
                    _validation("m::s1::brightness", False)])
    # v2 disagrees about s1 and additionally flags s0.
    io.write_jsonl(quality / "validation__v2.jsonl",
                   [_validation("m::s0::brightness", False),
                    _validation("m::s1::brightness", True)])
    bj = tmp_path / "results/biased_judgments/scoring__m.jsonl"
    recs = io.read_jsonl(bj, JudgeResult)
    for r in recs:
        r.biased_id = f"m::{r.sample_id}::{r.bias_type}"
    io.write_jsonl(bj, recs)


def test_two_validators_intersect_rather_than_union(tmp_path: Path):
    """The QC subset means "no validator could tell the images apart", so adding a
    validator can only SHRINK it. A union would let an image one model flagged be
    readmitted by another — the subset would grow as scrutiny increased."""
    _setup_two_validators(tmp_path)
    stats = aggregate(tmp_path / "results", roster=None)
    assert not [r for r in stats["qc"] if r["bias_type"] == "brightness"], (
        "no image passed BOTH validators, so the brightness QC cell must be empty"
    )


def test_validators_can_be_named_when_one_is_too_noisy_to_gate_on(tmp_path: Path):
    """A validator that flags 21% of a visually null re-encode removes images at
    random rather than removing bad ones; it is reported, not intersected."""
    _setup_two_validators(tmp_path)
    stats = aggregate(tmp_path / "results", roster=None, validators=["v1"])
    qc_bright = next(r for r in stats["qc"] if r["bias_type"] == "brightness")
    assert qc_bright["n"] == 1  # v1's verdict alone: only s0


def test_aggregate_refuses_to_mix_analysis_variables(tmp_path: Path):
    """Shifts are resolved per judge (each is joined against its own baselines), so
    one judge answering without a dimension used to resolve to `overall_score` while
    the others stayed on `fine_score` — a mean_shift column holding both 3-30 sums and
    1-10 overalls, in which the odd judge reads as far more robust than it is."""
    import pytest

    rj = tmp_path / "results/raw_judgments"
    bj = tmp_path / "results/biased_judgments"
    fine = [_score("fine", "s0", 3), _score("fine", "s1", 3)]
    for r in fine:
        r.fine_score = 9
    fine_b = [_score("fine", "s0", 2, "brightness"), _score("fine", "s1", 2, "brightness")]
    for r in fine_b:
        r.fine_score = 6
    io.write_jsonl(rj / "scoring__fine.jsonl", fine)
    io.write_jsonl(bj / "scoring__fine.jsonl", fine_b)
    # second judge: overall_score only, so it resolves to the coarser variable
    io.write_jsonl(rj / "scoring__coarse.jsonl", [_score("coarse", "s0", 3), _score("coarse", "s1", 3)])
    io.write_jsonl(bj / "scoring__coarse.jsonl", [
        _score("coarse", "s0", 2, "brightness"), _score("coarse", "s1", 2, "brightness"),
    ])

    with pytest.raises(ValueError, match="mix analysis variables"):
        aggregate(tmp_path / "results", roster=None)


def test_subset_filter_keeps_one_block_out_of_the_table(tmp_path: Path):
    """MEASURED: once the anchor arm ran, gpt-5.5's `padding` cell went from n=611
    (breadth) to n=1,196 and mean_shift from -1.75 to -1.04, because the result
    manifests are per judge and the anchor blocks are single-source and deliberately
    unbalanced. Claim A's table has to be able to ask for one block."""
    from edit_judge_bias.data.schema import SampleRecord
    from edit_judge_bias.experiments.aggregate_results import sample_ids_matching

    rj = tmp_path / "results/raw_judgments"
    bj = tmp_path / "results/biased_judgments"
    io.write_jsonl(rj / "scoring__m.jsonl", [_score("m", "b0", 3), _score("m", "a0", 3)])
    io.write_jsonl(bj / "scoring__m.jsonl", [
        _score("m", "b0", 1, "padding"),   # breadth item: shift -2
        _score("m", "a0", 3, "padding"),   # anchor item: shift 0
    ])
    samples = tmp_path / "samples.jsonl"
    io.write_jsonl(samples, [
        SampleRecord(sample_id="b0", source_dataset="s", edit_type="add",
                     content_category="object", original_image_path="o.png",
                     instruction="i", edit_model="e", edited_image_path="e.png",
                     metadata={"subset_block": "breadth"}),
        SampleRecord(sample_id="a0", source_dataset="s", edit_type="add",
                     content_category="object", original_image_path="o.png",
                     instruction="i", edit_model="e", edited_image_path="e.png",
                     metadata={"subset_block": "anchor", "anchor_source": "EBench-18K"}),
    ])

    pooled = aggregate(tmp_path / "results", roster=None)["scoring"][0]
    assert pooled["n"] == 2 and pooled["mean_shift"] == -1.0

    keep = sample_ids_matching(samples, {"subset_block": "breadth"})
    assert keep == {"b0"}
    only_breadth = aggregate(tmp_path / "results", roster=None, keep_sample_ids=keep)["scoring"][0]
    assert only_breadth["n"] == 1 and only_breadth["mean_shift"] == -2.0
