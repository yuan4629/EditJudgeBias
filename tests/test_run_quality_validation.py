"""Tests for the quality-validation runner (Milestone 6, mock validator)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml
from conftest import make_rgb_image

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import BiasedRecord, QualityValidationResult, SampleRecord
from edit_judge_bias.experiments.run_quality_validation import run


def _setup(root: Path, *, per_bias=0) -> Path:
    samples, biased = [], []
    for i in range(3):
        make_rgb_image(root / f"orig/{i}.png", size=(48, 48))
        make_rgb_image(root / f"edit/{i}.png", size=(48, 48), color=(90, 90, 90))
        samples.append(SampleRecord(
            sample_id=f"s{i}", source_dataset="I2EBench", edit_type="remove",
            content_category="human", original_image_path=f"orig/{i}.png",
            instruction="Remove the man.", edit_model="m", edited_image_path=f"edit/{i}.png",
        ))
        for bias in ("brightness", "padding"):
            make_rgb_image(root / f"biased/{bias}/{i}.png", size=(48, 48), color=(110, 110, 110))
            biased.append(BiasedRecord(
                biased_id=f"s{i}__{bias}", base_sample_id=f"s{i}", bias_type=bias,
                bias_strength=1.0, biased_image_path=f"biased/{bias}/{i}.png",
            ))
    io.write_jsonl(root / "samples.jsonl", samples)
    io.write_jsonl(root / "biased.jsonl", biased)
    cfg = {
        "samples": "samples.jsonl", "biased": "biased.jsonl", "per_bias": per_bias,
        "seed": 42, "judge": {"type": "mock", "model_name": "mock-validator"},
        "raw_dir": "results/raw_responses_validation",
        "out_manifest": "results/quality/validation__mock-validator.jsonl",
        "summary_csv": "results/metrics/quality_preservation_summary.csv",
        "failure_log": "results/logs/qv_failures.jsonl",
    }
    cfg_path = root / "qv.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _manifest(root: Path) -> Path:
    return root / "results/quality/validation__mock-validator.jsonl"


def test_validates_and_writes_results_with_ssim(tmp_path: Path):
    cfg = _setup(tmp_path)
    stats = run(cfg, root=tmp_path)
    assert stats.validated == 6  # 3 samples x 2 biases
    recs = io.read_jsonl(_manifest(tmp_path), QualityValidationResult)
    assert len(recs) == 6
    for r in recs:
        assert r.parse_success
        assert r.passed in (True, False)
        assert r.ssim is not None
        assert (tmp_path / r.raw_response_path).is_file()


def test_summary_csv_has_pass_rate_and_ssim(tmp_path: Path):
    cfg = _setup(tmp_path)
    run(cfg, root=tmp_path)
    csv_path = tmp_path / "results/metrics/quality_preservation_summary.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert {r["bias_type"] for r in rows} == {"brightness", "padding"}
    for r in rows:
        assert 0.0 <= float(r["mllm_pass_rate"]) <= 1.0
        assert float(r["mean_ssim"]) <= 1.0


def test_resumable(tmp_path: Path):
    cfg = _setup(tmp_path)
    run(cfg, root=tmp_path)
    stats2 = run(cfg, root=tmp_path)
    assert stats2.validated == 0 and stats2.skipped == 6


def test_per_bias_sampling_limits_calls(tmp_path: Path):
    cfg = _setup(tmp_path, per_bias=1)
    stats = run(cfg, root=tmp_path)
    assert stats.validated == 2  # 1 per bias x 2 biases


def test_dry_run_writes_no_manifest(tmp_path: Path):
    cfg = _setup(tmp_path)
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.validated == 6
    assert not _manifest(tmp_path).exists()


# --- WP-A4d: the `bias_types` allowlist -------------------------------------------
# A4d needs ONE of the four `edit_damage` conditions (110 images) and must not pay for
# the other three.  The knob is small; the tests below exist because two of its failure
# modes are silent.


def test_bias_types_selects_one_bucket(tmp_path: Path):
    cfg_path = _setup(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["bias_types"] = ["padding"]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    stats = run(cfg_path, root=tmp_path)
    assert stats.validated == 3  # 3 samples x 1 bias, not 6
    recs = io.read_jsonl(_manifest(tmp_path), QualityValidationResult)
    assert all(r.biased_id.endswith("__padding") for r in recs)


def test_bias_types_typo_raises_instead_of_collecting_nothing(tmp_path: Path):
    """`validated=0` is indistinguishable from "this arm already finished"."""
    cfg_path = _setup(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["bias_types"] = ["padidng"]  # transposed
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="padidng"):
        run(cfg_path, root=tmp_path)


def test_bias_types_with_per_bias_is_refused(tmp_path: Path):
    """The shared-RNG hazard WP-A3 traced, refused rather than silently re-drawing.

    `_select` shuffles every bucket from one `random.Random(seed)` built outside the
    loop, so dropping a bucket hands the survivors a different RNG state.  With
    `per_bias > 0` that changes WHICH rows are drawn.  Repairing it (reset per bucket)
    would re-draw the published `_select(110, seed=42)` sample, so the combination is
    refused instead.
    """
    cfg_path = _setup(tmp_path, per_bias=1)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["bias_types"] = ["padding"]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="per_bias"):
        run(cfg_path, root=tmp_path)


def test_bias_types_absent_leaves_published_selection_untouched(tmp_path: Path):
    """No config that omits the knob may change behaviour."""
    from edit_judge_bias.experiments.run_quality_validation import _select

    _setup(tmp_path)
    biased = io.read_jsonl(tmp_path / "biased.jsonl", BiasedRecord)
    assert [b.biased_id for b in _select(biased, 0, 42)] == [
        b.biased_id for b in _select(biased, 0, 42, None)
    ]
    assert [b.biased_id for b in _select(biased, 1, 42)] == [
        b.biased_id for b in _select(biased, 1, 42, [])
    ]
