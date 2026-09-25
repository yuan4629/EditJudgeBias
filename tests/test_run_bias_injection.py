"""Tests for the batch bias-injection runner (Milestone 2)."""

from __future__ import annotations

from pathlib import Path

import yaml
from conftest import make_rgb_image

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import BiasedRecord, SampleRecord
from edit_judge_bias.experiments.run_bias_injection import run


def _sample(i: int, root: Path, *, make_edited: bool = True) -> SampleRecord:
    edited_rel = f"edited/{i}.png"
    if make_edited:
        make_rgb_image(root / edited_rel, size=(80, 64))
    return SampleRecord(
        sample_id=f"s{i}",
        source_dataset="I2EBench",
        edit_type="remove",
        content_category="human",
        original_image_path=f"orig/{i}.png",
        instruction="Remove the red slippers",
        edit_model="m",
        edited_image_path=edited_rel,
    )


def _setup(root: Path, samples, *, biases=None) -> Path:
    io.write_jsonl(root / "samples.jsonl", samples)
    biases = biases if biases is not None else [
        {"bias_type": "brightness", "config": {"factor": 1.2}},
        {"bias_type": "padding", "config": {"ratio": 0.1}},
    ]
    cfg = {
        "samples": "samples.jsonl",
        "output_dir": "biased",
        "manifest_out": "biased.jsonl",
        "failure_log": "failures.jsonl",
        "seed": 42,
        "biases": biases,
    }
    cfg_path = root / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_run_writes_record_and_image_per_bias(tmp_path: Path):
    cfg = _setup(tmp_path, [_sample(0, tmp_path), _sample(1, tmp_path)])
    stats = run(cfg, root=tmp_path)
    assert stats.written == 4  # 2 samples x 2 biases
    assert stats.failed == 0
    assert dict(stats.per_bias) == {"brightness": 2, "padding": 2}

    records = io.read_jsonl(tmp_path / "biased.jsonl", BiasedRecord)
    assert len(records) == 4
    for rec in records:
        assert (tmp_path / rec.biased_image_path).is_file()
        assert rec.biased_id == f"{rec.base_sample_id}__{rec.bias_type}"
    # padding params captured
    pad = next(r for r in records if r.bias_type == "padding")
    assert pad.bias_params["padding_ratio"] == 0.1


def test_run_is_resumable(tmp_path: Path):
    cfg = _setup(tmp_path, [_sample(0, tmp_path), _sample(1, tmp_path)])
    run(cfg, root=tmp_path)
    stats2 = run(cfg, root=tmp_path)  # everything already done
    assert stats2.written == 0
    assert stats2.skipped == 4
    assert len(io.read_jsonl(tmp_path / "biased.jsonl", BiasedRecord)) == 4


def test_overwrite_rebuilds(tmp_path: Path):
    cfg = _setup(tmp_path, [_sample(0, tmp_path)])
    run(cfg, root=tmp_path)
    stats = run(cfg, root=tmp_path, overwrite=True)
    assert stats.written == 2
    assert stats.skipped == 0


def test_failure_is_logged_not_fatal(tmp_path: Path):
    # s0 has a real edited image; s1's edited image is missing -> should fail.
    samples = [_sample(0, tmp_path), _sample(1, tmp_path, make_edited=False)]
    cfg = _setup(tmp_path, samples, biases=[{"bias_type": "brightness", "config": {}}])
    stats = run(cfg, root=tmp_path)
    assert stats.written == 1
    assert stats.failed == 1
    failures = (tmp_path / "failures.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(failures) == 1
    assert "s1__brightness" in failures[0]


def test_dry_run_writes_nothing(tmp_path: Path):
    cfg = _setup(tmp_path, [_sample(0, tmp_path)])
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.written == 2
    assert not (tmp_path / "biased.jsonl").exists()
    assert not (tmp_path / "biased").exists()


def test_limit(tmp_path: Path):
    cfg = _setup(tmp_path, [_sample(i, tmp_path) for i in range(3)])
    stats = run(cfg, root=tmp_path, limit=1)
    assert stats.written == 2  # 1 sample x 2 biases


def test_bias_spec_from_file_reference(tmp_path: Path):
    (tmp_path / "configs/bias").mkdir(parents=True)
    (tmp_path / "configs/bias/brightness.yaml").write_text(
        yaml.safe_dump({"bias_type": "brightness", "config": {"factor": 1.3}}),
        encoding="utf-8",
    )
    cfg = _setup(tmp_path, [_sample(0, tmp_path)], biases=["configs/bias/brightness.yaml"])
    stats = run(cfg, root=tmp_path)
    assert stats.written == 1
    rec = io.read_jsonl(tmp_path / "biased.jsonl", BiasedRecord)[0]
    assert rec.bias_strength == 1.3
