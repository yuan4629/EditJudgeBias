"""Tests for the pairwise judge runner (Milestone 3)."""

from __future__ import annotations

from pathlib import Path

import yaml
from conftest import make_rgb_image

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult, PairRecord
from edit_judge_bias.experiments.run_pairwise_judge import run


def _setup(root: Path, *, include_swap=False, n=3) -> Path:
    pairs = []
    make_rgb_image(root / "orig.png")
    for i in range(n):
        make_rgb_image(root / f"a{i}.png")
        make_rgb_image(root / f"b{i}.png")
        pairs.append(PairRecord(
            pair_id=f"pair_{i}", sample_id_a=f"a{i}", sample_id_b=f"b{i}",
            original_image_path="orig.png", instruction="Make it red.",
            edited_image_a_path=f"a{i}.png", edited_image_b_path=f"b{i}.png",
            edit_model_a="m1", edit_model_b="m2", source_dataset="I2EBench",
            edit_type="color",
        ))
    io.write_jsonl(root / "pairs.jsonl", pairs)
    cfg = {
        "pairs": "pairs.jsonl",
        "prompt_style": "vanilla",
        "include_position_swap": include_swap,
        "judge": {"type": "mock", "model_name": "mock-judge"},
        "raw_dir": "results/raw_responses",
        "results_dir": "results",
        "seed": 42,
    }
    cfg_path = root / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _orig(root: Path) -> Path:
    return root / "results/raw_judgments/pairwise__mock-judge.jsonl"


def _bias(root: Path) -> Path:
    return root / "results/biased_judgments/pairwise__mock-judge.jsonl"


def test_pairwise_basic(tmp_path: Path):
    cfg = _setup(tmp_path, n=3)
    stats = run(cfg, root=tmp_path)
    assert stats.written == 3
    recs = io.read_jsonl(_orig(tmp_path), JudgeResult)
    assert len(recs) == 3
    for r in recs:
        assert r.task_type.value == "pairwise"
        assert r.winner in {"A", "B", "Tie"}
        assert r.bias_type is None
        assert (tmp_path / r.raw_response_path).is_file()


def test_position_swap_goes_to_biased_manifest(tmp_path: Path):
    cfg = _setup(tmp_path, include_swap=True, n=2)
    stats = run(cfg, root=tmp_path)
    assert stats.written == 4  # 2 normal + 2 swapped
    orig = io.read_jsonl(_orig(tmp_path), JudgeResult)
    bias = io.read_jsonl(_bias(tmp_path), JudgeResult)
    assert len(orig) == 2 and len(bias) == 2
    assert all(r.bias_type == "position" for r in bias)
    assert all(r.biased_side == "swap" for r in bias)


def test_resumable(tmp_path: Path):
    cfg = _setup(tmp_path, n=2)
    run(cfg, root=tmp_path)
    stats2 = run(cfg, root=tmp_path)
    assert stats2.written == 0 and stats2.skipped == 2


def test_dry_run_writes_nothing(tmp_path: Path):
    cfg = _setup(tmp_path, n=2)
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.written == 2
    assert not _orig(tmp_path).exists()


# --------------------------------------------------------------------------- #
# one-sided image bias (full-version pairwise arm)                             #
# --------------------------------------------------------------------------- #
def _one_sided_setup(root: Path, *, n=4, side=None, bias_types=("padding",)) -> Path:
    from edit_judge_bias.data.schema import BiasedRecord

    cfg_path = _setup(root, n=n)
    biased = []
    for i in range(n):
        for bt in bias_types:
            for sid in (f"a{i}", f"b{i}"):
                make_rgb_image(root / f"{sid}__{bt}.png")
                biased.append(BiasedRecord(
                    biased_id=f"{sid}__{bt}", base_sample_id=sid, bias_type=bt,
                    bias_strength=1.0, biased_image_path=f"{sid}__{bt}.png",
                ))
    io.write_jsonl(root / "biased_members.jsonl", biased)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    spec = {"biased": "biased_members.jsonl", "bias_types": list(bias_types)}
    if side:
        spec["side"] = side
    data["one_sided_biases"] = spec
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return cfg_path


def test_one_sided_bias_replaces_exactly_one_image(tmp_path: Path):
    cfg = _one_sided_setup(tmp_path, n=2, side="a")
    run(cfg, root=tmp_path)
    recs = [r for r in io.read_jsonl(_bias(tmp_path), JudgeResult)
            if r.bias_type == "padding"]
    assert len(recs) == 2
    assert all(r.biased_side == "a" for r in recs)
    assert all(r.bias_params["biased_sample_id"].startswith("a") for r in recs)


def test_one_sided_side_is_seeded_not_always_slot_a(tmp_path: Path):
    """Always dressing slot A would make the one-sided effect inseparable from the
    position bias measured in the same run."""
    cfg = _one_sided_setup(tmp_path, n=24)
    run(cfg, root=tmp_path)
    sides = {r.biased_side for r in io.read_jsonl(_bias(tmp_path), JudgeResult)
             if r.bias_type == "padding"}
    assert sides == {"a", "b"}


def test_one_sided_skips_pairs_whose_member_was_not_injected(tmp_path: Path):
    """A missing biased image is skipped, not crashed on and not silently swapped
    for the unbiased one (which would dilute the effect toward zero)."""
    cfg = _one_sided_setup(tmp_path, n=2, side="a")
    rows = [r for r in io.read_jsonl(tmp_path / "biased_members.jsonl",
                                     __import__("edit_judge_bias.data.schema",
                                                fromlist=["BiasedRecord"]).BiasedRecord)
            if r.base_sample_id != "a0"]
    io.write_jsonl(tmp_path / "biased_members.jsonl", rows)
    run(cfg, root=tmp_path)
    recs = [r for r in io.read_jsonl(_bias(tmp_path), JudgeResult)
            if r.bias_type == "padding"]
    assert len(recs) == 1 and recs[0].pair_id == "pair_1"


def test_pairwise_judge_config_argument_overrides(tmp_path: Path):
    cfg = _setup(tmp_path, n=1)
    other = tmp_path / "judge_other.yaml"
    other.write_text(yaml.safe_dump(
        {"type": "mock", "model_name": "other-judge"}), encoding="utf-8")
    run(cfg, root=tmp_path, judge_config=other)
    assert (tmp_path / "results/raw_judgments/pairwise__other-judge.jsonl").exists()
