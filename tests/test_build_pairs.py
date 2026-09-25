"""Tests for the pairwise builder (Milestone 1)."""

from __future__ import annotations

from typing import List

from edit_judge_bias.data.build_pairs import build_pairs, load_config
from edit_judge_bias.data.schema import SampleRecord


def _sample(stem: str, model: str, instruction: str = "Remove the man.") -> SampleRecord:
    return SampleRecord(
        sample_id=f"i2e_{stem}_{model}",
        source_dataset="I2EBench",
        edit_type="remove",
        content_category="human",
        original_image_path=f"orig/{stem}.png",
        instruction=instruction,
        edit_model=model,
        edited_image_path=f"edit/{model}/{stem}.png",
    )


def _group(stem: str, models: List[str], instruction: str = "Remove the man."):
    return [_sample(stem, m, instruction) for m in models]


def test_pairs_across_distinct_models_only():
    samples = _group("a", ["m1", "m2", "m3"])
    pairs = build_pairs(samples, {"seed": 1, "max_pairs_per_group": 0})
    assert len(pairs) == 3  # C(3,2)
    for p in pairs:
        assert p.edit_model_a != p.edit_model_b
        assert p.original_image_path.as_posix() == "orig/a.png"
        assert p.edit_type.value == "remove"
    # unique, sequential ids
    assert [p.pair_id for p in pairs] == ["pair_000001", "pair_000002", "pair_000003"]


def test_groups_split_by_original_and_instruction():
    samples = _group("a", ["m1", "m2"]) + _group("b", ["m1", "m2"])
    # same models, different originals -> two independent groups, no cross pairs.
    pairs = build_pairs(samples, {"seed": 1, "max_pairs_per_group": 0})
    assert len(pairs) == 2
    origins = {p.original_image_path.as_posix() for p in pairs}
    assert origins == {"orig/a.png", "orig/b.png"}


def test_singleton_group_yields_no_pairs():
    pairs = build_pairs(_group("a", ["m1"]), {"max_pairs_per_group": 0})
    assert pairs == []


def test_duplicate_model_in_group_is_deduped():
    samples = _group("a", ["m1", "m1", "m2"])
    pairs = build_pairs(samples, {"max_pairs_per_group": 0})
    assert len(pairs) == 1  # only m1-m2


def test_cap_per_group_and_determinism():
    samples = _group("a", [f"m{i}" for i in range(5)])  # C(5,2)=10
    cfg = {"seed": 7, "max_pairs_per_group": 3}
    p1 = build_pairs(samples, cfg)
    p2 = build_pairs(samples, cfg)
    assert len(p1) == 3
    assert [(p.edit_model_a, p.edit_model_b) for p in p1] == [
        (p.edit_model_a, p.edit_model_b) for p in p2
    ]


def test_load_config_merges_defaults(tmp_path):
    cfg_path = tmp_path / "pairs.yaml"
    cfg_path.write_text("max_pairs_per_group: 2\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg["max_pairs_per_group"] == 2
    assert cfg["seed"] == 42  # default preserved
