"""Tests for A-class (prompt-level) bias specs and their wiring into the runners.

A-class biases (§4.A / §5.4) change only the judge prompt: the image shown is the
ordinary edited image. They were implemented in the prompt builders long before the
runners could reach them; these tests pin the wiring so that gap cannot reopen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edit_judge_bias.experiments import judge_common
import yaml
from conftest import make_rgb_image

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult, PairRecord, SampleRecord
from edit_judge_bias.experiments.prompt_bias import (
    bandwagon_side,
    load_specs,
    permuted_name_map,
    resolve_model_name,
)
from edit_judge_bias.experiments.run_pairwise_judge import run as run_pairwise
from edit_judge_bias.experiments.run_scoring_judge import run as run_scoring


# --------------------------------------------------------------------------- #
# prompt_bias helpers                                                          #
# --------------------------------------------------------------------------- #
def test_permuted_name_map_is_a_deterministic_derangement():
    names = ["magicbrush", "instructpix2pix", "hive", "mgie", "iedit"]
    m1 = permuted_name_map(names, 42)
    m2 = permuted_name_map(list(reversed(names)), 42)
    assert m1 == m2                       # order of the roster must not matter
    assert set(m1) == set(names)
    assert set(m1.values()) == set(names)  # a permutation, not a collapse
    # No fixed point: the A3 control must never name the true generator.
    assert all(src != dst for src, dst in m1.items())


def test_permuted_name_map_single_name_roster_is_returned_unchanged():
    # A one-model roster has no derangement; the caller gets the identity rather
    # than an exception, so a single-model source does not abort a whole run.
    assert permuted_name_map(["only"], 42) == {"only": "only"}


@pytest.mark.parametrize(
    "spec,expected",
    [
        ({"model_name_source": "true_name"}, "magicbrush"),
        ({"model_name_source": "none"}, None),
        ({}, None),
        ({"model_name_source": "literal", "model_name": "GPT-Image"}, "GPT-Image"),
    ],
)
def test_resolve_model_name(spec, expected):
    assert resolve_model_name(spec, "magicbrush", {"magicbrush": "hive"}) == expected


def test_resolve_model_name_permuted_uses_the_map():
    spec = {"model_name_source": "permuted"}
    assert resolve_model_name(spec, "magicbrush", {"magicbrush": "hive"}) == "hive"


def test_bandwagon_side_seeded_is_deterministic_and_not_always_a():
    spec = {"bandwagon_target": "seeded"}
    sides = [bandwagon_side(spec, f"pair_{i}", 42) for i in range(40)]
    assert set(sides) == {"A", "B"}  # seeded endorsement must not favour slot A,
    #                                  else A2 would be confounded with A1 position bias
    assert sides == [bandwagon_side(spec, f"pair_{i}", 42) for i in range(40)]


def test_bandwagon_side_explicit_and_off():
    assert bandwagon_side({"bandwagon_target": "B"}, "p", 42) == "B"
    assert bandwagon_side({}, "p", 42) is None


def test_bandwagon_side_rejects_garbage():
    with pytest.raises(ValueError):
        bandwagon_side({"bandwagon_target": "left"}, "p", 42)


def test_load_specs_rejects_duplicate_and_missing_bias_type():
    with pytest.raises(ValueError):
        load_specs({"prompt_biases": [{"bandwagon": True}]})
    with pytest.raises(ValueError):
        load_specs({"prompt_biases": [{"bias_type": "x"}, {"bias_type": "x"}]})


# --------------------------------------------------------------------------- #
# scoring runner wiring                                                        #
# --------------------------------------------------------------------------- #
def _scoring_cfg(root: Path, prompt_biases: list) -> Path:
    samples = []
    for i, model in enumerate(["magicbrush", "hive"]):
        make_rgb_image(root / f"orig/{i}.png")
        make_rgb_image(root / f"edit/{i}.png")
        samples.append(SampleRecord(
            sample_id=f"s{i}", source_dataset="I2EBench", edit_type="remove",
            content_category="human", original_image_path=f"orig/{i}.png",
            instruction="Remove the man.", edit_model=model,
            edited_image_path=f"edit/{i}.png",
        ))
    io.write_jsonl(root / "samples.jsonl", samples)
    cfg = {
        "samples": "samples.jsonl",
        "score_original": True,
        "score_biased": False,
        "prompt_style": "vanilla",
        "prompt_biases": prompt_biases,
        "judge": {"type": "mock", "model_name": "mock-judge"},
        "raw_dir": "results/raw_responses",
        "results_dir": "results",
        "seed": 42,
    }
    path = root / "exp.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _scoring_biased(root: Path):
    return io.read_jsonl(
        root / "results/biased_judgments/scoring__mock-judge.jsonl", JudgeResult
    )


def test_scoring_prompt_bias_adds_one_condition_per_spec(tmp_path: Path):
    cfg = _scoring_cfg(tmp_path, [
        {"bias_type": "bandwagon", "bandwagon": True},
        {"bias_type": "model_name", "model_name_source": "true_name"},
    ])
    stats = run_scoring(cfg, root=tmp_path)
    assert stats.written == 6  # 2 unbiased baseline + 2 samples x 2 prompt biases

    biased = _scoring_biased(tmp_path)
    assert len(biased) == 4
    assert {r.bias_type for r in biased} == {"bandwagon", "model_name"}
    # A-class verdicts route to biased_judgments/ like any other bias, and the
    # baseline manifest keeps only the unbiased condition.
    orig = io.read_jsonl(
        tmp_path / "results/raw_judgments/scoring__mock-judge.jsonl", JudgeResult
    )
    assert len(orig) == 2 and all(r.bias_type is None for r in orig)


def test_scoring_prompt_bias_records_params_and_reaches_the_prompt(tmp_path: Path):
    cfg = _scoring_cfg(tmp_path, [
        {"bias_type": "bandwagon", "bandwagon": True},
        {"bias_type": "model_name", "model_name_source": "true_name"},
    ])
    run_scoring(cfg, root=tmp_path)
    by_type = {}
    for r in _scoring_biased(tmp_path):
        by_type.setdefault(r.bias_type, []).append(r)

    assert all(r.bias_params.get("bandwagon") for r in by_type["bandwagon"])
    # A3 names the true generator, one per sample.
    assert {r.bias_params["model_name"] for r in by_type["model_name"]} == {
        "magicbrush", "hive"
    }
    assert all(
        r.bias_params["model_name_source"] == "true_name" for r in by_type["model_name"]
    )
    # The injected sentence really is in the prompt the judge saw. The mock judge
    # hashes the prompt, so a prompt that never changed would be undetectable here
    # -- assert against the saved raw request instead of trusting the score.
    prompt = _built_scoring_prompt(bandwagon=True)
    assert "previous reviewers rated this edit very highly" in prompt.lower()
    assert "produced by the magicbrush model" in _built_scoring_prompt(
        model_name="magicbrush"
    ).lower()


def _built_scoring_prompt(**kwargs) -> str:
    from edit_judge_bias.prompts import build_scoring_prompt

    return build_scoring_prompt("Remove the man.", "vanilla", **kwargs)


def test_scoring_prompt_bias_changes_the_prompt_the_judge_sees(tmp_path: Path):
    """The runner must pass the flags through, not merely tag the result row."""
    seen: list[str] = []

    import edit_judge_bias.experiments.run_scoring_judge as mod

    class RecordingAdapter:
        model_name = "mock-judge"

        def score(self, prompt, images):
            seen.append(prompt)
            return '{"overall_score": 3, "instruction_adherence": 3, ' \
                   '"editing_quality": 3, "detail_preservation": 3, "reason": "ok"}'

    cfg = _scoring_cfg(tmp_path, [{"bias_type": "bandwagon", "bandwagon": True}])
    import pytest as _pytest  # local alias to keep monkeypatch out of the signature

    mp = _pytest.MonkeyPatch()
    mp.setattr(judge_common, "build_adapter", lambda *a, **k: RecordingAdapter())
    try:
        run_scoring(cfg, root=tmp_path)
    finally:
        mp.undo()

    bandwagon_prompts = [p for p in seen if "previous reviewers" in p.lower()]
    assert len(bandwagon_prompts) == 2      # exactly the two A2 conditions
    assert len(seen) == 4                   # and the two baselines stayed clean


def test_scoring_prompt_bias_is_resumable(tmp_path: Path):
    cfg = _scoring_cfg(tmp_path, [{"bias_type": "bandwagon", "bandwagon": True}])
    run_scoring(cfg, root=tmp_path)
    stats = run_scoring(cfg, root=tmp_path)
    assert stats.written == 0 and stats.skipped == 4


def test_scoring_permuted_model_name_never_names_the_true_generator(tmp_path: Path):
    cfg = _scoring_cfg(tmp_path, [
        {"bias_type": "model_name_permuted", "model_name_source": "permuted"},
    ])
    run_scoring(cfg, root=tmp_path)
    injected = {r.sample_id: r.bias_params["model_name"] for r in _scoring_biased(tmp_path)}
    assert injected == {"s0": "hive", "s1": "magicbrush"}  # swapped, deterministic


# --------------------------------------------------------------------------- #
# pairwise runner wiring                                                       #
# --------------------------------------------------------------------------- #
def _pairwise_cfg(root: Path, prompt_biases: list, n: int = 4) -> Path:
    pairs = []
    make_rgb_image(root / "orig.png")
    for i in range(n):
        make_rgb_image(root / f"a{i}.png")
        make_rgb_image(root / f"b{i}.png")
        pairs.append(PairRecord(
            pair_id=f"pair_{i}", sample_id_a=f"a{i}", sample_id_b=f"b{i}",
            original_image_path="orig.png", instruction="Make it red.",
            edited_image_a_path=f"a{i}.png", edited_image_b_path=f"b{i}.png",
            edit_model_a="magicbrush", edit_model_b="hive",
            source_dataset="I2EBench", edit_type="color",
        ))
    io.write_jsonl(root / "pairs.jsonl", pairs)
    cfg = {
        "pairs": "pairs.jsonl",
        "prompt_style": "vanilla",
        "include_position_swap": False,
        "prompt_biases": prompt_biases,
        "judge": {"type": "mock", "model_name": "mock-judge"},
        "raw_dir": "results/raw_responses",
        "results_dir": "results",
        "seed": 42,
    }
    path = root / "exp.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_pairwise_prompt_bias_conditions_and_params(tmp_path: Path):
    cfg = _pairwise_cfg(tmp_path, [
        {"bias_type": "bandwagon", "bandwagon_target": "seeded"},
        {"bias_type": "model_name", "model_name_source": "true_name"},
    ], n=4)
    stats = run_pairwise(cfg, root=tmp_path)
    assert stats.written == 12  # 4 baseline + 4 x 2 prompt-bias conditions

    biased = io.read_jsonl(
        tmp_path / "results/biased_judgments/pairwise__mock-judge.jsonl", JudgeResult
    )
    assert len(biased) == 8
    bw = [r for r in biased if r.bias_type == "bandwagon"]
    assert {r.bias_params["bandwagon_target"] for r in bw} <= {"A", "B"}
    mn = [r for r in biased if r.bias_type == "model_name"]
    assert all(
        r.bias_params["model_name_a"] == "magicbrush"
        and r.bias_params["model_name_b"] == "hive"
        for r in mn
    )
    # biased_side stays None: nothing was swapped, only the prompt text changed.
    assert all(r.biased_side is None for r in biased)


def test_pairwise_prompt_bias_coexists_with_position_swap(tmp_path: Path):
    cfg = _pairwise_cfg(tmp_path, [
        {"bias_type": "bandwagon", "bandwagon_target": "A"},
    ], n=3)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["include_position_swap"] = True
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    stats = run_pairwise(cfg, root=tmp_path)
    assert stats.written == 9  # 3 baseline + 3 position swap + 3 bandwagon
    biased = io.read_jsonl(
        tmp_path / "results/biased_judgments/pairwise__mock-judge.jsonl", JudgeResult
    )
    assert {r.bias_type for r in biased} == {"position", "bandwagon"}
    assert len({r.result_id for r in biased}) == 6  # ids stay unique across conditions
