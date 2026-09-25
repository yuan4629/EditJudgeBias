"""Tests for the scoring judge runner (Milestone 3)."""

from __future__ import annotations

from pathlib import Path

import yaml
from conftest import make_rgb_image

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import BiasedRecord, JudgeResult, SampleRecord
# The runners build their adapter through judge_common.adapter_for, so this module
# is the single seam a fake adapter has to be injected at.
from edit_judge_bias.experiments import judge_common
from edit_judge_bias.experiments.run_scoring_judge import run


def _setup(root: Path, *, response_style="valid", score_biased=True) -> Path:
    samples = []
    biased = []
    for i in range(2):
        make_rgb_image(root / f"orig/{i}.png")
        make_rgb_image(root / f"edit/{i}.png")
        samples.append(SampleRecord(
            sample_id=f"s{i}", source_dataset="I2EBench", edit_type="remove",
            content_category="human", original_image_path=f"orig/{i}.png",
            instruction="Remove the man.", edit_model="m",
            edited_image_path=f"edit/{i}.png",
        ))
        make_rgb_image(root / f"biased/{i}.png")
        biased.append(BiasedRecord(
            biased_id=f"s{i}__brightness", base_sample_id=f"s{i}",
            bias_type="brightness", bias_strength=1.2,
            biased_image_path=f"biased/{i}.png",
        ))
    io.write_jsonl(root / "samples.jsonl", samples)
    io.write_jsonl(root / "biased.jsonl", biased)

    cfg = {
        "samples": "samples.jsonl",
        "biased": "biased.jsonl",
        "score_original": True,
        "score_biased": score_biased,
        "prompt_style": "vanilla",
        "judge": {"type": "mock", "model_name": "mock-judge", "response_style": response_style},
        "raw_dir": "results/raw_responses",
        "results_dir": "results",
        "seed": 42,
    }
    cfg_path = root / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _orig_manifest(root: Path) -> Path:
    return root / "results/raw_judgments/scoring__mock-judge.jsonl"


def _bias_manifest(root: Path) -> Path:
    return root / "results/biased_judgments/scoring__mock-judge.jsonl"


def test_scores_original_and_biased_into_separate_manifests(tmp_path: Path):
    cfg = _setup(tmp_path)
    stats = run(cfg, root=tmp_path)
    assert stats.written == 4  # 2 original + 2 biased
    assert stats.parse_failures == 0

    orig = io.read_jsonl(_orig_manifest(tmp_path), JudgeResult)
    bias = io.read_jsonl(_bias_manifest(tmp_path), JudgeResult)
    assert len(orig) == 2 and len(bias) == 2
    assert all(r.bias_type is None for r in orig)
    assert all(r.bias_type == "brightness" for r in bias)
    for r in orig + bias:
        assert r.task_type.value == "scoring"
        assert r.parse_success and 1 <= r.overall_score <= 10
        # Every row records the scale it was produced on and the analysis variable.
        assert r.score_scale == 10
        assert r.fine_score == (
            r.instruction_adherence + r.editing_quality + r.detail_preservation
        )
        # raw response was persisted before parsing
        assert (tmp_path / r.raw_response_path).is_file()


def test_resumable(tmp_path: Path):
    cfg = _setup(tmp_path)
    run(cfg, root=tmp_path)
    stats2 = run(cfg, root=tmp_path)
    assert stats2.written == 0
    assert stats2.skipped == 4


def test_parse_failure_is_recorded_not_dropped(tmp_path: Path):
    cfg = _setup(tmp_path, response_style="invalid", score_biased=False)
    stats = run(cfg, root=tmp_path)
    assert stats.written == 2  # still written
    assert stats.parse_failures == 2
    recs = io.read_jsonl(_orig_manifest(tmp_path), JudgeResult)
    for r in recs:
        assert r.parse_success is False
        assert r.parse_error
        assert r.overall_score is None
        assert (tmp_path / r.raw_response_path).is_file()  # raw still saved


def test_dry_run_writes_nothing(tmp_path: Path):
    cfg = _setup(tmp_path)
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.written == 4
    assert not _orig_manifest(tmp_path).exists()
    assert not (tmp_path / "results/raw_responses").exists()


def test_limit(tmp_path: Path):
    cfg = _setup(tmp_path)
    stats = run(cfg, root=tmp_path, limit=3)
    assert stats.written == 3


def test_concurrent_workers_produce_same_results(tmp_path: Path):
    cfg = _setup(tmp_path)
    import yaml

    data = yaml.safe_load(cfg.read_text())
    data["workers"] = 4
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    stats = run(cfg, root=tmp_path)
    assert stats.written == 4 and stats.parse_failures == 0
    orig = io.read_jsonl(_orig_manifest(tmp_path), JudgeResult)
    bias = io.read_jsonl(_bias_manifest(tmp_path), JudgeResult)
    assert len(orig) == 2 and len(bias) == 2
    assert {r.result_id for r in orig + bias} | set()  # all ids present, no dupes
    assert len({r.result_id for r in orig + bias}) == 4


def test_api_failure_is_logged_and_not_marked_done(tmp_path: Path, monkeypatch):
    # An adapter whose calls always raise simulates transient API errors.
    import edit_judge_bias.experiments.run_scoring_judge as mod

    class FailingAdapter:
        model_name = "mock-judge"

        def score(self, prompt, images):
            raise RuntimeError("boom")

    monkeypatch.setattr(judge_common, "build_adapter", lambda *a, **k: FailingAdapter())
    cfg = _setup(tmp_path, score_biased=False)
    # add a failure log to the config
    import yaml

    data = yaml.safe_load(cfg.read_text())
    data["failure_log"] = "results/logs/fail.jsonl"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    stats = run(cfg, root=tmp_path)
    assert stats.api_failures == 2
    assert stats.written == 0
    # nothing marked done -> a retry run would attempt them again
    assert not _orig_manifest(tmp_path).exists()
    assert (tmp_path / "results/logs/fail.jsonl").read_text().count("\n") == 2


# --------------------------------------------------------------------------- #
# subset_filter / max_samples / repeat (full-version arms)                     #
# --------------------------------------------------------------------------- #
def _blocked_setup(root: Path, **cfg_extra) -> Path:
    """Four samples in two blocks, each with one biased image."""
    samples, biased = [], []
    for i in range(4):
        make_rgb_image(root / f"orig/{i}.png")
        make_rgb_image(root / f"edit/{i}.png")
        make_rgb_image(root / f"biased/{i}.png")
        samples.append(SampleRecord(
            sample_id=f"s{i}", source_dataset="I2EBench", edit_type="remove",
            content_category="human", original_image_path=f"orig/{i}.png",
            instruction="Remove the man.", edit_model="m",
            edited_image_path=f"edit/{i}.png",
            metadata={"subset_block": "breadth" if i < 2 else "anchor",
                      "anchor_source": None if i < 2 else "EBench-18K"},
        ))
        biased.append(BiasedRecord(
            biased_id=f"s{i}__brightness", base_sample_id=f"s{i}",
            bias_type="brightness", bias_strength=1.2,
            biased_image_path=f"biased/{i}.png",
        ))
    io.write_jsonl(root / "samples.jsonl", samples)
    io.write_jsonl(root / "biased.jsonl", biased)
    cfg = {
        "samples": "samples.jsonl", "biased": "biased.jsonl",
        "score_original": True, "score_biased": True, "prompt_style": "vanilla",
        "judge": {"type": "mock", "model_name": "mock-judge"},
        "raw_dir": "results/raw_responses", "results_dir": "results", "seed": 42,
        **cfg_extra,
    }
    cfg_path = root / "exp_blocks.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_subset_filter_selects_one_block_without_join_misses(tmp_path: Path):
    """The breadth and anchor blocks share one manifest but get different condition
    counts, so a filtered-out sample must not be reported as a dangling join."""
    cfg = _blocked_setup(tmp_path, subset_filter={"subset_block": "breadth"})
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.written == 4          # 2 samples x (baseline + brightness)
    assert stats.join_misses == 0


def test_subset_filter_accepts_a_list_of_values(tmp_path: Path):
    cfg = _blocked_setup(tmp_path, subset_filter={"anchor_source": ["EBench-18K"]})
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.written == 4          # the 2 anchored samples


def test_bias_types_narrows_the_conditions(tmp_path: Path):
    cfg = _blocked_setup(tmp_path, bias_types=["padding"])  # none injected
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.written == 4          # baselines only


def test_a_genuinely_dangling_biased_row_is_still_a_join_miss(tmp_path: Path):
    cfg = _blocked_setup(tmp_path, subset_filter={"subset_block": "breadth"})
    io.append_jsonl(tmp_path / "biased.jsonl", BiasedRecord(
        biased_id="ghost__brightness", base_sample_id="not_a_sample",
        bias_type="brightness", bias_strength=1.2, biased_image_path="biased/0.png",
    ))
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.join_misses == 1


def test_max_samples_caps_samples_not_targets(tmp_path: Path):
    """A repeat arm capped by --limit would take N baselines and zero repeats."""
    cfg = _blocked_setup(tmp_path, score_biased=False, max_samples=2, repeat=2)
    stats = run(cfg, root=tmp_path, dry_run=True)
    assert stats.written == 4          # 2 samples x 2 passes


def test_repeat_emits_distinct_result_ids(tmp_path: Path):
    cfg = _blocked_setup(tmp_path, score_biased=False, repeat=2)
    run(cfg, root=tmp_path)
    rows = io.read_jsonl(_orig_manifest(tmp_path), JudgeResult)
    ids = [r.result_id for r in rows]
    assert len(ids) == len(set(ids)) == 8
    repeats = [r for r in rows if r.bias_params.get("repeat_index") == 2]
    assert len(repeats) == 4 and all(r.result_id.endswith("::rep2") for r in repeats)


def test_repeat_refuses_to_run_against_a_warm_cache(tmp_path: Path):
    """A cached second ask returns the first answer verbatim, so the arm would
    report a noise floor of exactly zero that it never measured."""
    cfg = _blocked_setup(tmp_path, score_biased=False, repeat=2)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["judge"]["cache_dir"] = "results/cache"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    with __import__("pytest").raises(ValueError, match="cache"):
        run(cfg, root=tmp_path, dry_run=True)


def test_judge_config_argument_overrides_the_config_file(tmp_path: Path):
    cfg = _blocked_setup(tmp_path, score_biased=False)
    other = tmp_path / "judge_other.yaml"
    other.write_text(yaml.safe_dump(
        {"type": "mock", "model_name": "other-judge"}), encoding="utf-8")
    run(cfg, root=tmp_path, judge_config=other)
    assert (tmp_path / "results/raw_judgments/scoring__other-judge.jsonl").exists()
