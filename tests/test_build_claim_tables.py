"""Tests for the frozen claim tables (claim A / claim B / pairwise / retest)."""

# ⚠️ Every builder call below passes `roster=None` on purpose.  The default is
# `PUBLISHED_ROSTER`, which is the five judges whose numbers are in the paper -- that
# default exists so a forgotten argument silently OMITS a new judge (loudly: it prints
# a note) rather than silently ADMITTING one into a published BH family, which is the
# corruption the roster was added to prevent.  These tests use fixture judge names, so
# they are about the mechanics and must opt out of family membership explicitly.


from __future__ import annotations

import csv
from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_claim_tables import (
    attach_bh,
    build_all,
    build_claim_a,
    build_claim_b,
    build_pairwise,
    build_retest,
)


def _score(model, sid, score, bias=None, *, scale=10, fine=None, repeat=1):
    return JudgeResult(
        result_id=f"{model}::{sid}::{bias}::{repeat}", judge_model=model,
        task_type="scoring", prompt_type="vanilla_scoring", raw_response_path="r.txt",
        parse_success=True, sample_id=sid, bias_type=bias,
        biased_id=f"{sid}__{bias}" if bias else None,
        overall_score=score, fine_score=fine, score_scale=scale,
        bias_params={"repeat_index": repeat} if repeat > 1 else {},
    )


def _pw(model, pair_id, winner, *, bias=None, side=None):
    return JudgeResult(
        result_id=f"{model}::{pair_id}::{bias}::{side}", judge_model=model,
        task_type="pairwise", prompt_type="vanilla_pairwise", raw_response_path="r.txt",
        parse_success=True, pair_id=pair_id, winner=winner,
        bias_type=bias, biased_side=side,
    )


def _sample(sid, *, block=None, anchor=None, human=None):
    rec = SampleRecord(
        sample_id=sid, source_dataset="src", edit_type="add", content_category="object",
        original_image_path=f"{sid}_o.jpg", instruction="do", edit_model="m",
        edited_image_path=f"{sid}_e.jpg", human_score=human,
    )
    if block:
        rec.metadata.subset_block = block  # type: ignore[attr-defined]
    if anchor:
        rec.metadata.anchor_source = anchor  # type: ignore[attr-defined]
    return rec


def _write_scoring(root: Path, model: str, originals, biased) -> None:
    io.write_jsonl(root / "raw_judgments" / f"scoring__{model}.jsonl", originals)
    io.write_jsonl(root / "biased_judgments" / f"scoring__{model}.jsonl", biased)


# --------------------------------------------------------------------------- #
# claim A                                                                     #
# --------------------------------------------------------------------------- #
def test_claim_a_carries_raw_placebo_and_the_equivalence_bound(tmp_path: Path):
    results = tmp_path / "v2"
    originals = [_score("gpt-5.5", f"s{i}", 5) for i in range(6)]
    biased = (
        # sham drifts UP by 1 on every sample: the placebo is not zero.
        [_score("gpt-5.5", f"s{i}", 6, "sham") for i in range(6)]
        # padding sits at the baseline: raw shift 0, but -1 against the placebo.
        + [_score("gpt-5.5", f"s{i}", 5, "padding") for i in range(6)]
    )
    _write_scoring(results, "gpt-5.5", originals, biased)

    rows = {r["bias_type"]: r for r in build_claim_a(results, roster=None)}
    pad = rows["padding"]
    assert pad["mean_shift"] == 0.0                 # headline: nothing happened
    assert pad["shift_vs_placebo"] == -1.0          # robustness: it did, vs the placebo
    assert pad["control_shift"] == 1.0
    assert pad["inside_placebo_bound"] is False     # 0.0 is outside [1.0, 1.0]
    assert rows["sham"]["family"] == "control"      # the control is not a hypothesis


def test_claim_a_excludes_the_control_from_the_bh_family(tmp_path: Path):
    results = tmp_path / "v2"
    originals = [_score("m", f"s{i}", 5) for i in range(8)]
    biased = ([_score("m", f"s{i}", 6, "sham") for i in range(8)]
              + [_score("m", f"s{i}", 3, "padding") for i in range(8)]
              + [_score("m", f"s{i}", 4, "text_overlay") for i in range(8)])
    _write_scoring(results, "m", originals, biased)

    rows = {r["bias_type"]: r for r in build_claim_a(results, roster=None)}
    assert rows["sham"]["q_value"] is None
    # m = 2, not 3: the control must not inflate the family it is the yardstick for.
    assert rows["padding"]["family"] == "claim_A"
    assert rows["padding"]["q_value"] is not None


def test_claim_a_respects_the_subset_filter(tmp_path: Path):
    results = tmp_path / "v2"
    originals = [_score("m", "s_breadth", 5), _score("m", "s_anchor", 5)]
    biased = [_score("m", "s_breadth", 3, "padding"),
              _score("m", "s_anchor", 9, "padding")]
    _write_scoring(results, "m", originals, biased)

    pooled = {r["bias_type"]: r for r in build_claim_a(results, roster=None)}
    assert pooled["padding"]["n"] == 2 and pooled["padding"]["mean_shift"] == 1.0

    kept = {r["bias_type"]: r for r in build_claim_a(results, {"s_breadth"}, roster=None)}
    assert kept["padding"]["n"] == 1 and kept["padding"]["mean_shift"] == -2.0


def test_claim_a_refuses_to_mix_analysis_variables(tmp_path: Path):
    """One judge resolving to overall_score while the rest stay on fine_score puts
    3-30 sums and 1-10 overalls in one mean_shift column."""
    results = tmp_path / "v2"
    _write_scoring(results, "fine",
                   [_score("fine", f"s{i}", 5, fine=15) for i in range(4)],
                   [_score("fine", f"s{i}", 4, "padding", fine=12) for i in range(4)])
    _write_scoring(results, "coarse",
                   [_score("coarse", f"s{i}", 5) for i in range(4)],
                   [_score("coarse", f"s{i}", 4, "padding") for i in range(4)])
    with pytest.raises(ValueError, match="mix analysis variables"):
        build_claim_a(results, roster=None)


# --------------------------------------------------------------------------- #
# claim B                                                                     #
# --------------------------------------------------------------------------- #
def test_claim_b_filters_on_anchor_source_not_on_the_block(tmp_path: Path):
    """39 of the 624 anchor samples were ALSO drawn into the breadth block. Filtering
    claim B by `subset_block == anchor` would silently drop them from the source they
    belong to, so the filter is `anchor_source` and only `anchor_source`."""
    results = tmp_path / "v2"
    samples = [
        _sample("shared", block="breadth", anchor="EBench-18K", human=0.9),
        _sample("anchor1", block="anchor", anchor="EBench-18K", human=0.5),
        _sample("anchor2", block="anchor", anchor="EBench-18K", human=0.1),
        _sample("plain", block="breadth", human=0.7),  # no anchor: not claim B's
    ]
    ids = ["shared", "anchor1", "anchor2", "plain"]
    _write_scoring(
        results, "m",
        [_score("m", sid, 9 - 2 * i) for i, sid in enumerate(ids)],
        [_score("m", sid, 5, "padding") for sid in ids],
    )
    [row] = build_claim_b(samples, results, biases=("padding",), n_boot=20, roster=None)
    assert row["anchor_source"] == "EBench-18K"
    assert row["n_paired"] == 3, "the shared sample must not be dropped"


def test_claim_b_reports_whether_the_rho_ci_excludes_zero(tmp_path: Path):
    results = tmp_path / "v2"
    samples = [_sample(f"s{i}", anchor="A", human=float(i)) for i in range(8)]
    _write_scoring(
        results, "m",
        [_score("m", f"s{i}", i + 1) for i in range(8)],       # perfectly ranked
        [_score("m", f"s{i}", 8 - i, "padding") for i in range(8)],  # exactly reversed
    )
    [row] = build_claim_b(samples, results, biases=("padding",), n_boot=50, roster=None)
    assert row["spearman_original"] == 1.0 and row["spearman_biased"] == -1.0
    assert row["rho_ci_excludes_zero"] is True


# --------------------------------------------------------------------------- #
# pairwise + retest                                                           #
# --------------------------------------------------------------------------- #
def test_pairwise_tables_carry_both_bh_families(tmp_path: Path):
    results = tmp_path / "v2"
    io.write_jsonl(results / "raw_judgments" / "pairwise__m.jsonl",
                   [_pw("m", f"p{i}", "A") for i in range(4)])
    io.write_jsonl(results / "biased_judgments" / "pairwise__m.jsonl",
                   [_pw("m", f"p{i}", "B", bias="position", side="swap") for i in range(4)]
                   + [_pw("m", f"p{i}", "B", bias="padding", side="b") for i in range(4)])

    tables = build_pairwise(results, decisive_prefs={f"p{i}": "a" for i in range(4)},
                            roster=None)
    [one] = tables["one_sided"]
    assert one["bias_advantage"] == 1.0
    assert "q_value_advantage" in one and "q_value_accuracy" in one
    assert "q_value" not in one, "an unqualified q_value would not say which family"
    [joint] = tables["position_joint"]
    assert joint["same_slot_rate"] == 0.0 and joint["count_A_B"] == 4


def test_retest_table_is_per_judge(tmp_path: Path):
    results = tmp_path / "v2"
    io.write_jsonl(results / "raw_judgments" / "scoring__m.jsonl", [
        _score("m", "s0", 3), _score("m", "s0", 5, repeat=2),
        _score("m", "s1", 4), _score("m", "s1", 4, repeat=2),
    ])
    [row] = build_retest(results, roster=None)
    assert row["n"] == 2 and row["mean_abs_delta"] == 1.0
    assert row["identical_rate"] == 0.5 and row["mean_delta"] == 1.0


def test_attach_bh_stamps_the_family_on_every_row():
    rows = [{"p": 0.001}, {"p": 0.9}, {"p": None}]
    attach_bh(rows, family="demo", p_key="p")
    assert [r["family"] for r in rows] == ["demo"] * 3
    assert rows[0]["significant_bh"] is True and rows[1]["significant_bh"] is False
    assert rows[2]["q_value"] is None and rows[2]["significant_bh"] is None


# --------------------------------------------------------------------------- #
# driver                                                                      #
# --------------------------------------------------------------------------- #
def test_build_all_writes_six_tables(tmp_path: Path):
    results = tmp_path / "v2"
    samples_path = tmp_path / "samples.jsonl"
    io.write_jsonl(samples_path, [
        _sample("s0", block="breadth", anchor="A", human=0.1),
        _sample("s1", block="breadth", anchor="A", human=0.5),
        _sample("s2", block="breadth", anchor="A", human=0.9),
    ])
    ids = ["s0", "s1", "s2"]
    _write_scoring(
        results, "m",
        [_score("m", sid, 3 + i) for i, sid in enumerate(ids)],
        [_score("m", sid, 5, "sham") for sid in ids]
        + [_score("m", sid, 2, "padding") for sid in ids],
    )
    io.write_jsonl(results / "raw_judgments" / "pairwise__m.jsonl", [_pw("m", "p0", "A")])
    io.write_jsonl(results / "biased_judgments" / "pairwise__m.jsonl",
                   [_pw("m", "p0", "B", bias="position", side="swap")])

    build_all(results, samples_path=samples_path, roster=None,
              subset_filter={"subset_block": "breadth"}, n_boot=20)
    metrics = results / "metrics"
    for name in ("claim_a", "claim_b", "pairwise_one_sided", "position",
                 "position_joint", "retest"):
        assert (metrics / f"{name}.csv").exists(), name
    rows = list(csv.DictReader((metrics / "claim_a.csv").open(encoding="utf-8")))
    assert {r["bias_type"] for r in rows} == {"sham", "padding"}
