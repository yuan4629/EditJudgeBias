"""Tests for judge-human agreement metrics (claim B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.metrics.agreement import (
    compute_agreement,
    compute_agreement_grid,
    cluster_bootstrap_ci,
    derive_pairs,
    build_items,
    spearman,
)


def _sample(sid, turn, human, source="EBench-18K"):
    return SampleRecord(
        sample_id=sid, source_dataset=source, edit_type="add",
        content_category="object", original_image_path=f"img/{turn}.png",
        instruction=f"instruction {turn}", edit_model=sid.split("_")[-1],
        edited_image_path=f"edit/{sid}.png", human_score=human,
    )


def _res(sid, score, *, bias=None, model="j", scale=10):
    return JudgeResult(
        result_id=f"{sid}::{bias or 'base'}", judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt", parse_success=True,
        sample_id=sid, overall_score=score, score_scale=scale, bias_type=bias,
    )


def _world(n_turns=6, editors=4, *, biased_shift=None):
    """Judge tracks the humans exactly; `biased_shift` optionally reverses it."""
    samples, base, biased = [], [], []
    for t in range(n_turns):
        for m in range(editors):
            sid = f"t{t}_m{m}"
            human = 0.1 * (m + 1)
            samples.append(_sample(sid, f"turn{t}", human))
            base.append(_res(sid, m + 1))
            if biased_shift is not None:
                biased.append(_res(sid, biased_shift(m), bias="padding"))
    return samples, base, biased


# --------------------------------------------------------------------------- #
# building blocks                                                             #
# --------------------------------------------------------------------------- #
def test_spearman_is_none_when_a_column_is_constant():
    """Realistic, not exotic: a judge answering "2" to everything has no ranking.
    None keeps that visible instead of poisoning a mean with nan."""
    assert spearman([1, 2, 3], [5, 5, 5]) is None
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_spearman_needs_three_points():
    assert spearman([1, 2], [1, 2]) is None


def test_cluster_bootstrap_resamples_clusters_not_items():
    """A statistic constant within clusters must have a degenerate interval no
    matter how many items each cluster holds."""
    clusters = [[1] * 8, [1] * 8, [1] * 8]
    ci = cluster_bootstrap_ci(clusters, lambda xs: sum(xs) / len(xs), n_boot=50)
    assert ci == (1.0, 1.0)


def test_cluster_bootstrap_needs_two_clusters():
    assert cluster_bootstrap_ci([[1, 2, 3]], lambda xs: sum(xs)) is None


# --------------------------------------------------------------------------- #
# derived pairs                                                                #
# --------------------------------------------------------------------------- #
def test_derived_pairs_stay_inside_a_turn():
    samples, base, biased = _world(n_turns=3, editors=4, biased_shift=lambda m: m + 1)
    items = build_items(samples, base, biased, judge_model="j", bias_type="padding")
    pairs = derive_pairs(items)
    assert len(pairs) == 3 * 6  # C(4,2) per turn, never across turns


def test_indecisive_human_gaps_are_dropped_not_scored_wrong():
    """A judge that disagrees about two edits the raters could not separate has
    not erred; counting it as an error dilutes every accuracy toward 50%."""
    samples = [_sample("a", "t", 0.50), _sample("b", "t", 0.5001),
               _sample("c", "t", 0.90)]
    base = [_res("a", 1), _res("b", 2), _res("c", 3)]
    items = build_items(samples, base, [], judge_model="j", bias_type="padding")
    # SD of {0.50, 0.5001, 0.90} is ~0.23, so 0.5 SD ~ 0.115: a-vs-b is out.
    prefs = derive_pairs(items, source_sd={"EBench-18K": 0.23})
    assert len(prefs) == 2


# --------------------------------------------------------------------------- #
# the reported row                                                            #
# --------------------------------------------------------------------------- #
def test_perfect_agreement_is_unchanged_by_a_harmless_bias():
    samples, base, biased = _world(biased_shift=lambda m: m + 1)
    st = compute_agreement(samples, base, biased, judge_model="j",
                           bias_type="padding", n_boot=100)
    assert st.spearman_original == pytest.approx(1.0)
    assert st.spearman_biased == pytest.approx(1.0)
    assert st.spearman_delta == pytest.approx(0.0)
    assert st.accuracy_original == 1.0 and st.accuracy_biased == 1.0
    assert st.mcnemar_b == 0 and st.mcnemar_c == 0


def test_a_bias_that_reverses_the_ranking_shows_up_in_both_measures():
    samples, base, biased = _world(biased_shift=lambda m: 4 - m)
    st = compute_agreement(samples, base, biased, judge_model="j",
                           bias_type="padding", n_boot=100)
    assert st.spearman_original == pytest.approx(1.0)
    assert st.spearman_biased == pytest.approx(-1.0)
    assert st.spearman_delta == pytest.approx(-2.0)
    assert st.accuracy_biased == 0.0
    assert st.mcnemar_b == st.n_pairs and st.mcnemar_c == 0
    assert st.mcnemar_p_unclustered is not None and st.mcnemar_p_unclustered < 0.05
    # The cluster-level test is the one that gates the claim, so it has to fire here too:
    # every turn moved the same way, which is as clean a signal as this design can carry.
    assert st.accuracy_p_cluster is not None and st.accuracy_p_cluster < 0.05
    assert st.n_clusters_graded == st.n_clusters


def test_damage_confined_to_one_turn_fools_the_pair_level_test_but_not_the_cluster_one():
    """★ The defect this table was published with, reduced to its smallest form.

    `derive_pairs` explodes one turn of E editors into C(E,2) within-turn comparisons. Put
    ALL the damage in a single turn and the pair-level McNemar sees C(E,2) discordant trials
    pointing the same way and calls it significant — but the evidence is ONE turn, and the
    cluster-level test says so. On the real anchor blocks (8 editors, ~18 comparisons/turn)
    this inflation is what took claim B's accuracy half to 13/40 BH-significant cells, 10 of
    which contradicted their own correctly-clustered CI.
    """
    samples, base, biased = [], [], []
    for t in range(8):
        for m in range(4):
            sid = f"t{t}_m{m}"
            samples.append(_sample(sid, f"turn{t}", 0.1 * (m + 1)))
            base.append(_res(sid, m + 1))
            # Only turn 0 is damaged; everywhere else the bias is a harmless +1.
            biased.append(_res(sid, (4 - m) if t == 0 else (m + 1), bias="padding"))

    st = compute_agreement(samples, base, biased, judge_model="j",
                           bias_type="padding", n_boot=100)
    assert st.mcnemar_b == 6 and st.mcnemar_c == 0           # C(4,2) from the one bad turn
    assert st.mcnemar_p_unclustered < 0.05                    # "significant" on 1 turn of data
    assert st.n_clusters_graded == 8
    assert st.accuracy_p_cluster is not None
    assert st.accuracy_p_cluster > 0.05                       # one cluster is not evidence


def test_cluster_net_discordance_is_one_value_per_turn():
    """Signed per turn: fixes minus breaks, so a turn where the bias does both cancels."""
    from edit_judge_bias.metrics.agreement import DerivedPair, cluster_net_discordance

    def pair(turn, before_ok, after_ok):
        # human prefers "a"; the judge is right when it also says "a"
        return DerivedPair(turn=(turn, "i"), human_pref="a",
                           judge_pref_original="a" if before_ok else "b",
                           judge_pref_biased="a" if after_ok else "b")

    graded = [pair("T1", True, False), pair("T1", True, False),   # T1: two breaks   -> -2
              pair("T2", True, False), pair("T2", False, True),   # T2: one each     ->  0
              pair("T3", False, True)]                            # T3: one fix      -> +1
    assert cluster_net_discordance(graded) == [-2.0, 0.0, 1.0]


def test_clusters_are_turns_not_items():
    samples, base, biased = _world(n_turns=6, editors=4, biased_shift=lambda m: m + 1)
    st = compute_agreement(samples, base, biased, judge_model="j",
                           bias_type="padding", n_boot=50)
    assert st.n_items == 24 and st.n_clusters == 6


def test_samples_without_a_human_score_are_skipped():
    samples, base, biased = _world(n_turns=2, editors=4, biased_shift=lambda m: m + 1)
    samples[0] = samples[0].model_copy(update={"human_score": None})
    st = compute_agreement(samples, base, biased, judge_model="j",
                           bias_type="padding", n_boot=50)
    assert st.n_items == 7


def test_mixing_score_scales_raises():
    samples, base, biased = _world(n_turns=2, editors=4, biased_shift=lambda m: m + 1)
    biased[0] = biased[0].model_copy(update={"score_scale": 5})
    with pytest.raises(ValueError, match="score scales"):
        compute_agreement(samples, base, biased, judge_model="j", bias_type="padding")


def test_grid_covers_every_judge_and_bias_present():
    samples, base, _ = _world(n_turns=3, editors=4)
    biased = []
    for model in ("j", "k"):
        for bias in ("padding", "sham"):
            for t in range(3):
                for m in range(4):
                    biased.append(_res(f"t{t}_m{m}", m + 1, bias=bias, model=model))
            base = base + [_res(f"t{t}_m{m}", m + 1, model=model)
                           for t in range(3) for m in range(4)]
    rows = compute_agreement_grid(samples, base, biased, n_boot=20)
    assert {(r.judge_model, r.bias_type) for r in rows} == {
        ("j", "padding"), ("j", "sham"), ("k", "padding"), ("k", "sham")
    }


def test_as_row_is_flat_and_csv_ready():
    samples, base, biased = _world(n_turns=3, editors=4, biased_shift=lambda m: m + 1)
    row = compute_agreement(samples, base, biased, judge_model="j",
                            bias_type="padding", n_boot=20).as_row()
    assert {"judge_model", "bias_type", "n_clusters", "spearman_delta",
            "accuracy_delta", "accuracy_p_cluster", "mcnemar_p_unclustered",
            "n_clusters_graded"} <= set(row)
    # The pair-level p must never reappear under a name that hides its assumption.
    assert "mcnemar_p" not in row
    assert all(not isinstance(v, (list, dict)) for v in row.values())


def test_a_judge_that_always_ties_scores_zero_and_says_so():
    """The mock judge scores from a prompt hash, and every editor in a turn shares
    an instruction — so it answers "tie" to every comparison. Accuracy 0.0 with a
    tie rate of 1.0 is the honest reading; accuracy 0.0 alone looks like a bug."""
    samples, base, biased = _world(n_turns=3, editors=4, biased_shift=lambda m: 7)
    base = [_res(sid.sample_id, 7) for sid in samples]
    st = compute_agreement(samples, base, biased, judge_model="j",
                           bias_type="padding", n_boot=20)
    assert st.tie_rate_original == 1.0 and st.accuracy_original == 0.0
