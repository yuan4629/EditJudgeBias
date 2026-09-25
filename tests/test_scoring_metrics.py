"""Tests for scoring metrics (Milestone 5)."""

from __future__ import annotations

import pytest

from edit_judge_bias.data.schema import JudgeResult
from edit_judge_bias.metrics.scoring_metrics import (
    compute_placebo_contrast,
    compute_retest_stats,
    compute_score_shifts,
)


def _orig(model, sid, score, parsed=True):
    return JudgeResult(
        result_id=f"o::{model}::{sid}", judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt",
        parse_success=parsed, sample_id=sid, overall_score=score if parsed else None,
    )


def _biased(model, sid, bias, score, parsed=True):
    return JudgeResult(
        result_id=f"b::{model}::{sid}::{bias}", judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt",
        parse_success=parsed, sample_id=sid, bias_type=bias, biased_id=f"{sid}__{bias}",
        overall_score=score if parsed else None,
    )


def test_basic_shift_sir_asc():
    origs = [_orig("m", "s0", 3), _orig("m", "s1", 3)]
    biased = [_biased("m", "s0", "brightness", 5), _biased("m", "s1", "brightness", 4)]
    [st] = compute_score_shifts(origs, biased)
    assert st.judge_model == "m" and st.bias_type == "brightness"
    assert st.n == 2
    assert st.mean_shift == 1.5      # (2 + 1) / 2
    assert st.sir == 1.0             # both biased > original
    assert st.asc == 1.5


def test_negative_and_zero_shifts():
    origs = [_orig("m", "s0", 4), _orig("m", "s1", 4), _orig("m", "s2", 4)]
    biased = [_biased("m", "s0", "padding", 5), _biased("m", "s1", "padding", 4),
              _biased("m", "s2", "padding", 2)]
    [st] = compute_score_shifts(origs, biased)
    assert st.mean_shift == (1 + 0 - 2) / 3
    assert st.sir == 1 / 3           # only s0 inflated
    assert st.asc == (1 + 0 + 2) / 3


def test_unpaired_biased_is_skipped():
    origs = [_orig("m", "s0", 3)]
    biased = [_biased("m", "s0", "b", 4), _biased("m", "s_missing", "b", 5)]
    [st] = compute_score_shifts(origs, biased)
    assert st.n == 1


def test_parse_failures_excluded():
    origs = [_orig("m", "s0", 3), _orig("m", "s1", 3, parsed=False)]
    biased = [_biased("m", "s0", "b", 5), _biased("m", "s1", "b", 5, parsed=False)]
    [st] = compute_score_shifts(origs, biased)
    assert st.n == 1


def test_groups_by_model_and_bias():
    origs = [_orig("m1", "s0", 3), _orig("m2", "s0", 3)]
    biased = [_biased("m1", "s0", "brightness", 4), _biased("m2", "s0", "padding", 5)]
    stats = compute_score_shifts(origs, biased)
    keys = {(s.judge_model, s.bias_type) for s in stats}
    assert keys == {("m1", "brightness"), ("m2", "padding")}


def test_as_row_keys():
    origs = [_orig("m", "s0", 3), _orig("m", "s1", 2)]
    biased = [_biased("m", "s0", "b", 5), _biased("m", "s1", "b", 4)]
    row = compute_score_shifts(origs, biased)[0].as_row()
    assert {"judge_model", "bias_type", "n", "mean_shift", "sir", "asc", "p_value"} <= set(row)


# --------------------------------------------------------------------------- #
# score scale / analysis variable                                              #
# --------------------------------------------------------------------------- #
def _scaled(rec, scale, fine=None):
    return rec.model_copy(update={"score_scale": scale, "fine_score": fine})


def test_mixing_score_scales_raises():
    """A 1-5 pilot judgment and a 1-10 main-grid judgment are not commensurable;
    pooling them would produce a mean_shift with no unit."""
    origs = [_scaled(_orig("m", "s0", 3), 5), _scaled(_orig("m", "s1", 3), 10)]
    biased = [_scaled(_biased("m", "s0", "b", 4), 5)]
    with pytest.raises(ValueError, match="score scales"):
        compute_score_shifts(origs, biased)


def test_fine_score_preferred_when_every_row_has_one():
    origs = [_scaled(_orig("m", "s0", 3), 10, fine=9)]
    biased = [_scaled(_biased("m", "s0", "b", 2), 10, fine=6)]
    [st] = compute_score_shifts(origs, biased)
    assert st.mean_shift == -3.0  # 6 - 9, i.e. on the 3-30 sum, not the 1-10 overall


def test_falls_back_to_overall_when_fine_score_absent():
    """Archived pilot rows carry no fine_score; they must still aggregate."""
    origs = [_orig("m", "s0", 3), _orig("m", "s1", 3)]
    biased = [_biased("m", "s0", "b", 2), _biased("m", "s1", "b", 2)]
    [st] = compute_score_shifts(origs, biased)
    assert st.mean_shift == -1.0


def test_a_few_rows_missing_fine_score_are_dropped_not_the_variable():
    """MEASURED on the breadth grid: kimi-k2.5 omitted `detail_preservation` in 4 of
    611 baseline answers (the parser accepts that — overall_score was present). The
    old all-or-nothing rule read those 4 rows as proof the judge had no fine_score and
    fell back to `overall_score` for kimi ALONE, so one mean_shift column carried both
    3-30 sums and 1-10 overalls and kimi looked ~10x more robust than it is."""
    origs = [_scaled(_orig("m", f"s{i}", 3), 10, fine=9) for i in range(40)]
    biased = [_scaled(_biased("m", f"s{i}", "b", 2), 10, fine=6) for i in range(40)]
    origs.append(_scaled(_orig("m", "s40", 3), 10))          # no fine_score: 1 of 41
    biased.append(_scaled(_biased("m", "s40", "b", 2), 10))
    [st] = compute_score_shifts(origs, biased)
    assert st.score_field == "fine_score"
    assert st.n == 40                # the incomplete item is dropped, not the variable
    assert st.mean_shift == -3.0     # on the 3-30 sum


def test_many_rows_missing_fine_score_raises_instead_of_degrading():
    """Past the drop threshold it is no longer a handful of bad answers, and quietly
    switching that judge to a coarser variable is what produced the mixed column."""
    origs = [_scaled(_orig("m", "s0", 3), 10, fine=9), _scaled(_orig("m", "s1", 3), 10)]
    biased = [_scaled(_biased("m", "s0", "b", 2), 10, fine=6),
              _scaled(_biased("m", "s1", "b", 2), 10)]
    with pytest.raises(ValueError, match="lack 'fine_score'"):
        compute_score_shifts(origs, biased)


def test_explicit_score_field_overrides_resolution():
    origs = [_scaled(_orig("m", "s0", 3), 10, fine=9), _scaled(_orig("m", "s1", 3), 10)]
    biased = [_scaled(_biased("m", "s0", "b", 2), 10, fine=6),
              _scaled(_biased("m", "s1", "b", 2), 10)]
    [st] = compute_score_shifts(origs, biased, score_field="overall_score")
    assert st.score_field == "overall_score" and st.n == 2 and st.mean_shift == -1.0


# --------------------------------------------------------------------------- #
# placebo contrast (bias vs sham, paired within sample)                        #
# --------------------------------------------------------------------------- #
def test_placebo_contrast_is_paired_within_sample():
    biased = [
        _biased("m", "s0", "sham", 5), _biased("m", "s1", "sham", 3),
        _biased("m", "s0", "padding", 2), _biased("m", "s1", "padding", 2),
    ]
    [st] = compute_placebo_contrast(biased)
    assert st.bias_type == "padding" and st.control_bias == "sham"
    assert st.n == 2
    assert st.mean_contrast == (2 - 5 + 2 - 3) / 2  # -2.0, not mean(padding)-mean(sham)
    assert st.mean_control == 4.0


def test_placebo_contrast_excludes_the_control_cell_itself():
    """sham vs sham is identically zero and would only pad the BH family."""
    biased = [_biased("m", "s0", "sham", 5), _biased("m", "s0", "padding", 4)]
    assert [st.bias_type for st in compute_placebo_contrast(biased)] == ["padding"]


def test_placebo_contrast_needs_both_halves_of_the_pair():
    biased = [
        _biased("m", "s0", "sham", 5),
        _biased("m", "s0", "padding", 4), _biased("m", "s1", "padding", 4),
    ]
    [st] = compute_placebo_contrast(biased)
    assert st.n == 1  # s1 has no sham answer to contrast against


def test_placebo_contrast_skips_a_judge_with_no_control_arm():
    """A judge whose sham cell failed has no placebo; it must be absent, not zero."""
    biased = [_biased("m1", "s0", "sham", 5), _biased("m1", "s0", "padding", 4),
              _biased("m2", "s0", "padding", 4)]
    assert {st.judge_model for st in compute_placebo_contrast(biased)} == {"m1"}


def test_placebo_contrast_rejects_mixed_scales():
    biased = [_scaled(_biased("m", "s0", "sham", 5), 5),
              _scaled(_biased("m", "s0", "padding", 4), 10)]
    with pytest.raises(ValueError, match="score scales"):
        compute_placebo_contrast(biased)


# --------------------------------------------------------------------------- #
# test-retest noise floor                                                      #
# --------------------------------------------------------------------------- #
def _repeat(rec, index):
    return rec.model_copy(update={
        "result_id": f"{rec.result_id}::rep{index}",
        "bias_params": {"repeat_index": index},
    })


def test_retest_reports_absolute_movement_and_signed_drift():
    """Both numbers are needed: gemini moves 3.34 points per item with NO net drift,
    gpt-5.5 moves 2.90 WITH a +1.34 drift, and only the second is a threat to a
    before-after design."""
    rows = []
    for i, (first, second) in enumerate([(3, 5), (4, 4), (2, 5)]):
        base = _orig("m", f"s{i}", first)
        rows += [base, _repeat(base.model_copy(update={"overall_score": second}), 2)]
    [st] = compute_retest_stats(rows)
    assert st.n == 3
    assert st.mean_abs_delta == (2 + 0 + 3) / 3
    assert st.identical_rate == 1 / 3
    assert st.mean_delta == (2 + 0 + 3) / 3  # all non-negative here: a real drift


def test_retest_ignores_samples_asked_only_once():
    rows = [_orig("m", "s0", 3)]
    assert compute_retest_stats(rows) == []


def test_retest_ignores_biased_rows():
    """The repeat arm re-asks BASELINES; a biased row in the same file is a different
    question and its difference is a bias effect, not noise."""
    base = _orig("m", "s0", 3)
    rows = [base, _repeat(base.model_copy(update={"overall_score": 4}), 2),
            _biased("m", "s0", "padding", 9)]
    [st] = compute_retest_stats(rows)
    assert st.n == 1 and st.mean_delta == 1.0
