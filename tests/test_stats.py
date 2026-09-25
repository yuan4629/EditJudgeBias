"""Tests for statistical helpers (Milestone 5)."""

from __future__ import annotations

from edit_judge_bias.metrics.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    mcnemar_pvalue,
    wilcoxon_pvalue,
)


def test_bootstrap_ci_brackets_mean():
    vals = [1.0, 1.0, 1.0, 1.0]
    lo, hi = bootstrap_ci(vals, seed=1)
    assert lo == hi == 1.0  # zero variance


def test_bootstrap_ci_none_for_singleton():
    assert bootstrap_ci([1.0]) is None


def test_wilcoxon_detects_consistent_positive_shift():
    p = wilcoxon_pvalue([1, 2, 2, 3, 1, 2, 3, 2])
    assert p is not None and p < 0.05


def test_wilcoxon_none_when_all_zero():
    assert wilcoxon_pvalue([0, 0, 0]) is None


def test_mcnemar_symmetric_is_nonsignificant():
    assert mcnemar_pvalue(5, 5) == 1.0


def test_mcnemar_extreme_is_significant():
    p = mcnemar_pvalue(10, 0)
    assert p is not None and p < 0.05


def test_mcnemar_none_when_no_discordant():
    assert mcnemar_pvalue(0, 0) is None


def test_bh_largest_pvalue_is_unchanged():
    # q_(m) = p_(m) * m/m; the rest are pulled up from it.
    qs = benjamini_hochberg([0.01, 0.02, 0.03, 0.04])
    assert qs[-1] == 0.04
    assert all(q >= p for q, p in zip(qs, [0.01, 0.02, 0.03, 0.04]))


def test_bh_is_monotone_in_p():
    qs = benjamini_hochberg([0.001, 0.9, 0.04, 0.02])
    ordered = [q for _, q in sorted(zip([0.001, 0.9, 0.04, 0.02], qs))]
    assert ordered == sorted(ordered)


def test_bh_keeps_none_in_place_and_excludes_it_from_m():
    """A cell whose test was undefined must not shift its neighbours' q-values.

    Dropping the None, correcting the rest and zipping back is the obvious
    implementation and it silently pairs each q with the WRONG row.
    """
    with_none = benjamini_hochberg([0.01, None, 0.02])
    without = benjamini_hochberg([0.01, 0.02])
    assert with_none[1] is None
    assert [with_none[0], with_none[2]] == without  # m counted the 2 real tests


def test_bh_all_none_returns_all_none():
    assert benjamini_hochberg([None, None]) == [None, None]


def test_bh_empty():
    assert benjamini_hochberg([]) == []


def test_bh_never_exceeds_one():
    assert all(q <= 1.0 for q in benjamini_hochberg([0.6, 0.7, 0.99]))
