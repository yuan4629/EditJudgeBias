"""Tests for statistical helpers (Milestone 5)."""

from __future__ import annotations

import pytest

from edit_judge_bias.metrics.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    mcnemar_pvalue,
    minimum_detectable_effect,
    wilcoxon_pvalue,
    wilson_interval,
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


@pytest.mark.parametrize(
    "successes,n,lo,hi",
    [
        # Derived from the closed form independently of the implementation:
        #   d = 1 + z^2/n ; centre = (p + z^2/2n)/d ; half = z*sqrt(p(1-p)/n + z^2/4n^2)/d
        # with z = 1.959963985. (20,20) matching the textbook 0.8389 is the anchor.
        (20, 20, 0.838875, 1.0),       # p=1.0 must NOT get a zero-width interval
        (0, 20, 0.0, 0.161125),        # p=0.0 likewise
        (8, 20, 0.218807, 0.613418),
        (1, 31, 0.005717, 0.161941),
        (17, 20, 0.639581, 0.947631),
        (27, 41, 0.505498, 0.784412),
    ],
)
def test_wilson_interval_values(successes, n, lo, hi):
    got_lo, got_hi = wilson_interval(successes, n)
    assert got_lo == pytest.approx(lo, abs=1e-5)
    assert got_hi == pytest.approx(hi, abs=1e-5)


def test_wilson_interval_contains_the_point_estimate_and_is_empty_at_n_zero():
    for successes, n in [(3, 7), (8, 20), (30, 31)]:
        lo, hi = wilson_interval(successes, n)
        assert lo <= successes / n <= hi
    assert wilson_interval(0, 0) == (None, None)


def test_mde_scales_as_one_over_sqrt_n():
    """One number, derived one way, so the write-up cannot quote two different MDEs."""
    assert minimum_detectable_effect(611) == pytest.approx(0.113)
    assert minimum_detectable_effect(41) == pytest.approx(0.436, abs=1e-3)
    assert minimum_detectable_effect(0) is None
