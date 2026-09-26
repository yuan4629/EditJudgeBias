"""Statistical helpers for the metrics layer.

Bootstrap CIs and a Wilcoxon signed-rank test for paired score shifts, a McNemar
test for pairwise flips, and a Benjamini-Hochberg correction for the fact that the
main grid runs dozens of them. All functions are defensive: degenerate inputs
(empty, all-zero, all-equal) return None rather than raising, so a sparse pilot
cell never crashes aggregation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap CI for the mean. None if fewer than 2 values."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_means(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """The bootstrap distribution of the mean that :func:`bootstrap_ci` summarises.

    Same generator and draw order as :func:`bootstrap_ci`, so its 2.5% / 97.5%
    quantiles are that function's interval. Used where a quantile other than the CI
    endpoints is needed (the noise floors). None if fewer than 2 values.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    return arr[idx].mean(axis=1)


def wilcoxon_pvalue(shifts: Sequence[float]) -> Optional[float]:
    """Wilcoxon signed-rank p-value that the median shift differs from 0.

    None if there is no non-zero difference (test undefined) or too few samples.
    """
    arr = np.asarray(shifts, dtype=float)
    nonzero = arr[arr != 0]
    if nonzero.size < 1:
        return None
    from scipy.stats import wilcoxon

    try:
        return float(wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return None


def benjamini_hochberg(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """BH-adjusted q-values, POSITIONALLY ALIGNED with the input.

    §8.3 asks for Benjamini-Hochberg and the grid needs it: claim A alone is 5
    judges x 12 cues = 60 tests, so at alpha=0.05 three "significant" cells are
    expected from noise even if no cue did anything. The family a q-value belongs
    to is a reporting decision, not a property of the number — callers pass one
    family at a time and stamp its name next to the column.

    `None` p-values (a test that was undefined: an empty cell, an all-zero shift)
    are passed through as `None` and are excluded from `m`. Dropping them and
    zipping the result back would shift every q onto the wrong row, which is the
    same class of silent misalignment as the joins documented in
    :mod:`edit_judge_bias.metrics.pairwise_metrics` — hence the index bookkeeping
    below rather than a comprehension.

    Monotonicity is enforced by the usual reverse cumulative minimum, so a q-value
    is never smaller than that of a larger p-value in the same family, and every
    q is clamped to <= 1.
    """
    indexed = [(i, float(p)) for i, p in enumerate(pvalues) if p is not None]
    out: List[Optional[float]] = [None] * len(pvalues)
    m = len(indexed)
    if m == 0:
        return out
    indexed.sort(key=lambda ip: ip[1])
    running = 1.0
    # Walk from the largest p down: q_(k) = min(q_(k+1), p_(k) * m / k).
    for rank in range(m, 0, -1):
        idx, p = indexed[rank - 1]
        running = min(running, p * m / rank)
        out[idx] = min(1.0, running)
    return out


def mcnemar_pvalue(b: int, c: int) -> Optional[float]:
    """Two-sided McNemar exact/asymptotic p-value for discordant counts (b, c).

    b, c are the off-diagonal counts (one direction of flip vs the other).
    None if there are no discordant pairs.
    """
    n = b + c
    if n == 0:
        return None
    from scipy.stats import binomtest

    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def fisher_exact_pvalue(a: int, b: int, c: int, d: int) -> Optional[float]:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]].

    Used to ask whether two INDEPENDENT pass rates differ — e.g. a bias arm's
    quality-preservation rate against the same validator's `sham` false-flag floor.
    A bare `rate < floor` comparison is not enough there: a validator whose floor is
    109/110 marks almost every arm as "below" on a one-image difference, which reads
    as evidence of damage when it is sampling noise. None if either row is empty.
    """
    if a + b <= 0 or c + d <= 0:
        return None
    from scipy.stats import fisher_exact

    return float(fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])


def wilson_interval(successes: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    """95% Wilson score interval, delegating to `visualization.style.wilson`.

    The project keeps exactly one Wilson implementation (used by the RR/CR, coverage and
    validator-pass-rate figures). It is imported lazily because `visualization.style`
    pulls in matplotlib at import time, and a metrics module must stay importable
    without a plotting stack.

    Returns `(None, None)` for `n == 0` rather than NaN: an empty interval is "not
    measured", and "not measured" never renders as a number.
    """
    if n <= 0:
        return (None, None)
    from edit_judge_bias.visualization.style import wilson  # lazy: matplotlib at import

    lo, hi = wilson(successes, n)
    return (round(float(lo), 6), round(float(hi), 6))


def minimum_detectable_effect(
    n: int, *, reference_n: int = 611, reference_mde: float = 0.113
) -> Optional[float]:
    """Scale the project's measured breadth MDE to a smaller paired n.

    Exists so the write-up quotes one number derived one way. The dataset report
    measured 0.113 SD at n=611 for the breadth block's paired test; MDE goes as 1/sqrt(n).

    ⚠️ **UNITS: SD of the PAIRED DIFFERENCE**, at alpha=0.05 two-sided and 80% power — the
    reference is exactly `(z_.975 + z_.80)/sqrt(n)` (0.1133 at n=611, matching the measured
    0.113 to three decimals). A shift standardised by SD(score) is NOT comparable to this
    number. Mixing the two denominators once already doubled a published ratio.
    """
    if n <= 0:
        return None
    return reference_mde * (reference_n / n) ** 0.5
