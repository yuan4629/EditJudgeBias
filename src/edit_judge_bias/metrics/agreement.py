"""Judge-human agreement, before and after a bias (claim B).

Claim A shows the judge's score *moves* under a perturbation that provably did not
change editing quality. The obvious rebuttal is that the deduction was justified —
the padded image really is worse. This module answers that: if the perturbation
pushed the judge **away from human judgement**, the deduction was not a correction.

Two agreement measures, deliberately different in kind:

``spearman``
    Rank correlation between the judge's score and the human score, item by item.
    Rank-only on purpose — EBench's MOS is a z-scored relative scale, so its
    absolute values are not comparable to a 1-10 judge score even in principle.

``derived pairwise accuracy``
    Within a turn (one original image + one instruction, several editors), turn
    both score columns into preferences and ask how often the judge's ordering
    matches the humans'. This is the quantity a benchmark actually uses a judge
    for, and it survives monotone rescaling of either column.

**Clusters, not items, are the sample size.** Several editors of the same turn
share an original image, an instruction and a difficulty; their errors are
correlated. Every interval here is a *cluster bootstrap over turns*, which is also
why the human-anchor blocks are drawn as whole turns
(``build_full_manifest.group_take``) rather than scattered across many.

Novelty note for the write-up: recomputing judge-human agreement under
perturbation is NOT new — it has been done at least five times. What is defensible
here is the three-way conjunction: a quality-preserving perturbation, preservation
*verified* independently, and agreement recomputed. Never claim a first.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from edit_judge_bias.data.build_full_manifest import human_score_sd
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.metrics.scoring_metrics import (
    check_single_scale,
    is_retest_repeat,
    resolve_score_field,
)
from edit_judge_bias.metrics.stats import mcnemar_pvalue, wilcoxon_pvalue

#: How big a human score gap has to be, in SDs of that source's own human scores,
#: before a derived preference counts as decisive. Matching
#: `build_full_manifest.decisive_tier` on purpose: a pair the sampler considered
#: too close to call must not be scored as a judge error here either.
DEFAULT_MIN_EFFECT = 0.5


# --------------------------------------------------------------------------- #
# building blocks                                                             #
# --------------------------------------------------------------------------- #
def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rho, or None when it is undefined (n<3, or either side constant).

    A constant column is the realistic failure here, not a rare one: a judge that
    answers "2" to almost everything has no ranking to correlate. Returning None
    rather than nan keeps that visible instead of poisoning a mean.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    from scipy.stats import spearmanr

    rho = spearmanr(xs, ys).statistic
    return None if rho is None or math.isnan(float(rho)) else float(rho)


def cluster_bootstrap_ci(
    clusters: Sequence[Sequence],
    stat_fn: Callable[[List], Optional[float]],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Optional[Tuple[float, float]]:
    """Percentile CI resampling whole CLUSTERS with replacement.

    Resampling items instead would treat 8 editors of one turn as 8 independent
    observations and report an interval roughly sqrt(8) times too narrow.
    """
    if len(clusters) < 2:
        return None
    rng = np.random.default_rng(seed)
    n = len(clusters)
    values: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        items = [item for i in idx for item in clusters[i]]
        val = stat_fn(items)
        if val is not None and not math.isnan(val):
            values.append(val)
    if len(values) < 2:
        return None
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# joining judge results to human labels                                       #
# --------------------------------------------------------------------------- #
@dataclass
class AgreementItem:
    """One sample with a human score and both judge scores (unbiased / biased)."""

    sample_id: str
    turn: Tuple[str, str]
    source_dataset: str
    human_score: float
    judge_original: float
    judge_biased: Optional[float] = None


def _turn_of(rec: SampleRecord) -> Tuple[str, str]:
    return (rec.original_image_path.as_posix(), rec.instruction)


def _scores_by_sample(results: Iterable[JudgeResult], field_name: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for r in results:
        value = getattr(r, field_name, None)
        if r.parse_success and value is not None and r.sample_id and not is_retest_repeat(r):
            out[(r.judge_model, r.sample_id)] = float(value)
    return out


def build_items(
    samples: Iterable[SampleRecord],
    original_results: Iterable[JudgeResult],
    biased_results: Iterable[JudgeResult],
    *,
    judge_model: str,
    bias_type: str,
    score_field: Optional[str] = None,
) -> List[AgreementItem]:
    """Join one (judge, bias) cell onto the human-labelled samples."""
    original_results = list(original_results)
    biased_results = [r for r in biased_results if r.bias_type == bias_type]
    check_single_scale(original_results + biased_results, label="agreement inputs")
    if score_field is None:
        score_field = resolve_score_field(original_results + biased_results)

    base = _scores_by_sample(original_results, score_field)
    after = _scores_by_sample(biased_results, score_field)

    items: List[AgreementItem] = []
    for s in samples:
        if s.human_score is None:
            continue
        judged = base.get((judge_model, s.sample_id))
        if judged is None:
            continue  # no unbiased score to compare against
        items.append(AgreementItem(
            sample_id=s.sample_id,
            turn=_turn_of(s),
            source_dataset=s.source_dataset,
            human_score=float(s.human_score),
            judge_original=judged,
            judge_biased=after.get((judge_model, s.sample_id)),
        ))
    return items


def by_turn(items: Sequence[AgreementItem]) -> List[List[AgreementItem]]:
    """Group items into clusters — the unit every interval is resampled over."""
    groups: Dict[Tuple[str, str], List[AgreementItem]] = defaultdict(list)
    for it in items:
        groups[it.turn].append(it)
    return [groups[k] for k in sorted(groups)]


# --------------------------------------------------------------------------- #
# the two agreement measures                                                  #
# --------------------------------------------------------------------------- #
def _rho(items: Sequence[AgreementItem], *, biased: bool) -> Optional[float]:
    """Rank correlation on the PAIRED items only, in both directions.

    Both sides must be measured on the same items or the difference is not a
    before-after comparison. Computing "before" over every joined item while
    "after" silently drops the ones whose biased call failed to parse would let a
    non-random handful of dropped items masquerade as an agreement drop — and a
    bias that provokes unparseable answers is exactly the case where those drops
    are least random.
    """
    pairs = [
        (it.human_score, it.judge_biased if biased else it.judge_original)
        for it in items
        if it.judge_biased is not None
    ]
    if not pairs:
        return None
    return spearman([h for h, _ in pairs], [j for _, j in pairs])


@dataclass
class DerivedPair:
    """One within-turn comparison: who do the humans prefer, and who does the judge?"""

    turn: Tuple[str, str]
    human_pref: str
    judge_pref_original: str
    judge_pref_biased: Optional[str]

    @property
    def correct_original(self) -> bool:
        return self.judge_pref_original == self.human_pref

    @property
    def correct_biased(self) -> Optional[bool]:
        if self.judge_pref_biased is None:
            return None
        return self.judge_pref_biased == self.human_pref


def _pref(a: float, b: float, eps: float = 0.0) -> str:
    if a > b + eps:
        return "a"
    if b > a + eps:
        return "b"
    return "tie"


def derive_pairs(
    items: Sequence[AgreementItem],
    *,
    min_effect: float = DEFAULT_MIN_EFFECT,
    source_sd: Optional[Dict[str, float]] = None,
) -> List[DerivedPair]:
    """All within-turn comparisons whose human preference is decisive.

    Indecisive comparisons are dropped rather than scored: a judge that disagrees
    about two edits the raters themselves could not separate has not made an error,
    and counting it as one would dilute every accuracy toward 50%. "Decisive" is
    measured in SDs of that source's own human scores for the same reason the
    sampler does it that way — a fixed absolute gap means something different on
    each source's scale.
    """
    sds = source_sd or {}
    out: List[DerivedPair] = []
    for cluster in by_turn(items):
        for x, y in combinations(cluster, 2):
            sd = sds.get(x.source_dataset)
            gap = abs(x.human_score - y.human_score)
            if sd and gap < min_effect * sd:
                continue
            human = _pref(x.human_score, y.human_score)
            if human == "tie":
                continue
            biased_pref = (
                _pref(x.judge_biased, y.judge_biased)
                if x.judge_biased is not None and y.judge_biased is not None
                else None
            )
            out.append(DerivedPair(
                turn=x.turn,
                human_pref=human,
                judge_pref_original=_pref(x.judge_original, y.judge_original),
                judge_pref_biased=biased_pref,
            ))
    return out


def cluster_net_discordance(graded: Sequence[DerivedPair]) -> List[float]:
    """Per-turn net discordance `d_t = c_t - b_t`, one value per turn.

    ★ THIS IS THE UNIT AT WHICH CLAIM B'S ACCURACY TEST IS ACTUALLY INDEPENDENT.
    `derive_pairs` explodes each anchor turn (8 editors) into ~18 within-turn comparisons in
    which every item reappears up to 7 times. A McNemar over those comparisons counts them as
    ~18 independent Bernoulli trials and reports a p-value roughly sqrt(design effect) too
    small; the measured design effect on this data reaches 4.3.

    Aggregating to `d_t` first and then testing symmetry about zero over TURNS keeps the same
    estimand (does the bias break more comparisons than it fixes?) at a level where the
    observations are independent by construction — the anchor blocks were drawn whole-turn
    precisely so this would be true. Turns with `d_t == 0` fall out of the signed-rank test,
    which is the direct analogue of McNemar conditioning on discordant pairs.
    """
    per_turn: Dict[Tuple[str, str], int] = defaultdict(int)
    for p in graded:
        if p.correct_original and not p.correct_biased:
            per_turn[p.turn] -= 1
        elif not p.correct_original and p.correct_biased:
            per_turn[p.turn] += 1
        else:
            per_turn[p.turn] += 0
    return [float(per_turn[k]) for k in sorted(per_turn)]


def _accuracy(pairs: Sequence[DerivedPair], *, biased: bool) -> Optional[float]:
    """Share of decisive human comparisons the judge orders the same way.

    A judge tie counts as WRONG, not as half credit and not as an exclusion. The
    human preference here is decisive by construction, so "I cannot tell" is a
    failure to reproduce it — and a bias that makes the judge *more* indecisive has
    damaged its validity in exactly the way claim B is about. The cost is that the
    number is not comparable to a 50% coin flip, which is why `tie_rate_*` is
    reported next to it.
    """
    if biased:
        vals = [p.correct_biased for p in pairs if p.correct_biased is not None]
    else:
        vals = [p.correct_original for p in pairs]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def _tie_rate(pairs: Sequence[DerivedPair], *, biased: bool) -> Optional[float]:
    prefs = [
        p.judge_pref_biased if biased else p.judge_pref_original
        for p in pairs
        if not biased or p.judge_pref_biased is not None
    ]
    if not prefs:
        return None
    return sum(1 for p in prefs if p == "tie") / len(prefs)


# --------------------------------------------------------------------------- #
# the reported row                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class AgreementStats:
    judge_model: str
    bias_type: str
    n_items: int
    #: Items carrying BOTH judge scores — the set every measure below is computed
    #: on. Equal to `n_items` in a complete grid; smaller when a biased call failed.
    n_paired: int = 0
    n_clusters: int = 0
    spearman_original: Optional[float] = None
    spearman_biased: Optional[float] = None
    spearman_delta: Optional[float] = None
    spearman_delta_ci_low: Optional[float] = None
    spearman_delta_ci_high: Optional[float] = None
    n_pairs: int = 0
    # Without these the accuracy is unreadable: a judge that answers "tie" to every
    # comparison scores 0.0, which looks like a bug rather than like indecision.
    tie_rate_original: Optional[float] = None
    tie_rate_biased: Optional[float] = None
    accuracy_original: Optional[float] = None
    accuracy_biased: Optional[float] = None
    accuracy_delta: Optional[float] = None
    accuracy_delta_ci_low: Optional[float] = None
    accuracy_delta_ci_high: Optional[float] = None
    mcnemar_b: int = 0
    mcnemar_c: int = 0
    #: ⚠️ McNemar over ALL derived pairs, which treats the ~18 within-turn comparisons of one
    #: turn as independent observations. They are not: the anchor blocks are 48 turns x 8
    #: editors, every item appears in up to 7 comparisons, and the measured design effect
    #: (cluster-robust variance / naive) runs to 4.3. Kept as a descriptive statistic and
    #: NAMED so it can never be mistaken for the test the claim needs; `accuracy_p_cluster`
    #: is what enters BH. Reporting it uncorrected produced 13/40 BH-significant cells of
    #: which 10 had a (correctly clustered) accuracy CI straddling zero — the tell.
    mcnemar_p_unclustered: Optional[float] = None
    #: The cluster-level analogue: per-turn net discordance d_t = c_t - b_t, tested for
    #: symmetry about 0 by Wilcoxon signed-rank over TURNS. Zero-discordance turns drop out,
    #: exactly as McNemar conditions on discordant pairs — so this is the same test moved up
    #: to the level at which the observations are actually independent.
    accuracy_p_cluster: Optional[float] = None
    #: Turns contributing at least one graded pair — the n of the test above.
    n_clusters_graded: int = 0
    items: List[AgreementItem] = field(default_factory=list, repr=False)

    def as_row(self) -> dict:
        def r(v, nd=4):
            return None if v is None else round(v, nd)

        return {
            "judge_model": self.judge_model,
            "bias_type": self.bias_type,
            "n_items": self.n_items,
            "n_paired": self.n_paired,
            "n_clusters": self.n_clusters,
            "spearman_original": r(self.spearman_original),
            "spearman_biased": r(self.spearman_biased),
            "spearman_delta": r(self.spearman_delta),
            "spearman_delta_ci_low": r(self.spearman_delta_ci_low),
            "spearman_delta_ci_high": r(self.spearman_delta_ci_high),
            "n_pairs": self.n_pairs,
            "tie_rate_original": r(self.tie_rate_original),
            "tie_rate_biased": r(self.tie_rate_biased),
            "accuracy_original": r(self.accuracy_original),
            "accuracy_biased": r(self.accuracy_biased),
            "accuracy_delta": r(self.accuracy_delta),
            "accuracy_delta_ci_low": r(self.accuracy_delta_ci_low),
            "accuracy_delta_ci_high": r(self.accuracy_delta_ci_high),
            "mcnemar_b": self.mcnemar_b,
            "mcnemar_c": self.mcnemar_c,
            "n_clusters_graded": self.n_clusters_graded,
            "accuracy_p_cluster": (
                None if self.accuracy_p_cluster is None else round(self.accuracy_p_cluster, 6)
            ),
            "mcnemar_p_unclustered": (
                None if self.mcnemar_p_unclustered is None
                else round(self.mcnemar_p_unclustered, 6)
            ),
        }


def compute_agreement(
    samples: Sequence[SampleRecord],
    original_results: Iterable[JudgeResult],
    biased_results: Iterable[JudgeResult],
    *,
    judge_model: str,
    bias_type: str,
    score_field: Optional[str] = None,
    min_effect: float = DEFAULT_MIN_EFFECT,
    n_boot: int = 1000,
    seed: int = 42,
) -> AgreementStats:
    """One (judge, bias) row: agreement before, after, and the clustered CI of the drop."""
    joined = build_items(
        samples, original_results, biased_results,
        judge_model=judge_model, bias_type=bias_type, score_field=score_field,
    )
    # Everything below is a BEFORE-vs-AFTER comparison, so it runs on the items
    # that have both halves. `n_items` keeps the join size visible next to it: a
    # large gap between the two means biased calls went missing and the row should
    # be read with that in mind, not silently averaged with complete ones.
    items = [it for it in joined if it.judge_biased is not None]
    clusters = by_turn(items)
    stats = AgreementStats(
        judge_model=judge_model, bias_type=bias_type,
        n_items=len(joined), n_paired=len(items),
        n_clusters=len(clusters), items=items,
    )
    if not items:
        return stats

    sds = human_score_sd(samples)

    stats.spearman_original = _rho(items, biased=False)
    stats.spearman_biased = _rho(items, biased=True)
    if stats.spearman_original is not None and stats.spearman_biased is not None:
        stats.spearman_delta = stats.spearman_biased - stats.spearman_original

        def rho_delta(sub: List[AgreementItem]) -> Optional[float]:
            before, after = _rho(sub, biased=False), _rho(sub, biased=True)
            return None if before is None or after is None else after - before

        ci = cluster_bootstrap_ci(clusters, rho_delta, n_boot=n_boot, seed=seed)
        if ci:
            stats.spearman_delta_ci_low, stats.spearman_delta_ci_high = ci

    pairs = derive_pairs(items, min_effect=min_effect, source_sd=sds)
    stats.n_pairs = len(pairs)
    if pairs:
        stats.tie_rate_original = _tie_rate(pairs, biased=False)
        stats.tie_rate_biased = _tie_rate(pairs, biased=True)
        stats.accuracy_original = _accuracy(pairs, biased=False)
        stats.accuracy_biased = _accuracy(pairs, biased=True)
        if stats.accuracy_original is not None and stats.accuracy_biased is not None:
            stats.accuracy_delta = stats.accuracy_biased - stats.accuracy_original

            def acc_delta(sub: List[AgreementItem]) -> Optional[float]:
                sub_pairs = derive_pairs(sub, min_effect=min_effect, source_sd=sds)
                before = _accuracy(sub_pairs, biased=False)
                after = _accuracy(sub_pairs, biased=True)
                return None if before is None or after is None else after - before

            ci = cluster_bootstrap_ci(clusters, acc_delta, n_boot=n_boot, seed=seed)
            if ci:
                stats.accuracy_delta_ci_low, stats.accuracy_delta_ci_high = ci

        # McNemar on the SAME derived pairs before vs after — paired, but NOT independent:
        # the pairs are within-turn comparisons, so one turn contributes ~18 of them.
        # Kept for description and named `_unclustered`; the test that enters BH is the
        # cluster-level one below.
        graded = [p for p in pairs if p.correct_biased is not None]
        stats.mcnemar_b = sum(1 for p in graded if p.correct_original and not p.correct_biased)
        stats.mcnemar_c = sum(1 for p in graded if not p.correct_original and p.correct_biased)
        stats.mcnemar_p_unclustered = mcnemar_pvalue(stats.mcnemar_b, stats.mcnemar_c)

        discordance = cluster_net_discordance(graded)
        stats.n_clusters_graded = len(discordance)
        stats.accuracy_p_cluster = wilcoxon_pvalue(discordance)
    return stats


def compute_agreement_grid(
    samples: Sequence[SampleRecord],
    original_results: Sequence[JudgeResult],
    biased_results: Sequence[JudgeResult],
    **kwargs,
) -> List[AgreementStats]:
    """Every (judge_model, bias_type) cell present in the biased results."""
    cells = sorted({
        (r.judge_model, r.bias_type)
        for r in biased_results
        if r.bias_type and r.parse_success
    })
    return [
        compute_agreement(
            samples, original_results, biased_results,
            judge_model=model, bias_type=bias, **kwargs,
        )
        for model, bias in cells
    ]
