"""Scoring metrics: score shift, inflation rate, absolute change (§8.1).

Pairs each biased scoring result with the unbiased ("original") score of the same
sample under the same judge, then aggregates per (judge_model, bias_type):

    score_shift      = biased_score - original_score        (per sample)
    mean_score_shift = mean(score_shift)
    SIR              = P(biased_score > original_score)      (Score Inflation Rate)
    ASC              = mean(|score_shift|)                   (Absolute Score Change)

A positive mean_shift with SIR > 0.5 and a significant Wilcoxon p-value is the
core "the judge rewards a spurious cue it shouldn't" signal.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from edit_judge_bias.data.schema import JudgeResult
from edit_judge_bias.metrics.stats import bootstrap_ci, wilcoxon_pvalue


@dataclass
class ScoringShiftStats:
    judge_model: str
    bias_type: str
    n: int
    mean_original: float
    mean_biased: float
    mean_shift: float
    sir: float
    asc: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_value: Optional[float] = None
    score_field: str = "overall_score"
    shifts: List[float] = field(default_factory=list, repr=False)

    def as_row(self) -> dict:
        return {
            "judge_model": self.judge_model,
            "bias_type": self.bias_type,
            "score_field": self.score_field,
            "n": self.n,
            "mean_original": round(self.mean_original, 4),
            "mean_biased": round(self.mean_biased, 4),
            "mean_shift": round(self.mean_shift, 4),
            "sir": round(self.sir, 4),
            "asc": round(self.asc, 4),
            "ci_low": None if self.ci_low is None else round(self.ci_low, 4),
            "ci_high": None if self.ci_high is None else round(self.ci_high, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
        }


#: Preference order for the analysis variable. `fine_score` (the sum of the three
#: dimensions) is markedly finer-grained than `overall_score` — on the pilot
#: responses gemini-3.5-flash used 3 distinct `overall_score` values but 7 distinct
#: sums — so it wins whenever the rows carry one.
SCORE_FIELDS = ("fine_score", "overall_score")

#: How many parsed rows may lack the preferred field before resolution refuses to
#: silently drop them. A judge that omits a dimension in a handful of answers is a
#: few unusable items; one that omits it in a tenth of them is a different judge and
#: needs a deliberate decision, not a quiet fallback to a coarser variable.
MAX_MISSING_SHARE = 0.05


def check_single_scale(results: Iterable[JudgeResult], *, label: str = "") -> Optional[int]:
    """Return the one `score_scale` present, or raise if several are mixed.

    A 1-5 pilot judgment and a 1-10 main-grid judgment are not commensurable: their
    difference is not a score shift, and averaging them produces a number with no
    unit.

    An UNSTAMPED row (score_scale=None) counts as its own scale here. Every such row
    is a pilot-era judgment written before the field existed, i.e. a 1-5 answer; the
    permissive reading — "None is compatible with anything" — is exactly the silent
    failure this guard exists to prevent, because the pilot and the v2 subset share
    129 sample_ids and would otherwise pool without a word.
    """
    results = list(results)
    scales = {r.score_scale for r in results}
    if len(scales) > 1:
        shown = sorted(("unstamped (pre-1-10 pilot)" if s is None else str(s)) for s in scales)
        raise ValueError(
            f"{label or 'results'} mix judge score scales {shown}; "
            "scores on different scales cannot be pooled — filter to one scale first"
        )
    return next(iter(scales), None)


def is_retest_repeat(result: JudgeResult) -> bool:
    """Is this row the SECOND (or later) ask of an identical question?

    The test-retest arm re-asks unbiased baselines with the cache off, so its rows
    carry no bias_type and land in the same manifest as the baselines they repeat.
    They must never be read as baselines: `(judge, sample_id)` is not unique across
    them, so a last-write-wins lookup would silently swap the baseline of every
    repeated sample for its retest answer — 200 of the breadth block's 611.
    """
    return int((result.bias_params or {}).get("repeat_index", 1)) > 1


def retest_pairs(results: Iterable[JudgeResult]) -> Dict[Tuple[str, str], List[int]]:
    """(judge_model, sample_id) -> the scores of each ask, first pass first.

    The arm's whole output. Keeping the reader next to the filter that hides these
    rows from every other metric is deliberate: they are excluded from the main
    tables, not discarded.
    """
    field = resolve_score_field(list(results))
    out: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for r in results:
        value = getattr(r, field, None)
        if not (r.parse_success and value is not None and r.sample_id and not r.bias_type):
            continue
        rep = int((r.bias_params or {}).get("repeat_index", 1))
        out.setdefault((r.judge_model, r.sample_id), []).append((rep, value))
    return {
        key: [v for _, v in sorted(vals)]
        for key, vals in out.items()
        if len(vals) > 1
    }


@dataclass
class RetestStats:
    """One judge's answer to "does it give the same answer to the same question?"."""

    judge_model: str
    n: int
    score_field: str
    mean_abs_delta: float
    identical_rate: float
    #: SIGNED drift, second ask minus first. Sign matters more than magnitude here:
    #: the biased conditions were asked AFTER the baselines, so a positive drift
    #: makes every reported deflation an understatement rather than an artefact.
    mean_delta: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_value: Optional[float] = None

    def as_row(self) -> dict:
        def r(v, nd=4):
            return None if v is None else round(v, nd)

        return {
            "judge_model": self.judge_model,
            "score_field": self.score_field,
            "n": self.n,
            "mean_abs_delta": r(self.mean_abs_delta),
            "identical_rate": r(self.identical_rate),
            "mean_delta": r(self.mean_delta),
            "ci_low": r(self.ci_low),
            "ci_high": r(self.ci_high),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
        }


def compute_retest_stats(
    results: Iterable[JudgeResult], *, seed: int = 42
) -> List[RetestStats]:
    """Per-judge test-retest noise floor from the repeat arm.

    The arm asks an identical question twice with the response cache off. At
    temperature 0 the two answers are *supposed* to be identical; the gap between
    that expectation and reality is the floor every effect in claim A has to clear.
    Two numbers say different things and both are reported: `mean_abs_delta` is how
    far a single item moves, `mean_delta` is whether the movement has a direction.

    Reads the arm through :func:`retest_pairs`, which is the only place that treats
    the `::rep2` rows as data rather than hiding them.
    """
    results = list(results)
    check_single_scale(results, label="retest results")
    score_field = resolve_score_field(results)

    by_judge: Dict[str, List[int]] = {}
    for (judge, _sample_id), scores in retest_pairs(results).items():
        # Consecutive asks: (2nd - 1st), (3rd - 2nd), ... A `repeat: 2` arm gives
        # one delta per sample, which is the only shape used so far.
        by_judge.setdefault(judge, []).extend(
            b - a for a, b in zip(scores, scores[1:])
        )

    stats: List[RetestStats] = []
    for judge, deltas in sorted(by_judge.items()):
        if not deltas:
            continue
        ci = bootstrap_ci(deltas, seed=seed)
        stats.append(RetestStats(
            judge_model=judge,
            n=len(deltas),
            score_field=score_field,
            mean_abs_delta=statistics.fmean(abs(d) for d in deltas),
            identical_rate=sum(1 for d in deltas if d == 0) / len(deltas),
            mean_delta=statistics.fmean(deltas),
            ci_low=ci[0] if ci else None,
            ci_high=ci[1] if ci else None,
            p_value=wilcoxon_pvalue(deltas),
        ))
    return stats


@dataclass
class PlaceboContrastStats:
    """One (judge, bias) cell measured against the sham arm instead of the baseline."""

    judge_model: str
    bias_type: str
    control_bias: str
    n: int
    mean_biased: float
    mean_control: float
    mean_contrast: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_value: Optional[float] = None
    score_field: str = "overall_score"
    contrasts: List[float] = field(default_factory=list, repr=False)

    def as_row(self) -> dict:
        def r(v, nd=4):
            return None if v is None else round(v, nd)

        return {
            "judge_model": self.judge_model,
            "bias_type": self.bias_type,
            "control_bias": self.control_bias,
            "score_field": self.score_field,
            "n": self.n,
            "mean_biased": r(self.mean_biased),
            "mean_control": r(self.mean_control),
            "mean_contrast": r(self.mean_contrast),
            "ci_low": r(self.ci_low),
            "ci_high": r(self.ci_high),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
        }


def compute_placebo_contrast(
    biased_results: Iterable[JudgeResult],
    *,
    control_bias: str = "sham",
    seed: int = 42,
    score_field: Optional[str] = None,
) -> List[PlaceboContrastStats]:
    """Per (judge, bias) contrast against the placebo arm, paired within sample.

    WHY THIS IS A ROBUSTNESS COLUMN AND NOT THE HEADLINE. The sham arm's own shift
    is not zero — measured on the breadth block it is +0.30 for gpt-5.5 and +0.59
    for gemini-3.5-flash, both significant. Some of that is plumbing every biased
    condition shares (a new file, a fresh call, a second ask), and subtracting it
    de-confounds the estimate; `d_i = biased_i - sham_i` is the contrast that does
    so item by item.

    But sham is a JPEG round trip saved back to PNG, while every other injector is a
    lossless PNG transform — so the placebo is NOT a strict sub-component of the
    conditions it is being subtracted from, and any part of its offset caused by the
    codec itself is over-subtracted. Its designed role is an *equivalence bound*
    ("perturbing the file without perturbing the picture moves the judge at most
    this much"), which is a screen, not a correction.

    Consequence, and the reason this function exists separately rather than
    replacing :func:`compute_score_shifts`: the correction moves `saturation` and
    `aesthetic_filter` from inside the placebo bound to significant on 3-4 judges.
    The headline stays on the raw shift, which is the conservative reading; this is
    reported next to it so the dependence is visible instead of assumed away.

    The control cell itself is excluded from the output — its contrast with itself
    is identically zero and would only pad the multiple-comparison family.
    """
    biased_results = list(biased_results)
    check_single_scale(biased_results, label="placebo contrast results")
    if score_field is None:
        score_field = resolve_score_field(biased_results)

    # (judge, bias) -> {sample_id: score}. Same dedup guard as compute_score_shifts:
    # a (judge, biased_id) written twice must not be counted twice.
    by_cell: Dict[Tuple[str, str], Dict[str, int]] = {}
    seen_biased: set = set()
    for r in biased_results:
        value = getattr(r, score_field, None)
        if not (r.parse_success and value is not None and r.sample_id and r.bias_type):
            continue
        dedup_key = (r.judge_model, r.biased_id)
        if r.biased_id is not None and dedup_key in seen_biased:
            continue
        seen_biased.add(dedup_key)
        by_cell.setdefault((r.judge_model, r.bias_type), {})[r.sample_id] = value

    controls = {
        judge: scores for (judge, bias), scores in by_cell.items() if bias == control_bias
    }

    stats: List[PlaceboContrastStats] = []
    for (judge, bias), scores in sorted(by_cell.items()):
        if bias == control_bias:
            continue
        control = controls.get(judge)
        if not control:
            continue  # no placebo arm for this judge; the cell has no contrast
        ids = sorted(set(scores) & set(control))
        if not ids:
            continue
        biased = [scores[i] for i in ids]
        ctrl = [control[i] for i in ids]
        contrasts = [b - c for b, c in zip(biased, ctrl)]
        ci = bootstrap_ci(contrasts, seed=seed)
        stats.append(PlaceboContrastStats(
            judge_model=judge,
            bias_type=bias,
            control_bias=control_bias,
            n=len(ids),
            mean_biased=statistics.fmean(biased),
            mean_control=statistics.fmean(ctrl),
            mean_contrast=statistics.fmean(contrasts),
            ci_low=ci[0] if ci else None,
            ci_high=ci[1] if ci else None,
            p_value=wilcoxon_pvalue(contrasts),
            score_field=score_field,
            contrasts=contrasts,
        ))
    return stats


def resolve_score_field(results: Iterable[JudgeResult]) -> str:
    """Pick the analysis variable: `fine_score` whenever the rows carry one.

    Resolved once over the whole set rather than per row, because a column that is
    half dimension-sums and half overall scores is not on any single scale. Rows
    missing the resolved field are then dropped item-wise by the callers, which keeps
    the column single-scale — that, not the field choice, is what the all-or-nothing
    reading used to buy.

    WHY NOT all-or-nothing. Measured on the breadth grid: kimi-k2.5 omitted
    `detail_preservation` in 4 of its 611 baseline answers (the parser accepts that —
    `overall_score` was there), and the old rule read those 4 rows as proof that the
    whole judge had no `fine_score`. It silently fell back to `overall_score` for
    kimi alone while gpt-5.5 and gemini stayed on `fine_score`, so one `mean_shift`
    column held both 3-30 sums and 1-10 overalls and kimi looked an order of
    magnitude more robust than it was. `fine_score` is a project-level decision (it
    is *the* analysis variable), not something four bad answers get to revoke.

    The `overall_score` fallback exists for one real case: pilot-era rows, written
    before `fine_score` existed, where no row has one.
    """
    parsed = [r for r in results if r.parse_success]
    if not parsed:
        return "overall_score"
    for name in SCORE_FIELDS:
        present = sum(1 for r in parsed if getattr(r, name, None) is not None)
        if not present:
            continue
        missing_share = 1 - present / len(parsed)
        if missing_share > MAX_MISSING_SHARE:
            raise ValueError(
                f"{missing_share:.1%} of parsed rows lack '{name}' "
                f"({len(parsed) - present} of {len(parsed)}); too many to drop "
                f"item-wise. Pass score_field= explicitly to state which variable "
                f"this set is analysed on."
            )
        return name
    return "overall_score"


def _parsed_scores_by_sample(
    results: Iterable[JudgeResult], score_field: str = "overall_score"
) -> Dict[Tuple[str, str], int]:
    """(judge_model, sample_id) -> score for parsed scoring results.

    Retest repeats are skipped — see :func:`is_retest_repeat`.
    """
    out: Dict[Tuple[str, str], int] = {}
    for r in results:
        value = getattr(r, score_field, None)
        if r.parse_success and value is not None and r.sample_id and not is_retest_repeat(r):
            out[(r.judge_model, r.sample_id)] = value
    return out


def compute_score_shifts(
    original_results: Iterable[JudgeResult],
    biased_results: Iterable[JudgeResult],
    *,
    seed: int = 42,
    include_biased_ids: Optional[set] = None,
    score_field: Optional[str] = None,
) -> List[ScoringShiftStats]:
    """Per (judge_model, bias_type) score-shift stats, paired by sample within model.

    If `include_biased_ids` is given, only biased results whose `biased_id` is in
    that set are counted — used for the quality-controlled shift (QC_shift, §8.1)
    over images that passed quality-preservation validation.

    `score_field` defaults to whichever of :data:`SCORE_FIELDS` every parsed row
    carries. Shifts are reported in units of that field, so a 1-10 grid's shifts are
    on a 3-30 sum and a 1-5 pilot's on a 1-5 overall — never mixed, see
    :func:`check_single_scale`.
    """
    original_results = list(original_results)
    biased_results = list(biased_results)
    check_single_scale(original_results + biased_results, label="scoring results")
    if score_field is None:
        score_field = resolve_score_field(original_results + biased_results)

    base = _parsed_scores_by_sample(original_results, score_field)

    grouped: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    seen_biased: set = set()  # defensive: never count a (model, biased_id) twice
    for r in biased_results:
        value = getattr(r, score_field, None)
        if not (r.parse_success and value is not None and r.sample_id):
            continue
        dedup_key = (r.judge_model, r.biased_id)
        if r.biased_id is not None and dedup_key in seen_biased:
            continue
        seen_biased.add(dedup_key)
        if include_biased_ids is not None and r.biased_id not in include_biased_ids:
            continue
        orig = base.get((r.judge_model, r.sample_id))
        if orig is None:
            continue  # no unbiased score to compare against
        grouped.setdefault((r.judge_model, r.bias_type or "unknown"), []).append(
            (orig, value)
        )

    return [
        shift_stats_from_pairs(model, bias_type, pairs,
                               score_field=score_field, seed=seed)
        for (model, bias_type), pairs in sorted(grouped.items())
    ]


def shift_stats_from_pairs(
    judge_model: str,
    bias_type: str,
    pairs: Sequence[Tuple[float, float]],
    *,
    score_field: str,
    seed: int = 42,
) -> ScoringShiftStats:
    """One (judge, cue) claim-A row from already-paired `(original, biased)` scores.

    Split out of :func:`compute_score_shifts` so a caller whose scores are NOT ints on
    a `JudgeResult` runs the same estimator rather than a second copy of it. The one
    such caller today is the z-normalised ensemble in `build_mitigation_tables`: a
    z-score is a float and every score field on `JudgeResult` is `Optional[int]`, so
    it cannot go through the synthetic-row path the raw ensemble uses.
    ⚠️ Two copies of an effect-size estimator is the same failure this project already
    paid for with judge discovery living in three modules — one of them silently
    emptied a published table. An estimator gets exactly one implementation.
    """
    origs = [o for o, _ in pairs]
    bias = [b for _, b in pairs]
    shifts = [b - o for o, b in pairs]
    n = len(shifts)
    ci = bootstrap_ci(shifts, seed=seed)
    return ScoringShiftStats(
        judge_model=judge_model,
        bias_type=bias_type,
        n=n,
        mean_original=sum(origs) / n,
        mean_biased=sum(bias) / n,
        mean_shift=sum(shifts) / n,
        sir=sum(1 for s in shifts if s > 0) / n,
        asc=sum(abs(s) for s in shifts) / n,
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        p_value=wilcoxon_pvalue(shifts),
        score_field=score_field,
        shifts=shifts,
    )
