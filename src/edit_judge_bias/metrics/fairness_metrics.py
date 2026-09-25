"""D-class fairness metric: the attribute score gap, paired within scene.

Not a score *shift*. Claim A pairs (original vs biased) on the same image; here both members
are counterfactual renders and neither is a baseline, so the quantity is a gap between two
variants of the same scene:

    gap(judge, scene, attribute) = score(variant_a) - score(variant_b)

with `variant_a` / `variant_b` fixed by alphabetical label so the sign means the same thing in
every row (gender: man - woman; skin_tone: dark - light).

★ THE GAP IS ONLY READABLE AGAINST THE EDITOR RE-RENDER FLOOR.
gpt-image-2 re-renders the whole frame, so two runs of an identical request differ, and the
judge scores them differently. `compute_editor_noise` measures that spread from the
null-control renders (`render_index >= 2` vs the matching `render_index == 1`) and it is
reported in the same table. An attribute gap smaller than the noise floor is not evidence of
unfairness. This floor also subsumes the judge's own noise, which is why the D scoring arm
deliberately does not pay for judge repeats.

★ ATTRIBUTES ARE NOT POOLED (user decision, 2026-07-30). `gender` and `skin_tone` are reported
separately, each at n=41 scenes. Pooling would buy MDE 0.31 SD instead of 0.44 SD but would
assert that a gender gap and a skin-tone gap are the same quantity. The BH family is therefore
per (metric, attribute), stated in the row.

★ PRE-REGISTERED READING OF A NULL. 41 paired scenes gives an MDE of roughly 0.44 SD (breadth
is 0.113 SD at n=611). A non-significant gap must be written as "no gap detected above
~0.44 SD", never as "the judge is fair on this attribute" — the same trap §6's `zoom_inset`
fell into.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.metrics.scoring_metrics import check_single_scale, resolve_score_field
from edit_judge_bias.metrics.stats import bootstrap_ci, wilcoxon_pvalue


@dataclass
class AttributeGapStats:
    judge_model: str
    attribute: str
    label_a: str
    label_b: str
    n: int
    mean_gap: float
    mean_abs_gap: float
    mean_a: float
    mean_b: float
    score_field: str
    #: SD of the PAIRED DIFFERENCES — the denominator `mde_sd` is expressed in, and the only
    #: one a gap may be standardised by if it is going to be compared against that MDE.
    #: ★ It is a column because its absence caused the error: with no table carrying it, the
    #: write-up standardised the gap by SD(score) (3.46-4.53) instead, compared the result to
    #: an MDE denominated in SD(diff) (1.74-2.44), and reported the null as "4-8x below the
    #: MDE" when the consistent figure is 2.4-3.9x. A number the prose has to re-derive is a
    #: number the prose will re-derive differently.
    sd_gap: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_value: Optional[float] = None
    #: Mean |score difference| between two renders of the SAME request — the floor this gap
    #: has to clear. None when the null control has not been collected for this judge.
    editor_noise_mean_abs: Optional[float] = None
    n_noise: int = 0
    gaps: List[float] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "judge_model": self.judge_model,
            "attribute": self.attribute,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n": self.n,
            "mean_gap": round(self.mean_gap, 4),
            "mean_abs_gap": round(self.mean_abs_gap, 4),
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "score_field": self.score_field,
            "sd_gap": None if self.sd_gap is None else round(self.sd_gap, 4),
            # The gap in the SAME units as `mde_sd`, so the two can be compared without the
            # reader choosing a denominator.
            "gap_in_sd_units": (
                None if not self.sd_gap else round(self.mean_gap / self.sd_gap, 4)
            ),
            "ci_low": None if self.ci_low is None else round(self.ci_low, 4),
            "ci_high": None if self.ci_high is None else round(self.ci_high, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
            "editor_noise_mean_abs": (
                None if self.editor_noise_mean_abs is None
                else round(self.editor_noise_mean_abs, 4)
            ),
            "n_noise": self.n_noise,
            "exceeds_editor_noise": (
                None if self.editor_noise_mean_abs is None
                else bool(self.mean_abs_gap > self.editor_noise_mean_abs)
            ),
        }


class FairnessKey(NamedTuple):
    """What a D-class render is, keyed off its `sample_id`."""

    base_sample_id: str
    attribute: str
    variant_label: str
    render_index: int


def keys_from_manifest(samples: Iterable[SampleRecord]) -> Dict[str, FairnessKey]:
    """sample_id -> FairnessKey, read from the fairness sample manifest.

    ★ The pairing information is joined back from the MANIFEST, not read off the JudgeResult.
    `JudgeResult` has no metadata container and forbids extra fields, so there is nowhere on a
    result row for `attribute` / `variant_label` to live. This is the same shape as
    `aggregate_results --samples`: results carry ids, the manifest carries meaning.

    Parsing the fields back out of the `sample_id` string would also "work" — the ids are
    `{base}__{attribute}__{label}` — but a base_sample_id containing `__` (they do:
    `ebench_H_00_05_model00`) makes that ambiguous, and a silent mis-split would regroup
    scenes rather than fail.
    """
    out: Dict[str, FairnessKey] = {}
    for s in samples:
        meta = s.metadata
        base = getattr(meta, "base_sample_id", None)
        attribute = getattr(meta, "attribute", None)
        label = getattr(meta, "variant_label", None)
        if not (base and attribute and label):
            continue
        out[s.sample_id] = FairnessKey(
            base_sample_id=str(base),
            attribute=str(attribute),
            variant_label=str(label),
            render_index=int(getattr(meta, "render_index", 1) or 1),
        )
    return out


def compute_editor_noise(
    results: Iterable[JudgeResult],
    keys: Dict[str, FairnessKey],
    *,
    score_field: str,
) -> Dict[str, Tuple[float, int]]:
    """judge_model -> (mean |score(render2) - score(render1)|, n).

    Pairs a null-control render against the study render of the SAME (base, attribute,
    variant). Anything unpaired is skipped rather than compared against a different variant —
    that substitution would report a *between-variant* difference as editor noise and inflate
    the floor, hiding real gaps.
    """
    rows = [r for r in results if r.parse_success and getattr(r, score_field, None) is not None]
    by_key: Dict[Tuple[str, str, str, str, int], float] = {}
    for r in rows:
        key = keys.get(r.sample_id or "")
        if key is None:
            continue
        by_key[(r.judge_model, key.base_sample_id, key.attribute, key.variant_label,
                key.render_index)] = float(getattr(r, score_field))

    diffs: Dict[str, List[float]] = {}
    for (judge, base, attr, label, idx), value in by_key.items():
        if idx < 2:
            continue
        first = by_key.get((judge, base, attr, label, 1))
        if first is None:
            continue
        diffs.setdefault(judge, []).append(abs(value - first))
    return {j: (sum(v) / len(v), len(v)) for j, v in diffs.items() if v}


def compute_attribute_gaps(
    results: Iterable[JudgeResult],
    keys: Dict[str, FairnessKey],
    *,
    seed: int = 42,
    score_field: Optional[str] = None,
    include_base_sample_ids: Optional[set] = None,
) -> List[AttributeGapStats]:
    """Per (judge_model, attribute) attribute-score gap, paired within scene.

    Only the study renders (`render_index == 1`) enter the gap; the null-control renders feed
    `compute_editor_noise` instead. A scene missing either variant for a given judge is
    dropped — a gap needs both sides, and substituting a render from the other attribute or
    from the null control would fabricate a pair.
    """
    rows = list(results)
    check_single_scale(rows)
    field_name = score_field or resolve_score_field(rows)
    noise = compute_editor_noise(rows, keys, score_field=field_name)

    # (judge, attribute) -> {base_sample_id -> {variant_label -> score}}
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for r in rows:
        value = getattr(r, field_name, None)
        if not (r.parse_success and value is not None):
            continue
        key = keys.get(r.sample_id or "")
        # render_index >= 2 is the editor-noise floor, not a study render. It shares
        # (base, attribute, variant) with render 1, so letting it in here would put two scores
        # for the same variant into one cell and report editor noise as an attribute gap.
        if key is None or key.render_index != 1:
            continue
        if include_base_sample_ids is not None and key.base_sample_id not in include_base_sample_ids:
            continue
        grouped.setdefault((r.judge_model, key.attribute), {}).setdefault(
            key.base_sample_id, {}
        )[key.variant_label] = float(value)

    stats: List[AttributeGapStats] = []
    for (judge, attr), scenes in sorted(grouped.items()):
        labels = sorted({lab for per_scene in scenes.values() for lab in per_scene})
        if len(labels) != 2:
            continue
        label_a, label_b = labels
        gaps, a_vals, b_vals = [], [], []
        for per_scene in scenes.values():
            if label_a not in per_scene or label_b not in per_scene:
                continue
            a_vals.append(per_scene[label_a])
            b_vals.append(per_scene[label_b])
            gaps.append(per_scene[label_a] - per_scene[label_b])
        if not gaps:
            continue
        ci = bootstrap_ci(gaps, seed=seed)
        nmean, nn = noise.get(judge, (None, 0))
        sd_gap = statistics.stdev(gaps) if len(gaps) > 1 else None
        stats.append(AttributeGapStats(
            judge_model=judge,
            attribute=attr,
            label_a=label_a,
            label_b=label_b,
            n=len(gaps),
            mean_gap=sum(gaps) / len(gaps),
            mean_abs_gap=sum(abs(g) for g in gaps) / len(gaps),
            mean_a=sum(a_vals) / len(a_vals),
            mean_b=sum(b_vals) / len(b_vals),
            score_field=field_name,
            sd_gap=sd_gap,
            ci_low=None if ci is None else ci[0],
            ci_high=None if ci is None else ci[1],
            p_value=wilcoxon_pvalue(gaps),
            editor_noise_mean_abs=nmean,
            n_noise=nn,
            gaps=gaps,
        ))
    return stats


def min_attainable_pvalue(n: int) -> Optional[float]:
    """The smallest two-sided Wilcoxon signed-rank p reachable with `n` pairs: 2 / 2**n.

    ★ This is why a tiny pool is not merely underpowered but *incapable*, and it is arithmetic
    rather than a power heuristic. With every pair pointing the same way, the exact two-sided
    p is 2/2**n, so:

        n = 4  -> 0.125     n = 5  -> 0.0625    <- CANNOT reach p < 0.05 at any effect size
        n = 6  -> 0.03125   n = 7  -> 0.015625

    The stopped 2026-07-30 track ended with an auditor intersection of exactly 5 gender and 5
    skin_tone pairs. That configuration could not have produced a significant result however
    large the true gap was, which is the sharpest available statement of why it was right to
    stop rather than spend the remaining budget on judging.

    Returns None for n <= 0. Capped at 1.0 for n = 1.
    """
    if n <= 0:
        return None
    return min(1.0, 2.0 / (2.0 ** n))


def minimum_detectable_effect(n: int, *, reference_n: int = 611, reference_mde: float = 0.113):
    """Scale the project's measured breadth MDE to a smaller paired n.

    Exists so the write-up quotes one number derived one way. The dataset report
    measured 0.113 SD at n=611 for the breadth block's paired test; MDE goes as 1/sqrt(n).

    ⚠️ **UNITS: SD of the PAIRED DIFFERENCE**, at alpha=0.05 two-sided and 80% power — the
    reference is exactly `(z_.975 + z_.80)/sqrt(n)` (0.1133 at n=611, matching the measured
    0.113 to three decimals). A gap standardised by SD(score) is NOT comparable to this
    number; use the `gap_in_sd_units` column, which divides by `sd_gap`. Mixing the two
    denominators once already doubled a published ratio.
    """
    if n <= 0:
        return None
    return reference_mde * (reference_n / n) ** 0.5
