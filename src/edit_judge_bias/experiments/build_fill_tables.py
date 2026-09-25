"""The FILL v2 cells: every one measured against the fill's own `sham`.

WHAT THE FILL IS.  Arms 3 and 4 asked four anchor cues and three one-sided pairwise cues.
FILL v2 (collected 2026-09-13/14 into `results/v2_fill`) asked the rest: seven one-sided
image cues plus `bandwagon` and `model_name` on the 616 judged pairs, eight cues on the
585 anchor-block samples, and a `sham` placebo in both arms.

WHERE ITS ROWS GO.  Into the published files that hold the same kind of cell -- user
decision, 2026-09-15.  `build_claim_tables.build_all(fill_dir=...)` appends these rows to
`claim_b.csv` and `pairwise_one_sided.csv` (and to the `a5_` copies) and writes
`fill_collection_drift.csv`.  This module has no command line of its own on purpose: a
second writer of one file is the last-write-wins hazard, and a side file beside the
published one is exactly what the merge retired.  Every row says where it came from
(`collection`) and what it was measured against (`reference`); the fill's BH families are
declared apart (`... [fill]`), so the merge re-corrects no main-grid q-value.

WHY EVERY CELL HERE IS MEASURED AGAINST THE FILL'S OWN `sham`, NOT THE JULY BASELINE.
The fill re-asked no unbiased baseline, and the relay instrument is not stationary: on
the same anchors, the September `sham` sits well above the July baseline for
gemini-3.5-flash and gpt-5.5 while the other judges do not move
(`fill_collection_drift.csv` carries the numbers).  A cue-vs-July contrast would book
that drift as a cue effect on two judges.  The September `sham` was asked in the same
run, on the same items, with the same dressed side on every pair (measured 616/616), so
a cue-vs-sham contrast cancels the between-collection drift by design.  It is the
estimator `claim_a.csv` already reports as `shift_vs_placebo`, applied to EVERY judge
rather than switched per judge, so no cell's reference was picked after seeing its
effect.

WHAT IT DOES NOT CANCEL -- the caveats `READING` repeats for these rows:
  * drift WITHIN the fill run (conditions were asked in priority order over ~6 h);
  * relay channel changes WITHIN the fill run (gemini's conditions were served by more
    than one upstream channel, per the 2026-09-14 routing audit);
  * `sham` is a JPEG q95 round-trip saved as PNG: a zero-DOSE control, not a
    dose-matched one, exactly as in claim A.

Rows (``a5_`` files with ``--roster a5``):

``pairwise_one_sided.csv``, collection ``fill_v2``
    `compute_one_sided_bias` with the fill's `sham` verdict as the baseline.  Image cues
    use the recorded dressed side; `bandwagon` uses its endorsed slot
    (`bias_params.bandwagon_target`) as the side it pulls toward.  `model_name` names both
    slots, so it has no one-sided reading and is not tabulated (see `NOT_TABULATED`).

``claim_b.csv``, collection ``fill_v2``
    `compute_agreement_grid` with the fill's `sham` scores as the "before" column, per
    anchor source.  Rank half flagged by its cluster-bootstrap CI; accuracy half corrected
    within the fill family.

``fill_collection_drift.csv``
    Why the reference is `sham`: per judge, the fill's `sham` against the July baseline on
    the same items -- the score shift and the dressed-side win rate.  Descriptive, not a
    hypothesis family.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.aggregate_results import PUBLISHED_ROSTER, _judges
from edit_judge_bias.experiments.build_claim_tables import _read_judge, attach_bh
from edit_judge_bias.metrics.agreement import compute_agreement_grid
from edit_judge_bias.metrics.pairwise_metrics import compute_one_sided_bias
from edit_judge_bias.metrics.scoring_metrics import compute_score_shifts, is_retest_repeat

#: The condition every fill cell is measured against, and how the tables name it.
REFERENCE = "sham"
REFERENCE_LABEL = "sham (same collection)"
COLLECTION = "fill_v2 (2026-09-13/14)"
FILL_TAG = " [fill]"

#: What `configs/experiment/pairwise_fill_v2.yaml` asked, apart from the reference.
PAIRWISE_IMAGE_CUES: Tuple[str, ...] = (
    "saturation", "watermark", "aesthetic_filter", "region_annotation",
    "zoom_inset", "detail_caption", "distraction",
)
#: Prompt cues that DO have a one-sided reading, and the bias_params key naming the slot.
ENDORSED_SIDE_KEY: Dict[str, str] = {"bandwagon": "bandwagon_target"}
#: Asked, but with no one-sided reading.  Kept so that an absent row is a statement.
NOT_TABULATED: Dict[str, str] = {
    "model_name": "names both slots' editors, so no side is dressed and the one-sided "
                  "advantage is undefined",
}
#: What `configs/experiment/scoring_anchor_fill_v2.yaml` asked, apart from the reference.
ANCHOR_CUES: Tuple[str, ...] = (
    "saturation", "watermark", "aesthetic_filter", "zoom_inset", "detail_caption",
    "distraction", "bandwagon", "model_name",
)
#: A condition holding less than this share of its design is unfinished (or lost) and
#: may not enter a table.  The finished arm's worst condition is 613 of 616.
MIN_FILL_COVERAGE = 0.95


# --------------------------------------------------------------------------- #
# inputs                                                                      #
# --------------------------------------------------------------------------- #
def _fill_rows(fill_dir: Path, task: str, model: str) -> List[JudgeResult]:
    path = fill_dir / "biased_judgments" / f"{task}__{model}.jsonl"
    if not path.exists():
        # Raise rather than skip: a rebuild that quietly drops the fill would leave the
        # main table describing cues it no longer has cells for.
        raise FileNotFoundError(
            f"{path} is missing. claim_b.csv / pairwise_one_sided.csv carry rows built "
            "from results/v2_fill; restore it (it travels in the judgments branch of the "
            "delivery package)."
        )
    return _read_judge(path)


def anchor_only_ids(samples: Sequence[SampleRecord]) -> set:
    """The anchor arm's design minus the 39 samples the breadth block already judged."""
    return {
        s.sample_id for s in samples
        if getattr(s.metadata, "anchor_source", None)
        and getattr(s.metadata, "subset_block", None) != "breadth"
    }


def check_coverage(rows: Sequence[JudgeResult], *, judge: str, task: str,
                   conditions: Sequence[str], expected: int) -> None:
    """Refuse a judge whose fill is missing a condition or holds only part of one."""
    if expected <= 0:
        raise ValueError(f"{task}: no design to measure {judge}'s fill against")
    counts = Counter(r.bias_type for r in rows)
    short = {c: counts.get(c, 0) for c in conditions
             if counts.get(c, 0) < MIN_FILL_COVERAGE * expected}
    if short:
        raise ValueError(
            f"{task} fill for {judge} is incomplete: {short} of {expected} expected rows "
            f"per condition (minimum {MIN_FILL_COVERAGE:.0%}). The fill joins the tables "
            "only after its collection has ended."
        )


def with_endorsed_side(rows: Sequence[JudgeResult]) -> List[JudgeResult]:
    """Give each prompt cue with an endorsed slot that slot as its `biased_side`.

    `bandwagon` tells the judge that most reviewers prefer one displayed image; the
    one-sided question is whether the verdict moves toward THAT slot.  One-sided rows are
    never display-swapped, so displayed 'A' is slot 'a'.  Every other row passes through
    untouched.
    """
    out: List[JudgeResult] = []
    for r in rows:
        key = ENDORSED_SIDE_KEY.get(r.bias_type or "")
        if key is None:
            out.append(r)
            continue
        target = str((r.bias_params or {}).get(key, "")).strip().upper()
        if target not in ("A", "B"):
            raise ValueError(f"{r.result_id}: {key}={target!r}, expected 'A' or 'B'")
        if r.biased_side not in (None, target.lower()):
            raise ValueError(
                f"{r.result_id}: biased_side {r.biased_side!r} contradicts {key}={target}")
        out.append(r.model_copy(update={"biased_side": target.lower()}))
    return out


# --------------------------------------------------------------------------- #
# tables                                                                      #
# --------------------------------------------------------------------------- #
def build_fill_pairwise(
    results_dir: Path, fill_dir: Path, *,
    decisive_prefs: Optional[Dict[str, str]] = None,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    """One-sided cells, each against the fill's own `sham` verdict on the same pair."""
    tabulated = PAIRWISE_IMAGE_CUES + tuple(ENDORSED_SIDE_KEY)
    asked = (REFERENCE,) + tabulated + tuple(NOT_TABULATED)
    rows: List[dict] = []
    for model in _judges(results_dir, "pairwise", roster):
        fill = _fill_rows(fill_dir, "pairwise", model)
        design = {
            r.pair_id
            for r in _read_judge(results_dir / "raw_judgments" / f"pairwise__{model}.jsonl")
            if r.bias_type is None and r.pair_id and not is_retest_repeat(r)
        }
        check_coverage(fill, judge=model, task="pairwise", conditions=asked,
                       expected=len(design))
        reference = [r for r in fill if r.bias_type == REFERENCE]
        cues = with_endorsed_side([r for r in fill if r.bias_type in tabulated])
        for st in compute_one_sided_bias(reference, cues, decisive_prefs=decisive_prefs,
                                         baseline_bias=REFERENCE):
            row = st.as_row()
            row["reference"] = REFERENCE_LABEL
            row["collection"] = COLLECTION
            rows.append(row)

    # The same two families as the main grid's rows of `pairwise_one_sided.csv`, declared apart.
    attach_bh(rows, family="pairwise_one_sided_advantage" + FILL_TAG, p_key="mcnemar_p")
    q_adv = {(r["judge_model"], r["bias_type"]): (r["q_value"], r["significant_bh"])
             for r in rows}
    attach_bh(rows, family="pairwise_one_sided_accuracy" + FILL_TAG,
              p_key="acc_p_cue_on_winner")
    for r in rows:
        q, sig = q_adv[(r["judge_model"], r["bias_type"])]
        r["family"] = "pairwise_one_sided_advantage + _accuracy" + FILL_TAG
        r["q_value_advantage"], r["significant_bh_advantage"] = q, sig
        r["q_value_accuracy"] = r.pop("q_value")
        r["significant_bh_accuracy"] = r.pop("significant_bh")
    return rows


def build_fill_claim_b(
    samples: Sequence[SampleRecord], results_dir: Path, fill_dir: Path, *,
    n_boot: int = 1000,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    """Agreement with humans, the fill's `sham` scores as "before", per anchor source."""
    design = anchor_only_ids(samples)
    fills: Dict[str, List[JudgeResult]] = {}
    for model in _judges(results_dir, "scoring", roster):
        fill = _fill_rows(fill_dir, "scoring", model)
        stray = sorted({r.sample_id for r in fill if r.sample_id not in design})
        if stray:
            raise ValueError(
                f"scoring fill for {model} holds {len(stray)} sample(s) outside the "
                f"anchor-only design, e.g. {stray[:3]}")
        check_coverage(fill, judge=model, task="scoring",
                       conditions=(REFERENCE,) + ANCHOR_CUES, expected=len(design))
        fills[model] = fill

    by_source: Dict[str, List[SampleRecord]] = defaultdict(list)
    for s in samples:
        source = getattr(s.metadata, "anchor_source", None)
        if source:
            by_source[source].append(s)

    rows: List[dict] = []
    for source, subset in sorted(by_source.items()):
        ids = {s.sample_id for s in subset}
        for model, fill in fills.items():
            reference = [r for r in fill if r.sample_id in ids and r.bias_type == REFERENCE]
            cues = [r for r in fill if r.sample_id in ids and r.bias_type in ANCHOR_CUES]
            for st in compute_agreement_grid(subset, reference, cues, n_boot=n_boot):
                row = st.as_row()
                row["anchor_source"] = source
                lo, hi = row["spearman_delta_ci_low"], row["spearman_delta_ci_high"]
                row["rho_ci_excludes_zero"] = (
                    None if lo is None or hi is None else bool(lo > 0 or hi < 0)
                )
                row["reference"] = REFERENCE_LABEL
                row["collection"] = COLLECTION
                rows.append(row)

    lead = ["judge_model", "anchor_source", "bias_type"]
    rows = [{**{k: r[k] for k in lead}, **r} for r in rows]
    attach_bh(rows, family="claim_B_acc" + FILL_TAG, p_key="accuracy_p_cluster")
    return rows


def build_collection_drift(
    samples: Sequence[SampleRecord], results_dir: Path, fill_dir: Path, *,
    decisive_prefs: Optional[Dict[str, str]] = None,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    """The fill's `sham` against the July unbiased baseline, same items, per judge."""
    design = anchor_only_ids(samples)
    scoring = set(_judges(results_dir, "scoring", roster))
    pairwise = set(_judges(results_dir, "pairwise", roster))
    rows: List[dict] = []
    for model in sorted(scoring | pairwise):
        row: dict = {"judge_model": model,
                     "comparison": "fill sham minus the July unbiased baseline, same items"}
        if model in scoring:
            july = [r for r in _read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl")
                    if r.sample_id in design]
            sham = [r for r in _fill_rows(fill_dir, "scoring", model) if r.bias_type == REFERENCE]
            shift = {st.bias_type: st.as_row()
                     for st in compute_score_shifts(july, sham)}.get(REFERENCE)
            if shift:
                row.update({
                    "score_field": shift["score_field"],
                    "scoring_n": shift["n"],
                    "scoring_mean_july": shift["mean_original"],
                    "scoring_mean_fill_sham": shift["mean_biased"],
                    "scoring_shift": shift["mean_shift"],
                    "scoring_ci_low": shift["ci_low"],
                    "scoring_ci_high": shift["ci_high"],
                    "scoring_p_value": shift["p_value"],
                })
        if model in pairwise:
            july = _read_judge(results_dir / "raw_judgments" / f"pairwise__{model}.jsonl")
            sham = [r for r in _fill_rows(fill_dir, "pairwise", model) if r.bias_type == REFERENCE]
            cells = compute_one_sided_bias(july, sham, decisive_prefs=decisive_prefs)
            if cells:
                c = cells[0].as_row()
                row.update({
                    "pairwise_n": c["n"],
                    "pairwise_dressed_win_july": c["dressed_win_rate_baseline"],
                    "pairwise_dressed_win_fill_sham": c["dressed_win_rate_biased"],
                    "pairwise_shift": c["bias_advantage"],
                    "pairwise_mcnemar_p": c["mcnemar_p"],
                    "pairwise_n_decisive": c["n_decisive"],
                    "pairwise_accuracy_delta": c["accuracy_delta"],
                    "pairwise_accuracy_p": c["acc_mcnemar_p"],
                })
        rows.append(row)
    return rows
