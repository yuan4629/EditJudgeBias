"""§9.5 mitigation tables computed entirely from data already on disk — no new calls.

The study design lists five mitigation arms. Two of them cost nothing to run
*now*, because the grid already collected everything they need, and both are measured
at the same scale as the headline (1-10 judges, `fine_score` 3-30, the breadth block):

- **`swap_and_average`** (§9.5 arm 4, protocol level) — every pair was judged in BOTH
  display orders (the baseline condition and the `position` swap). Reconciling the two
  verdicts is what a deployment would do to cancel position bias, and the question is
  whether it buys agreement with humans. ⚠️ There is **nothing numeric to average**: a
  pairwise `JudgeResult` carries only `winner`, so this is verdict RECONCILIATION, and
  the rule for "one order said Tie, the other picked a side" is a real choice that moves
  the answer (Tie counts range from 17 to 117 of 616 across judges). Both policies are
  therefore reported side by side rather than one being buried in the code.

  **TWO significance tests, because the table publishes TWO accuracies and they move in
  OPPOSITE directions.** This was a real reporting defect: the McNemar scored an
  undecided verdict as wrong, so its p-value tested `acc_all`, and it was printed beside
  `acc_on_decided`. On `gpt-4o-viescore / reconciled_strict` that meant publishing
  "0.7264 -> 0.8273, +10.1pp" (the quoted evidence that the mitigation works) next to
  `p=1.7e-21`, which was certifying `acc_all` 0.6849 -> 0.5251 = **-16.0pp**, the exact
  opposite conclusion. Every column of that test now carries an `_acc_all` suffix, and
  the estimand the write-up actually quotes is tested separately and named
  `_on_retained`: the arm's accuracy on the pairs it still answers, against the SAME
  pairs' accuracy under the deployed single-order baseline (a paired McNemar on the
  retained subset — both verdicts are on disk for every pair).
  ★ **That test settles the question, and not in the mitigation's favour.**
  `n_retained_verdict_differs_from_base` is **0 on 5/5 judges** for
  `reconciled_strict`: a pair survives strict reconciliation exactly when both orders
  picked the same side, and the reconciled verdict is then the base-order verdict, so
  on the pairs it keeps the arm returns literally the same answers as the deployed
  baseline. `acc_delta_on_retained` is therefore 0.0000 by construction and the paired
  test is UNDEFINED (b=c=0), not merely non-significant — the whole published
  `acc_on_decided` gain (viescore 0.7264 -> 0.8273) is subset selection.
  ⚠️ `*_on_retained` is **conditional on a subset the judge selected itself**
  (`retained_subset_selection` spells out the rule per arm): a pair survives strict
  reconciliation exactly when the judge agreed with itself across display orders, which
  is correlated with getting the pair right. It is the right test for "is the mitigated
  protocol more accurate *when it answers*", and it is NOT an unbiased estimate of the
  protocol's accuracy on the population — that is what `acc_all` + `coverage` are for.

- **`ensemble_median`** (§9.5 arm 5, ensemble level, multi-JUDGE variant) — the median
  of the five judges' `fine_score` as a single synthetic judge, with claim A recomputed
  against it. The plan's arm 5 is a multi-PROMPT ensemble; that variant only has pilot
  data (1-5 scale), so the deviation is deliberate and named.
  **The expected result is negative and that is the point**: 5 of the cues deflate on
  5/5 judges and `bandwagon` inflates on 5/5, and a median cannot cancel a bias every
  member shares. Ensembling removes independent noise, not correlated bias.

  ★ **Reported in TWO normalisations, and the difference is a correction (WP-F1a).**
  The raw-score version -- what a deployer who medians five raw scores actually gets --
  says the ensemble is WORSE than the member average on 11 of 13 cues, and the
  mechanism offered for that was "the median discards the most robust member as an
  outlier". That mechanism was reading a SCALE OFFSET: on the breadth block four
  baselines sit at 13.4-15.6 `fine_score` and `gpt-4o-viescore` sits at 19.76, so
  viescore is near-permanently the panel MAXIMUM and a median of five raw scores can
  near-permanently not select it, cue or no cue. Standardise each member by its own
  baseline first (`normalisation=z`) and **11/13 becomes 2/13**, both survivors
  differing in the third decimal, and the placebo goes from AMPLIFIED
  (+0.204 -> +0.389) to ATTENUATED (+0.041 -> +0.010).
  ⇒ The negative result survives and its statement changes: **an ensemble FOLLOWS its
  members' average and cannot remove a bias they share** -- it does not add damage of
  its own. Both blocks stay in the table; neither is the other's correction.

Both tables carry the cost column §9.5 demands (`:781-783`: never report that mitigation
worked without reporting what it cost). For M1 the cost is COVERAGE — how many pairs
stop having an answer; for M2 it is `mean_original`, the first-order signal that a
method merely flattened every score instead of ignoring the cue.

    python -m edit_judge_bias.experiments.build_mitigation_tables \\
        --results-dir results/v2 \\
        --samples data/manifests/samples_judge_v2.jsonl \\
        --full-samples data/manifests/samples_full_v2.jsonl \\
        --full-pairs data/manifests/pairs_full_v2.jsonl \\
        --subset-filter subset_block=breadth
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.data.schema import JudgeResult, PairRecord, SampleRecord, TaskType
from edit_judge_bias.experiments.aggregate_results import sample_ids_matching
from edit_judge_bias.experiments.build_claim_tables import (
    PUBLISHED_ROSTER,
    ROSTERS,
    _judges,
    _read_judge,
    attach_bh,
    write_csv,
)
from edit_judge_bias.metrics.pairwise_metrics import (
    _content_winner,
    _winner_by_pair,
    decisive_human_prefs,
)
from edit_judge_bias.metrics.scoring_metrics import (
    ScoringShiftStats,
    compute_score_shifts,
    is_retest_repeat,
    resolve_score_field,
    shift_stats_from_pairs,
)
from edit_judge_bias.metrics.stats import benjamini_hochberg, mcnemar_pvalue

#: The synthetic judge M2 writes. Named, not a real model, so no table can confuse it
#: with a collected arm.
ENSEMBLE = "ensemble-median"

#: The two normalisations M2 reports, in one table, one column apart.
#: ⚠️ `raw` medians the five judges' `fine_score` as collected, which is what a
#: deployer who pipes five judges into a median actually gets — and it is also
#: SCALE-MIXED: on the breadth block the five baselines are 13.4-15.6 except
#: `gpt-4o-viescore` at 19.76, so viescore is very nearly always the panel maximum and
#: the median can barely ever select it, FOR REASONS THAT HAVE NOTHING TO DO WITH
#: BIAS. The published mechanism sentence ("the median throws the most robust member
#: away as an outlier") was reading that offset. `z` standardises each judge by its own
#: baseline mean/sd first, which is the comparison the mechanism claim needs.
RAW = "raw"
Z = "z"

#: Stamped on the M2 rows that are NOT the ensemble. Those 65 rows are claim A for the
#: five panel judges recomputed on the ensemble's common subset — i.e. the SAME
#: hypotheses `claim_a.csv` already corrects. BH-ing them together with the 13 ensemble
#: rows gave one hypothesis two different q-values ("gpt-5.5 x aesthetic_filter != 0"
#: was q=0.059068 here and q=0.062039 there), so they now carry no q at all and name
#: the file that owns them.
MEMBER_FAMILY = "reference (claim_a.csv)"

#: How each arm's `*_on_retained` subset was selected. Never a random subset: the arm
#: itself decides which pairs it still answers, and that decision is correlated with
#: the outcome being measured. Printed in the table so a reader cannot miss it.
RETAINED_SELECTION = {
    "base_order_only":
        "decided under the base order (outcome-correlated: the judge's own Ties dropped)",
    "swapped_order_only":
        "decided under the swapped order (outcome-correlated: the judge's own Ties dropped)",
    "reconciled_strict":
        "both display orders picked the SAME side (outcome-correlated: "
        "the judge agreeing with itself)",
    "reconciled_lenient":
        "the two display orders did not contradict (outcome-correlated: "
        "the judge agreeing with itself)",
}

#: The five §2.1 judges. The ensemble is only defined where all of them answered.
ROSTER = (
    "gpt-5.5",
    "gemini-3.5-flash",
    "gpt-4o-viescore",
    "qwen3.5-plus",
    "kimi-k2.5",
)

#: A reconciled verdict that is not a side. Kept distinct from "Tie": a Tie is the
#: judge saying the two edits are equal, `UNDECIDED` is the protocol saying the judge
#: contradicted itself and the answer was withdrawn.
UNDECIDED = "undecided"


# --------------------------------------------------------------------------- #
# M1 — swap-and-average (protocol-level position mitigation)                   #
# --------------------------------------------------------------------------- #
def _reconcile(base: str, swap: str, *, lenient: bool) -> str:
    """Combine the same pair's verdict under both display orders (content sides).

    `base` and `swap` are already mapped to content sides ('a'/'b'/'Tie'). Under
    `lenient`, a Tie yields to the order that committed to a side; under strict, any
    disagreement withdraws the answer.
    """
    if base == swap:
        return base
    if base == "Tie" or swap == "Tie":
        if not lenient:
            return UNDECIDED
        return swap if base == "Tie" else base
    return UNDECIDED  # the two orders picked opposite sides


def _arm_row(
    judge: str, arm: str, verdicts: Dict[str, str], prefs: Dict[str, str]
) -> Tuple[dict, Dict[str, bool], Dict[str, bool]]:
    """Score one arm against the decisive human preferences.

    Returns the row plus per-pair correctness AND per-pair decidedness, so a later
    McNemar can pair this arm against the single-order baseline both on every decisive
    pair (`acc_all`) and on the subset this arm still answers (`acc_on_decided`).
    """
    correct: Dict[str, bool] = {}
    decided_by_pair: Dict[str, bool] = {}
    n_decided = 0
    for pair_id, verdict in verdicts.items():
        human = prefs.get(pair_id)
        if human is None:
            continue
        decided = verdict in ("a", "b")
        decided_by_pair[pair_id] = decided
        n_decided += int(decided)
        correct[pair_id] = decided and verdict == human
    n = len(correct)
    n_correct = sum(correct.values())
    return (
        {
            "judge_model": judge,
            "arm": arm,
            "n_pairs": n,
            "n_decided": n_decided,
            # The COST of the protocol: pairs that no longer have an answer.
            "coverage": round(n_decided / n, 4) if n else "",
            "n_correct": n_correct,
            # Accuracy among pairs the arm still answers — flattered by low coverage,
            # which is why it is never reported without `coverage` beside it, and why
            # its own significance test is the `*_on_retained` one, not `*_acc_all`.
            "acc_on_decided": round(n_correct / n_decided, 4) if n_decided else "",
            # Accuracy over every decisive pair, an undecided answer counting as wrong.
            # This is the comparable number across arms because the denominator is fixed.
            "acc_all": round(n_correct / n, 4) if n else "",
        },
        correct,
        decided_by_pair,
    )


def build_swap_average(
    results_dir: Path, *, decisive_prefs: Dict[str, str],
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    """Single-order vs order-reconciled agreement with humans, per judge.

    ⚠️ THE ROSTER, NOT THE COVERAGE GUARD, IS WHAT KEEPS THIS TABLE STABLE -- caught
    2026-08-18 by rebuilding mid-collection and diffing.  This builder used to call
    `_discover_models` directly, which asks only "is this judge finished?", and WP-A5's
    `qwen3-vl-32b-instruct` passed that test at 578 of 616 base-order pairs.  It entered
    the table as a 5th judge, took the BH family from 20 rows to 25, and moved five
    PUBLISHED q-values -- `gemini x swapped_order_only` on the retained subset went
    0.022036 -> 0.026443 while staying "significant", i.e. the paper's number changed and
    nothing about the output said so.

    The coverage guard cannot express this: by design it ADMITS a judge once collection
    completes, and the user ruled (2026-08-18) that A5's open-weights judges form their
    own declared family so the published five stay byte-stable permanently.  That is a
    roster question, and `_judges` is the one place that answers it.

    The same reasoning already protected `claim_a` / `claim_b` / `retest` / the pairwise
    tables and `build_analysis_tables` (hardcoded `ROSTER`), and `build_ensemble_median`
    below was safe for the same reason -- this builder was the single remaining door.
    """
    rows: List[dict] = []
    for model in _judges(results_dir, "pairwise", roster):
        base_raw = _read_judge(results_dir / "raw_judgments" / f"pairwise__{model}.jsonl")
        biased = _read_judge(results_dir / "biased_judgments" / f"pairwise__{model}.jsonl")
        # Both sides filter to their own condition, and the baseline lookup also drops
        # the CR repeats that share its `bias_type=None` and `pair_id`.
        base = _winner_by_pair(base_raw, bias_type=None)
        swap = _winner_by_pair(biased, bias_type="position")
        pairs = sorted(
            {p for (m, p) in base if m == model} & {p for (m, p) in swap if m == model}
        )
        if not pairs:
            continue

        as_base = {p: _content_winner(base[(model, p)], False) for p in pairs}
        as_swap = {p: _content_winner(swap[(model, p)], True) for p in pairs}
        arms = {
            "base_order_only": as_base,
            "swapped_order_only": as_swap,
            "reconciled_strict": {
                p: _reconcile(as_base[p], as_swap[p], lenient=False) for p in pairs
            },
            "reconciled_lenient": {
                p: _reconcile(as_base[p], as_swap[p], lenient=True) for p in pairs
            },
        }

        scored = {
            arm: _arm_row(model, arm, verdicts, decisive_prefs)
            for arm, verdicts in arms.items()
        }
        baseline_correct = scored["base_order_only"][1]
        for arm, (row, correct, decided) in scored.items():
            verdicts = arms[arm]
            row["n_pairs_judged"] = len(pairs)
            is_baseline = arm == "base_order_only"

            # --- test 1: acc_all, over EVERY decisive pair, undecided counting as wrong.
            # Paired against the deployed single-order protocol: b = right before and
            # wrong after, c = the reverse. This is the number `acc_all` reports, and the
            # suffix says so — it is NOT a test of `acc_on_decided`, which moves the
            # other way whenever the mitigation buys accuracy by withdrawing answers.
            b = sum(1 for p, ok in baseline_correct.items() if ok and not correct.get(p, False))
            c = sum(1 for p, ok in baseline_correct.items() if not ok and correct.get(p, False))
            row["mcnemar_b_acc_all"] = b
            row["mcnemar_c_acc_all"] = c
            row["mcnemar_p_acc_all"] = None if is_baseline else mcnemar_pvalue(b, c)

            # --- test 2: the estimand the write-up quotes. Restrict to the pairs this
            # arm still answers and compare its accuracy there against the BASELINE's
            # accuracy on those same pairs (paired McNemar; a baseline Tie counts as
            # wrong, exactly as it does everywhere else in this table).
            retained = [p for p, ok in decided.items() if ok]
            n_ret = len(retained)
            base_ok = sum(1 for p in retained if baseline_correct.get(p, False))
            arm_ok = sum(1 for p in retained if correct.get(p, False))
            b2 = sum(1 for p in retained
                     if baseline_correct.get(p, False) and not correct.get(p, False))
            c2 = sum(1 for p in retained
                     if not baseline_correct.get(p, False) and correct.get(p, False))
            row["n_retained"] = n_ret
            row["baseline_acc_on_retained"] = round(base_ok / n_ret, 4) if n_ret else ""
            row["acc_delta_on_retained"] = (
                round(arm_ok / n_ret - base_ok / n_ret, 4) if n_ret else ""
            )
            # ★ The mechanism, in one integer. `reconciled_strict` retains a pair
            # exactly when both orders picked the SAME side, and then the reconciled
            # verdict IS the base-order verdict — so this is 0 for every judge and the
            # arm CANNOT be more accurate than the baseline on the pairs it keeps.
            # Its entire published `acc_on_decided` gain (viescore 0.7264 -> 0.8273) is
            # therefore subset selection, not a better answer, and the paired test is
            # undefined (b=c=0) rather than merely non-significant.
            row["n_retained_verdict_differs_from_base"] = sum(
                1 for p in retained if verdicts[p] != as_base[p]
            )
            row["mcnemar_b_on_retained"] = b2
            row["mcnemar_c_on_retained"] = c2
            row["mcnemar_p_on_retained"] = None if is_baseline else mcnemar_pvalue(b2, c2)
            # ⚠️ The retained subset is never random — the arm chose it, on a variable
            # correlated with being right. Spelled out per row so no reader can miss it.
            row["retained_subset_outcome_selected"] = True
            row["retained_subset_selection"] = RETAINED_SELECTION.get(arm, "")

            # Every row carries every key, so the CSV column order cannot depend on
            # which arm happened to be written first.
            for suffix in ("_acc_all", "_on_retained"):
                row[f"family{suffix}"] = "baseline"
                row[f"q_value{suffix}"] = None
                row[f"significant_bh{suffix}"] = None
            rows.append(row)

    tested = [r for r in rows if r["arm"] != "base_order_only"]
    # Two families, because they are two different hypotheses about two different
    # quantities on two different denominators — not one family of 30.
    _stamp_bh(tested, family="mitigation_swap_acc_all",
              p_key="mcnemar_p_acc_all", suffix="_acc_all")
    _stamp_bh(tested, family="mitigation_swap_acc_on_retained",
              p_key="mcnemar_p_on_retained", suffix="_on_retained")
    return sorted(rows, key=lambda r: (r["judge_model"], r["arm"]))


def _stamp_bh(rows: List[dict], *, family: str, p_key: str, suffix: str) -> None:
    """`attach_bh`, but writing family/q/significant under a suffixed column name.

    One table, two families: `attach_bh`'s fixed `family`/`q_value`/`significant_bh`
    names can only carry one of them, and an unlabelled q beside two p-columns is the
    defect this module was fixing.
    """
    qs = benjamini_hochberg([r.get(p_key) for r in rows])
    for row, q in zip(rows, qs):
        row[f"family{suffix}"] = family
        row[f"q_value{suffix}"] = None if q is None else round(q, 6)
        row[f"significant_bh{suffix}"] = None if q is None else bool(q < 0.05)


# --------------------------------------------------------------------------- #
# M2 — five-judge ensemble median (ensemble-level mitigation)                  #
# --------------------------------------------------------------------------- #
def _scores(results: Sequence[JudgeResult], field: str) -> Dict[Tuple[str, Optional[str]], int]:
    """(sample_id, bias_type) -> score, for one judge's parsed non-repeat rows.

    ⚠️ The key is `(sample_id, bias_type)` and NOT `biased_id`: `bandwagon` and
    `model_name` are prompt-level cues that never produced an image, so their rows
    carry `biased_id=None` and a biased_id join drops both conditions silently.
    """
    out: Dict[Tuple[str, Optional[str]], int] = {}
    for r in results:
        if not r.parse_success or not r.sample_id or is_retest_repeat(r):
            continue
        value = getattr(r, field, None)
        if value is None:
            continue
        out[(r.sample_id, r.bias_type)] = int(value)
    return out


def _synthetic(
    sample_id: str, bias_type: Optional[str], score: int, field: str
) -> JudgeResult:
    """One ensemble row shaped exactly like a collected one so the metrics accept it."""
    suffix = f"__{bias_type}" if bias_type else ""
    kwargs = {
        "result_id": f"score::{ENSEMBLE}::vanilla::{sample_id}{suffix}",
        "judge_model": ENSEMBLE,
        "task_type": TaskType.SCORING,
        # Synthetic rows have no raw model response; the path is a marker, not a file.
        "raw_response_path": Path("synthetic/ensemble_median"),
        "parse_success": True,
        "prompt_type": "vanilla_scoring",
        "sample_id": sample_id,
        "bias_type": bias_type,
        # Deliberately left None: `biased_id` is a real artefact id elsewhere and this
        # row has no artefact. compute_score_shifts only uses it to dedup, and the
        # construction below is already one row per (sample_id, bias_type).
        "biased_id": None,
        "score_scale": 10,
        field: score,
    }
    return JudgeResult(**kwargs)


def build_ensemble_median(
    results_dir: Path, keep_sample_ids: Optional[set] = None
) -> List[dict]:
    """Claim A recomputed for the median of the roster, against each member."""
    def keep(results: List[JudgeResult]) -> List[JudgeResult]:
        if keep_sample_ids is None:
            return results
        return [r for r in results if r.sample_id in keep_sample_ids]

    originals: Dict[str, List[JudgeResult]] = {}
    biased: Dict[str, List[JudgeResult]] = {}
    for model in ROSTER:
        originals[model] = keep(
            _read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl")
        )
        biased[model] = keep(
            _read_judge(results_dir / "biased_judgments" / f"scoring__{model}.jsonl")
        )
    present = [m for m in ROSTER if biased.get(m)]
    if len(present) < 2:
        return []

    pooled = [r for m in present for r in originals[m] + biased[m]]
    field = resolve_score_field(pooled)

    orig_scores = {m: _scores(originals[m], field) for m in present}
    bias_scores = {m: _scores(biased[m], field) for m in present}

    # A cell counts only where EVERY member answered, so the median is always over the
    # same panel; a median of whichever judges happened to parse would change composition
    # cell by cell and the "ensemble" would not be one thing.
    common_orig = set.intersection(*[{k for k in s if k[1] is None} for s in orig_scores.values()])
    common_bias = set.intersection(*[set(s) for s in bias_scores.values()])

    ens_orig = [
        _synthetic(sid, None, int(statistics.median(orig_scores[m][(sid, None)] for m in present)), field)
        for (sid, _b) in sorted(common_orig)
    ]
    ens_bias = [
        _synthetic(sid, bias, int(statistics.median(bias_scores[m][(sid, bias)] for m in present)), field)
        for (sid, bias) in sorted(common_bias)
    ]

    # Restrict every member to the SAME cells, so "the ensemble beat gpt-5.5" is not an
    # artefact of the two being measured on different samples.
    ids_by_bias: Dict[str, set] = {}
    for sid, bias in common_bias:
        ids_by_bias.setdefault(bias or "unknown", set()).add(sid)
    orig_ids = {sid for sid, _ in common_orig}

    rows: List[dict] = []
    member_stats: Dict[Tuple[str, str], ScoringShiftStats] = {}
    for arm in list(present) + [ENSEMBLE]:
        if arm == ENSEMBLE:
            o_rows, b_rows = ens_orig, ens_bias
        else:
            o_rows = [r for r in originals[arm] if r.sample_id in orig_ids]
            b_rows = [
                r for r in biased[arm]
                if r.sample_id in ids_by_bias.get(r.bias_type or "unknown", set())
            ]
        for st in compute_score_shifts(o_rows, b_rows, score_field=field):
            rows.append(_ensemble_row(st, arm, RAW))
            if arm != ENSEMBLE:
                member_stats[(arm, st.bias_type)] = st

    rows += _z_normalised_rows(
        present, orig_scores, bias_scores, common_orig, common_bias, field,
        member_stats,
    )

    _attach_reduction(rows)
    # ⚠️ ONLY the 13 ensemble rows are this table's hypotheses. The 65 member rows are
    # claim A for the five panel judges, recomputed on the ensemble's common subset —
    # the same hypotheses `claim_a.csv` states and corrects. Putting all 78 in one BH
    # family gave "gpt-5.5 x aesthetic_filter != 0" q=0.059068 here and q=0.062039
    # there: one hypothesis, two q-values, and a reader free to quote the smaller. The
    # members stay in the table (the ensemble is meaningless without what it is built
    # from) but carry no q and name the file that owns their correction.
    # One BH family per normalisation, never one over both: the raw and the z ensemble
    # ask the SAME 13 questions twice, so pooling them would correct 26 p-values for 13
    # hypotheses and hand a reader two q-values per hypothesis to choose between.
    attach_bh([r for r in rows if r["is_ensemble"] and r["normalisation"] == RAW],
              family="mitigation_ensemble", p_key="p_value")
    attach_bh([r for r in rows if r["is_ensemble"] and r["normalisation"] == Z],
              family="mitigation_ensemble_z", p_key="p_value")
    for r in rows:
        if not r["is_ensemble"]:
            # A member's z row is its raw row divided by a positive constant, so it is
            # the SAME hypothesis (identical Wilcoxon p, identical SIR) expressed in SD
            # units -- still `claim_a.csv`'s to correct, in either normalisation.
            r["family"] = MEMBER_FAMILY
            r["q_value"] = None
            r["significant_bh"] = None
    return sorted(
        rows,
        key=lambda r: (r["bias_type"], r["normalisation"], r["is_ensemble"], r["arm"]),
    )


def _ensemble_row(st, arm: str, normalisation: str) -> dict:
    """One M2 table row, built the same way whichever normalisation produced it."""
    r = st.as_row()
    return {
        "bias_type": r["bias_type"],
        "arm": arm,
        "normalisation": normalisation,
        "is_ensemble": arm == ENSEMBLE,
        "score_field": r["score_field"],
        "n": r["n"],
        # The trade-off signal: an ensemble that merely flattens absolute scores
        # shows it here, not in mean_shift.
        "mean_original": r["mean_original"],
        "mean_shift": r["mean_shift"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "sir": r["sir"],
        "p_value": r["p_value"],
    }


def _in_sd_units(st: ScoringShiftStats, mu: float, sd: float) -> ScoringShiftStats:
    """One member's claim-A row expressed in that member's own baseline SDs.

    ⚠️ This is a RESCALE of the already-computed row, not a recomputation, and the
    difference is not cosmetic. Recomputing it from z-scored pairs -- shift =
    `(b-mu)/sd - (o-mu)/sd` instead of `(b-o)/sd` -- is algebraically the same number
    and numerically is not: the raw shifts are INTEGERS with hundreds of exact ties,
    and routing each through two roundings splits those ties apart. The Wilcoxon tie
    correction then reads a different sample: measured here, 33 of 65 member rows moved
    (`gemini x aesthetic_filter` p 0.349007 -> 0.263429) for a transformation that
    cannot move a rank test at all. Same hypothesis, same data, a p-value invented by
    floating point.

    So `p_value` and `sir` are carried over unchanged -- a strictly positive rescale
    preserves every sign and every rank, by definition -- and only the quantities that
    carry units are divided.
    """
    return ScoringShiftStats(
        judge_model=st.judge_model,
        bias_type=st.bias_type,
        n=st.n,
        mean_original=(st.mean_original - mu) / sd,
        mean_biased=(st.mean_biased - mu) / sd,
        mean_shift=st.mean_shift / sd,
        sir=st.sir,
        asc=st.asc / sd,
        ci_low=None if st.ci_low is None else st.ci_low / sd,
        ci_high=None if st.ci_high is None else st.ci_high / sd,
        p_value=st.p_value,
        score_field=st.score_field,
        shifts=[x / sd for x in st.shifts],
    )


def _z_normalised_rows(
    present: Sequence[str],
    orig_scores: Dict[str, Dict[Tuple[str, Optional[str]], int]],
    bias_scores: Dict[str, Dict[Tuple[str, Optional[str]], int]],
    common_orig: set,
    common_bias: set,
    field: str,
    member_stats: Dict[Tuple[str, str], ScoringShiftStats],
) -> List[dict]:
    """The same ensemble, after each member is standardised by its OWN baseline.

    Why this exists: the raw-score version reports the ensemble as worse than the
    member average on 11 of 13 cues, and the mechanism offered for that was "the
    median discards the most robust member as an outlier". But the five baselines are
    not on one scale -- `gpt-4o-viescore` sits ~4-6 `fine_score` points above the other
    four -- so viescore is near-permanently the panel MAXIMUM and a median over raw
    scores is near-permanently unable to select it, cue or no cue. Standardising each
    member first removes that offset and leaves the question the sentence was trying to
    answer.

    ⚠️ The raw rows are NOT superseded and are not deleted: they are what a deployer who
    medians five raw scores actually gets. What the z rows correct is the *mechanism*.

    Returns `[]` rather than raising when the panel cannot be standardised -- a member
    with a degenerate baseline (sd = 0). That is a real state in fixtures and tiny
    rosters, it is not an error, and the raw block still stands on its own.
    """
    baselines: Dict[str, Tuple[float, float]] = {}
    for m in present:
        vals = [float(orig_scores[m][k]) for k in sorted(common_orig)]
        if len(vals) < 2:
            return []
        sd = statistics.stdev(vals)   # sample sd; at n in the hundreds the n-1 vs n
        if sd <= 0:                   # choice does not reach the 4th decimal
            print(f"  skip z-normalised ensemble: {m} has a degenerate baseline "
                  f"(sd=0 over {len(vals)} originals)")
            return []
        baselines[m] = (statistics.fmean(vals), sd)

    def z(model: str, value: float) -> float:
        mu, sd = baselines[model]
        return (float(value) - mu) / sd

    ens_orig = {
        sid: statistics.median(z(m, orig_scores[m][(sid, None)]) for m in present)
        for (sid, _b) in common_orig
    }
    cells: Dict[Optional[str], List[str]] = {}
    for sid, bias in common_bias:
        cells.setdefault(bias, []).append(sid)

    rows: List[dict] = []
    for bias in sorted(cells, key=lambda b: b or "unknown"):
        label = bias or "unknown"
        sids = sorted(s for s in cells[bias] if s in ens_orig)
        if not sids:
            continue
        ens_pairs = [
            (ens_orig[s],
             statistics.median(z(m, bias_scores[m][(s, bias)]) for m in present))
            for s in sids
        ]
        # The ensemble row is genuinely recomputed, not rescaled: median(z) is a
        # different statistic from z(median), which is the entire point of the block.
        rows.append(_ensemble_row(
            shift_stats_from_pairs(ENSEMBLE, label, ens_pairs, score_field=field),
            ENSEMBLE, Z,
        ))
        for m in present:
            st = member_stats.get((m, label))
            if st is None:      # a member with no row for this cue: nothing to rescale
                continue
            mu, sd = baselines[m]
            rows.append(_ensemble_row(_in_sd_units(st, mu, sd), m, Z))
    return rows


def _attach_reduction(rows: List[dict]) -> None:
    """Per bias, compare the ensemble against the members it is built from.

    `abs_reduction > 0` means the ensemble sits closer to zero than the average member
    — the same convention the pilot's prompt-level mitigation analysis used, so the
    two mitigation tables read the same way.
    """
    # ⚠️ Keyed by (normalisation, cue), not by cue: a z ensemble compared against raw
    # members would divide SD units by score points and print the ratio as a
    # "reduction". Same defect class as mixing two score scales in one column.
    members: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        if not r["is_ensemble"] and isinstance(r["mean_shift"], (int, float)):
            members.setdefault(
                (r["normalisation"], r["bias_type"]), []
            ).append(float(r["mean_shift"]))
    for r in rows:
        vals = members.get((r["normalisation"], r["bias_type"]), [])
        if not r["is_ensemble"] or not vals:
            r["mean_member_shift"] = ""
            r["worst_member_shift"] = ""
            r["abs_reduction"] = ""
            r["pct_reduction"] = ""
            continue
        mean_member = sum(vals) / len(vals)
        r["mean_member_shift"] = round(mean_member, 4)
        r["worst_member_shift"] = round(max(vals, key=abs), 4)
        abs_red = abs(mean_member) - abs(float(r["mean_shift"]))
        r["abs_reduction"] = round(abs_red, 4)
        r["pct_reduction"] = round(abs_red / abs(mean_member), 4) if mean_member else ""


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def build_all(
    results_dir: Path,
    *,
    samples_path: Path,
    full_samples: Optional[Path] = None,
    full_pairs: Optional[Path] = None,
    subset_filter: Optional[dict] = None,
    out_dir: Optional[Path] = None,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
    prefix: str = "",
) -> Dict[str, List[dict]]:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "metrics"
    keep_ids = sample_ids_matching(samples_path, subset_filter) if subset_filter else None

    prefs: Dict[str, str] = {}
    if full_samples and full_pairs:
        # ⚠️ The decisive threshold is an SD of each source's OWN human scores and must
        # be computed on the full pool: on the judging subset the same rule marks 293
        # pairs instead of 438 and drops GenAI-Bench entirely.
        prefs = decisive_human_prefs(
            io.read_jsonl(Path(full_samples), SampleRecord),
            io.read_jsonl(Path(full_pairs), PairRecord),
        )

    tables = {
        "mitigation_swap_average": build_swap_average(
            results_dir, decisive_prefs=prefs, roster=roster
        ),
    }
    # M2's estimand is "the MEDIAN of the panel", which is only defined for a panel: over
    # two members a median IS the mean, a different statistic wearing the same column
    # name.  So a small roster gets no ensemble row rather than a silently redefined one.
    if roster is None or len(roster) >= 3:
        tables["mitigation_ensemble"] = build_ensemble_median(results_dir, keep_ids)
    else:
        print(f"  skip mitigation_ensemble: a median needs >=3 members, roster has "
              f"{len(roster)}")
    # `prefix` keeps a non-published roster in its OWN files -- writing a second roster's
    # rows over `mitigation_swap_average.csv` is exactly the corruption the roster exists
    # to prevent.
    for name, rows in tables.items():
        write_csv(out_dir / f"{prefix}{name}.csv", rows)
    return tables


def _print_summary(tables: Dict[str, List[dict]], n_decisive: int) -> None:
    print(f"=== §9.5 M1 swap-and-average (protocol level), {n_decisive} decisive pairs ===")
    print("two estimands, two BH families: acc_all (every decisive pair, undecided = "
          "wrong)\nand acc_on_decided vs the baseline ON THE SAME RETAINED PAIRS "
          "(subset chosen by the judge)")
    print(f"{'judge':<18}{'arm':<20}{'n':>5}{'cover':>8}{'acc_dec':>9}{'base_ret':>9}"
          f"{'d_ret':>8}{'q_ret':>9}{'acc_all':>9}{'q_all':>9}")
    for r in tables["mitigation_swap_average"]:
        def _q(key: str) -> str:
            return "" if r[key] is None else f"{r[key]:.4f}"
        print(f"{r['judge_model']:<18}{r['arm']:<20}{r['n_pairs']:>5}"
              f"{str(r['coverage']):>8}{str(r['acc_on_decided']):>9}"
              f"{str(r['baseline_acc_on_retained']):>9}"
              f"{str(r['acc_delta_on_retained']):>8}{_q('q_value_on_retained'):>9}"
              f"{str(r['acc_all']):>9}{_q('q_value_acc_all'):>9}")

    if "mitigation_ensemble" not in tables:
        # a roster too small for a median; `build_all` already said so and skipped it
        return
    print("\n=== §9.5 M2 five-judge ensemble median (claim A) ===")
    print(f"{'bias':<18}{'arm':<18}{'n':>6}{'orig':>8}{'shift':>9}{'absRed':>9}")
    for r in tables["mitigation_ensemble"]:
        mark = " *" if r["is_ensemble"] else "  "
        print(f"{r['bias_type']:<18}{r['arm']:<18}{r['n']:>6}{r['mean_original']:>8}"
              f"{r['mean_shift']:>9}{str(r['abs_reduction']):>9}{mark}")


def main(argv: Optional[List[str]] = None) -> int:
    root = default_root()
    ap = argparse.ArgumentParser(description="Build the §9.5 zero-cost mitigation tables.")
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--samples", type=Path,
                    default=root / "data" / "manifests" / "samples_judge_v2.jsonl")
    ap.add_argument("--full-samples", type=Path, default=None,
                    help="full sample pool — the decisive threshold must be computed on it")
    ap.add_argument("--full-pairs", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--subset-filter", action="append", default=None, metavar="KEY=VALUE")
    ap.add_argument("--roster", choices=sorted(ROSTERS), default="published",
                    help="declared judge family; anything but `published` writes to "
                         "prefixed files so the published table cannot be overwritten")
    args = ap.parse_args(argv)

    flt = None
    if args.subset_filter:
        flt = {}
        for item in args.subset_filter:
            key, _, value = item.partition("=")
            if not _:
                ap.error(f"--subset-filter expects KEY=VALUE, got {item!r}")
            flt[key] = value

    tables = build_all(
        args.results_dir,
        samples_path=args.samples,
        full_samples=args.full_samples,
        full_pairs=args.full_pairs,
        subset_filter=flt,
        out_dir=args.out_dir,
        roster=ROSTERS[args.roster],
        prefix="" if args.roster == "published" else f"{args.roster}_",
    )
    n_decisive = 0
    if args.full_samples and args.full_pairs:
        n_decisive = len(decisive_human_prefs(
            io.read_jsonl(args.full_samples, SampleRecord),
            io.read_jsonl(args.full_pairs, PairRecord),
        ))
    _print_summary(tables, n_decisive)
    out = args.out_dir or (Path(args.results_dir) / "metrics")
    print(f"\nwrote {len(tables)} tables -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
