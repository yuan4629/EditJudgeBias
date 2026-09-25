"""Tests for the §9.5 zero-cost mitigation tables (swap-and-average, ensemble median)."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult
from edit_judge_bias.experiments.build_mitigation_tables import (
    ENSEMBLE,
    MEMBER_FAMILY,
    RAW,
    ROSTER,
    UNDECIDED,
    Z,
    _reconcile,
    build_all,
    build_ensemble_median,
    build_swap_average,
)


def _score(model, sid, fine, bias=None, *, scale=10, repeat=1, biased_id=...):
    return JudgeResult(
        result_id=f"{model}::{sid}::{bias}::{repeat}", judge_model=model,
        task_type="scoring", prompt_type="vanilla_scoring", raw_response_path="r.txt",
        parse_success=True, sample_id=sid, bias_type=bias,
        biased_id=(f"{sid}__{bias}" if bias else None) if biased_id is ... else biased_id,
        fine_score=fine, score_scale=scale,
        bias_params={"repeat_index": repeat} if repeat > 1 else {},
    )


def _pw(model, pair_id, winner, *, bias=None, side=None, repeat=1):
    return JudgeResult(
        result_id=f"{model}::{pair_id}::{bias}::{side}::{repeat}", judge_model=model,
        task_type="pairwise", prompt_type="vanilla_pairwise", raw_response_path="r.txt",
        parse_success=True, pair_id=pair_id, winner=winner,
        bias_type=bias, biased_side=side,
        bias_params={"repeat_index": repeat} if repeat > 1 else {},
    )


def _write_pairwise(root: Path, model: str, base, biased) -> None:
    io.write_jsonl(root / "raw_judgments" / f"pairwise__{model}.jsonl", base)
    io.write_jsonl(root / "biased_judgments" / f"pairwise__{model}.jsonl", biased)


# --------------------------------------------------------------------------- #
# M1 — swap-and-average                                                       #
# --------------------------------------------------------------------------- #
# These test the reconciliation ARITHMETIC on a synthetic judge, so they pass
# `roster=None` ("every complete judge on disk").  Saying it explicitly is the point:
# the default roster is `PUBLISHED_ROSTER`, and a test that silently relied on
# discovery-by-glob is how `qwen3-vl-32b-instruct` reached the published BH family.

def test_reconciliation_withdraws_a_verdict_the_two_orders_contradict():
    # Same content side under both orders -> keep it.
    assert _reconcile("a", "a", lenient=False) == "a"
    # Opposite sides -> the judge contradicted itself; no policy rescues this.
    assert _reconcile("a", "b", lenient=False) == UNDECIDED
    assert _reconcile("a", "b", lenient=True) == UNDECIDED
    # One order abstained: this is the choice that moves the number, so both are kept.
    assert _reconcile("Tie", "b", lenient=False) == UNDECIDED
    assert _reconcile("Tie", "b", lenient=True) == "b"
    assert _reconcile("Tie", "Tie", lenient=True) == "Tie"


def test_swap_average_reports_coverage_as_the_cost_of_the_protocol(tmp_path: Path):
    """§9.5 forbids reporting that a mitigation worked without its cost. Here the cost
    is not accuracy but COVERAGE: p2's two orders disagree, so the reconciled protocol
    stops answering it, and `acc_on_decided` rises only because the denominator fell."""
    results = tmp_path / "v2"
    base = [_pw("j", "p1", "A"), _pw("j", "p2", "A"), _pw("j", "p3", "A")]
    swapped = [
        _pw("j", "p1", "B", bias="position", side="swap"),   # B under swap == side 'a'
        _pw("j", "p2", "A", bias="position", side="swap"),   # A under swap == side 'b'
        _pw("j", "p3", "B", bias="position", side="swap"),
    ]
    _write_pairwise(results, "j", base, swapped)
    prefs = {"p1": "a", "p2": "a", "p3": "a"}

    rows = {r["arm"]: r for r in build_swap_average(results, decisive_prefs=prefs, roster=None)}
    assert rows["base_order_only"]["coverage"] == 1.0
    assert rows["base_order_only"]["acc_all"] == 1.0
    strict = rows["reconciled_strict"]
    assert strict["n_decided"] == 2 and strict["coverage"] < 1.0
    assert strict["acc_on_decided"] == 1.0     # flattered by the smaller denominator
    assert strict["acc_all"] < 1.0             # the comparable number: p2 lost its answer


def test_a_pairwise_cr_repeat_is_not_read_as_the_base_order(tmp_path: Path):
    """The CR arm re-asks the BASELINE with the cache off, so its rows carry
    `bias_type=None` and the same `pair_id` and land in the same file, written later. A
    last-write-wins lookup would reconcile the swap against a *repeat* instead of the
    baseline — and the repeat here answers the opposite way, so the bug is visible as a
    verdict that should never be withdrawn."""
    results = tmp_path / "v2"
    base = [
        _pw("j", "p1", "A"),                    # the baseline
        _pw("j", "p1", "B", repeat=2),          # the CR repeat, written after it
    ]
    swapped = [_pw("j", "p1", "B", bias="position", side="swap")]  # == side 'a', agrees
    _write_pairwise(results, "j", base, swapped)

    rows = {r["arm"]: r for r in build_swap_average(results, decisive_prefs={"p1": "a"}, roster=None)}
    assert rows["reconciled_strict"]["n_decided"] == 1
    assert rows["reconciled_strict"]["acc_all"] == 1.0


def test_the_significance_test_names_which_accuracy_it_tests(tmp_path: Path):
    """DEFECT: one McNemar scored undecided as WRONG — i.e. it tested `acc_all` — and
    was printed beside `acc_on_decided`, the column the write-up quotes. On the real
    grid that published `gpt-4o-viescore / reconciled_strict` acc_on_decided
    0.7264 -> 0.8273 (+10.1pp, "the mitigation works") next to p=1.7e-21, which was
    certifying acc_all 0.6849 -> 0.5251 = -16.0pp, the opposite conclusion.

    Constructed so the two estimands point in OPPOSITE directions: reconciliation drops
    the three pairs the baseline got right by guessing but keeps the one it got wrong
    and the swap fixes. `acc_on_decided` rises, `acc_all` falls, and each column must
    carry its own p/q under a name that says which one it tested."""
    results = tmp_path / "v2"
    base = [_pw("j", f"p{i}", "A") for i in range(4)]
    swapped = [
        # p0: both orders say content side 'a' -> retained, and correct.
        _pw("j", "p0", "B", bias="position", side="swap"),
        # p1..p3: the orders contradict -> withdrawn, though the base order was right.
        _pw("j", "p1", "A", bias="position", side="swap"),
        _pw("j", "p2", "A", bias="position", side="swap"),
        _pw("j", "p3", "A", bias="position", side="swap"),
    ]
    _write_pairwise(results, "j", base, swapped)
    prefs = {"p0": "a", "p1": "a", "p2": "a", "p3": "a"}

    rows = {r["arm"]: r for r in build_swap_average(results, decisive_prefs=prefs, roster=None)}
    strict = rows["reconciled_strict"]
    # The two quantities disagree in SIGN, which is the whole point.
    assert strict["acc_on_decided"] == 1.0 and rows["base_order_only"]["acc_on_decided"] == 1.0
    assert strict["acc_all"] == 0.25 and rows["base_order_only"]["acc_all"] == 1.0

    # Every acc_all column says acc_all in its name...
    assert strict["mcnemar_b_acc_all"] == 3 and strict["mcnemar_c_acc_all"] == 0
    assert strict["family_acc_all"] == "mitigation_swap_acc_all"
    assert strict["significant_bh_acc_all"] is not None
    # ...and none of the old ambiguous names survives, so nothing can quote a q-value
    # without knowing which accuracy it belongs to.
    for gone in ("mcnemar_p", "mcnemar_b", "mcnemar_c", "q_value", "significant_bh",
                 "family"):
        assert gone not in strict

    # The estimand actually claimed is tested separately, on the retained pairs only,
    # against the SAME pairs under the baseline.
    assert strict["n_retained"] == 1
    assert strict["baseline_acc_on_retained"] == 1.0
    assert strict["acc_delta_on_retained"] == 0.0
    assert strict["family_on_retained"] == "mitigation_swap_acc_on_retained"
    # And the retained subset is flagged as chosen by the judge, not at random.
    assert strict["retained_subset_outcome_selected"] is True
    assert "agreeing with itself" in strict["retained_subset_selection"]


def test_strict_reconciliation_returns_the_baseline_verdict_on_every_pair_it_keeps(
    tmp_path: Path,
):
    """★ Why `acc_on_decided` could never have been evidence that the protocol answers
    BETTER. A pair survives strict reconciliation exactly when both display orders
    picked the same side, and the reconciled verdict is then the base-order verdict —
    so on its retained subset the arm is verdict-for-verdict the deployed baseline. The
    paired test is UNDEFINED (b=c=0), not non-significant, and the entire published
    gain is subset selection. Measured on the real grid: 0 differing verdicts on 5/5
    judges."""
    results = tmp_path / "v2"
    base = [_pw("j", "p0", "A"), _pw("j", "p1", "B"), _pw("j", "p2", "Tie")]
    swapped = [
        _pw("j", "p0", "B", bias="position", side="swap"),    # agrees: side 'a'
        _pw("j", "p1", "A", bias="position", side="swap"),    # agrees: side 'b'
        _pw("j", "p2", "A", bias="position", side="swap"),    # base Tie, swap side 'b'
    ]
    _write_pairwise(results, "j", base, swapped)
    prefs = {"p0": "a", "p1": "a", "p2": "b"}

    rows = {r["arm"]: r for r in build_swap_average(results, decisive_prefs=prefs, roster=None)}
    strict = rows["reconciled_strict"]
    assert strict["n_retained"] == 2
    assert strict["n_retained_verdict_differs_from_base"] == 0
    assert strict["mcnemar_b_on_retained"] == 0 and strict["mcnemar_c_on_retained"] == 0
    assert strict["mcnemar_p_on_retained"] is None      # undefined, not "n.s."
    assert strict["acc_delta_on_retained"] == 0.0

    # The LENIENT policy is the one that can actually change an answer it keeps: it
    # promotes the swap's side when the base order abstained.
    lenient = rows["reconciled_lenient"]
    assert lenient["n_retained"] == 3
    assert lenient["n_retained_verdict_differs_from_base"] == 1
    assert lenient["acc_delta_on_retained"] > 0


def test_an_incomplete_judge_reports_its_own_n(tmp_path: Path):
    """qwen3.5-plus has both orders parsed for 545 of 616 pairs. A judge that answered
    fewer pairs must not be reported on the full denominator."""
    results = tmp_path / "v2"
    base = [_pw("j", "p1", "A"), _pw("j", "p2", "A")]
    swapped = [_pw("j", "p1", "B", bias="position", side="swap")]  # p2 never swapped
    _write_pairwise(results, "j", base, swapped)
    rows = build_swap_average(results, decisive_prefs={"p1": "a", "p2": "a"}, roster=None)
    assert all(r["n_pairs_judged"] == 1 for r in rows)


# --------------------------------------------------------------------------- #
# M2 — ensemble median                                                        #
# --------------------------------------------------------------------------- #
def _write_roster(root: Path, per_judge_bias_scores: dict, sids=("s0", "s1", "s2")) -> None:
    for model in ROSTER:
        originals = [_score(model, sid, 15) for sid in sids]
        biased = []
        for bias, score in per_judge_bias_scores[model].items():
            for sid in sids:
                biased.append(_score(model, sid, score, bias,
                                     biased_id=None if bias in ("bandwagon", "model_name")
                                     else ...))
        io.write_jsonl(root / "raw_judgments" / f"scoring__{model}.jsonl", originals)
        io.write_jsonl(root / "biased_judgments" / f"scoring__{model}.jsonl", biased)


def test_ensemble_join_keeps_the_prompt_level_biases(tmp_path: Path):
    """`bandwagon` and `model_name` never produced an image, so their rows carry
    `biased_id=None`. Joining the five judges on `biased_id` drops both conditions
    silently — and bandwagon is the only cue in the study that INFLATES, so losing it
    would remove the one direction that contradicts the deflation story."""
    results = tmp_path / "v2"
    _write_roster(results, {m: {"padding": 12, "bandwagon": 17} for m in ROSTER})
    rows = build_ensemble_median(results)
    ens = {r["bias_type"] for r in rows if r["is_ensemble"]}
    assert ens == {"padding", "bandwagon"}


def test_ensemble_is_defined_only_where_every_member_answered(tmp_path: Path):
    """A median over whichever judges happened to parse would change panel composition
    cell by cell, so 'the ensemble' would not be one instrument."""
    results = tmp_path / "v2"
    _write_roster(results, {m: {"padding": 12} for m in ROSTER})
    # Drop one sample from one member's biased file.
    path = results / "biased_judgments" / f"scoring__{ROSTER[0]}.jsonl"
    kept = [r for r in io.read_jsonl(path, JudgeResult) if r.sample_id != "s2"]
    io.write_jsonl(path, kept)

    rows = build_ensemble_median(results)
    ens = next(r for r in rows if r["is_ensemble"])
    assert ens["n"] == 2  # s2 has no ensemble value at all
    # ...and every member is scored on that same pair of samples, so the comparison
    # against the ensemble is not an artefact of different sample sets.
    assert {r["n"] for r in rows if r["bias_type"] == "padding"} == {2}


def test_ensemble_median_cannot_cancel_a_bias_every_member_shares(tmp_path: Path):
    """The headline negative result, in miniature: four members deflate by 3 and the
    fifth is robust. The median follows the majority and DISCARDS the robust member as
    an outlier, so the ensemble is no better than the members — measured on the real
    grid it is worse for 11 of 13 cues (zoom_inset -3.50 against a member mean of
    -2.72, because gpt-4o-viescore's +0.11 is exactly what a median throws away)."""
    results = tmp_path / "v2"
    scores = {m: {"padding": 12} for m in ROSTER}      # 15 -> 12, a -3 shift
    scores[ROSTER[0]] = {"padding": 15}                # one robust member, shift 0
    _write_roster(results, scores)

    rows = {(r["arm"], r["bias_type"]): r for r in build_ensemble_median(results)}
    assert rows[(ROSTER[0], "padding")]["mean_shift"] == 0.0
    ens = rows[(ENSEMBLE, "padding")]
    assert ens["mean_shift"] == -3.0                   # the majority, undiluted
    assert ens["abs_reduction"] < 0                    # WORSE than the member average


def test_only_the_ensemble_rows_are_this_tables_hypotheses(tmp_path: Path):
    """DEFECT: all 78 rows went into one BH family, but 65 of them are claim A for the
    five panel judges recomputed on the ensemble's common subset — the SAME hypotheses
    `claim_a.csv` already corrects. That gave "gpt-5.5 x aesthetic_filter != 0"
    q=0.059068 here and q=0.062039 there: one hypothesis, two q-values, and a reader
    free to quote whichever is smaller. Only the ensemble rows may carry a q now."""
    results = tmp_path / "v2"
    _write_roster(results, {m: {"padding": 12, "bandwagon": 17} for m in ROSTER})
    rows = build_ensemble_median(results)

    members = [r for r in rows if not r["is_ensemble"]]
    ens = [r for r in rows if r["is_ensemble"]]
    assert len(members) == len(ROSTER) * 2 and len(ens) == 2

    for r in members:
        assert r["family"] == MEMBER_FAMILY == "reference (claim_a.csv)"
        assert r["q_value"] is None
        assert r["significant_bh"] is None
        assert r["p_value"] is not None      # the p stays; only the q is withheld
    for r in ens:
        assert r["family"] == "mitigation_ensemble"

    # And the correction is over the ensemble rows ALONE: with m=2 here, BH on a single
    # tiny p gives q = p * 2 / 1 for the larger of the two, never p * 78 / rank.
    ps = sorted(r["p_value"] for r in ens)
    qs = sorted(r["q_value"] for r in ens)
    assert qs[-1] == round(min(1.0, ps[-1] * len(ens) / len(ens)), 6)


def test_build_all_writes_both_mitigation_tables(tmp_path: Path):
    results = tmp_path / "v2"
    _write_roster(results, {m: {"padding": 12} for m in ROSTER})
    _write_pairwise(results, "j", [_pw("j", "p1", "A")],
                    [_pw("j", "p1", "B", bias="position", side="swap")])
    samples = tmp_path / "samples.jsonl"
    io.write_jsonl(samples, [])

    tables = build_all(results, samples_path=samples)
    for name in ("mitigation_swap_average", "mitigation_ensemble"):
        path = results / "metrics" / f"{name}.csv"
        assert path.exists(), name
        assert name in tables
    rows = list(csv.DictReader((results / "metrics" / "mitigation_ensemble.csv")
                               .open(encoding="utf-8")))
    assert any(r["arm"] == ENSEMBLE for r in rows)


# --------------------------------------------------------------------------- #
# The roster fence — added 2026-08-18 after this builder was found to be the one
# door the 2026-08-17 audit left open.  It imported `_discover_models`, which made
# it LOOK guarded; but coverage answers "is this judge finished?", and WP-A5's
# `qwen3-vl-32b-instruct` answered yes at 578 of 616 base-order pairs while its arm
# was still running.  It joined the table, took the BH family from 20 rows to 25,
# and moved five published q-values without changing a single conclusion — which is
# precisely why nothing downstream would ever have caught it.
# --------------------------------------------------------------------------- #
def test_a_judge_outside_the_published_roster_cannot_enter_the_published_table(tmp_path: Path):
    results = tmp_path / "v2"
    prefs = {"p1": "a", "p2": "a"}
    for model in ("gpt-5.5", "gemini-3.5-flash", "gpt-4o-viescore",
                  "qwen3.5-plus", "kimi-k2.5", "qwen3-vl-32b-instruct"):
        _write_pairwise(
            results, model,
            [_pw(model, "p1", "A"), _pw(model, "p2", "A")],
            [_pw(model, "p1", "B", bias="position", side="swap"),
             _pw(model, "p2", "B", bias="position", side="swap")],
        )

    judges = {r["judge_model"] for r in build_swap_average(results, decisive_prefs=prefs)}
    assert "qwen3-vl-32b-instruct" not in judges, (
        "a complete off-roster judge still changes every q-value in this BH family"
    )
    assert len(judges) == 5


def test_the_new_roster_gets_its_own_file_rather_than_overwriting_the_published_one(tmp_path: Path):
    """A second roster written over `mitigation_swap_average.csv` would BE the corruption
    the roster exists to prevent, so it is the prefix that is pinned, not just the rows."""
    from edit_judge_bias.experiments.build_claim_tables import A5_ROSTER
    from edit_judge_bias.experiments.build_mitigation_tables import build_all

    results = tmp_path / "v2"
    for model in ("gpt-5.5", "qwen3-vl-32b-instruct"):
        _write_pairwise(results, model, [_pw(model, "p1", "A")],
                        [_pw(model, "p1", "B", bias="position", side="swap")])
    samples = tmp_path / "samples.jsonl"
    samples.write_text("", encoding="utf-8")

    build_all(results, samples_path=samples, roster=A5_ROSTER, prefix="a5_")
    assert (results / "metrics" / "a5_mitigation_swap_average.csv").exists()
    assert not (results / "metrics" / "mitigation_swap_average.csv").exists()


def test_a_two_member_roster_gets_no_ensemble_row_because_a_median_needs_a_panel(tmp_path: Path, capsys):
    """Over two members the median IS the mean — a different statistic under the same
    column name, which is the kind of substitution no reader can see."""
    from edit_judge_bias.experiments.build_claim_tables import A5_ROSTER
    from edit_judge_bias.experiments.build_mitigation_tables import build_all

    results = tmp_path / "v2"
    _write_pairwise(results, "gpt-5.5", [_pw("gpt-5.5", "p1", "A")],
                    [_pw("gpt-5.5", "p1", "B", bias="position", side="swap")])
    samples = tmp_path / "samples.jsonl"
    samples.write_text("", encoding="utf-8")

    tables = build_all(results, samples_path=samples, roster=A5_ROSTER, prefix="a5_")
    assert "mitigation_ensemble" not in tables
    assert "a median needs >=3 members" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# M2 — the same ensemble after each member is standardised (WP-F1a)           #
# --------------------------------------------------------------------------- #
def _baseline_scores(model: str):
    """The originals `_write_offset_roster` writes for one member."""
    base = dict(zip(ROSTER, (12, 13, 14, 15, 25)))[model]
    return [base + (j % 3) for j in range(8)]


def _write_offset_roster(root: Path) -> None:
    """A panel where one member simply scores HIGHER, and is also the robust one.

    This is the real panel in miniature: on the breadth block four baselines sit at
    13.4-15.6 `fine_score` and `gpt-4o-viescore` sits at 19.76. Here the four sit at
    12/13/14/15 and the fifth at 25, and only the fifth resists the cue.
    """
    sids = [f"s{i}" for i in range(8)]
    offsets = {m: base for m, base in zip(ROSTER, (12, 13, 14, 15, 25))}
    robust = ROSTER[4]
    for i, model in enumerate(ROSTER):
        base = offsets[model]
        # A little spread, so the baseline sd is not degenerate.
        originals = [_score(model, sid, base + (j % 3)) for j, sid in enumerate(sids)]
        biased = [
            _score(model, sid, (base + (j % 3)) - (0 if model == robust else 3),
                   "padding")
            for j, sid in enumerate(sids)
        ]
        io.write_jsonl(root / "raw_judgments" / f"scoring__{model}.jsonl", originals)
        io.write_jsonl(root / "biased_judgments" / f"scoring__{model}.jsonl", biased)


def test_the_ensembles_11_of_13_is_partly_a_scale_artefact():
    """WP-F1a, pinned on the frozen table because the mechanism only exists there.

    The published reading is "the median is WORSE than the member average on 11 of 13
    cues, because it discards the most robust member as an outlier". Half of that is an
    artefact of the members not being on one scale: `gpt-4o-viescore` baselines ~4-6
    `fine_score` points above the other four, so it is near-permanently the panel
    MAXIMUM and a median of five raw scores can near-permanently not select it -- for
    reasons that have nothing to do with any cue. Standardise each member by its own
    baseline first and 11/13 becomes 2/13, with both survivors differing in the third
    decimal.

    ⚠️ The raw rows are still the honest answer to "what does a deployer who medians
    five raw scores get?". What moves is the MECHANISM sentence, and the abstract's
    third point with it.
    """
    path = (Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"
            / "mitigation_ensemble.csv")
    if not path.exists():
        pytest.skip("mitigation_ensemble.csv not built")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    ens = {(r["normalisation"], r["bias_type"]): r for r in rows
           if r["is_ensemble"] == "True"}
    worse = {n: sorted(cue for (nn, cue) in ens if nn == n
                       and float(ens[(nn, cue)]["abs_reduction"]) < 0)
             for n in (RAW, Z)}
    assert len(worse[RAW]) == 11, worse[RAW]
    assert worse[Z] == ["aesthetic_filter", "saturation"], worse[Z]
    # ...and the two survivors are ties on any reading a paper could quote.
    for cue in worse[Z]:
        assert abs(float(ens[(Z, cue)]["abs_reduction"])) < 0.02

    # The single sentence that has to change: raw AMPLIFIES the placebo's drift, z
    # ATTENUATES it. Same data, and the published direction is the scale-mixed one.
    for norm, amplified in ((RAW, True), (Z, False)):
        r = ens[(norm, "sham")]
        bigger = abs(float(r["mean_shift"])) > abs(float(r["mean_member_shift"]))
        assert bigger is amplified, (norm, r["mean_shift"], r["mean_member_shift"])

    # And the offset that causes it, so the explanation cannot drift from the data:
    # viescore's baseline is the panel's highest by a clear margin.
    base = {r["arm"]: float(r["mean_original"]) for r in rows
            if r["normalisation"] == RAW and r["bias_type"] == "sham"
            and r["is_ensemble"] == "False"}
    top = max(base, key=base.get)
    assert top == "gpt-4o-viescore"
    assert base[top] - max(v for k, v in base.items() if k != top) > 3.0


def test_a_members_z_row_is_its_raw_row_rescaled_and_not_a_second_test(tmp_path: Path):
    """DEFECT, caught while building this block: computing a member's z shift as
    `(b-mu)/sd - (o-mu)/sd` instead of rescaling `(b-o)` gives a DIFFERENT p-value.

    The raw shifts are integers with hundreds of exact ties; routing each through two
    roundings splits those ties apart and the Wilcoxon tie correction then reads a
    different sample. On the real grid 33 of 65 member rows moved (gemini x
    aesthetic_filter p 0.349007 -> 0.263429) under a transformation that cannot move a
    rank test at all. A member's z row must therefore be the SAME hypothesis in SD
    units: same n, same SIR, same p, and a mean_shift divided by that member's sd.
    """
    results = tmp_path / "v2"
    _write_offset_roster(results)
    rows = {(r["normalisation"], r["arm"], r["bias_type"]): r
            for r in build_ensemble_median(results)}
    for m in ROSTER:
        raw, z = rows[(RAW, m, "padding")], rows[(Z, m, "padding")]
        assert z["n"] == raw["n"]
        assert z["sir"] == raw["sir"]
        assert z["p_value"] == raw["p_value"]
        assert z["family"] == raw["family"] == MEMBER_FAMILY
        # exactly the raw shift divided by that member's own baseline sd
        sd = statistics.stdev(_baseline_scores(m))
        assert z["mean_shift"] == round(raw["mean_shift"] / sd, 4)


def test_the_reduction_column_never_compares_sd_units_against_score_points(
    tmp_path: Path,
):
    """`abs_reduction` is `|member mean| - |ensemble|`. Keyed by cue alone it would
    have averaged the raw members and the z members into one baseline and divided SD
    units by score points -- the same defect class as one `mean_shift` column carrying
    two score scales."""
    results = tmp_path / "v2"
    _write_offset_roster(results)
    rows = [r for r in build_ensemble_median(results) if r["is_ensemble"]]
    for r in rows:
        members = [x["mean_shift"] for x in build_ensemble_median(results)
                   if not x["is_ensemble"]
                   and x["bias_type"] == r["bias_type"]
                   and x["normalisation"] == r["normalisation"]]
        assert r["mean_member_shift"] == round(sum(members) / len(members), 4)


def test_the_two_normalisations_are_two_bh_families(tmp_path: Path):
    """The raw and the z ensemble ask the same 13 questions twice. One family over
    both would correct 26 p-values for 13 hypotheses and hand a reader two q-values
    per hypothesis to pick from -- the exact defect that split the member rows out."""
    results = tmp_path / "v2"
    _write_offset_roster(results)
    rows = build_ensemble_median(results)
    fams = {r["family"] for r in rows if r["is_ensemble"]}
    assert fams == {"mitigation_ensemble", "mitigation_ensemble_z"}
    for r in rows:
        if r["is_ensemble"]:
            assert r["family"].endswith("_z") == (r["normalisation"] == Z)


def test_a_degenerate_baseline_drops_the_z_block_instead_of_dividing_by_zero(
    tmp_path: Path, capsys
):
    """A member whose baseline has sd = 0 cannot be standardised. That is a real state
    (a tiny roster, a fixture), not an error, and the raw block still stands on its
    own -- so the z rows are withheld and the reason is printed, never a silent NaN."""
    results = tmp_path / "v2"
    _write_roster(results, {m: {"padding": 12} for m in ROSTER})   # every original = 15
    rows = build_ensemble_median(results)
    assert {r["normalisation"] for r in rows} == {RAW}
    assert "degenerate baseline" in capsys.readouterr().out
