"""WP-A5's frozen tables — the open-weights replication judge.

Two things get pinned here and they are different in kind.

**The isolation property.**  `qwen3-vl-32b-instruct` is a SIXTH judge that is
deliberately not in the published BH family (user decision, 2026-08-18), so the
published tables must be unable to see it.  That is not a nice-to-have: adding a
judge to a declared family rewrites every q-value in it, which is exactly what
`build_mitigation_tables` did on 2026-08-18 before it was routed through the
roster.  These tests assert the separation from both sides.

**The replication itself.**  A5 exists to answer "does claim A hold on a judge
that was never part of the design?".  It does, and one cell of the answer is
worth a test of its own: this judge has the LOWEST self-noise in the study and
the second-worst position robustness, which is the cleanest available refutation
of "the position effect is just noise" -- measured against the instrument's own
noise rather than against another judge.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"
A5_JUDGE = "qwen3-vl-32b-instruct"


def _rows(name: str):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not built")
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("table", [
    "claim_a.csv", "claim_b.csv", "retest.csv", "position.csv",
    "pairwise_consistency.csv", "pairwise_one_sided.csv",
    "mitigation_swap_average.csv", "quality_combined.csv",
])
def test_the_a5_judge_is_invisible_to_every_published_table(table: str):
    """Its presence anywhere here means a published BH family silently grew."""
    rows = _rows(table)
    judges = {r.get("judge_model") or r.get("validator_model") or "" for r in rows}
    assert A5_JUDGE not in judges, (
        f"{table} admitted the A5 judge; every q-value in its family is now different "
        "from the one in the paper, and no conclusion has to change for that to matter"
    )


def test_the_a5_tables_contain_only_the_a5_judge():
    for name in ("a5_claim_a.csv", "a5_claim_b.csv", "a5_position.csv",
                 "a5_retest.csv", "a5_mitigation_swap_average.csv"):
        rows = _rows(name)
        assert {r["judge_model"] for r in rows} == {A5_JUDGE}, name


# --------------------------------------------------------------------------- #
# the replication
# --------------------------------------------------------------------------- #
def test_the_five_deflating_headline_cues_replicate_on_an_outside_judge():
    """Claim A's headline is that these five deflate on every judge tested."""
    rows = {r["bias_type"]: r for r in _rows("a5_claim_a.csv")}
    for cue in ("distraction", "region_annotation", "padding",
                "text_overlay", "brightness"):
        assert float(rows[cue]["mean_shift"]) < 0, cue
        assert float(rows[cue]["p_value"]) < 0.05, cue


def test_bandwagon_is_still_the_cue_that_inflates():
    rows = {r["bias_type"]: r for r in _rows("a5_claim_a.csv")}
    shift = float(rows["bandwagon"]["mean_shift"])
    assert shift > 0 and float(rows["bandwagon"]["p_value"]) < 0.05
    # inside the published +0.87..+1.58 band, i.e. a replication rather than a new size
    assert 0.87 <= shift <= 1.58


def test_the_placebo_is_the_cleanest_in_the_study_which_is_why_two_small_cues_clear_it():
    """`saturation` and `aesthetic_filter` are significant HERE and mostly not on the
    published panel.  The mechanism is not "this judge is more foolable" -- its sham
    drift is the smallest measured (+0.06 against gpt-5.5's +0.30 and gemini's +0.59),
    so its equivalence bound is tighter and smaller effects become visible.  A better
    instrument, not a larger bias; the two must be reported together."""
    rows = {r["bias_type"]: r for r in _rows("a5_claim_a.csv")}
    assert abs(float(rows["sham"]["mean_shift"])) < 0.1
    assert float(rows["sham"]["p_value"]) > 0.05
    for cue in ("saturation", "aesthetic_filter"):
        assert abs(float(rows[cue]["mean_shift"])) < 0.25


def test_claim_b_reproduces_the_headline_cue_and_only_it():
    """Published claim B: `region_annotation` is the one cue that reorders, and it
    shows up on the strong anchor.  One cell out of eight is a much weaker result
    than the published 8-of-10, so "the headline cue replicates" is writable and
    "claim B replicates" is not."""
    # The main grid's family only: since 2026-09-15 the file also carries the FILL v2 rows,
    # corrected in their own family (`claim_B_acc [fill] [a5 roster]`).
    rows = [r for r in _rows("a5_claim_b.csv") if r["family"] == "claim_B_acc [a5 roster]"]
    assert len(rows) == 8
    sig = [r for r in rows if r["significant_bh"] == "True"]
    assert len(sig) == 1
    assert sig[0]["bias_type"] == "region_annotation"
    assert int(sig[0]["n_items"]) == 384          # EBench-18K, the strong anchor
    assert float(sig[0]["accuracy_delta"]) < 0


def test_the_slot_flips_this_judge_38x_more_often_than_its_own_noise_does():
    """★ The strongest form of "the position effect is not noise" in the project.

    Both numbers are strict flip rates on the SAME judge and the same 616 pairs:
    one when only the display order changed, one when nothing changed and the
    question was simply asked again.  No cross-judge comparison and no modelling
    assumption stands between them.
    """
    pos = _rows("a5_position.csv")[0]
    cr = _rows("a5_pairwise_consistency.csv")[0]
    swap_flip = float(pos["strict_flip_rate"])
    self_flip = float(cr["strict_flip_rate"])
    assert self_flip < 0.02 < 0.4 < swap_flip
    assert swap_flip / self_flip > 30


def test_the_most_self_consistent_judge_is_not_the_most_invariant_one():
    """§7.2's fourth axis.  Self-consistency and invariance come apart, and this
    judge is the demonstration: best-in-study on retest, second-worst on RR.

    ⚠️ Deliberately no correlation coefficient.  Over six judges the rank
    correlation is -0.886, sitting exactly on the n=6 critical value, and this
    project has already been burned by quoting a rho at n=5 that swung from
    -0.700 to +0.800 under an equally reasonable operationalisation (section 7.2).
    "Not the same thing" is robust; "anti-correlated" is not writable.
    """
    a5_retest = _rows("a5_retest.csv")[0]
    published = _rows("retest.csv")
    assert float(a5_retest["identical_rate"]) > max(
        float(r["identical_rate"]) for r in published
    ), "A5 is meant to be the most self-consistent judge measured"

    a5_rr = float(_rows("a5_position.csv")[0]["rr"])
    worse = [r for r in _rows("position.csv") if float(r["rr"]) < a5_rr]
    assert len(worse) == 1, "and simultaneously second-worst on display order"


def test_protocol_reconciliation_is_still_abstention_not_repair():
    """The paper's closing argument, on a sixth judge: strict reconciliation
    changes zero verdicts and buys its precision entirely with coverage."""
    rows = {r["arm"]: r for r in _rows("a5_mitigation_swap_average.csv")}
    strict = rows["reconciled_strict"]
    assert float(strict["acc_delta_on_retained"]) == 0.0
    assert int(strict["n_retained_verdict_differs_from_base"]) == 0
    assert float(strict["coverage"]) < float(rows["base_order_only"]["coverage"]) - 0.5


def test_the_a5_family_stamp_names_its_roster():
    """The `family` column exists so a reader can tell WHICH multiple-comparison
    correction a q belongs to.  Until 2026-08-19 both `claim_a.csv` and `a5_claim_a.csv`
    stamped the bare string `claim_A` -- two INDEPENDENT corrections wearing one label.
    Separate files make that harmless right up until somebody concatenates them, which is
    precisely the operation the a5 arm's declared-independence rule forbids and precisely
    the operation a bare shared label invites.

    Both directions are pinned: the a5 rows say which roster they are corrected within,
    and the published rows are NOT retagged (that would rewrite 65 frozen cells to fix a
    problem the published family does not have)."""
    import csv
    from pathlib import Path

    metrics = Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"
    a5 = metrics / "a5_claim_a.csv"
    pub = metrics / "claim_a.csv"
    if not (a5.exists() and pub.exists()):
        pytest.skip("claim A tables not built")

    def families(path):
        with path.open(encoding="utf-8-sig", newline="") as fh:
            return {r["family"] for r in csv.DictReader(fh)}

    assert all("[a5 roster]" in f for f in families(a5)), families(a5)
    assert not any("roster" in f for f in families(pub)), families(pub)
