"""Tests for pairwise position / stickiness / one-sided metrics (Milestone 5)."""

from __future__ import annotations

from edit_judge_bias.data.schema import JudgeResult, PairRecord, SampleRecord
from edit_judge_bias.metrics.pairwise_metrics import (
    compute_consistency,
    compute_one_sided_bias,
    compute_position_flips,
    decisive_human_prefs,
    position_joint,
)


def _pw(model, pair_id, winner, *, position=False):
    return JudgeResult(
        result_id=f"{model}::{pair_id}::{'swap' if position else 'orig'}",
        judge_model=model, task_type="pairwise", prompt_type="vanilla_pairwise",
        raw_response_path="r.txt", parse_success=True, pair_id=pair_id,
        winner=winner, bias_type=("position" if position else None),
        biased_side=("swap" if position else None),
    )


def test_content_fair_judge_no_flip():
    # Content-fair: when slots swap, the displayed winner flips too -> same content.
    orig = [_pw("m", "p0", "A"), _pw("m", "p1", "B")]
    swap = [_pw("m", "p0", "B", position=True), _pw("m", "p1", "A", position=True)]
    [st] = compute_position_flips(orig, swap)
    assert st.n == 2
    assert st.rr == 1.0
    assert st.strict_flip_rate == 0.0


def test_position_biased_judge_all_flip():
    # Judge always picks slot A regardless of content -> content winner flips.
    orig = [_pw("m", "p0", "A"), _pw("m", "p1", "B")]
    swap = [_pw("m", "p0", "A", position=True), _pw("m", "p1", "B", position=True)]
    [st] = compute_position_flips(orig, swap)
    assert st.rr == 0.0
    assert st.strict_flip_rate == 1.0
    assert st.posA_rate_original == 0.5
    assert st.posA_rate_swapped == 0.5  # p0->A, p1->B
    assert st.p_value is not None


def test_only_shared_pairs_counted():
    orig = [_pw("m", "p0", "A"), _pw("m", "p1", "A")]
    swap = [_pw("m", "p0", "B", position=True)]  # p1 has no swap verdict
    [st] = compute_position_flips(orig, swap)
    assert st.n == 1


def test_tie_counts_as_soft_not_strict():
    orig = [_pw("m", "p0", "A")]
    swap = [_pw("m", "p0", "Tie", position=True)]
    [st] = compute_position_flips(orig, swap)
    assert st.soft_flip_rate == 1.0
    assert st.strict_flip_rate == 0.0


def _one_sided(model, pair_id, winner, bias, side="a"):
    """An arm-④ one-sided row: same pair_id, a bias on ONE member, no display swap."""
    return JudgeResult(
        result_id=f"{model}::{pair_id}::{bias}::{side}",
        judge_model=model, task_type="pairwise", prompt_type="vanilla_pairwise",
        raw_response_path="r.txt", parse_success=True, pair_id=pair_id,
        winner=winner, bias_type=bias, biased_side=side,
    )


def test_one_sided_rows_do_not_masquerade_as_the_position_swap():
    """MEASURED on gpt-5.5's 3,080 arm-④ rows: the biased pairwise manifest holds four
    conditions sharing one pair_id, so an unfiltered last-write-wins join handed the
    position metric a one-sided row — which is NOT display-swapped — and the content
    mapping inverted it. RR came out 0.151 (below the ~0.48 you would get by chance)
    for a judge that is in fact position-robust."""
    orig = [_pw("m", f"p{i}", "A") for i in range(4)]
    swap = [_pw("m", f"p{i}", "B", position=True) for i in range(4)]  # content-fair
    # written AFTER the position rows, as the runner does
    noise = [_one_sided("m", f"p{i}", "A", "padding") for i in range(4)]
    [st] = compute_position_flips(orig, swap + noise)
    assert st.n == 4
    assert st.rr == 1.0            # the padding rows must not be read as the swap
    assert st.strict_flip_rate == 0.0


def test_baselines_are_not_read_from_a_biased_row():
    orig = [_pw("m", "p0", "A"), _one_sided("m", "p0", "B", "text_overlay")]
    swap = [_pw("m", "p0", "B", position=True)]
    [st] = compute_position_flips(orig, swap)
    assert st.n == 1 and st.rr == 1.0


def test_a_pairwise_retest_repeat_is_not_read_as_the_baseline():
    """The CR arm re-asks the BASELINE with the cache off, so its rows carry
    bias_type=None and the same pair_id, land in the same manifest, and are written
    later. Without the repeat filter the swap would be compared against the repeat
    answer and RR would measure retest noise instead of position robustness."""
    base = _pw("m", "p0", "A")
    repeat = base.model_copy(update={
        "result_id": "m::p0::orig::rep2", "winner": "B",
        "bias_params": {"repeat_index": 2},
    })
    swap = [_pw("m", "p0", "B", position=True)]
    [st] = compute_position_flips([base, repeat], swap)
    assert st.n == 1 and st.rr == 1.0  # A -> B across orders = content-fair


# --------------------------------------------------------------------------- #
# slot stickiness                                                             #
# --------------------------------------------------------------------------- #
def test_joint_separates_a_sticky_judge_from_a_content_fair_one():
    """RR cannot tell these apart at a glance; the displayed-letter joint can."""
    orig = [_pw("m", f"p{i}", "A") for i in range(4)]
    fair = [_pw("m", f"p{i}", "B", position=True) for i in range(4)]
    [st] = position_joint(orig, fair)
    assert st.n == 4 and st.same_slot_rate == 0.0
    assert st.counts[("A", "B")] == 4

    sticky = [_pw("m", f"p{i}", "A", position=True) for i in range(4)]
    [st] = position_joint(orig, sticky)
    assert st.same_slot_rate == 1.0
    assert st.as_row()["count_A_A"] == 4


def test_joint_row_has_all_nine_cells():
    orig = [_pw("m", "p0", "Tie")]
    swap = [_pw("m", "p0", "Tie", position=True)]
    row = position_joint(orig, swap)[0].as_row()
    assert row["count_Tie_Tie"] == 1
    assert sum(row[f"count_{a}_{b}"] for a in ("A", "B", "Tie") for b in ("A", "B", "Tie")) == 1


def test_repeated_ties_are_not_slot_stickiness():
    """MEASURED: gemini repeats itself on 17.1% of pairs but 14.9pp of that is
    Tie->Tie — consistently indecisive, not slot-bound. Pooling the two would report
    it as four times stickier than it is."""
    orig = [_pw("m", "p0", "Tie"), _pw("m", "p1", "Tie"), _pw("m", "p2", "A")]
    swap = [_pw("m", "p0", "Tie", position=True), _pw("m", "p1", "Tie", position=True),
            _pw("m", "p2", "A", position=True)]
    [st] = position_joint(orig, swap)
    assert st.same_slot_rate == 1.0     # every answer repeated
    assert st.sticky_rate == 1 / 3      # only p2 named the same SLOT twice
    # ...and that is the strict content flip, by construction. The two tables come
    # from different functions, so pinning the identity makes a join bug loud.
    [flip] = compute_position_flips(orig, swap)
    assert st.sticky_rate == flip.strict_flip_rate


# --------------------------------------------------------------------------- #
# consistency rate                                                            #
# --------------------------------------------------------------------------- #
def _repeat(rec, winner, index=2):
    return rec.model_copy(update={
        "result_id": f"{rec.result_id}::rep{index}", "winner": winner,
        "bias_params": {"repeat_index": index},
    })


def test_consistency_pairs_the_two_asks():
    rows = []
    for i, (first, second) in enumerate([("A", "A"), ("A", "B"), ("A", "Tie")]):
        base = _pw("m", f"p{i}", first)
        rows += [base, _repeat(base, second)]
    [st] = compute_consistency(rows)
    assert st.n == 3
    assert st.cr == 1 / 3
    assert st.strict_flip_rate == 1 / 3   # A -> B
    assert st.tie_churn_rate == 1 / 3     # A -> Tie


def test_consistency_ignores_pairs_asked_once_and_biased_rows():
    rows = [_pw("m", "p0", "A"),                                   # never repeated
            _pw("m", "p1", "A"), _repeat(_pw("m", "p1", "A"), "A"),
            _one_sided("m", "p1", "B", "padding", side="a")]       # a different question
    [st] = compute_consistency(rows)
    assert st.n == 1 and st.cr == 1.0


# --------------------------------------------------------------------------- #
# one-sided image bias                                                        #
# --------------------------------------------------------------------------- #
def test_one_sided_reads_the_dressed_side_per_row():
    """The runner picks the dressed slot as a seeded function of the pair id (measured
    843 'a' / 1005 'b' on the grid), so that a one-sided effect cannot be confused
    with position bias. Assuming slot A mirrors half the pairs."""
    base = [_pw("m", "p0", "A"), _pw("m", "p1", "B")]
    # Both pairs move ONTO the dressed side, but the dressed side differs.
    after = [_one_sided("m", "p0", "A", "padding", side="a"),
             _one_sided("m", "p1", "B", "padding", side="b")]
    [st] = compute_one_sided_bias(base, after)
    assert st.n == 2
    assert st.dressed_win_rate_baseline == 1.0 and st.dressed_win_rate_biased == 1.0
    assert st.bias_advantage == 0.0


def test_one_sided_bias_advantage_is_paired_and_signed():
    base = [_pw("m", f"p{i}", "A") for i in range(4)]
    # b is dressed everywhere; two pairs move onto it, two stay on A.
    after = [_one_sided("m", "p0", "B", "text_overlay", side="b"),
             _one_sided("m", "p1", "B", "text_overlay", side="b"),
             _one_sided("m", "p2", "A", "text_overlay", side="b"),
             _one_sided("m", "p3", "A", "text_overlay", side="b")]
    [st] = compute_one_sided_bias(base, after)
    assert st.dressed_win_rate_baseline == 0.0   # baseline always picked A
    assert st.dressed_win_rate_biased == 0.5
    assert st.bias_advantage == 0.5
    assert (st.mcnemar_b, st.mcnemar_c) == (0, 2)  # all movement toward the cue
    assert st.flip_toward_dressed == 2 and st.flip_away_from_dressed == 0


def test_one_sided_ignores_the_position_swap_and_other_conditions():
    """All five arm-④ conditions share a pair_id; only rows carrying an a/b side and
    this bias's own name belong to this cell."""
    base = [_pw("m", "p0", "A")]
    rows = [_pw("m", "p0", "B", position=True),               # biased_side="swap"
            _one_sided("m", "p0", "B", "padding", side="b"),
            _one_sided("m", "p0", "A", "brightness", side="a")]
    stats = {st.bias_type: st for st in compute_one_sided_bias(base, rows)}
    assert set(stats) == {"padding", "brightness"}
    assert stats["padding"].n == 1 and stats["brightness"].n == 1


def test_pooled_accuracy_is_null_when_the_cue_is_randomly_placed():
    """A cue that deflates whichever side it sits on lands on the human's pick about
    half the time and on the other edit the rest, so the POOLED accuracy delta is ~0
    however strong the cue is. Only the conditional split shows the damage.

    Here the judge picks slot a at baseline on all four pairs and the cue (always on
    a) pushes it off; the humans prefer a on two of them and b on the other two.
    """
    base, after, prefs = [], [], {}
    for i in range(4):
        prefs[f"p{i}"] = "a" if i < 2 else "b"
        base.append(_pw("m", f"p{i}", "A"))
        after.append(_one_sided("m", f"p{i}", "B", "padding", side="a"))
    [st] = compute_one_sided_bias(base, after, decisive_prefs=prefs)
    assert st.accuracy_baseline == 0.5 and st.accuracy_biased == 0.5
    assert st.accuracy_delta == 0.0            # 2 lost, 2 gained: pooled says nothing
    assert st.n_cue_on_winner == 2 and st.acc_delta_cue_on_winner == -1.0
    assert st.n_cue_on_loser == 2 and st.acc_delta_cue_on_loser == 1.0


def test_one_sided_ties_count_as_wrong_and_are_reported():
    base = [_pw("m", "p0", "A")]
    after = [_one_sided("m", "p0", "Tie", "padding", side="b")]
    [st] = compute_one_sided_bias(base, after, decisive_prefs={"p0": "a"})
    assert st.accuracy_baseline == 1.0 and st.accuracy_biased == 0.0
    assert st.tie_rate_baseline == 0.0 and st.tie_rate_biased == 1.0


# --------------------------------------------------------------------------- #
# decisive human preferences                                                  #
# --------------------------------------------------------------------------- #
def _sample(sid, source, human):
    return SampleRecord(
        sample_id=sid, source_dataset=source, edit_type="add",
        content_category="object", original_image_path=f"{sid}_o.jpg",
        instruction="do", edit_model="m", edited_image_path=f"{sid}_e.jpg",
        human_score=human,
    )


def _pair(pid, source, gap, pref):
    return PairRecord(
        pair_id=pid, sample_id_a=f"{pid}a", sample_id_b=f"{pid}b",
        original_image_path="o.jpg", instruction="do",
        edited_image_a_path="a.jpg", edited_image_b_path="b.jpg",
        edit_model_a="m1", edit_model_b="m2", source_dataset=source,
        edit_type="add", ground_truth_preference=pref, pair_quality_gap=gap,
    )


def test_decisive_must_be_computed_on_the_full_pool_not_the_subset():
    """"Decisive" is a gap in SDs of that source's OWN human scores, and the judging
    subset is a tier-ordered draw whose spread is no longer the population's.

    MEASURED on the real manifests: the full pool marks 438 of the 616 judged pairs
    decisive; recomputing the identical rule on the subset marks only 293 and drops a
    whole source. The mechanism is reproduced here in miniature — the subset kept the
    extreme samples, so its SD is larger, so its 0.5*SD bar is higher, so a pair the
    full pool calls decisive silently disappears.
    """
    # Full pool: mostly identical human scores plus two extremes -> small SD.
    full_samples = ([_sample(f"s{i}", "src", 5.0) for i in range(11)]
                    + [_sample("s_lo", "src", 0.0), _sample("s_hi", "src", 10.0)])
    pairs = [_pair("p_mid", "src", gap=2.0, pref="a")]
    # Subset: only the extremes survived the draw -> much larger SD, much higher bar.
    sub_samples = [_sample("s_lo", "src", 0.0), _sample("s_hi", "src", 10.0)]

    from_full = decisive_human_prefs(full_samples, pairs)
    from_subset = decisive_human_prefs(sub_samples, pairs)
    assert "p_mid" in from_full
    assert "p_mid" not in from_subset, (
        "this test only exercises the trap if the subset is stricter than the pool"
    )


def test_decisive_drops_ties_and_unlabelled_pairs():
    samples = [_sample(f"s{i}", "src", float(i)) for i in range(6)]
    pairs = [
        _pair("p_big", "src", gap=5.0, pref="a"),
        _pair("p_tie", "src", gap=5.0, pref="tie"),
        _pair("p_none", "src", gap=5.0, pref=None),
    ]
    assert decisive_human_prefs(samples, pairs) == {"p_big": "a"}
