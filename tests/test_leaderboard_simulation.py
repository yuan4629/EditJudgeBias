"""Guards for the WP-A1e editor-leaderboard simulation.

The table turns a score shift into the quantity the field publishes — a ranking over
editors. Four things can go wrong, and each one produces a leaderboard that looks
plausible:

1. **the turn confound comes back.** EBench-18K runs 17 editors over 48 turns with 8
   per turn, so a raw per-editor mean measures which turns an editor drew as much as
   how good it is. Centring inside the turn is the correction, and it must be a no-op
   on the complete ImagenHub design or it would be doing something else as well.
2. **a missing biased score silently falls back to the clean one.** That understates
   every effect and would be invisible in the output.
3. **a uniform shift registers as a reordering.** If every editor loses the same amount
   the board must not move; a test that cannot detect a no-op cannot be trusted on a
   real one.
4. **the sign of `mean_rank_drop` flips.** Rank 1 is the best editor, so falling means
   the number goes UP, and getting that backwards inverts the whole scenario.
"""

# ⚠️ The fixture calls below pass `roster=None` on purpose.  The default is
# `PUBLISHED_ROSTER` -- the five judges whose numbers are in the paper -- so that a
# forgotten argument silently OMITS a new judge (and says so) rather than silently
# ADMITTING one into a published table.  These tests use synthetic judge names, so they
# are about the mechanics and must opt out of family membership explicitly.


from __future__ import annotations

from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_leaderboard_simulation import (
    Board,
    build_boards,
    build_leaderboard_rows,
    kendall_tau,
    ranks_from_scores,
    turn_centred_means,
    _swap_scores,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "v2"

on_disk = pytest.mark.skipif(
    not (RESULTS / "raw_judgments").exists(), reason="v2 judgments not on disk"
)


# --------------------------------------------------------------------------- #
# a synthetic anchor block                                                     #
# --------------------------------------------------------------------------- #
def _sample(sid, editor, turn, source="src", human=None):
    rec = SampleRecord(
        sample_id=sid, source_dataset=source, edit_type="add", content_category="object",
        original_image_path=f"{turn}.jpg", instruction=f"do {turn}", edit_model=editor,
        edited_image_path=f"{sid}_e.jpg", human_score=human,
    )
    rec.metadata.anchor_source = source  # type: ignore[attr-defined]
    return rec


def _result(model, sid, fine, bias=None):
    return JudgeResult(
        result_id=f"score::{model}::vanilla::{sid}::{bias}", judge_model=model,
        task_type="scoring", prompt_type="vanilla_scoring", raw_response_path="r.txt",
        parse_success=True, sample_id=sid, bias_type=bias,
        biased_id=f"{sid}__{bias}" if bias else None,
        overall_score=max(1, min(10, round(fine / 3))), fine_score=int(fine),
        score_scale=10,
    )


def _tree(tmp_path, *, deflation, n_turns=6, n_editors=4, quality=(24, 20, 16, 12)):
    """`n_editors` editors of fixed quality over `n_turns` turns of varying difficulty.

    `deflation` maps editor -> points removed by the cue, so a caller can build a
    perfectly uniform shift (no reordering) or a targeted one.
    """
    results = tmp_path / "v2"
    samples, originals, biased = [], [], []
    for t in range(n_turns):
        difficulty = 3 * t          # a turn effect big enough to swamp the quality gaps
        for e in range(n_editors):
            editor = f"ed{e}"
            sid = f"t{t}_{editor}"
            samples.append(_sample(sid, editor, f"turn{t}", human=quality[e] / 30))
            originals.append(_result("j1", sid, quality[e] + difficulty))
            biased.append(_result("j1", sid, quality[e] + difficulty - deflation[editor],
                                  "padding"))
    io.write_jsonl(results / "raw_judgments" / "scoring__j1.jsonl", originals)
    io.write_jsonl(results / "biased_judgments" / "scoring__j1.jsonl", biased)
    return results, samples


# --------------------------------------------------------------------------- #
# turn centring                                                                #
# --------------------------------------------------------------------------- #
def test_turn_centring_removes_the_difficulty_confound():
    """Two editors, disjoint turns of different difficulty; centring must decide.

    A raw mean says the editor that drew the easy turn is better. This is not a
    hypothetical shape — it is exactly the EBench-18K anchor, 8 of 17 editors per turn.
    """
    turns = {"easy": {"good": "a", "weak": "b"}, "hard": {"good": "c", "weak": "d"}}
    scores = {"a": 28.0, "b": 26.0, "c": 14.0, "d": 12.0}
    centred = turn_centred_means(turns, scores)
    assert centred["good"] > centred["weak"]
    # and on a turn where only one editor has a score, that turn contributes nothing
    partial = turn_centred_means({"solo": {"good": "a"}}, {"a": 28.0})
    assert partial == {}


def test_centring_cannot_change_the_ranking_on_a_complete_design(tmp_path):
    results, samples = _tree(tmp_path, deflation={f"ed{e}": 0 for e in range(4)})
    board = build_boards(samples, results, roster=None)[0]
    raw: dict[str, list[float]] = {}
    for members in board.turns.values():
        for editor, sid in members.items():
            raw.setdefault(editor, []).append(board.clean[sid])
    raw_ranks = ranks_from_scores({e: sum(v) / len(v) for e, v in raw.items()})
    assert raw_ranks == ranks_from_scores(turn_centred_means(board.turns, board.clean))


def test_rank_one_is_the_best_editor():
    ranks = ranks_from_scores({"best": 10.0, "mid": 5.0, "worst": 1.0})
    assert ranks == {"best": 1.0, "mid": 2.0, "worst": 3.0}


# --------------------------------------------------------------------------- #
# the swap                                                                     #
# --------------------------------------------------------------------------- #
def test_a_missing_biased_score_is_dropped_never_replaced_by_the_clean_one(tmp_path):
    results, samples = _tree(tmp_path, deflation={f"ed{e}": 4 for e in range(4)})
    board = build_boards(samples, results, roster=None)[0]
    victim = sorted(board.biased["padding"])[0]
    del board.biased["padding"][victim]
    swapped = _swap_scores(board, "padding", ["ed0"])
    assert victim not in swapped or swapped[victim] != board.clean[victim]
    # every other ed0 sample really did take the biased value
    for members in board.turns.values():
        sid = members["ed0"]
        if sid in swapped and sid != victim:
            assert swapped[sid] == board.biased["padding"][sid]


def test_only_the_named_editors_are_swapped(tmp_path):
    results, samples = _tree(tmp_path, deflation={f"ed{e}": 4 for e in range(4)})
    board = build_boards(samples, results, roster=None)[0]
    swapped = _swap_scores(board, "padding", ["ed2"])
    for members in board.turns.values():
        for editor, sid in members.items():
            expected = (board.biased["padding"][sid] if editor == "ed2"
                        else board.clean[sid])
            assert swapped[sid] == expected


# --------------------------------------------------------------------------- #
# the two scenarios                                                            #
# --------------------------------------------------------------------------- #
def test_a_perfectly_uniform_cue_does_not_move_the_board(tmp_path):
    """The no-op case. If this fires, nothing else in the table can be believed."""
    results, samples = _tree(tmp_path, deflation={f"ed{e}": 5 for e in range(4)})
    rows = {r["scenario"]: r for r in build_leaderboard_rows(samples, results, n_boot=50, roster=None)}
    uniform = rows["all_editors"]
    assert uniform["kendall_tau_clean_vs_biased"] == pytest.approx(1.0)
    assert uniform["n_rank_changes"] == 0
    assert uniform["top1_changed"] is False
    assert uniform["delta_tau_human"] == pytest.approx(0.0)


def test_a_targeted_cue_makes_that_editor_fall_and_the_sign_says_so(tmp_path):
    """Rank 1 is best, so falling is a POSITIVE `mean_rank_drop`."""
    deflation = {"ed0": 9, "ed1": 0, "ed2": 0, "ed3": 0}   # only the leader is hit
    results, samples = _tree(tmp_path, deflation=deflation)
    rows = {r["scenario"]: r for r in build_leaderboard_rows(samples, results, n_boot=50, roster=None)}
    single = rows["single_editor"]
    assert single["worst_editor"] == "ed0"
    assert single["max_rank_drop"] > 0
    assert single["n_editors_falling"] >= 1
    assert single["leader_loses_top1"] is True
    # ed0 is the clean leader and a 9-point self-inflicted deduction drops it below
    # ed1 (4 points behind) and ed2 (8 behind) but not below ed3 (12 behind).
    assert single["max_rank_drop"] == pytest.approx(2.0)


def test_kendall_tau_needs_three_editors_to_mean_anything():
    assert kendall_tau({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 1.0}) is None
    assert kendall_tau({"a": 1.0, "b": 2.0, "c": 3.0},
                       {"a": 1.0, "b": 2.0, "c": 3.0}) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the frozen grid                                                              #
# --------------------------------------------------------------------------- #
@on_disk
def test_frozen_grid_covers_five_judges_two_anchors_four_cues():
    samples = io.read_jsonl(REPO / "data" / "manifests" / "samples_judge_v2.jsonl",
                            SampleRecord)
    rows = build_leaderboard_rows(samples, RESULTS, n_boot=30)
    for scenario in ("all_editors", "single_editor"):
        sub = [r for r in rows if r["scenario"] == scenario]
        assert len(sub) == 40, scenario
    ebench = [r for r in rows if r["anchor_source"] == "EBench-18K"]
    assert {r["n_editors"] for r in ebench} == {17}
    assert {r["n_turns"] for r in ebench} == {48}
    imagenhub = [r for r in rows if r["anchor_source"] == "ImagenHub"]
    assert {r["n_editors"] for r in imagenhub} == {8}
    assert {r["n_turns"] for r in imagenhub} == {30}


# --------------------------------------------------------------------------- #
# the figure                                                                   #
# --------------------------------------------------------------------------- #
def test_figure_renders_and_refuses_a_half_table(tmp_path):
    """F12 needs BOTH scenarios; one alone would draw a panel against nothing."""
    from edit_judge_bias.visualization.plot_leaderboard import plot_leaderboard_simulation

    results, samples = _tree(tmp_path, deflation={"ed0": 9, "ed1": 0, "ed2": 0, "ed3": 0})
    rows = build_leaderboard_rows(samples, results, n_boot=20, roster=None)
    out = plot_leaderboard_simulation(rows, tmp_path / "f12.png")
    assert out is not None and out.exists() and out.stat().st_size > 10_000

    only_one = [r for r in rows if r["scenario"] == "all_editors"]
    assert plot_leaderboard_simulation(only_one, tmp_path / "half.png") is None


def test_figure_counts_the_human_agreement_intervals_from_the_data(tmp_path, monkeypatch):
    """The footnote's "all 40 cover zero" must be counted, not remembered."""
    from edit_judge_bias.visualization import plot_leaderboard

    captured: list[str] = []
    real_text = None

    results, samples = _tree(tmp_path, deflation={"ed0": 9, "ed1": 0, "ed2": 0, "ed3": 0})
    rows = build_leaderboard_rows(samples, results, n_boot=20, roster=None)
    for r in rows:
        if r["scenario"] == "all_editors":
            r["delta_tau_ci_excludes_zero"] = True

    import matplotlib.figure

    real_text = matplotlib.figure.Figure.text

    def spy(self, *args, **kwargs):
        if args and isinstance(args[-1], str):
            captured.append(args[-1])
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "text", spy)
    plot_leaderboard.plot_leaderboard_simulation(rows, tmp_path / "f12b.png")
    footnote = "\n".join(captured)
    assert "0 of 1 cluster-bootstrap intervals cover zero" in footnote


@on_disk
def test_an_editor_that_captions_its_own_outputs_loses_the_top_spot():
    """A1e's reason for existing, as a regression test.

    On the 17-editor EBench board, `text_overlay` applied to one editor alone costs the
    leader first place on every judge — with no change in edit quality. If that ever
    stops being true the deployment paragraph has to be rewritten.
    """
    samples = io.read_jsonl(REPO / "data" / "manifests" / "samples_judge_v2.jsonl",
                            SampleRecord)
    rows = [r for r in build_leaderboard_rows(samples, RESULTS, n_boot=30)
            if r["scenario"] == "single_editor" and r["anchor_source"] == "EBench-18K"
            and r["bias_type"] == "text_overlay"]
    assert len(rows) == 5
    assert all(r["leader_loses_top1"] for r in rows)
    assert max(r["max_rank_drop"] for r in rows) >= 5
