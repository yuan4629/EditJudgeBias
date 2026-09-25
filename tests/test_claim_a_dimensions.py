"""WP-F1b — claim A decomposed into the three dimensions the judge was asked for.

Two kinds of test live here and they are different in kind.

**The decomposition property**, pinned on the frozen table: the three dimensions are
computed on the same complete-case subset as `claim_a.csv` and their `mean_shift`
values ADD UP to that table's, cell by cell. Without that, "instruction adherence moved
on 4 judges" would be a claim about a different sample from the headline it qualifies,
and nobody reading the paper could tell.

**The reading**, also pinned on the frozen table, because it is what the abstract and
§7 now say: a cue that is applied after the edit and that the annotators were told to
ignore cannot legitimately move *instruction adherence*, so a deflation there is the
one form of this effect that cannot be argued away as a justified deduction. Three
cues do it, `brightness` essentially does not, and `bandwagon` does the reverse.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult
from edit_judge_bias.experiments.build_claim_a_dimensions import (
    DIMENSIONS,
    build_claim_a_dimensions,
)
from edit_judge_bias.experiments.build_claim_tables import PUBLISHED_ROSTER

METRICS = Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"


def _frozen(name: str):
    path = METRICS / name
    if not path.exists():
        pytest.skip(f"{name} not built")
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def _score(model, sid, ia, eq, dp, bias=None):
    dims = (ia, eq, dp)
    return JudgeResult(
        result_id=f"{model}::{sid}::{bias}", judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt", parse_success=True,
        sample_id=sid, bias_type=bias,
        biased_id=f"{sid}__{bias}" if bias else None,
        instruction_adherence=ia, editing_quality=eq, detail_preservation=dp,
        fine_score=sum(dims) if all(d is not None for d in dims) else None,
        score_scale=10,
    )


# --------------------------------------------------------------------------- #
# the decomposition property
# --------------------------------------------------------------------------- #
def test_the_three_dimensions_add_up_to_the_published_headline():
    """The whole point of building this as a decomposition rather than as a second
    analysis: every cell is on `claim_a.csv`'s own complete-case subset, so the three
    dimension shifts sum to the published `fine_score` shift and the same `n`.

    If this ever stops holding, the dimension table has quietly acquired its own sample
    and none of the sentences it supports mean what they say."""
    dims = _frozen("claim_a_by_dimension.csv")
    head = _frozen("claim_a.csv")
    by_cell = {}
    for r in dims:
        by_cell.setdefault((r["judge_model"], r["bias_type"]), []).append(r)

    assert len(dims) == len(head) * len(DIMENSIONS)
    for h in head:
        cell = by_cell[(h["judge_model"], h["bias_type"])]
        assert len(cell) == len(DIMENSIONS)
        assert {r["n"] for r in cell} == {h["n"]}
        total = sum(float(r["mean_shift"]) for r in cell)
        # each term is rounded to 4dp, so three of them can be off by at most 1.5e-4
        assert abs(total - float(h["mean_shift"])) < 2e-4, (h["judge_model"],
                                                            h["bias_type"], total)


def test_each_dimension_is_its_own_bh_family():
    """Three dimensions are three questions. One family over all 195 rows would
    correct each dimension for the other two, and `sham` is the control in every one
    of them -- it is what the family is measured against, not a member of it."""
    rows = _frozen("claim_a_by_dimension.csv")
    for dim in DIMENSIONS:
        fam = {r["family"] for r in rows if r["dimension"] == dim
               and r["bias_type"] != "sham"}
        assert fam == {f"claim_A_{dim}"}
    assert {r["family"] for r in rows if r["bias_type"] == "sham"} == {"control"}
    assert all(r["q_value"] == "" for r in rows if r["bias_type"] == "sham")


# --------------------------------------------------------------------------- #
# the reading the paper now quotes
# --------------------------------------------------------------------------- #
def _signed(rows, cue, dim):
    sel = [r for r in rows if r["bias_type"] == cue and r["dimension"] == dim]
    down = [r for r in sel if r["significant_bh"] == "True" and float(r["mean_shift"]) < 0]
    up = [r for r in sel if r["significant_bh"] == "True" and float(r["mean_shift"]) > 0]
    return len(down), len(up), len(sel)


def test_instruction_adherence_is_where_the_effect_cannot_be_argued_away():
    """The cue is pasted on AFTER the edit and the validators and annotators were told
    to ignore it, so the correct instruction-adherence answer cannot move. It moves
    anyway, downward, on 4-5 of 5 judges for three cues.

    ⚠️ Counted with the sign, never as one "significant" count: three cues in this
    table are significant in BOTH directions across the panel, and a single count would
    be the same composite-rate defect this project already corrected once."""
    rows = _frozen("claim_a_by_dimension.csv")
    ia = "instruction_adherence"
    assert _signed(rows, "distraction", ia)[0] == 5
    assert _signed(rows, "text_overlay", ia)[0] == 4
    assert _signed(rows, "region_annotation", ia)[0] == 4
    assert _signed(rows, "padding", ia)[0] == 4
    # ...and the placebo does not move any dimension, which is what makes the rest readable
    for dim in DIMENSIONS:
        assert _signed(rows, "sham", dim)[:2] == (0, 0)


def test_brightness_must_not_be_listed_beside_the_other_four():
    """`brightness` deflates `fine_score` on 5/5 judges, but almost none of it is on
    instruction adherence -- it lands on editing quality and detail preservation, which
    is exactly what a global colour shift that really did degrade the irrelevant
    regions would look like. So it is the one headline cue whose deflation has a
    defensible reading, and the paper may not present it as unarguable bias."""
    rows = _frozen("claim_a_by_dimension.csv")
    assert _signed(rows, "brightness", "instruction_adherence")[0] <= 1
    assert _signed(rows, "brightness", "editing_quality")[0] == 5


def test_bandwagon_raises_the_score_for_following_an_instruction_it_never_touched():
    """The sharpest single cell in the study: `bandwagon` is a line of fabricated
    social proof in the PROMPT. It cannot have changed how well the edit followed the
    instruction, and every judge raises instruction adherence anyway."""
    rows = _frozen("claim_a_by_dimension.csv")
    for dim in DIMENSIONS:
        down, up, n = _signed(rows, "bandwagon", dim)
        assert (down, up, n) == (0, 5, 5), (dim, down, up, n)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_a_row_missing_one_dimension_leaves_the_other_two_as_well(tmp_path: Path):
    """Complete-case is decided on `fine_score` (all three dimensions), NOT per
    dimension. Deciding it per dimension would give each dimension its own item set,
    and "IA moved on 5 judges but DP on 4" would then be partly a statement about which
    answers happened to parse -- the mixed-variable defect this project already paid
    for once, one level down."""
    results = tmp_path / "v2"
    for model in PUBLISHED_ROSTER:
        io.write_jsonl(results / "raw_judgments" / f"scoring__{model}.jsonl", [
            _score(model, "s0", 5, 5, 5),
            _score(model, "s1", 5, 5, 5),
            _score(model, "s2", 5, 5, None),      # kimi-style: one dimension omitted
        ])
        io.write_jsonl(results / "biased_judgments" / f"scoring__{model}.jsonl", [
            _score(model, "s0", 4, 5, 5, "padding"),
            _score(model, "s1", 4, 5, 5, "padding"),
            _score(model, "s2", 4, 5, 5, "padding"),
        ])
    rows = build_claim_a_dimensions(results)
    # s2 is outside claim A's sample entirely, so every dimension reports n=2 ...
    assert {r["n"] for r in rows} == {2}
    # ... and the decomposition still holds: -1 on IA, 0 elsewhere.
    got = {r["dimension"]: r["mean_shift"] for r in rows}
    assert got == {"instruction_adherence": -1.0,
                   "editing_quality": 0.0,
                   "detail_preservation": 0.0}


def test_a_judge_outside_the_roster_cannot_enter_the_table(tmp_path: Path):
    """Same guard as every other claim table: a sixth judge in the directory is a
    sixth judge in the BH family, and that rewrites every q-value in it."""
    results = tmp_path / "v2"
    for model in list(PUBLISHED_ROSTER) + ["qwen3-vl-32b-instruct"]:
        io.write_jsonl(results / "raw_judgments" / f"scoring__{model}.jsonl",
                       [_score(model, f"s{i}", 5, 5, 5) for i in range(4)])
        io.write_jsonl(results / "biased_judgments" / f"scoring__{model}.jsonl",
                       [_score(model, f"s{i}", 4, 5, 5, "padding") for i in range(4)])
    rows = build_claim_a_dimensions(results)
    assert {r["judge_model"] for r in rows} == set(PUBLISHED_ROSTER)
