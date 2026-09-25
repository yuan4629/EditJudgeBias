"""Tests for the shared instruction -> (edit_type, content_category) rules.

Every full-version source (ImagenHub, GenAI-Bench, MagicBrush dev) derives its
`edit_type` from this module, and `edit_type` is a §9.4 group key -- so a silent
change here silently rewrites the breakdown for ~3k samples. The ordering tests
below are the load-bearing ones: they fail if someone reorders EDIT_TYPE_RULES.
"""

from __future__ import annotations

import pytest

from edit_judge_bias.data.edit_type_rules import (
    EDIT_TYPE_RULES,
    classify_batch,
    classify_content_category,
    classify_edit_type,
    rules_as_config,
)

VALID_EDIT_TYPES = {"add", "remove", "replace", "color", "background", "low-level"}


def test_every_rule_declares_a_taxonomy_edit_type():
    for edit_type, _ in EDIT_TYPE_RULES:
        assert edit_type in VALID_EDIT_TYPES, edit_type


def test_empty_and_none_instructions_are_unclassified():
    assert classify_edit_type("") is None
    assert classify_edit_type("   ") is None
    assert classify_edit_type(None) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The explicit-imperative forms rules 1-6 have always covered.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("Add a horse to the image", "add"),
        ("insert a lamp on the table", "add"),
        ("remove the man from the photo", "remove"),
        ("erase the sign", "remove"),
        ("replace the frisbee with a ball", "replace"),
        ("change the frisbee into a ball", "replace"),
        ("change the background to a beach", "background"),
        ("change the color of the car to red", "color"),
        ("make it a black and white photo", "low-level"),
        ("blur the image", "low-level"),
        ("add a watermark", "low-level"),
    ],
)
def test_explicit_imperative_forms(instruction, expected):
    assert classify_edit_type(instruction) == expected


# --------------------------------------------------------------------------
# MagicBrush-style paraphrases (tiers 7-9, added 2026-07-25). Each case here is
# a real instruction taken from the built manifests, not an invented one.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "instruction,expected",
    [
        # tier 7 -- object introduction
        ("Have the cow wear a hat.", "add"),
        ("He should be eating a watermelon", "add"),
        ("let the plate contain ice cream", "add"),
        ("Let the bluebery cake be topped with chocolate syrup.", "add"),
        ("Have there be a dolphin jumping out of the water", "add"),
        ("What if the man had a hat?", "add"),
        ("let a herd of sheep block the taxi", "add"),
        ("A dog should be near the sheep.", "add"),
        ("It could be a microwave next to the woman.", "add"),
        ("let the bowl have chocolate sauce", "add"),
        ("cover the bread with sauce and salad", "add"),
        ("the ocean should have waves.", "add"),
        ("What if there was a drawing of a bird on his shirt?", "add"),
        # tier 8 -- colour by paraphrase
        ("Make all the grass green.", "color"),
        ("make one of the shoes black", "color"),
        ("The bed should be red.", "color"),
        ("let the cat have blue eyes", "color"),
        ("let the woman have blonde hair", "color"),
        ("Make the scarf multi-colored.", "color"),
        # tier 9 -- replace by paraphrase
        ("make it a pepperoni pizza", "replace"),
        ("Make the zebra a regular horse.", "replace"),
        ("Make the donut a cupcake.", "replace"),
        ("let the bed be wooden", "replace"),
        ("let it be a bullet train", "replace"),
        ("make the ramp cement", "replace"),
        ("turn the mountain in a waterfall", "replace"),
        # extended rules 1/4
        ("Could he be in the forest?", "background"),
        ("make the plate empty", "remove"),
        ("take the objects off the dresser", "remove"),
        ("leave nothing on top of the cheesecake", "remove"),
    ],
)
def test_magicbrush_paraphrase_forms(instruction, expected):
    assert classify_edit_type(instruction) == expected


# --------------------------------------------------------------------------
# Ordering guarantees. These are the reason EDIT_TYPE_RULES is an ordered list
# rather than a dict; each case would flip if the tiers were reordered.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "instruction,expected,why",
    [
        (
            "remove the man holding a cup",
            "remove",
            "explicit `remove` (rule 4) must beat the tier-7 `holding` add cue",
        ),
        (
            "replace the woman wearing a hat with a man",
            "replace",
            "explicit `replace` (rule 5) must beat the tier-7 `wearing` add cue",
        ),
        (
            "Have the woman be wearing a blue tank top",
            "add",
            "a newly introduced coloured object is an add, not a recolour: "
            "tier 7 must precede tier 8",
        ),
        (
            "Can we have a blue airplane?",
            "color",
            "`have a <colour> <noun>` is a recolour, so the tier-7 rule carries a "
            "colour lookahead and this falls through to tier 8",
        ),
        (
            "Let a green towel be hung in the bathroom.",
            "add",
            "`let a ...` introduces a new object even when a colour word follows",
        ),
        (
            "change the background to a beach",
            "background",
            "rule 1 must precede the generic replace rule",
        ),
        (
            "make it a black and white photo",
            "low-level",
            "rule 2 (style) must precede the colour rules",
        ),
        (
            "let the bluebery cake be topped with chocolate syrup.",
            "add",
            "tier 7 `topped with` must beat the tier-9 `let the X be Y` replace form",
        ),
    ],
)
def test_rule_order_is_load_bearing(instruction, expected, why):
    assert classify_edit_type(instruction) == expected, why


# --------------------------------------------------------------------------
# The action/pose/state gap is deliberate -- see TAXONOMY_GAP_NOTE. If a future
# rule starts labelling these, this test fails and forces the choice to be
# explicit rather than accidental.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "instruction",
    [
        "Let the woman smile.",
        "let the woman cry",
        "let the dog yawn",
        "make the cat lick its nose",
        "Open the zebra's mouth.",
        "Let the man press the keyboard.",
        "Have the person swing the racquet between his legs",
        "let the horse close its eyes",
    ],
)
def test_action_and_pose_edits_stay_unclassified(instruction):
    assert classify_edit_type(instruction) is None


# --------------------------------------------------------------------------
# content_category
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texts,expected",
    [
        (("add a hat to the man",), "human"),
        (("add a collar to the dog",), "animal"),
        (("put a boat on the lake",), "scenery"),
        (("place a cup on the table",), "object"),
        (("increase the saturation",), "global"),
    ],
)
def test_content_category(texts, expected):
    assert classify_content_category(*texts) == expected


def test_content_category_uses_the_caption_when_the_instruction_is_terse():
    assert classify_content_category("make it red", "") == "global"
    assert classify_content_category("make it red", "A dog on a couch.") == "animal"


def test_content_category_ignores_none_and_empty_texts():
    assert classify_content_category("", None, "a horse in a field") == "animal"  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# classify_batch / reporting
# --------------------------------------------------------------------------
def test_classify_batch_reports_coverage_and_the_unclassified_tail():
    instructions = [
        "add a horse",           # add
        "remove the sign",       # remove
        "Let the woman smile.",  # unclassified (action)
    ]
    edit_types, categories, report = classify_batch(instructions)
    assert edit_types == ["add", "remove", None]
    assert len(categories) == 3
    assert report.n == 3
    assert report.n_classified == 2
    assert report.coverage == pytest.approx(2 / 3)
    assert report.unclassified == ["Let the woman smile."]
    assert report.edit_types == {"add": 1, "remove": 1}
    assert "2/3" in report.summary()


def test_classify_batch_pads_short_caption_lists():
    edit_types, categories, report = classify_batch(
        ["add a horse", "add a bird"], captions=["A field."]
    )
    assert edit_types == ["add", "add"]
    assert report.n == 2
    assert categories[0] == "animal"


def test_classify_batch_on_no_input():
    edit_types, categories, report = classify_batch([])
    assert edit_types == [] and categories == []
    assert report.coverage == 0.0


def test_rules_as_config_is_serialisable_for_run_provenance():
    cfg = rules_as_config()
    assert set(cfg) <= VALID_EDIT_TYPES
    assert all(isinstance(v, list) and v and isinstance(v[0], str) for v in cfg.values())


def test_classification_is_case_insensitive_and_deterministic():
    assert classify_edit_type("ADD A HORSE") == "add"
    assert classify_edit_type("Add A Horse") == "add"
    assert [classify_edit_type("make it a pepperoni pizza") for _ in range(3)] == ["replace"] * 3
