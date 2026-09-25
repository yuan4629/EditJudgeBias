"""Tests for scoring/pairwise prompt builders (Milestone 3)."""

from __future__ import annotations

import pytest

from edit_judge_bias.prompts import build_pairwise_prompt, build_scoring_prompt
from edit_judge_bias.prompts.styles import (
    PAIRWISE_DEBIAS_LINE,
    SCORING_DEBIAS_LINE,
)


def test_scoring_vanilla_has_no_debias_line():
    p = build_scoring_prompt("Remove the man.", style="vanilla")
    assert "Remove the man." in p
    assert "overall_score" in p
    assert SCORING_DEBIAS_LINE not in p  # vanilla is a true baseline


def test_scoring_bias_aware_adds_debias_line():
    p = build_scoring_prompt("Remove the man.", style="bias_aware")
    assert SCORING_DEBIAS_LINE in p


def test_scoring_rubric_first_mentions_criteria_order():
    p = build_scoring_prompt("Remove the man.", style="rubric_first")
    assert "independently" in p.lower()
    assert SCORING_DEBIAS_LINE not in p


def test_pairwise_styles():
    base = build_pairwise_prompt("Make it red.", style="vanilla")
    assert "Make it red." in base
    assert "winner" in base
    assert PAIRWISE_DEBIAS_LINE not in base
    assert PAIRWISE_DEBIAS_LINE in build_pairwise_prompt("x", style="bias_aware")


def test_viescore_style_mentions_sc_pq_rubric():
    p = build_scoring_prompt("Remove the man.", style="viescore")
    assert "VIEScore" in p
    assert "Semantic Consistency" in p and "Perceptual Quality" in p
    assert "overall_score" in p  # still emits the standard JSON


# --------------------------------------------------------------------------- #
# rating scale                                                                 #
# --------------------------------------------------------------------------- #
def test_scoring_defaults_to_the_ten_point_scale():
    p = build_scoring_prompt("Remove the man.")
    assert "from 1 to 10" in p
    assert "<1-10>" in p
    assert "<1-5>" not in p


def test_scoring_scale_is_overridable():
    p = build_scoring_prompt("Remove the man.", scale=5)
    assert "from 1 to 5" in p and "<1-5>" in p


def test_viescore_line_tracks_the_scale():
    """Regression: the VIEScore rubric line hard-coded "1-5", so a 1-10 run would
    have told the judge two different scales in the same prompt."""
    p = build_scoring_prompt("x", style="viescore")
    assert "1-10 scale" in p and "1-5 scale" not in p


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        build_scoring_prompt("x", style="nonsense")


# --------------------------------------------------------------------------- #
# A-class prompt-level bias injections (A2 bandwagon, A3 self-preference)       #
# --------------------------------------------------------------------------- #
def test_scoring_bandwagon_off_by_default():
    p = build_scoring_prompt("Remove the man.")
    assert "rated" not in p.lower()


def test_scoring_bandwagon_and_model_name_inject():
    p = build_scoring_prompt("Remove the man.", bandwagon=True, model_name="gpt-5.5")
    assert "highly" in p.lower()
    assert "gpt-5.5" in p


def test_pairwise_bandwagon_targets_side():
    p = build_pairwise_prompt("Make it red.", bandwagon_target="B")
    assert "preferred Image B" in p
    assert "preferred Image A" not in p


def test_pairwise_model_names_label_both_sides():
    p = build_pairwise_prompt("Make it red.", model_name_a="qwen", model_name_b="kimi")
    assert "Image A was produced by qwen" in p
    assert "Image B was produced by kimi" in p


def test_prompt_bias_composes_with_vanilla_baseline():
    # Injecting a prompt bias must not pull in the debias (mitigation) line.
    p = build_scoring_prompt("x", bandwagon=True)
    assert SCORING_DEBIAS_LINE not in p
