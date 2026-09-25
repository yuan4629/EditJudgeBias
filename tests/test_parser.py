"""Tests for the robust judge-output parser (Milestone 3)."""

from __future__ import annotations

from edit_judge_bias.judges.parser import (
    extract_json_object,
    parse_pairwise,
    parse_scoring,
    parse_validation,
)

VALID = '{"instruction_adherence": 4, "editing_quality": 3, "detail_preservation": 5, "overall_score": 4, "reason": "ok"}'


# --------------------------------------------------------------------------- #
# extract_json_object                                                          #
# --------------------------------------------------------------------------- #
def test_extract_bare_json():
    assert extract_json_object(VALID)["overall_score"] == 4


def test_extract_from_markdown_fence():
    raw = f"Here you go:\n```json\n{VALID}\n```\nThanks!"
    assert extract_json_object(raw)["overall_score"] == 4


def test_extract_from_surrounding_prose():
    raw = f"My verdict is {VALID} — hope that helps."
    assert extract_json_object(raw)["editing_quality"] == 3


def test_extract_handles_braces_in_strings():
    raw = '{"reason": "contains { and } braces", "overall_score": 2}'
    obj = extract_json_object(raw)
    assert obj["overall_score"] == 2
    assert "{" in obj["reason"]


def test_extract_returns_none_on_garbage():
    assert extract_json_object("no json here at all") is None
    assert extract_json_object("") is None


# --------------------------------------------------------------------------- #
# parse_scoring                                                                #
# --------------------------------------------------------------------------- #
def test_parse_scoring_valid():
    r = parse_scoring(VALID)
    assert r.success
    assert r.overall_score == 4
    assert r.detail_preservation == 5
    assert r.reason == "ok"


def test_parse_scoring_clamps_out_of_range():
    r = parse_scoring('{"overall_score": 99}')
    assert r.success and r.overall_score == 10
    r2 = parse_scoring('{"overall_score": 0}')
    assert r2.success and r2.overall_score == 1


def test_parse_scoring_does_not_truncate_within_scale():
    """Regression: the clamp used to be hard-wired to 5, which would have turned
    every 6-10 answer of the 1-10 main grid into a silent 5."""
    r = parse_scoring('{"overall_score": 9}')
    assert r.success and r.overall_score == 9 and r.score_scale == 10


def test_parse_scoring_honours_explicit_scale():
    r = parse_scoring('{"overall_score": 9}', scale=5)
    assert r.success and r.overall_score == 5 and r.score_scale == 5


def test_parse_scoring_fine_score_is_dimension_sum():
    r = parse_scoring(VALID)
    assert r.fine_score == 4 + 3 + 5


def test_parse_scoring_fine_score_none_when_a_dimension_missing():
    r = parse_scoring('{"overall_score": 4, "editing_quality": 3}')
    assert r.success and r.fine_score is None


def test_parse_scoring_coerces_string_score():
    r = parse_scoring('{"overall_score": "4 out of 5"}')
    assert r.success and r.overall_score == 4


def test_parse_scoring_missing_overall_fails():
    r = parse_scoring('{"editing_quality": 4}')
    assert not r.success
    assert "overall_score" in r.error


def test_parse_scoring_no_json_fails():
    r = parse_scoring("I refuse to answer.")
    assert not r.success
    assert r.overall_score is None


# --------------------------------------------------------------------------- #
# parse_pairwise                                                               #
# --------------------------------------------------------------------------- #
def test_parse_pairwise_basic():
    assert parse_pairwise('{"winner": "A"}').winner == "A"
    assert parse_pairwise('{"winner": "tie"}').winner == "Tie"


def test_parse_pairwise_normalizes_phrases():
    assert parse_pairwise('{"winner": "Image B"}').winner == "B"
    assert parse_pairwise('{"winner": "neither"}').winner == "Tie"


def test_parse_pairwise_invalid_winner_fails():
    r = parse_pairwise('{"winner": "maybe C"}')
    assert not r.success


def test_parse_pairwise_no_json_fails():
    r = parse_pairwise("hmm")
    assert not r.success and r.winner is None


# --------------------------------------------------------------------------- #
# parse_validation (§6.2)                                                      #
# --------------------------------------------------------------------------- #
def test_parse_validation_pass():
    raw = ('{"instruction_adherence_changed": false, "editing_quality_changed": false, '
           '"detail_preservation_changed": false, "major_semantic_shift": false, '
           '"pass": true, "reason": "cosmetic only"}')
    r = parse_validation(raw)
    assert r.success and r.passed is True
    assert r.detail_preservation_changed is False


def test_parse_validation_fail_when_detail_changed():
    raw = ('{"instruction_adherence_changed": false, "editing_quality_changed": false, '
           '"detail_preservation_changed": true, "major_semantic_shift": false, '
           '"pass": false}')
    r = parse_validation(raw)
    assert r.success and r.passed is False
    assert r.detail_preservation_changed is True


def test_parse_validation_derives_pass_when_missing():
    raw = ('{"instruction_adherence_changed": false, "editing_quality_changed": true, '
           '"detail_preservation_changed": false, "major_semantic_shift": false}')
    r = parse_validation(raw)
    assert r.success and r.passed is False  # derived: a change -> not passed


def test_parse_validation_coerces_string_bools():
    raw = ('{"instruction_adherence_changed": "no", "editing_quality_changed": "no", '
           '"detail_preservation_changed": "no", "major_semantic_shift": "no", "pass": "yes"}')
    r = parse_validation(raw)
    assert r.success and r.passed is True


def test_parse_validation_no_json_fails():
    r = parse_validation("I cannot answer")
    assert not r.success and r.passed is None


def test_trailing_comma_before_brace_is_repaired():
    """MEASURED: gemini-3.5-flash emitted one otherwise-perfect answer whose only flaw
    was a comma before the closing brace, and strict json rejected the whole thing."""
    raw = '```json\n{\n  "instruction_adherence": 7,\n  "editing_quality": 6,\n' \
          '  "detail_preservation": 5,\n  "overall_score": 6,\n  "reason": "ok",\n}\n```'
    parsed = parse_scoring(raw, scale=10)
    assert parsed.success
    assert parsed.overall_score == 6 and parsed.fine_score == 18


def test_trailing_comma_in_nested_array_is_repaired():
    assert extract_json_object('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


def test_comma_inside_a_string_is_not_touched():
    obj = extract_json_object('{"reason": "added a border, then a caption", "overall_score": 4}')
    assert obj["reason"] == "added a border, then a caption"
