"""Tests for the MockJudgeAdapter (Milestone 3)."""

from __future__ import annotations

from pathlib import Path

from edit_judge_bias.judges import build_adapter
from edit_judge_bias.judges.base import JudgeRequest
from edit_judge_bias.judges.mock_judge import MockJudgeAdapter
from edit_judge_bias.judges.parser import parse_pairwise, parse_scoring


def _score_req(prompt="score this"):
    return JudgeRequest(prompt=prompt, images=[Path("o.png"), Path("e.png")], task="scoring")


def test_mock_is_deterministic():
    a = MockJudgeAdapter()
    r1 = a.generate(_score_req())
    r2 = a.generate(_score_req())
    assert r1 == r2


def test_mock_varies_by_prompt():
    a = MockJudgeAdapter()
    assert a.generate(_score_req("p1")) != a.generate(_score_req("p2"))


def test_mock_scoring_output_parses():
    a = MockJudgeAdapter()
    parsed = parse_scoring(a.generate(_score_req()))
    assert parsed.success
    assert 1 <= parsed.overall_score <= 10


def test_mock_honours_score_scale():
    """The mock must span the scale the runner asks for, or a mock dry-run would
    exercise a range the real grid never produces."""
    a = MockJudgeAdapter(score_scale=5)
    seen = {parse_scoring(a.generate(_score_req(f"p{i}")), scale=5).overall_score
            for i in range(80)}
    assert seen and max(seen) <= 5
    wide = {parse_scoring(a2.generate(_score_req(f"p{i}"))).overall_score
            for i in range(80) for a2 in [MockJudgeAdapter(score_scale=10)]}
    assert max(wide) > 5


def test_mock_pairwise_output_parses():
    a = MockJudgeAdapter()
    req = JudgeRequest("compare", [Path("o.png"), Path("a.png"), Path("b.png")], "pairwise")
    parsed = parse_pairwise(a.generate(req))
    assert parsed.success
    assert parsed.winner in {"A", "B", "Tie"}


def test_mock_response_styles_round_trip_through_parser():
    for style in ("valid", "fenced", "prose"):
        a = MockJudgeAdapter(response_style=style)
        assert parse_scoring(a.generate(_score_req())).success, style


def test_mock_invalid_style_fails_parsing():
    a = MockJudgeAdapter(response_style="invalid")
    assert not parse_scoring(a.generate(_score_req())).success


def test_build_adapter_mock():
    a = build_adapter({"type": "mock", "model_name": "m1"})
    assert isinstance(a, MockJudgeAdapter)
    assert a.model_name == "m1"


def test_build_adapter_unknown_type_raises():
    import pytest

    with pytest.raises(ValueError):
        build_adapter({"type": "nonexistent-provider"})
