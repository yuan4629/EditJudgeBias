"""Tests for the D-S construct-validity screen.

Two properties carry real weight here and each has a test:

  * a missing or unparseable field must never PASS a scene. One of the four questions is an
    ethics hard stop, and "no child is present" is a claim that has to be positively observed;
  * the pass set must be refused outright when the screen's false-reject floor is unmeasured
    or too high, rather than returned with a caveat.
"""

from __future__ import annotations

import json

import pytest

from edit_judge_bias.experiments.run_construct_screen import (
    MAX_FALSE_REJECT,
    agreement_with_labels,
    false_reject_rate,
    passed_sample_ids,
    write_summary,
)
from edit_judge_bias.fairness.records import ConstructScreenResult
from edit_judge_bias.judges.parser import parse_construct_screen


def _row(sample_id, *, passed=True, control=False, ok=True, **fields):
    base = dict(is_photograph=True, single_subject=True, no_minor=True, skin_visible=True)
    base.update(fields)
    return ConstructScreenResult(sample_id=sample_id, is_control=control,
                                 passed=passed, parse_success=ok, **base)


# --------------------------------------------------------------------------- #
# Parsing: absence is never consent                                            #
# --------------------------------------------------------------------------- #
def test_all_four_true_passes():
    p = parse_construct_screen(
        '{"is_photograph":true,"single_subject":true,"no_minor":true,'
        '"skin_visible":true,"reason":"ok"}')
    assert p.success is True and p.passed is True


@pytest.mark.parametrize("missing", ["is_photograph", "single_subject",
                                     "no_minor", "skin_visible"])
def test_a_missing_field_never_passes(missing):
    """A dropped field can only ever cost a scene, never admit one.

    Over-rejecting shrinks n; under-rejecting puts a minor into a dataset of deliberately
    manipulated skin tones. Only one of those is recoverable.
    """
    obj = {"is_photograph": True, "single_subject": True,
           "no_minor": True, "skin_visible": True}
    obj.pop(missing)
    import json

    p = parse_construct_screen(json.dumps(obj))
    assert p.success is True
    assert p.passed is False, f"a missing {missing} must not pass"


def test_a_volunteered_pass_cannot_override_a_false_field():
    """The model saying "pass": true while answering no_minor false must not win."""
    p = parse_construct_screen(
        '{"is_photograph":true,"single_subject":true,"no_minor":false,'
        '"skin_visible":true,"pass":true}')
    assert p.passed is False


def test_unparseable_response_is_a_failure_not_a_pass():
    p = parse_construct_screen("I cannot help with that.")
    assert p.success is False and p.passed is None


def test_fenced_json_is_recovered():
    p = parse_construct_screen(
        'Here you go:\n```json\n{"is_photograph": true, "single_subject": true,'
        ' "no_minor": true, "skin_visible": true}\n```')
    assert p.success is True and p.passed is True


# --------------------------------------------------------------------------- #
# The floor gates the pass set                                                 #
# --------------------------------------------------------------------------- #
def test_false_reject_rate_counts_only_controls():
    rows = [_row("c1", control=True, passed=True),
            _row("c2", control=True, passed=False),
            _row("s1", passed=False)]
    assert false_reject_rate(rows) == pytest.approx(0.5)


def test_no_controls_means_the_pass_set_is_refused():
    """An unmeasured floor cannot be gated on -- that is the glm-4v lesson in code."""
    with pytest.raises(RuntimeError, match="never measured"):
        passed_sample_ids([_row("s1")])


def test_a_shredding_screen_is_refused():
    """Rejecting good scenes at random is not filtering, however safe it looks."""
    controls = [_row(f"c{i}", control=True, passed=i < 7) for i in range(10)]
    with pytest.raises(RuntimeError, match="above the"):
        passed_sample_ids(controls + [_row("s1")])


def test_a_clean_floor_returns_only_study_rows():
    controls = [_row(f"c{i}", control=True, passed=True) for i in range(10)]
    got = passed_sample_ids(controls + [_row("s1", passed=True),
                                        _row("s2", passed=False)])
    assert got == {"s1"}, "controls must not leak into the pool they calibrate"


def test_the_ceiling_is_the_same_as_the_pair_gates():
    assert MAX_FALSE_REJECT == 0.10


def test_parse_failures_never_enter_the_pass_set():
    controls = [_row(f"c{i}", control=True, passed=True) for i in range(10)]
    got = passed_sample_ids(controls + [_row("s1", passed=True, ok=False)])
    assert got == set()


# --------------------------------------------------------------------------- #
# Agreement is reported in BOTH directions                                     #
# --------------------------------------------------------------------------- #
def test_agreement_reports_both_error_directions():
    """A floor alone hides the error that matters most for the ethics question."""
    rows = [_row("u1", passed=True), _row("u2", passed=False),
            _row("x1", passed=True), _row("x2", passed=False)]
    labels = {"usable": ["u1", "u2"], "unusable": ["x1", "x2"]}
    got = agreement_with_labels(rows, labels)
    assert got["false_reject"] == 1
    assert got["false_accept"] == 1
    assert got["agreement"] == pytest.approx(0.5)


def test_a_minor_admitted_by_the_screen_is_named_not_just_counted():
    """A false accept on the ethics question is not a rate, it is a list to act on."""
    rows = [_row("kid", passed=True), _row("cgi", passed=True)]
    labels = {"usable": [], "unusable": ["kid", "cgi"],
              "unusable_reasons": {"kid": "MINOR", "cgi": "non-photographic (CGI)"}}
    got = agreement_with_labels(rows, labels)
    assert got["minor_false_accept"] == ["kid"]


def test_summary_separates_the_floor_row(tmp_path):
    write_summary(tmp_path / "s.csv", [
        _row("c1", control=True, passed=True),
        _row("s1", passed=True), _row("s2", passed=False),
    ])
    text = (tmp_path / "s.csv").read_text(encoding="utf-8")
    assert "false_reject_floor" in text and "measured" in text


# --------------------------------------------------------------------------- #
# ★★ The pass list must be rewritten by the run that measures it (2026-08-01)  #
# --------------------------------------------------------------------------- #
def _res(sample_id, *, passed=True, parse_success=True, is_control=False):
    return ConstructScreenResult(
        sample_id=sample_id, validator_model="gpt-4o-mini", source_dataset="omniedit_train",
        is_photograph=passed, single_subject=passed, no_minor=passed, skin_visible=passed,
        passed=passed, parse_success=parse_success, is_control=is_control,
        raw_response_path=f"raw/{sample_id}.txt",
    )


def test_the_pass_list_is_rewritten_from_the_full_done_set(tmp_path):
    """The stale-gate bug, pinned.

    The pass list gates pass 2 of the pool build. It used to be produced out-of-band, so after
    the screen increment took the measured set 664 -> 1,309 rows (396 -> 785 passes), the file
    on disk was still the stale 396. `--include-ids` filters to whatever it is handed, so pass 2
    would have rebuilt the pool at HALF the earned n and reported a clean result.
    """
    from edit_judge_bias.experiments import run_construct_screen as RCS

    out = tmp_path / "passed.json"
    ctrl = [_res(f"k{i}", is_control=True) for i in range(10)]
    n = RCS.write_passed(out, [_res("a"), _res("b"), _res("c", passed=False)] + ctrl)
    assert n == 2
    assert json.loads(out.read_text(encoding="utf-8")) == ["a", "b"]

    # A later run that has seen more rows must OVERWRITE, not append or keep the old file.
    n2 = RCS.write_passed(
        out, [_res("a"), _res("b"), _res("c", passed=False), _res("d")] + ctrl)
    assert n2 == 3
    assert json.loads(out.read_text(encoding="utf-8")) == ["a", "b", "d"]


def test_controls_and_parse_failures_never_enter_the_pass_list(tmp_path):
    """Controls are calibration, not pool members; an unparsed row has no verdict to trust."""
    from edit_judge_bias.experiments import run_construct_screen as RCS

    out = tmp_path / "passed.json"
    RCS.write_passed(out, [_res("real"), _res("unparsed", parse_success=False)]
                     + [_res(f"k{i}", is_control=True) for i in range(10)])
    assert json.loads(out.read_text(encoding="utf-8")) == ["real"]
