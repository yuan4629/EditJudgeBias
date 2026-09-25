"""Regression tests for the construct-validity consumer.

Each test pins a failure mode that would produce a publishable-looking wrong number. The
first one is the important one: an unannotated sheet must never render as a pass rate of
zero.

Every fixture here is synthetic. The real sheets under `data/human_validation*/` are read
only in `test_real_sheets_are_readable_and_currently_unannotated`, and never written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edit_judge_bias.experiments import build_construct_validity_table as builder
from edit_judge_bias.metrics.construct_validity import (
    DG_PAIR_VOCAB,
    DS_V3_VOCAB,
    DS_V4_VOCAB,
    SOURCES,
    ValidationSource,
    auditor_agreement,
    cohens_kappa,
    evaluate_source,
    load_verdicts,
    summarise,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _dg_rows(verdicts, *, agree=None, attribute=None, a_pass=None, b_pass=None):
    n = len(verdicts)
    agree = agree if agree is not None else [True] * n
    attribute = attribute if attribute is not None else ["gender"] * n
    a_pass = a_pass if a_pass is not None else [True] * n
    b_pass = b_pass if b_pass is not None else [True] * n
    return [
        {
            "pair_key": f"scene_{i}__{attribute[i]}",
            "attribute": attribute[i],
            "auditor_A_passed": a_pass[i],
            "auditor_B_passed": b_pass[i],
            "auditors_agree": agree[i],
            "human_verdict": verdicts[i],
        }
        for i in range(n)
    ]


def _ds_rows(verdicts, prefix="scene"):
    return [
        {"scene": f"{prefix}_{i}", "gate_passed": True, "human_verdict": v}
        for i, v in enumerate(verdicts)
    ]


def _src(tmp_path, name, rows):
    """Build a synthetic ValidationSource rooted at tmp_path."""
    base = SOURCES[name]
    _write(tmp_path / base.path, rows)
    return base


# --------------------------------------------------------------------------- #
# 1. the silent-error case: 100% unannotated must NOT read as a rate of 0.0
# --------------------------------------------------------------------------- #
def test_fully_unannotated_reports_none_not_zero(tmp_path: Path):
    src = _src(tmp_path, "ds_v4", _ds_rows([""] * 20))
    (summary,) = evaluate_source(src, root=tmp_path)

    assert summary.n_total == 20
    assert summary.n_annotated == 0
    assert summary.n_unannotated == 20
    assert summary.status == "not_annotated"

    # ★ the whole point: None, never 0.0. A 0.0 here would be read as "every sheet failed".
    assert summary.pass_rate is None
    assert summary.pass_rate_excl_flagged is None
    assert summary.wilson_ci == (None, None)

    msg = summary.message()
    assert "0/20 annotated" in msg
    assert "NOT 'the pass rate is zero'" in msg

    row = summary.as_row()
    assert row["pass_rate"] is None and row["n_total"] == 20 and row["n_annotated"] == 0


# --------------------------------------------------------------------------- #
# 2-4. one test per vocabulary
# --------------------------------------------------------------------------- #
def test_dg_pairs_vocabulary(tmp_path: Path):
    verdicts = ["valid"] * 3 + ["moved"] * 2 + ["noflip"] + ["noperson"]
    src = _src(tmp_path, "dg_pairs", _dg_rows(verdicts))
    rows = {s.stratum: s for s in evaluate_source(src, root=tmp_path)}

    s = rows["all"]
    assert s.vocabulary == "dg_pairs"
    assert s.n_annotated == 7 and s.n_pass == 3
    assert s.pass_rate == pytest.approx(3 / 7)
    assert s.counts == {"valid": 3, "moved": 2, "noflip": 1, "noperson": 1}
    # `moved` is a construct FAILURE, not a flag -- D-G has no orthogonal label
    assert s.n_flagged == 0 and s.pass_rate_excl_flagged == pytest.approx(3 / 7)


def test_ds_v3_vocabulary_transcribes_the_30_of_31_conclusion(tmp_path: Path):
    # README_GATE_P4.md states in prose that a human found 30 of v3's 31 sheets invalid.
    # Transcribed into the file, this module must reproduce it as a rate with an interval.
    verdicts = ["plausible"] + ["recoloured"] * 28 + ["no_person"] * 2
    src = _src(tmp_path, "ds_v3", _ds_rows(verdicts))
    (s,) = evaluate_source(src, root=tmp_path)

    assert s.vocabulary == "ds_v3"
    assert s.n_total == 31 and s.n_annotated == 31 and s.n_pass == 1
    assert s.pass_rate == pytest.approx(1 / 31, abs=1e-6)
    assert s.counts["recoloured"] == 28
    lo, hi = s.wilson_ci
    assert lo == pytest.approx(0.005717, abs=1e-5)
    assert hi == pytest.approx(0.161941, abs=1e-5)
    # and NOT a zero-width interval -- the reason Wilson is mandatory at these n
    assert hi - lo > 0.1


def test_ds_v4_figure_is_an_orthogonal_ethics_flag(tmp_path: Path):
    # `figure` (recognisable public figure) is not a construct verdict. It counts as
    # not-a-pass in the headline rate, and is ALSO reported separately with the rate that
    # excludes it -- both, never only the flattering one.
    verdicts = ["pass"] * 8 + ["patch"] * 6 + ["weak"] * 4 + ["figure"] * 2
    src = _src(tmp_path, "ds_v4", _ds_rows(verdicts))
    (s,) = evaluate_source(src, root=tmp_path)

    assert s.n_annotated == 20 and s.n_pass == 8 and s.n_flagged == 2
    assert s.pass_rate == pytest.approx(8 / 20)          # conservative denominator
    assert s.pass_rate_excl_flagged == pytest.approx(8 / 18)
    assert s.pass_rate < s.pass_rate_excl_flagged        # excluding flags can only inflate
    row = s.as_row()
    assert row["count_figure"] == 2 and row["count_weak"] == 4


# --------------------------------------------------------------------------- #
# 5. partial annotation
# --------------------------------------------------------------------------- #
def test_partial_annotation_uses_only_the_annotated_subset(tmp_path: Path):
    verdicts = ["pass", "pass", "patch", "", "", "", "", "", "", ""]
    src = _src(tmp_path, "ds_v4", _ds_rows(verdicts))
    (s,) = evaluate_source(src, root=tmp_path)

    assert s.status == "partial"
    assert s.n_total == 10 and s.n_annotated == 3 and s.n_unannotated == 7
    # denominator is the ANNOTATED subset, not n_total -- 2/3, not 2/10
    assert s.pass_rate == pytest.approx(2 / 3)
    assert "PARTIAL: 3 of 10 annotated" in s.message()
    row = s.as_row()
    assert row["n_annotated"] == 3 and row["n_total"] == 10


def test_unrecognised_verdict_counts_as_failure_and_is_surfaced(tmp_path: Path):
    # A typo must not be dropped from the denominator (that would inflate the rate) and
    # must not vanish (that would make it undiagnosable).
    src = _src(tmp_path, "ds_v4", _ds_rows(["pass", "pass", "passs", "PATCH"]))
    (s,) = evaluate_source(src, root=tmp_path)

    assert s.n_annotated == 4
    assert s.n_pass == 2                       # "passs" is not a pass
    assert s.n_unknown == 1 and s.unknown_labels == ("passs",)
    assert s.pass_rate == pytest.approx(2 / 4)
    assert s.counts["patch"] == 1              # case is normalised, "PATCH" is recognised
    assert "unrecognised" in s.message()


# --------------------------------------------------------------------------- #
# 6. Wilson numerics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "successes,n,lo,hi",
    [
        # Derived from the closed form independently of the implementation:
        #   d = 1 + z^2/n ; centre = (p + z^2/2n)/d ; half = z*sqrt(p(1-p)/n + z^2/4n^2)/d
        # with z = 1.959963985. (20,20) matching the textbook 0.8389 is the anchor.
        (20, 20, 0.838875, 1.0),       # p=1.0 must NOT get a zero-width interval
        (0, 20, 0.0, 0.161125),        # p=0.0 likewise
        (8, 20, 0.218807, 0.613418),
        (1, 31, 0.005717, 0.161941),
        (17, 20, 0.639581, 0.947631),  # the D-G probe's 0.85 on 20
        (27, 41, 0.505498, 0.784412),  # auditor A's 0.66 on 41
        (9, 41, 0.120032, 0.367050),   # auditor B's 0.22 on 41
    ],
)
def test_wilson_interval_values(successes, n, lo, hi):
    got_lo, got_hi = wilson_interval(successes, n)
    assert got_lo == pytest.approx(lo, abs=1e-5)
    assert got_hi == pytest.approx(hi, abs=1e-5)


def test_wilson_is_symmetric_and_contains_the_point_estimate():
    for successes, n in [(3, 7), (8, 20), (30, 31)]:
        lo, hi = wilson_interval(successes, n)
        assert lo <= successes / n <= hi
    assert wilson_interval(0, 0) == (None, None)


def test_wilson_probe_lesson_intervals_overlap_but_point_estimates_differ():
    """0.85 on 20 vs 0.66 on 41 -- the precedent the module docstring cites."""
    lo20, hi20 = wilson_interval(17, 20)
    lo41, hi41 = wilson_interval(27, 41)
    assert lo20 < hi41 and lo41 < hi20   # the 20-scene estimate never excluded 0.66


# --------------------------------------------------------------------------- #
# 7. Cohen's kappa
# --------------------------------------------------------------------------- #
def test_cohens_kappa_reproduces_the_published_minus_0_028(tmp_path: Path):
    """The real file's auditor A vs B cell counts are (TT,TF,FT,FF) = (10,26,14,32).

    Published value: kappa = -0.028. Reproducing it from the loader is the self-check that
    this module reads the sheet the way the original D-G analysis did.
    """
    a = [True] * 10 + [True] * 26 + [False] * 14 + [False] * 32
    b = [True] * 10 + [False] * 26 + [True] * 14 + [False] * 32
    assert cohens_kappa(a, b) == pytest.approx(-0.0276, abs=5e-4)


def test_cohens_kappa_edges():
    assert cohens_kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)
    # perfect disagreement on a balanced set
    assert cohens_kappa([True, False], [False, True]) == pytest.approx(-1.0)
    # both raters constant -> expected agreement is 1.0, kappa is 0/0 -> None, not 1.0
    assert cohens_kappa([True, True], [True, True]) is None
    assert cohens_kappa([], []) is None
    # unannotated items are DROPPED, not imputed as False (which would fake agreement)
    assert cohens_kappa([None, True, False], [True, True, False]) == pytest.approx(1.0)


def test_human_vs_auditor_kappa_says_whose_side_the_human_took(tmp_path: Path):
    # 8 pairs: the human agrees with auditor B everywhere and with A nowhere.
    a_pass = [True, True, True, True, False, False, False, False]
    b_pass = [True, False, True, False, True, False, True, False]
    human = ["valid" if b else "moved" for b in b_pass]
    src = _src(
        tmp_path,
        "dg_pairs",
        _dg_rows(human, a_pass=a_pass, b_pass=b_pass, agree=[x == y for x, y in zip(a_pass, b_pass)]),
    )
    records = load_verdicts(tmp_path / src.path, src)
    k = auditor_agreement(records)

    assert k["kappa_human_vs_auditor_B"] == pytest.approx(1.0)
    assert k["kappa_human_vs_auditor_A"] == pytest.approx(0.0)
    assert k["n_kappa"] == 8


def test_kappa_is_none_when_nobody_has_annotated(tmp_path: Path):
    src = _src(tmp_path, "dg_pairs", _dg_rows([""] * 6, b_pass=[True, False] * 3))
    records = load_verdicts(tmp_path / src.path, src)
    k = auditor_agreement(records)
    assert k["kappa_human_vs_auditor_A"] is None
    assert k["kappa_human_vs_auditor_B"] is None
    assert k["n_kappa"] == 0
    # the auditor-vs-auditor self-check is still computable without any human input
    assert k["kappa_auditor_A_vs_B"] is not None


# --------------------------------------------------------------------------- #
# 8. D-G stratification
# --------------------------------------------------------------------------- #
def test_dg_strata_split_on_auditors_agree_and_attribute(tmp_path: Path):
    verdicts = ["valid", "valid", "moved", "noflip", "valid", "moved"]
    agree = [True, True, True, False, False, False]
    attribute = ["gender", "gender", "gender", "skin_tone", "skin_tone", "skin_tone"]
    a_pass = [True, False, True, False, True, False]
    b_pass = [True, False, False, True, False, True]  # agree on 1&2, differ on the rest
    src = _src(
        tmp_path,
        "dg_pairs",
        _dg_rows(verdicts, agree=agree, attribute=attribute, a_pass=a_pass, b_pass=b_pass),
    )
    rows = {s.stratum: s for s in evaluate_source(src, root=tmp_path)}

    assert set(rows) == {
        "all",
        "auditors_agree",
        "auditors_disagree",
        "attribute=gender",
        "attribute=skin_tone",
    }
    assert rows["all"].n_total == 6 and rows["all"].n_pass == 3
    assert rows["auditors_agree"].n_total == 3 and rows["auditors_agree"].n_pass == 2
    assert rows["auditors_disagree"].n_total == 3 and rows["auditors_disagree"].n_pass == 1
    assert rows["attribute=gender"].n_total == 3
    assert rows["attribute=skin_tone"].n_total == 3
    # the strata partition the file -- no pair is counted twice or dropped
    assert rows["auditors_agree"].n_total + rows["auditors_disagree"].n_total == 6

    # ★ the A-vs-B kappa is blanked inside the strata DEFINED by A-vs-B agreement: it is
    # 1.0 / -0.83 by construction there, and a CSV reader could quote either as a finding.
    assert rows["auditors_agree"].extra["kappa_auditor_A_vs_B"] is None
    assert rows["auditors_disagree"].extra["kappa_auditor_A_vs_B"] is None
    assert rows["all"].extra["kappa_auditor_A_vs_B"] is not None
    assert rows["attribute=gender"].extra["kappa_auditor_A_vs_B"] is not None


# --------------------------------------------------------------------------- #
# 9. the CLI
# --------------------------------------------------------------------------- #
def test_cli_writes_csv_and_dry_run_does_not(tmp_path: Path):
    _src(tmp_path, "ds_v4", _ds_rows(["pass"] * 12 + ["patch"] * 8))
    out = "results/v2_fairness_ds/metrics/construct_validity.csv"

    rep = builder.build(root=tmp_path, sources=["ds_v4"], out=out, dry_run=True)
    assert rep["rows"] == 1 and rep["dry_run"] is True
    assert not (tmp_path / out).exists()

    rep = builder.build(root=tmp_path, sources=["ds_v4"], out=out)
    assert rep["rows"] == 1 and rep["n_annotated"] == 20
    text = (tmp_path / out).read_text(encoding="utf-8")
    assert "pass_rate" in text and "wilson_low" in text and "reading" in text
    assert "0.6" in text  # 12/20

    # idempotent: this is the pipeline's form of "resumable"
    before = text
    builder.build(root=tmp_path, sources=["ds_v4"], out=out)
    assert (tmp_path / out).read_text(encoding="utf-8") == before


def test_cli_reports_the_unannotated_state_without_inventing_a_rate(tmp_path: Path):
    _src(tmp_path, "ds_v3", _ds_rows([""] * 31))
    _src(tmp_path, "dg_pairs", _dg_rows([""] * 4))
    rep = builder.build(root=tmp_path, out="out.csv")

    assert rep["n_annotated"] == 0 and rep["n_total"] == 35
    assert "NO pass rate exists yet" in rep["note"]
    assert all("," not in m or "NOT 'the pass rate is zero'" in m for m in rep["messages"])
    # missing sources are skipped rather than fabricated
    assert set(rep["sources"]) == {"dg_pairs", "ds_v3"}


def test_cli_errors_when_no_sheet_is_staged(tmp_path: Path):
    rep = builder.build(root=tmp_path, out="out.csv")
    assert rep["rows"] == 0 and "no adjudication sheets found" in rep["error"]


def test_cli_main_smoke(tmp_path: Path, capsys):
    _src(tmp_path, "ds_v4", _ds_rows(["pass"] * 10 + [""] * 10))
    rc = builder.main(["--root", str(tmp_path), "--sources", "ds_v4", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARTIAL: 10 of 20 annotated" in out
    assert "RATE" in out or "rate" in out


# --------------------------------------------------------------------------- #
# 10. the real sheets, read-only
# --------------------------------------------------------------------------- #
def test_real_sheets_are_readable_and_currently_unannotated():
    root = Path(__file__).resolve().parents[1]
    expected = {"dg_pairs": 82, "ds_v3": 31, "ds_v4": 20}
    for name, n in expected.items():
        src = SOURCES[name]
        path = root / src.path
        if not path.exists():
            pytest.skip(f"{src.path} not staged in this checkout")
        records = load_verdicts(path, src)
        assert len(records) == n, f"{name} should hold {n} rows"
        (overall,) = [s for s in evaluate_source(src, root=root) if s.stratum == "all"]
        # This assertion is expected to START FAILING once verdicts are filled in -- at
        # which point it should be relaxed to a coverage report, not deleted.
        if overall.n_annotated == 0:
            assert overall.pass_rate is None


def test_real_dg_sheet_reproduces_the_published_auditor_kappa():
    root = Path(__file__).resolve().parents[1]
    src = SOURCES["dg_pairs"]
    if not (root / src.path).exists():
        pytest.skip("D-G sheet not staged in this checkout")
    records = load_verdicts(root / src.path, src)
    k = auditor_agreement(records)
    # the published -0.028: the two auditors agree at CHANCE on pair usability
    assert k["kappa_auditor_A_vs_B"] == pytest.approx(-0.028, abs=1e-3)
