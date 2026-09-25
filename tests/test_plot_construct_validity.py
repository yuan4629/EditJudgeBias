"""Tests for F10 — the construct-validity figure.

Each of these pins something that is INVISIBLE in the rendered PNG but would ship a wrong
claim if it broke:

1. the interval is the stored **Wilson** interval, drawn as two endpoints. A symmetric
   normal-approximation interval populates the same two columns and looks perfectly
   plausible on the page — at ``28/31`` it puts the lower endpoint at 0.799 instead of
   0.751 and the upper one *above 1.0* — so the figure recomputes Wilson and refuses to
   draw anything that does not match;
2. the three sources never share an axis. They are all proportions, so nothing about the
   units stops a reader ranking them; only the question differs, and the question is not
   in the number;
3. the kappa signs. ``kappa(human, B) > 0 > kappa(human, A)`` is the whole right panel; a
   sign flip would reverse the paper's conclusion about which auditor the human backs and
   the figure would look exactly as convincing;
4. the deliberately-blank ``kappa_auditor_A_vs_B`` inside the agreement-defined strata
   stays blank. It is +-1 *by construction* there, and a figure that filled it in would be
   quotable as a finding;
5. a missing column raises instead of drawing a silently empty row (an empty row in a
   7-row figure reads as "zero");
6. an unannotated source is drawn as "no rate", never as 0.0;
7. rendering is byte-identical on a re-run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from edit_judge_bias.visualization.plot_construct_validity import (
    KAPPA_SERIES,
    auditor_marginals,
    build_blocks,
    census_line,
    check_columns,
    check_one_vocabulary_per_axis,
    estimates,
    kappa_marks,
    plot_construct_validity,
)
from edit_judge_bias.visualization.style import wilson

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "v2_fairness_ds" / "metrics" / "construct_validity.csv"
SHEET = ROOT / "data" / "human_validation_fairness" / "pairs.jsonl"


# --------------------------------------------------------------------------- #
# fixtures — miniature versions of the frozen table
# --------------------------------------------------------------------------- #
def _row(source, vocabulary, stratum, n_pass, n_ann, labels, *, arm="D-G",
         n_total=None, counts=None, **extra):
    """One CSV-shaped row with a *correct* Wilson interval, so tests start from valid data."""
    lo, hi = wilson(n_pass, n_ann) if n_ann else (None, None)
    row = {
        "source": source,
        "arm": arm,
        "stratum": stratum,
        "vocabulary": vocabulary,
        "status": "complete",
        "n_total": str(n_total if n_total is not None else n_ann),
        "n_annotated": str(n_ann),
        "n_pass": str(n_pass),
        "pass_rate": "" if not n_ann else f"{n_pass / n_ann:.6f}",
        "wilson_low": "" if not n_ann else f"{lo:.6f}",
        "wilson_high": "" if not n_ann else f"{hi:.6f}",
        "n_unknown": "0",
        "unknown_labels": "",
        "n_flagged": "0",
        "pass_rate_excl_flagged": "",
    }
    for label in labels:
        row[f"count_{label}"] = str((counts or {}).get(label, 0))
    row.update(extra)
    return row


DG_LABELS = ("valid", "moved", "noflip", "noperson")
V4_LABELS = ("pass", "patch", "weak", "figure")


def _dg_rows():
    """The five D-G strata, with the real counts and the real kappas."""
    return [
        _row("dg_pairs", "dg_pairs", "all", 42, 82, DG_LABELS,
             counts=dict(valid=42, moved=25, noflip=1, noperson=14),
             kappa_human_vs_auditor_A="0.075919",
             kappa_human_vs_auditor_B="0.468787",
             kappa_auditor_A_vs_B="-0.027569", n_kappa="82"),
        _row("dg_pairs", "dg_pairs", "auditors_agree", 18, 42, DG_LABELS,
             kappa_human_vs_auditor_A="0.485294",
             kappa_human_vs_auditor_B="0.485294",
             kappa_auditor_A_vs_B="", n_kappa="42"),
        _row("dg_pairs", "dg_pairs", "auditors_disagree", 24, 40, DG_LABELS,
             kappa_human_vs_auditor_A="-0.489362",
             kappa_human_vs_auditor_B="0.433962",
             kappa_auditor_A_vs_B="", n_kappa="40"),
    ]


def _ds_rows():
    return [
        _row("ds_v3", "ds_v3", "all", 28, 31,
             ("plausible", "recoloured", "leaked", "no_person"), arm="D-S",
             counts=dict(plausible=28, recoloured=3)),
        _row("ds_v4", "ds_v4", "all", 17, 20, V4_LABELS, arm="D-S",
             counts=dict(pass_=0, patch=1, weak=2, figure=0)),
    ]


def _all_rows():
    return _dg_rows() + _ds_rows()


def _sheet(tmp_path: Path) -> Path:
    """A miniature D-G sheet: 6 pairs with a known human/A/B pattern."""
    rows = [
        # human, A, B  -- three contested pairs (A != B) and three concurring ones
        ("valid", True, False),     # contested, human sides with A
        ("moved", True, False),     # contested, human sides with B
        ("valid", False, True),     # contested, human sides with B
        ("valid", True, True),      # concurring
        ("moved", False, False),    # concurring
        ("noperson", True, True),   # concurring; A and B both pass what the human rejects
    ]
    path = tmp_path / "pairs.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, (verdict, a, b) in enumerate(rows):
            fh.write(json.dumps({
                "pair_key": f"p{i}", "attribute": "gender",
                "auditor_A_passed": a, "auditor_B_passed": b,
                "auditors_agree": a == b, "human_verdict": verdict}) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 1. the interval — endpoints, and Wilson specifically
# --------------------------------------------------------------------------- #
def test_the_interval_is_asymmetric_and_is_drawn_as_endpoints():
    block = build_blocks(_ds_rows())[0]
    est = estimates(block)[0]
    assert (est.n_pass, est.n_annotated) == (28, 31)
    # the endpoints are absolute positions, not a half-width around the point
    assert est.lo < est.rate < est.hi
    below, above = est.rate - est.lo, est.hi - est.rate
    assert above == pytest.approx(0.064, abs=5e-3)
    assert below == pytest.approx(0.152, abs=5e-3)
    # a symmetric bar would be wrong by more than a third of the interval's own width
    assert abs(below - above) > 0.3 * (est.hi - est.lo)


def test_estimates_refuse_a_symmetric_normal_approximation_interval():
    """The one way this figure could silently go wrong: a table that switched to +-1.96 SE.

    The normal approximation fills exactly the same two columns and, at 28/31, puts the
    upper endpoint ABOVE 1.0 — an impossible pass rate that a bar chart would happily draw.
    """
    row = _row("ds_v3", "ds_v3", "all", 28, 31,
               ("plausible", "recoloured", "leaked", "no_person"), arm="D-S")
    p, n = 28 / 31, 31
    half = 1.959963985 * math.sqrt(p * (1 - p) / n)
    row["wilson_low"], row["wilson_high"] = f"{p - half:.6f}", f"{p + half:.6f}"
    with pytest.raises(ValueError, match="not the Wilson interval"):
        estimates(build_blocks([row])[0])


def test_estimates_refuse_a_rate_that_arrives_without_its_interval():
    row = _row("ds_v4", "ds_v4", "all", 17, 20, V4_LABELS, arm="D-S")
    row["wilson_low"] = ""
    with pytest.raises(ValueError, match="wilson_low and wilson_high"):
        estimates(build_blocks([row])[0])


def test_an_unannotated_source_has_no_rate_rather_than_a_zero():
    """`nobody has looked yet` and `every sheet failed` must not be the same pixel."""
    row = _row("ds_v4", "ds_v4", "all", 0, 0, V4_LABELS, arm="D-S", n_total=20)
    est = estimates(build_blocks([row])[0])[0]
    assert est.rate is None and est.lo is None and est.hi is None
    assert est.annotated is False
    assert est.n_total == 20


# --------------------------------------------------------------------------- #
# 2. three sources, three rulers
# --------------------------------------------------------------------------- #
def test_each_source_gets_its_own_axis_with_its_own_question_and_label_set():
    blocks = build_blocks(_all_rows())
    assert [b.source for b in blocks] == ["dg_pairs", "ds_v3", "ds_v4"]
    # a different question per block is the whole reason they are not one axis
    assert len({b.question for b in blocks}) == 2      # ds_v3 and ds_v4 share a question
    assert len({b.labels for b in blocks}) == 3        # but never a label set
    assert blocks[0].labels == DG_LABELS
    assert blocks[2].labels == V4_LABELS
    # `all` is always the first (emphasised) row of its block
    assert [r["stratum"] for r in blocks[0].rows][0] == "all"
    assert len(blocks[0].rows) == 3


def test_two_vocabularies_can_never_be_routed_onto_one_axis():
    mixed = [_dg_rows()[0], dict(_ds_rows()[0], source="dg_pairs")]
    with pytest.raises(ValueError, match="two vocabularies on one axis"):
        check_one_vocabulary_per_axis(mixed, "dg_pairs")
    with pytest.raises(ValueError, match="two vocabularies on one axis"):
        build_blocks(mixed)


def test_an_unknown_vocabulary_raises_rather_than_being_drawn_unlabelled():
    row = _row("new_arm", "ds_v9", "all", 1, 2, ("a", "b"))
    with pytest.raises(ValueError, match="unknown vocabulary"):
        build_blocks([row])


def test_a_missing_column_raises_instead_of_drawing_a_blank_row():
    rows = [dict(r) for r in _dg_rows()]
    for r in rows:
        r.pop("wilson_high")
    with pytest.raises(ValueError, match="missing column"):
        check_columns(rows)
    with pytest.raises(ValueError, match="wilson_high"):
        build_blocks(rows)
    with pytest.raises(ValueError):
        check_columns([])


def test_the_census_line_prints_a_zero_ethics_flag():
    """`figure: 0 recorded` is a claim about the sheet and must be visible, not omitted."""
    v4 = build_blocks(_ds_rows())[1]
    line = census_line(v4)
    assert "patch 1" in line and "weak 2" in line
    assert "0 recorded" in line and "figure" in line
    # the flag is NOT counted among the construct failures
    assert line.startswith("3 failures")


# --------------------------------------------------------------------------- #
# 3. the kappas — sign, stagger, and the blank that must stay blank
# --------------------------------------------------------------------------- #
def test_kappa_signs_point_at_auditor_b_in_every_stratum():
    marks = kappa_marks(_dg_rows())
    by = {(m.stratum, m.series): m.value for m in marks}
    for stratum in ("all", "auditors_agree", "auditors_disagree"):
        a = by[(stratum, "kappa_human_vs_auditor_A")]
        b = by[(stratum, "kappa_human_vs_auditor_B")]
        assert b >= a, f"{stratum}: the human should track auditor B at least as closely"
    # the contested stratum is the tie-break, and it is signed: B positive, A negative
    assert by[("auditors_disagree", "kappa_human_vs_auditor_B")] > 0
    assert by[("auditors_disagree", "kappa_human_vs_auditor_A")] < 0
    # ... and the published A-vs-B tie is reproduced on the only stratum that carries it
    assert by[("all", "kappa_auditor_A_vs_B")] == pytest.approx(-0.0276, abs=1e-3)


def test_kappa_marks_stay_visible_when_two_series_share_a_value():
    """On `auditors_agree` both kappas are +0.485; without a stagger one mark hides the other."""
    marks = [m for m in kappa_marks(_dg_rows()) if m.stratum == "auditors_agree"]
    assert len(marks) == 2
    assert marks[0].value == pytest.approx(marks[1].value)
    assert marks[0].offset != marks[1].offset
    assert {s.offset for s in KAPPA_SERIES} == {-0.21, 0.0, 0.21}


def test_the_circular_a_vs_b_kappa_is_not_drawn_inside_an_agreement_stratum():
    """It is +-1 by construction there — a property of the split, never a measurement."""
    marks = kappa_marks(_dg_rows())
    circular = [m for m in marks
                if m.series == "kappa_auditor_A_vs_B"
                and m.stratum in ("auditors_agree", "auditors_disagree")]
    assert circular == []
    assert any(m.series == "kappa_auditor_A_vs_B" and m.stratum == "all" for m in marks)


# --------------------------------------------------------------------------- #
# 4. the marginals — counted, never taken on trust
# --------------------------------------------------------------------------- #
def test_auditor_marginals_are_counted_from_the_sheet(tmp_path: Path):
    m = auditor_marginals(_sheet(tmp_path))
    assert (m.n_annotated, m.n_human, m.n_auditor_a, m.n_auditor_b) == (6, 3, 4, 3)
    assert m.contested_n == 3
    assert m.contested_with_a + m.contested_with_b == m.contested_n
    assert (m.contested_with_a, m.contested_with_b) == (1, 2)
    # the cell that separates "a conservative filter" from "a different opinion":
    # A passes both the `moved` and the `noperson` pair the human rejects, B only the
    # `noperson` one — i.e. B is the more nested of the two, as it is on the real sheet.
    assert m.a_pass_human_reject == 2
    assert m.b_pass_human_reject == 1


def test_auditor_marginals_return_none_when_the_sheet_is_absent(tmp_path: Path):
    assert auditor_marginals(tmp_path / "nope.jsonl") is None


# --------------------------------------------------------------------------- #
# 5. the figure
# --------------------------------------------------------------------------- #
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_figure_renders_and_is_md5_idempotent(tmp_path: Path):
    marginals = auditor_marginals(_sheet(tmp_path))
    out = tmp_path / "construct_validity.png"
    made = plot_construct_validity(_all_rows(), out, marginals=marginals)
    assert made == out and out.exists() and out.stat().st_size > 10_000
    first = _md5(out)
    plot_construct_validity(_all_rows(), out, marginals=marginals)
    assert _md5(out) == first


def test_figure_renders_without_the_sheet_and_says_so(tmp_path: Path):
    out = tmp_path / "no_marginals.png"
    assert plot_construct_validity(_all_rows(), out, marginals=None).exists()


def test_figure_refuses_an_empty_table(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_construct_validity([], tmp_path / "x.png")


# --------------------------------------------------------------------------- #
# 6. the real artefacts, when they are in this checkout
# --------------------------------------------------------------------------- #
def _read(path: Path):
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_frozen_table_carries_every_column_this_figure_reads():
    rows = _read(FROZEN)
    if not rows:
        pytest.skip("construct_validity.csv not present in this checkout")
    blocks = build_blocks(rows)
    assert [b.source for b in blocks] == ["dg_pairs", "ds_v3", "ds_v4"]
    # estimates() re-derives every Wilson interval, so this asserts the whole table
    for block in blocks:
        for est in estimates(block):
            assert est.annotated, f"{block.source}/{est.stratum} is not annotated"
    top = {b.source: estimates(b)[0] for b in blocks}
    assert (top["dg_pairs"].n_pass, top["dg_pairs"].n_annotated) == (42, 82)
    assert (top["ds_v3"].n_pass, top["ds_v3"].n_annotated) == (28, 31)
    assert (top["ds_v4"].n_pass, top["ds_v4"].n_annotated) == (17, 20)


def test_the_real_sheet_reproduces_the_headline_and_its_counterweight():
    """★ The two sentences the right panel must always carry together.

    Pattern: the human sides with the STRICT auditor B. Strictness: the human is
    nevertheless the most PERMISSIVE of the three. If either half ever stops holding, the
    caption is wrong and this test says so instead of the figure quietly over-claiming.
    """
    m = auditor_marginals(SHEET)
    if m is None:
        pytest.skip("D-G adjudication sheet not present in this checkout")
    assert (m.n_annotated, m.n_human, m.n_auditor_a, m.n_auditor_b) == (82, 42, 36, 24)
    # the counterweight: the human is the most permissive rater of the three
    assert m.n_human > m.n_auditor_a > m.n_auditor_b
    # the tie-break, on the pairs where exactly one auditor can be right
    assert (m.contested_n, m.contested_with_b, m.contested_with_a) == (40, 28, 12)
    # B is nearly nested inside the human read; A is not
    assert m.b_pass_human_reject == 2 and m.a_pass_human_reject == 16

    rows = _read(FROZEN)
    if not rows:
        pytest.skip("construct_validity.csv not present in this checkout")
    by = {(k.stratum, k.series): k.value
          for k in kappa_marks([r for r in rows if r["source"] == "dg_pairs"])}
    assert by[("all", "kappa_human_vs_auditor_B")] > 0 > by[("auditors_disagree",
                                                             "kappa_human_vs_auditor_A")]
    assert by[("auditors_disagree", "kappa_human_vs_auditor_B")] > 0


def test_the_rendered_figure_matches_a_fresh_render_of_the_frozen_table(tmp_path: Path):
    """`bash scripts/07_plot_figures.sh` must be a no-op on an unchanged tree."""
    rows = _read(FROZEN)
    shipped = ROOT / "results" / "v2_fairness_ds" / "figures" / "construct_validity.png"
    if not rows or not shipped.exists():
        pytest.skip("frozen table or rendered figure not present in this checkout")
    fresh = plot_construct_validity(rows, tmp_path / "construct_validity.png",
                                    marginals=auditor_marginals(SHEET))
    assert _md5(fresh) == _md5(shipped)
