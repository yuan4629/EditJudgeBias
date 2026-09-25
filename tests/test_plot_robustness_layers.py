"""Tests for F9 — the cross-layer robustness figure.

What these pin (each one is invisible in the rendered PNG but would ship a wrong
claim):

1. the three layer scalars are computed from the right rows in the right units —
   in particular ``sham`` (``family="control"``) never enters the absolute-score
   mean, because the placebo defines the equivalence bound rather than being a
   cue to resist, and the absolute layer reads the per-dimension ``asc`` (the
   Method's invariance cell), never a signed ``mean_shift`` in which opposite-sign
   changes cancel;
2. ``rank_judges`` honours the direction argument, which INVERTS between layers
   (lower |Δ| and lower |Δρ| are good, higher RR is good). Getting this wrong
   would draw one of the three axes upside down and the figure would still look
   perfectly plausible;
3. the ranking scalar uses |Δρ|, so two cues of opposite sign cannot cancel and
   report a reordering judge as a stable one;
4. the figure renders from the FROZEN tables' own column names and the crossing
   count it prints is real — if the data ever stopped crossing, the count would
   go to zero rather than the caption quietly over-claiming.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from edit_judge_bias.visualization.plot_robustness_layers import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    absolute_score_robustness,
    crossings,
    plot_robustness_layers,
    position_robustness,
    rank_judges,
    rank_table,
    ranking_robustness,
)

JUDGES = ["gpt-5.5", "gemini-3.5-flash", "kimi-k2.5", "qwen3.5-plus", "gpt-4o-viescore"]


# --------------------------------------------------------------------------- #
# fixtures — miniature versions of the three frozen tables
# --------------------------------------------------------------------------- #
def _claim_a_dim_rows():
    """Two cues × two dimensions per judge plus the control arm, with a known mean asc.

    Every cue cell has ``mean_shift`` 0.0: the per-item changes cancel inside each
    cell, so a layer that read the signed shift would call every judge perfectly
    stable.
    """
    cells = {
        "gpt-5.5": (2.0, 1.0, 1.4, 1.6),            # mean asc 1.50
        "gemini-3.5-flash": (1.5, 0.9, 1.2, 1.2),   # 1.20
        "kimi-k2.5": (1.0, 0.6, 0.8, 0.8),          # 0.80
        "qwen3.5-plus": (1.4, 0.6, 1.0, 1.0),       # 1.00
        "gpt-4o-viescore": (0.6, 0.2, 0.4, 0.4),    # 0.40  <- most robust here
    }
    dims = ("instruction_adherence", "detail_preservation")
    keys = [(cue, dim) for cue in ("distraction", "bandwagon") for dim in dims]
    rows = []
    for judge, values in cells.items():
        for (cue, dim), asc in zip(keys, values):
            rows.append(dict(judge_model=judge, bias_type=cue, dimension=dim,
                             asc=str(asc), mean_shift="0.0", family=f"claim_A_{dim}"))
        for dim in dims:
            # the placebo: a big movement that must NOT enter the mean
            rows.append(dict(judge_model=judge, bias_type="sham", dimension=dim,
                             asc="9.0", mean_shift="9.0", family="control"))
    return rows


def _claim_b_rows():
    """|Δρ| means engineered so viescore is WORST on ranking."""
    deltas = {
        "gpt-5.5": (-0.05, 0.03),          # mean |Δρ| 0.040
        "gemini-3.5-flash": (-0.04, 0.02), # 0.030
        "kimi-k2.5": (-0.02, 0.00),        # 0.010
        "qwen3.5-plus": (-0.03, 0.01),     # 0.020
        "gpt-4o-viescore": (-0.14, 0.10),  # 0.120  <- least robust here
    }
    rows = []
    for judge, (a, b) in deltas.items():
        for src, delta in (("EBench-18K", a), ("ImagenHub", b)):
            rows.append(dict(judge_model=judge, anchor_source=src,
                             bias_type="region_annotation",
                             spearman_delta=str(delta), accuracy_delta="-0.09",
                             rho_ci_excludes_zero="True", family="claim_B_acc"))
    return rows


def _position_rows():
    rr = {"gemini-3.5-flash": 0.8994, "gpt-5.5": 0.8458, "kimi-k2.5": 0.7955,
          "gpt-4o-viescore": 0.6883, "qwen3.5-plus": 0.3523}
    return [dict(judge_model=j, bias_type="position", n="616", rr=str(v))
            for j, v in rr.items()]


# --------------------------------------------------------------------------- #
# 1. the scalars
# --------------------------------------------------------------------------- #
def test_absolute_score_scalar_excludes_the_placebo_arm():
    layer = absolute_score_robustness(_claim_a_dim_rows())
    # 9.0 would dominate every mean if `sham` leaked in
    assert layer.values["gpt-5.5"] == pytest.approx(1.50)
    assert layer.values["gpt-4o-viescore"] == pytest.approx(0.40)
    assert all(n == 4 for n in layer.support.values())
    assert layer.better == LOWER_IS_BETTER
    assert "rating points" in layer.unit


def test_absolute_score_scalar_is_the_per_item_change_not_the_signed_shift():
    """Every toy cue cell has mean_shift 0.0; changes that cancel still count."""
    layer = absolute_score_robustness(_claim_a_dim_rows())
    assert min(layer.values.values()) > 0
    assert layer.values["kimi-k2.5"] == pytest.approx(0.80)


def test_absolute_score_scalar_refuses_the_summed_table():
    """`claim_a.csv` has an `asc` column too, on the 3-30 sum; it must not be read."""
    summed = [dict(judge_model="j", bias_type="padding", mean_shift="-0.5",
                   asc="1.9", family="claim_A")]
    with pytest.raises(ValueError, match="claim_a_by_dimension"):
        absolute_score_robustness(summed)


def test_ranking_scalar_is_absolute_so_opposite_signs_cannot_cancel():
    rows = [
        dict(judge_model="j", anchor_source="A", bias_type="c",
             spearman_delta="-0.30", family="claim_B_acc"),
        dict(judge_model="j", anchor_source="B", bias_type="c",
             spearman_delta="0.30", family="claim_B_acc"),
    ]
    layer = ranking_robustness(rows)
    # a signed mean would be 0.0 and would call this judge perfectly stable
    assert layer.values["j"] == pytest.approx(0.30)
    assert layer.better == LOWER_IS_BETTER


def test_position_scalar_is_rr_verbatim_and_higher_is_better():
    layer = position_robustness(_position_rows())
    assert layer.values["qwen3.5-plus"] == pytest.approx(0.3523)
    assert layer.better == HIGHER_IS_BETTER
    assert layer.support["gpt-5.5"] == 616


# --------------------------------------------------------------------------- #
# 2. ranking direction — the axis that would silently flip
# --------------------------------------------------------------------------- #
def test_rank_judges_inverts_with_the_direction_argument():
    values = {"a": 0.10, "b": 0.90}
    assert rank_judges(values, LOWER_IS_BETTER) == {"a": 1, "b": 2}
    assert rank_judges(values, HIGHER_IS_BETTER) == {"b": 1, "a": 2}


def test_rank_judges_requires_an_explicit_direction():
    with pytest.raises(ValueError):
        rank_judges({"a": 1.0}, "best")


def test_rank_judges_gives_exact_ties_the_same_rank():
    ranks = rank_judges({"a": 0.5, "b": 0.5, "c": 0.9}, LOWER_IS_BETTER)
    assert ranks["a"] == ranks["b"] == 1
    assert ranks["c"] == 3


def test_rank_table_reproduces_the_headline_crossing():
    layers = [absolute_score_robustness(_claim_a_dim_rows()),
              ranking_robustness(_claim_b_rows()),
              position_robustness(_position_rows())]
    table = rank_table(layers)
    # the whole point of the figure: best on layer 1, worst on layer 2
    assert table["gpt-4o-viescore"][0] == 1
    assert table["gpt-4o-viescore"][1] == len(table)
    assert table["gemini-3.5-flash"][2] == 1
    assert crossings(layers) > 0


# --------------------------------------------------------------------------- #
# 3. the figure
# --------------------------------------------------------------------------- #
def test_figure_renders_and_is_idempotent(tmp_path: Path):
    out = tmp_path / "robustness_layers.png"
    made = plot_robustness_layers(_claim_a_dim_rows(), _claim_b_rows(),
                                  _position_rows(), out)
    assert made == out and out.exists() and out.stat().st_size > 10_000
    first = out.read_bytes()
    plot_robustness_layers(_claim_a_dim_rows(), _claim_b_rows(), _position_rows(), out)
    assert len(out.read_bytes()) == len(first)


def test_figure_refuses_a_judge_set_with_no_common_layer(tmp_path: Path):
    a = [dict(judge_model="only-a", bias_type="x", dimension="instruction_adherence",
              asc="1.0", family="claim_A_instruction_adherence")]
    b = [dict(judge_model="only-b", anchor_source="A", bias_type="x",
              spearman_delta="0.1", family="claim_B_acc")]
    p = [dict(judge_model="only-c", bias_type="position", n="10", rr="0.5")]
    with pytest.raises(ValueError, match="all three layers"):
        plot_robustness_layers(a, b, p, tmp_path / "x.png")


# --------------------------------------------------------------------------- #
# 4. the real tables, when they are on disk
# --------------------------------------------------------------------------- #
def _read(path: Path):
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_frozen_tables_carry_every_column_this_figure_reads():
    root = Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"
    claim_a_dim, claim_b = _read(root / "claim_a_by_dimension.csv"), _read(root / "claim_b.csv")
    position, claim_a = _read(root / "position.csv"), _read(root / "claim_a.csv")
    if not (claim_a_dim and claim_b and position and claim_a):
        pytest.skip("frozen v2 metric tables not present in this checkout")
    # claim_b.csv also holds the FILL v2 rows (RESULTS 26.6); figure 5 ranks the main
    # grid's cells only, exactly as `plot_robustness_layers.main` reads them.
    from edit_judge_bias.experiments.build_claim_tables import main_grid_rows

    claim_b = main_grid_rows(claim_b)
    assert claim_b, "claim_b.csv has no main-grid rows"

    for name, rows, cols in (
        ("claim_a_by_dimension", claim_a_dim, {"judge_model", "bias_type", "dimension",
                                               "asc", "family"}),
        ("claim_b", claim_b, {"judge_model", "bias_type", "spearman_delta",
                              "accuracy_delta"}),
        ("position", position, {"judge_model", "rr", "n"}),
    ):
        assert cols <= set(rows[0]), f"{name}.csv is missing {cols - set(rows[0])}"

    layers = [absolute_score_robustness(claim_a_dim), ranking_robustness(claim_b),
              position_robustness(position)]
    assert set(JUDGES) <= set(layers[0].values)
    # the absolute-score mean is over the 36 cue x dimension cells, sham excluded
    assert set(layers[0].support.values()) == {36}
    # and the narrative the caption asserts is really in the frozen numbers
    table = rank_table(layers)
    assert table["gemini-3.5-flash"][0] == 5, "gemini should be last on the absolute layer"
    assert table["gemini-3.5-flash"][2] == 1, "gemini should be first on protocol"
    assert table["gpt-4o-viescore"][1] == 5, "viescore should be last on ranking"
    assert crossings(layers) == 8
    # the caveat line: sizing each cue's AVERAGE signed change instead puts viescore first
    signed: dict = {}
    for r in claim_a:
        if r["family"] == "claim_A":
            signed.setdefault(r["judge_model"], []).append(abs(float(r["mean_shift"])))
    assert min(signed, key=lambda j: sum(signed[j]) / len(signed[j])) == "gpt-4o-viescore"
