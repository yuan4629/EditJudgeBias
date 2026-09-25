"""Tests for the paper figure set.

These pin four things that are invisible in a rendered PNG but would each ship a
wrong claim:

1. the pilot tree can never write a v2 basename (mock-judge data reaching a paper);
2. the claim-A forest reads ``claim_a.csv``, not ``scoring_shift.csv`` (a table with
   no CI, no BH q and no placebo bound cannot support the figure's encoding);
3. ``sham`` is drawn below a rule in its own strip (its empty q is by design, and in
   the main strip it would read as "not significant");
4. the D-S builder refuses to put two BH families on one axis.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

from edit_judge_bias.experiments.plot_results import figure_prefix, make_figures
from edit_judge_bias.visualization.plot_fairness_ds import (
    check_one_family_per_panel,
    family_group,
    plot_ds_null_vs_control,
)
from edit_judge_bias.visualization.plot_forests import (
    claim_a_layout,
    plot_claim_a_dimension_forest,
    plot_claim_a_forest,
    split_control_rows,
)
from edit_judge_bias.visualization.style import wilson, wilson_from_rate

JUDGES = ["gpt-5.5", "gemini-3.5-flash", "kimi-k2.5", "qwen3.5-plus", "gpt-4o-viescore"]


# --------------------------------------------------------------------------- #
# fixtures — miniature versions of the frozen tables
# --------------------------------------------------------------------------- #
def _claim_a_rows():
    rows = []
    for judge in JUDGES:
        for cue, shift in (("distraction", -3.2), ("brightness", -0.45),
                           ("bandwagon", 1.5), ("saturation", -0.05)):
            rows.append(dict(
                judge_model=judge, bias_type=cue, score_field="fine_score", n=611,
                mean_shift=shift, ci_low=shift - 0.2, ci_high=shift + 0.2,
                q_value="0.0001", significant_bh="True",
                inside_placebo_bound="False", family="claim_A"))
        # the control arm: no q, no significance flag, no placebo verdict — by design
        rows.append(dict(
            judge_model=judge, bias_type="sham", score_field="fine_score", n=611,
            mean_shift=0.30, ci_low=0.08, ci_high=0.53, q_value="",
            significant_bh="", inside_placebo_bound="", family="control"))
    return rows


def _ds_rows():
    rows = []
    for judge in JUDGES:
        rows.append(dict(judge_model=judge, attribute="skin_tone", n=740,
                         mean_gap=0.05, ci_low=-0.07, ci_high=0.17, q_value="0.34",
                         significant_bh="False", mde_sd="0.1025",
                         family="fairness_gap:skin_tone"))
        rows.append(dict(judge_model=judge, attribute="dose_control", n=480,
                         mean_gap=0.70, ci_low=0.48, ci_high=0.94, q_value="0.0",
                         significant_bh="True", mde_sd="0.1272",
                         family="fairness_gap:dose_control"))
        for arm, sign in (("dark", -1), ("light", -1)):
            rows.append(dict(
                judge_model=judge, attribute=f"dose_vs_placebo:{arm}", n=480,
                mean_gap=sign * 0.26, ci_low=sign * 0.44, ci_high=sign * 0.09,
                q_value="0.001", significant_bh="True", mde_sd="0.1272",
                family=f"fairness_contrast:dose_vs_placebo:{arm}"))
            rows.append(dict(
                judge_model=judge, attribute=f"nonskin_vs_skin:{arm}", n=480,
                mean_gap=0.50, ci_low=0.30, ci_high=0.72, q_value="0.0",
                significant_bh="True", mde_sd="0.1272",
                family=f"fairness_contrast:nonskin_vs_skin:{arm}"))
    return rows


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# 1. the pilot prefix
# --------------------------------------------------------------------------- #
def test_pilot_tree_gets_a_prefix_and_v2_trees_do_not(tmp_path: Path):
    assert figure_prefix(tmp_path / "results" / "metrics") == "pilot_"
    assert figure_prefix(tmp_path / "results" / "v2" / "metrics") == ""
    assert figure_prefix(tmp_path / "results" / "v2_fairness_ds" / "metrics") == ""


def test_pilot_figures_cannot_collide_with_v2_basenames(tmp_path: Path):
    """MEASURED HAZARD: the pilot archive is 1-5-scale data that still contains
    `mock-judge` and the dead `gpt-5.4-nano`, and it used to render to exactly the
    basenames the v2 grid uses. A \\includegraphics resolving to the wrong tree would
    have shipped mock-judge numbers with nothing on the page to give it away."""
    shared = [dict(judge_model="mock-judge", bias_type="padding", mean_shift=-0.4,
                   sir=0.3, ci_low=-0.6, ci_high=-0.2, n=50)]
    _write_csv(tmp_path / "results" / "metrics" / "scoring_shift.csv", shared)
    _write_csv(tmp_path / "results" / "v2" / "metrics" / "scoring_shift.csv", shared)

    pilot = make_figures(tmp_path / "results" / "metrics",
                         tmp_path / "results" / "figures", root=tmp_path)
    v2 = make_figures(tmp_path / "results" / "v2" / "metrics",
                      tmp_path / "results" / "v2" / "figures", root=tmp_path)

    assert pilot and v2
    assert all(p.name.startswith("pilot_") for p in pilot)
    assert not any(p.name.startswith("pilot_") for p in v2)
    assert not {p.name for p in pilot} & {p.name for p in v2}


# --------------------------------------------------------------------------- #
# 2. F2's source table
# --------------------------------------------------------------------------- #
def test_claim_a_forest_needs_claim_a_csv_not_scoring_shift(tmp_path: Path):
    """`scoring_shift.csv` carries the point estimate and nothing else — no CI, no BH
    q, no per-judge n, no `inside_placebo_bound`. The forest encodes all four, so it
    must be driven by the frozen claim table."""
    metrics = tmp_path / "results" / "v2" / "metrics"
    figures = tmp_path / "results" / "v2" / "figures"
    _write_csv(metrics / "scoring_shift.csv",
               [dict(judge_model=j, bias_type="padding", mean_shift=-0.4, sir=0.3)
                for j in JUDGES])

    made = {p.name for p in make_figures(metrics, figures, root=tmp_path)}
    assert "claim_a_forest.png" not in made, (
        "scoring_shift.csv alone must not produce the claim A forest"
    )

    _write_csv(metrics / "claim_a.csv", _claim_a_rows())
    made = {p.name for p in make_figures(metrics, figures, root=tmp_path)}
    assert "claim_a_forest.png" in made


def test_claim_a_forest_reads_the_columns_scoring_shift_lacks():
    """A row stripped of the claim-table columns must not silently render as an
    effect: without `significant_bh` the marker is hollow, not filled."""
    rows = _claim_a_rows()
    for r in rows:
        if r["bias_type"] == "distraction":
            r["significant_bh"] = ""
    layout = claim_a_layout(rows)
    assert "distraction" in layout.cues  # still drawn, just not as an effect


# --------------------------------------------------------------------------- #
# 3. the sham strip
# --------------------------------------------------------------------------- #
def test_sham_rows_are_split_out_of_the_claim_family():
    claim, control = split_control_rows(_claim_a_rows())
    assert {r["bias_type"] for r in control} == {"sham"}
    assert "sham" not in {r["bias_type"] for r in claim}


def test_sham_is_drawn_in_its_own_strip_below_the_rule():
    """Its q is empty BY DESIGN — sham defines the equivalence bound, so it is not
    tested against it. In the main strip five blank-q rows read as "not significant",
    i.e. as robustness that was never measured."""
    layout = claim_a_layout(_claim_a_rows())
    assert layout.rule_y is not None
    sham_y = [y for (_, cue), y in layout.y_by_key.items() if cue == "sham"]
    claim_y = [y for (_, cue), y in layout.y_by_key.items() if cue != "sham"]
    assert len(sham_y) == len(JUDGES)
    assert max(claim_y) < layout.rule_y < min(sham_y)
    control_bands = [c for c, _, _, is_ctl in layout.bands if is_ctl]
    assert control_bands == ["sham"]
    assert "sham" not in layout.cues


def test_claim_a_forest_renders(tmp_path: Path):
    out = plot_claim_a_forest(_claim_a_rows(), tmp_path / "f2.png")
    assert out.exists() and out.stat().st_size > 0


def test_claim_a_forest_refuses_a_control_only_table():
    _, control = split_control_rows(_claim_a_rows())
    with pytest.raises(ValueError, match="no non-control rows"):
        claim_a_layout(control)


# --------------------------------------------------------------------------- #
# 4. F1 must not pool BH families
# --------------------------------------------------------------------------- #
def test_family_group_strips_only_the_arm_suffix():
    assert family_group("fairness_contrast:dose_vs_placebo:dark") == \
        "fairness_contrast:dose_vs_placebo"
    assert family_group("fairness_gap:skin_tone") == "fairness_gap:skin_tone"


def test_one_panel_may_hold_the_two_arms_of_one_contrast():
    rows = [dict(family="fairness_contrast:dose_vs_placebo:dark"),
            dict(family="fairness_contrast:dose_vs_placebo:light")]
    assert check_one_family_per_panel(rows) == "fairness_contrast:dose_vs_placebo"


def test_one_panel_may_not_hold_two_contrasts():
    rows = [dict(family="fairness_gap:skin_tone"),
            dict(family="fairness_gap:dose_control")]
    with pytest.raises(ValueError, match="refusing to pool two BH families"):
        check_one_family_per_panel(rows, "A")


def test_ds_figure_raises_when_a_panel_would_pool_two_families(tmp_path: Path):
    """The null and its control are two measurements. Sharing one ruler under one
    apparent correction is the specific misreading this figure exists to prevent."""
    rows = copy.deepcopy(_ds_rows())
    for r in rows:
        if r["attribute"] == "skin_tone":
            r["family"] = "fairness_gap:dose_control"  # smuggle a second family in
            break
    with pytest.raises(ValueError, match="refusing to pool two BH families"):
        plot_ds_null_vs_control(rows, tmp_path / "f1.png")


def test_ds_figure_renders_the_four_panels(tmp_path: Path):
    out = plot_ds_null_vs_control(_ds_rows(), tmp_path / "f1.png")
    assert out.exists() and out.stat().st_size > 0


def test_ds_figure_runs_without_the_mde_reference(tmp_path: Path):
    """SD(paired difference) comes from the raw judgments, which a fresh checkout may
    not have. The MDE ticks are then simply not drawn — never drawn against
    SD(score), which would put the threshold ~3.5x too far out."""
    out = plot_ds_null_vs_control(_ds_rows(), tmp_path / "f1b.png", paired_sd={})
    assert out.exists()


# --------------------------------------------------------------------------- #
# uncertainty helper
# --------------------------------------------------------------------------- #
def test_wilson_interval_is_not_degenerate_at_the_boundaries():
    """A pass rate of 110/110 must not draw a zero-width bar — that reads as
    certainty, and the whole point of F8 is that 1.000 on n=110 is not certain."""
    lo, hi = wilson(110, 110)
    assert hi == 1.0 and 0.95 < lo < 1.0
    lo, hi = wilson(0, 110)
    assert lo == 0.0 and 0.0 < hi < 0.05


def test_wilson_from_rate_matches_wilson_from_counts():
    assert wilson_from_rate(0.9, 110) == pytest.approx(wilson(99, 110))


# --------------------------------------------------------------------------- #
# WP-A3 — the quality figure must not assert a conclusion it did not compute
# --------------------------------------------------------------------------- #

def _quality_row(validator, cue, *, rate, floor, sig_matched=False, sig_fisher=False):
    return {
        "validator_model": validator,
        "bias_type": cue,
        "mllm_pass_rate": rate,
        "control_pass_rate": floor,
        "n_mllm": 110,
        "below_floor_significant_matched": sig_matched,
        "below_floor_significant": sig_fisher,
        "floor_q_matched": 0.0195 if sig_matched else 0.9,
        "n_discordant_b": 9 if sig_matched else 1,
        "n_discordant_c": 0,
        "passes_85_gate": rate >= 0.85,
    }


def test_quality_headline_is_derived_not_asserted():
    """Until 2026-08-17 this figure's title was the hardcoded sentence 'No cue is
    significantly below its own validator's false-flag floor'. WP-A3's matched floor
    falsified it, and a hardcoded title would have shipped the falsified claim inside
    the figure while `quality_combined.csv` beside it said the opposite."""
    from edit_judge_bias.visualization.plot_quality import quality_headline

    clean = [_quality_row("v1", "a", rate=0.99, floor=0.99),
             _quality_row("v2", "a", rate=0.99, floor=0.99)]
    assert quality_headline(clean).startswith("No cue is significantly below")

    dirty = clean + [_quality_row("v1", "zoom_inset", rate=0.918, floor=0.9962,
                                  sig_matched=True),
                     _quality_row("v2", "zoom_inset", rate=0.736, floor=0.8974,
                                  sig_matched=True)]
    head = quality_headline(dirty)
    assert "zoom_inset" in head and "2 of 2 validators" in head


def test_quality_figure_marks_significance_from_the_MATCHED_column():
    """The unmatched Fisher test borrows power from images the arm never ran on (it is
    110 vs 780), so it must not drive the figure's encoding. `gpt-4o-mini x text_overlay`
    is the real cell where the two disagree: Fisher q=0.0045, matched q=0.2148."""
    from edit_judge_bias.visualization.plot_quality import (
        quality_headline,
        significant_cells,
    )

    rows = [
        _quality_row("v1", "text_overlay", rate=0.900, floor=0.9756,
                     sig_matched=False, sig_fisher=True),
        _quality_row("v1", "other", rate=0.99, floor=0.9756),
    ]
    assert significant_cells(rows) == [], "a Fisher-only cell must not be marked"
    assert quality_headline(rows).startswith("No cue is significantly below")


def test_quality_headline_reflects_the_frozen_table_if_it_exists():
    """Guards against the figure and the CSV drifting apart on disk."""
    table = Path("results/v2/metrics/quality_combined.csv")
    if not table.exists():
        pytest.skip("quality_combined.csv not built")
    from edit_judge_bias.visualization.plot_quality import (
        quality_headline,
        significant_cells,
    )

    with table.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "below_floor_significant_matched" in rows[0], (
        "the matched columns are missing — WP-A3's rebuild did not run"
    )
    sig = significant_cells(rows)
    head = quality_headline(rows)
    if sig:
        assert all(cue in head for cue in {c for _, c in sig})
    else:
        assert head.startswith("No cue is significantly below")


# --------------------------------------------------------------------------- #
# F13 — claim A by dimension (WP-F1b)                                         #
# --------------------------------------------------------------------------- #
def _dimension_rows():
    """The same cells as `_claim_a_rows`, split so the three dimensions sum to it."""
    from edit_judge_bias.experiments.build_claim_a_dimensions import DIMENSIONS
    rows = []
    for judge in JUDGES:
        for cue, parts in (("distraction", (-1.0, -1.1, -1.1)),
                           ("brightness", (-0.02, -0.23, -0.20)),
                           ("bandwagon", (0.4, 0.55, 0.55)),
                           ("saturation", (0.0, -0.05, 0.0))):
            for dim, v in zip(DIMENSIONS, parts):
                sig = "True" if abs(v) > 0.1 else ""
                rows.append(dict(
                    judge_model=judge, bias_type=cue, dimension=dim,
                    score_field=dim, n=611, mean_shift=v,
                    ci_low=v - 0.06, ci_high=v + 0.06,
                    q_value="0.0001" if sig else "0.4", significant_bh=sig,
                    family=f"claim_A_{dim}"))
        for dim in DIMENSIONS:
            rows.append(dict(
                judge_model=judge, bias_type="sham", dimension=dim, score_field=dim,
                n=611, mean_shift=0.10, ci_low=0.02, ci_high=0.18,
                q_value="", significant_bh="", family="control"))
    return rows


def test_the_three_panels_share_one_cue_order_and_one_x_axis(tmp_path: Path):
    """Both are load-bearing. The figure exists to be read ACROSS panels ("this cue
    moves detail preservation but not instruction adherence"), which a per-panel cue
    order destroys; and the instruction-adherence effects are the smallest of the
    three, which a per-panel x-axis would rescale into looking the same size as the
    detail-preservation ones — the opposite of what the data says."""
    from edit_judge_bias.experiments.build_claim_a_dimensions import DIMENSIONS
    from edit_judge_bias.visualization.plot_forests import dimension_shared_xlim

    rows = _dimension_rows()
    out = plot_claim_a_dimension_forest(rows, tmp_path / "f13.png")
    assert out.exists() and out.stat().st_size > 0

    shared = dimension_shared_xlim(rows)
    for dim in DIMENSIONS:
        panel = [r for r in rows if r["dimension"] == dim]
        own = dimension_shared_xlim(panel)
        assert shared[0] <= own[0] and own[1] <= shared[1]
    # ...and at least one panel is strictly narrower than the shared range, i.e. the
    # sharing is doing work rather than being a no-op on this table.
    assert any(dimension_shared_xlim([r for r in rows if r["dimension"] == d]) != shared
               for d in DIMENSIONS)


def test_the_dimension_panels_are_ordered_by_the_summed_effect(tmp_path: Path):
    """The shared cue order is claim A's own — cues by the SUMMED effect, i.e. by
    `claim_a.csv`'s `mean_shift` — rather than a fourth ordering nobody declared."""
    from edit_judge_bias.visualization.plot_forests import claim_a_layout
    rows = _dimension_rows()
    summed = {}
    for r in rows:
        key = (r["judge_model"], r["bias_type"])
        summed[key] = summed.get(key, 0.0) + r["mean_shift"]
    layout = claim_a_layout([{"judge_model": j, "bias_type": c, "mean_shift": v}
                             for (j, c), v in sorted(summed.items())])
    assert layout.cues[0] == "distraction"
    assert "sham" not in layout.cues            # control strip, below the rule


def test_the_dimension_figure_is_rendered_from_its_own_frozen_table(tmp_path: Path):
    """It must come from `claim_a_by_dimension.csv`, which is a decomposition of
    `claim_a.csv` on the same complete-case cells. Rendering it from anything else
    would let the two figures disagree about a cell that is by construction the same
    cell."""
    metrics = tmp_path / "results" / "v2" / "metrics"
    figures = tmp_path / "results" / "v2" / "figures"
    _write_csv(metrics / "claim_a.csv", _claim_a_rows())
    made = {p.name for p in make_figures(metrics, figures, root=tmp_path)}
    assert "claim_a_dimension_forest.png" not in made

    _write_csv(metrics / "claim_a_by_dimension.csv", _dimension_rows())
    made = {p.name for p in make_figures(metrics, figures, root=tmp_path)}
    assert "claim_a_dimension_forest.png" in made
