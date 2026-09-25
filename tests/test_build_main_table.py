"""Tests for the one-table per-cue summary (`main_table.csv`)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from edit_judge_bias.experiments.aggregate_results import PUBLISHED_ROSTER
from edit_judge_bias.experiments.build_claim_tables import MAIN_COLLECTION
from edit_judge_bias.experiments.build_fill_tables import COLLECTION, REFERENCE_LABEL
from edit_judge_bias.experiments.build_main_table import (
    BASELINE_LABEL,
    CUE_CLASS,
    build_main_table,
)

JUDGES = list(PUBLISHED_ROSTER)
MAIN = {"reference": BASELINE_LABEL, "collection": MAIN_COLLECTION}
FILL = {"reference": REFERENCE_LABEL, "collection": COLLECTION}
PW_FAMILY = "pairwise_one_sided_advantage + _accuracy"


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys or ["judge_model"], restval="")
        writer.writeheader()
        writer.writerows(rows)


def _a(judge, cue, shift, *, sig=True, inside=False, lo=None, hi=None):
    return {"judge_model": judge, "bias_type": cue, "mean_shift": shift,
            "ci_low": shift - 0.1 if lo is None else lo,
            "ci_high": shift + 0.1 if hi is None else hi,
            "significant_bh": sig, "inside_placebo_bound": inside, "family": "claim_A"}


def _pw(judge, cue, adv, sig, family=PW_FAMILY, prov=None):
    return {"judge_model": judge, "bias_type": cue, "bias_advantage": adv,
            "significant_bh_advantage": sig, **(prov or {}), "family": family}


def _cb(judge, cue, drho, moved, source="EBench-18K", family="claim_B_acc", prov=None):
    return {"judge_model": judge, "anchor_source": source, "bias_type": cue,
            "spearman_delta": drho, "rho_ci_excludes_zero": moved, **(prov or {}),
            "family": family}


def _tables(tmp_path: Path, *, claim_a, pairwise=(), claim_b=()) -> Path:
    m = tmp_path / "metrics"
    _write(m / "claim_a.csv", list(claim_a))
    _write(m / "pairwise_one_sided.csv", list(pairwise))
    _write(m / "claim_b.csv", list(claim_b))
    return m


def test_claim_a_counts_a_judge_only_when_significant_and_outside_the_bound(tmp_path: Path):
    rows = [_a(j, "padding", -1.0) for j in JUDGES[:3]]
    rows.append(_a(JUDGES[3], "padding", 0.4, inside=True))    # significant, inside the bound
    rows.append(_a(JUDGES[4], "padding", -0.2, sig=False))
    rows += [{**_a(j, "sham", 0.3), "family": "control", "significant_bh": ""} for j in JUDGES]
    [row] = build_main_table(_tables(tmp_path, claim_a=rows))
    assert row["cue"] == "padding", "the control is not a cue"
    assert row["claim_a_effect_judges"] == 3 and row["claim_a_judges"] == 5
    assert row["claim_a_direction"] == "down"
    assert row["claim_a_median_shift"] == -1.0


def test_counted_effects_whose_mean_ci_still_contains_zero_are_reported(tmp_path: Path):
    rows = [_a(j, "region_annotation", -1.0) for j in JUDGES[:4]]
    rows.append(_a(JUDGES[4], "region_annotation", -0.34, lo=-0.73, hi=0.01))
    [row] = build_main_table(_tables(tmp_path, claim_a=rows))
    assert row["claim_a_effect_judges"] == 5
    assert row["claim_a_effects_ci_crossing_zero"] == 1


def test_each_layer_names_the_baseline_its_cells_were_measured_against(tmp_path: Path):
    """One file per layer holds both collections; the reference comes from the rows."""
    claim_a = [_a(j, c, -1.0) for j in JUDGES for c in ("padding", "watermark")]
    pairwise = ([_pw(j, "padding", -0.05, True, prov=MAIN) for j in JUDGES]
                + [_pw(j, "watermark", 0.01, False, family=PW_FAMILY + " [fill]", prov=FILL)
                   for j in JUDGES])
    claim_b = ([_cb(j, "padding", -0.1, j == JUDGES[0], prov=MAIN) for j in JUDGES]
               + [_cb(j, "watermark", 0.0, False, family="claim_B_acc [fill]", prov=FILL)
                  for j in JUDGES])
    rows = {r["cue"]: r for r in build_main_table(
        _tables(tmp_path, claim_a=claim_a, pairwise=pairwise, claim_b=claim_b))}
    assert rows["padding"]["pairwise_reference"] == BASELINE_LABEL
    assert rows["watermark"]["pairwise_reference"] == REFERENCE_LABEL
    assert rows["padding"]["pairwise_significant_cells"] == 5
    assert rows["padding"]["pairwise_direction"] == "down"
    assert rows["padding"]["claim_b_rank_changed_cells"] == 1 and rows["padding"]["claim_b_cells"] == 5
    assert rows["watermark"]["claim_b_reference"] == REFERENCE_LABEL
    assert rows["watermark"]["claim_b_rank_changed_cells"] == 0


def test_a_table_without_provenance_columns_reads_as_the_main_grid(tmp_path: Path):
    claim_a = [_a(j, "padding", -1.0) for j in JUDGES]
    pairwise = [_pw(j, "padding", -0.05, True) for j in JUDGES]
    [row] = build_main_table(_tables(tmp_path, claim_a=claim_a, pairwise=pairwise))
    assert row["pairwise_reference"] == BASELINE_LABEL


def test_a_cue_with_cells_in_both_collections_is_refused(tmp_path: Path):
    claim_a = [_a(j, "padding", -1.0) for j in JUDGES]
    pairwise = ([_pw(j, "padding", -0.05, True, prov=MAIN) for j in JUDGES]
                + [_pw(j, "padding", -0.05, True, family=PW_FAMILY + " [fill]", prov=FILL)
                   for j in JUDGES])
    with pytest.raises(ValueError, match="exactly one collection"):
        build_main_table(_tables(tmp_path, claim_a=claim_a, pairwise=pairwise))


def test_a_cue_whose_cells_mix_references_is_refused(tmp_path: Path):
    claim_a = [_a(j, "padding", -1.0) for j in JUDGES]
    odd = {"reference": REFERENCE_LABEL, "collection": MAIN_COLLECTION}
    claim_b = ([_cb(j, "padding", -0.1, False, prov=MAIN) for j in JUDGES[:3]]
               + [_cb(j, "padding", -0.1, False, prov=odd) for j in JUDGES[3:]])
    with pytest.raises(ValueError, match="mixes the references"):
        build_main_table(_tables(tmp_path, claim_a=claim_a, claim_b=claim_b))


def test_a_cue_without_a_one_sided_reading_says_why(tmp_path: Path):
    claim_a = [_a(j, "model_name", 0.1, sig=False) for j in JUDGES]
    [row] = build_main_table(_tables(tmp_path, claim_a=claim_a))
    assert row["pairwise_cells"] is None
    assert "names both slots" in row["note"]


def test_a_judge_outside_the_published_roster_is_refused(tmp_path: Path):
    claim_a = [_a(j, "padding", -1.0) for j in JUDGES] + [_a("qwen3-vl-32b-instruct", "padding", -1.0)]
    with pytest.raises(ValueError, match="outside the published roster"):
        build_main_table(_tables(tmp_path, claim_a=claim_a))


def test_a_row_stamped_with_another_rosters_family_is_refused(tmp_path: Path):
    claim_a = [_a(j, "padding", -1.0) for j in JUDGES]
    pairwise = [_pw(j, "padding", -0.05, True, family=PW_FAMILY + " [a5 roster]") for j in JUDGES]
    with pytest.raises(ValueError, match="another roster"):
        build_main_table(_tables(tmp_path, claim_a=claim_a, pairwise=pairwise))


def test_rows_run_pixel_then_content_then_prompt(tmp_path: Path):
    claim_a = [_a(j, c, -1.0) for j in JUDGES for c in ("bandwagon", "zoom_inset", "padding")]
    rows = build_main_table(_tables(tmp_path, claim_a=claim_a))
    assert [r["cue_class"] for r in rows] == ["pixel", "content", "prompt"]


def test_every_claim_a_cue_has_a_declared_class():
    assert set(CUE_CLASS) == {
        "brightness", "saturation", "watermark", "text_overlay", "padding", "aesthetic_filter",
        "zoom_inset", "detail_caption", "distraction", "region_annotation",
        "bandwagon", "model_name",
    }
