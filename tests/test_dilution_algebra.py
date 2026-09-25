"""Guards for the WP-A1f dilution algebra.

The paper argues the D-S null is not an artefact of invalid scenes. The argument is four
lines of arithmetic on three measured inputs, and until now it lived only in prose —
the same shape as an earlier SD-denominator error that had to be corrected, where a
number the tables did not carry was re-derived in a sentence and derived wrong.

Three ways this can go wrong, all of them producing a table that looks right:

1. **phi is read as the PASS rate instead of the failure rate.** `1/(1-0.85)` and
   `1/(1-0.15)` differ by a factor of 5.7, and both are plausible-looking numbers.
2. **the binding case becomes the mean gap instead of the largest.** Dilution has to
   explain the BIGGEST observed gap; averaging makes the argument look stronger than
   the data support.
3. **the two SD denominators get mixed again.** `mde_sd` is in SD(paired difference);
   a gap standardised by SD(score) is a different quantity and roughly doubles the
   ratio.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from edit_judge_bias.experiments.build_dilution_algebra import (
    construct_rates,
    dilution_rows,
    ita_dose,
)

REPO = Path(__file__).resolve().parents[1]
DS_METRICS = REPO / "results" / "v2_fairness_ds" / "metrics"
MANIFEST = REPO / "data" / "manifests" / "samples_fairness_ds_judge_v4.jsonl"

on_disk = pytest.mark.skipif(
    not (DS_METRICS / "attribute_gaps.csv").exists(), reason="D-S tables not on disk"
)


def _toy_metrics(tmp_path: Path) -> Path:
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True)
    with (metrics / "attribute_gaps.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "judge_model", "attribute", "n", "mean_gap", "sd_gap", "gap_in_sd_units",
            "mde_sd",
        ])
        w.writeheader()
        w.writerows([
            # a small gap and a large one, so the binding case has to be chosen
            {"judge_model": "small", "attribute": "skin_tone", "n": "100",
             "mean_gap": "0.04", "sd_gap": "2.0", "gap_in_sd_units": "0.02",
             "mde_sd": "0.10"},
            {"judge_model": "large", "attribute": "skin_tone", "n": "100",
             "mean_gap": "-0.10", "sd_gap": "2.0", "gap_in_sd_units": "-0.05",
             "mde_sd": "0.10"},
            # a different family must not leak in
            {"judge_model": "small", "attribute": "dose_control", "n": "100",
             "mean_gap": "0.90", "sd_gap": "2.0", "gap_in_sd_units": "0.45",
             "mde_sd": "0.12"},
        ])
    return metrics


def test_phi_is_the_failure_rate_not_the_pass_rate(tmp_path):
    """80 of 100 pass -> phi = 0.20. Reading it as 0.80 inflates the algebra 4x."""
    metrics = _toy_metrics(tmp_path)
    construct = {"phi_observed": 0.20, "phi_wilson_high": 0.30}
    rows = {r["judge_model"]: r for r in dilution_rows(metrics, construct=construct)}
    # observed 0.05, phi 0.20 -> true = 0.05 / 0.80 = 0.0625
    assert rows["large"]["implied_true_at_phi_observed"] == pytest.approx(0.0625)
    assert rows["large"]["implied_true_at_phi_high"] == pytest.approx(0.05 / 0.70, abs=1e-6)


def test_breakeven_phi_is_where_the_diluted_observation_reaches_the_mde(tmp_path):
    metrics = _toy_metrics(tmp_path)
    rows = {r["judge_model"]: r
            for r in dilution_rows(metrics, construct={"phi_observed": 0.2,
                                                       "phi_wilson_high": 0.3})}
    # 0.05 / (1 - phi) = 0.10  ->  phi = 0.5
    assert rows["large"]["phi_breakeven"] == pytest.approx(0.5)
    # 0.02 / (1 - phi) = 0.10  ->  phi = 0.8
    assert rows["small"]["phi_breakeven"] == pytest.approx(0.8)
    assert rows["large"]["breakeven_outside_wilson_high"] is True


def test_the_binding_case_is_the_largest_gap_not_the_average(tmp_path):
    metrics = _toy_metrics(tmp_path)
    rows = dilution_rows(metrics, construct={"phi_observed": 0.2, "phi_wilson_high": 0.3})
    binding = [r for r in rows if r["scope"] == "binding_case"]
    assert len(binding) == 1
    assert "large" in binding[0]["judge_model"]
    assert binding[0]["abs_gap_in_sd_units"] == pytest.approx(0.05)


def test_only_the_named_attribute_family_enters(tmp_path):
    metrics = _toy_metrics(tmp_path)
    rows = dilution_rows(metrics, construct={"phi_observed": 0.2, "phi_wilson_high": 0.3})
    assert {r["attribute"] for r in rows} == {"skin_tone"}
    assert len(rows) == 3  # two judges plus the binding-case summary


def test_a_scene_missing_one_arm_contributes_no_separation(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rows = [
        {"metadata": {"base_sample_id": "a", "variant_label": "dark",
                      "achieved_delta_ita": -30.0}},
        {"metadata": {"base_sample_id": "a", "variant_label": "light",
                      "achieved_delta_ita": 31.0}},
        {"metadata": {"base_sample_id": "b", "variant_label": "dark",
                      "achieved_delta_ita": -28.0}},   # no light arm
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dose = ita_dose(manifest)
    assert dose["n_scenes"] == 1
    assert dose["separation_median"] == pytest.approx(61.0)


# --------------------------------------------------------------------------- #
# the frozen artefacts                                                         #
# --------------------------------------------------------------------------- #
@on_disk
def test_reproduces_every_number_section_6_5_a_prints():
    """0.051 / 2.02x / 0.36 / 0.068 / 1.52x / 0.579 — now from the frozen tables.

    The binding case is `qwen3.5-plus`, the judge with the largest |gap|. If any of
    these moves, §6.5(a) has to be rewritten, so they are pinned here rather than
    remembered.
    """
    construct = construct_rates(DS_METRICS)
    assert construct["phi_observed"] == pytest.approx(0.15)
    assert construct["phi_wilson_high"] == pytest.approx(0.36, abs=5e-3)

    binding = next(r for r in dilution_rows(DS_METRICS) if r["scope"] == "binding_case")
    assert "qwen3.5-plus" in binding["judge_model"]
    assert binding["implied_true_at_phi_observed"] == pytest.approx(0.051, abs=5e-4)
    assert binding["mde_over_implied_at_phi_observed"] == pytest.approx(2.02, abs=5e-3)
    assert binding["implied_true_at_phi_high"] == pytest.approx(0.068, abs=5e-4)
    assert binding["mde_over_implied_at_phi_high"] == pytest.approx(1.52, abs=5e-3)
    assert binding["phi_breakeven"] == pytest.approx(0.579, abs=5e-4)
    assert binding["breakeven_outside_wilson_high"] is True


@on_disk
def test_the_gap_and_the_mde_come_from_the_same_row_in_the_same_units():
    """Both sides in SD(paired difference). Mixing denominators doubled a ratio once."""
    with (DS_METRICS / "attribute_gaps.csv").open(encoding="utf-8", newline="") as fh:
        gaps = {r["judge_model"]: r for r in csv.DictReader(fh)
                if r["attribute"] == "skin_tone"}
    for row in dilution_rows(DS_METRICS):
        if row["scope"] != "per_judge":
            continue
        source = gaps[row["judge_model"]]
        assert row["abs_gap_in_sd_units"] == pytest.approx(
            abs(float(source["gap_in_sd_units"])), abs=1e-9)
        assert row["mde_sd"] == pytest.approx(float(source["mde_sd"]), abs=1e-9)
        # and the gap in SD units really is mean_gap / sd_gap, not mean_gap / SD(score)
        assert row["abs_gap_in_sd_units"] == pytest.approx(
            abs(float(source["mean_gap"]) / float(source["sd_gap"])), abs=1e-3)


@on_disk
def test_the_dose_is_not_diluted_and_the_published_two_numbers_are_corrected():
    """61.3 is the mean not the median, and nothing is below 10 deg.

    Both corrections make §6.5(a) stronger, and both are measurements rather than
    arguments — which is why they get a test instead of a footnote.
    """
    dose = ita_dose(MANIFEST)
    assert dose["n_scenes"] == 743
    assert dose["separation_mean"] == pytest.approx(61.263, abs=5e-3)
    assert dose["separation_median"] == pytest.approx(61.196, abs=5e-3)
    assert dose["separation_mean"] != dose["separation_median"]
    assert dose["n_below_weak_threshold"] == 0
    # zero by construction: the per-arm floor is half the two-sided separation
    assert dose["separation_min"] >= 2 * dose["gate_floor_per_arm_degrees"]
