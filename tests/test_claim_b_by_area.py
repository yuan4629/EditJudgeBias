"""WP-F1c — `region_annotation`'s box geometry, and claim B stratified by it.

The defect this table exists for: the injector estimates the edit region as the largest
connected component of |edited − original|, so an editor that re-renders the whole frame
gets a box around the whole frame. Box SIZE therefore carries information about edit
QUALITY, in the very cells claim B is measured on.

Three things are pinned here.

**That the stratifier is measuring the published estimand.** Its unstratified row must
reproduce `claim_b.csv` field for field. Without that, a stratified row differing from
the headline would be uninterpretable — is it the stratum, or is it a second, subtly
different pipeline?

**The confound itself**, as a number with a turn-clustered interval: bigger boxes go to
worse edits.

**What the stratification shows and what it must not be read to show.** The decay is
concentrated where the box is a box; where it covers the frame, it mostly vanishes. That
does NOT license "large boxes are harmless" — a frame-filling border is a different
stimulus, not a smaller dose — and it does not license the pooled Δρ being read as a
distance-to-edit-region ordering, which is the sentence it retires.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from edit_judge_bias.experiments.build_claim_b_by_area import (
    CUE,
    SCHEMES,
    load_area_fracs,
)

METRICS = Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"


def _frozen(name: str):
    path = METRICS / name
    if not path.exists():
        pytest.skip(f"{name} not built")
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


# --------------------------------------------------------------------------- #
# the stratifier measures the published estimand
# --------------------------------------------------------------------------- #
def test_the_unstratified_row_reproduces_claim_b_field_for_field():
    """The `scheme=none` rows go through this module's own join, filter and bootstrap,
    and must land on `claim_b.csv`'s numbers exactly. That equality is what makes every
    stratified row in the file comparable to the headline."""
    strata = [r for r in _frozen("claim_b_by_area_stratum.csv") if r["scheme"] == "none"]
    published = [r for r in _frozen("claim_b.csv") if r["bias_type"] == CUE]
    assert len(strata) == len(published) == 10
    idx = {(r["judge_model"], r["anchor_source"]): r for r in strata}
    fields = ("n_items", "n_paired", "n_clusters", "spearman_original",
              "spearman_biased", "spearman_delta", "spearman_delta_ci_low",
              "spearman_delta_ci_high", "accuracy_delta", "accuracy_p_cluster",
              "mcnemar_b", "mcnemar_c")
    for p in published:
        r = idx[(p["judge_model"], p["anchor_source"])]
        for k in fields:
            assert p[k] == r[k], (p["judge_model"], p["anchor_source"], k)


def test_the_unstratified_rows_carry_no_q_of_their_own():
    """They restate a hypothesis `claim_b.csv` already corrects. Two q-values for one
    hypothesis is the defect that split the ensemble table's member rows out of its
    family, and it would be the same defect here."""
    rows = [r for r in _frozen("claim_b_by_area_stratum.csv") if r["scheme"] == "none"]
    assert {r["family"] for r in rows} == {"reference (claim_b.csv)"}
    assert all(r["q_value"] == "" for r in rows)


# --------------------------------------------------------------------------- #
# the confound
# --------------------------------------------------------------------------- #
def test_worse_edits_get_bigger_boxes_on_both_anchors():
    """The confound, measured rather than asserted: box area is negatively rank-
    correlated with the human score, and the interval that says so is resampled over
    TURNS, because the 8 editors of one turn share an original image."""
    rows = _frozen("anchor_area_frac_distribution.csv")
    assert {r["anchor_source"] for r in rows} == {"EBench-18K", "ImagenHub"}
    for r in rows:
        assert float(r["spearman_area_vs_human"]) < 0
        assert r["spearman_ci_excludes_zero"] == "True"
        # and the distribution the paper never reported: a third of the boxes are
        # essentially the whole frame
        assert float(r["share_gt_0.9"]) > 0.30


def test_the_unclustered_p_is_named_as_such():
    """Same discipline as `mcnemar_p_unclustered`: an unclustered p over within-turn
    items is anti-conservative here, so it may be reported but may never be the column
    a claim rests on."""
    rows = _frozen("anchor_area_frac_distribution.csv")
    for r in rows:
        assert "spearman_p_unclustered" in r
        assert r["spearman_p_unclustered"] not in (None, "")


# --------------------------------------------------------------------------- #
# what the stratification shows
# --------------------------------------------------------------------------- #
def _excluding_zero(rows, scheme, stratum):
    cells = [r for r in rows if r["scheme"] == scheme and r["stratum"] == stratum]
    return sum(1 for r in cells if r["rho_ci_excludes_zero"] == "True"), len(cells)


def test_the_headline_survives_inside_a_stratum_but_the_pooled_ordering_does_not():
    """Both halves of the finding, and they point in different directions:

    - the decay is still there on small boxes (7 of 10 cells, against 8 of 10 pooled,
      on 40% fewer items and fragmented clusters) — so claim B does not rest on the
      confound;
    - and it is nearly gone on frame-filling boxes (2 of 10), so the pooled estimate
      averages two very different regimes and cannot be read as a statement about how
      close the perturbation sits to the edit region.
    """
    rows = _frozen("claim_b_by_area_stratum.csv")
    small = _excluding_zero(rows, "binary", "<0.5")
    large = _excluding_zero(rows, "binary", ">=0.5")
    pooled = _excluding_zero(rows, "none", "all")
    assert pooled == (8, 10)
    assert small[0] >= 6 and small[1] == 10
    assert large[0] <= 3
    assert small[0] > large[0]
    # the four-way split shows it as a gradient, ending at zero on the frame-filling band
    assert _excluding_zero(rows, "quartet", ">=0.9")[0] == 0


def test_every_stratified_row_reports_what_stratifying_cost_it():
    """Strata cut turns apart — a turn is 8 editors of one image and they land in
    different bands — so a stratified interval is wider than the pooled one for reasons
    that are not about the effect. Each row must carry its cluster count and how many
    of its clusters are still whole turns, or that cost is invisible."""
    rows = _frozen("claim_b_by_area_stratum.csv")
    for r in rows:
        assert r["n_clusters"] and r["mean_cluster_size"]
        assert r["share_whole_turns"] != ""
    # the unstratified rows keep whole turns; the stratified ones almost never do
    whole = {r["scheme"]: float(r["share_whole_turns"]) for r in rows}
    assert min(float(r["share_whole_turns"]) for r in rows
               if r["scheme"] == "none") >= 0.9
    assert max(float(r["share_whole_turns"]) for r in rows
               if r["scheme"] != "none") < 0.2, whole


def test_both_stratification_schemes_are_published():
    """Reporting only the cut that came out better would be choosing it after seeing
    the answer. Both are in the file, and the four-way one is the reason the binary
    result is not a threshold artefact."""
    rows = _frozen("claim_b_by_area_stratum.csv")
    assert {r["scheme"] for r in rows} == set(SCHEMES) | {"none"}
    for scheme, bands in SCHEMES.items():
        got = {r["stratum"] for r in rows if r["scheme"] == scheme}
        assert got == {name for name, _lo, _hi in bands}


# --------------------------------------------------------------------------- #
# the geometry is read, never re-derived
# --------------------------------------------------------------------------- #
def test_area_is_read_from_the_manifest_the_injector_wrote(tmp_path: Path):
    """Re-deriving the box from the images would measure today's estimator against
    yesterday's stimuli. The geometry comes from `bias_params`, and only from the
    region_annotation rows — every other cue's params have no `area_frac` and must not
    be swept in under some other key."""
    manifest = tmp_path / "biased.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in [
        {"base_sample_id": "a", "bias_type": CUE,
         "bias_params": {"area_frac": 0.5, "region_method": "diff"}},
        {"base_sample_id": "b", "bias_type": CUE,
         "bias_params": {"area_frac": 0.9, "region_method": "mask"}},
        {"base_sample_id": "c", "bias_type": "padding",
         "bias_params": {"padding_ratio": 0.1}},
        {"base_sample_id": "d", "bias_type": CUE, "bias_params": {}},
    ]) + "\n", encoding="utf-8")
    area, methods = load_area_fracs(manifest)
    assert area == {"a": 0.5, "b": 0.9}
    assert methods == {"diff": 1, "mask": 1}
