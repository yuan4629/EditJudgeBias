"""Guards for the WP-A1 sensitivity tables.

The A1a table exists to answer "does the headline survive if the suspect stimuli are
deleted", so it is worth exactly what its stimulus bookkeeping is worth. Three things
can go silently wrong and each produces a plausible-looking CSV:

1. the violation count is taken off a reconstructed *canvas* rather than the pixels the
   injector actually changed (this really happened — it is the 22-vs-21 correction);
2. the exclusion is applied to the biased rows but not the baselines, so `mean_shift`
   becomes a difference between two different sample sets; and
3. the frozen pre-fix record is overwritten after WP-A2 re-injects, at which point the
   count is 0 by construction and the whole comparison evaporates.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from edit_judge_bias.bias.distraction import DistractionInjector, _intersection
from edit_judge_bias.experiments.build_sensitivity_tables import (
    CHANGED_PIXEL_THRESHOLD,
    REGION_CUES,
    build_exclusion_sensitivity,
    build_mos_stratum_sensitivity,
    changed_pixel_bbox,
    mask_affected_ids,
    minimum_candidate_overlap,
    spec_violating_ids,
    sticker_rect,
    within_source_tertiles,
)

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "results" / "v2" / "metrics"
AUDIT = METRICS / "mask_polarity_audit.json"
VIOLATIONS = METRICS / "distraction_spec_violation_audit.json"
BIASED = REPO / "data" / "manifests" / "biased_samples_full_v2.jsonl"
SAMPLES = REPO / "data" / "manifests" / "samples_judge_v2.jsonl"

on_disk = pytest.mark.skipif(
    not (AUDIT.exists() and VIOLATIONS.exists()), reason="v2 audits not on disk"
)


# --------------------------------------------------------------------------- #
# geometry: tie the reconstruction to the injector, not to a remembered number #
# --------------------------------------------------------------------------- #
def test_reconstructed_canvas_contains_the_pixels_the_injector_changed(tmp_path):
    """Run the injector, then rebuild its canvas from the params it recorded.

    This is the load-bearing geometry test: it ties `sticker_rect` to the code that
    actually pasted the pixels, so a future change to `_candidate_positions` or to the
    margin breaks a test instead of quietly shifting all 91 rectangles.

    It also pins the exact asymmetry the 22-vs-21 correction rests on: the changed
    pixels are a STRICT subset of the canvas, because the assets are drawn on a
    transparent square.
    """
    from PIL import Image

    image = Image.new("RGB", (400, 300), (120, 120, 120))
    out, params = DistractionInjector().transform(
        image, {"max_area_frac": 0.04, "margin_frac": 0.02, "asset": "star"},
        random.Random(42), {},
    )
    # `sticker_rect` recovers the frame size from `avoided_bbox`, so this test is only
    # valid where that bbox IS the frame. With no original and no mask the fallback is
    # a centred crop, not the frame — so pin the sticker to a full-frame region the way
    # the 91 affected samples were injected, and assert the recovery explicitly.
    params["avoided_bbox"] = [0, 0, 400, 300]

    edited_p, biased_p = tmp_path / "e.png", tmp_path / "b.png"
    image.save(edited_p)
    out.save(biased_p)

    canvas = sticker_rect(params)
    assert (canvas[2] - canvas[0], canvas[3] - canvas[1]) == tuple(params["sticker_size"])

    ink = changed_pixel_bbox(edited_p, biased_p)
    assert ink is not None
    assert canvas[0] <= ink[0] and canvas[1] <= ink[1]
    assert ink[2] <= canvas[2] and ink[3] <= canvas[3]
    # STRICT subset for a star — this is the whole reason a canvas count overcounts.
    assert (ink[2] - ink[0]) * (ink[3] - ink[1]) < (canvas[2] - canvas[0]) * (canvas[3] - canvas[1])


def test_canvas_can_overlap_a_region_the_ink_does_not():
    """The 22-vs-21 mechanism, as a standalone property rather than an anecdote."""
    from PIL import Image

    out, params = DistractionInjector().transform(
        Image.new("RGB", (400, 300), (10, 10, 10)),
        {"max_area_frac": 0.04, "margin_frac": 0.02, "asset": "star"},
        random.Random(0), {},
    )
    params["avoided_bbox"] = [0, 0, 400, 300]   # see the note in the test above
    canvas = sticker_rect(params)
    # A 3px sliver at the canvas's top-left corner: inside the canvas, but a star has
    # nothing drawn in its corners.
    sliver = (canvas[0], canvas[1], canvas[0] + 3, canvas[1] + 3)
    assert _intersection(canvas, sliver) > 0
    px = out.load()
    assert all(px[x, y] == (10, 10, 10)
               for x in range(sliver[0], sliver[2]) for y in range(sliver[1], sliver[3]))


# --------------------------------------------------------------------------- #
# the frozen violation record                                                  #
# --------------------------------------------------------------------------- #
@on_disk
def test_frozen_violation_audit_has_the_measured_shape():
    a = json.loads(VIOLATIONS.read_text(encoding="utf-8"))
    assert a["n_candidates"] == 91
    assert a["n_violating"] == 21
    # The canvas count is retained so the correction to the canvas-geometry count is auditable.
    assert a["n_violating_canvas_geometry"] == 22
    assert a["canvas_only_false_alarms"] == ["mb_362944_1_human_annotator"]
    assert a["changed_pixel_threshold"] == CHANGED_PIXEL_THRESHOLD
    # Every measured ink box must sit inside the canvas the reconstruction predicts.
    assert a["reconstruction_control"]["ink_bbox_inside_reconstructed_canvas"] == "91/91"
    assert sum(1 for r in a["per_sample"] if r["violates_spec"]) == 21


@on_disk
def test_violations_are_substantial_not_marginal():
    """A one-pixel touch and a fully-covered edit region are not the same finding."""
    a = json.loads(VIOLATIONS.read_text(encoding="utf-8"))
    assert 0 < a["frac_inside_min"] <= a["frac_inside_median"] <= a["frac_inside_max"] <= 1.0
    # Measured 2026-08-16: median 0.888, 13 over 50%, 10 over 90%.
    assert a["frac_inside_median"] > 0.5
    assert a["n_over_90pct_inside"] >= 10


@on_disk
def test_the_two_exclusion_levels_are_nested():
    """Not nested means the trajectory is not a trajectory."""
    spec = set(spec_violating_ids(VIOLATIONS))
    affected = mask_affected_ids(AUDIT)
    assert len(spec) == 21 and len(affected) == 91
    assert spec < affected


@on_disk
def test_violations_are_read_against_the_corrected_region_never_the_published_one():
    """Reading them against the published (full-frame) bbox marks all 91 as violations.

    That mistake makes the defect look four times worse than it is, and it is invisible
    in the output because '91 violations' is also a plausible number.
    """
    a = json.loads(VIOLATIONS.read_text(encoding="utf-8"))
    published_bbox_hits = sum(
        1 for r in a["per_sample"] if r["changed_px_total"] > 0
    )  # every sticker trivially sits inside the full frame
    assert published_bbox_hits == 91
    assert a["n_violating"] == 21


# --------------------------------------------------------------------------- #
# the exclusion arithmetic                                                     #
# --------------------------------------------------------------------------- #
@on_disk
def test_exclusion_drops_the_same_samples_from_baseline_and_biased():
    """`mean_shift` must never compare two different sample sets.

    `build_claim_a` filters both arms with one keep-set, so the guard here is that the
    per-level `n` really falls by the number of dropped samples that the cue actually
    covers — an exclusion applied to only one arm leaves `n` unchanged.
    """
    rows = build_exclusion_sensitivity(
        REPO / "results" / "v2",
        samples_path=SAMPLES,
        subset_filter={"subset_block": "breadth"},
        audit_path=AUDIT,
        violation_audit=VIOLATIONS,
    )
    by_level = {}
    for r in rows:
        by_level.setdefault(r["level"], []).append(r)
    assert set(by_level) == {"published", "drop_spec_viol", "drop_mask_affected"}
    assert all(len(v) == 15 for v in by_level.values()), "3 cues x 5 judges per level"

    assert by_level["published"][0]["n_samples_dropped"] == 0
    assert by_level["drop_spec_viol"][0]["n_samples_dropped"] == 21
    assert by_level["drop_mask_affected"][0]["n_samples_dropped"] == 91

    # n must strictly decrease as more samples are dropped, for every cell.
    key = lambda r: (r["judge_model"], r["bias_type"])
    pub = {key(r): r["n"] for r in by_level["published"]}
    for level, dropped in (("drop_spec_viol", 21), ("drop_mask_affected", 91)):
        for r in by_level[level]:
            assert r["n"] < pub[key(r)], f"{level} {key(r)} kept n unchanged"
            assert pub[key(r)] - r["n"] <= dropped


@on_disk
def test_lost_significance_is_undefined_for_never_significant_cells():
    """A cell that was never significant cannot "stop surviving".

    `zoom_inset x gpt-4o-viescore` is published at +0.11 with q=0.91. Scoring it as a
    loss at every exclusion level would report the headline as fragile in a place where
    nothing was ever claimed — the exact overstatement this column exists to prevent.
    """
    rows = build_exclusion_sensitivity(
        REPO / "results" / "v2",
        samples_path=SAMPLES,
        subset_filter={"subset_block": "breadth"},
        audit_path=AUDIT,
        violation_audit=VIOLATIONS,
    )
    viescore_zoom = [
        r for r in rows
        if r["judge_model"] == "gpt-4o-viescore" and r["bias_type"] == "zoom_inset"
    ]
    assert len(viescore_zoom) == 3
    assert all(r["lost_significance"] is None for r in viescore_zoom)
    assert all(r["published_significant_bh"] is not True for r in viescore_zoom)

    # And the headline the paper actually leans on: at the harshest exclusion, the
    # cells that WERE significant almost all still are.
    harsh = [r for r in rows if r["level"] == "drop_mask_affected"]
    tested = [r for r in harsh if r["lost_significance"] is not None]
    assert len(tested) == 14, "14 of the 15 published cells were significant"
    assert sum(1 for r in tested if r["lost_significance"]) <= 1


@on_disk
def test_builder_refuses_to_pool_the_blocks():
    """The n=1,196 accident, as a guard rather than as a comment."""
    with pytest.raises(ValueError, match="subset filter"):
        build_exclusion_sensitivity(
            REPO / "results" / "v2",
            samples_path=SAMPLES,
            subset_filter=None,
            audit_path=AUDIT,
            violation_audit=VIOLATIONS,
        )


def test_region_cues_match_the_audit_script():
    """One list, two files — drift silently changes the affected-image count."""
    src = (REPO / "scripts" / "audit_mask_polarity.py").read_text(encoding="utf-8")
    assert len(REGION_CUES) == 3
    for cue in REGION_CUES:
        assert cue in src


# --------------------------------------------------------------------------- #
# WP-A2 — can the violation be avoided at all?                                 #
# --------------------------------------------------------------------------- #
def test_minimum_candidate_overlap_is_zero_when_a_slot_is_clear():
    """A small region in one corner leaves seven clear slots, so the floor is 0."""
    params = {"sticker_size": [100, 100]}
    assert minimum_candidate_overlap(params, (0, 0, 150, 150), (1000, 1000)) == 0


def test_minimum_candidate_overlap_is_positive_when_the_region_is_the_whole_frame():
    """The case that survives WP-A2: the dataset's own mask covers the picture.

    `DistractionInjector` picks the least-overlapping of eight slots, so a positive
    floor means NO placement satisfies "outside the edit region". Reporting such a
    sample as a placement failure would misdescribe the repair.
    """
    params = {"sticker_size": [100, 100]}
    floor = minimum_candidate_overlap(params, (0, 0, 1000, 1000), (1000, 1000))
    assert floor == 100 * 100


def test_minimum_candidate_overlap_reads_the_frame_from_the_image_not_the_bbox():
    """`avoided_bbox` stopped being the frame the moment the fix landed.

    Before WP-A2 the recorded region WAS the whole frame, so `sticker_rect` could
    recover the canvas from it. Afterwards it is the small real region, and reusing it
    lays the eight candidate slots out inside that region instead of around it — which
    makes an AVOIDABLE case report as unavoidable, i.e. it would excuse a real
    placement failure. The numbers below are that mistake, in both directions.
    """
    params = {"sticker_size": [100, 100], "avoided_bbox": [0, 0, 500, 500]}
    region = (0, 0, 500, 500)
    on_frame = minimum_candidate_overlap(params, region, (1000, 1000))
    on_bbox = minimum_candidate_overlap(params, region, (500, 500))
    assert on_frame == 0, "a quarter-frame region leaves the far corners clear"
    assert on_bbox == 100 * 100, "laid out inside the region, every slot 'must' overlap"


@on_disk
def test_the_postfix_record_exists_and_every_survivor_is_unavoidable():
    """WP-A2's real acceptance criterion — NOT "the count reaches zero".

    The WP-A2 design asked for 0 of 91. Measured after the re-injection it
    is 3 of 91, and all three are samples whose true edit region covers 67-100% of the
    frame: the specification has no solution there. The criterion that means something
    is "every survivor is unavoidable", which is what this asserts.
    """
    postfix = METRICS / "distraction_spec_violation_postfix.json"
    if not postfix.exists():
        pytest.skip("WP-A2 verify has not been run")
    after = json.loads(postfix.read_text(encoding="utf-8"))
    before = json.loads(VIOLATIONS.read_text(encoding="utf-8"))
    assert before["n_violating"] == 21, "the frozen pre-fix record must never move"
    assert after["n_violating"] < before["n_violating"]
    assert after["n_violating_unavoidable"] == after["n_violating"]
    assert after["changed_px_inside_region_total"] < (
        before.get("changed_px_inside_region_total")
        or sum(r["changed_px_inside_region"] for r in before["per_sample"])
    )
    for row in after["per_sample"]:
        if row["violates_spec"]:
            assert row["min_possible_overlap_px"] > 0
            assert row["region_frac_of_frame"] > 0.5


# --------------------------------------------------------------------------- #
# A1d — claim A inside human-MOS strata                                        #
# --------------------------------------------------------------------------- #
def _mos_sample(sid: str, source: str, human, *, block: str = "breadth"):
    from edit_judge_bias.data.schema import SampleRecord

    rec = SampleRecord(
        sample_id=sid, source_dataset=source, edit_type="add", content_category="object",
        original_image_path=f"{sid}_o.jpg", instruction="do", edit_model="m",
        edited_image_path=f"{sid}_e.jpg", human_score=human,
    )
    rec.metadata.subset_block = block  # type: ignore[attr-defined]
    return rec


def test_tertiles_are_cut_inside_each_source_not_across_them():
    """A pooled cut would be a source selector wearing a quality label.

    EBench-18K's MOS lives in roughly [0.29, 0.71] and ImagenHub's in [0, 1]; pooling
    them would put most of one source in one stratum. This is the same failure mode as
    the decisive-pair threshold, which had to be measured in SDs of each source's own
    human scores for exactly this reason.
    """
    samples = (
        [_mos_sample(f"e{i}", "EBench-18K", 0.40 + 0.002 * i) for i in range(30)]
        + [_mos_sample(f"h{i}", "ImagenHub", 0.0 + 0.03 * i) for i in range(30)]
    )
    strata = within_source_tertiles(samples, {s.sample_id for s in samples})
    for key in ("mos_tertile_low", "mos_tertile_mid", "mos_tertile_high"):
        sources = {s.source_dataset for s in samples if s.sample_id in strata[key]}
        assert sources == {"EBench-18K", "ImagenHub"}, f"{key} collapsed to {sources}"


def test_identical_human_scores_never_land_in_different_strata():
    """A 13-level scale cannot be split into equal thirds; the cut wins, not the count."""
    samples = ([_mos_sample(f"a{i}", "S", 0.5) for i in range(10)]
               + [_mos_sample(f"b{i}", "S", 0.9) for i in range(5)])
    strata = within_source_tertiles(samples, {s.sample_id for s in samples})
    tied = {s.sample_id for s in samples if s.human_score == 0.5}
    landed = [k for k in ("mos_tertile_low", "mos_tertile_mid", "mos_tertile_high")
              if strata[k] & tied]
    assert len(landed) == 1, f"the 0.5 block was split across {landed}"


def test_published_and_has_human_strata_are_what_they_say():
    samples = ([_mos_sample(f"a{i}", "S", 0.1 * i) for i in range(6)]
               + [_mos_sample(f"n{i}", "S", None) for i in range(4)])
    keep = {s.sample_id for s in samples}
    strata = within_source_tertiles(samples, keep)
    assert strata["published"] == keep
    assert strata["has_human"] == {f"a{i}" for i in range(6)}
    assert not any(s.startswith("n") for s in strata["mos_tertile_high"])


def test_scope_restriction_drops_the_other_sources_entirely():
    samples = ([_mos_sample(f"e{i}", "EBench-18K", 0.1 * i) for i in range(6)]
               + [_mos_sample(f"h{i}", "ImagenHub", 0.1 * i) for i in range(6)])
    strata = within_source_tertiles(
        samples, {s.sample_id for s in samples}, sources={"EBench-18K"}
    )
    assert all(s.startswith("e") for s in strata["has_human"])
    # `published` is the untouched reference and is deliberately NOT scoped: the
    # trajectory columns compare each stratum against the number in the paper.
    assert len(strata["published"]) == 12


@on_disk
def test_mos_builder_refuses_to_pool_the_blocks():
    with pytest.raises(ValueError, match="subset filter"):
        build_mos_stratum_sensitivity(
            REPO / "results" / "v2", samples_path=SAMPLES, subset_filter=None,
        )


@on_disk
def test_mos_strata_report_power_so_a_null_cannot_be_misread():
    """`detectable` must be about power, and it must sometimes be False.

    A tertile is a third of the human-labelled samples, so the smallest headline cue
    (`brightness`, -0.26..-0.57 points) sits below the stratum's own MDE. If every row
    came back detectable the column would be decorative and the `zoom_inset` trap would
    be open again.
    """
    rows = build_mos_stratum_sensitivity(
        REPO / "results" / "v2", samples_path=SAMPLES,
        subset_filter={"subset_block": "breadth"}, scopes=({"EBench-18K"},),
    )
    tertile = [r for r in rows if r["stratum"] == "mos_tertile_high"]
    assert tertile and all(r["mde_sd"] is not None for r in tertile)
    assert any(r["detectable"] is False for r in tertile)
    for r in tertile:
        if r["sd_shift"] and r["mean_shift"] is not None:
            assert r["shift_in_sd_units"] == pytest.approx(
                r["mean_shift"] / r["sd_shift"], abs=1e-3
            )
    # The published stratum is the whole breadth block, so it must reproduce claim A.
    published = {(r["judge_model"], r["bias_type"]): r
                 for r in rows if r["stratum"] == "published"}
    assert published[("gpt-5.5", "padding")]["n"] == 611
    assert published[("gpt-5.5", "padding")]["shift_delta_vs_published"] == 0.0


@on_disk
def test_the_deflation_is_not_carried_by_the_worst_edits():
    """A1d's reason for existing: does the effect survive on the best-rated third?

    Answer, on EBench-18K's continuous MOS: `distraction` keeps its sign and its BH
    significance on 5/5 judges in the top tertile, and its magnitude is LARGER there
    than in the published table. If this ever inverts, the paper's extrapolation
    sentence has to change, so it is a test rather than a remembered result.
    """
    rows = build_mos_stratum_sensitivity(
        REPO / "results" / "v2", samples_path=SAMPLES,
        subset_filter={"subset_block": "breadth"}, scopes=({"EBench-18K"},),
    )
    top = [r for r in rows
           if r["stratum"] == "mos_tertile_high" and r["bias_type"] == "distraction"]
    assert len(top) == 5
    assert all(r["sign_preserved"] for r in top)
    assert all(r["significant_bh"] for r in top)
    assert min(r["mean_shift"] for r in top) < min(r["published_mean_shift"] for r in top)


# --------------------------------------------------------------------------- #
# WP-F1d — the zero-difference policy                                         #
# --------------------------------------------------------------------------- #
def _zero_policy_rows():
    import csv
    path = (Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"
            / "claim_a_zero_policy_sensitivity.csv")
    if not path.exists():
        pytest.skip("claim_a_zero_policy_sensitivity.csv not built")
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def test_the_wilcox_column_reproduces_the_published_p_and_q():
    """The whole table is only readable if its `wilcox` half IS claim A. If the two
    disagreed anywhere, a difference in the `pratt` column could be the policy or could
    be a second pipeline, and nobody could tell which."""
    import csv
    claim_a = (Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"
               / "claim_a.csv")
    if not claim_a.exists():
        pytest.skip("claim_a.csv not built")
    published = list(csv.DictReader(claim_a.open(encoding="utf-8-sig")))
    rows = {(r["judge_model"], r["bias_type"]): r for r in _zero_policy_rows()}
    assert len(published) == len(rows) == 65
    for r in published:
        z = rows[(r["judge_model"], r["bias_type"])]
        assert z["n"] == r["n"]
        assert z["p_wilcox"] == r["p_value"]
        if r["bias_type"] != "sham":
            assert z["q_wilcox"] == r["q_value"]


def test_no_headline_cue_depends_on_the_zero_policy():
    """The published p-values discard items whose score did not move (`zero_method
    ="wilcox"`), and between 3% and 72% of a cell's items are such zeros — so the
    policy is a real analysis degree of freedom, not a formality.

    Nine of the 60 tested cells change verdict under Pratt (3 out, 6 in) and NOT ONE of
    them is a headline cue: every flip is in aesthetic_filter / model_name /
    detail_caption / saturation, i.e. exactly the group the paper already reports as
    inside the placebo bound."""
    rows = _zero_policy_rows()
    headline = {"distraction", "text_overlay", "padding", "region_annotation",
                "brightness", "bandwagon", "zoom_inset"}
    flipped = [r for r in rows if r["policy_changes_verdict"] == "True"]
    assert flipped, "the sensitivity check is only worth reporting if it can fire"
    assert not {r["bias_type"] for r in flipped} & headline, [
        (r["judge_model"], r["bias_type"]) for r in flipped
    ]


def test_the_two_policies_get_two_bh_families():
    """A q from one zero policy has no meaning against the other's: they are two
    corrections of two different 60-test families, not one family of 120."""
    rows = _zero_policy_rows()
    tested = [r for r in rows if r["bias_type"] != "sham"]
    assert {r["family"] for r in tested} == {"claim_A (wilcox) + claim_A_pratt"}
    assert {r["family"] for r in rows if r["bias_type"] == "sham"} == {"control"}
    for r in rows:
        if r["bias_type"] == "sham":
            assert r["q_wilcox"] == "" and r["q_pratt"] == ""
        else:
            assert r["q_wilcox"] and r["q_pratt"]
