"""WP-A6 table builder: the guards that keep a human leg from being over-read.

Every test here corresponds to a way this arm could produce a number that looks
publishable and is not:

* a per-cue rate quoted without the same annotator's own placebo floor;
* a kappa quoted without the marginals that cap it;
* a kappa of 0 on a class with one instance, read as disagreement;
* a human-vs-validator join that silently compares different pictures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from edit_judge_bias.experiments import build_human_validation_tables as B

LAB = B.LABEL_COL


def _write_resolved(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["index", "image_file", "bias_type", "biased_id",
                            "base_sample_id", LAB])
        w.writeheader()
        for i, (cue, label) in enumerate(rows, start=1):
            w.writerow({"index": i, "image_file": f"images/{i:03d}.png",
                        "bias_type": cue, "biased_id": f"s{i}__{cue}",
                        "base_sample_id": f"s{i}", LAB: label})
    return path


def _labels(pairs):
    return {bid: (label, cue) for bid, label, cue in pairs}


# --------------------------------------------------------------------------- #
# 1. the placebo floor is the denominator, not a cue
# --------------------------------------------------------------------------- #
def test_a_package_without_a_placebo_row_is_refused():
    """A per-cue rate with no floor beside it is not a measurement.

    The whole reason `sham` occupies 30 of the 180 slots is that a human also has
    a false-flag rate, and every other row's claim is "more than that".  Writing
    the table anyway would hand the paper six rates with nothing to read them
    against, and nothing downstream would notice.
    """
    human = _labels([("a__zoom_inset", "Yes", "zoom_inset")])
    with pytest.raises(ValueError, match="false-flag floor"):
        B.build_human_cue_rates(human)


def test_the_placebo_row_is_excluded_from_its_own_bh_family():
    human = _labels(
        [(f"s{i}__sham", "No", "sham") for i in range(10)]
        + [(f"s{i}__zoom_inset", "Yes" if i < 3 else "No", "zoom_inset") for i in range(10)]
    )
    rows = B.build_human_cue_rates(human)
    sham = [r for r in rows if r["bias_type"] == "sham"]
    assert sham and all(r["fisher_p_vs_floor"] is None for r in sham)
    assert all(r["q_value"] is None for r in sham)
    assert all(r["is_placebo"] for r in sham)


def test_both_thresholds_are_reported_and_are_separate_bh_families():
    """`Slightly` is sub-decision-threshold by the annotator's own definition.

    Folding it into the failure class measures a different construct than the one
    the study gates on, so the two readings must never share a family: a cue that
    only moves under the loose reading would otherwise borrow significance from
    the strict one, or spend it.
    """
    human = _labels(
        [(f"s{i}__sham", "No", "sham") for i in range(30)]
        + [(f"s{i}__zoom", "Slightly" if i < 7 else "No", "zoom_inset") for i in range(30)]
    )
    rows = B.build_human_cue_rates(human)
    thresholds = {r["threshold"] for r in rows}
    assert thresholds == {"strict_yes", "loose_any"}
    assert len({r["family"] for r in rows}) == 2
    strict = next(r for r in rows if r["threshold"] == "strict_yes" and r["bias_type"] == "zoom_inset")
    loose = next(r for r in rows if r["threshold"] == "loose_any" and r["bias_type"] == "zoom_inset")
    assert strict["n_fail"] == 0 and loose["n_fail"] == 7


# --------------------------------------------------------------------------- #
# 2. kappa may not be reported without the ceiling its marginals impose
# --------------------------------------------------------------------------- #
def test_kappa_ships_the_ceiling_its_own_marginals_impose():
    """kappa == kappa_max means "they agreed on everything they arithmetically could".

    Without `kappa_max` the same 0.41 reads as "moderate agreement between two
    annotators who often differed about items", which is the opposite of what a
    maximal diagonal says.  This is the frozen 2026-08-18 case.
    """
    primary = _labels(
        [(f"i{i}", "No", "sham") for i in range(55)]
        + [("x1", "No", "zoom_inset"), ("x2", "No", "zoom_inset"),
           ("x3", "No", "region_annotation"), ("x4", "Slightly", "zoom_inset"),
           ("x5", "Yes", "zoom_inset")]
    )
    second = {b: "No" for b in primary}
    second.update({"x1": "Slightly", "x2": "Slightly", "x3": "Slightly",
                   "x4": "Slightly", "x5": "Slightly"})
    out = B.build_iaa(primary, second, n_boot=200)
    three = out["three_level"]
    assert three["kappa_max"] is not None
    assert three["kappa_over_kappa_max"] == pytest.approx(1.0, abs=1e-9)
    # and the prevalence-robust statistics disagree with Cohen by a wide margin,
    # which is exactly why all of them are reported rather than one.
    assert three["gwets_ac1"] > three["cohens_kappa"] + 0.3


def test_a_binary_fold_with_one_positive_is_stamped_degenerate():
    """The paper's gate fold has 1 positive in 60 draws; its kappa of 0 is arithmetic.

    Same failure mode as reading WP-A4c's `b=0` on a cue already at 1.000 as
    evidence that the template does not matter.  `degenerate` exists so the zero
    can never be quoted as an agreement statistic.
    """
    primary = _labels([(f"i{i}", "No", "sham") for i in range(59)] + [("y", "Yes", "zoom_inset")])
    second = {b: "No" for b in primary}
    out = B.build_iaa(primary, second, n_boot=100)
    assert out["binary"]["yes_vs_rest"]["degenerate"] is True
    assert out["binary"]["yes_vs_rest"]["annotator2_positive"] == 0


def test_the_second_sheet_is_joined_through_its_own_key_not_by_row_number(tmp_path):
    """The IAA package is renumbered on purpose.

    Joining on `index` would pair unrelated pictures and still produce a plausible
    kappa, so the loader must go through the second package's own key file.
    """
    d = tmp_path / "iaa"
    d.mkdir()
    with (d / B.IAA_KEY).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "biased_id"])
        w.writeheader()
        w.writerow({"index": 1, "biased_id": "zzz__zoom_inset"})
        w.writerow({"index": 2, "biased_id": "aaa__sham"})
    with (d / "labels.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", LAB])
        w.writeheader()
        w.writerow({"index": 1, LAB: "Yes"})
        w.writerow({"index": 2, LAB: "No"})
    got = B.load_second_annotator(d)
    assert got == {"zzz__zoom_inset": "Yes", "aaa__sham": "No"}


# --------------------------------------------------------------------------- #
# 3. the human-vs-validator join must be per image, not per cue name
# --------------------------------------------------------------------------- #
def test_a_partial_validator_overlap_is_refused_not_silently_dropped():
    """The first human leg overlapped the validator arm by 5 of 150 images.

    It was joined on the `bias_type` string, so it compared one leg's pictures
    with the other leg's other pictures and reported an agreement rate anyway.
    Dropping the missing rows here would reproduce that defect at a smaller scale;
    refusing makes it a build failure instead of a paper sentence.
    """
    human = _labels([("a__sham", "No", "sham"), ("b__zoom_inset", "Yes", "zoom_inset")])
    validators = {"v": {"a__sham": True}}
    with pytest.raises(ValueError, match="never shown to this validator"):
        B.build_human_validator_agreement(human, validators)


def test_cue_level_convergence_is_separated_from_image_level_agreement():
    """Two instruments can rank the cues identically and share no picture.

    `both_fail_*` plus `expected_both_strict_if_independent` are what stop that
    from being written either as "they agree" or as "they disagree": at these base
    rates the expected overlap is a fraction of an image, so an observed zero
    carries no information in either direction.
    """
    human = _labels(
        [(f"s{i}__sham", "No", "sham") for i in range(30)]
        + [(f"s{i}__zoom", "Yes" if i < 2 else "No", "zoom_inset") for i in range(30)]
    )
    # the validator also fails exactly 2 zoom_inset images -- different ones
    passes = {b: True for b in human}
    passes["s28__zoom"] = False
    passes["s29__zoom"] = False
    rows = B.build_human_validator_agreement(human, {"v": passes})
    zoom = next(r for r in rows if r["bias_type"] == "zoom_inset")
    assert zoom["human_fail_strict"] == 2 and zoom["validator_fail"] == 2
    assert zoom["both_fail_strict"] == 0
    assert zoom["expected_both_strict_if_independent"] == pytest.approx(2 * 2 / 30, abs=1e-4)


# --------------------------------------------------------------------------- #
# 4. the frozen artefacts
# --------------------------------------------------------------------------- #
RESULTS = Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"
PACKAGE = Path(__file__).resolve().parents[1] / "data" / "human_validation_v2"


@pytest.mark.skipif(not (RESULTS / "human_validation_cue_rates.csv").exists(),
                    reason="frozen human-validation tables not built")
def test_frozen_cue_rates_cover_the_six_cues_that_had_no_human_label():
    """These six are exactly the `n_human = 0` set the limitations section named."""
    rows = list(csv.DictReader((RESULTS / "human_validation_cue_rates.csv").open(encoding="utf-8")))
    cues = {r["bias_type"] for r in rows}
    assert cues == {"sham", "zoom_inset", "detail_caption", "distraction",
                    "region_annotation", "aesthetic_filter"}
    assert all(int(r["n"]) == 30 for r in rows)


@pytest.mark.skipif(not (RESULTS / "human_validation_cue_rates.csv").exists(),
                    reason="frozen human-validation tables not built")
def test_frozen_no_cue_clears_multiplicity_on_the_PREREGISTERED_threshold():
    """`strict_yes` is the threshold the instrument declared BEFORE any labelling.

    `INSTRUCTIONS.txt`, handed to the annotator with the images, says verbatim "Only No
    and Slightly count as quality-preserved" and "IF YOU ARE UNSURE -- Use Slightly".
    So `Yes` alone is the pre-registered failure criterion and `loose_any` is a post-hoc
    sensitivity reading by construction: it reclassifies as a failure the very category
    the instructions defined as a pass AND as the hedge.

    ⚠️ WHY THIS DISTINCTION IS LOAD-BEARING, 2026-08-18.  The sheets were labelled
    THREE times, rounds two and three each following disclosure of the previous
    round's result.  The lenient reading moved every time and in one direction
    (`zoom_inset` 7/30 -> 10/30 -> 11/30, q 0.0527 -> 0.0040 -> 0.0016, crossing the
    line).  On THIS threshold not one cell moved across all three rounds -- every cue
    identical, `zoom_inset` 2/30 at q=1.0 throughout.  That is what makes the
    re-annotation reportable rather than a garden of forking paths, and it is only
    checkable because the criterion was written down before any labelling.

    ⚠️ What is stable is the COUNT, not the two images: the third round moved one
    `Yes` off image 167 and onto image 101.  "The gate is stable" is a statement about
    whether the cue separates from the floor, not about which picture failed.
    """
    rows = list(csv.DictReader((RESULTS / "human_validation_cue_rates.csv").open(encoding="utf-8")))
    strict = {r["bias_type"]: r for r in rows if r["threshold"] == "strict_yes"}
    assert all(r["above_floor_significant"] == "False" for r in strict.values())
    assert int(strict["zoom_inset"]["n_fail"]) == 2
    assert float(strict["zoom_inset"]["q_value"]) == 1.0
    assert int(strict["sham"]["n_fail"]) == 0


@pytest.mark.skipif(not (RESULTS / "human_validation_cue_rates.csv").exists(),
                    reason="frozen human-validation tables not built")
def test_frozen_the_one_significant_cell_is_on_the_exploratory_threshold_only():
    """`zoom_inset` clears BH under the lenient reading and nothing else does.

    This is the one place the human is MORE sensitive than either gating validator
    (q=0.0040 against their q=0.103 / q=0.051), so it must never be quoted as if the
    three instruments agreed -- they bracket `zoom_inset`, which is what "boundary
    case" means.  The placebo staying at 0/30 across both passes is what says the
    added sensitivity is cue-selective rather than a lowered threshold.
    """
    rows = list(csv.DictReader((RESULTS / "human_validation_cue_rates.csv").open(encoding="utf-8")))
    loose = {r["bias_type"]: r for r in rows if r["threshold"] == "loose_any"}
    sig = [c for c, r in loose.items() if r["above_floor_significant"] == "True"]
    assert sig == ["zoom_inset"]
    assert int(loose["zoom_inset"]["n_fail"]) == 11
    assert float(loose["zoom_inset"]["q_value"]) < 0.05
    assert int(loose["sham"]["n_fail"]) == 0
    # The other three C-class cues are what retires the "B is more provable than C is
    # circular" argument in section 7.1 -- and the claim is made on the PRE-REGISTERED
    # gate, where all three are byte-identical to the placebo in every round.  On the
    # lenient reading they drift by a picture or two between rounds (`distraction` 0 ->
    # 0 -> 1), which is exactly why the argument must not be pinned there.
    strict = {r["bias_type"]: r for r in rows if r["threshold"] == "strict_yes"}
    for cue in ("distraction", "detail_caption", "region_annotation"):
        assert int(strict[cue]["n_fail"]) == 0
        assert int(loose[cue]["n_fail"]) <= 1


@pytest.mark.skipif(not (RESULTS / "human_validation_iaa.json").exists(),
                    reason="frozen IAA not built")
def test_frozen_iaa_kappa_tracks_its_own_ceiling_across_three_rounds():
    """★ Three rounds of labelling, and kappa alone cannot tell them apart.

        round   kappa    kappa_max   reading
          1     0.4074    0.4074     agreed on everything arithmetically available
                                     -> the low kappa was a PREVALENCE artefact
          2     0.2899    0.8817     the ceiling rose and kappa fell -> the room for
                                     agreement appeared and was not taken: REAL disagreement
          3     0.8930    0.8930     ceiling held high AND kappa caught up -> real, high agreement

    Rounds 1 and 3 have the same ratio (1.000) and wildly different kappas; round 2 has
    a middling kappa for a completely different reason.  Reporting kappa alone conflates
    them, and Gwet's AC1 (0.929 / 0.892 / 0.982) and PABAK (0.900 / 0.850 / 0.975) barely
    move across all three -- which is why all four ship in one file and none may be
    quoted alone.  This test pins the final round and the property that produced the
    reading, not just the number.
    """
    out = json.loads((RESULTS / "human_validation_iaa.json").read_text(encoding="utf-8"))
    assert out["n_shared"] == 60
    assert set(out["cue_mix"].values()) == {10}
    three = out["three_level"]
    assert three["kappa_over_kappa_max"] == pytest.approx(1.0, abs=1e-6), (
        "the annotators are back at the ceiling their marginals allow"
    )
    assert three["kappa_max"] > 0.85, "and the ceiling itself is high, unlike round 1"
    assert three["cohens_kappa"] > 0.8
    assert three["observed_agreement"] == pytest.approx(0.9833, abs=1e-4)

    # exactly one disagreement, one notch wide.  Round 2's No<->Yes pair is gone.
    assert len(out["disagreements"]) == 1
    assert {out["disagreements"][0]["annotator1"],
            out["disagreements"][0]["annotator2"]} == {"Yes", "Slightly"}

    # ⚠️ The loose fold -- the one the single significant cue-rate cell is built on --
    # is now PERFECT, which retires the caveat that it sat on the least reliable notch.
    assert out["binary"]["any_change_vs_none"]["cohens_kappa"] == pytest.approx(1.0)
    # ⚠️ The pre-registered gate's fold is still arithmetic, not evidence: 1-2 positives
    # in 60, and its kappa equals its own ceiling, so it cannot go higher at this base rate.
    gate = out["binary"]["yes_vs_rest"]
    assert gate["degenerate"] is True
    assert gate["cohens_kappa"] == pytest.approx(gate["kappa_max"], abs=1e-6)


def test_the_builder_refuses_to_run_on_a_resolved_sheet_that_is_stale(tmp_path: Path):
    """★ This mismatch happened twice on 2026-08-18 and nothing complained either time.

    The tables come from the `unblind` output; the annotator edits `labels.csv`.  Skip
    the resolve step and every table is computed from the PREVIOUS labels while both
    files look freshly written -- once producing a kappa that joined a new annotator-2
    sheet to a stale annotator-1 sheet, once producing a commit whose two files
    disagreed by one row.  Any pipeline with a resolve stage has this shape: the input
    the human touches is not the input the code reads.
    """
    from edit_judge_bias.experiments.build_human_validation_tables import (
        check_resolved_is_current,
    )

    header = "index,image_file,instruction," + LAB
    (tmp_path / "labels.csv").write_text(
        header + "\n1,images/001.png,do a thing,Yes\n", encoding="utf-8")
    (tmp_path / "human_validation_labels_v2.csv").write_text(
        "index,image_file,bias_type,biased_id,base_sample_id," + LAB
        + "\n1,images/001.png,sham,s__sham,s,No\n", encoding="utf-8")

    with pytest.raises(SystemExit) as err:
        check_resolved_is_current(tmp_path)
    assert "unblind" in str(err.value), "the error must name the command that fixes it"


@pytest.mark.skipif(not (RESULTS / "human_validator_agreement.csv").exists(),
                    reason="frozen agreement table not built")
def test_frozen_every_human_labelled_image_was_also_seen_by_every_validator():
    """100% overlap is the property that makes this leg comparable at all.

    The first human package managed 3.3%.  If a future rebuild drops below 100%
    the builder raises; this asserts the frozen table was in fact built at 180.
    """
    rows = list(csv.DictReader((RESULTS / "human_validator_agreement.csv").open(encoding="utf-8")))
    alls = [r for r in rows if r["bias_type"] == "ALL"]
    assert len(alls) == 3
    assert all(int(r["n_images"]) == 180 for r in alls)
