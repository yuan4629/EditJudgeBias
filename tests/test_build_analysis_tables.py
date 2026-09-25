"""Tests for the §6 quality + §9.4 breakdown table builder."""

from __future__ import annotations

import csv
from pathlib import Path

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import (
    JudgeResult,
    QualityValidationResult,
    SampleRecord,
)
from edit_judge_bias.experiments.build_analysis_tables import (
    EXPLORATORY_FAMILY,
    _load_sample_groups,
    build_all,
    build_cell_coverage,
    build_positive_control,
    build_quality_combined,
)
from edit_judge_bias.metrics.stats import fisher_exact_pvalue


def _score(model, sid, score, bias=None):
    biased_id = f"{sid}__{bias}" if bias else None
    return JudgeResult(
        result_id=f"{model}::{sid}::{bias}", judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt", parse_success=True,
        sample_id=sid, bias_type=bias, biased_id=biased_id, overall_score=score,
    )


def _sample(sid, edit_type, content):
    return SampleRecord(
        sample_id=sid, source_dataset="t", edit_type=edit_type,
        content_category=content, original_image_path="o.jpg",
        instruction="do", edit_model="m", edited_image_path="e.jpg",
    )


def _val(biased_id, ssim, passed):
    return QualityValidationResult(
        instruction_adherence_changed=False, editing_quality_changed=not passed,
        detail_preservation_changed=False, major_semantic_shift=False,
        passed=passed, biased_id=biased_id, parse_success=True, ssim=ssim,
    )


def _setup(root: Path):
    # gpt-5.5 is in the roster; mock-judge must be excluded from breakdowns.
    rj = root / "results/raw_judgments"
    bj = root / "results/biased_judgments"
    for model in ("gpt-5.5", "mock-judge"):
        io.write_jsonl(rj / f"scoring__{model}.jsonl",
                       [_score(model, "s_add", 3), _score(model, "s_rem", 3)])
        io.write_jsonl(bj / f"scoring__{model}.jsonl", [
            _score(model, "s_add", 2, "padding"),   # add: shift -1
            _score(model, "s_rem", 1, "padding"),   # remove: shift -2
        ])
    manifest = root / "data/manifests/samples_pilot.jsonl"
    io.write_jsonl(manifest, [
        _sample("s_add", "add", "human"),
        _sample("s_rem", "remove", "object"),
    ])
    return manifest


def test_breakdown_splits_by_edit_type_and_excludes_non_roster(tmp_path: Path):
    manifest = _setup(tmp_path)
    tables = build_all(tmp_path / "results", manifest=manifest,
                       human_csv=tmp_path / "nope.csv")
    by_edit = tables["by_edit"]
    # Only gpt-5.5 (roster), 1 bias x 2 edit_types = 2 cells; mock-judge excluded.
    assert {r["judge_model"] for r in by_edit} == {"gpt-5.5"}
    assert len(by_edit) == 2
    add = next(r for r in by_edit if r["edit_type"] == "add")
    rem = next(r for r in by_edit if r["edit_type"] == "remove")
    assert add["n"] == 1 and add["mean_shift"] == -1.0
    assert rem["n"] == 1 and rem["mean_shift"] == -2.0


def test_breakdown_by_content_category(tmp_path: Path):
    manifest = _setup(tmp_path)
    tables = build_all(tmp_path / "results", manifest=manifest,
                       human_csv=tmp_path / "nope.csv")
    cats = {r["content_category"] for r in tables["by_content"]}
    assert cats == {"human", "object"}
    csv_path = tmp_path / "results/metrics/scoring_shift_by_content_category.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert all("content_category" in r for r in rows)


def test_quality_combined_folds_ssim_mllm_and_human(tmp_path: Path):
    results = tmp_path / "results"
    io.write_jsonl(results / "quality/validation__v.jsonl", [
        _val("a__padding", 1.0, True),
        _val("b__padding", 1.0, True),
        _val("c__brightness", 0.90, True),
        _val("d__brightness", 0.85, False),
    ])
    human = tmp_path / "human.csv"
    with human.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "image_file", "bias_type", "biased_id",
                    "base_sample_id", "instruction",
                    "quality_changed (No/Slightly/Yes)"])
        w.writerow([1, "i.png", "padding", "a__padding", "a", "do", "No"])
        w.writerow([2, "i.png", "padding", "b__padding", "b", "do", "Slightly"])
        w.writerow([3, "i.png", "brightness", "c__brightness", "c", "do", "Yes"])

    rows = build_quality_combined(results, human)
    pad = next(r for r in rows if r["bias_type"] == "padding")
    assert pad["mean_ssim"] == 1.0 and pad["mllm_pass_rate"] == 1.0
    assert pad["human_no"] == 1 and pad["human_slightly"] == 1
    assert pad["human_preserved_rate"] == 1.0  # No + Slightly over 2

    bri = next(r for r in rows if r["bias_type"] == "brightness")
    assert bri["mllm_pass_rate"] == 0.5  # 1 of 2 passed
    assert bri["min_ssim"] == 0.85
    assert bri["human_yes"] == 1 and bri["human_preserved_rate"] == 0.0


def test_quality_table_carries_the_sham_false_flag_floor(tmp_path: Path):
    """A validator flags some fraction of a VISUALLY NULL re-encode; that fraction is
    its false-flag floor, and a bias passing at the same rate has not been shown to
    change anything. MEASURED: distraction 0.982 against a sham floor of 0.982."""
    results = tmp_path / "results"
    io.write_jsonl(results / "quality/validation__v.jsonl", [
        _val("a__sham", 1.0, True), _val("b__sham", 1.0, False),      # floor = 0.5
        _val("c__padding", 1.0, True), _val("d__padding", 1.0, True),  # 1.0 > floor
        _val("e__zoom_inset", 0.8, False), _val("f__zoom_inset", 0.8, False),
    ])
    rows = {r["bias_type"]: r for r in build_quality_combined(results, tmp_path / "no.csv")}
    assert rows["padding"]["control_pass_rate"] == 0.5
    assert rows["padding"]["above_control_floor"] is True
    assert rows["padding"]["passes_85_gate"] is True
    assert rows["zoom_inset"]["above_control_floor"] is False
    assert rows["zoom_inset"]["passes_85_gate"] is False


def test_validators_are_reported_per_row_and_never_pooled(tmp_path: Path):
    """Pooling validators moved a headline. The builder used to glob every
    `validation__*.jsonl` into one average, so adding a noisy third validator dragged
    BOTH the pass rate and the sham floor, and `above_control_floor` — the single fact
    §6 rests on — could flip because of which validators happened to be on disk. Here
    `strict` sees zoom_inset as clean above its own perfect floor while `noisy` fails a
    fifth of everything: pooled, zoom_inset would read 0.5 against a 0.75 floor (BELOW,
    a false boundary case); per-validator, each verdict stands on its own evidence."""
    results = tmp_path / "results"
    io.write_jsonl(results / "quality/validation__strict.jsonl", [
        _val("a__sham", 1.0, True), _val("b__sham", 1.0, True),          # floor = 1.0
        _val("e__zoom_inset", 0.8, True), _val("f__zoom_inset", 0.8, True),
    ])
    io.write_jsonl(results / "quality/validation__noisy.jsonl", [
        _val("g__sham", 1.0, True), _val("h__sham", 1.0, False),          # floor = 0.5
        _val("i__zoom_inset", 0.8, False), _val("j__zoom_inset", 0.8, False),
    ])
    rows = build_quality_combined(results, tmp_path / "no.csv")
    by = {(r["validator_model"], r["bias_type"]): r for r in rows}

    assert by[("strict", "zoom_inset")]["control_pass_rate"] == 1.0
    assert by[("strict", "zoom_inset")]["mllm_pass_rate"] == 1.0
    assert by[("strict", "zoom_inset")]["above_control_floor"] is True
    # The noisy validator is scored against ITS OWN floor, not the strict one's.
    assert by[("noisy", "zoom_inset")]["control_pass_rate"] == 0.5
    assert by[("noisy", "zoom_inset")]["mllm_pass_rate"] == 0.0
    assert by[("noisy", "zoom_inset")]["above_control_floor"] is False


def test_below_the_floor_requires_significance_not_just_a_lower_rate(tmp_path: Path):
    """A bare `rate < floor` comparison over-flags whenever the floor is near-perfect.
    MEASURED on the real grid: gpt-4o-mini's sham floor is 109/110, which puts 9 of its
    11 arms "below" it on one- or two-image differences that Fisher cannot separate
    from noise. Here `tiny` misses the floor by a single image and must NOT be reported
    as damage, while `real` misses it by half the arm and must be."""
    results = tmp_path / "results"
    rows_in = [_val(f"s{i}__sham", 1.0, True) for i in range(40)]
    rows_in += [_val(f"t{i}__tiny", 1.0, i > 0) for i in range(40)]      # 39/40
    rows_in += [_val(f"r{i}__real", 1.0, i >= 20) for i in range(40)]    # 20/40
    io.write_jsonl(results / "quality/validation__v.jsonl", rows_in)

    by = {r["bias_type"]: r for r in build_quality_combined(results, tmp_path / "no.csv")}
    assert by["tiny"]["above_control_floor"] is False        # numerically below...
    assert by["tiny"]["below_floor_significant"] is False    # ...but not distinguishable
    assert by["real"]["below_floor_significant"] is True
    assert by["real"]["floor_q_value"] < 0.05
    # The control arm is the reference, so it never tests against itself.
    assert by["sham"]["floor_p_value"] == ""


def test_validators_flag_narrows_the_quality_table(tmp_path: Path):
    results = tmp_path / "results"
    io.write_jsonl(results / "quality/validation__keep.jsonl", [_val("a__padding", 1.0, True)])
    io.write_jsonl(results / "quality/validation__drop.jsonl", [_val("b__padding", 1.0, False)])
    rows = build_quality_combined(results, tmp_path / "no.csv", validators=["keep"])
    assert {r["validator_model"] for r in rows} == {"keep"}


def test_a_validator_without_a_sham_arm_issues_no_floor_verdict(tmp_path: Path):
    """glm-4v stopped at 507/1210 and never reached sham, so its false-flag floor is
    UNMEASURED. Blank is the honest output; a pooled floor borrowed from another
    validator would have manufactured a verdict its own data cannot support."""
    results = tmp_path / "results"
    io.write_jsonl(results / "quality/validation__nosham.jsonl", [
        _val("a__padding", 1.0, True), _val("b__padding", 1.0, False),
    ])
    row = next(r for r in build_quality_combined(results, tmp_path / "no.csv"))
    assert row["control_pass_rate"] == ""
    assert row["above_control_floor"] == ""
    assert row["passes_85_gate"] is False  # the 85% gate is absolute, still computable


# --------------------------------------------------------------------------- #
# WP-A6a: two human packages, and how much of each the validator actually saw #
# --------------------------------------------------------------------------- #
def _human_csv(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bias_type", "biased_id",
                                           "quality_changed (No/Slightly/Yes)"])
        w.writeheader()
        for bias, biased_id, label in rows:
            w.writerow({"bias_type": bias, "biased_id": biased_id,
                        "quality_changed (No/Slightly/Yes)": label})
    return path


def test_two_human_packages_covering_different_cues_are_both_counted(tmp_path: Path):
    """The published 150 cover five B-class cues; WP-A6a's 180 cover the six that had
    none. They are disjoint, so reading only one silently reports n_human=0 for the
    other's cues — which is the very gap the second package exists to close."""
    results = tmp_path / "results"
    io.write_jsonl(results / "quality/validation__v.jsonl", [
        _val("a__padding", 1.0, True), _val("b__zoom_inset", 1.0, True),
    ])
    old = _human_csv(tmp_path / "old.csv", [("padding", "a__padding", "No")])
    new = _human_csv(tmp_path / "new.csv", [("zoom_inset", "b__zoom_inset", "Slightly")])

    by = {r["bias_type"]: r for r in build_quality_combined(results, [old, new])}
    assert by["padding"]["n_human"] == 1 and by["zoom_inset"]["n_human"] == 1
    assert by["padding"]["human_package"] == "old"
    assert by["zoom_inset"]["human_package"] == "new"


def test_n_human_matched_counts_labels_the_validator_actually_saw(tmp_path: Path):
    """MEASURED on the real grid: 150 human labels and 1,210 validator verdicts share
    exactly 5 pictures, because the human package read the pilot manifests while the
    validators ran v2 and the legs were aligned only by the cue STRING. A cue can
    therefore show n_human=30 and still have no human evidence about the images the
    validator judged. Here `stale` is labelled on pictures the validator never saw."""
    results = tmp_path / "results"
    io.write_jsonl(results / "quality/validation__v.jsonl", [
        _val(f"v{i}__stale", 1.0, True) for i in range(5)
    ] + [_val(f"v{i}__fresh", 1.0, True) for i in range(5)])
    human = _human_csv(tmp_path / "h.csv",
                       [("stale", f"pilot{i}__stale", "No") for i in range(3)]
                       + [("fresh", f"v{i}__fresh", "No") for i in range(3)])

    by = {r["bias_type"]: r for r in build_quality_combined(results, human)}
    assert by["stale"]["n_human"] == 3 and by["stale"]["n_human_matched"] == 0
    assert by["fresh"]["n_human"] == 3 and by["fresh"]["n_human_matched"] == 3


def test_a_single_human_csv_path_still_works(tmp_path: Path):
    """Backwards compatibility: `human_csv` was one Path for the whole published run."""
    results = tmp_path / "results"
    io.write_jsonl(results / "quality/validation__v.jsonl", [_val("a__padding", 1.0, True)])
    human = _human_csv(tmp_path / "h.csv", [("padding", "a__padding", "Yes")])
    row = next(r for r in build_quality_combined(results, human))
    assert row["n_human"] == 1 and row["human_yes"] == 1
    assert row["human_preserved_rate"] == 0.0


# --------------------------------------------------------------------------- #
# WP-A3: the MATCHED floor                                                    #
# --------------------------------------------------------------------------- #
def test_the_matched_columns_expose_a_floor_measured_on_different_pictures(tmp_path: Path):
    """This is the defect WP-A3 exists to repair, reproduced at toy scale.

    `run_quality_validation._select` shuffles each bias bucket with an un-reset RNG, so
    on the real grid `sham`'s 110 images overlap each arm's 110 by only 7-13. Fisher
    still returns a p-value — it is designed for two independent rates — and every
    "below the floor" verdict therefore mixes the arm's effect with "these pictures are
    harder". The matched columns cannot be fooled the same way: with disjoint draws they
    report n=0 and refuse to issue a verdict, which is the honest reading."""
    results = tmp_path / "results"
    rows_in = [_val(f"s{i}__sham", 1.0, True) for i in range(20)]       # 20/20
    rows_in += [_val(f"a{i}__arm", 1.0, i >= 10) for i in range(20)]    # 10/20, no overlap
    io.write_jsonl(results / "quality/validation__v.jsonl", rows_in)

    arm = {r["bias_type"]: r for r in build_quality_combined(results, tmp_path / "no.csv")}["arm"]
    assert arm["below_floor_significant"] is True      # Fisher: happily significant
    assert arm["n_matched"] == 0                       # ...on zero shared pictures
    assert arm["matched_design_complete"] is False
    assert arm["floor_p_matched"] == "" and arm["floor_q_matched"] == ""
    assert arm["below_floor_significant_matched"] == ""


def test_a_matched_arm_that_loses_images_the_null_perturbation_keeps_is_flagged(tmp_path: Path):
    """The post-A3 shape: same base images on both sides, so the pairing is complete and
    the discordant counts carry the whole test. b = the arm failed where sham passed."""
    results = tmp_path / "results"
    rows_in = [_val(f"s{i}__sham", 1.0, True) for i in range(12)]
    rows_in += [_val(f"s{i}__arm", 1.0, i >= 10) for i in range(12)]    # fails 10 of 12
    io.write_jsonl(results / "quality/validation__v.jsonl", rows_in)

    arm = {r["bias_type"]: r for r in build_quality_combined(results, tmp_path / "no.csv")}["arm"]
    assert arm["n_matched"] == 12 and arm["matched_design_complete"] is True
    assert (arm["n_discordant_b"], arm["n_discordant_c"]) == (10, 0)
    assert arm["floor_p_matched"] == round(2 * 0.5 ** 10, 6)
    assert arm["floor_q_matched"] < 0.05
    assert arm["below_floor_significant_matched"] is True


def test_no_discordant_pairs_leaves_the_matched_test_undefined_not_null(tmp_path: Path):
    """Acceptance criterion 5 of the WP-A3 design, and it is already the observed state on the 11
    images `gpt-4o-mini x text_overlay` shares with sham today: both sides fail the SAME
    picture, b = c = 0, and `metrics/stats.py` returns None. "Undefined" and "not
    significant" are different results and the table must not print them the same way."""
    results = tmp_path / "results"
    rows_in = [_val(f"s{i}__sham", 1.0, i > 0) for i in range(11)]      # fails s0
    rows_in += [_val(f"s{i}__arm", 1.0, i > 0) for i in range(11)]      # fails s0 too
    io.write_jsonl(results / "quality/validation__v.jsonl", rows_in)

    arm = {r["bias_type"]: r for r in build_quality_combined(results, tmp_path / "no.csv")}["arm"]
    assert arm["n_matched"] == 11 and arm["matched_design_complete"] is True
    assert (arm["n_discordant_b"], arm["n_discordant_c"]) == (0, 0)
    assert arm["floor_p_matched"] == "" and arm["floor_q_matched"] == ""
    assert arm["below_floor_significant_matched"] == ""


def test_an_arm_that_beats_its_own_floor_is_never_called_damage(tmp_path: Path):
    """McNemar is two-sided, so a significant q says only "these differ". Direction has
    to come from the discordant counts: c > b means the arm passed MORE than the null
    perturbation, which is not evidence that the cue broke preservation."""
    results = tmp_path / "results"
    rows_in = [_val(f"s{i}__sham", 1.0, i >= 10) for i in range(12)]    # sham fails 10
    rows_in += [_val(f"s{i}__arm", 1.0, True) for i in range(12)]       # arm passes all
    io.write_jsonl(results / "quality/validation__v.jsonl", rows_in)

    arm = {r["bias_type"]: r for r in build_quality_combined(results, tmp_path / "no.csv")}["arm"]
    assert (arm["n_discordant_b"], arm["n_discordant_c"]) == (0, 10)
    assert arm["floor_q_matched"] < 0.05
    assert arm["below_floor_significant_matched"] is False


def test_partial_overlap_is_visible_as_a_number_not_hidden(tmp_path: Path):
    """The state the table is in TODAY: 7-13 of each arm's 110 images have a sham
    verdict. The matched test still computes, but on a fraction of the arm — so the row
    must say so, or a q computed on 11 pairs reads as if it came from 110."""
    results = tmp_path / "results"
    rows_in = [_val(f"s{i}__sham", 1.0, True) for i in range(4)]
    rows_in += [_val(f"s{i}__arm", 1.0, False) for i in range(4)]        # 4 shared
    rows_in += [_val(f"x{i}__arm", 1.0, True) for i in range(16)]        # 16 unshared
    io.write_jsonl(results / "quality/validation__v.jsonl", rows_in)

    arm = {r["bias_type"]: r for r in build_quality_combined(results, tmp_path / "no.csv")}["arm"]
    assert arm["n_mllm"] == 20
    assert arm["n_matched"] == 4
    assert arm["matched_design_complete"] is False


def test_the_matched_columns_do_not_disturb_the_published_fisher_columns(tmp_path: Path):
    """Criterion 4 of §2.11: the arm side does not change a single image, so every
    Fisher number must survive the new code path byte for byte. Recomputed here from
    `metrics.stats` directly rather than copied from a previous run."""
    results = tmp_path / "results"
    rows_in = [_val(f"s{i}__sham", 1.0, True) for i in range(20)]
    rows_in += [_val(f"a{i}__arm", 1.0, i >= 6) for i in range(20)]      # 14/20
    io.write_jsonl(results / "quality/validation__v.jsonl", rows_in)

    arm = {r["bias_type"]: r for r in build_quality_combined(results, tmp_path / "no.csv")}["arm"]
    assert arm["floor_p_value"] == round(fisher_exact_pvalue(14, 6, 20, 0), 6)
    assert arm["mllm_pass_rate"] == 0.7 and arm["control_pass_rate"] == 1.0
    assert arm["above_control_floor"] is False


# --------------------------------------------------------------------------- #
# subset filter, coverage, positive control                                   #
# --------------------------------------------------------------------------- #
def _blocked(rec, block):
    rec.metadata.subset_block = block  # type: ignore[attr-defined]
    return rec


def test_subset_filter_restricts_the_breakdown_to_one_block(tmp_path: Path):
    """The three blocks were judged under different numbers of conditions, so pooling
    them changes a cell without changing its heading."""
    rj = tmp_path / "results/raw_judgments"
    bj = tmp_path / "results/biased_judgments"
    io.write_jsonl(rj / "scoring__gpt-5.5.jsonl",
                   [_score("gpt-5.5", "s_breadth", 3), _score("gpt-5.5", "s_anchor", 3)])
    io.write_jsonl(bj / "scoring__gpt-5.5.jsonl", [
        _score("gpt-5.5", "s_breadth", 2, "padding"),   # shift -1
        _score("gpt-5.5", "s_anchor", 9, "padding"),    # shift +6, other block
    ])
    manifest = tmp_path / "m.jsonl"
    io.write_jsonl(manifest, [
        _blocked(_sample("s_breadth", "add", "human"), "breadth"),
        _blocked(_sample("s_anchor", "add", "human"), "anchor"),
    ])

    pooled = build_all(tmp_path / "results", manifest=manifest, human_csv=tmp_path / "n.csv")
    [cell] = pooled["by_edit"]
    assert cell["n"] == 2 and cell["mean_shift"] == 2.5

    filtered = build_all(tmp_path / "results", manifest=manifest,
                         human_csv=tmp_path / "n.csv",
                         subset_filter={"subset_block": "breadth"})
    [cell] = filtered["by_edit"]
    assert cell["n"] == 1 and cell["mean_shift"] == -1.0


def test_coverage_table_names_the_empty_cells(tmp_path: Path):
    """The interaction is reported only where it exists; the holes go on the page."""
    samples = [_sample("s0", "add", "human"), _sample("s1", "remove", "object")]
    rows = build_cell_coverage(samples)
    assert len(rows) == 4  # 2 edit_types x 2 content_categories
    empty = {(r["edit_type"], r["content_category"]) for r in rows if r["empty"]}
    assert empty == {("add", "object"), ("remove", "human")}


def test_positive_control_splits_the_operator_collision(tmp_path: Path):
    """`Enhance the brightness` IS the brightness injector, so on low-level samples the
    cue re-does the edit. Those cells are the positive control; the same bias with them
    removed is the check that the headline is not driven by them."""
    rj = tmp_path / "results/raw_judgments"
    bj = tmp_path / "results/biased_judgments"
    io.write_jsonl(rj / "scoring__gpt-5.5.jsonl",
                   [_score("gpt-5.5", "s_low", 5), _score("gpt-5.5", "s_add", 5)])
    io.write_jsonl(bj / "scoring__gpt-5.5.jsonl", [
        _score("gpt-5.5", "s_low", 9, "brightness"),   # collision: +4
        _score("gpt-5.5", "s_add", 4, "brightness"),   # clean: -1
        _score("gpt-5.5", "s_add", 4, "padding"),      # not a colliding operator
    ])
    manifest = tmp_path / "m.jsonl"
    io.write_jsonl(manifest, [
        _sample("s_low", "low-level", "global"),
        _sample("s_add", "add", "human"),
    ])
    rows = build_positive_control(tmp_path / "results", _load_sample_groups(manifest))
    by_subset = {r["subset"]: r for r in rows}
    assert {r["bias_type"] for r in rows} == {"brightness"}  # padding is not colliding
    assert by_subset["collision"]["n"] == 1 and by_subset["collision"]["mean_shift"] == 4.0
    assert by_subset["clean"]["n"] == 1 and by_subset["clean"]["mean_shift"] == -1.0
    assert by_subset["pooled"]["n"] == 2 and by_subset["pooled"]["mean_shift"] == 1.5


def _many_judge_rows(root: Path, judges, biases, sids):
    """One scoring grid per judge over `sids`, half low-level (collision), half not."""
    rj, bj = root / "results/raw_judgments", root / "results/biased_judgments"
    for j in judges:
        io.write_jsonl(rj / f"scoring__{j}.jsonl", [_score(j, s, 5) for s in sids])
        io.write_jsonl(bj / f"scoring__{j}.jsonl",
                       [_score(j, s, 5 - (i % 3), b)
                        for b in biases for i, s in enumerate(sids)])
    manifest = root / "m.jsonl"
    io.write_jsonl(manifest, [
        _sample(s, "low-level" if i % 2 else "add", "global") for i, s in enumerate(sids)
    ])
    return manifest


def test_positive_control_declares_a_bh_family_per_subset(tmp_path: Path):
    """DEFECT: the study's ONLY positive control shipped 45 p-values (23 below 0.05) and
    is quoted cell by cell, with no multiplicity correction and no family label at all.

    Family = one per `subset`, judges pooled exactly as `claim_A` pools them. NOT one
    family of 45: `pooled` is the deterministic union of `collision` and `clean` over
    the same judge x cue cells, so a single family would count most observations twice
    and make `m` depend on how many nested views the table prints."""
    judges = ("gpt-5.5", "gemini-3.5-flash", "kimi-k2.5")
    manifest = _many_judge_rows(
        tmp_path, judges, ("brightness", "saturation", "aesthetic_filter"),
        [f"s{i}" for i in range(8)],
    )
    rows = build_positive_control(tmp_path / "results", _load_sample_groups(manifest))

    assert len(rows) == len(judges) * 3 * 3          # judges x cues x subsets
    families = {}
    for r in rows:
        assert r["family"] in (
            "positive_control_collision", "positive_control_clean",
            "positive_control_pooled",
        ), r["family"]
        assert r["family"].endswith(r["subset"])     # the label names its own subset
        families.setdefault(r["family"], []).append(r)
    assert len(families) == 3
    assert {len(v) for v in families.values()} == {len(judges) * 3}

    # Every defined p carries a q, and BH is over its OWN subset (m = 9 here), so a
    # q can never be smaller than its p and is never scaled by the full 27.
    for r in rows:
        if isinstance(r["p_value"], float):
            assert isinstance(r["q_value"], float)
            assert r["q_value"] >= r["p_value"] - 1e-12
            assert r["significant_bh"] is (r["q_value"] < 0.05)
    smallest = min((r for r in rows if r["family"] == "positive_control_clean"),
                   key=lambda r: r["p_value"])
    assert smallest["q_value"] <= round(smallest["p_value"] * 9, 6) + 1e-9


def test_the_exploratory_breakdowns_never_ship_a_bare_p_value(tmp_path: Path):
    """DEFECT: 390 + 325 p-values with no q and no family. A within-slice BH would be
    worse than a label here — the cells are slices of claim A's rows and the table is
    read in both directions, so WHICH of the 30/25 slicings gets quoted would stay
    uncorrected. So: no q-value column at all, and every row says so."""
    manifest = _setup(tmp_path)
    tables = build_all(tmp_path / "results", manifest=manifest,
                       human_csv=tmp_path / "nope.csv")

    for key in ("by_edit", "by_content"):
        assert tables[key], key
        for r in tables[key]:
            assert r["family"] == EXPLORATORY_FAMILY == "uncorrected — exploratory"
            # No q-value column: a p beside an empty q reads as "corrected, n.s.".
            assert "q_value" not in r and "significant_bh" not in r

    for name in ("scoring_shift_by_edit_type", "scoring_shift_by_content_category"):
        header = (tmp_path / "results/metrics" / f"{name}.csv").read_text(
            encoding="utf-8").splitlines()[0]
        assert header.endswith(",p_value,family"), header


def test_quality_only_writes_one_table_and_joins_no_human_package(tmp_path: Path):
    """The FILL v2 P4 path: a validator arm on its own images.  Only quality_combined.csv is
    written, and no human package -- drawn from other pictures -- is joined by cue name."""
    from edit_judge_bias.experiments.build_analysis_tables import main

    results = tmp_path / "v2_fill"
    io.write_jsonl(results / "quality/validation__v.jsonl", [
        _val("a__sham", 1.0, True), _val("b__sham", 1.0, True),
        _val("a__padding", 0.9, True), _val("b__padding", 0.9, False),
    ])
    assert main(["--results-dir", str(results), "--quality-only"]) == 0
    assert sorted(p.name for p in (results / "metrics").iterdir()) == ["quality_combined.csv"]
    with (results / "metrics" / "quality_combined.csv").open(encoding="utf-8") as fh:
        rows = {r["bias_type"]: r for r in csv.DictReader(fh)}
    assert rows["padding"]["n_matched"] == "2", "paired against sham on the same two pictures"
    assert rows["padding"]["n_discordant_b"] == "1"
    assert {r["n_human"] for r in rows.values()} == {"0"}


def test_quality_only_refuses_a_tree_without_verdicts(tmp_path: Path):
    import pytest

    from edit_judge_bias.experiments.build_analysis_tables import main

    with pytest.raises(SystemExit):
        main(["--results-dir", str(tmp_path / "empty"), "--quality-only"])
    assert not (tmp_path / "empty" / "metrics").exists()
