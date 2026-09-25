"""Tests for the D-S `dose_control` decomposition.

The frozen `dose_control` row (sham - sham_nonskin) is a sound positive control and an unsound
mechanism claim: `sham` carries dose 0, so the pair differs in dose AND region at once. This
module splits it, and every test here pins a way the split could silently lie — a contrast
standing on a different scene set than its neighbour, a sign flip from label ordering, or the
area-matched-not-dose-matched caveat getting detached from the number it qualifies.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import (
    ContentCategory,
    EditType,
    JudgeResult,
    SampleMetadata,
    SampleRecord,
)
from edit_judge_bias.experiments import build_dose_control_decomposition as DC
from edit_judge_bias.fairness.records import AttributeInjectionRecord
from edit_judge_bias.metrics.fairness_metrics import compute_attribute_gaps, keys_from_manifest

ARMS = ("dark", "light", "sham", "sham_nonskin")
#: The emitter puts the two sham arms under their own contrast family; the decomposition has to
#: cross that boundary, which is exactly why it remaps the attribute.
FAMILY_OF = {"dark": "skin_tone", "light": "skin_tone",
             "sham": "dose_control", "sham_nonskin": "dose_control"}


def _sample(base: str, label: str) -> SampleRecord:
    return SampleRecord(
        sample_id=f"{base}__skin_tone__{label}",
        source_dataset="omniedit_train",
        edit_type=EditType.ADD,
        content_category=ContentCategory.HUMAN,
        original_image_path=f"o/{base}__{label}.png",
        instruction="Add a blue vase",
        edit_model="omniedit_train",
        edited_image_path=f"e/{base}__{label}.png",
        metadata=SampleMetadata(
            base_sample_id=base, attribute=FAMILY_OF[label],
            variant_label=label, render_index=1,
        ),
    )


def _result(judge: str, base: str, label: str, fine: Optional[int]) -> JudgeResult:
    return JudgeResult(
        result_id=f"score::{judge}::vanilla::{base}__skin_tone__{label}",
        judge_model=judge,
        task_type="scoring",
        prompt_type="vanilla_scoring",
        raw_response_path=f"raw/{judge}/{base}__{label}.txt",
        parse_success=True,
        sample_id=f"{base}__skin_tone__{label}",
        overall_score=8,
        fine_score=fine,
        score_scale=10,
    )


def _inj(base: str, label: str, *, mean: float, mass: float) -> AttributeInjectionRecord:
    return AttributeInjectionRecord(
        injection_id=f"{base}__skin_tone__{label}__edited",
        base_sample_id=base, attribute="skin_tone", variant_label=label,
        role="study" if label in ("dark", "light") else label,
        applied_to="edited",
        source_image_path=f"s/{base}.jpg", injected_image_path=f"i/{base}__{label}.png",
        instruction="Add a blue vase", source_dataset="omniedit_train", edit_type="add",
        edit_model="omniedit_train",
        mask_kind="nonskin" if label == "sham_nonskin" else "skin",
        mask_sha256=f"mask_{base}", mask_frac=0.11,
        achieved_delta_ita=-30.0, delta_e00_mean=mean, delta_e00_mass=mass,
        max_outside_diff=0, max_inside_diff=30, outside_mask_identical=True,
    )


def _tree(
    tmp_path: Path,
    scores: Dict[str, Dict[str, Dict[str, Optional[int]]]],
    injections: Sequence[AttributeInjectionRecord] = (),
) -> dict:
    """scores: judge -> base -> label -> fine_score (None = answered but missing a dimension)."""
    bases = sorted({b for per_judge in scores.values() for b in per_judge})
    (tmp_path / "data/manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results/ds/raw_judgments").mkdir(parents=True, exist_ok=True)
    io.write_jsonl(tmp_path / "data/manifests/judge.jsonl",
                   [_sample(b, lab) for b in bases for lab in ARMS])
    for judge, per_base in scores.items():
        rows: List[JudgeResult] = [
            _result(judge, base, lab, fine)
            for base, per_label in per_base.items()
            for lab, fine in per_label.items()
        ]
        io.write_jsonl(tmp_path / f"results/ds/raw_judgments/scoring__{judge}.jsonl", rows)
    if injections:
        io.write_jsonl(tmp_path / "data/manifests/inj.jsonl", list(injections))
    return DC.build(
        root=tmp_path,
        results_dir="results/ds",
        samples="data/manifests/judge.jsonl",
        injections="data/manifests/inj.jsonl" if injections else None,
    )


def _rows(tmp_path: Path) -> List[dict]:
    import csv
    path = tmp_path / "results/ds/metrics/dose_control_decomposition.csv"
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _scene(dark: int, light: int, sham: int, nonskin: int) -> Dict[str, Optional[int]]:
    return {"dark": dark, "light": light, "sham": sham, "sham_nonskin": nonskin}


# --------------------------------------------------------------------------- #
# ★ the identity that makes the decomposition a decomposition                  #
# --------------------------------------------------------------------------- #
def test_the_two_components_reconstruct_the_frozen_dose_control_row(tmp_path: Path):
    """dose_control == nonskin_vs_skin - dose_vs_placebo, exactly, on the same scenes.

    If this ever fails, one of the three numbers is standing on a different scene set and the
    published `dose_control` cannot be read as the sum of the two mechanisms it contains.
    """
    scores = {"j": {f"s{i}": _scene(24 - i % 2, 24, 25, 23) for i in range(6)}}
    _tree(tmp_path, scores)
    rows = {(r["attribute"], r["judge_model"]): float(r["mean_gap"]) for r in _rows(tmp_path)}

    samples = io.read_jsonl(tmp_path / "data/manifests/judge.jsonl", SampleRecord)
    results = io.read_jsonl(
        tmp_path / "results/ds/raw_judgments/scoring__j.jsonl", JudgeResult)
    frozen = {s.attribute: s.mean_gap
              for s in compute_attribute_gaps(results, keys_from_manifest(samples))}

    for arm in ("dark", "light"):
        recomposed = rows[(f"nonskin_vs_skin:{arm}", "j")] - rows[(f"dose_vs_placebo:{arm}", "j")]
        assert recomposed == pytest.approx(frozen["dose_control"], abs=1e-9)


def test_gap_sign_is_study_minus_reference_for_every_contrast(tmp_path: Path):
    """Labels sort alphabetically inside the metric; a rename would silently flip the table."""
    scores = {"j": {f"s{i}": _scene(20, 22, 26, 18) for i in range(5)}}
    _tree(tmp_path, scores)
    for r in _rows(tmp_path):
        study, reference = DC.CONTRASTS[r["attribute"]]
        assert (r["label_a"], r["label_b"]) == (study, reference)
    rows = {(r["attribute"], r["judge_model"]): float(r["mean_gap"]) for r in _rows(tmp_path)}
    assert rows[("dose_vs_placebo:dark", "j")] == pytest.approx(20 - 26)     # skin dose costs
    assert rows[("nonskin_vs_skin:dark", "j")] == pytest.approx(20 - 18)     # off-skin costs more


# --------------------------------------------------------------------------- #
# ★ one scene set per judge, decided on the analysis variable                  #
# --------------------------------------------------------------------------- #
def test_a_row_that_parsed_without_the_analysis_variable_is_not_complete(tmp_path: Path):
    """kimi answers that parse but omit a dimension carry `fine_score = None`.

    Counting those as answered put that judge's four contrasts on two different scene sets
    (476 vs 474 in the real arm), which is the drift this table exists to avoid.
    """
    # 6 scenes so the one missing value stays under `MAX_MISSING_SHARE`; above it
    # `resolve_score_field` refuses to pick a variable at all, which is a different guard.
    scores = {"j": {f"s{i}": _scene(20, 21, 25, 18) for i in range(6)}}
    scores["j"]["s0"]["sham_nonskin"] = None
    rep = _tree(tmp_path, scores)
    assert rep["scenes_by_judge"] == {"j": 5}
    assert {int(r["n"]) for r in _rows(tmp_path)} == {5}


def test_completeness_is_per_judge_not_a_global_intersection(tmp_path: Path):
    """A refusal belongs to (judge, image); a global intersection punishes every judge for it."""
    a = {f"s{i}": _scene(20, 21, 25, 18) for i in range(4)}
    b = {f"s{i}": _scene(20, 21, 25, 18) for i in range(4)}
    del b["s3"]["dark"]                       # only judge `b` never answered this arm
    rep = _tree(tmp_path, {"a": a, "b": b})
    assert rep["scenes_by_judge"] == {"a": 4, "b": 3}


def test_bh_family_is_the_contrast_not_the_arm(tmp_path: Path):
    """`:dark` and `:light` are one manipulation at opposite sign on the same scenes.

    Filing them as two families of 5 instead of one of 10 is the least conservative split
    available and would let a borderline cell through because its twin was counted separately.
    """
    scores = {j: {f"s{i}": _scene(20, 21, 25, 18) for i in range(6)} for j in ("a", "b")}
    _tree(tmp_path, scores)
    rows = _rows(tmp_path)
    families = {r["attribute"]: r["family"] for r in rows}
    assert families == {
        f: f"fairness_contrast:{f.split(':', 1)[0]}" for f in DC.CONTRASTS
    }
    # Each family holds both arms for every judge — on the real arm that is 2 x 5 = 10 rows.
    sizes = collections.Counter(r["family"] for r in rows)
    assert set(sizes.values()) == {2 * len(scores)}


# --------------------------------------------------------------------------- #
# ★ the caveat travels with the number                                         #
# --------------------------------------------------------------------------- #
def test_dose_ratio_columns_land_only_on_the_nonskin_rows(tmp_path: Path):
    """`dose_vs_placebo` shares one mask and has no dose-matching caveat to carry."""
    scores = {"j": {f"s{i}": _scene(20, 21, 25, 18) for i in range(4)}}
    inj = [_inj(f"s{i}", lab, mean=8.0, mass=800.0) for i in range(4) for lab in ("dark", "light")]
    inj += [_inj(f"s{i}", "sham_nonskin", mean=12.0, mass=1200.0) for i in range(4)]
    _tree(tmp_path, scores, injections=inj)
    for r in _rows(tmp_path):
        if r["attribute"] in DC.NONSKIN_FAMILIES:
            assert float(r["dose_mean_ratio_median"]) == pytest.approx(1.5)
            assert float(r["dose_mass_ratio_median"]) == pytest.approx(1.5)
            assert float(r["dose_within_2x_frac"]) == pytest.approx(1.0)
        else:
            assert r["dose_mean_ratio_median"] == ""


def test_dose_ratio_is_measured_only_over_the_scenes_the_contrast_used(tmp_path: Path):
    """An injection for a scene the judge never completed must not enter the ratio."""
    inj = [_inj("s0", lab, mean=8.0, mass=800.0) for lab in ("dark", "light")]
    inj += [_inj("s0", "sham_nonskin", mean=12.0, mass=1200.0)]
    inj += [_inj("s9", "dark", mean=1.0, mass=100.0),
            _inj("s9", "sham_nonskin", mean=99.0, mass=9900.0)]   # never judged
    ratios = DC.nonskin_dose_ratios(inj, study_arm="dark", scenes={"s0"})
    assert ratios["n_dose_pairs"] == 1
    assert ratios["dose_mean_ratio_median"] == pytest.approx(1.5)


def test_failed_injections_never_enter_the_dose_ratio(tmp_path: Path):
    """`success=False` rows carry whatever the solver reached before giving up."""
    ok = _inj("s0", "dark", mean=8.0, mass=800.0)
    bad = _inj("s0", "sham_nonskin", mean=99.0, mass=9900.0)
    bad.success = False
    assert DC.nonskin_dose_ratios([ok, bad], study_arm="dark")["n_dose_pairs"] == 0
