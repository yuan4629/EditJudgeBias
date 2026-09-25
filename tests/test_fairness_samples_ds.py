"""Tests for the D-S sample emitter — injections + construction gate -> judgeable SampleRecords.

This is the seam between "images exist on disk" and "the panel can be paid for", and every
failure mode it has is silent: a dropped arm shrinks n without saying so, a mislabelled family
pools a fairness gap with its own control, and an emitted orphan is discarded later inside the
metric where nobody counts it. Each test below pins one of those.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import (
    ContentCategory,
    EditType,
    JudgeResult,
    SampleRecord,
)
from edit_judge_bias.experiments import build_fairness_samples_ds as DS
from edit_judge_bias.fairness.records import AttributeInjectionRecord
from edit_judge_bias.metrics.fairness_metrics import (
    compute_attribute_gaps,
    compute_editor_noise,
    keys_from_manifest,
)

ARMS = (("dark", "study"), ("light", "study"),
        ("sham", "sham"), ("sham_nonskin", "sham_nonskin"))


def _pool_sample(sample_id: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        source_dataset="omniedit_train",
        edit_type=EditType.ADD,
        content_category=ContentCategory.GLOBAL,   # OmniEdit's own label, deliberately wrong
        original_image_path=f"data/images/original/{sample_id}.jpg",
        instruction="Add a blue vase",
        edit_model="omniedit_train_probe",
        edited_image_path=f"data/images/edited/{sample_id}.jpg",
    )


def _inj(base: str, label: str, role: str, applied_to: str, **kw) -> AttributeInjectionRecord:
    row = dict(
        injection_id=f"{base}__skin_tone__{label}__{applied_to}",
        base_sample_id=base, attribute="skin_tone", variant_label=label, role=role,
        applied_to=applied_to,
        source_image_path=f"src/{base}_{applied_to}.jpg",
        injected_image_path=f"inj/{base}__{label}__{applied_to}.png",
        instruction="Add a blue vase", source_dataset="omniedit_train", edit_type="add",
        edit_model="omniedit_train_probe", mask_kind="skin", mask_sha256=f"mask_{base}",
        mask_frac=0.11, achieved_delta_ita=-30.2, delta_e00_mean=8.2,
        max_outside_diff=0, max_inside_diff=30, outside_mask_identical=True,
    )
    row.update(kw)
    return AttributeInjectionRecord(**row)


def _scene(base: str, arms=ARMS, members=("original", "edited")):
    return [_inj(base, label, role, m) for label, role in arms for m in members]


def _write(tmp_path: Path, rows, scenes=("s1", "s2"), passed=None) -> dict:
    (tmp_path / "data/manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results/v2_fairness/metrics").mkdir(parents=True, exist_ok=True)
    io.write_jsonl(tmp_path / "data/manifests/pool.jsonl",
                   [_pool_sample(s) for s in scenes])
    io.write_jsonl(tmp_path / "data/manifests/inj.jsonl", rows)
    gate = None
    if passed is not None:
        gate = "results/v2_fairness/metrics/gate.json"
        (tmp_path / gate).write_text(
            '{"passed_pair_keys": %s}' % str(list(passed)).replace("'", '"'),
            encoding="utf-8")
    return DS.build(root=tmp_path, pool="data/manifests/pool.jsonl",
                    injections="data/manifests/inj.jsonl", gate_json=gate,
                    out_path="data/manifests/out.jsonl")


def _emitted(tmp_path: Path):
    return io.read_jsonl(tmp_path / "data/manifests/out.jsonl", SampleRecord)


# --------------------------------------------------------------------------- #
# ★ Four arms become two contrast families                                     #
# --------------------------------------------------------------------------- #
def test_four_arms_map_onto_two_paired_families(tmp_path: Path):
    """The metric needs exactly two labels per attribute; the injector produces four arms."""
    rep = _write(tmp_path, _scene("s1"), scenes=("s1",))
    assert rep["samples_out"] == 4
    assert rep["by_family"] == {"dose_control": 2, "skin_tone": 2}
    fam = {r.sample_id: (r.metadata.attribute, r.metadata.variant_label)
           for r in _emitted(tmp_path)}
    assert fam["s1__skin_tone__dark"] == ("skin_tone", "dark")
    assert fam["s1__skin_tone__light"] == ("skin_tone", "light")
    assert fam["s1__skin_tone__sham"] == ("dose_control", "sham")
    assert fam["s1__skin_tone__sham_nonskin"] == ("dose_control", "sham_nonskin")


def test_one_item_joins_both_members_of_the_pair(tmp_path: Path):
    """original member -> `original_image_path`, edited member -> `edited_image_path`."""
    _write(tmp_path, _scene("s1"), scenes=("s1",))
    rec = next(r for r in _emitted(tmp_path) if r.sample_id == "s1__skin_tone__dark")
    assert str(rec.original_image_path).endswith("s1__dark__original.png")
    assert str(rec.edited_image_path).endswith("s1__dark__edited.png")
    # The detector measured a person; OmniEdit's own label says "global" and is not trusted.
    assert rec.content_category == ContentCategory.HUMAN


def test_render_index_is_one_so_no_editor_noise_floor_is_fabricated(tmp_path: Path):
    """Deterministic injection has no editor to re-run; an index >= 2 would invent a floor."""
    _write(tmp_path, _scene("s1"), scenes=("s1",))
    samples = _emitted(tmp_path)
    assert {r.metadata.render_index for r in samples} == {1}
    keys = keys_from_manifest(samples)
    assert compute_editor_noise([], keys, score_field="fine_score") == {}


# --------------------------------------------------------------------------- #
# ★ The gate is pairwise and takes the sham arms with it                       #
# --------------------------------------------------------------------------- #
def test_failed_gate_drops_the_scene_from_every_family(tmp_path: Path):
    """A sham floor computed over scenes the study arms never used is not that arm's floor."""
    rep = _write(tmp_path, _scene("s1") + _scene("s2"), passed={"s1__skin_tone"})
    assert rep["samples_out"] == 4
    # Counted in judged items: s2's four arms, not its two families.
    assert rep["dropped"]["failed_construction_gate"] == 4
    assert {r.metadata.base_sample_id for r in _emitted(tmp_path)} == {"s1"}


def test_without_a_gate_file_nothing_is_gated(tmp_path: Path):
    rep = _write(tmp_path, _scene("s1") + _scene("s2"))
    assert rep["gate_applied"] is False
    assert rep["samples_out"] == 8


# --------------------------------------------------------------------------- #
# ★ Incomplete units are dropped where they can be counted                     #
# --------------------------------------------------------------------------- #
def test_arm_missing_a_member_is_dropped_not_half_emitted(tmp_path: Path):
    """One member alone is not a judgeable item — the pair IS the item."""
    rows = [r for r in _scene("s1")
            if not (r.variant_label == "dark" and r.applied_to == "edited")]
    rep = _write(tmp_path, rows, scenes=("s1",))
    assert any("incomplete_members" in k for k in rep["dropped"])
    # `dark` is gone, so `skin_tone` is one-armed and must go too.
    assert rep["by_family"] == {"dose_control": 2}
    assert any(k.startswith("incomplete_pair:skin_tone") for k in rep["dropped"])


def test_one_armed_family_cannot_produce_a_paired_difference(tmp_path: Path):
    rows = [r for r in _scene("s1") if r.variant_label != "light"]
    rep = _write(tmp_path, rows, scenes=("s1",))
    assert rep["by_family"] == {"dose_control": 2}
    assert any(k.startswith("incomplete_pair:skin_tone") for k in rep["dropped"])


def test_base_missing_from_the_pool_is_dropped(tmp_path: Path):
    rep = _write(tmp_path, _scene("s9"), scenes=("s1",))
    assert rep["samples_out"] == 0
    assert rep["dropped"]["base_sample_missing_from_pool"] == 4


# --------------------------------------------------------------------------- #
# ★ Locality is a precondition, not a quality score                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [
    {"max_outside_diff": 1},
    {"outside_mask_identical": False},
    {"success": False},
])
def test_a_row_that_is_not_byte_local_never_reaches_a_judge(tmp_path: Path, kw):
    """The sham arms are outside the gate's pairwise check, so this is asserted here."""
    rows = _scene("s1")
    rows[0] = _inj("s1", "dark", "study", "original", **kw)
    rep = _write(tmp_path, rows, scenes=("s1",))
    assert rep["by_family"] == {"dose_control": 2}


def test_duplicate_member_row_is_reported_rather_than_last_write_wins(tmp_path: Path):
    rows = _scene("s1") + [_inj("s1", "dark", "study", "original",
                                injected_image_path="inj/OTHER.png")]
    rep = _write(tmp_path, rows, scenes=("s1",))
    assert rep["dropped"]["duplicate_member_row"] == 1
    rec = next(r for r in _emitted(tmp_path) if r.sample_id == "s1__skin_tone__dark")
    assert "OTHER" not in str(rec.original_image_path)


# --------------------------------------------------------------------------- #
# ★ Round trip: the emitted manifest actually drives the paired metric         #
# --------------------------------------------------------------------------- #
def _judged(sample_id: str, score: int, judge: str = "j") -> JudgeResult:
    return JudgeResult(
        result_id=f"score::{judge}::vanilla::{sample_id}",
        judge_model=judge, task_type="scoring", prompt_type="vanilla_scoring",
        raw_response_path=f"raw/{sample_id}.txt", sample_id=sample_id,
        overall_score=score, instruction_adherence=score, editing_quality=score,
        detail_preservation=score, fine_score=score * 3, score_scale=10, parse_success=True,
    )


def test_emitted_manifest_feeds_the_gap_with_the_documented_sign(tmp_path: Path):
    """skin_tone gap = dark - light, and the control family never joins it."""
    _write(tmp_path, _scene("s1") + _scene("s2"))
    keys = keys_from_manifest(_emitted(tmp_path))
    results = [
        _judged("s1__skin_tone__dark", 8), _judged("s1__skin_tone__light", 6),
        _judged("s2__skin_tone__dark", 7), _judged("s2__skin_tone__light", 6),
        _judged("s1__skin_tone__sham", 7), _judged("s1__skin_tone__sham_nonskin", 7),
        _judged("s2__skin_tone__sham", 7), _judged("s2__skin_tone__sham_nonskin", 7),
    ]
    stats = {s.attribute: s for s in compute_attribute_gaps(results, keys)}
    assert set(stats) == {"skin_tone", "dose_control"}
    skin = stats["skin_tone"]
    assert (skin.label_a, skin.label_b) == ("dark", "light")
    assert skin.n == 2
    assert skin.mean_gap == pytest.approx(4.5)          # (+6 and +3) in fine_score points
    dose = stats["dose_control"]
    assert (dose.label_a, dose.label_b) == ("sham", "sham_nonskin")
    assert dose.mean_gap == pytest.approx(0.0)


def test_missing_injection_manifest_reports_rather_than_raises(tmp_path: Path):
    (tmp_path / "data/manifests").mkdir(parents=True, exist_ok=True)
    io.write_jsonl(tmp_path / "data/manifests/pool.jsonl", [_pool_sample("s1")])
    rep = DS.build(root=tmp_path, pool="data/manifests/pool.jsonl",
                   injections="data/manifests/nope.jsonl", gate_json=None)
    assert rep["samples_out"] == 0 and "error" in rep and rep["out_path"] is None


# --------------------------------------------------------------------------- #
# ★ The two-manifest separation, added 2026-08-01 after it self-destructed once #
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[1]


def _yaml(rel: str) -> dict:
    import yaml

    return yaml.safe_load((_ROOT / rel).read_text(encoding="utf-8"))


def test_the_screen_reads_the_candidate_pool_and_the_injector_reads_the_screened_one():
    """The collision that would have silently emptied the paid screen.

    Pass 2 of the pool build originally wrote its SCREENED output back over the same path the
    construct screen reads as INPUT. Nothing errors: the moment pass 2 ran, the screen's input
    became its own already-passed subset, so the next increment would truthfully report zero
    new candidates while every remaining scene sat unscreened in a file nobody read again.
    """
    screen_in = _yaml("configs/fairness/construct_screen.yaml")["samples"]
    inject_in = _yaml("configs/fairness/skin_tone_inject_v4.yaml")["pool"]
    pool_out = _yaml("configs/fairness/pool_ds_omniedit.yaml")["out_path"]

    assert screen_in == pool_out, "the screen must measure pass 1's candidate pool"
    assert inject_in != screen_in, "pass 2's output must not overwrite the screen's input"
    assert "screened" in inject_in


def test_the_driver_redirects_pass_two_away_from_the_candidate_manifest():
    sh = (_ROOT / "scripts/fairness_ds.sh").read_text(encoding="utf-8")
    assert "--out data/manifests/samples_fairness_ds_v4_screened.jsonl" in sh
    assert "--screens-out results/v2_fairness/metrics/pool_screens_ds_v4_screened.jsonl" in sh


def test_k_is_unset_by_default_while_the_screen_is_incomplete():
    """`include_ids` filters BEFORE the seeded shuffle, so a k-prefix on a partial screen is
    not a subset of the k-prefix on the complete one. k unset keeps every passing scene, which
    makes topping the screen up purely additive."""
    sh = (_ROOT / "scripts/fairness_ds.sh").read_text(encoding="utf-8")
    assert 'K="${K:-}"' in sh
    assert '${K:+--k "$K"}' in sh, "an empty K must omit the flag, not pass --k ''"


# --------------------------------------------------------------------------- #
# ★★ The two families cover different populations — measured 2026-08-01        #
# --------------------------------------------------------------------------- #
def test_the_report_names_the_common_subset_the_controlled_contrast_needs(tmp_path):
    """`sham_nonskin` is impossible when skin exceeds 55.6% of the person.

    The pilot's 20 scenes split 10/10 on exactly that, with `person_frac` identical between the
    groups (0.346 vs 0.345) and skin/person 0.656 vs 0.304. So `dose_control` lives on the more
    clothed half. Subtracting a control measured on smaller skin regions understates the
    artefact penalty and inflates the residual attributed to skin tone — the unsafe direction.
    The report must therefore surface the scenes where BOTH families exist.
    """
    # s1 is a clothed subject (all four arms); s2 is mostly-skin, so `sham_nonskin` was
    # impossible and the injector wrote nothing for it — exactly the pilot's 10/10 split.
    no_dose = tuple(a for a in ARMS if a[0] != "sham_nonskin")
    rep = _write(tmp_path, _scene("s1") + _scene("s2", arms=no_dose))

    assert rep["scenes_by_family"]["skin_tone"] > rep["scenes_by_family"]["dose_control"]
    assert rep["common_subset_scenes"] == rep["scenes_by_family"]["dose_control"]
    assert rep["common_subset_mde_sd"] is not None
    assert set(rep["mde_by_family"]) == {"skin_tone", "dose_control"}
    # A smaller family must not report a smaller (better) MDE.
    assert rep["mde_by_family"]["dose_control"] >= rep["mde_by_family"]["skin_tone"]


def test_a_one_armed_dose_control_is_dropped_with_a_named_reason(tmp_path):
    """Silence is the failure mode this project keeps paying for: a scene missing
    `sham_nonskin` must not simply vanish into a smaller n."""
    no_dose = tuple(a for a in ARMS if a[0] != "sham_nonskin")
    rep = _write(tmp_path, _scene("s1") + _scene("s2", arms=no_dose))
    reasons = [k for k in rep["dropped"] if k.startswith("incomplete_pair:dose_control")]
    assert reasons, rep["dropped"]
