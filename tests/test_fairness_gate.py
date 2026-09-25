"""Tests for the D-class construction gate and the deterministic injection runner.

The gate replaces an MLLM judgment that collapsed (two auditors at Cohen's kappa +0.130 on
`scene_preserved`, -0.028 on pair validity). Its whole value is that it is arithmetic, so every
test here pins a way the arithmetic could go quietly wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from edit_judge_bias.data import io
from edit_judge_bias.experiments import run_attribute_inject as INJ
from edit_judge_bias.experiments import run_construction_gate as GATE
from edit_judge_bias.fairness import attribute_injectors as AI
from edit_judge_bias.fairness.records import AttributeInjectionRecord
from edit_judge_bias.metrics.fairness_metrics import min_attainable_pvalue


def _row(**kw) -> AttributeInjectionRecord:
    base = dict(
        injection_id="s__skin_tone__dark__original",
        base_sample_id="s", attribute="skin_tone", variant_label="dark",
        role=AI.ROLE_STUDY, applied_to=AI.APPLIED_ORIGINAL,
        source_image_path="o.png", injected_image_path="i.png",
        instruction="Add a hat", source_dataset="EBench-18K", edit_type="add",
        edit_model="m0", mask_kind="skin", mask_sha256="abc123", mask_frac=0.15,
        achieved_delta_ita=-30.0, delta_e00_mean=8.0,
        max_outside_diff=0, max_inside_diff=30, outside_mask_identical=True,
    )
    base.update(kw)
    return AttributeInjectionRecord(**base)


def _cfg(**kw) -> dict:
    cfg = dict(GATE.DEFAULTS)
    cfg["verify_pixels"] = False          # unit tests check the logic, not the filesystem
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------- #
# ★ No tolerance on locality                                                   #
# --------------------------------------------------------------------------- #
def test_one_nonzero_outside_pixel_fails_the_row(tmp_path: Path):
    """A single stray pixel is a bug (lossy resave / resize / alpha leak), not noise."""
    ok, reasons = GATE._check_row(_row(max_outside_diff=1), tmp_path, _cfg())
    assert ok is False
    assert any("outside_mask_diff" in r for r in reasons)


def test_outside_identical_false_fails_even_at_zero_diff(tmp_path: Path):
    ok, reasons = GATE._check_row(
        _row(outside_mask_identical=False), tmp_path, _cfg()
    )
    assert ok is False
    assert "outside_mask_not_identical" in reasons


def test_a_clean_row_passes(tmp_path: Path):
    ok, reasons = GATE._check_row(_row(), tmp_path, _cfg())
    assert ok is True and reasons == []


# --------------------------------------------------------------------------- #
# ★ The dose floor applies to the ORIGINAL member only                         #
# --------------------------------------------------------------------------- #
def test_ita_dose_floor_applies_to_the_original_member(tmp_path: Path):
    ok, reasons = GATE._check_row(
        _row(applied_to=AI.APPLIED_ORIGINAL, achieved_delta_ita=-19.0), tmp_path, _cfg()
    )
    assert ok is False
    assert any("below_floor" in r for r in reasons)


def test_ita_dose_floor_does_not_apply_to_the_edited_member(tmp_path: Path):
    """★ MEASURED. 16 of 62 edited rows read below the floor and NONE of them is a defect.

    The mask is derived from the original, so on the edited image it covers different pixels
    (one scene's instruction replaces the background outright), and applying the same Lab offset
    to a different starting colour yields a different ITA change because ITA is nonlinear:
    `ebench_H_09_02` reads +17.6 on the original and -38.8 under the same mask on the edited
    image. Gating the edited member on ITA would have discarded 9 good scenes. The PHYSICAL
    dose is what must match, and it does (mean dE00 6.07 vs 5.89 on that very pair).
    """
    ok, reasons = GATE._check_row(
        _row(injection_id="s__skin_tone__dark__edited",
             applied_to=AI.APPLIED_EDITED, achieved_delta_ita=-17.7),
        tmp_path, _cfg(),
    )
    assert ok is True, reasons


def test_sham_rows_are_exempt_from_the_dose_floor(tmp_path: Path):
    """The sham arm's whole point is a zero dose; a floor would reject it by definition."""
    ok, reasons = GATE._check_row(
        _row(role=AI.ROLE_SHAM, variant_label="sham",
             achieved_delta_ita=0.0, max_inside_diff=3),
        tmp_path, _cfg(),
    )
    assert ok is True, reasons


def test_study_arm_that_changed_nothing_fails(tmp_path: Path):
    """A silent no-op would otherwise pass every other check."""
    ok, reasons = GATE._check_row(_row(max_inside_diff=1), tmp_path, _cfg())
    assert ok is False
    assert any("inside_diff" in r for r in reasons)


@pytest.mark.parametrize("mask_frac", [0.0001, 0.95])
def test_mask_size_bounds(tmp_path: Path, mask_frac):
    """Too small is not a manipulation; too large stops being local.

    This study measured GLOBAL tonal cues (`saturation`, `aesthetic_filter`) as null on 4/5
    judges while LOCAL readable ones deflate on 5/5, so locality is what keeps the arm
    interpretable at all.
    """
    ok, reasons = GATE._check_row(_row(mask_frac=mask_frac), tmp_path, _cfg())
    assert ok is False
    assert any("mask_frac" in r for r in reasons)


# --------------------------------------------------------------------------- #
# ★ Pairwise application and the shared mask                                   #
# --------------------------------------------------------------------------- #
def _write_manifest(tmp_path: Path, rows) -> Path:
    path = tmp_path / "data/manifests/inj.jsonl"
    io.write_jsonl(path, rows)
    return path


def _full_pair(**overrides):
    """Four study rows: 2 arms x 2 members, all clean unless overridden."""
    rows = []
    for label, delta in (("dark", -30.0), ("light", 30.0)):
        for member in (AI.APPLIED_ORIGINAL, AI.APPLIED_EDITED):
            rows.append(_row(
                injection_id=f"s__skin_tone__{label}__{member}",
                variant_label=label, applied_to=member, achieved_delta_ita=delta,
            ))
    for row in rows:
        for key, value in overrides.items():
            setattr(row, key, value)
    return rows


def _run_gate(tmp_path: Path, rows, **cfg_over) -> dict:
    manifest = _write_manifest(tmp_path, rows)
    cfg = {
        "injections": manifest.relative_to(tmp_path).as_posix(),
        "out_json": "gate.json",
        "verify_pixels": False,
        **cfg_over,
    }
    path = tmp_path / "gate.yaml"
    path.write_text(json.dumps(cfg), encoding="utf-8")   # YAML is a superset of JSON
    return GATE.run(path, root=tmp_path)


def test_one_bad_arm_drops_the_whole_pair(tmp_path: Path):
    """★ The gate is PAIRWISE. A one-sided drop leaves an orphan the paired metric discards
    later and silently -- the same reasoning the published `build_fairness_samples` gate uses.
    """
    rows = _full_pair()
    rows[0].max_outside_diff = 5          # break exactly one of the four rows
    rows[0].outside_mask_identical = False
    report = _run_gate(tmp_path, rows)
    assert report["pairs_total"] == 1
    assert report["pairs_passed"] == 0
    assert report["pairs_dropped"] == 1


def test_a_clean_pair_passes(tmp_path: Path):
    report = _run_gate(tmp_path, _full_pair())
    assert report["pairs_passed"] == 1
    assert report["rows_failing"] == 0


def test_arms_must_share_one_mask_digest(tmp_path: Path):
    """Shared mask is what makes segmentation error common-mode instead of a between-arm
    difference. Without this check a mask bug could masquerade as an attribute effect.
    """
    rows = _full_pair()
    for row in rows:
        if row.variant_label == "light":
            row.mask_sha256 = "different"
    report = _run_gate(tmp_path, rows)
    assert report["pairs_passed"] == 0
    assert report["mask_conflicts"] == 1


def test_incomplete_arms_are_dropped(tmp_path: Path):
    rows = [r for r in _full_pair() if r.variant_label == "dark"]
    report = _run_gate(tmp_path, rows)
    assert report["pairs_passed"] == 0


def test_cross_member_dose_inconsistency_is_caught(tmp_path: Path):
    """dE00 is the physical dose; if it differs wildly between members the pair is not one pair."""
    rows = _full_pair()
    for row in rows:
        if row.applied_to == AI.APPLIED_EDITED:
            row.delta_e00_mean = 40.0     # 5x the original member's 8.0
    report = _run_gate(tmp_path, rows)
    assert report["pairs_passed"] == 0
    assert report["dose_conflicts"] >= 1


def test_gate_reports_mde_and_min_attainable_p(tmp_path: Path):
    report = _run_gate(tmp_path, _full_pair())
    assert report["mde_sd"] is not None
    assert report["min_attainable_p"] == pytest.approx(1.0)   # n=1 -> capped


# --------------------------------------------------------------------------- #
# ★ Re-verification against the SAVED pixels, including the resize case        #
# --------------------------------------------------------------------------- #
def _save(path: Path, arr) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_pixel_reverify_catches_a_leak_the_manifest_denies(tmp_path: Path):
    """The manifest value was computed in memory BEFORE the write, so it cannot catch a bad
    save. Re-reading is the only way to know what is actually on disk.
    """
    rng = np.random.default_rng(3)
    source = rng.integers(0, 255, (40, 40, 3), dtype="uint8")
    leaked = source.copy()
    leaked[:, :] = (leaked.astype("int16") + 30).clip(0, 255).astype("uint8")  # whole frame
    _save(tmp_path / "o.png", source)
    _save(tmp_path / "i.png", leaked)

    row = _row(mask_frac=0.15)            # claims only 15% could have changed
    ok, worst = GATE._verify_pixels(row, tmp_path)
    assert ok is False


def test_pixel_reverify_accepts_a_change_inside_the_declared_support(tmp_path: Path):
    rng = np.random.default_rng(4)
    source = rng.integers(0, 255, (40, 40, 3), dtype="uint8")
    injected = source.copy()
    injected[:6, :] = 0                   # 15% of the frame
    _save(tmp_path / "o.png", source)
    _save(tmp_path / "i.png", injected)

    ok, _worst = GATE._verify_pixels(_row(mask_frac=0.16), tmp_path)
    assert ok is True


def test_resized_source_is_reverified_with_the_same_resample(tmp_path: Path):
    """★ A GATE BUG THE GATE CAUGHT ON ITSELF.

    12 of the 31 D-S scenes ship a smaller original than edited image, so the edited member is
    resampled onto the original's grid before injection (the mask comes from the original and
    both members must share a pixel grid). The first version of this check re-read the edited
    source at its ON-DISK size and failed all 24 such rows -- 12 scenes x 2 study arms -- on a
    shape mismatch that was not a defect. `source_resized` on the row tells the gate to
    reproduce the identical PIL BICUBIC resample.
    """
    from PIL import Image

    rng = np.random.default_rng(5)
    big = rng.integers(0, 255, (80, 80, 3), dtype="uint8")
    _save(tmp_path / "o.png", big)                       # source on disk: 80x80
    small = np.asarray(Image.fromarray(big).resize((40, 40), Image.BICUBIC))
    _save(tmp_path / "i.png", small)                     # injected: 40x40, unchanged content

    without = GATE._verify_pixels(_row(source_resized=False), tmp_path)
    assert without[0] is False, "a shape mismatch must fail when no resize was declared"

    with_flag = GATE._verify_pixels(_row(source_resized=True, mask_frac=0.05), tmp_path)
    assert with_flag[0] is True, "declaring the resize must let the gate reproduce it"


# --------------------------------------------------------------------------- #
# min_attainable_pvalue — the arithmetic behind G1's hard stop                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n, expected", [
    (4, 0.125), (5, 0.0625), (6, 0.03125), (7, 0.015625),
])
def test_min_attainable_pvalue(n, expected):
    """★ n=5 cannot reach p<0.05 at ANY effect size, which is why the stopped track's 5+5
    auditor intersection was incapable rather than merely underpowered.
    """
    assert min_attainable_pvalue(n) == pytest.approx(expected)


def test_min_attainable_pvalue_edge_cases():
    assert min_attainable_pvalue(0) is None
    assert min_attainable_pvalue(-3) is None
    assert min_attainable_pvalue(1) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The injection runner                                                         #
# --------------------------------------------------------------------------- #
def test_injection_id_distinguishes_the_two_members():
    """Both members are separate rows, so `applied_to` must be part of the id or the second
    write would look like a duplicate and be skipped by resume.
    """
    a = INJ._injection_id("s", "skin_tone", "dark", AI.APPLIED_ORIGINAL)
    b = INJ._injection_id("s", "skin_tone", "dark", AI.APPLIED_EDITED)
    assert a != b
    assert a.endswith("__original") and b.endswith("__edited")


def test_injection_ids_are_unique_across_arms_and_members():
    ids = {
        INJ._injection_id("s", "skin_tone", label, member)
        for label in ("dark", "light", "sham", "sham_nonskin")
        for member in INJ.MEMBERS
    }
    assert len(ids) == 8


def test_resume_skips_ids_already_on_disk(tmp_path: Path):
    manifest = _write_manifest(tmp_path, [_row()])
    done = INJ._load_done(manifest)
    assert "s__skin_tone__dark__original" in done


def test_load_done_on_a_missing_manifest_is_empty(tmp_path: Path):
    assert INJ._load_done(tmp_path / "nope.jsonl") == set()


def test_nan_never_reaches_the_manifest():
    """`ita_before` is NaN for an empty mask; a NaN in JSONL is not valid JSON."""
    assert INJ._finite(float("nan")) is None
    assert INJ._finite(None) is None
    assert INJ._finite(30.123456) == pytest.approx(30.1235)


# --------------------------------------------------------------------------- #
# ★ The original-derived mask must still land on skin in the RE-RENDERED member #
# --------------------------------------------------------------------------- #
def test_coverage_check_is_off_by_default(tmp_path: Path):
    """The published v3 gate must reproduce: its records predate this measurement."""
    assert GATE.DEFAULTS["min_edited_mask_coverage"] is None
    ok, _reasons = GATE._check_row(
        _row(applied_to=AI.APPLIED_EDITED, edited_mask_skin_coverage=None),
        tmp_path, _cfg())
    assert ok is True


def test_low_coverage_on_the_edited_member_fails(tmp_path: Path):
    """A mask that recolours skin on one side and background on the other is not a pair.

    Measured over 60 random OmniEdit pairs: median coverage 0.971, but 1 in 60 below 0.5.
    Small, silent, and exactly the class of defect this project keeps finding after the fact.
    """
    ok, reasons = GATE._check_row(
        _row(applied_to=AI.APPLIED_EDITED, edited_mask_skin_coverage=0.21),
        tmp_path, _cfg(min_edited_mask_coverage=0.5))
    assert ok is False
    assert any("edited_mask_coverage" in r for r in reasons)


def test_healthy_coverage_passes(tmp_path: Path):
    ok, _reasons = GATE._check_row(
        _row(applied_to=AI.APPLIED_EDITED, edited_mask_skin_coverage=0.97),
        tmp_path, _cfg(min_edited_mask_coverage=0.5))
    assert ok is True


def test_an_unmeasured_coverage_is_not_a_pass(tmp_path: Path):
    """A missing measurement is not a passing one -- the same rule as the screen's fields."""
    ok, reasons = GATE._check_row(
        _row(applied_to=AI.APPLIED_EDITED, edited_mask_skin_coverage=None),
        tmp_path, _cfg(min_edited_mask_coverage=0.5))
    assert ok is False
    assert any("unmeasured" in r for r in reasons)


def test_the_original_member_is_exempt(tmp_path: Path):
    """On the original the mask IS the one derived from it, so coverage is 1.0 by construction.

    Checking it there would dilute the gate's view of the only number that carries information.
    """
    ok, _reasons = GATE._check_row(
        _row(applied_to=AI.APPLIED_ORIGINAL, edited_mask_skin_coverage=None),
        tmp_path, _cfg(min_edited_mask_coverage=0.5))
    assert ok is True
