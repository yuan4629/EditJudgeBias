"""Tests for the D-class person/face/head/skin region instrument.

Every test pins a measured failure mode. The instrument exists because the 2026-07-30
D-class gate collapsed on `scene_preserved` (two auditors at Cohen's kappa +0.130), and the
fix is to make locality arithmetic instead of a judgment -- so the arithmetic has to be
trustworthy, and an instrument that fails silently is worse than no instrument at all.

Synthetic only: no model weights, no data tree, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from edit_judge_bias.fairness import person_region as PR


# --------------------------------------------------------------------------- #
# The self-check is the instrument's own contract                              #
# --------------------------------------------------------------------------- #
def test_self_check_passes_and_covers_every_named_control():
    checks = PR.self_check(strict=True)
    names = {c.name for c in checks}
    for required in (
        "skin_needs_person",
        "skin_without_person_overfires",
        "feather_support_contains_mask",
        "cache_roundtrip_bit_identical",
        "config_digest_tracks_config",
        "outside_identity_catches_one_pixel",
        "fails_loudly_on_unreadable",
    ):
        assert required in names, f"self_check lost its {required!r} control"
    assert all(c.passed for c in checks)


def test_self_check_raises_rather_than_returning_a_failed_report():
    """strict=True must RAISE. A silently-failed instrument is the whole hazard here.

    Direct descendant of a real incident: the ORB near-duplicate check once reported
    "0 duplicates" because it was handed paths instead of signatures and the TypeError was
    swallowed by `except Exception: continue`.
    """
    # Force a failure by making the skin rule impossible to satisfy: an empty Cr window
    # means the chroma test can never fire, so `skin_without_person_overfires` must fail.
    broken = {"skin": {"cr": (255, 255), "cb": (255, 255)}}
    with pytest.raises(PR.RegionError, match="self-check failed"):
        PR.self_check(cfg=broken, strict=True)
    # And with strict=False the same config must report the failure instead of hiding it.
    checks = PR.self_check(cfg=broken, strict=False)
    assert any(not c.passed for c in checks)


# --------------------------------------------------------------------------- #
# ★ The skin rule needs a person prior                                         #
# --------------------------------------------------------------------------- #
def test_skin_mask_is_bounded_by_the_person_prior():
    """★ MEASURED: the bare chroma rule selected a median 0.285 of the WHOLE FRAME.

    Over 57 candidate images the classic YCrCb+HSV skin test fired on sand and wood, with a
    maximum of 0.994 of the frame. Recolouring a beach is not a skin-tone manipulation, so
    the person prior is mandatory for study rows.
    """
    frame = np.zeros((100, 100, 3), dtype="uint8")
    frame[:, :] = (200, 150, 130)                 # skin-coloured everywhere
    person = np.zeros((100, 100), dtype=bool)
    person[:10, :] = True                          # only 10% of the frame is a person

    bounded, stats = PR.skin_mask(frame, person)
    assert bounded.mean() <= 0.10 + 1e-9
    assert stats["require_person"] is True
    assert stats["chroma_frac"] > 0.9, "the control only means something if chroma over-fires"

    unbounded, _ = PR.skin_mask(frame, None, require_person=False)
    assert unbounded.mean() >= 0.9


def test_skin_mask_refuses_to_run_without_a_person_when_required():
    """Passing no person mask must raise, not quietly fall back to the whole frame."""
    frame = np.full((32, 32, 3), 200, dtype="uint8")
    with pytest.raises(PR.RegionError, match="needs a person mask"):
        PR.skin_mask(frame, None, require_person=True)


# --------------------------------------------------------------------------- #
# ★ Face boxes need the person overlap filter                                  #
# --------------------------------------------------------------------------- #
def test_face_box_off_the_person_is_rejected():
    """★ MEASURED: Haar's largest "faces" in this pool were a burger and a TV remote.

    Face area fractions of 0.629 (a burger, instruction "Remove the lettuce from the
    burger") and 0.473 (a TV remote, "turn the remote into a pizza"). Admitting those is
    how the previous attempt ended up judging gulls, a squirrel, a zebra and dolls.

    Driven through `face_boxes` with a stubbed detector so the test does not depend on Haar
    firing on synthetic pixels.
    """
    frame = np.zeros((100, 100, 3), dtype="uint8")
    person = np.zeros((100, 100), dtype=bool)
    person[60:100, 60:100] = True                  # the person is bottom-right

    on_person = (65, 65, 85, 85)
    off_person = (5, 5, 25, 25)                    # e.g. the burger

    import edit_judge_bias.fairness.person_region as mod
    original = mod._raw_face_boxes
    try:
        mod._raw_face_boxes = lambda rgb, *, cfg: [off_person, on_person]
        kept = PR.face_boxes(frame, person=person)
    finally:
        mod._raw_face_boxes = original

    assert [box for box, _ in kept] == [on_person]
    assert kept[0][1] == pytest.approx(1.0)


def test_face_boxes_without_a_person_mask_keep_everything():
    """The filter is opt-in via `person=`; the pool builder always passes one."""
    frame = np.zeros((60, 60, 3), dtype="uint8")
    import edit_judge_bias.fairness.person_region as mod
    original = mod._raw_face_boxes
    try:
        mod._raw_face_boxes = lambda rgb, *, cfg: [(1, 1, 11, 11)]
        kept = PR.face_boxes(frame, person=None)
    finally:
        mod._raw_face_boxes = original
    assert len(kept) == 1 and kept[0][1] == 1.0


# --------------------------------------------------------------------------- #
# Loud failure, never an empty mask                                            #
# --------------------------------------------------------------------------- #
def test_unreadable_image_raises_instead_of_yielding_an_empty_mask(tmp_path: Path):
    """An empty mask would pass every downstream check as "nothing to change".

    That is strictly worse than a crash: the row would be silently dropped from the pool
    (or injected as a no-op) and the run would look healthy.
    """
    with pytest.raises(PR.RegionError, match="unreadable image"):
        PR.person_mask(tmp_path / "does_not_exist.png")

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a png at all")
    with pytest.raises(PR.RegionError):
        PR.person_mask(corrupt)


def test_rgb_array_with_wrong_shape_raises():
    with pytest.raises(PR.RegionError):
        PR._as_rgb_array(np.zeros((10, 10), dtype="uint8"))


# --------------------------------------------------------------------------- #
# Cache determinism                                                            #
# --------------------------------------------------------------------------- #
def test_mask_png_roundtrip_is_bit_identical(tmp_path: Path):
    """The cached PNG is the source of truth, so it must survive the round trip exactly.

    DeepLabV3 argmax can flip boundary pixels between CUDA and CPU. If the gate recomputed
    the mask instead of re-reading the PNG, byte-identity could fail for a reason unrelated
    to the injection.
    """
    mask = np.zeros((50, 40), dtype=bool)
    mask[10:30, 5:25] = True
    path = PR.save_mask_png(mask, tmp_path / "m.png")
    assert (PR.load_mask_png(path) == mask).all()

    from PIL import Image
    with Image.open(path) as im:
        assert im.mode == "L"
        assert set(np.unique(np.asarray(im))) <= {0, 255}


def test_config_digest_changes_with_the_config(tmp_path: Path):
    base = dict(PR.DEFAULT_REGION_CONFIG)
    d1 = PR.config_digest(base)
    assert d1 == PR.config_digest(dict(base)), "digest must be stable for equal configs"
    assert d1 != PR.config_digest({**base, "infer_long_edge": 999})
    assert d1 != PR.config_digest({**base, "person_prob_threshold": 0.9})


def test_cache_path_flattens_the_image_path_so_sources_cannot_collide(tmp_path: Path):
    """Two trees in this pool reuse filenames, so a basename key would collide.

    `model00/H_00_05.jpg` exists under both `sourceimg_h` and `targetimg_h`.
    """
    a = PR.cache_path_for("tmp_data/editing_all/sourceimg_h/model00/H_00_05.jpg",
                          "person", cache_dir=tmp_path, digest="abc123")
    b = PR.cache_path_for("tmp_data/editing_all/targetimg_h/model00/H_00_05.jpg",
                          "person", cache_dir=tmp_path, digest="abc123")
    assert a != b


def test_mask_sha256_proves_two_arms_shared_one_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    assert PR.mask_sha256(mask) == PR.mask_sha256(mask.copy())
    other = mask.copy()
    other[0, 0] = True
    assert PR.mask_sha256(mask) != PR.mask_sha256(other)


# --------------------------------------------------------------------------- #
# Geometry helpers                                                             #
# --------------------------------------------------------------------------- #
def test_head_region_is_clipped_to_the_person():
    """Growing the face box must never escape the person mask.

    Otherwise a generative head flip would repaint background it was never authorised to
    touch, and the arms would differ in scene as well as in attribute.
    """
    person = np.zeros((100, 100), dtype=bool)
    person[30:70, 30:70] = True
    head = PR.head_region((40, 40, 60, 60), person)
    assert head.any()
    assert not (head & ~person).any()


def test_feather_support_is_a_superset_of_the_mask_and_bounded():
    mask = np.zeros((80, 80), dtype=bool)
    mask[30:50, 30:50] = True
    alpha = PR.feather(mask, radius_frac=0.05, short_edge=80)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert ((alpha > 0) | ~mask).all(), "alpha>0 must contain the whole mask"
    assert (alpha > 0).sum() >= mask.sum()


def test_feather_with_zero_radius_is_the_mask_itself():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    alpha = PR.feather(mask, radius_frac=0.0, short_edge=20)
    assert ((alpha > 0) == mask).all()


# --------------------------------------------------------------------------- #
# The invariant that replaced `scene_preserved`                                #
# --------------------------------------------------------------------------- #
def test_outside_mask_identical_has_no_tolerance():
    """One stray pixel is a bug (lossy resave, resize, alpha leak), not noise."""
    base = np.zeros((40, 40, 3), dtype="uint8")
    inside = np.zeros((40, 40), dtype=bool)
    inside[20:30, 20:30] = True

    same = base.copy()
    ok, worst = PR.outside_mask_identical(base, same, inside)
    assert ok and worst == 0

    leaked = base.copy()
    leaked[0, 0] = (1, 0, 0)
    ok, worst = PR.outside_mask_identical(base, leaked, inside)
    assert not ok and worst == 1

    # A change INSIDE the mask is expected and must not trip the check.
    changed = base.copy()
    changed[20:30, 20:30] = (200, 100, 50)
    ok, _ = PR.outside_mask_identical(base, changed, inside)
    assert ok


def test_outside_mask_identical_rejects_mismatched_shapes():
    base = np.zeros((10, 10, 3), dtype="uint8")
    with pytest.raises(PR.RegionError):
        PR.outside_mask_identical(base, np.zeros((8, 8, 3), dtype="uint8"),
                                  np.zeros((10, 10), dtype=bool))
    with pytest.raises(PR.RegionError):
        PR.outside_mask_identical(base, base.copy(), np.zeros((4, 4), dtype=bool))


# --------------------------------------------------------------------------- #
# The artefact control region                                                  #
# --------------------------------------------------------------------------- #
def test_nonskin_control_stays_on_the_person_and_matches_area():
    """The artefact control must be the SAME person's non-skin pixels, area-matched.

    Equal *area* is not equal *saliency*: recolouring a wall is less noticed than
    recolouring a person, so a background control estimates only a LOWER BOUND on the
    artefact penalty -- and subtracting a lower bound inflates the residual one would then
    call bias, an error in the unsafe direction.
    """
    person = np.zeros((100, 100), dtype=bool)
    person[20:80, 20:80] = True
    skin = np.zeros((100, 100), dtype=bool)
    skin[20:40, 20:40] = True                       # face; the rest of the person is clothing

    region, stats = PR.nonskin_same_person_region(person, skin, target_area=int(skin.sum()))
    assert region is not None
    assert not (region & skin).any(), "the control must not overlap skin"
    assert not (region & ~person).any(), "the control must stay on the person"
    assert 0.8 <= stats["area_ratio"] <= 1.25


def test_nonskin_control_is_dropped_not_faked_when_no_region_exists():
    """A scene with no non-skin person pixels loses THIS ARM ONLY, and it is counted."""
    person = np.zeros((40, 40), dtype=bool)
    person[10:20, 10:20] = True
    skin = person.copy()                            # the person is entirely skin
    region, stats = PR.nonskin_same_person_region(person, skin, target_area=50)
    assert region is None
    assert stats["reason"] == "no_nonskin_region_on_person"


def test_nonskin_control_rejects_a_bad_area_match():
    """Better to drop the arm than to report a control carrying a different pixel budget."""
    person = np.zeros((60, 60), dtype=bool)
    person[0:30, 0:30] = True
    skin = np.zeros((60, 60), dtype=bool)
    skin[0:29, 0:29] = True                         # almost all of the person is skin
    region, stats = PR.nonskin_same_person_region(person, skin, target_area=800)
    assert region is None
    assert "reason" in stats


# --------------------------------------------------------------------------- #
# Config merging                                                               #
# --------------------------------------------------------------------------- #
def test_user_config_merges_one_level_into_subdicts():
    merged = PR._merge_cfg({"skin": {"y_min": 99}})
    assert merged["skin"]["y_min"] == 99
    assert merged["skin"]["cr"] == PR.DEFAULT_REGION_CONFIG["skin"]["cr"], (
        "overriding one skin key must not drop the others"
    )
    assert merged["face"] == PR.DEFAULT_REGION_CONFIG["face"]


def test_defaults_require_the_person_prior():
    """A regression guard on the value that matters most in this module."""
    assert PR.DEFAULT_REGION_CONFIG["skin"]["require_person"] is True
    assert PR.DEFAULT_REGION_CONFIG["face"]["min_person_overlap"] >= 0.5
