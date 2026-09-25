"""Tests for the degenerate-homography screen on ORB near-duplicate detection.

Added 2026-08-01, after the OmniEdit A8 gate produced this project's first 7 ORB hits above
`MIN_INLIERS` and **all seven were false positives** — every one against a low-texture
incumbent image (a dark kitchen drawer, a hazy cityscape, a white dog in snow). The failure is
dangerous precisely because it looks like a real find: the inlier count clears the calibrated
threshold, and only the geometry shows the fit is rank-deficient.

Two directions are pinned here, and the second matters as much as the first: the screen must
kill degenerate fits AND must not touch a genuine same-photo match.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from edit_judge_bias.data import overlap


def _diag(**kw) -> dict:
    base = {"good": 200, "inliers": 60, "det_h": 1.0,
            "hull_frac_a": 0.6, "hull_frac_b": 0.6,
            "keypoints_a": 500, "keypoints_b": 500}
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# ★ The three checks, and the proof that none of them is sufficient alone      #
# --------------------------------------------------------------------------- #
def test_a_healthy_fit_is_not_flagged():
    assert overlap.degenerate_reason(_diag()) is None


def test_rank_deficient_homography_is_flagged():
    """|det H| = 0.000 in all seven measured false positives."""
    assert "degenerate_homography" in overlap.degenerate_reason(_diag(det_h=0.0))


def test_inliers_piled_onto_one_spot_are_flagged():
    """A genuine same-photo match spreads over the frame; a pile-up does not."""
    reason = overlap.degenerate_reason(_diag(hull_frac_b=0.0))
    assert reason is not None and "not_spread" in reason


def test_more_inliers_than_keypoints_is_flagged():
    """Many source points matched onto one target point is a pile-up, not a correspondence."""
    assert overlap.degenerate_reason(
        _diag(inliers=300, keypoints_b=267)) == "more_inliers_than_keypoints"


def test_a_keypoint_floor_alone_would_have_missed_two_of_the_seven():
    """The measured reason all three checks are kept together.

    Two of the seven false positives had a perfectly ordinary keypoint budget on both sides;
    what gave them away was the geometry. A screen built on keypoint counts alone passes this
    fixture, which is why it is not the screen.
    """
    sneaky = _diag(inliers=56, keypoints_a=500, keypoints_b=500, det_h=2.4e-14,
                   hull_frac_a=0.31, hull_frac_b=0.0)
    assert sneaky["inliers"] <= min(sneaky["keypoints_a"], sneaky["keypoints_b"])
    assert overlap.degenerate_reason(sneaky) is not None


def test_a_fit_too_weak_to_judge_is_left_alone():
    """Below 4 inliers there is no homography to call degenerate; do not invent a verdict."""
    assert overlap.degenerate_reason(_diag(inliers=3, det_h=None)) is None


# --------------------------------------------------------------------------- #
# ★ End to end on real pixels: the screen must not break a true match          #
# --------------------------------------------------------------------------- #
def _textured(seed: int, size: int = 320):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    import cv2

    # Blur a little so ORB finds corners rather than pure noise.
    return cv2.GaussianBlur(img, (3, 3), 0)


def _sig(img):
    import cv2

    orb = cv2.ORB_create(nfeatures=overlap.ORB_FEATURES)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(grey, None)
    return (kp, des)


def test_self_match_still_saturates_with_the_screen_on():
    """The instrument control the project runs before believing any verdict.

    If the screen broke this, every duplicate check would silently start reporting zero — the
    exact failure mode (`ransac_inliers(path, path)` swallowed by a bare except) that once made
    this project's duplicate check return "0 duplicates" and nearly published a wrong pool.
    """
    sig = _sig(_textured(42))
    raw = overlap.ransac_inliers(sig, sig)
    screened = overlap.ransac_inliers(sig, sig, screen_degenerate=True)
    assert raw > 100
    assert screened == raw


def test_a_crop_of_the_same_photo_survives_the_screen():
    """The pattern the near-duplicate dimension exists for: same photo, cropped + re-encoded."""
    import cv2

    img = _textured(7)
    crop = cv2.resize(img[24:296, 24:296], (320, 320))
    a, b = _sig(img), _sig(crop)
    screened = overlap.ransac_inliers(a, b, screen_degenerate=True)
    assert screened >= overlap.MIN_INLIERS
    assert screened == overlap.ransac_inliers(a, b)


def test_unrelated_textures_stay_below_threshold_either_way():
    a, b = _sig(_textured(1)), _sig(_textured(2))
    assert overlap.ransac_inliers(a, b) < overlap.MIN_INLIERS
    assert overlap.ransac_inliers(a, b, screen_degenerate=True) < overlap.MIN_INLIERS


def test_diagnostics_expose_what_the_verdict_rests_on():
    sig = _sig(_textured(3))
    diag = overlap.ransac_diagnostics(sig, sig)
    assert diag["inliers"] > 0
    assert diag["det_h"] is not None
    assert diag["hull_frac_a"] == pytest.approx(diag["hull_frac_b"], abs=1e-6)
    assert diag["keypoints_a"] == diag["keypoints_b"] > 0


# --------------------------------------------------------------------------- #
# ★ Default-off is a reproducibility contract, not a preference                #
# --------------------------------------------------------------------------- #
def test_the_screen_is_off_by_default_so_published_pools_reproduce():
    import inspect

    sig = inspect.signature(overlap.ransac_inliers)
    assert sig.parameters["screen_degenerate"].default is False
    from edit_judge_bias.experiments import build_fairness_pool as BFP

    assert inspect.signature(BFP.build).parameters["orb_screen_degenerate"].default is False
    assert inspect.signature(
        BFP.collapse_duplicates).parameters["screen_degenerate"].default is False
