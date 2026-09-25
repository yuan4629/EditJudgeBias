"""Tests for the D-class ITA (apparent skin lightness) manipulation.

Each test here pins a failure mode that would have produced a publishable-looking wrong
number. Two of them pin defects that were actually present and were caught by measurement
on 2026-07-30 -- they are marked ★.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from edit_judge_bias.fairness import ita as I


def _skin_patch(size=120, base=(198, 150, 128), noise=12.0, seed=42):
    """A textured skin-coloured frame with a masked square inside it."""
    rng = np.random.default_rng(seed)
    tex = rng.normal(0.0, noise, (size, size, 1))
    rgb = np.clip(np.asarray(base, dtype="float64")[None, None, :] + tex, 0, 255).astype("uint8")
    alpha = np.zeros((size, size), dtype="float32")
    alpha[size // 6: size - size // 6, size // 6: size - size // 6] = 1.0
    return rgb, alpha


# --------------------------------------------------------------------------- #
# ITA arithmetic                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lightness, blue_yellow, expected",
    [(50.0, 10.0, 0.0), (70.0, 0.0, 90.0), (30.0, 0.0, -90.0), (60.0, 10.0, 45.0)],
)
def test_ita_matches_hand_computed_values(lightness, blue_yellow, expected):
    assert I.ita_degrees(lightness, blue_yellow) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "value, expected_bin",
    [
        (70.0, "very_light"), (55.0, "very_light"),   # boundaries land in the lighter bin
        (54.9, "light"), (41.0, "light"),
        (40.9, "intermediate"), (28.0, "intermediate"),
        (27.9, "tan"), (10.0, "tan"),
        (9.9, "brown"), (-30.0, "brown"),
        (-30.1, "dark"), (-80.0, "dark"),
    ],
)
def test_fitzpatrick_bin_boundaries_are_deterministic(value, expected_bin):
    assert I.fitzpatrick_bin(value) == expected_bin


def test_weighted_median_ignores_unmasked_pixels():
    """The solved dose must depend on the subject's skin, not on the background."""
    rgb = np.zeros((20, 20, 3), dtype="uint8")
    rgb[:, :] = (20, 20, 20)          # very dark background
    rgb[5:10, 5:10] = (200, 150, 130)  # skin patch
    alpha = np.zeros((20, 20), dtype="float32")
    alpha[5:10, 5:10] = 1.0
    lab = I.rgb_to_lab(rgb)
    l_med, _a, b_med = I.weighted_median_lab(lab, alpha)
    # The masked patch is light; had the background leaked in, L* would collapse toward 8.
    assert l_med > 50.0
    assert I.ita_degrees(l_med, b_med) > 0.0


def test_weighted_median_raises_on_an_empty_mask():
    lab = I.rgb_to_lab(np.zeros((8, 8, 3), dtype="uint8"))
    with pytest.raises(ValueError):
        I.weighted_median_lab(lab, np.zeros((8, 8), dtype="float32"))


# --------------------------------------------------------------------------- #
# ★ The two defects caught by measurement                                      #
# --------------------------------------------------------------------------- #
def test_outside_the_mask_is_byte_identical():
    """★ MEASURED DEFECT. sRGB -> CIELAB -> sRGB is lossy even where the shift is zero.

    Returning the round trip directly left a max abs diff of **2** outside the mask, which
    silently breaks the one invariant the whole D-class gate rests on ("only masked pixels
    can differ"). `composite` writes the shifted pixels back into the ORIGINAL bytes, so
    outside the support the output is the original verbatim.

    No tolerance here on purpose: a single nonzero pixel is a bug, not noise.
    """
    rgb, alpha = _skin_patch()
    lab = I.rgb_to_lab(rgb)
    for delta in (+30.0, -30.0):
        solution = I.solve_matched_shift(lab, alpha, delta_ita=delta)
        out, _ = I.apply_shift(rgb, alpha, solution)
        diff = np.abs(out.astype("int32") - rgb.astype("int32")).max(axis=2)
        assert diff[alpha <= 0].max() == 0, f"delta={delta} leaked outside the mask"
        assert diff[alpha > 0].max() > 0, f"delta={delta} changed nothing inside the mask"


def test_shift_preserves_within_mask_contrast_symmetrically():
    """★ MEASURED DEFECT. A lightness-dependent roll-off compressed texture per arm.

    The earlier per-pixel roll-off `w = (100-L)/50` cost the lighten arm **14%** of its
    within-mask L* standard deviation (4.49 -> 3.84) while the darken arm lost none. An
    arm-dependent texture change is exactly the confound the design exists to remove, so
    the shift is strictly additive and both arms must preserve spread and rank order.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    rgb, alpha = _skin_patch()
    lab = I.rgb_to_lab(rgb)
    inside = alpha > 0
    before = I.rgb_to_lab(rgb)[..., 0][inside]

    spreads = {}
    for name, delta in (("light", +30.0), ("dark", -30.0)):
        solution = I.solve_matched_shift(lab, alpha, delta_ita=delta)
        out, _ = I.apply_shift(rgb, alpha, solution)
        after = I.rgb_to_lab(out)[..., 0][inside]
        rho = scipy_stats.spearmanr(before, after).statistic
        assert rho > 0.999, f"{name} arm reordered L* (Spearman {rho:.6f})"
        spreads[name] = float(after.std())
        assert spreads[name] == pytest.approx(float(before.std()), rel=0.05), (
            f"{name} arm changed within-mask spread"
        )
    # And the two arms must not differ from each other either.
    assert spreads["light"] == pytest.approx(spreads["dark"], rel=0.05)


def test_the_two_arms_carry_a_matched_dose():
    """Positive and negative deltas must move |ITA| by the same amount.

    Pushing both arms to Fitzpatrick extremes instead would bind dose to the arm label: a
    subject at ITA 40 deg sits 15 deg from a light target and 70 deg from a dark one, so
    the dark arm would carry ~4.7x the perturbation and part of any measured "gap" would
    just be a gap in how much the image was disturbed.
    """
    rgb, alpha = _skin_patch()
    lab = I.rgb_to_lab(rgb)
    light = I.solve_matched_shift(lab, alpha, delta_ita=+30.0)
    dark = I.solve_matched_shift(lab, alpha, delta_ita=-30.0)
    assert abs(light.achieved_delta_ita) == pytest.approx(abs(dark.achieved_delta_ita), abs=3.0)
    assert light.achieved_delta_ita > 0 and dark.achieved_delta_ita < 0


# --------------------------------------------------------------------------- #
# The solver's honesty                                                         #
# --------------------------------------------------------------------------- #
def test_achieved_dose_clears_the_gate_floor():
    """The DOSE is the contract; bin crossing is only a description.

    Bin membership cannot be a hard gate: the bins are 13-27 deg wide, so whether a matched
    +-30 deg shift crosses one depends on where the subject started. A subject at ITA 0 sits
    mid-`brown` (-30..10), and -30 lands exactly on the boundary without leaving the bin.
    So the gate must key on |achieved_delta_ita| >= MIN_CROSSING_DELTA and record the bin.
    """
    rgb, alpha = _skin_patch()
    lab = I.rgb_to_lab(rgb)
    for delta in (+30.0, -30.0):
        solution = I.solve_matched_shift(lab, alpha, delta_ita=delta)
        assert abs(solution.achieved_delta_ita) >= I.MIN_CROSSING_DELTA
        assert solution.bin_before in I.FITZPATRICK_ITA_BINS
        assert solution.bin_after in I.FITZPATRICK_ITA_BINS


def test_bin_crossing_is_not_guaranteed_for_a_mid_bin_subject():
    """Pins the reasoning above so nobody later "fixes" the gate into requiring a crossing."""
    # brown spans (-30, 10]; a subject at ITA 0 shifted -30 lands on -30, still brown.
    assert I.fitzpatrick_bin(0.0) == "brown"
    assert I.fitzpatrick_bin(-30.0) == "brown"


def test_reported_dose_does_not_move_with_the_feather_radius():
    """★ MEASURED DEFECT. The feather is cosmetic; it must not move a pass/fail boundary.

    Medians used to be taken over the whole feathered support, so band pixels receiving a
    partial shift dragged them: on a real photo the same requested -30 deg reported an
    achieved -32.2 at `radius_frac=0` but only -27.1 at `radius_frac=0.01`. Since the gate
    compares the achieved delta against MIN_CROSSING_DELTA, tuning the feather would have
    silently changed which rows pass. Measuring over the fully-dosed core cut the spread
    from 3.9 deg to 1.1 deg.
    """
    rgb, hard = _skin_patch()
    hard_bool = hard > 0
    short_edge = min(rgb.shape[:2])

    from edit_judge_bias.fairness import person_region as PR

    achieved = []
    for radius_frac in (0.0, 0.01, 0.03):
        alpha = PR.feather(hard_bool, radius_frac=radius_frac, short_edge=short_edge)
        solution = I.solve_matched_shift(I.rgb_to_lab(rgb), alpha, delta_ita=-30.0)
        _out, meas = I.apply_shift(rgb, alpha, solution)
        achieved.append(meas["delta_ita"])
        # The support grows with the feather but the core stays the measurement footprint.
        assert meas["core_frac"] <= meas["mask_frac"] + 1e-9
    assert max(achieved) - min(achieved) < 3.0, (
        f"achieved dose tracked the feather radius: {achieved}"
    )


def test_gamut_pressure_is_reported_and_the_shift_shrinks():
    """Clipping must be measured and the shift reduced, never silently clamped.

    A silent clamp would flatten skin texture and hand the arm an artefact the gate cannot
    see; a reported shrink lets the gate reject the row via `achieved_delta_ita`.
    """
    rgb = np.full((40, 40, 3), 252, dtype="uint8")   # near-white, no headroom to lighten
    alpha = np.ones((40, 40), dtype="float32")
    solution = I.solve_matched_shift(I.rgb_to_lab(rgb), alpha, delta_ita=+40.0,
                                     max_clipped_frac=0.02)
    assert solution.clipped_frac <= 0.02 or solution.shrunk
    # Whatever it managed, it must report the ACHIEVED delta, not the requested one.
    assert solution.requested_delta_ita == 40.0
    assert abs(solution.achieved_delta_ita) <= abs(solution.requested_delta_ita) + 1e-6


def test_solution_params_round_trip_to_a_manifest():
    rgb, alpha = _skin_patch()
    solution = I.solve_matched_shift(I.rgb_to_lab(rgb), alpha, delta_ita=-30.0)
    params = solution.as_params()
    for key in ("delta_l", "delta_b", "requested_delta_ita", "achieved_delta_ita",
                "ita_before", "ita_after", "bin_before", "bin_after", "clipped_frac"):
        assert key in params
    import json
    json.dumps(params)  # must be JSON-safe for the manifest


# --------------------------------------------------------------------------- #
# The same dose must reach both members of a pair                              #
# --------------------------------------------------------------------------- #
def test_one_solution_applied_to_two_images_uses_the_same_dose():
    """★ The solution is solved on the ORIGINAL and reused on the edited image verbatim.

    Re-solving per image would let the two members of one pair land on different achieved
    ITAs -- the pair would then differ in dose as well as in content, and the judge's score
    difference could not be attributed to the attribute.
    """
    rgb, alpha = _skin_patch()
    # An "edited" version: same subject, globally slightly brighter (a real editor's tone shift).
    edited = np.clip(rgb.astype("int16") + 8, 0, 255).astype("uint8")

    solution = I.solve_matched_shift(I.rgb_to_lab(rgb), alpha, delta_ita=-30.0)
    out_o, meas_o = I.apply_shift(rgb, alpha, solution)
    out_e, meas_e = I.apply_shift(edited, alpha, solution)

    # Identical Lab offsets applied to both members.
    assert meas_o["delta_ita"] == pytest.approx(meas_e["delta_ita"], abs=2.0)
    # And each member keeps byte-identity outside the mask against ITS OWN base.
    for base, out in ((rgb, out_o), (edited, out_e)):
        diff = np.abs(out.astype("int32") - base.astype("int32")).max(axis=2)
        assert diff[alpha <= 0].max() == 0


def test_composite_rejects_shape_mismatch():
    rgb, alpha = _skin_patch(size=40)
    with pytest.raises(ValueError):
        I.composite(rgb, np.zeros((30, 30, 3), dtype="uint8"), alpha)
    with pytest.raises(ValueError):
        I.composite(rgb, rgb.copy(), np.zeros((7, 7), dtype="float32"))


def test_composite_with_an_empty_alpha_returns_the_base_unchanged():
    rgb, _ = _skin_patch(size=32)
    other = np.zeros_like(rgb)
    out = I.composite(rgb, other, np.zeros((32, 32), dtype="float32"))
    assert (out == rgb).all()


def test_feathered_alpha_blends_inside_the_support_only():
    """A partially-transparent band must be treated as INSIDE the region.

    Byte-identity is asserted against `alpha > 0`, not against the hard mask, precisely so
    the feather band is never counted as "unchanged" while actually being modified.
    """
    rgb, alpha = _skin_patch()
    banded = alpha.copy()
    band = (slice(18, 20), slice(20, 100))
    banded[band] = 0.5
    solution = I.solve_matched_shift(I.rgb_to_lab(rgb), alpha, delta_ita=-30.0)
    out, _ = I.apply_shift(rgb, banded, solution)
    diff = np.abs(out.astype("int32") - rgb.astype("int32")).max(axis=2)
    assert diff[banded <= 0].max() == 0
    assert diff[banded == 0.5].max() > 0


def test_measure_reports_nothing_for_an_empty_mask_rather_than_guessing():
    rgb, _ = _skin_patch(size=32)
    stats = I.measure(rgb, np.zeros((32, 32), dtype="float32"))
    assert stats["ita_median"] is None
    assert stats["bin"] == "unknown"
    assert stats["mask_frac"] == 0.0
