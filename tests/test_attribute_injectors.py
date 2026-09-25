"""Tests for the D-class attribute injectors (the four arms).

Every test pins a failure mode that would have produced a publishable-looking wrong number.
Synthetic only -- no model weights, no data tree, no network.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from edit_judge_bias.fairness import attribute_injectors as AI
from edit_judge_bias.fairness import ita as I
from edit_judge_bias.fairness import person_region as PR


def _scene(size=120, seed=7):
    """A synthetic subject: skin-coloured face region plus a differently-coloured 'shirt'."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((size, size, 3), dtype="uint8")
    rgb[:, :] = (60, 90, 70)                                    # background
    person = np.zeros((size, size), dtype=bool)
    person[20:100, 30:90] = True                                 # the subject
    skin = np.zeros((size, size), dtype=bool)
    skin[25:55, 40:80] = True                                    # face
    tex = rng.normal(0.0, 9.0, (size, size, 1))
    face = np.clip(np.array([198, 150, 128])[None, None, :] + tex, 0, 255).astype("uint8")
    shirt = np.clip(np.array([70, 90, 190])[None, None, :] + tex, 0, 255).astype("uint8")
    rgb[person] = shirt[person]
    rgb[skin] = face[skin]
    return rgb, person, skin


def _region_set(rgb, person, skin) -> PR.RegionSet:
    h, w = rgb.shape[:2]
    head = np.zeros((h, w), dtype=bool)
    head[25:60, 38:82] = True
    return PR.RegionSet(
        image_path=None, width=w, height=h,
        person=person, person_method="synthetic", person_frac=float(person.mean()),
        face_box=(40, 25, 80, 55), face_frac=(40 * 30) / float(w * h),
        face_source="synthetic",
        head=head, head_frac=float(head.mean()),
        skin=skin, skin_frac=float(skin.mean()),
    )


def _prepared(delta_ita=30.0):
    rgb, person, skin = _scene()
    regions = _region_set(rgb, person, skin)
    injector = AI.get_attribute_injector("skin_tone", delta_ita=delta_ita)
    scene = injector.prepare_scene(rgb, base_sample_id="synthetic", regions=regions)
    return injector, scene, rgb, regions


# --------------------------------------------------------------------------- #
# Registry isolation                                                           #
# --------------------------------------------------------------------------- #
def test_attribute_injectors_are_not_in_the_abc_bias_registry():
    """★ A demographic injector in `bias/registry.py` is one typo from the main grid.

    The A/B/C experiment configs name registry keys by string and
    `tests/test_bias_registry.py` enumerates `available()`, so a D-class arm registered there
    could silently enter the invariance benchmark it does not belong to.
    """
    from edit_judge_bias.bias import registry as abc_registry

    abc_names = set(abc_registry.available())
    for name in AI.available_attribute_injectors():
        assert name not in abc_names, f"{name!r} leaked into the A/B/C registry"
    assert "skin_tone" in AI.available_attribute_injectors()


def test_unknown_attribute_raises_rather_than_dropping_an_arm():
    with pytest.raises(KeyError, match="unknown attribute injector"):
        AI.get_attribute_injector("eye_colour")


# --------------------------------------------------------------------------- #
# The arm set                                                                  #
# --------------------------------------------------------------------------- #
def test_both_controls_are_emitted_by_default():
    """`sham_nonskin` is not optional: without it there is no fairness claim to make.

    This study has measured that judges penalise local readable pixel changes on 5/5 judges,
    so a skin recolor could move scores because it IS a recolor. The same-person non-skin
    arm is what separates artefact sensitivity from a tone response.
    """
    arms = AI.get_attribute_injector("skin_tone").arms()
    roles = {a.role for a in arms}
    assert AI.ROLE_SHAM in roles
    assert AI.ROLE_SHAM_NONSKIN in roles
    labels = [a.label for a in arms]
    assert "dark" in labels and "light" in labels


def test_study_arms_are_signed_symmetrically_around_zero():
    arms = {a.label: a for a in AI.get_attribute_injector("skin_tone", delta_ita=30.0).arms()}
    assert arms["dark"].delta_ita == -30.0
    assert arms["light"].delta_ita == +30.0
    assert arms["sham"].delta_ita == 0.0


def test_alphabetical_label_order_fixes_the_reported_sign():
    """`compute_attribute_gaps` sorts labels, so the gap is dark - light. Pin it.

    Renaming an arm would silently flip every sign in the frozen table.
    """
    labels = sorted({a.label for a in AI.get_attribute_injector("skin_tone").arms()
                     if a.role == AI.ROLE_STUDY})
    assert labels == ["dark", "light"]


# --------------------------------------------------------------------------- #
# ★ One mask, one dose, shared                                                 #
# --------------------------------------------------------------------------- #
def test_both_study_arms_share_one_mask_digest():
    """Shared mask makes segmentation error COMMON-MODE.

    If the mask clips an ear it clips it identically in both arms, so it cannot produce a
    difference between them. `mask_sha256` makes that auditable instead of asserted.
    """
    injector, scene, rgb, _ = _prepared()
    results = {
        spec.label: injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
        for spec in injector.arms() if spec.role == AI.ROLE_STUDY
    }
    digests = {r.mask_sha256 for r in results.values()}
    assert len(digests) == 1 and digests != {""}


def test_the_same_dose_reaches_both_members_of_a_pair():
    """The dose is solved once on the ORIGINAL and reused verbatim on the edited image.

    Re-solving per member would let the two members land on different achieved shifts, so
    the pair would differ in dose as well as in content.
    """
    injector, scene, rgb, _ = _prepared()
    edited = np.clip(rgb.astype("int16") + 7, 0, 255).astype("uint8")
    dark = [s for s in injector.arms() if s.label == "dark"][0]

    on_original = injector.apply(rgb, scene, dark, applied_to=AI.APPLIED_ORIGINAL)
    on_edited = injector.apply(edited, scene, dark, applied_to=AI.APPLIED_EDITED)
    assert on_original.success and on_edited.success
    assert on_original.params["delta_l"] == on_edited.params["delta_l"]
    assert on_original.params["delta_b"] == on_edited.params["delta_b"]
    assert on_original.params["applied_to"] == "original"
    assert on_edited.params["applied_to"] == "edited"


def test_study_arms_carry_a_matched_perceptual_dose():
    injector, scene, rgb, _ = _prepared()
    doses = {}
    for spec in injector.arms():
        if spec.role != AI.ROLE_STUDY:
            continue
        result = injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
        doses[spec.label] = result.measured["delta_e00_mean"]
    assert doses["dark"] == pytest.approx(doses["light"], rel=0.25)


# --------------------------------------------------------------------------- #
# ★ Byte identity outside the mask                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", ["dark", "light", "sham", "sham_nonskin"])
def test_every_arm_is_byte_identical_outside_its_mask(label):
    """No tolerance. A single stray pixel means a lossy resave or an alpha leak, not noise."""
    injector, scene, rgb, _ = _prepared()
    spec = [s for s in injector.arms() if s.label == label][0]
    result = injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
    assert result.success, result.message
    assert result.measured["outside_mask_identical"] is True
    assert result.measured["max_outside_diff"] == 0


def test_study_arms_actually_change_something_inside_the_mask():
    """A byte-identical study arm would be a silent no-op passing every check."""
    injector, scene, rgb, _ = _prepared()
    for spec in injector.arms():
        if spec.role != AI.ROLE_STUDY:
            continue
        result = injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
        assert result.measured["max_inside_diff"] > 4, f"{spec.label} barely changed anything"
        assert abs(result.measured["delta_ita"]) >= I.MIN_CROSSING_DELTA * 0.6


# --------------------------------------------------------------------------- #
# ★ The sham arm must be a distinct request                                    #
# --------------------------------------------------------------------------- #
def test_sham_is_zero_dose_but_never_byte_identical():
    """★ A byte-identical sham reports a zero it never measured.

    `judges/openai_judge.py` caches responses on a hash of the payload INCLUDING the base64
    image, so an identical sham image is served the baseline's answer and the placebo floor
    comes back as exactly 0.0 -- fabricated by plumbing. `bias/sham.py:73-78` documents the
    same trap for the A/B/C placebo.
    """
    injector, scene, rgb, _ = _prepared()
    spec = [s for s in injector.arms() if s.role == AI.ROLE_SHAM][0]
    result = injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
    assert result.success
    assert result.image.tobytes() != rgb.tobytes(), "sham must be a distinct request"
    assert result.measured["delta_ita"] == pytest.approx(0.0, abs=1.0)


def test_sham_dose_is_far_smaller_than_a_study_arm():
    """Zero-DOSE, not zero-CHANGE: the code path's own numerical footprint is the placebo.

    Measured on a real photo: sham mean dE00 0.375 against ~8.9 for a study arm, i.e. ~24x
    smaller. The nonzero footprint is the measurement, not a bug to fix.
    """
    injector, scene, rgb, _ = _prepared()
    by_label = {}
    for spec in injector.arms():
        result = injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
        if result.success:
            by_label[spec.label] = result.measured.get("delta_e00_mean", float("nan"))
    assert by_label["sham"] < by_label["dark"] / 5.0


def test_force_distinct_flips_one_lsb_inside_the_mask_only():
    """The forced pixel must land inside the mask, or it would break byte-identity."""
    base = np.full((20, 20, 3), 128, dtype="uint8")
    alpha = np.zeros((20, 20), dtype="float32")
    alpha[5:10, 5:10] = 1.0
    out, forced = AI._force_distinct(base, base.copy(), alpha)
    assert forced is True
    diff = np.abs(out.astype("int32") - base.astype("int32")).max(axis=2)
    assert diff[alpha <= 0].max() == 0
    assert diff[alpha > 0].max() == 1


def test_force_distinct_is_a_no_op_when_already_different():
    base = np.full((10, 10, 3), 100, dtype="uint8")
    changed = base.copy()
    changed[0, 0] = (200, 100, 100)
    alpha = np.ones((10, 10), dtype="float32")
    out, forced = AI._force_distinct(base, changed, alpha)
    assert forced is False
    assert (out == changed).all()


# --------------------------------------------------------------------------- #
# ★ The artefact control                                                       #
# --------------------------------------------------------------------------- #
def test_nonskin_control_stays_on_the_person_and_off_the_skin():
    injector, scene, rgb, _ = _prepared()
    assert scene.nonskin is not None, "the synthetic subject has clothing; the arm must exist"
    assert not (scene.nonskin & scene.skin).any()
    assert scene.nonskin_stats["overlaps_skin"] == 0
    assert 0.8 <= scene.nonskin_stats["area_ratio"] <= 1.25


def test_nonskin_arm_is_dropped_and_reported_when_no_such_region_exists():
    """Losing one arm must not lose the scene, and it must be visible in the message."""
    rgb, person, _skin = _scene()
    all_skin = person.copy()                       # the subject is entirely skin
    regions = _region_set(rgb, person, all_skin)
    injector = AI.get_attribute_injector("skin_tone")
    scene = injector.prepare_scene(rgb, base_sample_id="s", regions=regions)
    assert scene.nonskin is None

    spec = [s for s in injector.arms() if s.role == AI.ROLE_SHAM_NONSKIN][0]
    result = injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
    assert result.success is False
    assert "nonskin" in result.message

    # ...and the study arms still work.
    dark = [s for s in injector.arms() if s.label == "dark"][0]
    assert injector.apply(rgb, scene, dark, applied_to=AI.APPLIED_ORIGINAL).success


def test_dose_comparability_columns_are_present_on_every_row():
    """The control is only usable if its dose can be CHECKED, not assumed.

    Measured on a real photo the control ran 20% lower in mean dE00 than the study arm while
    its total dE00 mass ran 8% higher, so both readings ride on the row.
    """
    injector, scene, rgb, _ = _prepared()
    for spec in injector.arms():
        result = injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
        if not result.success:
            continue
        for key in ("delta_e00_mean", "delta_e00_median", "delta_e00_mass", "core_px"):
            assert key in result.measured, f"{spec.label} is missing {key}"


# --------------------------------------------------------------------------- #
# Failure handling                                                             #
# --------------------------------------------------------------------------- #
def test_misaligned_member_is_reported_not_silently_resized():
    """A resize here would change the edit as well as the attribute.

    The runner must align the pair explicitly and record that it did; the injector refusing
    is what forces that to be a visible decision.
    """
    injector, scene, rgb, _ = _prepared()
    smaller = np.zeros((60, 60, 3), dtype="uint8")
    dark = [s for s in injector.arms() if s.label == "dark"][0]
    result = injector.apply(smaller, scene, dark, applied_to=AI.APPLIED_EDITED)
    assert result.success is False
    assert "does not match" in result.message


def test_params_are_json_safe_for_the_manifest():
    import json

    injector, scene, rgb, _ = _prepared()
    for spec in injector.arms():
        result = injector.apply(rgb, scene, spec, applied_to=AI.APPLIED_ORIGINAL)
        if result.success:
            json.dumps(result.params)
            json.dumps({k: v for k, v in result.measured.items()})
