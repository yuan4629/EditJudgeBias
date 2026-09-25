"""Tests for the detector-first D-class pool builder.

The screen that selects this pool decides the study's n, so every test here pins a way the
selection could go quietly wrong. The most important one is
`test_minor_word_screen_still_bites_without_the_human_category`: relaxing the
`content_category` screen must not relax the ethics hard stop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import ContentCategory, EditType, SampleRecord
from edit_judge_bias.experiments import build_fairness_pool as BFP
from edit_judge_bias.experiments.run_attribute_edit import _eligible, instruction_ok


def _sample(sample_id: str, *, instruction: str, category: ContentCategory,
            original: str, edited: str, source: str = "EBench-18K",
            edit_type: EditType = EditType.ADD) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id, source_dataset=source, edit_type=edit_type,
        content_category=category, original_image_path=original,
        instruction=instruction, edit_model="m0", edited_image_path=edited,
    )


def _write_image(path: Path, seed: int = 0, size=(256, 256)) -> Path:
    """A TEXTURED image.

    Deliberately not flat grey: ORB finds no keypoints in a featureless frame, so
    `validate_orb_instrument` would (correctly) refuse to certify a dedupe it could not
    perform, and the test would fail for a reason unrelated to what it is testing.
    """
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (120, 120, 120))
    draw = ImageDraw.Draw(img)
    for i in range(0, size[0], 7):
        shade = (i * 13 + seed * 37) % 200 + 30
        draw.rectangle([i, (i * 3 + seed) % size[1], i + 5, (i * 3 + seed) % size[1] + 9],
                      fill=(shade, (shade * 2) % 255, (shade * 3) % 255))
    img.save(path)
    return path


# --------------------------------------------------------------------------- #
# ★ The ethics hard stop survives the relaxation                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("instruction", [
    "Add a hat to the girl",
    "Let the boy hold a ball",
    "make the child smile",
    "put a scarf on the toddler",
    "give the baby a rattle",
    "the teenager should wear a cap",
])
def test_minor_word_screen_still_bites_without_the_human_category(instruction):
    """★ THE MOST IMPORTANT TEST IN THIS FILE.

    The detector-first pool turns OFF `content_category == "human"` to stop losing real
    candidates to a bad label. `_MINOR_WORDS` must be entirely unaffected by that: it is an
    ethics hard stop, not a quality filter, and it must never be relaxed to grow the pool.
    """
    assert instruction_ok(instruction) is False


@pytest.mark.parametrize("instruction", [
    "make the woman smile", "change his hat to red", "give her longer hair",
    "lighten the skin tone", "the man should hold a cup",
])
def test_attribute_word_screen_still_bites_without_the_human_category(instruction):
    """Never flip an attribute the instruction itself refers to."""
    assert instruction_ok(instruction) is False


@pytest.mark.parametrize("instruction", [
    "Add a Christmas hat to the bird",
    "Remove the rose from the hand",
    "Change the background to a park",
    "Replace the soccer ball with a basketball",
])
def test_clean_instructions_pass_both_screens(instruction):
    assert instruction_ok(instruction) is True


def test_eligible_is_unchanged_by_the_refactor():
    """`_eligible` is imported by tests and built the published 41-scene pool.

    It must remain exactly "human category AND clean instruction".
    """
    clean = _sample("a", instruction="Add a hat", category=ContentCategory.HUMAN,
                    original="o.png", edited="e.png")
    assert _eligible(clean) is True

    not_human = _sample("b", instruction="Add a hat", category=ContentCategory.ANIMAL,
                        original="o.png", edited="e.png")
    assert _eligible(not_human) is False
    # ...but the instruction screen alone accepts it, which is the whole point of the split.
    assert instruction_ok(not_human.instruction) is True

    dirty = _sample("c", instruction="Add a hat to the girl",
                    category=ContentCategory.HUMAN, original="o.png", edited="e.png")
    assert _eligible(dirty) is False


def test_relaxing_the_category_admits_mislabelled_people_only_via_the_detector(tmp_path: Path):
    """Turning off the label must widen the CANDIDATE set, not the accepted set.

    Acceptance still requires the measured person/face screen, so a mislabelled landscape
    gets a chance to be measured and then fails on its own merits.
    """
    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    samples = [
        _sample("labelled_human", instruction="Add a hat", category=ContentCategory.HUMAN,
                original="o.png", edited="e.png"),
        _sample("labelled_scenery", instruction="Add a bench",
                category=ContentCategory.SCENERY, original="o.png", edited="e.png"),
    ]
    strict = BFP.eligible_originals(samples, tmp_path, require_human_category=True)
    relaxed = BFP.eligible_originals(samples, tmp_path, require_human_category=False)
    # Both point at the same file, so path-dedupe keeps one either way -- what matters is that
    # the relaxed screen does not *reject* on the label.
    assert len(strict) == 1 and len(relaxed) == 1
    assert BFP.eligible_originals(samples[1:], tmp_path, require_human_category=True) == []
    assert len(BFP.eligible_originals(samples[1:], tmp_path,
                                     require_human_category=False)) == 1


# --------------------------------------------------------------------------- #
# ★ The ORB instrument must prove itself                                      #
# --------------------------------------------------------------------------- #
def test_orb_self_match_control_raises_when_the_matcher_is_degenerate(monkeypatch):
    """★ Turns a real incident into a test.

    A first attempt at the 70 -> 41 collapse reported "0 duplicates" -- clean, plausible and
    completely wrong -- because it called `ransac_inliers(path, path)` when the function takes
    signatures, and the TypeError was swallowed by `except Exception: continue`. A self-match
    saturates at 440..500 inliers, so a low self-match means every verdict is void.
    """
    monkeypatch.setattr(BFP, "ransac_inliers", lambda a, b: 0)
    with pytest.raises(RuntimeError, match="self-match"):
        BFP.validate_orb_instrument([(list(range(50)), None)])


def test_orb_validation_raises_when_nothing_is_readable():
    with pytest.raises(RuntimeError, match="no usable signatures"):
        BFP.validate_orb_instrument([None, None])


def test_orb_self_match_floor_is_relative_to_available_keypoints(monkeypatch):
    """A keypoint-poor image must not fail a matcher that is working correctly.

    An absolute inlier floor silently assumes image richness: a 37-keypoint frame can never
    reach 400 however well the matcher works, so the floor is a fraction of what is there.
    """
    monkeypatch.setattr(BFP, "ransac_inliers", lambda a, b: 35)
    report = BFP.validate_orb_instrument([(list(range(37)), None)])
    assert report["self_match_frac"] >= 0.8
    # The same 35 inliers against a rich signature must FAIL: 35 of 500 is not a self-match.
    with pytest.raises(RuntimeError, match="self-match"):
        BFP.validate_orb_instrument([(list(range(500)), None)])


def test_orb_validation_counts_unreadable_signatures(monkeypatch):
    """An unreadable signature is a measurement failure and must be COUNTED, not silent.

    `collapse_duplicates` skips such pairs, so without this count a pool could be full of
    unverified singletons and look fully deduped.
    """
    monkeypatch.setattr(BFP, "ransac_inliers", lambda a, b: 500)
    report = BFP.validate_orb_instrument(
        [(list(range(50)), None), None, (list(range(40)), None), None]
    )
    assert report["usable_signatures"] == 2
    assert report["unreadable_signatures"] == 2


# --------------------------------------------------------------------------- #
# Alignment screen                                                             #
# --------------------------------------------------------------------------- #
def test_face_box_iou_arithmetic():
    assert BFP._face_box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert BFP._face_box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Half-overlap along one axis: intersection 50, union 150.
    assert BFP._face_box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_alignment_cap_is_reported_not_silent(tmp_path: Path, monkeypatch):
    """A capped scan must never read as an exhaustive one ("no silent caps").

    EBench ships 17 editors per original; the arm needs one aligned editor, so the scan is
    capped -- and `n_align_checked` rides next to `n_aligned_editors` so the report says so.
    """
    from edit_judge_bias.fairness import person_region as pr

    _write_image(tmp_path / "o.png")
    siblings = []
    for i in range(10):
        _write_image(tmp_path / f"e{i}.png")
        siblings.append(_sample(f"s{i}", instruction="Add a hat",
                                category=ContentCategory.HUMAN,
                                original="o.png", edited=f"e{i}.png"))

    fake = pr.RegionSet(
        image_path=None, width=256, height=256,
        person=None, person_method="stub", person_frac=0.5,
        face_box=(40, 40, 120, 120), face_frac=0.1, face_source="stub",
        head=None, head_frac=0.1, skin=None, skin_frac=0.05,
    )
    monkeypatch.setattr(pr, "region_set", lambda *a, **k: fake)

    screens = BFP.screen_originals(
        siblings[:1], {"o.png": siblings}, tmp_path,
        min_face_frac=0.02, min_person_frac=0.05, align_iou=0.5, align_max_checks=4,
    )
    assert len(screens) == 1
    screen = screens[0]
    assert screen.passed_detector is True
    assert screen.n_align_checked == 4, "the cap must be recorded"
    assert screen.n_aligned_editors == 4
    assert len(screen.aligned_sample_ids) == 4


def test_detector_rejects_a_scene_with_no_face_on_the_person(tmp_path: Path, monkeypatch):
    from edit_judge_bias.fairness import person_region as pr

    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    sample = _sample("s", instruction="Add a hat", category=ContentCategory.HUMAN,
                     original="o.png", edited="e.png")
    faceless = pr.RegionSet(
        image_path=None, width=256, height=256, person=None, person_method="stub",
        person_frac=0.4, face_box=None, face_frac=0.0, face_source="none",
        head=None, head_frac=0.0, skin=None, skin_frac=0.0,
    )
    monkeypatch.setattr(pr, "region_set", lambda *a, **k: faceless)
    screens = BFP.screen_originals([sample], {"o.png": [sample]}, tmp_path)
    assert screens[0].passed_detector is False
    assert screens[0].reason == "no_face_box_on_person"


def test_region_error_is_recorded_not_raised(tmp_path: Path, monkeypatch):
    """One unreadable image must not abort the pool build, and the reason must be visible."""
    from edit_judge_bias.fairness import person_region as pr

    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    sample = _sample("s", instruction="Add a hat", category=ContentCategory.HUMAN,
                     original="o.png", edited="e.png")

    def boom(*a, **k):
        raise pr.RegionError("stub failure")

    monkeypatch.setattr(pr, "region_set", boom)
    screens = BFP.screen_originals([sample], {"o.png": [sample]}, tmp_path)
    assert screens[0].passed_detector is False
    assert "region_error" in screens[0].reason


# --------------------------------------------------------------------------- #
# Config handling                                                              #
# --------------------------------------------------------------------------- #
def test_unknown_config_key_raises(tmp_path: Path):
    """A silently-ignored `min_face_frac` would build a pool the config does not describe."""
    path = tmp_path / "pool.yaml"
    path.write_text("min_face_frac: 0.04\nmin_fase_frac: 0.02\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        BFP.load_config(path)


def test_config_round_trips_known_keys(tmp_path: Path):
    path = tmp_path / "pool.yaml"
    path.write_text(
        "detector: true\nmin_face_frac: 0.04\nexclude_edit_types: [color]\n", encoding="utf-8"
    )
    cfg = BFP.load_config(path)
    assert cfg["detector"] is True
    assert cfg["min_face_frac"] == 0.04
    assert cfg["exclude_edit_types"] == ["color"]


def test_default_build_is_the_published_screen():
    """The old behaviour must stay the default so the 41-scene pool reproduces."""
    import inspect

    sig = inspect.signature(BFP.build)
    assert sig.parameters["detector"].default is False
    assert sig.parameters["require_human_category"].default is True


# --------------------------------------------------------------------------- #
# Multi-manifest input                                                         #
# --------------------------------------------------------------------------- #
def test_overlapping_manifests_are_deduped_by_sample_id(tmp_path: Path):
    """`samples_full_v2` already contains an EBench prefix; double-counting would inflate n."""
    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    shared = _sample("dup", instruction="Add a hat", category=ContentCategory.HUMAN,
                     original="o.png", edited="e.png")
    only_b = _sample("solo", instruction="Add a bench", category=ContentCategory.HUMAN,
                     original="o.png", edited="e.png")
    a = tmp_path / "data/manifests/a.jsonl"
    b = tmp_path / "data/manifests/b.jsonl"
    io.write_jsonl(a, [shared])
    io.write_jsonl(b, [shared, only_b])

    report = BFP.build(
        root=tmp_path,
        samples_paths=["data/manifests/a.jsonl", "data/manifests/b.jsonl"],
        write=False,
    )
    assert report["total_samples"] == 2, "the shared sample_id must be counted once"


def test_exclude_edit_types_filters_before_counting(tmp_path: Path):
    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    rows = [
        _sample("keep", instruction="Add a hat", category=ContentCategory.HUMAN,
                original="o.png", edited="e.png", edit_type=EditType.ADD),
        _sample("drop", instruction="Repaint it", category=ContentCategory.HUMAN,
                original="o.png", edited="e.png", edit_type=EditType.COLOR),
    ]
    io.write_jsonl(tmp_path / "data/manifests/s.jsonl", rows)
    report = BFP.build(root=tmp_path, samples_path="data/manifests/s.jsonl",
                       exclude_edit_types=["color"], write=False)
    assert report["total_samples"] == 1
    assert report["excluded_edit_types"] == ["color"]


# --------------------------------------------------------------------------- #
# ★ Facial-feature screen (added 2026-07-31 from a measured gap)               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("instruction", [
    "Add eyes",
    "Add glasses",
    "Remove the sunglasses",
    "Make the lips fuller",
    "Change the nose shape",
    "Add freckles to the cheeks",
    "Whiten the teeth",
    "Add a tattoo",
    "Remove the makeup",
    "Put a mask on",
])
def test_facial_feature_instructions_are_screened_out(instruction):
    """★ MEASURED GAP. The OmniEdit probe surfaced `Add eyes` and `Add glasses` as USABLE.

    Both operate on the face, which is exactly where a skin-lightness manipulation acts, yet
    neither tripped the original screen: they name a facial *feature*, not a demographic
    attribute, and `face` alone did not cover them. A judge asked "were the eyes added
    correctly?" while we have recoloured the face is being asked two questions at once, and
    glasses additionally occlude the region the manipulation depends on.
    """
    assert instruction_ok(instruction) is False


@pytest.mark.parametrize("instruction", [
    "Add necklace",
    "Replace the denim jacket with white sweater",
    "Remove khaki pants",
    "turn the color of book to gray",
    "Replace the scarf with necklace",
])
def test_clothing_and_object_instructions_still_pass(instruction):
    """The screen must stay narrow enough to keep the data it should keep.

    Clothing and prop edits are exactly the "identity-irrelevant instruction" the ethics doc
    asks for: the edit acts on the garment, the manipulation acts on skin, disjoint regions.
    Over-screening these would delete the usable pool.
    """
    assert instruction_ok(instruction) is True


# --------------------------------------------------------------------------- #
# ★ The framing band, both bounds (v4 / OmniEdit)                              #
# --------------------------------------------------------------------------- #
def _stub_regions(*, person_frac=0.5, face_frac=0.08, skin_frac=0.09,
                  face_box=(40, 40, 120, 120)):
    from edit_judge_bias.fairness import person_region as pr

    return pr.RegionSet(
        image_path=None, width=256, height=256,
        person=None, person_method="stub", person_frac=person_frac,
        face_box=face_box, face_frac=face_frac, face_source="stub",
        head=None, head_frac=face_frac, skin=None, skin_frac=skin_frac,
    )


def _screen_one(tmp_path: Path, monkeypatch, regions, **kwargs):
    from edit_judge_bias.fairness import person_region as pr

    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    sample = _sample("s", instruction="Add a hat", category=ContentCategory.HUMAN,
                     original="o.png", edited="e.png")
    monkeypatch.setattr(pr, "region_set", lambda *a, **k: regions)
    return BFP.screen_originals([sample], {"o.png": [sample]}, tmp_path,
                                check_alignment=False, **kwargs)[0]


@pytest.mark.parametrize("face_frac,expect_pass", [
    (0.35, False),   # tight portrait crop -- no scene left for the instruction to act on
    (0.13, False),   # just above the band
    (0.12, True),    # head-and-shoulders, the upper edge
    (0.08, True),
    (0.04, True),    # half-body, the lower edge
    (0.03, False),   # below the band -- the manipulation is not perceptible
])
def test_framing_band_rejects_both_ends(tmp_path, monkeypatch, face_frac, expect_pass):
    """An upper bound is as load-bearing as the lower one.

    A lower bound alone admits FairFace/CelebA-style portrait crops, where `face_frac ~0.35`
    leaves no scene for the editing instruction to act on -- the sample fails construct
    validity from the opposite direction to a distant crowd shot. Both bounds together are
    what "single subject, head-and-shoulders to half-body" means.
    """
    screen = _screen_one(
        tmp_path, monkeypatch, _stub_regions(face_frac=face_frac),
        require_face=True, require_skin=True,
        min_face_frac=0.04, max_face_frac=0.12, min_skin_frac=0.04, max_skin_frac=1.0,
    )
    assert screen.passed_detector is expect_pass


def test_max_face_frac_defaults_to_no_upper_bound():
    """The published v3 pools must reproduce, so the band is opt-in."""
    assert BFP.DEFAULT_MAX_FACE_FRAC is None


# --------------------------------------------------------------------------- #
# ★ require_skin is decoupled from require_face                                #
# --------------------------------------------------------------------------- #
def test_require_skin_defaults_to_the_historical_coupling(tmp_path, monkeypatch):
    """`require_skin=None` reproduces v3: a face-gated arm ignores the skin floor.

    D-G composites a head and genuinely does not care about skin area, so this coupling was
    intentional. Pinning it means the v3 D-G pool (19 passed the detector, 2 of them below the
    skin floor) still rebuilds unchanged now that the parameter has been split out.
    """
    screen = _screen_one(
        tmp_path, monkeypatch, _stub_regions(skin_frac=0.001),
        require_face=True, min_face_frac=0.02, min_skin_frac=0.04,
    )
    assert screen.passed_detector is True, "a face-gated arm must not apply the skin floor"


def test_require_skin_true_alongside_require_face_applies_both(tmp_path, monkeypatch):
    """The combination v3's code could not express, and the whole reason for the split.

    A deterministic skin recolor run over a PORTRAIT corpus needs both: the face box to place
    the subject in the framing band, the skin fraction to have something to recolour.
    """
    screen = _screen_one(
        tmp_path, monkeypatch, _stub_regions(skin_frac=0.001),
        require_face=True, require_skin=True,
        min_face_frac=0.02, min_skin_frac=0.04, max_skin_frac=1.0,
    )
    assert screen.passed_detector is False
    assert screen.reason.startswith("skin_frac")


# --------------------------------------------------------------------------- #
# ★ Policy B: operator collision screened on the SOURCE task label             #
# --------------------------------------------------------------------------- #
def test_exclude_tasks_screens_raw_task_and_reports_both_sides(tmp_path: Path):
    """`env` arrives mapped to `background`, so an edit_type rule cannot express this.

    The exclusion is by OPERATOR CLASS on the dataset's own label. Screening our derived
    `edit_type` instead would either miss `env` entirely or take the whole `background`
    family with it -- and the measured collision is real (`env` median skin overlap 0.981)
    while `attribute_modification`, which the plan wanted excluded, is 0.172.
    """
    samples = []
    for i, task in enumerate(["addition", "env", "style", "swap"]):
        _write_image(tmp_path / f"o{i}.png", seed=i)
        _write_image(tmp_path / f"e{i}.png", seed=i + 50)
        rec = _sample(f"s{i}", instruction="Add a hat", category=ContentCategory.HUMAN,
                      original=f"o{i}.png", edited=f"e{i}.png")
        rec.metadata.raw_task = task
        samples.append(rec)
    io.write_jsonl(tmp_path / "m.jsonl", samples)

    rep = BFP.build(root=tmp_path, samples_path="m.jsonl", write=False,
                    require_human_category=False, exclude_tasks=["style", "env"])
    assert rep["dropped_by_task_exclusion"] == {"env": 1, "style": 1}
    assert rep["after_task_exclusion"] == 2
    assert rep["exclude_tasks"] == ["env", "style"]


def test_exclude_tasks_is_off_by_default(tmp_path: Path):
    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    rec = _sample("s", instruction="Add a hat", category=ContentCategory.HUMAN,
                  original="o.png", edited="e.png")
    rec.metadata.raw_task = "env"
    io.write_jsonl(tmp_path / "m.jsonl", [rec])
    rep = BFP.build(root=tmp_path, samples_path="m.jsonl", write=False,
                    require_human_category=False)
    assert "dropped_by_task_exclusion" not in rep
    assert rep["content_distinct_scenes"] == 1


# --------------------------------------------------------------------------- #
# ★ R-A: seeded shuffle, take prefix -- raising k must yield a STRICT SUPERSET #
# --------------------------------------------------------------------------- #
def _pool_of(tmp_path: Path, n: int) -> Path:
    samples = []
    for i in range(n):
        _write_image(tmp_path / f"o{i}.png", seed=i * 3 + 1)
        _write_image(tmp_path / f"e{i}.png", seed=i * 3 + 2)
        samples.append(_sample(f"s{i:03d}", instruction="Add a hat",
                               category=ContentCategory.HUMAN,
                               original=f"o{i}.png", edited=f"e{i}.png"))
    io.write_jsonl(tmp_path / "m.jsonl", samples)
    return tmp_path / "m.jsonl"


def _ids_at_k(tmp_path: Path, k):
    rep = BFP.build(root=tmp_path, samples_path="m.jsonl",
                    out_path="out.jsonl", write=True,
                    require_human_category=False, k=k)
    ids = [r.sample_id for r in io.read_jsonl(tmp_path / "out.jsonl", SampleRecord)]
    return rep, ids


def test_raising_k_yields_a_strict_superset(tmp_path: Path):
    """RED LINE R-A. This is what keeps already-paid work valid when the pool grows.

    Seeded-shuffle-then-take-prefix means the first k scenes never change as k rises, so
    injected images and judge calls bought at k=4 are still exactly the right ones at k=8.
    A `random.sample`-style draw would silently reshuffle and void all of it.
    """
    _pool_of(tmp_path, 12)
    _rep_small, small = _ids_at_k(tmp_path, 4)
    _rep_big, big = _ids_at_k(tmp_path, 8)
    assert len(small) == 4 and len(big) == 8
    assert big[:4] == small, "the prefix must be stable as k rises"
    assert set(small) < set(big)


def test_k_is_deterministic_across_rebuilds(tmp_path: Path):
    _pool_of(tmp_path, 10)
    _r1, first = _ids_at_k(tmp_path, 5)
    _r2, second = _ids_at_k(tmp_path, 5)
    assert first == second


def test_k_above_the_pool_is_a_no_op(tmp_path: Path):
    """Asking for more scenes than exist must take all of them, not raise or resample."""
    _pool_of(tmp_path, 6)
    rep, ids = _ids_at_k(tmp_path, 99)
    assert len(ids) == 6
    assert rep["scenes_before_k"] == 6
    assert rep["content_distinct_scenes"] == 6


def test_k_none_keeps_every_scene_in_collapse_order(tmp_path: Path):
    _pool_of(tmp_path, 5)
    rep, ids = _ids_at_k(tmp_path, None)
    assert ids == sorted(ids), "unsampled pools keep the deterministic collapse order"
    assert rep["k"] is None and rep["seed"] == 42


def test_k_and_scenes_before_k_are_both_reported(tmp_path: Path):
    """"No silent caps": a truncated pool must say what it was truncated from."""
    _pool_of(tmp_path, 9)
    rep, _ids = _ids_at_k(tmp_path, 3)
    assert rep["scenes_before_k"] == 9
    assert rep["content_distinct_scenes"] == 3
    assert rep["k"] == 3


def test_new_config_keys_round_trip(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "max_face_frac: 0.12\nrequire_skin: true\nexclude_tasks: [style, env]\n"
        "exclude_id_substrings: [_attr_mod_facial_]\nk: 611\nseed: 42\n", encoding="utf-8")
    loaded = BFP.load_config(cfg)
    assert loaded["max_face_frac"] == 0.12
    assert loaded["require_skin"] is True
    assert loaded["exclude_tasks"] == ["style", "env"]
    assert loaded["exclude_id_substrings"] == ["_attr_mod_facial_"]
    assert loaded["k"] == 611 and loaded["seed"] == 42


# --------------------------------------------------------------------------- #
# ★ The collision one level below the task label                               #
# --------------------------------------------------------------------------- #
def test_exclude_id_substrings_catches_the_sub_task_the_task_column_hides(tmp_path: Path):
    """OmniEdit labels 293 facial-expression edits `attribute_modification`.

    The coarse task column cannot distinguish "make him be angry" (acts on the face, where the
    skin manipulation acts -- a collision) from "turn the color of scroll to black" (a disjoint
    object, measured skin overlap 0.172 -- no collision). The finer sub-task is in the id, so
    the screen has to reach it there. Excluding the whole `attribute_modification` family, as
    the plan originally proposed, would take 281 clean colour scenes with it.
    """
    samples = []
    for i, sid in enumerate([
        "omniedit_train_task_attr_mod_facial_1",
        "omniedit_train_task_attr_mod_color_2",
        "omniedit_train_task_obj_add_3",
    ]):
        _write_image(tmp_path / f"o{i}.png", seed=i)
        _write_image(tmp_path / f"e{i}.png", seed=i + 50)
        rec = _sample(sid, instruction="Add a hat", category=ContentCategory.HUMAN,
                      original=f"o{i}.png", edited=f"e{i}.png")
        rec.metadata.raw_task = "attribute_modification" if "attr_mod" in sid else "addition"
        samples.append(rec)
    io.write_jsonl(tmp_path / "m.jsonl", samples)

    rep = BFP.build(root=tmp_path, samples_path="m.jsonl", write=False,
                    require_human_category=False,
                    exclude_id_substrings=["_attr_mod_facial_"])
    assert rep["dropped_by_id_exclusion"] == {"_attr_mod_facial_": 1}
    assert rep["after_id_exclusion"] == 2, "the colour sub-family must survive"


def test_exclude_id_substrings_is_off_by_default(tmp_path: Path):
    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    rec = _sample("omniedit_train_task_attr_mod_facial_1", instruction="Add a hat",
                  category=ContentCategory.HUMAN, original="o.png", edited="e.png")
    io.write_jsonl(tmp_path / "m.jsonl", [rec])
    rep = BFP.build(root=tmp_path, samples_path="m.jsonl", write=False,
                    require_human_category=False)
    assert "dropped_by_id_exclusion" not in rep
    assert rep["content_distinct_scenes"] == 1


@pytest.mark.parametrize("instruction", [
    "make him be angry",
    "let him be excited",
    "make the dryad feel amused",
    "portray Wanda with a sad expression",
    "let the elf cry",
    "make her look surprised",
    "give him a joyful mood",
])
def test_expression_instructions_are_screened_out(instruction):
    """The same collision, caught from text alone.

    OmniEdit encodes its sub-task in the id, so the pool config screens it there too -- but a
    corpus that publishes no sub-task label leaves this list as the only defence. It also
    catches 6 rows the id screen misses even where both apply.
    """
    assert instruction_ok(instruction) is False


@pytest.mark.parametrize("instruction", [
    "Add necklace",
    "Replace the denim jacket with white sweater",
    "turn the color of scroll to black",
    "Remove the vase from the table",
])
def test_non_expression_instructions_still_pass(instruction):
    """The expression list must not swallow the object edits the pool is built from."""
    assert instruction_ok(instruction) is True


# --------------------------------------------------------------------------- #
# ★ The construct screen's pass list feeds back into the build                 #
# --------------------------------------------------------------------------- #
def test_include_ids_filters_before_the_collapse_and_the_prefix(tmp_path: Path):
    """The screen must apply BEFORE the ORB collapse and the seeded prefix, not after.

    Filtering the finished manifest would take a prefix of a contaminated list and then delete
    from it -- a different, smaller sample than taking a prefix of the clean one.
    """
    import json as _json

    samples = []
    for i in range(6):
        _write_image(tmp_path / f"o{i}.png", seed=i * 5 + 1)
        _write_image(tmp_path / f"e{i}.png", seed=i * 5 + 2)
        samples.append(_sample(f"s{i:02d}", instruction="Add a hat",
                               category=ContentCategory.HUMAN,
                               original=f"o{i}.png", edited=f"e{i}.png"))
    io.write_jsonl(tmp_path / "m.jsonl", samples)
    (tmp_path / "pass.json").write_text(_json.dumps(["s00", "s02", "s04"]), encoding="utf-8")

    rep = BFP.build(root=tmp_path, samples_path="m.jsonl", out_path="out.jsonl", write=True,
                    require_human_category=False, include_ids="pass.json", k=2)
    assert rep["include_ids_listed"] == 3
    assert rep["dropped_by_include_ids"] == 3
    assert rep["after_include_ids"] == 3
    assert rep["scenes_before_k"] == 3, "the collapse must run on the SCREENED set"
    got = [r.sample_id for r in io.read_jsonl(tmp_path / "out.jsonl", SampleRecord)]
    assert len(got) == 2
    assert set(got) <= {"s00", "s02", "s04"}


def test_include_ids_accepts_a_passed_key_object(tmp_path: Path):
    import json as _json

    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    io.write_jsonl(tmp_path / "m.jsonl", [
        _sample("keep", instruction="Add a hat", category=ContentCategory.HUMAN,
                original="o.png", edited="e.png")])
    (tmp_path / "pass.json").write_text(_json.dumps({"passed": ["keep"]}), encoding="utf-8")
    rep = BFP.build(root=tmp_path, samples_path="m.jsonl", write=False,
                    require_human_category=False, include_ids="pass.json")
    assert rep["after_include_ids"] == 1


def test_include_ids_is_off_by_default(tmp_path: Path):
    _write_image(tmp_path / "o.png")
    _write_image(tmp_path / "e.png")
    io.write_jsonl(tmp_path / "m.jsonl", [
        _sample("s", instruction="Add a hat", category=ContentCategory.HUMAN,
                original="o.png", edited="e.png")])
    rep = BFP.build(root=tmp_path, samples_path="m.jsonl", write=False,
                    require_human_category=False)
    assert "include_ids" not in rep
    assert rep["content_distinct_scenes"] == 1


# --------------------------------------------------------------------------- #
# ★ The two-pass build must be reachable from the CLI                          #
# --------------------------------------------------------------------------- #
def test_cli_passes_include_ids_and_k_through_to_build(monkeypatch, tmp_path: Path):
    """Pass 2 replays pass 1's config with a screen list and a k.

    They are CLI flags rather than a second YAML on purpose: the only difference between the
    two passes must be on the command line, or the passes can drift apart in a screening
    parameter and the paid construct screen would then describe a pool nobody built.
    """
    seen = {}

    def _fake_build(**kwargs):
        seen.update(kwargs)
        return {"content_distinct_scenes": 0}

    monkeypatch.setattr(BFP, "build", _fake_build)
    cfg = tmp_path / "pool.yaml"
    cfg.write_text(
        "samples_path: s.jsonl\nout_path: o.jsonl\ndetector: true\n", encoding="utf-8")
    BFP.main(["--config", str(cfg), "--include-ids", "passed.json", "--k", "300"])
    assert seen["include_ids"] == "passed.json"
    assert seen["k"] == 300
    # ...and pass 1 leaves both unset rather than defaulting to something.
    seen.clear()
    BFP.main(["--config", str(cfg)])
    assert "include_ids" not in seen and "k" not in seen
