"""WP-A4a — the validator sensitivity positive control.

The tests that matter here are not "does it draw pixels". They are:
  - the damage LANDS, and lands where the edit is;
  - `severity` is monotone, so a detection threshold is measurable rather than binary;
  - the injector REFUSES to fabricate a control it cannot ground; and
  - it cannot leak into a judge arm, where it would be a quality-destroying cue sitting
    in a table whose entire premise is quality preservation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from edit_judge_bias.bias.edit_damage import EditDamageInjector
from edit_judge_bias.bias.registry import get_injector


def _scene(tmp_path: Path, box=(20, 20, 44, 44)):
    """An 'original' and an 'edited' that differ only inside `box`.

    The edit is a CHECKERBOARD, not a flat fill: `blur` mode destroys detail, and a
    uniform patch has none to destroy — it would blur to itself and the mode would test
    as a no-op while working perfectly on real photographs.
    """
    original = Image.new("RGB", (64, 64), (30, 90, 200))
    edited = original.copy()
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            edited.putpixel((x, y), (240, 230, 20) if (x + y) % 2 else (10, 10, 10))
    op, ep = tmp_path / "orig.png", tmp_path / "edit.png"
    original.save(op)
    edited.save(ep)
    return op, ep


def _apply(tmp_path: Path, **config):
    op, ep = _scene(tmp_path)
    out = tmp_path / "damaged.png"
    cfg = {"original_image_path_abs": str(op)}
    cfg.update(config)
    return EditDamageInjector().apply(ep, out, config=cfg), op, ep, out


def test_full_revert_puts_the_original_pixels_back(tmp_path: Path):
    """severity=1.0 is the ground truth the whole control rests on: the edit region
    becomes the original, so the instruction was provably not carried out there. No
    rater, no judgement call, no annotation budget."""
    res, op, ep, out = _apply(tmp_path, severity=1.0)
    assert res.success, res.message
    original, damaged = Image.open(op).convert("RGB"), Image.open(out).convert("RGB")
    # Inside the edit region the damaged image IS the original.
    assert damaged.getpixel((32, 32)) == original.getpixel((32, 32))
    # Outside it, nothing moved.
    assert damaged.getpixel((5, 5)) == Image.open(ep).convert("RGB").getpixel((5, 5))
    assert res.params["region_method"] == "diff"


def test_severity_is_monotone_so_a_threshold_is_measurable(tmp_path: Path):
    """A single all-or-nothing control answers "is it blind"; a ladder answers "how
    much damage does it take", which is the number §5.7's null verdicts need beside
    them."""
    changes = []
    for sev in (0.25, 0.5, 1.0):
        res, *_ = _apply(tmp_path, severity=sev)
        assert res.success, res.message
        changes.append(res.params["mean_abs_change"])
    assert changes == sorted(changes) and changes[0] > 0
    assert changes[2] > changes[0] * 2


def test_the_damage_is_measured_on_pixels_not_inferred_from_geometry(tmp_path: Path):
    """`distraction`'s spec violation was found because rebuilt CANVAS geometry was
    trusted over pixels that never landed (22 reported, 21 real). The same mistake is
    available here, so the manifest carries a measured change, not a bbox area."""
    res, *_ = _apply(tmp_path, severity=1.0)
    p = res.params
    # 24x24 damaged out of 64x64 = 0.1406 of the frame.
    assert p["changed_frac"] == pytest.approx(24 * 24 / (64 * 64), abs=0.01)
    assert p["mean_abs_change"] > 0


def test_revert_refuses_the_fallback_box_instead_of_faking_a_control(tmp_path: Path):
    """Guard 2. Without an original the region estimator returns a centered box covering
    60% of the frame; damaging that would produce a picture that LOOKS like a positive
    control while proving nothing about the edit. Failing loudly is the only safe
    behaviour — and `apply` turns it into success=False, so a batch still continues."""
    _, ep = _scene(tmp_path)
    res = EditDamageInjector().apply(ep, tmp_path / "x.png", config={})
    assert res.success is False
    assert "fallback" in res.message


def test_blur_mode_damages_detail_without_needing_the_original(tmp_path: Path):
    """The second failure mode: the edit is still there but its detail is destroyed.
    That maps onto a DIFFERENT validator question (detail preservation vs instruction
    adherence), so the two modes probe the instrument in two places."""
    res, _, ep, out = _apply(tmp_path, mode="blur")
    assert res.success, res.message
    assert res.params["mean_abs_change"] > 0
    assert res.params["mode"] == "blur"


def test_bad_configuration_is_rejected_not_silently_clamped(tmp_path: Path):
    for bad in ({"severity": 0.0}, {"severity": 1.5}, {"mode": "smear"}):
        res, *_ = _apply(tmp_path, **bad)
        assert res.success is False, bad


def test_it_is_registered_but_reachable_only_by_name(tmp_path: Path):
    """It must ride the identical code path (anything special-cased would not be a
    control), so it is in the registry like `sham`."""
    assert get_injector("edit_damage") is not None


def test_the_control_never_enters_a_judge_arm_or_the_published_manifest():
    """The load-bearing guard. This injector destroys edit quality on purpose; a single
    row of it inside claim A would be a cue that genuinely lowers quality sitting in a
    table whose premise is that no cue does. Checked against the arm configs and the
    published biased manifest rather than against a promise."""
    repo = Path(__file__).resolve().parents[1]
    configs = sorted((repo / "configs" / "experiment").glob("*.yaml"))
    assert configs, "no configs found — the guard would pass vacuously"

    # (a) It must never be JUDGED at all. A judge arm scoring these images would put a
    #     cue that genuinely lowers quality inside a table whose premise is that no cue
    #     does.  There is no legitimate reason for a scoring/pairwise arm to name it.
    judged = [p for p in configs if p.name.startswith(("scoring_", "pairwise_"))]
    assert judged, "no judge arm configs found — the guard would pass vacuously"
    for cfg in judged:
        assert "edit_damage" not in cfg.read_text(encoding="utf-8"), cfg.name

    # (b) A VALIDATOR arm is allowed to name it — WP-A4b measures validator sensitivity
    #     on exactly these images, and that is the arm's whole purpose.  What must never
    #     happen is such a config writing into the published result tree, where
    #     `build_quality_combined` globs `validation__*.jsonl` and would read the control
    #     as an eleventh cue.  So the invariant is about the DESTINATION, not the name.
    published_roots = ("results/v2/", "results\\v2\\", "results/v2_fairness")
    named_control = False
    for cfg in configs:
        text = cfg.read_text(encoding="utf-8")
        if "edit_damage" not in text:
            continue
        named_control = True
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or ":" not in stripped:
                continue
            value = stripped.split(":", 1)[1].strip()
            assert not value.startswith(published_roots), (
                f"{cfg.name} names the edit_damage control AND writes into the published "
                f"tree ({stripped!r}) — the control would be read as an eleventh cue"
            )
    assert named_control, (
        "no config names edit_damage — either the injection config was deleted or this "
        "guard has gone vacuous"
    )

    # (c) It must never reach the published biased manifest, which every judge arm reads.
    manifest = repo / "data" / "manifests" / "biased_samples_full_v2.jsonl"
    if manifest.exists():
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    assert json.loads(line)["bias_type"] != "edit_damage"
