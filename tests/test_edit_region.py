"""Tests for the C-class edit-region estimator (edit_region.py, §5.4 C0)."""

from __future__ import annotations

from pathlib import Path

from conftest import make_rgb_image
from PIL import Image

from edit_judge_bias.bias.edit_region import EditRegion, estimate_edit_region


def _pair(dirp: Path, box=(90, 70, 140, 110), size=(200, 160)):
    """An original and an edited image differing only inside `box`."""
    orig = Image.new("RGB", size, (120, 120, 120))
    edited = orig.copy()
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            edited.putpixel((x, y), (210, 40, 40))
    op, ep = dirp / "o.png", dirp / "e.png"
    orig.save(op)
    edited.save(ep)
    return op, ep


def test_diff_localizes_the_changed_box(tmp_path: Path):
    op, ep = _pair(tmp_path, box=(90, 70, 140, 110))
    reg = estimate_edit_region(ep, original=op)
    assert reg.method == "diff"
    l, t, r, b = reg.bbox
    # The bbox should tightly enclose the painted square (allow a few px of slack).
    assert abs(l - 90) <= 4 and abs(t - 70) <= 4
    assert abs(r - 140) <= 4 and abs(b - 110) <= 4
    assert 0 < reg.area_frac < 0.3


def test_fallback_when_no_original(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", size=(200, 160))
    reg = estimate_edit_region(src, original=None)
    assert reg.method == "fallback"
    assert isinstance(reg, EditRegion)
    # Centered box roughly centered in the frame.
    cx, cy = reg.center
    assert 80 <= cx <= 120 and 60 <= cy <= 100


def test_mask_takes_priority(tmp_path: Path):
    op, ep = _pair(tmp_path)
    mask = Image.new("L", (200, 160), 0)
    for x in range(10, 40):
        for y in range(10, 30):
            mask.putpixel((x, y), 255)
    mp = tmp_path / "m.png"
    mask.save(mp)
    reg = estimate_edit_region(ep, original=op, mask=mp)
    assert reg.method == "mask"
    assert reg.bbox == (10, 10, 40, 30)


def test_missing_edited_is_non_fatal(tmp_path: Path):
    reg = estimate_edit_region(tmp_path / "nope.png")
    assert reg.method == "fallback"


# --------------------------------------------------------------------------- #
# ★ Mask polarity — each test here pins a bug that shipped judged data.        #
#                                                                              #
# MagicBrush stores its 349 edit masks as RGBA with **alpha <= 127 marking the  #
# edited region**. `_load` used to do `convert("RGB")`, dropping alpha, and the #
# surviving RGB is bright almost everywhere -- so `point(p > 0).getbbox()`      #
# returned the WHOLE FRAME. Measured 2026-07-30 over 30 judged samples that     #
# ship a mask, IoU against the |edited-original| diff region: alpha_low 0.573,  #
# luma_high 0.008. The old reading was uncorrelated with the edit, not merely   #
# imprecise, and it reached 273 already-judged biased images.                   #
# --------------------------------------------------------------------------- #
def _rgba_alpha_low_mask(size=(200, 160), box=(10, 10, 40, 30)) -> Image.Image:
    """A MagicBrush-shaped mask: opaque photo content, edited region punched into alpha."""
    # RGB is bright everywhere -- this is exactly what made the flattened read return
    # the full frame. Keep it non-constant so the test cannot pass by accident.
    mask = Image.new("RGBA", size, (200, 180, 160, 255))
    for x in range(size[0]):
        for y in range(size[1]):
            mask.putpixel((x, y), (150 + (x % 50), 140 + (y % 40), 130, 255))
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            r, g, b, _ = mask.getpixel((x, y))
            mask.putpixel((x, y), (r, g, b, 0))
    return mask


def test_rgba_mask_region_comes_from_alpha_not_luma(tmp_path: Path):
    """The whole bug in one assertion: alpha carries the region, RGB does not."""
    op, ep = _pair(tmp_path)
    mp = tmp_path / "m.png"
    _rgba_alpha_low_mask(box=(10, 10, 40, 30)).save(mp)

    reg = estimate_edit_region(ep, original=op, mask=mp)
    assert reg.method == "mask"
    assert reg.mask_polarity == "alpha_low"
    assert reg.bbox == (10, 10, 40, 30)
    # The frame is 200x160 = 32,000 px; the region is 30x20 = 600. A full-frame read
    # would report 1.0, which is precisely the defect.
    assert reg.area_frac < 0.05


def test_luma_high_reproduces_the_published_full_frame_read(tmp_path: Path):
    """`luma_high` must stay bit-exact, or the §6 ablation cannot reproduce the paper."""
    op, ep = _pair(tmp_path)
    mp = tmp_path / "m.png"
    _rgba_alpha_low_mask().save(mp)

    reg = estimate_edit_region(ep, original=op, mask=mp, mask_polarity="luma_high")
    assert reg.method == "mask"
    assert reg.mask_polarity == "luma_high"
    assert reg.bbox == (0, 0, 200, 160)
    assert reg.area_frac == 1.0


def test_auto_falls_back_to_luma_when_alpha_carries_no_region(tmp_path: Path):
    """A fully opaque RGBA mask has no alpha signal; reading it would select everything.

    This is why `_alpha_of` rejects a degenerate band instead of trusting the mode.
    """
    op, ep = _pair(tmp_path)
    mask = Image.new("RGBA", (200, 160), (0, 0, 0, 255))
    for x in range(10, 40):
        for y in range(10, 30):
            mask.putpixel((x, y), (255, 255, 255, 255))
    mp = tmp_path / "m.png"
    mask.save(mp)

    reg = estimate_edit_region(ep, original=op, mask=mp)
    assert reg.mask_polarity == "luma_high"
    assert reg.bbox == (10, 10, 40, 30)


def test_plain_greyscale_mask_is_unaffected(tmp_path: Path):
    """Sources that ship an L-mode mask must behave exactly as before."""
    op, ep = _pair(tmp_path)
    mask = Image.new("L", (200, 160), 0)
    for x in range(5, 25):
        for y in range(5, 20):
            mask.putpixel((x, y), 255)
    mp = tmp_path / "m.png"
    mask.save(mp)

    reg = estimate_edit_region(ep, original=op, mask=mp)
    assert reg.mask_polarity == "luma_high"
    assert reg.bbox == (5, 5, 25, 20)


def test_explicit_alpha_polarity_on_a_maskless_mode_declines_rather_than_guessing(
    tmp_path: Path,
):
    """Asking for a channel the mask lacks must fall through to `diff`, not read another.

    Silently reading luma when `alpha_low` was requested is how a config typo becomes an
    invisible change of manipulation dose.
    """
    op, ep = _pair(tmp_path, box=(90, 70, 140, 110))
    mask = Image.new("L", (200, 160), 0)
    for x in range(5, 25):
        for y in range(5, 20):
            mask.putpixel((x, y), 255)
    mp = tmp_path / "m.png"
    mask.save(mp)

    reg = estimate_edit_region(ep, original=op, mask=mp, mask_polarity="alpha_low")
    assert reg.method == "diff"
    assert reg.mask_polarity is None


def test_alpha_high_is_the_complement(tmp_path: Path):
    op, ep = _pair(tmp_path)
    mp = tmp_path / "m.png"
    _rgba_alpha_low_mask(box=(10, 10, 40, 30)).save(mp)

    reg = estimate_edit_region(ep, original=op, mask=mp, mask_polarity="alpha_high")
    assert reg.mask_polarity == "alpha_high"
    # The opaque part is everything except the punched box, so its bbox is the frame.
    assert reg.bbox == (0, 0, 200, 160)


def test_unknown_polarity_raises(tmp_path: Path):
    op, ep = _pair(tmp_path)
    try:
        estimate_edit_region(ep, original=op, mask_polarity="sideways")
    except ValueError as exc:
        assert "mask_polarity" in str(exc)
    else:  # pragma: no cover - the raise is the contract
        raise AssertionError("an unknown mask_polarity must raise, not pick a default")
