"""Tests for the PIL bias injectors (Milestone 2)."""

from __future__ import annotations

from pathlib import Path

from conftest import make_rgb_image
from PIL import Image

from edit_judge_bias.bias.aesthetic_filter import AestheticFilterInjector
from edit_judge_bias.bias.brightness import BrightnessInjector
from edit_judge_bias.bias.detail_caption import DetailCaptionInjector
from edit_judge_bias.bias.distraction import DistractionInjector
from edit_judge_bias.bias.padding import PaddingInjector
from edit_judge_bias.bias.region_annotation import RegionAnnotationInjector
from edit_judge_bias.bias.saturation import SaturationInjector
from edit_judge_bias.bias.sham import ShamInjector
from edit_judge_bias.bias.text_overlay import (
    TextOverlayInjector,
    extract_keyword,
    truncate_words,
)
from edit_judge_bias.bias.watermark import WatermarkInjector
from edit_judge_bias.bias.zoom_inset import ZoomInsetInjector


def _mean(img: Image.Image):
    px = list(img.getdata())
    n = len(px)
    return tuple(sum(c[i] for c in px) / n for i in range(3))


# --------------------------------------------------------------------------- #
# Brightness / Saturation                                                      #
# --------------------------------------------------------------------------- #
def test_brightness_increases_luminance(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", color=(100, 100, 100))
    out = tmp_path / "out.png"
    res = BrightnessInjector().apply(src, out, config={"factor": 1.5})
    assert res.success and out.exists()
    assert res.bias_strength == 1.5
    assert _mean(Image.open(out))[0] > 100


def test_brightness_factor_below_one_darkens(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", color=(100, 100, 100))
    out = tmp_path / "out.png"
    BrightnessInjector().apply(src, out, config={"factor": 0.5})
    assert _mean(Image.open(out))[0] < 100


def test_saturation_zero_is_grayscale(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", color=(200, 50, 25))
    out = tmp_path / "out.png"
    SaturationInjector().apply(src, out, config={"factor": 0.0})
    r, g, b = Image.open(out).getpixel((10, 10))
    assert r == g == b  # desaturated -> equal channels


# --------------------------------------------------------------------------- #
# Watermark                                                                    #
# --------------------------------------------------------------------------- #
def test_watermark_preserves_size_and_marks_corner(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", size=(200, 160), color=(120, 120, 120))
    out = tmp_path / "out.png"
    res = WatermarkInjector().apply(src, out, config={"text": "Verified Edit"})
    assert res.success
    orig = Image.open(src)
    marked = Image.open(out)
    assert marked.size == orig.size
    assert res.params["overlay_text"] == "Verified Edit"
    # The bottom-right region must change; the top-left corner must not.
    assert _region_changed(orig, marked, box=(140, 110, 200, 160))
    assert not _region_changed(orig, marked, box=(0, 0, 40, 40))


def _region_changed(a: Image.Image, b: Image.Image, box) -> bool:
    return list(a.crop(box).getdata()) != list(b.crop(box).getdata())


# --------------------------------------------------------------------------- #
# Text overlay                                                                 #
# --------------------------------------------------------------------------- #
def test_extract_keyword_skips_verbs_and_stopwords():
    assert extract_keyword("Remove the red slippers from the image") == "slippers"
    assert extract_keyword("Add a cat to the shoe rack") == "rack"
    assert extract_keyword("") == ""


def test_truncate_words():
    assert truncate_words("a b c d e", 3) == "a b c"


def test_text_overlay_records_text_and_keeps_size(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", size=(200, 160))
    out = tmp_path / "out.png"
    res = TextOverlayInjector().apply(
        src, out, config={"mode": "keyword", "band": "top"},
        sample={"instruction": "Remove the red slippers"},
    )
    assert res.success
    assert res.params["overlay_text"] == "slippers"
    assert Image.open(out).size == (200, 160)
    assert _region_changed(Image.open(src), Image.open(out), box=(0, 0, 200, 30))


# --------------------------------------------------------------------------- #
# Padding                                                                      #
# --------------------------------------------------------------------------- #
def test_padding_grows_image_and_preserves_center(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", size=(100, 80), color=(123, 45, 67))
    out = tmp_path / "out.png"
    res = PaddingInjector().apply(src, out, config={"ratio": 0.10})
    assert res.success
    padded = Image.open(out)
    bx, by = int(round(100 * 0.10)), int(round(80 * 0.10))
    assert padded.size == (100 + 2 * bx, 80 + 2 * by)
    assert res.params["padding_ratio"] == 0.10
    # Border is black; the pasted center keeps the original color.
    assert padded.getpixel((0, 0)) == (0, 0, 0)
    assert padded.getpixel((bx + 5, by + 5)) == (123, 45, 67)


# --------------------------------------------------------------------------- #
# Full-version B6 / C1-C4 injectors                                            #
# --------------------------------------------------------------------------- #
def _edit_pair(tmp_path: Path, box=(90, 70, 140, 110), size=(200, 160)):
    """Original + edited differing only inside `box` (for C-class localization)."""
    orig = Image.new("RGB", size, (120, 120, 120))
    edited = orig.copy()
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            edited.putpixel((x, y), (210, 40, 40))
    op, ep = tmp_path / "o.png", tmp_path / "e.png"
    orig.save(op)
    edited.save(ep)
    return op, ep


def test_aesthetic_filter_keeps_size_and_changes_pixels(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", size=(200, 160), color=(120, 90, 70))
    out = tmp_path / "out.png"
    res = AestheticFilterInjector().apply(src, out, config={})
    assert res.success
    marked = Image.open(out)
    assert marked.size == (200, 160)
    assert res.bias_strength == 1.1
    # The filter chain actually changes pixels (tone + micro-contrast).
    assert list(Image.open(src).getdata()) != list(marked.getdata())


def test_region_annotation_box_draws_red_on_edit(tmp_path: Path):
    op, ep = _edit_pair(tmp_path)
    out = tmp_path / "out.png"
    res = RegionAnnotationInjector().apply(
        ep, out, config={"mode": "box", "original_image_path_abs": str(op)}
    )
    assert res.success
    assert res.params["region_method"] == "diff"
    # A red pixel must appear on the box border around the detected region.
    marked = Image.open(out)
    l, t, r, b = res.params["bbox"]
    assert marked.getpixel((l, (t + b) // 2))[0] > 180  # left edge is reddish


def test_region_annotation_arrow_variant(tmp_path: Path):
    op, ep = _edit_pair(tmp_path)
    out = tmp_path / "out.png"
    res = RegionAnnotationInjector().apply(
        ep, out, config={"mode": "arrow", "original_image_path_abs": str(op)}
    )
    assert res.success and res.params["mode"] == "arrow"
    assert Image.open(out).size == (200, 160)


def test_zoom_inset_pastes_in_corner_clear_of_edit(tmp_path: Path):
    op, ep = _edit_pair(tmp_path)
    out = tmp_path / "out.png"
    res = ZoomInsetInjector().apply(
        ep, out, config={"original_image_path_abs": str(op)}
    )
    assert res.success
    assert res.params["corner"] in {"top-left", "top-right", "bottom-left", "bottom-right"}
    assert Image.open(out).size == (200, 160)  # composition unchanged


def test_detail_caption_adds_bottom_strip_and_preserves_top(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", size=(200, 160), color=(80, 120, 160))
    out = tmp_path / "out.png"
    res = DetailCaptionInjector().apply(
        src, out, config={}, sample={"instruction": "replace the sofa with an armchair"}
    )
    assert res.success
    captioned = Image.open(out)
    assert captioned.size[0] == 200 and captioned.size[1] > 160  # grew downward
    assert "replace the sofa" in res.params["caption"]
    # Original pixels preserved at the top.
    assert captioned.getpixel((5, 5)) == (80, 120, 160)


def test_distraction_pastes_outside_edit_region(tmp_path: Path):
    op, ep = _edit_pair(tmp_path)
    out = tmp_path / "out.png"
    res = DistractionInjector().apply(
        ep, out, config={"original_image_path_abs": str(op)}, seed=7
    )
    assert res.success
    assert res.params["area_frac"] <= 0.04 + 1e-6
    assert res.params["asset"] in {"star", "heart", "circle", "triangle", "butterfly"}
    assert Image.open(out).size == (200, 160)


def test_distraction_asset_pin_is_reproducible(tmp_path: Path):
    op, ep = _edit_pair(tmp_path)
    o1, o2 = tmp_path / "a.png", tmp_path / "b.png"
    cfg = {"original_image_path_abs": str(op), "asset": "star"}
    DistractionInjector().apply(ep, o1, config=cfg, seed=1)
    DistractionInjector().apply(ep, o2, config=cfg, seed=1)
    assert o1.read_bytes() == o2.read_bytes()
    assert DistractionInjector().apply(ep, tmp_path / "c.png", config=cfg).params["asset"] == "star"


# --------------------------------------------------------------------------- #
# Cross-cutting: determinism + failure handling                               #
# --------------------------------------------------------------------------- #
def test_injection_is_deterministic(tmp_path: Path):
    src = make_rgb_image(tmp_path / "e.png", size=(150, 120))
    o1, o2 = tmp_path / "a.png", tmp_path / "b.png"
    WatermarkInjector().apply(src, o1, config={}, seed=7)
    WatermarkInjector().apply(src, o2, config={}, seed=7)
    assert o1.read_bytes() == o2.read_bytes()


def test_apply_missing_image_is_non_fatal(tmp_path: Path):
    res = BrightnessInjector().apply(tmp_path / "nope.png", tmp_path / "out.png")
    assert res.success is False
    assert res.message
    assert not (tmp_path / "out.png").exists()


# --------------------------------------------------------------------------- #
# Resolution invariance (added 2026-07-25).                                    #
#                                                                              #
# The full-version pool mixes 500x500, 512x512 and 1024x1024 images. Any bias  #
# sized in absolute pixels would cover a different share of a large frame than #
# a small one, so a measured score drop could partly be a function of          #
# resolution -- a confound that cannot be separated after the run. Every       #
# injector's footprint must therefore scale with the frame.                    #
# --------------------------------------------------------------------------- #
import pytest as _pytest

from edit_judge_bias.bias.base import relative_px as _relative_px
from edit_judge_bias.bias.registry import get_injector as _get_injector


def _structured(path, size):
    """Non-flat content, so overlays actually register as changed pixels."""
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256)
    img.save(path)
    return path


def _changed_fraction(before: Image.Image, after: Image.Image) -> float:
    a, b = before.convert("RGB"), after.convert("RGB")
    if a.size != b.size:  # canvas-extending bias: compare the shared top-left area
        w, h = min(a.size[0], b.size[0]), min(a.size[1], b.size[1])
        a, b = a.crop((0, 0, w, h)), b.crop((0, 0, w, h))
    pa, pb = list(a.getdata()), list(b.getdata())
    return sum(1 for x, y in zip(pa, pb) if x != y) / len(pa)


@_pytest.mark.parametrize(
    "bias_type",
    ["text_overlay", "watermark", "region_annotation", "zoom_inset", "distraction"],
)
def test_overlay_footprint_is_a_constant_fraction_of_the_frame(tmp_path, bias_type):
    """The same bias must cover ~the same *share* of a 512px and a 1024px frame."""
    fracs = {}
    for size in ((512, 512), (1024, 1024)):
        src = _structured(tmp_path / f"{bias_type}_{size[0]}.png", size)
        out = tmp_path / f"{bias_type}_{size[0]}_biased.png"
        res = _get_injector(bias_type).apply(
            src, out, config={}, sample={"instruction": "add a red hat to the man"}, seed=42
        )
        assert res.success, res.message
        with Image.open(src) as a, Image.open(out) as b:
            fracs[size[0]] = _changed_fraction(a, b)
    small, large = fracs[512], fracs[1024]
    assert small > 0.0005, f"{bias_type} changed almost nothing at 512px: {small}"
    # Rasterisation of text/strokes is not perfectly scale-free, so allow a 2x
    # band -- an absolute-pixel size would show a ~4x area ratio instead.
    assert 0.5 <= large / small <= 2.0, (
        f"{bias_type} footprint is resolution-dependent: "
        f"{small:.4f} at 512px vs {large:.4f} at 1024px"
    )


def test_aesthetic_filter_sharpening_radius_scales_with_resolution(tmp_path):
    from edit_judge_bias.bias.aesthetic_filter import AestheticFilterInjector

    radii = {}
    for size in ((512, 512), (1024, 1024)):
        src = _structured(tmp_path / f"ae_{size[0]}.png", size)
        res = AestheticFilterInjector().apply(
            src, tmp_path / f"ae_{size[0]}_out.png", config={}, sample={}, seed=42
        )
        assert res.success, res.message
        radii[size[0]] = res.params["unsharp"][0]
    assert radii[1024] == _pytest.approx(2 * radii[512], rel=0.05)


def test_padding_scales_by_construction(tmp_path):
    """padding is already a ratio of w/h; pinned so a refactor cannot regress it."""
    from edit_judge_bias.bias.padding import PaddingInjector

    for size, expect in (((512, 512), 512), ((1024, 1024), 1024)):
        src = _structured(tmp_path / f"pad_{size[0]}.png", size)
        out = tmp_path / f"pad_{size[0]}_out.png"
        res = PaddingInjector().apply(src, out, config={"ratio": 0.1}, sample={}, seed=42)
        assert res.success, res.message
        with Image.open(out) as im:
            assert im.size[0] == expect + 2 * round(expect * 0.1)


@_pytest.mark.parametrize(
    "bias_type,ratio_key,px_key",
    [
        ("region_annotation", "width_ratio", "width_px"),
        ("zoom_inset", "border_ratio", "border_px"),
    ],
)
def test_an_explicit_pixel_value_still_overrides_the_ratio(tmp_path, bias_type, ratio_key, px_key):
    """Ablations need to be able to pin an absolute size on purpose."""
    src = _structured(tmp_path / f"{bias_type}_ov.png", (512, 512))
    res = _get_injector(bias_type).apply(
        src,
        tmp_path / f"{bias_type}_ov_out.png",
        config={px_key: 9},
        sample={"instruction": "add a hat"},
        seed=42,
    )
    assert res.success, res.message
    assert res.params[px_key] == 9
    assert res.params[ratio_key] is None
    assert res.params[f"{ratio_key}_source"] == "explicit_px"


def test_relative_px_records_what_it_resolved():
    px, params = _relative_px(
        {}, ratio_key="w_ratio", px_key="w_px", short_edge=1024, default_ratio=0.01
    )
    assert px == 10 and params == {
        "w_px": 10,
        "w_ratio": 0.01,
        "w_ratio_source": "ratio",
    }
    px, params = _relative_px(
        {"w_px": 4}, ratio_key="w_ratio", px_key="w_px", short_edge=1024, default_ratio=0.01
    )
    assert px == 4 and params["w_ratio_source"] == "explicit_px"


def test_relative_px_never_returns_zero_on_a_tiny_image():
    px, _ = _relative_px(
        {}, ratio_key="w_ratio", px_key="w_px", short_edge=16, default_ratio=0.001
    )
    assert px == 1


# --------------------------------------------------------------------------- #
# Sham (placebo control)                                                       #
# --------------------------------------------------------------------------- #
def _smooth(path: Path, size=(120, 90)) -> Path:
    """A photo-like smooth gradient. A flat colour survives a lossy codec exactly,
    and per-pixel noise is a worst case no real photograph resembles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    img = Image.new("RGB", size)
    img.putdata([
        (30 + 200 * x // w, 20 + 200 * y // h, 60 + 150 * (x + y) // (w + h))
        for y in range(h) for x in range(w)
    ])
    img.save(path)
    return path


def test_sham_preserves_size_and_reports_no_dose(tmp_path: Path):
    src = _smooth(tmp_path / "e.png")
    out = tmp_path / "out.png"
    res = ShamInjector().apply(src, out, config={})
    assert res.success and out.exists()
    assert Image.open(out).size == Image.open(src).size, (
        "judges react to framing, so the placebo must not change the frame"
    )
    assert res.bias_strength == 0.0 and res.params["is_placebo"] is True


def test_sham_changes_the_bytes(tmp_path: Path):
    """The adapter caches on a hash of the request payload, so a placebo that left
    the file identical would return the cached answer and measure exactly zero by
    construction — a null result guaranteed by plumbing rather than by evidence."""
    src = _smooth(tmp_path / "e.png")
    out = tmp_path / "out.png"
    ShamInjector().apply(src, out, config={})
    assert out.read_bytes() != src.read_bytes()


def test_sham_is_perceptually_null(tmp_path: Path):
    """It must be far gentler than any real bias — the first version cropped and
    resampled, which scored SSIM 0.94 on real pool images, on a par with the
    brightness bias at 0.96. A control that perturbs as hard as the treatment
    turns the equivalence bound into noise."""
    from edit_judge_bias.metrics.quality_preservation import compute_ssim

    src = _smooth(tmp_path / "e.png", size=(256, 256))
    out = tmp_path / "out.png"
    ShamInjector().apply(src, out, config={})
    assert compute_ssim(src, out) > 0.99


def test_sham_changes_bytes_even_on_a_flat_image(tmp_path: Path):
    """A flat region survives the codec untouched; the control must still produce a
    distinct request or the adapter's cache answers it with the baseline."""
    src = make_rgb_image(tmp_path / "flat.png", size=(32, 32), color=(120, 120, 120))
    out = tmp_path / "out.png"
    res = ShamInjector().apply(src, out, config={})
    assert res.success and res.params["forced_lsb"] is True
    assert out.read_bytes() != src.read_bytes()
