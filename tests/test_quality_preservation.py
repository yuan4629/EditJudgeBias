"""Tests for automatic quality-preservation metrics (Milestone 6)."""

from __future__ import annotations

from pathlib import Path

from conftest import make_rgb_image
from PIL import Image

from edit_judge_bias.metrics.quality_preservation import compute_auto_metrics, compute_ssim


def test_ssim_identical_is_one(tmp_path: Path):
    p = make_rgb_image(tmp_path / "a.png", size=(64, 64), color=(100, 120, 140))
    assert compute_ssim(p, p) == 1.0


def test_ssim_handles_padding_size_mismatch(tmp_path: Path):
    base = make_rgb_image(tmp_path / "base.png", size=(64, 64), color=(120, 120, 120))
    # A padded version: larger canvas with the same content centered.
    padded = Image.new("RGB", (80, 80), (0, 0, 0))
    padded.paste(Image.open(base), (8, 8))
    padded_path = tmp_path / "padded.png"
    padded.save(padded_path)
    # Center content identical -> SSIM ~1 after the runner's center-crop alignment.
    assert compute_ssim(base, padded_path) > 0.99


def _noise_image(path: Path, size=(64, 64)) -> Path:
    """Structured content, so a misaligned crop actually costs SSIM.

    A flat colour patch scores ~1.0 at any offset, which is precisely why the
    detail_caption misalignment went unnoticed against the old flat-grey fixture.
    """
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256)
    img.save(path)
    return path


def test_ssim_handles_a_caption_strip_appended_below_the_frame(tmp_path: Path):
    """`detail_caption` extends the canvas downward only, leaving pixels untouched.

    A center-crop alignment shifts such an image vertically by half the strip and
    scored a spurious 0.46 (min 0.04) on real data; the anchor search must recover
    the exact 1.0.
    """
    base = _noise_image(tmp_path / "base.png")
    captioned = Image.new("RGB", (64, 90), (255, 255, 255))
    captioned.paste(Image.open(base), (0, 0))  # content at the TOP, strip below
    out = tmp_path / "captioned.png"
    captioned.save(out)
    assert compute_ssim(base, out) == 1.0


def test_ssim_alignment_is_symmetric_in_argument_order(tmp_path: Path):
    base = _noise_image(tmp_path / "base.png")
    captioned = Image.new("RGB", (64, 90), (255, 255, 255))
    captioned.paste(Image.open(base), (0, 0))
    out = tmp_path / "captioned.png"
    captioned.save(out)
    assert compute_ssim(out, base) == compute_ssim(base, out) == 1.0


def test_ssim_alignment_cannot_rescue_damaged_content(tmp_path: Path):
    """Maximising over anchors must not manufacture a passing score.

    The guarantee that makes the anchor search sound: content that really was
    destroyed scores low at *every* offset.
    """
    base = _noise_image(tmp_path / "base.png")
    wrecked = Image.new("RGB", (64, 90), (255, 255, 255))
    wrecked.paste(Image.open(base).rotate(90), (0, 0))
    out = tmp_path / "wrecked.png"
    wrecked.save(out)
    assert compute_ssim(base, out) < 0.5


def test_ssim_handles_one_axis_growing_and_the_other_shrinking(tmp_path: Path):
    """Neither image contains the other — must still return a number, not raise."""
    a = _noise_image(tmp_path / "a.png", size=(64, 48))
    b = _noise_image(tmp_path / "b.png", size=(48, 64))
    score = compute_ssim(a, b)
    assert isinstance(score, float) and -1.0 <= score <= 1.0


def test_auto_metrics_ssim_always_present(tmp_path: Path):
    a = make_rgb_image(tmp_path / "a.png", size=(48, 48))
    b = make_rgb_image(tmp_path / "b.png", size=(48, 48), color=(10, 10, 10))
    m = compute_auto_metrics(a, b)  # clip/lpips off by default
    assert "ssim" in m and isinstance(m["ssim"], float)
    assert m["clip_image_similarity"] is None and m["lpips"] is None
