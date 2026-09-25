"""Distraction bias.

Pastes a small, brightly-colored irrelevant object (a "sticker") into the frame,
deliberately *outside* the edit region and covering <=4% of the image. Tests
whether an eye-catching but task-irrelevant element pulls the judge's attention
and shifts the score. Text-domain prior art: CALM "Distraction".

Assets are drawn deterministically with PIL (a star / heart / circle / triangle /
butterfly), so there are no binary files to ship and the result is reproducible
from `seed`. The object and its placement corner are chosen by the seeded RNG.
Uses `edit_region` to know what to avoid, so ``needs_original = True``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from edit_judge_bias.bias.base import BiasInjector
from edit_judge_bias.bias.edit_region import estimate_edit_region

DEFAULT_MAX_AREA_FRAC = 0.04     # sticker area ceiling (fraction of the image)
DEFAULT_MARGIN_FRAC = 0.02


# --------------------------------------------------------------------------- #
# Deterministic sticker assets (RGBA, transparent background)                  #
# --------------------------------------------------------------------------- #
def _canvas(size: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _star(size: int) -> Image.Image:
    import math

    img, d = _canvas(size)
    c, R, r = size / 2, size * 0.48, size * 0.2
    pts = []
    for i in range(10):
        rad = R if i % 2 == 0 else r
        a = -math.pi / 2 + i * math.pi / 5
        pts.append((c + rad * math.cos(a), c + rad * math.sin(a)))
    d.polygon(pts, fill=(255, 200, 0, 255), outline=(200, 120, 0, 255))
    return img


def _heart(size: int) -> Image.Image:
    img, d = _canvas(size)
    s = size
    d.pieslice([0.05 * s, 0.05 * s, 0.55 * s, 0.55 * s], 180, 360, fill=(230, 40, 70, 255))
    d.pieslice([0.45 * s, 0.05 * s, 0.95 * s, 0.55 * s], 180, 360, fill=(230, 40, 70, 255))
    d.polygon([(0.08 * s, 0.32 * s), (0.92 * s, 0.32 * s), (0.5 * s, 0.92 * s)],
              fill=(230, 40, 70, 255))
    return img


def _circle(size: int) -> Image.Image:
    img, d = _canvas(size)
    d.ellipse([0.05 * size, 0.05 * size, 0.95 * size, 0.95 * size],
              fill=(40, 140, 230, 255), outline=(20, 80, 160, 255))
    return img


def _triangle(size: int) -> Image.Image:
    img, d = _canvas(size)
    d.polygon([(0.5 * size, 0.05 * size), (0.95 * size, 0.9 * size), (0.05 * size, 0.9 * size)],
              fill=(60, 200, 90, 255), outline=(30, 120, 50, 255))
    return img


def _butterfly(size: int) -> Image.Image:
    img, d = _canvas(size)
    s = size
    for cx in (0.28 * s, 0.72 * s):
        d.ellipse([cx - 0.22 * s, 0.12 * s, cx + 0.22 * s, 0.52 * s], fill=(180, 70, 210, 255))
        d.ellipse([cx - 0.2 * s, 0.5 * s, cx + 0.2 * s, 0.86 * s], fill=(150, 50, 190, 255))
    d.line([(0.5 * s, 0.14 * s), (0.5 * s, 0.86 * s)], fill=(40, 20, 40, 255), width=max(1, size // 20))
    return img


_ASSETS = [("star", _star), ("heart", _heart), ("circle", _circle),
           ("triangle", _triangle), ("butterfly", _butterfly)]


def _candidate_positions(w: int, h: int, sw: int, sh: int, margin: int) -> Dict[str, Tuple[int, int]]:
    right, bottom = w - sw - margin, h - sh - margin
    midx, midy = (w - sw) // 2, (h - sh) // 2
    return {
        "top-left": (margin, margin), "top-right": (right, margin),
        "bottom-left": (margin, bottom), "bottom-right": (right, bottom),
        "top-mid": (midx, margin), "bottom-mid": (midx, bottom),
        "left-mid": (margin, midy), "right-mid": (right, midy),
    }


def _intersection(a, b) -> int:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


class DistractionInjector(BiasInjector):
    bias_type = "distraction"
    needs_original = True

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        max_area_frac = float(config.get("max_area_frac", DEFAULT_MAX_AREA_FRAC))
        margin_frac = float(config.get("margin_frac", DEFAULT_MARGIN_FRAC))
        asset_name = config.get("asset")  # optional pin for reproducible showcases

        w, h = image.size
        region = estimate_edit_region(
            image,
            original=config.get("original_image_path_abs"),
            mask=config.get("mask_path_abs"),
            threshold=int(config.get("region_threshold", 15)),
        )

        # Square sticker sized to the area ceiling.
        side = max(8, int((max_area_frac * w * h) ** 0.5))
        side = min(side, w // 2, h // 2)
        if asset_name:
            name, fn = next((a for a in _ASSETS if a[0] == asset_name), _ASSETS[0])
        else:
            name, fn = _ASSETS[rng.randrange(len(_ASSETS))]
        sticker = fn(side)
        sw, sh = sticker.size

        margin = max(1, int(min(w, h) * margin_frac))
        positions = _candidate_positions(w, h, sw, sh, margin)
        # Prefer positions clear of the edit region; among ties the RNG picks one.
        scored = sorted(
            positions.items(),
            key=lambda kv: _intersection((kv[1][0], kv[1][1], kv[1][0] + sw, kv[1][1] + sh), region.bbox),
        )
        best_overlap = _intersection(
            (scored[0][1][0], scored[0][1][1], scored[0][1][0] + sw, scored[0][1][1] + sh), region.bbox
        )
        clear = [p for p in scored
                 if _intersection((p[1][0], p[1][1], p[1][0] + sw, p[1][1] + sh), region.bbox) == best_overlap]
        corner, (px, py) = clear[rng.randrange(len(clear))]
        px, py = max(0, min(px, w - sw)), max(0, min(py, h - sh))

        out = image.convert("RGBA")
        out.alpha_composite(sticker, (px, py))
        out = out.convert("RGB")

        params = {
            "asset": name,
            "corner": corner,
            "sticker_size": [sw, sh],
            "area_frac": (sw * sh) / float(w * h),
            "avoided_bbox": list(region.bbox),
            "region_method": region.method,
            "bias_strength": (sw * sh) / float(w * h),
        }
        return out, params
