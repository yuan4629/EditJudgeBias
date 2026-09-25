"""Zoom-inset bias.

Crops the estimated edit region, magnifies it, frames it with a white border, and
pastes it into the frame corner that overlaps the edit region *least* — a
"detail call-out" of the sort a diligent reviewer might add. Tests whether extra
visual verbosity that showcases the edit inflates the judge's score, even though
no editing quality changed.

Prior-art spirit: MM-JudgeBias "Detail-Description" verbosity bias, made concrete
for images. Uses `edit_region` to localize, so ``needs_original = True``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from PIL import Image, ImageOps

from edit_judge_bias.bias.base import REFERENCE_SHORT_EDGE, BiasInjector, relative_px
from edit_judge_bias.bias.edit_region import estimate_edit_region

DEFAULT_ZOOM = 2.0
DEFAULT_MAX_FRAC = 0.4     # inset must fit in a corner: cap each side to 40% of the frame
# Border as a fraction of the short edge, so the inset frame stays equally
# prominent at 500px and 1024px (3px at the 512px reference).
DEFAULT_BORDER_RATIO = 3 / REFERENCE_SHORT_EDGE
DEFAULT_BORDER_COLOR = (255, 255, 255)
DEFAULT_MARGIN_FRAC = 0.02


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


class ZoomInsetInjector(BiasInjector):
    bias_type = "zoom_inset"
    needs_original = True

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        zoom = float(config.get("zoom", DEFAULT_ZOOM))
        max_frac = float(config.get("max_frac", DEFAULT_MAX_FRAC))
        border_color = tuple(config.get("border_color", DEFAULT_BORDER_COLOR))
        margin_frac = float(config.get("margin_frac", DEFAULT_MARGIN_FRAC))

        w, h = image.size
        border, border_params = relative_px(
            config,
            ratio_key="border_ratio",
            px_key="border_px",
            short_edge=min(w, h),
            default_ratio=DEFAULT_BORDER_RATIO,
        )
        region = estimate_edit_region(
            image,
            original=config.get("original_image_path_abs"),
            mask=config.get("mask_path_abs"),
            threshold=int(config.get("region_threshold", 15)),
        )
        crop = image.crop(region.bbox)
        cw, ch = crop.size

        # Magnify, then cap so the framed inset fits within a corner.
        tw, th = max(1, int(cw * zoom)), max(1, int(ch * zoom))
        cap_w, cap_h = int(w * max_frac), int(h * max_frac)
        scale = min(1.0, cap_w / tw, cap_h / th)
        tw, th = max(1, int(tw * scale)), max(1, int(th * scale))
        inset = crop.resize((tw, th), Image.LANCZOS)
        inset = ImageOps.expand(inset, border=border, fill=border_color)
        iw, ih = inset.size

        margin = max(1, int(min(w, h) * margin_frac))
        candidates = {
            "top-left": (margin, margin),
            "top-right": (w - iw - margin, margin),
            "bottom-left": (margin, h - ih - margin),
            "bottom-right": (w - iw - margin, h - ih - margin),
        }
        # Paste into the corner whose footprint overlaps the edit region the least.
        corner, (px, py) = min(
            candidates.items(),
            key=lambda kv: _iou((kv[1][0], kv[1][1], kv[1][0] + iw, kv[1][1] + ih), region.bbox),
        )
        px, py = max(0, px), max(0, py)

        out = image.copy()
        out.paste(inset, (px, py))

        params = {
            "zoom": zoom,
            "bbox": list(region.bbox),
            "region_method": region.method,
            "inset_size": [iw, ih],
            "corner": corner,
            **border_params,
            "bias_strength": zoom,
        }
        return out, params
