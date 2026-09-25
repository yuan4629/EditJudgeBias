"""Region-annotation bias.

Draws an attention cue *on the edited region itself* — either a red bounding box
around it, or an arrow from a frame corner pointing at it. Tests whether visibly
"marking up" the edit (as a reviewer might) nudges a judge toward crediting it.

Two variants (config ``mode``):
- ``box``   — a 3px red rectangle around the estimated edit region.
- ``arrow`` — a red arrow from the nearest corner to the region center. Pointing
              the arrow *at the edit location* is our novelty vs. generic bbox
              overlays: it directly exploits edit-task localization.

The region is estimated by `edit_region.estimate_edit_region`, so this injector
sets ``needs_original = True`` and the batch runner supplies the original image.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw

from edit_judge_bias.bias.base import REFERENCE_SHORT_EDGE, BiasInjector, relative_px
from edit_judge_bias.bias.edit_region import estimate_edit_region

DEFAULT_MODE = "box"
DEFAULT_COLOR = (255, 0, 0)
# Stroke width as a fraction of the short edge, so a box on a 1024px image is as
# visually prominent as on a 512px one (3px at the 512px reference).
DEFAULT_WIDTH_RATIO = 3 / REFERENCE_SHORT_EDGE


def _region(image: Image.Image, config: Dict[str, Any]):
    return estimate_edit_region(
        image,
        original=config.get("original_image_path_abs"),
        mask=config.get("mask_path_abs"),
        threshold=int(config.get("region_threshold", 15)),
    )


def _nearest_corner(cx: int, cy: int, w: int, h: int) -> Tuple[int, int]:
    """The image corner farthest from the region center (so the arrow spans the frame)."""
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    return max(corners, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)


class RegionAnnotationInjector(BiasInjector):
    bias_type = "region_annotation"
    needs_original = True

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        mode = str(config.get("mode", DEFAULT_MODE))
        color = tuple(config.get("color", DEFAULT_COLOR))

        region = _region(image, config)
        w, h = image.size
        width, width_params = relative_px(
            config,
            ratio_key="width_ratio",
            px_key="width_px",
            short_edge=min(w, h),
            default_ratio=DEFAULT_WIDTH_RATIO,
        )
        out = image.copy()
        draw = ImageDraw.Draw(out)

        if mode == "arrow":
            cx, cy = region.center
            sx, sy = _nearest_corner(cx, cy, w, h)
            # Stop the arrow at the region boundary, not dead center, so it points *at* it.
            draw.line([(sx, sy), (cx, cy)], fill=color, width=width)
            self._arrowhead(draw, (sx, sy), (cx, cy), color, width)
        else:
            draw.rectangle(list(region.bbox), outline=color, width=width)

        params = {
            "mode": mode,
            "bbox": list(region.bbox),
            "region_method": region.method,
            "area_frac": region.area_frac,
            **width_params,
            "bias_strength": region.area_frac,
        }
        return out, params

    @staticmethod
    def _arrowhead(draw, start, end, color, width) -> None:
        sx, sy = start
        ex, ey = end
        ang = math.atan2(ey - sy, ex - sx)
        size = max(8, width * 5)
        for da in (math.radians(150), math.radians(-150)):
            hx = ex + size * math.cos(ang + da)
            hy = ey + size * math.sin(ang + da)
            draw.line([(ex, ey), (hx, hy)], fill=color, width=width)
