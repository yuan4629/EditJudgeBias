"""Aesthetic 'beauty filter' bias.

A deterministic post-processing chain of the kind photo apps apply by default —
it makes the image look "nicer" (punchier, warmer, crisper) without changing
what was edited. Tests whether a judge rewards surface prettiness that is
unrelated to instruction adherence or editing quality.

Fixed chain (all parameters config-overridable):

1. Contrast   ×1.1  (ImageEnhance.Contrast)
2. Warm LUT   R +8 / B -8, clamped 0..255 (a point op — a mild orange cast)
3. UnsharpMask radius 2, percent 60, threshold 3

The output keeps the input size exactly, so composition is untouched; only tone
and local micro-contrast move. `bias_strength` reports the contrast factor as a
single representative magnitude (see §9.3 for the ablation grid).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from PIL import Image, ImageEnhance, ImageFilter

from edit_judge_bias.bias.base import REFERENCE_SHORT_EDGE, BiasInjector

DEFAULT_CONTRAST = 1.1
DEFAULT_WARM_SHIFT = 8      # +delta to R, -delta to B
# UnsharpMask's radius is in pixels, so a fixed 2.0 sharpens a 512px image visibly
# and a 1024px image only locally -- the perceived "enhancement" would then scale
# with resolution and confound the measured effect. Expressed as a fraction of the
# short edge instead (2.0px at the 512px reference). Kept float, unlike the integer
# `base.relative_px` helper, because PIL accepts a fractional radius and rounding
# it would coarsen small images.
DEFAULT_UNSHARP_RADIUS_RATIO = 2.0 / REFERENCE_SHORT_EDGE
DEFAULT_UNSHARP_PERCENT = 60
DEFAULT_UNSHARP_THRESHOLD = 3


def _warm_lut(delta: int) -> Image.Image.point:
    """Build a per-channel LUT that shifts R up and B down by `delta` (clamped)."""
    up = [min(255, i + delta) for i in range(256)]
    down = [max(0, i - delta) for i in range(256)]
    identity = list(range(256))
    return up + identity + down  # concatenated R,G,B tables for Image.point


class AestheticFilterInjector(BiasInjector):
    bias_type = "aesthetic_filter"

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        contrast = float(config.get("contrast", DEFAULT_CONTRAST))
        warm = int(config.get("warm_shift", DEFAULT_WARM_SHIFT))
        percent = int(config.get("unsharp_percent", DEFAULT_UNSHARP_PERCENT))
        threshold = int(config.get("unsharp_threshold", DEFAULT_UNSHARP_THRESHOLD))

        short = min(image.size)
        if config.get("unsharp_radius") is not None:
            radius = float(config["unsharp_radius"])  # explicit absolute override
            radius_ratio = None
        else:
            radius_ratio = float(
                config.get("unsharp_radius_ratio", DEFAULT_UNSHARP_RADIUS_RATIO)
            )
            radius = max(0.5, short * radius_ratio)

        out = ImageEnhance.Contrast(image).enhance(contrast)
        if warm:
            out = out.point(_warm_lut(warm))  # RGB point op over the 3-channel LUT
        out = out.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )

        params = {
            "contrast": contrast,
            "warm_shift": warm,
            "unsharp": [radius, percent, threshold],
            "unsharp_radius_ratio": radius_ratio,
            "bias_strength": contrast,
        }
        return out, params
