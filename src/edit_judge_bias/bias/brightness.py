"""Brightness bias: scale image brightness.

Spurious visual cue — does the judge reward a brighter edit it shouldn't? MVP uses
factor 1.2 (subtle; keeps editing quality intact).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from PIL import Image, ImageEnhance

from edit_judge_bias.bias.base import BiasInjector

DEFAULT_FACTOR = 1.2


class BrightnessInjector(BiasInjector):
    bias_type = "brightness"

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        factor = float(config.get("factor", DEFAULT_FACTOR))
        out = ImageEnhance.Brightness(image).enhance(factor)
        return out, {"factor": factor, "bias_strength": factor}
