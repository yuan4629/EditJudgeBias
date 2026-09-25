"""Saturation bias: scale color saturation.

Spurious visual cue. MVP uses factor 1.2 (ImageEnhance.Color; 1.0 = unchanged,
0.0 = grayscale).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from PIL import Image, ImageEnhance

from edit_judge_bias.bias.base import BiasInjector

DEFAULT_FACTOR = 1.2


class SaturationInjector(BiasInjector):
    bias_type = "saturation"

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        factor = float(config.get("factor", DEFAULT_FACTOR))
        out = ImageEnhance.Color(image).enhance(factor)
        return out, {"factor": factor, "bias_strength": factor}
