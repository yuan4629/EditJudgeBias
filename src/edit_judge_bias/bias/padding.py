"""Black padding / size-framing bias.

Adds a uniform border around the image, changing composition and visual focus
while leaving the center content untouched. MVP uses a 10% black border. Output
dimensions change; the ratio and both sizes are recorded.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from PIL import Image

from edit_judge_bias.bias.base import BiasInjector

DEFAULT_RATIO = 0.10
DEFAULT_COLOR = "black"


class PaddingInjector(BiasInjector):
    bias_type = "padding"

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        ratio = float(config.get("ratio", DEFAULT_RATIO))
        color = config.get("color", DEFAULT_COLOR)

        w, h = image.size
        bx = int(round(w * ratio))
        by = int(round(h * ratio))
        out = Image.new("RGB", (w + 2 * bx, h + 2 * by), color)
        out.paste(image, (bx, by))  # center content preserved exactly

        params = {
            "padding_ratio": ratio,
            "color": color,
            "border_px": [bx, by],
            "original_size": [w, h],
            "padded_size": list(out.size),
            "bias_strength": ratio,
        }
        return out, params
