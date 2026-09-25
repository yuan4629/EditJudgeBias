"""Authenticity / authority overlay bias.

Stamps a small semi-transparent caption ("Reference Image", "Verified Edit", ...)
in a corner — testing whether an authority cue inflates the judge's score. Placed
at the edge so it does not occlude the edited subject. Defaults: corner =
bottom-right, font = 4% of the short edge, opacity = 0.45.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw

from edit_judge_bias.bias.base import BiasInjector
from edit_judge_bias.bias.fonts import load_font

DEFAULT_TEXT = "Reference Image"
DEFAULT_OPACITY = 0.45
DEFAULT_FONT_RATIO = 0.04
DEFAULT_POSITION = "bottom-right"
DEFAULT_MARGIN_RATIO = 0.03


def _anchor_xy(position: str, w: int, h: int, tw: int, th: int, margin: int) -> Tuple[int, int]:
    """Top-left pixel for a text box of size (tw, th) at a named corner."""
    right = w - tw - margin
    bottom = h - th - margin
    table = {
        "bottom-right": (right, bottom),
        "bottom-left": (margin, bottom),
        "top-right": (right, margin),
        "top-left": (margin, margin),
    }
    return table.get(position, table["bottom-right"])


class WatermarkInjector(BiasInjector):
    bias_type = "watermark"

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        text = str(config.get("text", DEFAULT_TEXT))
        opacity = float(config.get("opacity", DEFAULT_OPACITY))
        font_ratio = float(config.get("font_ratio", DEFAULT_FONT_RATIO))
        position = str(config.get("position", DEFAULT_POSITION))
        margin_ratio = float(config.get("margin_ratio", DEFAULT_MARGIN_RATIO))

        w, h = image.size
        short = min(w, h)
        font = load_font(int(short * font_ratio), config.get("font_path"))
        margin = max(1, int(short * margin_ratio))
        alpha = max(0, min(255, int(round(255 * opacity))))

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        x, y = _anchor_xy(position, w, h, tw, th, margin)
        # Subtle dark shadow for legibility on light backgrounds, then the caption.
        draw.text((x - l + 1, y - t + 1), text, font=font, fill=(0, 0, 0, alpha))
        draw.text((x - l, y - t), text, font=font, fill=(255, 255, 255, alpha))

        out = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        params = {
            "overlay_text": text,
            "opacity": opacity,
            "position": position,
            "font_size": font.size if hasattr(font, "size") else None,
            "bias_strength": opacity,
        }
        return out, params
