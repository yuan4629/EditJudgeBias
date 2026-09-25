"""Detail-caption bias.

Appends a clean white caption strip below the image with a professional-looking,
semantically-aligned template line — e.g. ``Edited: replace the sofa with a
leather armchair``. Unlike ``text_overlay`` (a raw keyword slapped over the
subject), this reads as a legitimate "edit description" panel; it tests whether a
tidy, on-topic explanation box earns unwarranted credit.

Prior-art: MM-JudgeBias "Detail-Description" (redundant aligned information). The
strip is *added below* the frame (like padding) so it never occludes the edit;
the original pixels are preserved exactly. No original image needed.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from edit_judge_bias.bias.base import BiasInjector
from edit_judge_bias.bias.fonts import load_font
from edit_judge_bias.bias.text_overlay import truncate_words

DEFAULT_TEMPLATE = "Edited: {instruction}"
DEFAULT_MAX_WORDS = 14
DEFAULT_FONT_RATIO = 0.035
DEFAULT_BG = (255, 255, 255)
DEFAULT_FG = (20, 20, 20)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    if not text:
        return [""]
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class DetailCaptionInjector(BiasInjector):
    bias_type = "detail_caption"

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        template = str(config.get("template", DEFAULT_TEMPLATE))
        max_words = int(config.get("max_words", DEFAULT_MAX_WORDS))
        font_ratio = float(config.get("font_ratio", DEFAULT_FONT_RATIO))
        bg = tuple(config.get("bg_color", DEFAULT_BG))
        fg = tuple(config.get("fg_color", DEFAULT_FG))

        instruction = sample.get("instruction") or config.get("text", "")
        summary = truncate_words(instruction, max_words) or "edit applied"
        caption = template.format(instruction=summary)

        w, h = image.size
        font = load_font(int(min(w, h) * font_ratio), config.get("font_path"))
        pad = max(4, int(min(w, h) * 0.015))

        measure = ImageDraw.Draw(image)
        lines = _wrap(measure, caption, font, w - 2 * pad)
        l, t, r, b = measure.textbbox((0, 0), "Ag", font=font)
        line_h = (b - t) + pad // 2
        strip_h = line_h * len(lines) + pad

        out = Image.new("RGB", (w, h + strip_h), bg)
        out.paste(image, (0, 0))  # original pixels preserved exactly
        draw = ImageDraw.Draw(out)
        ty = h + pad // 2
        for line in lines:
            draw.text((pad, ty), line, font=font, fill=fg)
            ty += line_h

        params = {
            "caption": caption,
            "n_lines": len(lines),
            "strip_h": strip_h,
            "original_size": [w, h],
            "output_size": list(out.size),
            "bias_strength": strip_h / float(h),
        }
        return out, params
