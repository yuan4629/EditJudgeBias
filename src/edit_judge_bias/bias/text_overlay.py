"""Instruction / keyword overlay bias.

Renders text drawn from the editing instruction onto a semi-transparent band at
the top or bottom edge — testing whether seeing the instruction (or its target
keyword) on the image nudges the judge toward "adherence". Two modes:

- ``keyword``     : a single salient content word from the instruction.
- ``instruction`` : the instruction truncated to `max_words` (default 12).

The exact rendered text is always recorded in params (`overlay_text`) so a later
OCR / area check (§6.1) can confirm it did not cover the edited subject.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from edit_judge_bias.bias.base import BiasInjector
from edit_judge_bias.bias.fonts import load_font

DEFAULT_MODE = "keyword"
DEFAULT_OPACITY = 0.5
DEFAULT_BAND = "top"
DEFAULT_FONT_RATIO = 0.05
DEFAULT_MAX_WORDS = 12

# Tokens stripped before picking a keyword: articles/prepositions + edit verbs.
_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "with", "and", "or", "this",
    "that", "it", "its", "image", "photo", "picture", "from", "into", "for",
    "please", "by", "as",
}
_EDIT_VERBS = {
    "remove", "add", "make", "change", "turn", "replace", "delete", "put", "set",
    "transform", "alter", "adjust", "modify", "convert", "give", "increase",
    "decrease", "fill",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def extract_keyword(instruction: str) -> str:
    """Pick a salient content word (heuristic, NLP-free): the last token that is
    neither a stopword nor an edit verb; falls back to the last word."""
    tokens = _WORD_RE.findall(instruction or "")
    if not tokens:
        return ""
    content = [t for t in tokens if t.lower() not in _STOPWORDS and t.lower() not in _EDIT_VERBS]
    return (content[-1] if content else tokens[-1])


def truncate_words(instruction: str, max_words: int) -> str:
    words = (instruction or "").split()
    return " ".join(words[:max_words])


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    """Greedy word-wrap so the band text never overflows the image width."""
    if not text:
        return [""]
    lines: List[str] = []
    cur = ""
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


class TextOverlayInjector(BiasInjector):
    bias_type = "text_overlay"

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        mode = str(config.get("mode", DEFAULT_MODE))
        opacity = float(config.get("opacity", DEFAULT_OPACITY))
        band = str(config.get("band", DEFAULT_BAND))
        font_ratio = float(config.get("font_ratio", DEFAULT_FONT_RATIO))
        max_words = int(config.get("max_words", DEFAULT_MAX_WORDS))

        instruction = sample.get("instruction") or config.get("text", "")
        if mode == "keyword":
            text = extract_keyword(instruction)
        else:
            text = truncate_words(instruction, max_words)
        text = text or "edit"

        w, h = image.size
        short = min(w, h)
        font = load_font(int(short * font_ratio), config.get("font_path"))
        alpha = max(0, min(255, int(round(255 * opacity))))
        pad = max(2, int(short * 0.01))

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        lines = _wrap(draw, text, font, w - 2 * pad)
        l, t, r, b = draw.textbbox((0, 0), "Ag", font=font)
        line_h = (b - t) + pad
        band_h = line_h * len(lines) + pad
        y0 = 0 if band == "top" else h - band_h

        # Dark translucent band, then white centered text — legible, edge-anchored.
        draw.rectangle([0, y0, w, y0 + band_h], fill=(0, 0, 0, alpha))
        ty = y0 + pad // 2
        text_alpha = min(255, alpha + 120)
        for line in lines:
            lw = draw.textlength(line, font=font)
            draw.text(((w - lw) / 2, ty), line, font=font, fill=(255, 255, 255, text_alpha))
            ty += line_h

        out = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        params = {
            "overlay_text": text,
            "mode": mode,
            "band": band,
            "opacity": opacity,
            "n_lines": len(lines),
            "bias_strength": opacity,
        }
        return out, params
