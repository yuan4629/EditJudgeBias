"""Robust TrueType font loading for text-based bias injectors.

Overlays size text relative to the image, which needs a
scalable font. We try a configured path, then common system fonts, and finally
Pillow's bundled default — so injectors never hard-fail for lack of a font.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import ImageFont

# Searched in order; first that loads wins. Bare names resolve via the OS font dirs.
_FONT_CANDIDATES = (
    "arial.ttf",
    "DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def load_font(size: int, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """Return a TrueType font at `size`px, falling back gracefully."""
    size = max(8, int(size))
    candidates = ([font_path] if font_path else []) + list(_FONT_CANDIDATES)
    for cand in candidates:
        if cand is None:
            continue
        try:
            return ImageFont.truetype(cand, size=size)
        except OSError:
            continue
    # Pillow >= 10 supports a sized default (a scalable DejaVu); older returns bitmap.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()
