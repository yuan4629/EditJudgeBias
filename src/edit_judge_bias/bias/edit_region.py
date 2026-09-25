"""Edit-region estimation — a shared tool for the C-class overlay biases (§5.4, C0).

C-class injectors (region_annotation, zoom_inset, distraction) need to know *where*
the edit happened so they can annotate it, zoom into it, or deliberately avoid it.
This module estimates that region, in the edited image's pixel coordinates, by the
best available evidence:

1. **mask**     — if the dataset ships an edit mask, use its bounding box (exact).
                  ⚠️ Reading it correctly means NOT flattening it to RGB; see
                  `MASK_POLARITY_DEFAULT` for the measurement and the blast radius.
2. **diff**     — else |edited - original|: per-pixel max-channel difference,
                  threshold, morphological open+close (de-speckle), then the bbox of
                  the largest connected component.
3. **fallback** — if there is no original (or the diff is empty), a centered box
                  covering `fallback_frac` of the frame, so callers never hard-fail.

The heavy path uses numpy + scipy.ndimage when importable; it degrades to a
PIL-only diff+morphology+bbox when they are not, so the bias layer keeps working
in a minimal (Pillow-only) environment. `estimate_edit_region` never raises for
ordinary inputs — a bad path or size mismatch falls through to the centered box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

Bbox = Tuple[int, int, int, int]  # (left, top, right, bottom); right/bottom exclusive

DEFAULT_THRESHOLD = 15       # per-channel diff (0..255) above which a pixel "changed"
DEFAULT_FALLBACK_FRAC = 0.6  # centered box side as a fraction of each dimension
DEFAULT_MORPH_FRAC = 0.01    # morphology kernel as a fraction of the short edge

_MASK_POLARITIES = frozenset({"auto", "alpha_low", "alpha_high", "luma_high"})

#: How to read a dataset-shipped edit mask.
#:
#: ★ MEASURED 2026-07-30, and it fixes a real defect. MagicBrush -- the only source in the
#: pool that ships masks (349 files, `has_mask` False for the other 13,520 samples) --
#: stores them as **RGBA where alpha <= 127 marks the edited region**. The old code path
#: did `Image.open(...).convert("RGB")`, which drops alpha; the surviving RGB is bright
#: almost everywhere, so `point(p > 0).getbbox()` returned the WHOLE FRAME.
#:
#: IoU against the `|edited - original|` diff region over 30 judged samples that ship a
#: mask:
#:     alpha <= 127 ("alpha_low")   mean 0.573  median 0.624   <-- the real region
#:     alpha  > 127 ("alpha_high")  mean 0.007  median 0.001
#:     luma   >   0 ("luma_high")   mean 0.008  median 0.001   <-- what the old code did
#: So the old reading was not merely imprecise, it was *uncorrelated* with the edit.
#:
#: Blast radius in the ALREADY-COLLECTED main grid: 91 of the 1,196 judged samples ship a
#: mask, and the three region-dependent C-class injectors (`region_annotation`,
#: `zoom_inset`, `distraction`) all consumed it, so 273 judged biased images (7.6% of each
#: of those cues' rows) were built against a full-frame region estimate. One of them is
#: `zoom_inset`, the contested boundary cue. That is dose heterogeneity in
#: published numbers: it must be disclosed, and `"luma_high"` exists so the published
#: behaviour stays exactly reproducible for the ablation.
MASK_POLARITY_DEFAULT = "auto"


@dataclass
class EditRegion:
    """Estimated edit location in edited-image pixels."""

    bbox: Bbox
    method: str          # "mask" | "diff" | "fallback"
    area_frac: float     # bbox area / image area (0..1)
    #: Which channel/threshold actually read the shipped mask ("alpha_low" |
    #: "alpha_high" | "luma_high"), or None when `method != "mask"`. Recorded so a
    #: manifest row can be audited after the fact -- see MASK_POLARITY_DEFAULT.
    mask_polarity: Optional[str] = None

    @property
    def center(self) -> Tuple[int, int]:
        l, t, r, b = self.bbox
        return (l + r) // 2, (t + b) // 2


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _clamp_bbox(bbox: Bbox, w: int, h: int) -> Bbox:
    l, t, r, b = bbox
    l, t = max(0, min(l, w - 1)), max(0, min(t, h - 1))
    r, b = max(l + 1, min(r, w)), max(t + 1, min(b, h))
    return (l, t, r, b)


def _area_frac(bbox: Bbox, w: int, h: int) -> float:
    l, t, r, b = bbox
    return ((r - l) * (b - t)) / float(max(1, w * h))


def _fallback(w: int, h: int, frac: float) -> EditRegion:
    bw, bh = int(round(w * frac)), int(round(h * frac))
    l, t = (w - bw) // 2, (h - bh) // 2
    bbox = _clamp_bbox((l, t, l + bw, t + bh), w, h)
    return EditRegion(bbox=bbox, method="fallback", area_frac=_area_frac(bbox, w, h))


def _load(img_or_path) -> Optional[Image.Image]:
    if img_or_path is None:
        return None
    if isinstance(img_or_path, Image.Image):
        return img_or_path.convert("RGB")
    try:
        with Image.open(img_or_path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _load_mask(img_or_path) -> Optional[Image.Image]:
    """Load a shipped edit mask WITHOUT flattening it.

    Deliberately not `_load`: that converts to RGB, which **drops the alpha channel**.
    MagicBrush -- the only source in the pool that ships masks (349 files) -- encodes the
    edited region in alpha, so flattening it silently destroys the mask. See
    MASK_POLARITY_DEFAULT for the measurement.
    """
    if img_or_path is None:
        return None
    if isinstance(img_or_path, Image.Image):
        return img_or_path.copy()
    try:
        with Image.open(img_or_path) as im:
            return im.copy()
    except Exception:
        return None


def _alpha_of(mk: Image.Image):
    """The mask's alpha band, or None when it has none / it is degenerate.

    Degenerate means every pixel sits on one side of the midpoint, i.e. alpha carries no
    region information (a fully opaque RGBA image is the common case). Returning None
    there is what makes `mask_polarity="auto"` fall back to the luma reading instead of
    selecting the whole frame or nothing.
    """
    if mk.mode not in ("RGBA", "LA", "PA"):
        return None
    band = mk.getchannel("A")
    lo, hi = band.getextrema()
    if lo > 127 or hi <= 127:
        return None
    return band


def _mask_bbox(mk: Image.Image, polarity: str) -> Tuple[Optional[Bbox], str]:
    """(bbox of the marked region, polarity actually used)."""
    alpha = _alpha_of(mk) if polarity in ("auto", "alpha_low", "alpha_high") else None
    if polarity in ("alpha_low", "alpha_high") and alpha is None:
        # Explicitly requested a channel this mask does not carry: say so rather than
        # quietly reading a different one.
        return None, polarity
    if alpha is not None:
        used = "alpha_high" if polarity == "alpha_high" else "alpha_low"
        keep = 255 if used == "alpha_high" else 0
        binary = alpha.point(lambda p, k=keep: 255 if ((p > 127) == (k == 255)) else 0)
        return binary.getbbox(), used
    binary = mk.convert("RGB").convert("L").point(lambda p: 255 if p > 0 else 0)
    return binary.getbbox(), "luma_high"


def _region_numpy(edited: Image.Image, original: Image.Image, threshold: int) -> Optional[Bbox]:
    """Largest-connected-component bbox via numpy + scipy.ndimage; None if unavailable/empty."""
    try:
        import numpy as np
        from scipy import ndimage
    except Exception:
        return None
    w, h = edited.size
    e = np.asarray(edited, dtype=np.int16)
    o = np.asarray(original, dtype=np.int16)
    mask = np.abs(e - o).max(axis=2) > threshold
    if not mask.any():
        return None
    k = max(1, int(round(min(w, h) * DEFAULT_MORPH_FRAC)))
    struct = np.ones((k, k), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=struct)
    mask = ndimage.binary_closing(mask, structure=struct)
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background
    largest = int(counts.argmax())
    ys, xs = np.where(labels == largest)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _region_pil(edited: Image.Image, original: Image.Image, threshold: int) -> Optional[Bbox]:
    """PIL-only fallback: diff -> threshold -> open/close -> bbox of all changed pixels."""
    from PIL import ImageChops, ImageFilter

    diff = ImageChops.difference(edited, original).convert("L")
    mask = diff.point(lambda p: 255 if p > threshold else 0)
    # Morphological open (MinFilter->MaxFilter) then close (MaxFilter->MinFilter).
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return mask.getbbox()


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def estimate_edit_region(
    edited,
    original=None,
    *,
    mask=None,
    threshold: int = DEFAULT_THRESHOLD,
    fallback_frac: float = DEFAULT_FALLBACK_FRAC,
    mask_polarity: str = MASK_POLARITY_DEFAULT,
) -> EditRegion:
    """Estimate where the edit is, in `edited`'s pixel coordinates.

    `edited` / `original` / `mask` may each be a PIL image, a path, or None.
    Always returns an EditRegion (falls back to a centered box).

    `mask_polarity` selects how a shipped mask is read -- see MASK_POLARITY_DEFAULT.
    Pass `"luma_high"` to reproduce the pre-2026-07-30 behaviour exactly.
    """
    if mask_polarity not in _MASK_POLARITIES:
        raise ValueError(
            f"mask_polarity must be one of {sorted(_MASK_POLARITIES)}; got {mask_polarity!r}"
        )
    ed = _load(edited)
    if ed is None:
        # Nothing to work with; return a nominal box so callers stay non-fatal.
        return EditRegion(bbox=(0, 0, 1, 1), method="fallback", area_frac=1.0)
    w, h = ed.size

    mk = _load_mask(mask)
    if mk is not None:
        if mk.size != ed.size:
            mk = mk.resize(ed.size, Image.NEAREST)
        bbox, polarity_used = _mask_bbox(mk, mask_polarity)
        if bbox is not None:
            bbox = _clamp_bbox(bbox, w, h)
            return EditRegion(
                bbox=bbox,
                method="mask",
                area_frac=_area_frac(bbox, w, h),
                mask_polarity=polarity_used,
            )

    orig = _load(original)
    if orig is not None:
        if orig.size != ed.size:
            orig = orig.resize(ed.size, Image.BILINEAR)
        bbox = _region_numpy(ed, orig, threshold) or _region_pil(ed, orig, threshold)
        if bbox is not None:
            bbox = _clamp_bbox(bbox, w, h)
            return EditRegion(bbox=bbox, method="diff", area_frac=_area_frac(bbox, w, h))

    return _fallback(w, h, fallback_frac)
