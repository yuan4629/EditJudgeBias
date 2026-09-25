"""Shared chart chrome for the paper figure set.

One palette, one set of mark specs, one uncertainty helper — so the figures read as
one system and so nothing about colour has to be re-argued per figure.

**Palette provenance.** The two *series* hues are categorical slots 1 and 2 of the
validated default palette; they were re-checked with the dataviz validator on the
light chart surface under ``--pairs all`` (the strict setting, because these are
scatter/dot forms where every pair can appear side by side)::

    node validate_palette.js "#2a78d6,#eb6834" --mode light --pairs all
      PASS lightness band · PASS chroma floor · PASS CVD separation (worst DE 24.7)
      PASS normal-vision floor (33.6) · PASS contrast vs surface  -> ALL CHECKS PASS

``MUTED`` is deliberately NOT a third series slot — it fails the chroma floor by
design, because it is the *de-emphasis* ink of the emphasis form (one entity in the
accent hue, the rest grey) and the axis/label ink. Any figure that greys a mark also
labels it, so identity is never carried by hue alone.

``CRITICAL`` is the reserved status colour. It appears exactly once in the set (the
one cell in the whole study that fails the 85% preservation gate) and always with a
printed label, never as colour alone.

**Greyscale.** Every encoding in this set is redundant with a non-colour channel:
significance is marker FILL plus line weight, identity is marker SHAPE plus a printed
tick label. Printing the figures in greyscale loses nothing load-bearing.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# --- series (validated categorical slots 1-2) ------------------------------- #
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# --- ink / chrome (never a series) ----------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BAND = "#f4f3ef"  # alternating group shading, one step off the surface

# --- status (reserved; icon + label always) -------------------------------- #
CRITICAL = "#d03b3b"

# The five judges, in one fixed order used by every figure, so a shape or a row
# position means the same thing everywhere (colour follows the entity, not the rank).
JUDGE_ORDER = [
    "gpt-5.5",
    "gemini-3.5-flash",
    "kimi-k2.5",
    "qwen3.5-plus",
    "gpt-4o-viescore",
]

#: Marker shape per judge — the greyscale-safe identity channel.
JUDGE_MARKER = {
    "gpt-5.5": "o",
    "gemini-3.5-flash": "s",
    "kimi-k2.5": "^",
    "qwen3.5-plus": "D",
    "gpt-4o-viescore": "v",
}

#: Short labels; the full model ids are printed once in each figure's footnote.
JUDGE_SHORT = {
    "gpt-5.5": "gpt-5.5",
    "gemini-3.5-flash": "gemini-3.5",
    "kimi-k2.5": "kimi-k2.5",
    "qwen3.5-plus": "qwen3.5",
    "gpt-4o-viescore": "viescore",
}

ROSTER_NOTE = ("judges: gpt-5.5 · gemini-3.5-flash · kimi-k2.5 · qwen3.5-plus · "
               "gpt-4o-viescore (VIEScore rubric on gpt-4o)")


def judge_sort_key(model: str) -> Tuple[int, str]:
    """Fixed roster order first, then anything unrecognised alphabetically."""
    try:
        return (JUDGE_ORDER.index(model), "")
    except ValueError:
        return (len(JUDGE_ORDER), model)


def marker_for(model: str) -> str:
    return JUDGE_MARKER.get(model, "o")


def short(model: str) -> str:
    return JUDGE_SHORT.get(model, model)


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
def wilson(successes: float, n: float, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval for a proportion.

    Used wherever a figure plots a *rate* whose table carries only the rate and its
    denominator (RR, CR, coverage, accuracy-on-decided, validator pass rate). The
    normal-approximation interval is wrong at exactly the values this study lives at
    — a pass rate of 1.000 on n=110 would get a zero-width bar, which would read as
    certainty. Wilson gives it [0.966, 1.0], which is the honest picture.
    """
    n = float(n)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(successes) / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_from_rate(rate: float, n: float) -> Tuple[float, float]:
    """Wilson interval when the table stored the rate rather than the count."""
    return wilson(round(float(rate) * float(n)), n)


# --------------------------------------------------------------------------- #
# chrome
# --------------------------------------------------------------------------- #
def f(row: dict, key: str) -> Optional[float]:
    """Float or None — metric CSVs write an empty cell where a value is undefined."""
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def truthy(row: dict, key: str) -> Optional[bool]:
    """Tri-state read of a CSV boolean: True / False / None (column absent or blank)."""
    v = row.get(key)
    if v in (None, ""):
        return None
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes"}


def new_figure(figsize, **kw):
    fig = plt.figure(figsize=figsize, facecolor=SURFACE, **kw)
    return fig


def style_axes(ax, *, xgrid: bool = True, ygrid: bool = False,
               spines: Sequence[str] = ("bottom",)) -> None:
    """Recessive chrome: hairline solid grid, no box, ticks that do not shout."""
    ax.set_facecolor(SURFACE)
    for name, sp in ax.spines.items():
        sp.set_visible(name in spines)
        sp.set_color(BASE)
        sp.set_linewidth(0.8)
    ax.grid(axis="x" if xgrid and not ygrid else ("y" if ygrid and not xgrid else "both"),
            color=GRID, linewidth=0.6, linestyle="-")
    if not xgrid and not ygrid:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=BASE)


def qtext(value: Optional[float]) -> str:
    """A q-value as it should be printed: never `0.000`, never a bare 0."""
    if value is None:
        return "—"
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def save(fig, out_path, *, dpi: int = 200):
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out_path
