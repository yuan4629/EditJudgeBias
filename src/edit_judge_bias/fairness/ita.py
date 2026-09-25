"""ITA (Individual Typology Angle) skin-lightness manipulation for the D-class track.

★ WHAT THIS MEASURES, AND WHAT IT DOES NOT

ITA is the standard photometric axis for apparent skin lightness (Chardon 1991; Del Bino &
Bernerd 2013):

    ITA = atan2(L* - 50, b*) * 180 / pi

in CIELAB, with larger angles meaning lighter skin. This module shifts the skin pixels of
one image along that axis and leaves every other pixel untouched.

**The construct is "apparent skin lightness", NOT ancestry.** ITA collapses skin to one
dimension; Fitzpatrick VI skin differs from *darkened* Fitzpatrick II skin in hue, chroma,
specular structure and subsurface scattering. Any write-up must say "apparent skin
lightness" and must not claim to have rendered a person of a different ancestry.

★ WHY THE SHIFT IS SYMMETRIC AND MATCHED, NOT "PUSH BOTH ARMS TO THE EXTREMES"

The obvious design -- send one arm to Fitzpatrick I-II and the other to V-VI -- silently
binds the perturbation *magnitude* to the arm label. A subject whose native ITA is 40 deg
sits 15 deg from a light target and 70 deg from a dark one, so the dark arm would carry
~4.7x the perturbation and any measured "gap" would be partly a gap in how much the image
was disturbed. Since this project has already measured that judges penalise local readable
pixel changes on 5/5 judges, that confound points straight at a false positive.

So `solve_matched_shift` moves **+delta and -delta around the subject's own median ITA**.
Both arms then carry an equal dose by construction, and delta is chosen large enough to
cross Fitzpatrick bins (bins are 13-27 deg wide, so delta=30 moves 40 deg -> 70/10 deg).

★ WHY GAMUT OVERFLOW IS REPORTED, NEVER SILENTLY CLIPPED

If a requested shift pushes pixels outside sRGB, clipping would quietly flatten skin
texture and the arm would carry an artefact the gate could not see. Instead the solver
bisects |dL| down until the clipped share is under `max_clipped_frac` and reports the
**achieved** delta and bin. That is what makes the gate's "achieved bin == target bin" a
real check rather than a restatement of the request.

★ TEXTURE PRESERVATION IS A TESTED PROPERTY, NOT A CLAIM

The shift is **strictly additive per pixel**, so within-mask contrast is preserved exactly.
`tests/test_ita.py` asserts the strong form: the per-pixel rank order of L* inside the mask is
unchanged (Spearman = 1.0) and the standard deviation is unchanged to within 5%.

An earlier version tapered the shift by pixel lightness to protect highlights. It was removed
because the taper is a function of L* and therefore *compresses* contrast -- asymmetrically
between the two arms: measured on a synthetic skin patch, the lighten arm lost 14% of its
within-mask L* spread (4.49 -> 3.84) while the darken arm lost none. An arm-dependent texture
change is precisely the confound this design exists to remove. Gamut pressure is handled the
honest way instead: measured, reported, and met by shrinking the shift symmetrically.

★ ONE MORE INVARIANT: OUTSIDE THE MASK, THE OUTPUT IS THE INPUT'S BYTES

`composite` is not optional. sRGB -> CIELAB -> sRGB is lossy even where the shift is exactly
zero (measured: max abs diff 2 outside the mask), which would silently break the one invariant
the D-class gate rests on. Shifted pixels are therefore written back into the ORIGINAL bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: Fitzpatrick-equivalent ITA bins in degrees (Chardon 1991; Del Bino & Bernerd 2013).
#: Bin widths are 13-27 deg, which is where MIN_CROSSING_DELTA comes from.
FITZPATRICK_ITA_BINS: Dict[str, Tuple[float, float]] = {
    "very_light": (55.0, 90.0),
    "light": (41.0, 55.0),
    "intermediate": (28.0, 41.0),
    "tan": (10.0, 28.0),
    "brown": (-30.0, 10.0),
    "dark": (-90.0, -30.0),
}

#: Default symmetric shift. Large enough to cross at least one bin from any starting point.
DEFAULT_DELTA_ITA = 30.0

#: The gate's floor on the ACHIEVED shift. A smaller shift is not a manipulation worth
#: judging, since the Fitzpatrick-equivalent bins are only 13-27 deg wide.
#:
#: ⚠️ Gate on this, NOT on "the bin changed". Whether a matched +-30 deg shift crosses a bin
#: depends on where the subject started: a subject at ITA 0 sits mid-`brown` (-30, 10], so
#: -30 lands exactly on the boundary and stays in the same bin. `bin_before` / `bin_after`
#: are descriptive; `achieved_delta_ita` is the contract.
MIN_CROSSING_DELTA = 25.0

#: Above this share of clipped pixels the solver shrinks the shift instead of clamping.
DEFAULT_MAX_CLIPPED_FRAC = 0.02

#: Alpha at or above which a pixel counts as "in the region" for MEASUREMENT.
#:
#: ★ Why measurement and application use different footprints. The shift is applied with the
#: feathered alpha, so band pixels receive a partial dose -- correct, that is what a feather
#: is for. But including them in the median made the *reported* dose depend on a cosmetic
#: parameter: measured 2026-07-30 on a real photo, the same requested -30 deg reported an
#: achieved -32.2 at `radius_frac=0` and only -27.1 at `radius_frac=0.01`, because the
#: half-shifted band drags the alpha-weighted median. The gate compares the achieved delta
#: against MIN_CROSSING_DELTA, so a feather tweak would have silently moved a pass/fail
#: boundary. Medians are therefore taken over the core.
#:
#: The threshold is 0.999, i.e. **fully dosed pixels only**, not merely "mostly inside".
#: A first attempt used 0.5 and still drifted (-32.2 to -28.3 across the same radii),
#: because pixels with alpha in [0.5, 1) receive a partial shift while counting in the
#: median. `feather` pins alpha to exactly 1.0 on its eroded core, so 0.999 selects
#: precisely the pixels that got the whole shift.
CORE_ALPHA = 0.999


def _np():
    try:
        import numpy as np  # noqa: PLC0415 - deferred, mirrors data/overlap.py::_cv2
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("ITA manipulation needs numpy (pip install numpy)") from exc
    return np


def _cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("ITA manipulation needs opencv-python (pip install opencv-python)") from exc
    return cv2


@dataclass
class ShiftSolution:
    """The solved shift plus everything needed to audit it."""

    delta_l: float
    delta_a: float
    delta_b: float
    requested_delta_ita: float
    achieved_delta_ita: float
    ita_before: float
    ita_after: float
    bin_before: str
    bin_after: str
    clipped_frac: float
    iterations: int
    shrunk: bool = False
    log: Dict[str, Any] = field(default_factory=dict)

    def as_params(self) -> Dict[str, Any]:
        return {
            "delta_l": round(self.delta_l, 4),
            "delta_a": round(self.delta_a, 4),
            "delta_b": round(self.delta_b, 4),
            "requested_delta_ita": round(self.requested_delta_ita, 3),
            "achieved_delta_ita": round(self.achieved_delta_ita, 3),
            "ita_before": round(self.ita_before, 3),
            "ita_after": round(self.ita_after, 3),
            "bin_before": self.bin_before,
            "bin_after": self.bin_after,
            "clipped_frac": round(self.clipped_frac, 5),
            "iterations": self.iterations,
            "shrunk": self.shrunk,
        }


# --------------------------------------------------------------------------- #
# ITA basics                                                                   #
# --------------------------------------------------------------------------- #
def ita_degrees(lightness, blue_yellow):
    """ITA in degrees from L* and b*. Accepts scalars or arrays."""
    np = _np()
    import math

    if np.isscalar(lightness) and np.isscalar(blue_yellow):
        return math.degrees(math.atan2(float(lightness) - 50.0, float(blue_yellow)))
    return np.degrees(np.arctan2(np.asarray(lightness, dtype="float64") - 50.0,
                                 np.asarray(blue_yellow, dtype="float64")))


def fitzpatrick_bin(ita: float) -> str:
    """Name the Fitzpatrick-equivalent bin an ITA falls in.

    Boundaries are half-open at the top so a value exactly on a boundary lands in the
    lighter bin, deterministically -- an ITA of exactly 55.0 is `very_light`.
    """
    value = float(ita)
    for name, (lo, hi) in FITZPATRICK_ITA_BINS.items():
        if lo <= value < hi:
            return name
    if value >= 90.0:
        return "very_light"
    return "dark"


def rgb_to_lab(rgb):
    """(H, W, 3) uint8 sRGB -> float32 CIELAB with L* in [0, 100], a*/b* centred on 0."""
    np = _np()
    cv2 = _cv2()
    arr = np.asarray(rgb)
    lab = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2LAB).astype("float32")
    out = np.empty_like(lab)
    out[..., 0] = lab[..., 0] * (100.0 / 255.0)
    out[..., 1] = lab[..., 1] - 128.0
    out[..., 2] = lab[..., 2] - 128.0
    return out


def lab_to_rgb(lab) -> Tuple[Any, float]:
    """CIELAB -> (uint8 sRGB, clipped share).

    The clipped share is the fraction of pixels whose L*/a*/b* had to be clamped into
    OpenCV's storage range. It is returned rather than swallowed so the caller can shrink
    the shift instead of flattening texture.
    """
    np = _np()
    cv2 = _cv2()
    lab = np.asarray(lab, dtype="float32")
    l_raw = lab[..., 0] * (255.0 / 100.0)
    a_raw = lab[..., 1] + 128.0
    b_raw = lab[..., 2] + 128.0
    out_of_range = (
        (l_raw < 0) | (l_raw > 255) | (a_raw < 0) | (a_raw > 255) | (b_raw < 0) | (b_raw > 255)
    )
    packed = np.empty(lab.shape, dtype="uint8")
    packed[..., 0] = np.clip(l_raw, 0, 255).astype("uint8")
    packed[..., 1] = np.clip(a_raw, 0, 255).astype("uint8")
    packed[..., 2] = np.clip(b_raw, 0, 255).astype("uint8")
    rgb = cv2.cvtColor(packed, cv2.COLOR_LAB2RGB)
    return rgb, float(out_of_range.mean())


def core_of(alpha):
    """The measurement footprint: pixels at or above `CORE_ALPHA`, else the whole support.

    Falling back to the support matters for a very small or heavily feathered region where
    no pixel reaches the core threshold -- reporting nothing there would be worse than
    reporting a diluted number, and the dilution is visible in `mask_frac`.
    """
    np = _np()
    a = np.asarray(alpha, dtype="float32")
    core = a >= CORE_ALPHA
    return core if core.any() else (a > 0)


def weighted_median_lab(lab, alpha) -> Tuple[float, float, float]:
    """Alpha-weighted median (L*, a*, b*) of the masked pixels.

    Median, not mean: a few specular highlights or a shadowed edge would drag a mean and
    make the solved shift depend on lighting rather than on the subject's skin.
    """
    np = _np()
    lab = np.asarray(lab, dtype="float64")
    weight = np.asarray(alpha, dtype="float64")
    sel = weight > 0
    if not sel.any():
        raise ValueError("weighted_median_lab: the mask selects no pixels")

    def _wmed(values):
        v = values[sel]
        w = weight[sel]
        order = np.argsort(v, kind="stable")
        v, w = v[order], w[order]
        cum = np.cumsum(w)
        if cum[-1] <= 0:
            raise ValueError("weighted_median_lab: mask weights sum to zero")
        return float(v[int(np.searchsorted(cum, cum[-1] / 2.0))])

    return _wmed(lab[..., 0]), _wmed(lab[..., 1]), _wmed(lab[..., 2])


# --------------------------------------------------------------------------- #
# The shift                                                                    #
# --------------------------------------------------------------------------- #
def _shift_once(lab, alpha, delta_l: float, delta_a: float, delta_b: float):
    """Apply a STRICTLY ADDITIVE Lab shift, weighted only by alpha.

    ★ There is deliberately no per-pixel roll-off here, and that is a correction of a real
    defect measured 2026-07-30. An earlier version tapered the shift by pixel lightness
    (`w = (100-L)/50` when lightening) to protect highlights. Because that weight is a
    *function of L*, it compresses contrast -- and it does so **asymmetrically between the
    two arms**: on a synthetic skin patch the lighten arm lost 14% of its within-mask L*
    standard deviation (4.49 -> 3.84, Spearman 0.9996) while the darken arm lost none
    (4.49 -> 4.49, Spearman 1.0). An arm-dependent texture change is precisely the
    confound this whole design exists to remove, so the roll-off is gone.

    Gamut pressure is handled the honest way instead: `solve_matched_shift` measures the
    clipped share and bisects the shift down until it is under `max_clipped_frac`, and
    reports the achieved delta. A measured, reported, arm-symmetric shrink beats an
    unmeasured per-pixel compression.
    """
    np = _np()
    lab = np.asarray(lab, dtype="float32").copy()
    alpha = np.asarray(alpha, dtype="float32")
    lab[..., 0] = lab[..., 0] + alpha * float(delta_l)
    lab[..., 1] = lab[..., 1] + alpha * float(delta_a)
    lab[..., 2] = lab[..., 2] + alpha * float(delta_b)
    return lab


def composite(base, shifted, alpha):
    """`base` everywhere alpha == 0, blended toward `shifted` as alpha rises.

    ★ THIS IS WHAT MAKES BYTE-IDENTITY TRUE, and it is not optional. Measured
    2026-07-30: returning the LAB->RGB round trip directly leaves a **max abs diff of 2**
    outside the mask, because sRGB -> CIELAB -> sRGB is lossy even where the shift is
    exactly zero. That silently violates the one invariant the D-class gate rests on. So
    the shifted pixels are composited back into the ORIGINAL bytes, and where alpha == 0
    the output is the original verbatim, by construction rather than by tolerance.
    """
    np = _np()
    base_arr = np.asarray(base)
    shifted_arr = np.asarray(shifted)
    if base_arr.shape != shifted_arr.shape:
        raise ValueError(f"composite shape mismatch: {base_arr.shape} vs {shifted_arr.shape}")
    a = np.asarray(alpha, dtype="float32")
    if a.shape != base_arr.shape[:2]:
        raise ValueError(f"alpha shape {a.shape} does not match image {base_arr.shape[:2]}")
    support = a > 0
    out = base_arr.copy()
    if not support.any():
        return out
    a3 = a[..., None] if base_arr.ndim == 3 else a
    blended = base_arr.astype("float32") * (1.0 - a3) + shifted_arr.astype("float32") * a3
    rounded = np.clip(np.rint(blended), 0, 255).astype(base_arr.dtype)
    out[support] = rounded[support]
    return out


def _target_lab_for_ita(l_med: float, b_med: float, target_ita: float) -> Tuple[float, float]:
    """Where (L*, b*) must land to hit `target_ita`, keeping the chroma radius fixed.

    Rotating along the ITA circle rather than moving L* alone: pure L* changes read as an
    exposure error, whereas real lightness differences move b* too.
    """
    import math

    radius = math.hypot(l_med - 50.0, b_med)
    if radius < 1e-6:
        radius = 1e-6
    angle = math.radians(target_ita)
    return 50.0 + radius * math.sin(angle), radius * math.cos(angle)


def solve_matched_shift(lab, alpha, *, delta_ita: float = DEFAULT_DELTA_ITA,
                        max_clipped_frac: float = DEFAULT_MAX_CLIPPED_FRAC,
                        max_iters: int = 12) -> ShiftSolution:
    """Solve a Lab shift moving the masked region's median ITA by `delta_ita` degrees.

    Sign convention: **positive `delta_ita` lightens**, negative darkens. Call it twice
    with +d and -d to get two arms carrying an equal dose (see the module docstring).

    ⚠️ THE ACHIEVED SHIFT IS EXACT ON THE IMAGE THE SOLUTION WAS SOLVED FROM, AND ONLY THERE.
    Measured over the real 31-scene pool: on the `original` member 0 of 62 study rows fall
    below the 25 deg dose floor (median 30.26). On the `edited` member 16 of 62 do -- and that
    is correct behaviour, not gamut loss. The mask is derived from the original, so on the
    edited image it covers *different pixels* (one scene's instruction replaces the background
    outright), and applying the same Lab offset to a different starting colour yields a
    different ITA change because ITA is a nonlinear function of Lab. `ebench_H_09_02` starts at
    ITA +17.6 on the original and -38.8 under the same mask on the edited image.

    The physical dose IS matched across members -- mean dE00 6.07 vs 5.89 on that pair -- which
    is exactly why `delta_e00_mean` is the cross-member consistency check and the ITA dose floor
    is evaluated on the `original` member only. Re-solving per member would give the two members
    different Lab offsets, which is the one thing this design must not do.

    An earlier version of this function took an `rgb=` argument and iterated the solve against
    the rendered image to "recover" the shortfall. It was removed: it was built on the
    misdiagnosis above, and measurement showed it changed nothing (15/124 -> 16/124) because
    there was never a shortfall on the member the solution is defined for.
    """
    np = _np()
    lab = np.asarray(lab, dtype="float32")
    alpha = np.asarray(alpha, dtype="float32")
    core = core_of(alpha)
    l_med, a_med, b_med = weighted_median_lab(lab, core)
    ita_before = float(ita_degrees(l_med, b_med))
    target_ita = ita_before + float(delta_ita)
    l_target, b_target = _target_lab_for_ita(l_med, b_med, target_ita)

    full_dl, full_db = l_target - l_med, b_target - b_med
    scale, shrunk, iterations = 1.0, False, 0
    best: Optional[Tuple[float, float, float, float, float]] = None
    for iterations in range(1, max_iters + 1):
        d_l, d_b = full_dl * scale, full_db * scale
        shifted = _shift_once(lab, alpha, d_l, 0.0, d_b)
        _rgb, clipped = lab_to_rgb(shifted)
        l_new, a_new, b_new = weighted_median_lab(shifted, core)
        ita_after = float(ita_degrees(l_new, b_new))
        best = (d_l, 0.0, d_b, clipped, ita_after)
        if clipped <= max_clipped_frac:
            break
        scale *= 0.75
        shrunk = True

    assert best is not None  # the loop always runs at least once
    d_l, d_a, d_b, clipped, ita_after = best

    return ShiftSolution(
        delta_l=float(d_l), delta_a=float(d_a), delta_b=float(d_b),
        requested_delta_ita=float(delta_ita),
        achieved_delta_ita=float(ita_after - ita_before),
        ita_before=ita_before, ita_after=ita_after,
        bin_before=fitzpatrick_bin(ita_before), bin_after=fitzpatrick_bin(ita_after),
        clipped_frac=float(clipped), iterations=iterations, shrunk=shrunk,
        log={"median_lab_before": [l_med, a_med, b_med], "target_ita": target_ita,
             "full_delta_l": float(full_dl), "full_delta_b": float(full_db),
             "final_scale": float(scale)},
    )


def apply_shift(rgb, alpha, solution: ShiftSolution) -> Tuple[Any, Dict[str, float]]:
    """Apply a solved shift to `rgb`. Returns (uint8 RGB, measurements).

    ★ The solution is solved ONCE on the original and applied verbatim here, so the
    original and the dataset's edited image receive the *same* dose. Re-solving per image
    would let the two members of one pair land on different achieved ITAs and make the pair
    internally inconsistent.
    """
    np = _np()
    lab = rgb_to_lab(rgb)
    shifted = _shift_once(lab, alpha, solution.delta_l, solution.delta_a, solution.delta_b)
    rendered, clipped = lab_to_rgb(shifted)
    # ★ Composite, never return the round trip directly -- see `composite`.
    out = composite(rgb, rendered, alpha)
    alpha_arr = np.asarray(alpha, dtype="float32")
    if (alpha_arr > 0).any():
        # Measure the SAVED result, not the intermediate Lab array: the composite +
        # uint8 rounding is what the judge actually sees, so that is what gets reported.
        # Measure over the CORE, so the reported dose does not move with the feather.
        core = core_of(alpha_arr)
        l_med, _a, b_med = weighted_median_lab(rgb_to_lab(out), core)
        ita_after = float(ita_degrees(l_med, b_med))
        l_before, _ab, b_before = weighted_median_lab(lab, core)
        ita_before = float(ita_degrees(l_before, b_before))
        core_frac = float(core.mean())
    else:
        ita_after = ita_before = float("nan")
        core_frac = 0.0
    return out, {
        "ita_before": ita_before,
        "ita_after": ita_after,
        "delta_ita": ita_after - ita_before,
        "bin_after": fitzpatrick_bin(ita_after) if ita_after == ita_after else "unknown",
        "clipped_frac": clipped,
        "mask_frac": float((alpha_arr > 0).mean()),
        "core_frac": core_frac,
    }


def measure(rgb, alpha) -> Dict[str, Any]:
    """Describe the masked region's apparent lightness, for the gate and the manifest."""
    np = _np()
    alpha_arr = np.asarray(alpha, dtype="float32")
    if not (alpha_arr > 0).any():
        return {"ita_median": None, "bin": "unknown", "l_std": None, "mask_frac": 0.0,
                "core_frac": 0.0}
    lab = rgb_to_lab(rgb)
    core = core_of(alpha_arr)
    l_med, _a_med, b_med = weighted_median_lab(lab, core)
    ita = float(ita_degrees(l_med, b_med))
    return {
        "ita_median": ita,
        "bin": fitzpatrick_bin(ita),
        "l_std": float(np.asarray(lab[..., 0])[core].std()),
        "mask_frac": float((alpha_arr > 0).mean()),
        "core_frac": float(core.mean()),
    }
