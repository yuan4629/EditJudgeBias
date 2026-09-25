"""D-class attribute injectors: apparent skin lightness, by construction.

★ WHY THESE ARE NOT IN `bias/registry.py`

That registry drives the A/B/C invariance benchmark: `configs/experiment/*.yaml` name its
keys and `tests/test_bias_registry.py` enumerates `available()`. A demographic injector
sitting there would be one config typo away from entering the main grid, where it does not
belong. Two further mismatches make the reuse wrong on the merits:

* `BiasInjector.transform(image, config, rng, sample)` sees ONE image and returns ONE image.
  A D-class arm needs *one mask and one solved dose shared across two arms and two members
  of a pair* -- there is nowhere in that signature to put it.
* `BiasAppliedTo` is `{edited_image, judge_prompt}`. A D-class attribute construction is
  applied to BOTH the original and the edited image, which is neither.

So this module carries its own three-function registry with the same shape
(`available_attribute_injectors` / `get_attribute_injector`) and none of the coupling.

★ THE FOUR ARMS, AND WHY THE LAST TWO ARE NOT OPTIONAL

    light / dark    the study arms: +delta and -delta around the subject's OWN median ITA
    sham            the same code path with delta = 0            -> false-positive floor
    sham_nonskin    the same dose on the SAME PERSON's non-skin  -> artefact-vs-attribute

`sham` is this project's established discipline (a null perturbation that 2 of 5 judges
still respond to, +0.30 and +0.59 on `fine_score`). `sham_nonskin` answers the question a
reviewer asks first: this study has already measured that judges penalise local, readable
pixel changes on 5/5 judges, so a skin recolor could move scores because it is a recolor,
not because of the tone. Recolouring the same subject's clothing or hair with a matched dose
separates the two. Without it there is no fairness claim to make -- only an artefact
measurement -- which is why `SkinToneITAInjector.arms()` emits it by default.

⚠️ **`sham_nonskin` matches the REQUESTED delta_ita and the AREA, not the achieved physical
dose.** Measured over the 992 v4 scenes that have one: median 0.90x the skin mask's area but
**1.29x its per-pixel mean dE00** (1.47x dE00 mass; p90 2.14x / 2.61x), because the same ITA
shift lands on differently-coloured pixels. The construction gate's `max_delta_e00_ratio` only
checks original-vs-edited *within* an arm and never study-vs-`sham_nonskin`. Consequence, and
it is load-bearing for any mechanism sentence: `study - sham_nonskin` is an UPPER BOUND on a
region effect, not an estimate of one. See `build_dose_control_decomposition`.

⚠️ **`sham` is a zero-DOSE arm, not a zero-CHANGE arm.** It runs the identical code path, so
it carries that path's numerical footprint: sRGB -> CIELAB -> sRGB plus the uint8 composite
leaves a real but tiny change inside the mask (measured on a real photo: max 5/255, mean
dE00 **0.375**, versus ~8.9 for a study arm -- a 24x smaller dose). That is exactly what a
placebo should be: `delta_ita == 0` with the pipeline's own noise included, directly
analogous to the A/B/C `sham` being a JPEG round trip rather than a no-op. Do not "fix" the
nonzero `max_inside_diff`; it is the measurement.

★ WHY THE MASK AND THE DOSE ARE SOLVED ONCE

Both arms of a scene share one mask (solved from the original) and one solved shift
magnitude. Two consequences, both load-bearing:

* segmentation error becomes **common-mode** -- if the mask clips an ear, it clips it
  identically in both arms, so it cannot produce a difference between them;
* the two arms carry an **equal dose**, so a measured gap cannot be a gap in how much the
  image was disturbed. `mask_sha256` is recorded on every row so this is auditable rather
  than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from edit_judge_bias.fairness import ita as _ita
from edit_judge_bias.fairness import person_region as _pr

#: Roles an arm can play in the analysis. `study` rows enter the gap; the two sham roles are
#: floors and are reported as their own rows, never inside the study's BH family -- the same
#: treatment `sham` already gets in the A/B/C tables.
ROLE_STUDY = "study"
ROLE_SHAM = "sham"
ROLE_SHAM_NONSKIN = "sham_nonskin"

#: Which image of the (original, edited) pair an injection was applied to. Both are needed:
#: the pair the judge sees is (original_variant, edited_variant).
APPLIED_ORIGINAL = "original"
APPLIED_EDITED = "edited"


@dataclass(frozen=True)
class InjectionSpec:
    """One arm of one attribute."""

    attribute: str          # "skin_tone"
    label: str              # "light" | "dark" | "sham" | "sham_nonskin"
    role: str               # ROLE_STUDY | ROLE_SHAM | ROLE_SHAM_NONSKIN
    delta_ita: float = 0.0
    mask_kind: str = "skin"  # "skin" | "nonskin"


@dataclass
class InjectionResult:
    """One injected image plus everything the gate and the manifest need."""

    image: Any                       # np.ndarray uint8 RGB
    alpha: Any                       # np.ndarray float32 in [0, 1]
    mask_sha256: str
    measured: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    forced_lsb: bool = False
    success: bool = True
    message: str = ""

    @classmethod
    def failed(cls, message: str) -> "InjectionResult":
        """A dropped arm, reported rather than raised, so one bad scene cannot abort a batch."""
        return cls(image=None, alpha=None, mask_sha256="", success=False, message=message)


@dataclass
class SceneMasks:
    """The mask and dose for ONE scene, solved once and shared by every arm and member."""

    base_sample_id: str
    skin: Any
    skin_alpha: Any
    skin_sha256: str
    nonskin: Optional[Any]
    nonskin_alpha: Optional[Any]
    nonskin_sha256: Optional[str]
    nonskin_stats: Dict[str, Any]
    person_frac: float
    face_frac: float
    person_method: str
    solutions: Dict[str, _ita.ShiftSolution] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class SkinToneITAInjector:
    """Apparent skin lightness via a matched symmetric ITA shift.

    The construct is **apparent skin lightness**, not ancestry -- see `fairness/ita.py`.
    """

    attribute = "skin_tone"

    def __init__(self, *, delta_ita: float = _ita.DEFAULT_DELTA_ITA,
                 include_sham: bool = True, include_sham_nonskin: bool = True):
        self.delta_ita = float(delta_ita)
        self.include_sham = bool(include_sham)
        self.include_sham_nonskin = bool(include_sham_nonskin)

    def arms(self) -> List[InjectionSpec]:
        """The arms this injector emits, study arms first.

        ⚠️ Labels are sorted alphabetically downstream by `compute_attribute_gaps` to fix the
        sign of the gap, so `dark` is label_a and `light` is label_b: the reported gap is
        **dark minus light**. Renaming an arm silently flips every sign in the table.
        """
        out = [
            InjectionSpec(self.attribute, "dark", ROLE_STUDY, -self.delta_ita, "skin"),
            InjectionSpec(self.attribute, "light", ROLE_STUDY, +self.delta_ita, "skin"),
        ]
        if self.include_sham:
            out.append(InjectionSpec(self.attribute, "sham", ROLE_SHAM, 0.0, "skin"))
        if self.include_sham_nonskin:
            out.append(
                InjectionSpec(self.attribute, "sham_nonskin", ROLE_SHAM_NONSKIN,
                              -self.delta_ita, "nonskin")
            )
        return out

    # ------------------------------------------------------------------ #
    # Step 1: solve the scene once                                        #
    # ------------------------------------------------------------------ #
    def prepare_scene(self, original_rgb, *, base_sample_id: str,
                      regions: _pr.RegionSet, cfg: Optional[Dict[str, Any]] = None,
                      ) -> SceneMasks:
        """Derive the shared mask(s) and solve each arm's dose on the ORIGINAL image."""
        np = _pr._np()
        cfg = _pr._merge_cfg(cfg)
        short_edge = min(regions.width, regions.height)
        radius_frac = float(cfg["feather"].get("radius_frac", 0.01))

        skin = np.asarray(regions.skin).astype(bool)
        skin_alpha = _pr.feather(skin, radius_frac=radius_frac, short_edge=short_edge)

        # The artefact control: same person, non-skin, area-matched to the skin region.
        nonskin, nonskin_stats = _pr.nonskin_same_person_region(
            regions.person, skin, target_area=int(skin.sum()), cfg=cfg
        )
        if nonskin is None:
            nonskin_alpha, nonskin_sha = None, None
        else:
            nonskin_alpha = _pr.feather(nonskin, radius_frac=radius_frac, short_edge=short_edge)
            nonskin_sha = _pr.mask_sha256(nonskin)

        scene = SceneMasks(
            base_sample_id=base_sample_id,
            skin=skin, skin_alpha=skin_alpha, skin_sha256=_pr.mask_sha256(skin),
            nonskin=nonskin, nonskin_alpha=nonskin_alpha, nonskin_sha256=nonskin_sha,
            nonskin_stats=nonskin_stats,
            person_frac=regions.person_frac, face_frac=regions.face_frac,
            person_method=regions.person_method,
            diagnostics={"feather_radius_frac": radius_frac,
                         "skin_frac": float(skin.mean()),
                         "support_frac": float((skin_alpha > 0).mean())},
        )

        # ★ Solve every dose ONCE, on the original, over the mask that arm will use.
        lab = _ita.rgb_to_lab(original_rgb)
        for spec in self.arms():
            alpha = skin_alpha if spec.mask_kind == "skin" else nonskin_alpha
            if alpha is None:
                continue
            if spec.role == ROLE_SHAM:
                # No dose to solve: the point of the arm is that nothing was applied. A
                # zero-delta ShiftSolution keeps the manifest schema uniform.
                scene.solutions[spec.label] = _ita.ShiftSolution(
                    delta_l=0.0, delta_a=0.0, delta_b=0.0,
                    requested_delta_ita=0.0, achieved_delta_ita=0.0,
                    ita_before=float("nan"), ita_after=float("nan"),
                    bin_before="unchanged", bin_after="unchanged",
                    clipped_frac=0.0, iterations=0,
                )
                continue
            # Solved on the ORIGINAL and reused verbatim on the edited member. The ITA
            # readout on the edited member will differ (the mask covers different pixels
            # there and ITA is nonlinear); `delta_e00_mean` is the cross-member dose check
            # and the ITA floor is evaluated on the original. See `ita.solve_matched_shift`.
            scene.solutions[spec.label] = _ita.solve_matched_shift(
                lab, alpha, delta_ita=spec.delta_ita
            )
        return scene

    # ------------------------------------------------------------------ #
    # Step 2: apply one arm to one member                                 #
    # ------------------------------------------------------------------ #
    def apply(self, rgb, scene: SceneMasks, spec: InjectionSpec, *,
              applied_to: str) -> InjectionResult:
        """Inject one arm into one image. Never raises for ordinary failures."""
        np = _pr._np()
        alpha = scene.skin_alpha if spec.mask_kind == "skin" else scene.nonskin_alpha
        sha = scene.skin_sha256 if spec.mask_kind == "skin" else scene.nonskin_sha256
        if alpha is None:
            return InjectionResult.failed(
                f"{spec.label}: no {spec.mask_kind} region on this scene "
                f"({scene.nonskin_stats.get('reason', 'unknown')})"
            )
        solution = scene.solutions.get(spec.label)
        if solution is None:
            return InjectionResult.failed(f"{spec.label}: no solved dose for this arm")

        base = np.asarray(rgb)
        if np.asarray(alpha).shape != base.shape[:2]:
            return InjectionResult.failed(
                f"{spec.label}: mask {np.asarray(alpha).shape} does not match "
                f"{applied_to} image {base.shape[:2]}; the pair members must be aligned"
            )

        out, measured = _ita.apply_shift(base, alpha, solution)
        forced_lsb = False
        if spec.role == ROLE_SHAM:
            out, forced_lsb = _force_distinct(base, out, alpha)

        ok, worst = _pr.outside_mask_identical(base, out, alpha)
        measured.update({"outside_mask_identical": bool(ok), "max_outside_diff": int(worst)})
        inside = np.abs(out.astype("int32") - base.astype("int32")).max(axis=2)
        support = np.asarray(alpha) > 0
        measured["max_inside_diff"] = int(inside[support].max()) if support.any() else 0
        measured.update(delta_e00_stats(base, out, alpha))

        return InjectionResult(
            image=out, alpha=alpha, mask_sha256=sha or "",
            measured=measured,
            params={
                "attribute": spec.attribute, "label": spec.label, "role": spec.role,
                "mask_kind": spec.mask_kind, "applied_to": applied_to,
                "requested_delta_ita": spec.delta_ita,
                "person_frac": scene.person_frac, "face_frac": scene.face_frac,
                "person_method": scene.person_method,
                **solution.as_params(),
            },
            forced_lsb=forced_lsb,
        )


def delta_e00_stats(base, out, alpha) -> Dict[str, Any]:
    """CIEDE2000 between `base` and `out` over the fully-dosed core.

    ★ WHY THIS IS MEASURED AND NOT ASSUMED. `sham_nonskin` only works as an artefact control
    if it carries a dose comparable to the study arm. Matching is done in ITA degrees, which
    is not the same as matching perceptual difference: measured on a real photo, the two
    study arms agreed to within 3% on mean dE00 (8.93 dark vs 8.68 light) but the non-skin
    control came in 20% LOWER per pixel (7.11) while its total dE00 **mass** ran 8% HIGHER
    (1.05e6 vs 0.97e6), because clothing has different chroma and its eroded core holds more
    pixels than the fragmented skin mask.

    Both readings are therefore recorded on every row, so dose comparability is checked
    against the data at analysis time instead of asserted here. If the control's mass drifts
    far from the study arm's, that is a fact the table must show, not one the injector should
    quietly iterate away.
    """
    np = _pr._np()
    try:
        from skimage.color import deltaE_ciede2000, rgb2lab  # noqa: PLC0415
    except ImportError:  # pragma: no cover - environment-dependent
        return {}
    core = np.asarray(alpha) >= _ita.CORE_ALPHA
    if not core.any():
        core = np.asarray(alpha) > 0
    if not core.any():
        return {"core_px": 0}
    lab_a = rgb2lab(np.asarray(base) / 255.0)
    lab_b = rgb2lab(np.asarray(out) / 255.0)
    de = deltaE_ciede2000(lab_a, lab_b)[core]
    return {
        "core_px": int(core.sum()),
        "delta_e00_mean": float(de.mean()),
        "delta_e00_median": float(np.median(de)),
        "delta_e00_mass": float(de.sum()),
    }


def _force_distinct(base, out, alpha) -> Tuple[Any, bool]:
    """Guarantee the sham arm is a byte-distinct request, and say when it had to be forced.

    ★ Without this the sham arm reports a zero it never measured. `judges/openai_judge.py`
    caches responses on a hash of the payload **including the base64 image**, so a
    byte-identical sham image is served the baseline's answer and the placebo floor comes
    back as exactly 0.0 -- fabricated by plumbing, not measured. `bias/sham.py:73-78`
    documents the same trap for the A/B/C placebo; this is the D-class copy of it.

    One LSB on one masked pixel is far below both perception and SSIM resolution. It is
    placed INSIDE the mask so the byte-identity assertion still holds outside it.
    """
    np = _pr._np()
    if out.tobytes() != np.asarray(base).tobytes():
        return out, False
    support = np.asarray(alpha) > 0
    ys, xs = np.where(support)
    out = out.copy()
    if len(ys) == 0:
        return out, False
    y, x = int(ys[0]), int(xs[0])
    out[y, x, 0] = out[y, x, 0] ^ 1
    return out, True


# --------------------------------------------------------------------------- #
# Registry — same shape as bias/registry.py, none of the coupling              #
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Any] = {"skin_tone": SkinToneITAInjector}


def available_attribute_injectors() -> List[str]:
    return sorted(_REGISTRY)


def get_attribute_injector(name: str, **kwargs) -> Any:
    """Resolve an attribute name to its injector.

    Raises on an unknown name rather than returning None: a typo must stop the run, not
    silently drop an arm from the grid.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown attribute injector {name!r}; available: {available_attribute_injectors()}"
        ) from exc
    return cls(**kwargs)
