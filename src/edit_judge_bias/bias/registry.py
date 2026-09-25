"""Bias registry: resolve a `bias_type` string (from YAML) to an injector.

Register new image injectors here. `position` is intentionally NOT here: it is a judge-protocol bias
(swap A/B order, no pixel change) handled by the pairwise runner, not a
BiasInjector — `get_injector("position")` raises a pointed error.
"""

from __future__ import annotations

from typing import Dict, List, Type

from edit_judge_bias.bias.aesthetic_filter import AestheticFilterInjector
from edit_judge_bias.bias.base import BiasInjector
from edit_judge_bias.bias.brightness import BrightnessInjector
from edit_judge_bias.bias.detail_caption import DetailCaptionInjector
from edit_judge_bias.bias.distraction import DistractionInjector
from edit_judge_bias.bias.edit_damage import EditDamageInjector
from edit_judge_bias.bias.padding import PaddingInjector
from edit_judge_bias.bias.region_annotation import RegionAnnotationInjector
from edit_judge_bias.bias.saturation import SaturationInjector
from edit_judge_bias.bias.sham import ShamInjector
from edit_judge_bias.bias.text_overlay import TextOverlayInjector
from edit_judge_bias.bias.watermark import WatermarkInjector
from edit_judge_bias.bias.zoom_inset import ZoomInsetInjector

_INJECTORS: List[Type[BiasInjector]] = [
    # B-class — pixel-level (MVP 5 + full-version aesthetic_filter).
    BrightnessInjector,
    SaturationInjector,
    WatermarkInjector,
    TextOverlayInjector,
    PaddingInjector,
    AestheticFilterInjector,
    # C-class — content/overlay (localize the edit via edit_region.py).
    RegionAnnotationInjector,
    ZoomInsetInjector,
    DetailCaptionInjector,
    DistractionInjector,
    # Control — a visually null perturbation, so the other ten have a noise floor
    # to be read against. Registered here so it rides the identical code path;
    # anything special-cased would not be a control.
    ShamInjector,
    # ⚠️ THE ONE INJECTOR THAT DELIBERATELY DESTROYS EDIT QUALITY. `sham` measures the
    # validator's SPECIFICITY (what it flags when nothing changed); this measures its
    # SENSITIVITY (what it misses when the edit is provably gone). Without the second
    # number, "all eleven cues preserve quality" could be a fact about the validator.
    # It is registered so it rides the identical code path — the same reason `sham` is —
    # but it appears in NO arm config and must never enter a judge grid or a claim table
    # (pinned by `tests/test_edit_damage.py`).
    EditDamageInjector,
]

_REGISTRY: Dict[str, Type[BiasInjector]] = {cls.bias_type: cls for cls in _INJECTORS}
# Convenience aliases for the overlay injector's two documented uses.
_ALIASES: Dict[str, str] = {
    "instruction_overlay": "text_overlay",
    "keyword_overlay": "text_overlay",
    "authenticity_overlay": "watermark",
    # WP-A4a's severity ladder. `biased_id` is `{sample}__{bias_type}` and the runner
    # takes the type from the CONFIG, so three severities of one injector need three
    # names or they overwrite each other's images and manifest rows. Aliases give that
    # without three near-identical classes; the severity itself stays in the bias config
    # where it is visible, and `edit_damage_100` is not a different injector, it is the
    # same one at severity 1.0 (asserted in tests/test_edit_damage.py).
    "edit_damage_100": "edit_damage",
    "edit_damage_50": "edit_damage",
    "edit_damage_25": "edit_damage",
    "edit_damage_blur": "edit_damage",
}

#: Judge-protocol biases that have no image injector (handled in pairwise judging).
PROTOCOL_BIASES = frozenset({"position"})


def register(cls: Type[BiasInjector]) -> Type[BiasInjector]:
    """Register an injector class (use as a decorator or call directly)."""
    if not cls.bias_type:
        raise ValueError(f"{cls.__name__} must set a non-empty bias_type")
    _REGISTRY[cls.bias_type] = cls
    return cls


def available() -> List[str]:
    """Sorted list of registered image-injector bias types."""
    return sorted(_REGISTRY)


def get_injector(bias_type: str) -> BiasInjector:
    """Instantiate the injector for `bias_type` (aliases resolved)."""
    key = _ALIASES.get(bias_type, bias_type)
    if key not in _REGISTRY:
        if bias_type in PROTOCOL_BIASES:
            raise KeyError(
                f"{bias_type!r} is a judge-protocol bias (pairwise A/B swap), not an "
                "image injector; it has no BiasInjector."
            )
        raise KeyError(
            f"unknown bias_type {bias_type!r}; registered: {available()}"
        )
    return _REGISTRY[key]()
