"""Bias injection layer: deterministic PIL injectors + registry (Milestone 2).

The MVP image injectors: brightness, saturation, watermark
(authenticity overlay), text_overlay (instruction/keyword), padding. The sixth MVP
bias, `position`, is a judge-protocol bias applied during pairwise judging.
"""

from edit_judge_bias.bias.base import BiasInjector, BiasResult
from edit_judge_bias.bias.brightness import BrightnessInjector
from edit_judge_bias.bias.padding import PaddingInjector
from edit_judge_bias.bias.registry import (
    PROTOCOL_BIASES,
    available,
    get_injector,
    register,
)
from edit_judge_bias.bias.saturation import SaturationInjector
from edit_judge_bias.bias.text_overlay import TextOverlayInjector
from edit_judge_bias.bias.watermark import WatermarkInjector

__all__ = [
    "BiasInjector",
    "BiasResult",
    "BrightnessInjector",
    "SaturationInjector",
    "WatermarkInjector",
    "TextOverlayInjector",
    "PaddingInjector",
    "available",
    "get_injector",
    "register",
    "PROTOCOL_BIASES",
]
