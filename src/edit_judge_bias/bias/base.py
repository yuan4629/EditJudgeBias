"""BiasInjector base class and result type.

Design principles enforced here so individual injectors stay tiny:

1. Originals are never overwritten — injectors only ever write to `output_path`.
2. Reproducible — a `seed` derives the `random.Random` passed to `transform`.
3. Non-fatal — `apply` catches errors and returns `success=False` with a message,
   so one corrupt image can't abort a whole batch (the runner logs and continues).

Subclasses implement `transform(image, config, rng, sample) -> (Image, params)`.
The returned `params` dict is free-form per injector but SHOULD include a
``bias_strength`` float (a representative magnitude stored on the BiasedRecord)
plus anything worth tracing later (overlay_text, padding_ratio, output size).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from edit_judge_bias.data.schema import BiasAppliedTo

PathLike = str | Path

#: Resolution the `*_px` legacy defaults were tuned against. A ratio is converted
#: to pixels as ``short_edge * ratio``, so a ratio of ``px / REFERENCE_SHORT_EDGE``
#: reproduces the old look on a 512px image and scales from there.
REFERENCE_SHORT_EDGE = 512


def relative_px(
    config: Dict[str, Any],
    *,
    ratio_key: str,
    px_key: str,
    short_edge: int,
    default_ratio: float,
    minimum: int = 1,
) -> Tuple[int, Dict[str, Any]]:
    """Resolve a size that must stay a constant *fraction* of the frame.

    The pool mixes 500x500, 512x512 and 1024x1024 images. An absolute-pixel
    annotation width or sharpening radius therefore covers a different share of a
    1024px frame than of a 512px one, which would make any measured bias effect
    partly a function of resolution -- a confound we cannot separate after the
    fact. Sizes are consequently expressed as a fraction of the short edge.

    An explicit `px_key` in the config still wins, so ablations can pin an
    absolute value on purpose. Returns (pixels, params) where params records both
    the ratio and the resolved pixels for the manifest.
    """
    if px_key in config and config[px_key] is not None:
        px = max(minimum, int(config[px_key]))
        return px, {px_key: px, ratio_key: None, f"{ratio_key}_source": "explicit_px"}
    ratio = float(config.get(ratio_key, default_ratio))
    px = max(minimum, int(round(short_edge * ratio)))
    return px, {px_key: px, ratio_key: ratio, f"{ratio_key}_source": "ratio"}


@dataclass
class BiasResult:
    """Outcome of one bias injection (mirrors the §5.3 return contract)."""

    bias_type: str
    biased_image_path: Path
    success: bool
    bias_strength: float = 0.0
    bias_applied_to: BiasAppliedTo = BiasAppliedTo.EDITED_IMAGE
    params: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class BiasInjector(ABC):
    """Abstract deterministic PIL bias injector applied to an edited image."""

    #: Registry key / stored `bias_type`. Set by each subclass.
    bias_type: str = ""
    #: Which input a bias perturbs. All MVP image injectors edit the edited image.
    bias_applied_to: BiasAppliedTo = BiasAppliedTo.EDITED_IMAGE
    #: C-class injectors localize the edit via |edited - original|. When True the
    #: batch runner resolves the sample's original image and passes its absolute
    #: path in config as ``original_image_path_abs`` (None when unavailable).
    needs_original: bool = False

    @abstractmethod
    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Return (biased RGB image, params dict). Pure; no disk i/o."""

    def apply(
        self,
        image_path: PathLike,
        output_path: PathLike,
        *,
        config: Optional[Dict[str, Any]] = None,
        sample: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> BiasResult:
        """Load `image_path`, transform, save to `output_path`, return a BiasResult.

        Never raises for ordinary failures (missing/corrupt image, bad config):
        those come back as `success=False` so the batch can continue.
        """
        output_path = Path(output_path)
        config = dict(config or {})
        rng = random.Random(seed)
        try:
            with Image.open(image_path) as im:
                img = im.convert("RGB")
            out_img, params = self.transform(img, config, rng, sample or {})
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(output_path)
            return BiasResult(
                bias_type=self.bias_type,
                biased_image_path=output_path,
                success=True,
                bias_strength=float(params.get("bias_strength", 0.0)),
                bias_applied_to=self.bias_applied_to,
                params=params,
            )
        except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
            return BiasResult(
                bias_type=self.bias_type,
                biased_image_path=output_path,
                success=False,
                bias_applied_to=self.bias_applied_to,
                message=f"{type(exc).__name__}: {exc}",
            )
