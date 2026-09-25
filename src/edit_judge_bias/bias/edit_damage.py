"""WP-A4a — the validator SENSITIVITY positive control. NOT a bias, NOT a cue.

★ WHY THIS EXISTS, AND WHY IT IS THE ONE INJECTOR THAT MUST NOT PRESERVE QUALITY.

Every other injector in this package is quality-preserving by construction, and the
whole study is conditional on that: a score shift only counts if the perturbation
provably did not change how well the edit was done. The instrument that certifies it is
the MLLM validator, and the only thing measured about that instrument so far is its
SPECIFICITY — its pass rate on `sham`, a visually null re-encode, which is its false-flag
floor (gemini 0.982, gpt-4o-mini 0.991, glm-4v 0.864).

Specificity alone cannot license the conclusion. A validator that answers "preserved" to
everything scores a perfect floor and passes all eleven cues, and "all eleven cues
preserve quality" would then be a fact about the validator, not about the cues. The
missing half is SENSITIVITY: shown an edit that demonstrably WAS damaged, does it say so?

This injector manufactures that ground truth. It reverts the estimated edit region back
towards the original image by a controlled fraction: at `severity = 1.0` the region is
the original's pixels, so the instruction was provably not carried out there — no
judgement call, no annotation, no rater. A validator that passes THAT cannot certify
anything, and one that catches it at 1.0 but not at 0.25 has a measurable threshold that
belongs beside every "not significantly below the floor" verdict in §5.7.

★ THREE GUARDS, because a deliberately damaging injector in a registry of preserving
ones is a foot-gun.

1. The `bias_type` is `edit_damage` and it appears in NO arm config. It is generated into
   its own manifest and validated on its own; nothing in the judge grid can reach it.
2. `revert` mode RAISES without an original image rather than degrading to the centered
   fallback box. A "positive control" that quietly damaged a box in the middle of the
   frame would be worthless in the exact direction that matters — it would look like a
   control while proving nothing.
3. Every output records `mean_abs_change` and `changed_frac` measured on the pixels, so
   "the damage actually landed" is a number in the manifest rather than an assumption.
   `distraction`'s spec violation was found precisely because rebuilt geometry was
   trusted over measured pixels; the same mistake is available here.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image, ImageChops, ImageFilter

from edit_judge_bias.bias.base import BiasInjector
from edit_judge_bias.bias.edit_region import estimate_edit_region

#: How much of the edit to undo. 1.0 = the region becomes the original's pixels.
DEFAULT_SEVERITY = 1.0
#: `revert` undoes the edit (breaks instruction adherence); `blur` destroys the detail
#: inside it (breaks detail preservation). Two different validator questions, on purpose.
DEFAULT_MODE = "revert"
#: Gaussian radius as a fraction of the region's short edge, so a 512px and a 1024px
#: frame are damaged comparably -- the same reasoning as `relative_px` for the cues.
DEFAULT_BLUR_RATIO = 0.08
MODES = ("revert", "blur")


class EditDamageInjector(BiasInjector):
    """Destroy the edit inside its own region, by a controlled amount."""

    bias_type = "edit_damage"
    needs_original = True

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        mode = str(config.get("mode", DEFAULT_MODE))
        if mode not in MODES:
            raise ValueError(f"edit_damage: mode must be one of {MODES}, got {mode!r}")
        severity = float(config.get("severity", DEFAULT_SEVERITY))
        if not 0.0 < severity <= 1.0:
            raise ValueError(f"edit_damage: severity must be in (0, 1], got {severity}")

        original_path = config.get("original_image_path_abs")
        region = estimate_edit_region(
            image,
            original=original_path,
            mask=config.get("mask_path_abs"),
            threshold=int(config.get("region_threshold", 15)),
        )
        if mode == "revert" and region.method == "fallback":
            # Guard 2. Damaging a centered box on an image whose edit we could not
            # locate produces a control that proves nothing about edit quality.
            raise ValueError(
                "edit_damage/revert needs a locatable edit region; got the fallback box "
                f"(original={original_path!r})"
            )

        box = region.bbox
        out = image.copy()
        if mode == "revert":
            with Image.open(original_path) as im:
                original = im.convert("RGB")
            resized = original.size != image.size
            if resized:
                original = original.resize(image.size)
            patch = Image.blend(image.crop(box), original.crop(box), severity)
        else:
            resized = False
            radius = max(1.0, min(box[2] - box[0], box[3] - box[1]) * DEFAULT_BLUR_RATIO)
            # Blur the WHOLE frame and take the region out of it, rather than blurring
            # the crop. Blurring a crop convolves against its own replicated edge, so
            # the region boundary stays razor-sharp and the pixels just inside it are
            # wrong — a seam that is an artefact of the renderer, not of the damage.
            # `severity` scales the blend, so this mode has the same monotone ladder.
            patch = Image.blend(
                image.crop(box),
                image.filter(ImageFilter.GaussianBlur(radius)).crop(box),
                severity,
            )
        out.paste(patch, box)

        diff = ImageChops.difference(image, out).convert("L")
        hist = diff.histogram()
        n_px = image.width * image.height
        changed = sum(hist[9:])  # threshold 8, matching the distraction spec audit
        mean_abs = sum(i * c for i, c in enumerate(hist)) / n_px
        params = {
            "bias_strength": severity,
            "mode": mode,
            "severity": severity,
            "region_bbox": list(box),
            "region_method": region.method,
            "region_area_frac": round(region.area_frac, 4),
            "original_resized": resized,
            # Guard 3: measured, not inferred. A run where these come back ~0 has
            # produced 110 pictures that look like a control and are not one.
            "mean_abs_change": round(mean_abs, 3),
            "changed_frac": round(changed / n_px, 4),
        }
        return out, params
