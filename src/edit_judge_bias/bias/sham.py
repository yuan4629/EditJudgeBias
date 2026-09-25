"""Sham (placebo) perturbation — the noise floor for every other bias.

This is not a bias. It is the control every "the judge moved when it shouldn't
have" result needs: an image that went through the same pipeline, came out as a
different file with different bytes, and is *visually the same picture*.

Why it is needed. The pilot's real effects are −0.18 … −0.58 score points, while
the judges' own absolute change per item (ASC) was 0.13 (gemini-3.5-flash) to 0.22
(gpt-4o-viescore) — the same order of magnitude. Without a placebo arm there is no
way to answer "is that shift bigger than the judge's own run-to-run noise?", and
the whole result rests on a p-value against a null nobody measured. With it, the
sham arm's confidence interval becomes an *equivalence bound*: perturbing the file
without perturbing the picture moves the judge at most this much.

The transform is a lossy re-encode at a quality where the loss is invisible. Two
properties matter, and the obvious alternatives fail one each:

- **Visually null.** Measured over 12 random pool images: this re-encode holds
  SSIM 0.9995 (worst 0.9987). The first version of this injector cropped 1px and
  resampled back, which scored 0.9385 (worst 0.8916) — on a par with the
  *brightness* bias at 0.96. That is a real perturbation wearing a control's name,
  and it would have made the equivalence bound meaningless.
- **The bytes always change.** A plain re-save is a no-op for a lossless PNG, and
  the judge adapter caches on a hash of the request payload — identical bytes
  would return the cached baseline answer, so the arm would measure exactly zero
  by construction rather than by evidence.

It doubles as the quality validator's own positive control: a validator that fails
to pass the sham at ~100% is miscalibrated, and its verdicts on the real biases
cannot be trusted either.
"""

from __future__ import annotations

import io
import random
from typing import Any, Dict, Tuple

from PIL import Image

from edit_judge_bias.bias.base import BiasInjector

#: JPEG quality for the round trip. High enough to be invisible (SSIM 0.999 on
#: real pool images), low enough that the codec always perturbs *something*.
DEFAULT_QUALITY = 95


class ShamInjector(BiasInjector):
    """A visually null perturbation. `bias_strength` is 0.0 — there is no dose."""

    bias_type = "sham"

    def transform(
        self,
        image: Image.Image,
        config: Dict[str, Any],
        rng: random.Random,
        sample: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        quality = int(config.get("quality", DEFAULT_QUALITY))

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as reopened:
            out = reopened.convert("RGB")

        # A perfectly flat region survives the codec untouched. That is fine for a
        # photo but not for a *control*: byte-identical output would be served from
        # the adapter's cache, and the arm would report a zero it never measured.
        # One LSB on one corner pixel is far below both perception and SSIM
        # resolution, and it guarantees the request is a distinct one.
        forced_lsb = False
        if out.tobytes() == image.tobytes():
            r, g, b = out.getpixel((0, 0))
            out.putpixel((0, 0), (r ^ 1, g, b))
            forced_lsb = True

        return out, {
            "quality": quality,
            "codec": "jpeg_roundtrip",
            "forced_lsb": forced_lsb,
            # No dose to report: the point of the arm is that nothing was applied.
            "bias_strength": 0.0,
            "is_placebo": True,
        }
