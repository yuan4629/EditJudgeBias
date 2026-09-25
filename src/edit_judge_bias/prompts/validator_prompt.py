"""Quality-preservation MLLM validator prompt.

A DISTINCT call that never participates in the main scoring/pairwise experiment.
It judges whether a cosmetic modification changed the *true* editing outcome — not
whether the cosmetic change is visible. Images are supplied to the adapter in the
order [original, edited (A), biased edited (B)].
"""

from __future__ import annotations

_TEMPLATE = """You are a strict validator for an image-editing benchmark. You check whether a \
COSMETIC modification changed the TRUE editing outcome of an edit.

You will receive:
1. The original image (before editing).
2. The editing instruction.
3. Edit A: the edited image.
4. Edit B: the SAME edit after a cosmetic modification (e.g. brightness/saturation \
change, a black border, a corner watermark, or a text band).

Editing instruction: "{instruction}"

Edit B differs from Edit A only by a superficial cosmetic change. Decide whether \
that change altered the ACTUAL editing result with respect to the instruction — NOT \
whether the cosmetic change is merely visible. Treat global brightness/saturation \
shifts, borders, watermarks, and edge text bands as cosmetic; mark a change only if \
they hide or alter the instruction-relevant content.

For each aspect, answer whether it CHANGED from Edit A to Edit B:
- instruction_adherence_changed: does B follow the instruction better/worse than A?
- editing_quality_changed: did the realism/coherence of the edited region change?
- detail_preservation_changed: is instruction-irrelevant content now hidden/altered?
- major_semantic_shift: did the actual scene/content meaningfully change?

"pass" is true ONLY if all four are false (the cosmetic change preserved quality).

Return only valid JSON:
{{
  "instruction_adherence_changed": true/false,
  "editing_quality_changed": true/false,
  "detail_preservation_changed": true/false,
  "major_semantic_shift": true/false,
  "pass": true/false,
  "reason": "<short reason>"
}}"""


# --------------------------------------------------------------------------- #
# WP-A4c — the C-class ablation variant.
# --------------------------------------------------------------------------- #
# WHY IT EXISTS.  The template above carries a construction asymmetry that was only
# noticed in 2026-08-06 and became load-bearing on 2026-08-17.  It tells the validator to
# treat certain artefacts as cosmetic, and every named example is B-CLASS: "global
# brightness/saturation shifts, borders, watermarks, and edge text bands".  The four
# C-class overlays -- a box or arrow drawn on the edit region, a magnified inset in a
# corner, a template caption bar, a sticker outside the edit region -- are NEVER named.
# So the template is, by construction, harsher on C-class cues than on B-class ones.
#
# That was a footnote while no cue was significantly below its floor.  After WP-A3's
# matched design it is not: `zoom_inset` (C-class, the largest overlay) is now
# significantly below its floor on two validators, and "the validator saw the inset and
# judged the content changed" is OBSERVATIONALLY EQUIVALENT to "the edit was destroyed".
# This variant is the only thing that separates them.
#
# ⚠️ DESIGN RULE: everything except the cosmetic-example list is byte-identical to the
# published template.  If any other wording moved, a difference in pass rates would no
# longer be attributable to the asymmetry being tested.  `tests/test_validator_prompt.py`
# asserts that the two templates differ in exactly the intended places.
_C_CLASS_EXAMPLES = (
    "a box or arrow drawn over the edited region, a magnified inset of the edited "
    "region placed in a corner, a caption bar describing the edit, or a small unrelated "
    "sticker placed away from the edited region"
)

def _sub_once(text: str, old: str, new: str) -> str:
    """`str.replace` that REFUSES to no-op.

    Found the hard way while building this variant: the published template uses
    backslash line continuations, so a pattern written with the newlines you see in the
    source file matches nothing -- and `str.replace` then returns the string unchanged
    and says nothing.  The result would have been an ablation that only half happened,
    which is worse than no ablation: the arm would run, cost money, and answer a question
    nobody could state.
    """
    if text.count(old) != 1:
        raise AssertionError(
            f"validator-prompt ablation expected exactly one occurrence of {old[:60]!r}, "
            f"found {text.count(old)} — the published template changed and the variant "
            "must be re-derived rather than silently no-op"
        )
    return text.replace(old, new)


_ABLATION_TEMPLATE = _sub_once(
    _sub_once(
        _TEMPLATE,
        "change, a black border, a corner watermark, or a text band).",
        f"change, a black border, a corner watermark, a text band, or {_C_CLASS_EXAMPLES}).",
    ),
    "Treat global brightness/saturation shifts, borders, watermarks, and edge text "
    "bands as cosmetic; mark a change only if they hide or alter the "
    "instruction-relevant content.",
    "Treat global brightness/saturation shifts, borders, watermarks, edge text bands, "
    f"and annotation overlays ({_C_CLASS_EXAMPLES}) as cosmetic; mark a change only if "
    "they hide or alter the instruction-relevant content.",
)

STYLES = ("default", "c_class_named")


def build_validator_prompt(instruction: str, style: str = "default") -> str:
    """`style="c_class_named"` adds the C-class overlays to the cosmetic-example list.

    The published grid used `default` and must keep using it; the variant exists only for
    the WP-A4c ablation, which re-runs the four C-class cues on the main validator.
    """
    if style not in STYLES:
        raise ValueError(f"unknown validator prompt style {style!r}; expected {STYLES}")
    template = _ABLATION_TEMPLATE if style == "c_class_named" else _TEMPLATE
    return template.format(instruction=instruction)
