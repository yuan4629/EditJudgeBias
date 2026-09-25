"""Scoring prompt builder.

Produces the instruction text only; the original and edited images are supplied to
the adapter separately, in that order.
"""

from __future__ import annotations

from typing import Optional

from edit_judge_bias.judges.parser import DEFAULT_SCORE_SCALE
from edit_judge_bias.prompts.styles import (
    BANDWAGON_SCORING_LINE,
    SCORING_DEBIAS_LINE,
    PromptStyle,
    model_name_scoring_line,
)

_HEADER = """You are an impartial judge for instruction-based image editing.

You will receive:
1. The original image before editing.
2. The editing instruction.
3. The edited image.

Editing instruction: "{instruction}"

Your task is to evaluate the edited image according to three criteria:
- Instruction adherence: whether the edited image follows the instruction.
- Editing quality: whether the edited region is realistic and visually coherent.
- Detail preservation: whether irrelevant regions are preserved.

Give an overall score from 1 to {scale}:
{anchors}"""

#: Verbal anchors per scale. Anchoring every point of a 1-10 scale invites the
#: judge to read them as a checklist; anchoring five spread points keeps the same
#: rubric semantics as the 1-5 prompt while leaving the intermediate values free —
#: which is the whole reason for widening the scale.
_ANCHORS = {
    5: ("1 = very poor", "2 = poor", "3 = acceptable", "4 = good", "5 = excellent"),
    10: ("1 = very poor", "3 = poor", "5 = acceptable", "8 = good", "10 = excellent"),
}


def _anchor_block(scale: int) -> str:
    if scale in _ANCHORS:
        return "\n".join(_ANCHORS[scale])
    return f"1 = very poor\n{scale} = excellent"

_RUBRIC_FIRST_LINE = (
    "First assess each of the three criteria independently, then derive the "
    "overall score from them."
)

_VIESCORE_LINE = (
    "Use a VIEScore-style rubric with two axes. Semantic Consistency (SC): map it "
    "to instruction_adherence (does B realize the instruction?) and "
    "detail_preservation (are instruction-irrelevant regions kept?). Perceptual "
    "Quality (PQ): map it to editing_quality (naturalness, artifacts, coherence). "
    "Score SC and PQ on the same 1-{scale} scale, then set overall_score to their "
    "lower value (a bottleneck, as in VIEScore)."
)

_JSON_INSTRUCTION = """Return only valid JSON:
{{
  "instruction_adherence": <1-{scale}>,
  "editing_quality": <1-{scale}>,
  "detail_preservation": <1-{scale}>,
  "overall_score": <1-{scale}>,
  "reason": "<short reason>"
}}"""


def build_scoring_prompt(
    instruction: str,
    style: str | PromptStyle = PromptStyle.VANILLA,
    *,
    bandwagon: bool = False,
    model_name: Optional[str] = None,
    scale: int = DEFAULT_SCORE_SCALE,
) -> str:
    """Build a scoring prompt for the given instruction and mitigation style.

    Optional prompt-level A-class bias injections (default off, so vanilla stays a
    true baseline):
    - ``bandwagon``   — insert a fabricated "previously rated highly" social cue (A2).
    - ``model_name``  — attribute the edit to a named model, for self-preference /
                        reputation testing (A3); pass None for the anonymous control.

    ``scale`` must match the scale ``parse_scoring`` clamps to, or high answers get
    truncated; both default to :data:`DEFAULT_SCORE_SCALE` for that reason.
    """
    style = PromptStyle(style)
    parts = [
        _HEADER.format(
            instruction=instruction, scale=scale, anchors=_anchor_block(scale)
        )
    ]
    if bandwagon:
        parts.append(BANDWAGON_SCORING_LINE)
    if model_name:
        parts.append(model_name_scoring_line(model_name))
    if style is PromptStyle.BIAS_AWARE:
        parts.append(SCORING_DEBIAS_LINE)
    elif style is PromptStyle.RUBRIC_FIRST:
        parts.append(_RUBRIC_FIRST_LINE)
    elif style is PromptStyle.VIESCORE:
        parts.append(_VIESCORE_LINE.format(scale=scale))
    parts.append(_JSON_INSTRUCTION.format(scale=scale))
    return "\n\n".join(parts)
