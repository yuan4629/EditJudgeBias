"""Pairwise prompt builder.

The original image, edited image A, and edited image B are supplied to the adapter
separately, in that order.
"""

from __future__ import annotations

from typing import Optional

from edit_judge_bias.prompts.styles import (
    PAIRWISE_DEBIAS_LINE,
    PromptStyle,
    bandwagon_pairwise_line,
    model_name_pairwise_line,
)

_HEADER = """You are an impartial judge for instruction-based image editing.

You will receive:
1. The original image before editing.
2. The editing instruction.
3. Edited image A.
4. Edited image B.

Editing instruction: "{instruction}"

Choose which edited image better satisfies the instruction while preserving \
irrelevant details and maintaining visual quality.

You may choose:
- "A"
- "B"
- "Tie\""""

_RUBRIC_FIRST_LINE = (
    "Compare the two images on instruction adherence, editing quality, and detail "
    "preservation before deciding."
)

_JSON_INSTRUCTION = """Return only valid JSON:
{{
  "winner": "A" | "B" | "Tie",
  "reason": "<short reason>"
}}"""


def build_pairwise_prompt(
    instruction: str,
    style: str | PromptStyle = PromptStyle.VANILLA,
    *,
    bandwagon_target: Optional[str] = None,
    model_name_a: Optional[str] = None,
    model_name_b: Optional[str] = None,
) -> str:
    """Build a pairwise prompt for the given instruction and mitigation style.

    Optional prompt-level A-class bias injections (default off):
    - ``bandwagon_target`` — "A"/"B": fabricated majority preference for that side (A2).
    - ``model_name_a`` / ``model_name_b`` — attribute each image to a named model for
                        self-preference / reputation testing (A3).
    """
    style = PromptStyle(style)
    parts = [_HEADER.format(instruction=instruction)]
    if bandwagon_target in ("A", "B"):
        parts.append(bandwagon_pairwise_line(bandwagon_target))
    if model_name_a or model_name_b:
        parts.append(model_name_pairwise_line(model_name_a, model_name_b))
    if style is PromptStyle.BIAS_AWARE:
        parts.append(PAIRWISE_DEBIAS_LINE)
    elif style is PromptStyle.RUBRIC_FIRST:
        parts.append(_RUBRIC_FIRST_LINE)
    parts.append(_JSON_INSTRUCTION.format())
    return "\n\n".join(parts)
