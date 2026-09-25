"""Judge prompt builders.

`PromptStyle` selects the mitigation variant (§7.3): `vanilla` is a true neutral
baseline, `bias_aware` adds the "ignore watermarks/borders/brightness/..." warning,
`rubric_first` stresses scoring the three criteria before the overall score.

Note: §7.1's printed scoring template bundles the anti-bias sentence into what it
calls the base prompt. We deliberately move that sentence into `bias_aware` so the
`vanilla` baseline carries no debiasing hint — otherwise the mitigation experiment
(§9.5) would have nothing to compare against.
"""

from edit_judge_bias.prompts.scoring_prompt import build_scoring_prompt
from edit_judge_bias.prompts.pairwise_prompt import build_pairwise_prompt
from edit_judge_bias.prompts.styles import PROMPT_STYLES, PromptStyle

__all__ = [
    "build_scoring_prompt",
    "build_pairwise_prompt",
    "PromptStyle",
    "PROMPT_STYLES",
]
