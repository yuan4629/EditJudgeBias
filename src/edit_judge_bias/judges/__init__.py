"""Judge adapter layer (Milestone 3, mock-first).

`build_adapter` resolves a judge config dict to an adapter instance. Only the mock
adapter exists now; Milestone 4 registers real OpenAI-compatible adapters here,
gated behind `--use-api`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from edit_judge_bias.judges.base import JudgeAdapter, JudgeRequest
from edit_judge_bias.judges.mock_judge import MockJudgeAdapter
from edit_judge_bias.judges.parser import DEFAULT_SCORE_SCALE

__all__ = ["JudgeAdapter", "JudgeRequest", "MockJudgeAdapter", "build_adapter"]

_API_TYPES = {"openai", "openai_compatible"}


def build_adapter(
    config: Dict[str, Any],
    *,
    use_api: bool = False,
    root: Optional[Path] = None,
) -> JudgeAdapter:
    """Instantiate a judge adapter from a config dict.

    config = {"type": "mock"|"openai", "model_name": ..., ...}. Real API adapters
    require use_api=True so mock runs can never silently hit the network or spend
    money; the key is read from an env var inside the adapter, never from config.
    """
    jtype = config.get("type", "mock")
    if jtype == "mock":
        return MockJudgeAdapter(
            model_name=config.get("model_name", "mock-judge"),
            response_style=config.get("response_style", "valid"),
            seed=int(config.get("seed", 0)),
            score_scale=int(config.get("score_scale", DEFAULT_SCORE_SCALE)),
        )
    if jtype in _API_TYPES:
        if not use_api:
            raise PermissionError(
                f"judge type {jtype!r} makes paid API calls; pass --use-api to allow it"
            )
        from edit_judge_bias.judges.openai_judge import OpenAICompatibleJudge

        return OpenAICompatibleJudge.from_config(config, root=root)
    raise ValueError(f"unknown judge type {jtype!r}")
