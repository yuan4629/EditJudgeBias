"""JudgeAdapter interface.

Adapters are intentionally thin and provider-agnostic: given a built prompt and an
ordered list of image paths, return the model's **raw** text response. Prompt
construction lives in `prompts/`, parsing in `judges/parser.py`, and orchestration
(saving raw text before parsing, building JudgeResults, resuming) in the runners.

Real network adapters (Milestone 4) subclass this and gate calls behind `--use-api`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass
class JudgeRequest:
    """A single judge call: the prompt plus the images it references, in order."""

    prompt: str
    images: List[Path]
    task: str  # "scoring" | "pairwise"


class JudgeAdapter(ABC):
    """Base class for MLLM judges. Implement `generate`; get score/compare free."""

    #: Stored on every JudgeResult as `judge_model`.
    model_name: str = "base"

    @abstractmethod
    def generate(self, request: JudgeRequest) -> str:
        """Return the model's raw text response for `request` (no parsing)."""

    def score(self, prompt: str, images: Sequence[Path]) -> str:
        """Scoring call: images are [original, edited]."""
        return self.generate(JudgeRequest(prompt, [Path(p) for p in images], "scoring"))

    def compare(self, prompt: str, images: Sequence[Path]) -> str:
        """Pairwise call: images are [original, edited_A, edited_B]."""
        return self.generate(JudgeRequest(prompt, [Path(p) for p in images], "pairwise"))
