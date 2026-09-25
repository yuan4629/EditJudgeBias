"""MockJudgeAdapter — deterministic, offline judge for mock-first development.

Produces well-formed (or, on request, deliberately messy) responses derived from a
hash of the prompt, so runs are reproducible and cacheable without any network or
API key. It is blind to image *content* (it only sees paths), so it cannot fake a
bias effect — that is intentional: bias signal must come from real judges (M4).

`response_style` exercises the parser's robustness:
    valid   -> bare JSON
    fenced  -> ```json ... ``` block
    prose   -> JSON embedded in surrounding text
    invalid -> no JSON at all (forces a parse_error)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from edit_judge_bias.judges.base import JudgeAdapter, JudgeRequest
from edit_judge_bias.judges.parser import DEFAULT_SCORE_SCALE


@dataclass
class MockJudgeAdapter(JudgeAdapter):
    model_name: str = "mock-judge"
    response_style: str = "valid"
    seed: int = 0
    #: Must track the scale the runner asks for in the prompt, or a mock dry-run
    #: would exercise a range the real grid never sees.
    score_scale: int = DEFAULT_SCORE_SCALE

    def _rng_int(self, *parts: str) -> int:
        h = hashlib.sha256(("|".join((str(self.seed), *parts))).encode("utf-8"))
        return int.from_bytes(h.digest()[:8], "big")

    def _payload(self, request: JudgeRequest) -> dict:
        base = self._rng_int(request.task, request.prompt)
        if request.task == "pairwise":
            winner = ["A", "B", "Tie"][base % 3]
            return {"winner": winner, "reason": "mock pairwise verdict"}
        if request.task == "validation":
            # Mostly "preserved" (pass), with occasional flagged change for variety.
            changed = (base % 5) == 0
            return {
                "instruction_adherence_changed": False,
                "editing_quality_changed": changed,
                "detail_preservation_changed": False,
                "major_semantic_shift": False,
                "pass": not changed,
                "reason": "mock validation verdict",
            }
        # scoring: derive four 1..score_scale scores deterministically
        n = max(1, int(self.score_scale))
        ia = 1 + (base % n)
        eq = 1 + ((base >> 3) % n)
        dp = 1 + ((base >> 6) % n)
        overall = round((ia + eq + dp) / 3)
        return {
            "instruction_adherence": ia,
            "editing_quality": eq,
            "detail_preservation": dp,
            "overall_score": overall,
            "reason": "mock scoring verdict",
        }

    def generate(self, request: JudgeRequest) -> str:
        if self.response_style == "invalid":
            return "I cannot produce a structured judgment for this input."
        body = json.dumps(self._payload(request), ensure_ascii=False)
        if self.response_style == "fenced":
            return f"```json\n{body}\n```"
        if self.response_style == "prose":
            return f"Sure, here is my evaluation:\n{body}\nLet me know if you need more."
        return body
