"""Robust parsing of judge JSON outputs.

MLLMs often wrap JSON in prose or ```json fences, or emit minor noise. These
parsers extract and validate the expected object, **never raise**, and on failure
return a result with `success=False` and an `error` message so the runner can
persist the raw response and continue (the batch must not abort on one bad parse).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from `text`.

    Tries, in order: a ```json fenced block, a direct parse, then the first
    balanced ``{...}`` span. Returns the dict, or None if nothing parses.
    """
    if not text:
        return None

    candidates = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text.strip())
    span = _first_balanced_object(text)
    if span is not None:
        candidates.append(span)

    for cand in candidates:
        for text_try in (cand, _drop_trailing_commas(cand)):
            try:
                obj = json.loads(text_try)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _drop_trailing_commas(text: Optional[str]) -> Optional[str]:
    """Repair `{"a": 1,}` — a comma before the closing brace, which JSON rejects.

    Measured: gemini-3.5-flash emitted exactly this once in the anchor arm (a complete,
    well-formed answer whose only flaw was the comma) and the row was written as a
    parse failure. Retrying costs nothing — an identical payload is served from the
    response cache — but only if the parser can read the answer the second time.
    """
    if not isinstance(text, str):
        return text
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the first brace-balanced substring, respecting strings/escapes."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


#: Top of the judge's rating scale. The pilot ran on 1-5 and it was too coarse to
#: carry claim B: measured over 480 pilot judgments gemini-3.5-flash used only 3 of
#: the 5 levels (64% of its answers were a bare "2") and gpt-5.5 emitted a 5 once,
#: so a Spearman against EBench's continuous MOS was attenuated by ties before the
#: bias was even injected. The main grid therefore runs on 1-10. `score_scale` is
#: stamped on every JudgeResult so 1-5 pilot rows can never be silently averaged
#: together with 1-10 rows.
DEFAULT_SCORE_SCALE = 10

#: The scale the pilot (M3-M6) ran on. Kept as a named constant so re-parsing the
#: archived pilot responses stays a one-word change rather than a magic number.
PILOT_SCORE_SCALE = 5


def _coerce_score(value, *, scale: int = DEFAULT_SCORE_SCALE) -> Optional[int]:
    """Coerce a 1..`scale` score to a clamped int, or None if not numeric.

    Clamping is to the *caller's* scale: hard-coding 5 here while the prompt asks
    for 1-10 would silently collapse every 6-10 answer onto 5.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(1, min(scale, int(round(value))))
    if isinstance(value, str):
        m = re.search(r"-?\d+(\.\d+)?", value)
        if m:
            return max(1, min(scale, int(round(float(m.group())))))
    return None


@dataclass
class ParsedScoring:
    success: bool
    overall_score: Optional[int] = None
    instruction_adherence: Optional[int] = None
    editing_quality: Optional[int] = None
    detail_preservation: Optional[int] = None
    #: Sum of the three dimensions (3..3*scale) — the analysis variable. It exists
    #: because it is markedly finer-grained than `overall_score` at no extra cost:
    #: on the same pilot responses gemini went from 3 distinct levels to 7,
    #: gpt-5.5 from 5 to 12, gpt-4o-viescore from 5 to 13.
    fine_score: Optional[int] = None
    score_scale: int = DEFAULT_SCORE_SCALE
    reason: str = ""
    error: str = ""


@dataclass
class ParsedPairwise:
    success: bool
    winner: Optional[str] = None  # "A" | "B" | "Tie"
    reason: str = ""
    error: str = ""


def parse_scoring(raw: str, *, scale: int = DEFAULT_SCORE_SCALE) -> ParsedScoring:
    obj = extract_json_object(raw)
    if obj is None:
        return ParsedScoring(
            success=False, score_scale=scale, error="no JSON object found in response"
        )
    overall = _coerce_score(obj.get("overall_score"), scale=scale)
    if overall is None:
        return ParsedScoring(
            success=False,
            score_scale=scale,
            error="missing or non-numeric 'overall_score'",
        )
    dims = [
        _coerce_score(obj.get(key), scale=scale)
        for key in ("instruction_adherence", "editing_quality", "detail_preservation")
    ]
    return ParsedScoring(
        success=True,
        overall_score=overall,
        instruction_adherence=dims[0],
        editing_quality=dims[1],
        detail_preservation=dims[2],
        # Only defined when all three dimensions parsed: a sum over a missing term
        # would sit on a different scale than its neighbours in the same column.
        fine_score=sum(dims) if all(d is not None for d in dims) else None,
        score_scale=scale,
        reason=str(obj.get("reason", "")),
    )


_WINNER_MAP = {
    "a": "A", "image a": "A", "edited image a": "A", "first": "A",
    "b": "B", "image b": "B", "edited image b": "B", "second": "B",
    "tie": "Tie", "equal": "Tie", "draw": "Tie", "neither": "Tie", "both": "Tie",
}


def _normalize_winner(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in {"a", "b", "tie"}:
        return key.upper() if key != "tie" else "Tie"
    return _WINNER_MAP.get(key)


def parse_pairwise(raw: str) -> ParsedPairwise:
    obj = extract_json_object(raw)
    if obj is None:
        return ParsedPairwise(success=False, error="no JSON object found in response")
    winner = _normalize_winner(obj.get("winner"))
    if winner is None:
        return ParsedPairwise(
            success=False, error=f"missing or unrecognized 'winner': {obj.get('winner')!r}"
        )
    return ParsedPairwise(success=True, winner=winner, reason=str(obj.get("reason", "")))


_TRUE = {"true", "yes", "1", "y", "t", "changed"}
_FALSE = {"false", "no", "0", "n", "f", "unchanged", "none"}


def _coerce_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return None


@dataclass
class ParsedValidation:
    success: bool
    instruction_adherence_changed: Optional[bool] = None
    editing_quality_changed: Optional[bool] = None
    detail_preservation_changed: Optional[bool] = None
    major_semantic_shift: Optional[bool] = None
    passed: Optional[bool] = None
    reason: str = ""
    error: str = ""


@dataclass
class ParsedAttributeValidation:
    """Verdict on one D-class counterfactual pair (fairness track).

    Deliberately NOT folded into `ParsedValidation`: that one records four *changed* flags
    where every True is bad, while here `attribute_flipped` True is REQUIRED and False is the
    failure. Sharing the container would invite a sign error in whichever metric read it
    second.
    """

    success: bool
    person_legible: Optional[bool] = None
    attribute_flipped: Optional[bool] = None
    scene_preserved: Optional[bool] = None
    passed: Optional[bool] = None
    reason: str = ""
    error: str = ""


@dataclass
class ParsedConstructScreen:
    """Verdict on ONE image's construct validity for the D-class skin-lightness arm.

    Separate from `ParsedValidation` and `ParsedAttributeValidation` for the same reason those
    two are separate from each other: here every True is REQUIRED, whereas `ParsedValidation`
    records four *changed* flags where every True is bad. Sharing a container across those
    polarities is how a sign error gets into whichever metric reads it second.
    """

    success: bool
    is_photograph: Optional[bool] = None
    single_subject: Optional[bool] = None
    no_minor: Optional[bool] = None
    skin_visible: Optional[bool] = None
    passed: Optional[bool] = None
    reason: str = ""
    error: str = ""


def parse_construct_screen(raw: str) -> ParsedConstructScreen:
    """Parse the construct-screen JSON. `pass` is DERIVED, and a missing field never passes.

    ★ The derivation ignores any `pass` the model volunteered when a field contradicts it.
    All four questions must be positively observed -- "there is no child in this photograph"
    is a claim, not a default -- so an absent or unparseable field has to cost the scene. The
    asymmetry is deliberate: over-rejecting shrinks n, under-rejecting puts a minor in a
    dataset of deliberately manipulated skin tones, and only one of those is recoverable.
    """
    obj = extract_json_object(raw)
    if obj is None:
        return ParsedConstructScreen(success=False, error="no JSON object found in response")
    fields = {
        k: _coerce_bool(obj.get(k))
        for k in ("is_photograph", "single_subject", "no_minor", "skin_visible")
    }
    if all(v is None for v in fields.values()):
        return ParsedConstructScreen(success=False, error="no recognizable screen fields")
    return ParsedConstructScreen(
        success=True,
        **fields,
        passed=all(v is True for v in fields.values()),
        reason=str(obj.get("reason", "")),
    )


def parse_attribute_validation(raw: str) -> ParsedAttributeValidation:
    """Parse the attribute-pair validator JSON. `pass` is derived if omitted.

    Derivation is strict: a missing field cannot count as satisfied, because two of the three
    conditions are things that must be positively observed (a legible person, an actual flip).
    Treating an absent field as True would silently admit unusable pairs.
    """
    obj = extract_json_object(raw)
    if obj is None:
        return ParsedAttributeValidation(success=False, error="no JSON object found in response")
    fields = {
        k: _coerce_bool(obj.get(k))
        for k in ("person_legible", "attribute_flipped", "scene_preserved")
    }
    if all(v is None for v in fields.values()) and "pass" not in obj:
        return ParsedAttributeValidation(success=False, error="no recognizable validation fields")
    passed = _coerce_bool(obj.get("pass"))
    if passed is None:
        passed = all(v is True for v in fields.values())
    return ParsedAttributeValidation(
        success=True,
        person_legible=fields["person_legible"],
        attribute_flipped=fields["attribute_flipped"],
        scene_preserved=fields["scene_preserved"],
        passed=passed,
        reason=str(obj.get("reason", "")),
    )


def parse_validation(raw: str) -> ParsedValidation:
    """Parse the §6.2 quality-validator JSON. `pass` is derived if omitted."""
    obj = extract_json_object(raw)
    if obj is None:
        return ParsedValidation(success=False, error="no JSON object found in response")
    changed = {
        k: _coerce_bool(obj.get(k))
        for k in (
            "instruction_adherence_changed",
            "editing_quality_changed",
            "detail_preservation_changed",
            "major_semantic_shift",
        )
    }
    if all(v is None for v in changed.values()) and "pass" not in obj:
        return ParsedValidation(success=False, error="no recognizable validation fields")
    passed = _coerce_bool(obj.get("pass"))
    if passed is None:
        # Derive: pass iff no aspect changed (None treated as "not changed").
        passed = not any(v is True for v in changed.values())
    return ParsedValidation(
        success=True,
        instruction_adherence_changed=changed["instruction_adherence_changed"],
        editing_quality_changed=changed["editing_quality_changed"],
        detail_preservation_changed=changed["detail_preservation_changed"],
        major_semantic_shift=changed["major_semantic_shift"],
        passed=passed,
        reason=str(obj.get("reason", "")),
    )
