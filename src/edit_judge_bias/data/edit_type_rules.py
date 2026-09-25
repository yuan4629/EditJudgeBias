"""Deterministic instruction -> (edit_type, content_category) classification.

I2EBench ships explicit per-category labels, so `build_i2ebench.py` just maps them.
The full-version sources added in 2026-07 (ImagenHub museum, GenAI-Bench
`image_edition`, MagicBrush dev) ship **no edit-type label at all** — yet
`edit_type` and `content_category` are the group keys the §9.4 breakdown is built
on and are required by `SampleRecord`. This module derives them from the
instruction text with ordered keyword/regex rules.

Why rules and not an LLM: they are free, deterministic (so a rebuild is
reproducible, per the seed-controlled convention), and auditable by a reviewer —
and `classify_batch` reports coverage so an unclassifiable tail is visible rather
than silently mislabelled. Callers decide what to do with `None`: skip the row, or
fall back to a configured default.

Rule order is load-bearing: "change the background to a beach" must resolve to
`background`, not `replace`, so the background rule is consulted before the
generic replace rule. Same for style/low-level before `color`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "EDIT_TYPE_RULES",
    "CONTENT_CATEGORY_RULES",
    "classify_edit_type",
    "classify_content_category",
    "classify_batch",
    "ClassificationReport",
]


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


_COLOR_WORD = (
    r"red|orange|yellow|green|blue|purple|pink|brown|black|white|gray|grey|"
    r"golden|silver|beige|teal|turquoise|magenta|cyan|violet|maroon|blonde|blond|"
    r"dark|multi.?colo(?:u)?red"
)

_SCENE_WORD = (
    r"forest|desert|city|beach|ocean|sea|mountains?|park|snow|jungle|countryside|"
    r"space|street|woods|field|garden|kitchen|library|museum|zoo|farm"
)

_MATERIAL_WORD = (
    r"wooden|wood|metal|metallic|cement|concrete|plastic|glass|stone|brick|marble|"
    r"leather|wool|steel|golden|silk|paper"
)

# Ordered (edit_type, pattern). FIRST MATCH WINS — see module docstring.
EDIT_TYPE_RULES: List[Tuple[str, re.Pattern]] = [
    # 1. background: the edit target is the scene/backdrop.
    (
        "background",
        _rx(
            r"\bbackground\b",
            r"\bbackdrop\b",
            r"\bscenery behind\b",
            r"\b(?:change|replace|make|set|swap)\b[^.]{0,30}\b(?:scene|setting|location|environment)\b",
            # Relocate the subject into a new scene: "Could he be in the forest?"
            r"\b(?:be|move|put|place|stand|sit)\b[^.]{0,20}\b(?:in|at|on) the (?:" + _SCENE_WORD + r")\b",
        ),
    ),
    # 2. low-level: global photometric / restoration / style transfer.
    (
        "low-level",
        _rx(
            r"\bblur|\bdeblur|\bsharpen|\bdenois|\bnoise\b|\bgrain\b",
            r"\bhaze|\bfog(?:gy)?\b|\bdehaze",
            r"\brain(?:y|drops)?\b|\bsnow(?:y|flakes)?\b",
            r"\bshadow\b|\breflection\b",
            r"\bbright(?:er|ness)?\b|\bdark(?:er|en)\b|\bdim(?:mer)?\b|\bexposure\b|\bcontrast\b",
            r"\bblack and white\b|\bgrayscale\b|\bgreyscale\b|\bsepia\b|\bmonochrome\b",
            r"\bstyle\b|\bpainting\b|\bsketch\b|\bcartoon\b|\banime\b|\bwatercolor\b|"
            r"\boil painting\b|\bvan gogh\b|\bpicasso\b|\bpixel art\b|\bvintage\b|\bretro\b",
            r"\bmake (?:it|the (?:image|photo|picture)) look like\b",
            r"\bwatermark\b",
            r"\blow.?light\b|\bnight time\b|\bsunset\b(?= lighting)",
        ),
    ),
    # 3. color: recolour an existing object (checked after style so
    #    "make it a black and white photo" is low-level, not color).
    (
        "color",
        _rx(
            r"\brecolo(?:u)?r\b",
            r"\bchange (?:the )?colo(?:u)?r\b",
            r"\bcolo(?:u)?r (?:of|the)\b",
            r"\bpaint\b(?!ing)",
            r"\bmake (?:it|the|his|her|their|its)\b[^.]{0,30}\b(?:" + _COLOR_WORD + r")\b",
            r"\bturn\b[^.]{0,30}\b(?:" + _COLOR_WORD + r")\b",
            r"\b(?:" + _COLOR_WORD + r")\b[^.]{0,15}\b(?:instead|colo(?:u)?r)\b",
        ),
    ),
    # 4. remove.
    (
        "remove",
        _rx(
            r"\bremove\b",
            r"\bdelete\b",
            r"\berase\b",
            r"\btake\b[^.]{0,25}\b(?:out|away|off)\b",
            r"\bget rid of\b",
            r"\bleave nothing\b",
            r"\bwithout the\b",
            r"\bmake\b[^.]{0,20}\bdisappear\b",
            r"\bclear (?:away|out)\b",
            r"\bmake\b[^.]{0,20}\bempty\b",
        ),
    ),
    # 5. replace / swap (before `add`: "replace X with Y" also contains no add cue,
    #    but "change X into Y" must not fall through to add).
    (
        "replace",
        _rx(
            r"\breplace\b",
            r"\bswap\b",
            r"\bsubstitute\b",
            r"\bchange\b[^.]{0,40}\b(?:into|to|for|with)\b",
            r"\bturn\b[^.]{0,40}\binto\b",
            r"\bmake\b[^.]{0,20}\b(?:a|an|the)\b[^.]{0,30}\binstead\b",
            r"\binstead of\b",
        ),
    ),
    # 6. add.
    (
        "add",
        _rx(
            r"\badd\b",
            r"\bput\b",
            r"\bplace\b",
            r"\binsert\b",
            r"\binclude\b",
            r"\bgive (?:him|her|it|them|the)\b",
            r"\blet there be\b",
            r"\bthere should be\b",
            r"\bmake (?:him|her|it|them)\b[^.]{0,20}\b(?:wear|hold|carry)\b",
        ),
    ),
    # ---------------------------------------------------------------------
    # Paraphrase tiers (7-9), appended 2026-07-25.
    #
    # The full-version sources (ImagenHub / GenAI-Bench / MagicBrush dev) all
    # descend from MagicBrush, whose prompts are crowd-written paraphrases rather
    # than imperatives -- "Have the cow wear a hat", "let the bed be wooden",
    # "Make all the grass green". Rules 1-6 matched only 56-69% of them. These
    # tiers sit AFTER the explicit-verb rules on purpose: "remove the man holding
    # a cup" must stay `remove`, not become `add` because it contains "holding".
    # ---------------------------------------------------------------------
    # 7. add by object-introduction verb. Placed before the paraphrase colour
    #    tier because a *newly introduced* coloured object is an `add`, not a
    #    recolour: "Have the woman be wearing a blue tank top" -> add.
    (
        "add",
        _rx(
            r"\b(?:wear|wears|wearing)\b",
            r"\b(?:hold|holds|holding|carry|carries|carrying)\b",
            r"\b(?:eat|eats|eating|drink|drinks|drinking)\b",
            r"\btopped with\b",
            r"\b(?:contain|contains|containing)\b",
            r"\bthere (?:be|is|was|were|should be)\b",
            # "What if the man had a hat?" -- but NOT "Can we have a blue
            # airplane?", which is a recolour, hence the colour lookahead.
            r"\b(?:has|have|had|having)\s+(?:a|an)\s+(?!(?:" + _COLOR_WORD + r")\b)",
            r"\bshould (?:have|contain|include)\b",
            # Introduce a new indefinite entity: "let a herd of sheep block the
            # taxi", "Let a green towel be hung in the bathroom." No colour
            # lookahead here -- "let a <colour> <noun>" still introduces a new
            # object, unlike the "have a blue airplane" recolour form above.
            r"\blet\s+(?:a|an)\s",
            # "let the bowl have chocolate sauce" -- but not "let the cat have
            # blue eyes", which is a recolour.
            r"\blet\s+(?:the|it|this|that|him|her|them)\b[^.]{0,25}\bhave\s+"
            r"(?!(?:" + _COLOR_WORD + r")\b)",
            r"\bcover\b[^.]{0,30}\bwith\b",
            # "A dog should be near the sheep." / "it should be a tennis ball on the glove."
            r"\b(?:should|could|would|might)\s+be\b[^.]{0,40}\b(?:near|next to|beside|nearby|on|in|behind|above|under)\b",
            r"\blet water\b",
            r"\b(?:drawing|picture|print|logo|pattern) of\b",
        ),
    ),
    # 8. colour by paraphrase -- the rule-3 forms demand `make it/the/his/...`,
    #    so "Make all the grass green" and "The bed should be red." fell through.
    (
        "color",
        _rx(
            r"\bmake\s+(?:all\s+)?(?:the|it|this|that|his|her|their|its|one of the|a|an)\b"
            r"[^.]{0,30}\b(?:" + _COLOR_WORD + r")\b",
            r"\b(?:should be|be|have|has|had)\b[^.]{0,15}\b(?:" + _COLOR_WORD + r")\b",
        ),
    ),
    # 9. replace by paraphrase: "let the bed be wooden", "make it a pepperoni
    #    pizza", "Make the zebra a regular horse", "make the ramp cement".
    #    The negative lookahead keeps pose/state edits ("Let the faucet be turned
    #    on.") unclassified rather than mislabelling them -- see module docstring
    #    on the action/pose taxonomy gap.
    (
        "replace",
        _rx(
            r"\blet\s+(?:it|the|this|that|him|her|them)\b[^.]{0,30}\b(?:be|become)\b"
            r"(?!\s+(?:turned|opened|closed|standing|sitting|lying|open|closed))",
            r"\bmake\s+(?:it|the|this|that|him|her|them|one of the)\b[^.]{0,30}\b(?:a|an)\b",
            r"\bmake\s+(?:it|the|this|that)\b[^.]{0,25}\b(?:" + _MATERIAL_WORD + r")\b",
            # Tolerate the crowd-written "into" typo: "turn the mountain in a waterfall".
            r"\bturn\b[^.]{0,40}\bin\s+(?:a|an)\b",
            r"\bnot a\b",
        ),
    ),
]

# Edits the 6-class taxonomy has no slot for: pose, action, facial expression and
# open/closed state ("let the woman cry", "Open the zebra's mouth."). They stay
# deliberately unclassified -- forcing them into add/replace would corrupt the
# 9.4 breakdown. Callers see None and fall back to their configured default with
# `edit_type_source="default"` recorded, so the stratum stays filterable.
TAXONOMY_GAP_NOTE = "action/pose/state edits are out of the 6-class taxonomy by design"

# Ordered (content_category, pattern). FIRST MATCH WINS. `global` is the fallback,
# matching `build_i2ebench.py`'s `default_content_category`.
CONTENT_CATEGORY_RULES: List[Tuple[str, re.Pattern]] = [
    (
        "human",
        _rx(
            r"\b(?:man|men|woman|women|person|people|boy|girl|child|children|kid|baby|"
            r"guy|lady|human|player|rider|worker|chef|doctor|crowd|couple|he|she|his|her)\b",
            r"\b(?:face|hair|hand|arm|leg|shirt|dress|jacket|hat|glasses|shoes)\b",
        ),
    ),
    (
        "animal",
        _rx(
            r"\b(?:dog|puppy|cat|kitten|bird|horse|cow|sheep|goat|pig|bear|elephant|"
            r"giraffe|zebra|lion|tiger|monkey|rabbit|fish|shark|duck|chicken|animal|"
            r"pet|deer|fox|wolf|butterfly|insect)\b",
        ),
    ),
    (
        "scenery",
        _rx(
            r"\b(?:sky|cloud|mountain|beach|ocean|sea|lake|river|forest|tree|field|"
            r"grass|sunset|sunrise|landscape|street|city|building|road|garden|park|"
            r"desert|snowfield|waterfall)\b",
        ),
    ),
    (
        "object",
        _rx(
            r"\b(?:car|truck|bus|bike|bicycle|motorcycle|train|boat|plane|table|chair|"
            r"cup|bottle|plate|bowl|book|phone|laptop|computer|ball|frisbee|umbrella|"
            r"clock|lamp|flower|plant|food|pizza|cake|fruit|apple|banana|box|bag|toy)\b",
        ),
    ),
]


def classify_edit_type(instruction: str) -> Optional[str]:
    """Return one of the 6 edit_type values, or None if no rule matches."""
    text = (instruction or "").strip()
    if not text:
        return None
    for edit_type, pattern in EDIT_TYPE_RULES:
        if pattern.search(text):
            return edit_type
    return None


def classify_content_category(*texts: str) -> str:
    """Return a content_category from the first matching rule; `global` otherwise.

    Accepts several text fields (e.g. instruction + caption) — captions describe
    the scene far better than a terse instruction does, so pass them when the
    source provides them.
    """
    blob = " ".join(t for t in texts if t)
    for category, pattern in CONTENT_CATEGORY_RULES:
        if pattern.search(blob):
            return category
    return "global"


@dataclass
class ClassificationReport:
    """Coverage of a rule pass, so an unclassified tail stays visible."""

    n: int = 0
    n_classified: int = 0
    edit_types: Counter = None  # type: ignore[assignment]
    content_categories: Counter = None  # type: ignore[assignment]
    unclassified: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.edit_types = self.edit_types or Counter()
        self.content_categories = self.content_categories or Counter()
        self.unclassified = self.unclassified if self.unclassified is not None else []

    @property
    def coverage(self) -> float:
        return self.n_classified / self.n if self.n else 0.0

    def summary(self) -> str:
        return (
            f"classified={self.n_classified}/{self.n} ({self.coverage:.1%}) "
            f"edit_types={dict(self.edit_types)} "
            f"content={dict(self.content_categories)} "
            f"unclassified_examples={self.unclassified[:5]}"
        )


def classify_batch(
    instructions: Iterable[str],
    captions: Optional[Iterable[str]] = None,
) -> Tuple[List[Optional[str]], List[str], ClassificationReport]:
    """Classify many instructions, returning (edit_types, content_categories, report)."""
    instructions = list(instructions)
    caps = list(captions) if captions is not None else [""] * len(instructions)
    if len(caps) < len(instructions):
        caps += [""] * (len(instructions) - len(caps))

    report = ClassificationReport()
    edit_types: List[Optional[str]] = []
    categories: List[str] = []
    for instr, cap in zip(instructions, caps):
        report.n += 1
        et = classify_edit_type(instr)
        if et is None:
            report.unclassified.append(instr)
        else:
            report.n_classified += 1
            report.edit_types[et] += 1
        cc = classify_content_category(instr, cap)
        report.content_categories[cc] += 1
        edit_types.append(et)
        categories.append(cc)
    return edit_types, categories, report


def rules_as_config() -> Dict[str, List[str]]:
    """Expose the rule patterns for logging into a run manifest (traceability)."""
    return {et: [pattern.pattern] for et, pattern in EDIT_TYPE_RULES}
