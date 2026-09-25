"""Pydantic record schemas for every JSONL artifact in the pipeline.

Four primary record types trace every downstream artifact back to an original
`sample_id`:

- SampleRecord            — one edited-image output.
- PairRecord              — two edited outputs of the same (original, instruction).
- BiasedRecord            — one bias injection over a base sample (never overwrites).
- JudgeResult             — a scoring or pairwise verdict from a judge model.
- QualityValidationResult — the independent MLLM quality-preservation check (§6.2).

Conventions: pydantic for all schemas, `pathlib.Path` for
paths. JSON serialization goes through `model_dump(mode="json", by_alias=True)` in
`io.py`, which renders `Path` as a POSIX-style string and respects field aliases
(notably the reserved word `pass` -> field `passed`).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
)

# A pathlib.Path that always serializes to a POSIX-style string in JSON, so
# manifests are identical across Windows/Unix. In Python it stays a real Path.
PathField = Annotated[
    Path,
    PlainSerializer(lambda p: Path(p).as_posix(), return_type=str, when_used="json"),
]


# --------------------------------------------------------------------------- #
# Controlled vocabularies (these group keys drive the per-group breakdowns)    #
# --------------------------------------------------------------------------- #
class EditType(str, Enum):
    """Edit-type taxonomy. Dataset-specific categories map onto these in M1."""

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    COLOR = "color"
    BACKGROUND = "background"
    LOW_LEVEL = "low-level"


class ContentCategory(str, Enum):
    """Content category of the original image."""

    HUMAN = "human"
    ANIMAL = "animal"
    OBJECT = "object"
    SCENERY = "scenery"
    GLOBAL = "global"


class BiasAppliedTo(str, Enum):
    """Where a bias is injected: the edited pixels, or the judge protocol/prompt."""

    EDITED_IMAGE = "edited_image"
    JUDGE_PROMPT = "judge_prompt"


class TaskType(str, Enum):
    """Which judge protocol produced a JudgeResult."""

    SCORING = "scoring"
    PAIRWISE = "pairwise"


# --------------------------------------------------------------------------- #
# Shared base                                                                  #
# --------------------------------------------------------------------------- #
class _Record(BaseModel):
    """Common config for every record type.

    - ``populate_by_name`` so fields can be set by their Python name *or* alias.
    - ``extra="forbid"`` to catch typos / drifted schemas early in a research
      pipeline where a silently-dropped field would corrupt analysis.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# --------------------------------------------------------------------------- #
# SampleRecord (§3.1)                                                          #
# --------------------------------------------------------------------------- #
class SampleMetadata(BaseModel):
    """Per-sample auxiliary info. All optional; defaults are null-equivalent."""

    model_config = ConfigDict(extra="allow")

    has_mask: bool = False
    mask_path: Optional[PathField] = None
    original_width: Optional[int] = None
    original_height: Optional[int] = None


class SampleRecord(_Record):
    """One edited-image output: the atomic unit of the benchmark."""

    sample_id: str
    source_dataset: str
    edit_type: EditType
    content_category: ContentCategory
    original_image_path: PathField
    instruction: str
    edit_model: str
    edited_image_path: PathField
    reference_image_path: Optional[PathField] = None
    human_score: Optional[float] = None
    auto_score: Optional[float] = None
    metadata: SampleMetadata = Field(default_factory=SampleMetadata)


# --------------------------------------------------------------------------- #
# PairRecord (§3.2)                                                            #
# --------------------------------------------------------------------------- #
class PairRecord(_Record):
    """Two edited outputs of the same (original image, instruction)."""

    pair_id: str
    sample_id_a: str
    sample_id_b: str
    original_image_path: PathField
    instruction: str
    edited_image_a_path: PathField
    edited_image_b_path: PathField
    edit_model_a: str
    edit_model_b: str
    ground_truth_preference: Optional[str] = None
    pair_quality_gap: Optional[float] = None
    source_dataset: str
    edit_type: EditType


# --------------------------------------------------------------------------- #
# BiasedRecord (§3.3)                                                          #
# --------------------------------------------------------------------------- #
class QualityPreservation(BaseModel):
    """Quality-preservation evidence for one biased image (§6.1/§6.2).

    Populated incrementally: automatic metrics first, then the MLLM validator,
    then optional human validation. All fields stay null until measured.
    """

    model_config = ConfigDict(extra="forbid")

    lpips: Optional[float] = None
    ssim: Optional[float] = None
    clip_image_similarity: Optional[float] = None
    mllm_validation_pass: Optional[bool] = None
    human_validation_pass: Optional[bool] = None


class BiasedRecord(_Record):
    """One bias injection over a base sample. Originals are never overwritten."""

    biased_id: str
    base_sample_id: str
    bias_type: str
    bias_strength: float
    biased_image_path: PathField
    bias_applied_to: BiasAppliedTo = BiasAppliedTo.EDITED_IMAGE
    # Injector-applied parameters worth keeping for traceability and later checks:
    # e.g. overlay_text (for the §6.1 OCR check), padding_ratio, output image size.
    bias_params: dict = Field(default_factory=dict)
    quality_preservation: QualityPreservation = Field(
        default_factory=QualityPreservation
    )


# --------------------------------------------------------------------------- #
# JudgeResult (§12.1 scoring / §12.2 pairwise — unified)                       #
# --------------------------------------------------------------------------- #
class JudgeResult(_Record):
    """A scoring or pairwise verdict.

    `task_type` disambiguates which block of fields is meaningful. Raw model
    output is always persisted at `raw_response_path` *before* parsing, so a
    parse failure (`parse_success=False`) never loses data (§ "Raw vs parsed").
    """

    result_id: str
    judge_model: str
    task_type: TaskType
    prompt_type: str
    raw_response_path: PathField
    parse_success: bool

    # Provenance of what was judged. For an unbiased judgment, bias_type is null.
    bias_type: Optional[str] = None
    biased_id: Optional[str] = None
    # Parameters of a *prompt-level* (A-class) bias, e.g. {"bandwagon": true} or
    # {"model_name": "MagicBrush", "model_name_source": "true_name"}. Image-level
    # biases keep their parameters on the BiasedRecord instead, reached via
    # `biased_id`, so this stays empty for them.
    bias_params: dict = Field(default_factory=dict)

    # Scoring branch (task_type == "scoring").
    sample_id: Optional[str] = None
    overall_score: Optional[int] = None
    instruction_adherence: Optional[int] = None
    editing_quality: Optional[int] = None
    detail_preservation: Optional[int] = None
    # Sum of the three dimensions — the primary analysis variable, far finer than
    # `overall_score` (measured on pilot responses: 3 distinct levels -> 7 for
    # gemini-3.5-flash, 5 -> 12 for gpt-5.5).
    fine_score: Optional[int] = None
    # Top of the rating scale this row was produced on. The pilot ran 1-5 and the
    # main grid runs 1-10; without this field the two would average together into
    # a meaningless column. Optional so archived pilot rows still load.
    score_scale: Optional[int] = None

    # Pairwise branch (task_type == "pairwise").
    pair_id: Optional[str] = None
    biased_side: Optional[str] = None
    winner: Optional[str] = None

    reason: Optional[str] = None
    parse_error: Optional[str] = None
    created_at: Optional[str] = None


# --------------------------------------------------------------------------- #
# QualityValidationResult (§6.2)                                              #
# --------------------------------------------------------------------------- #
class QualityValidationResult(_Record):
    """Output of the independent MLLM quality-preservation validator (§6.2).

    This validator never participates in the main judging experiment. `pass` is a
    Python keyword, so the field is `passed` with a "pass" (de)serialization alias.
    """

    # Optional so a parse failure can still be recorded (parse_success=False);
    # for a successful validation all five carry the model's verdict.
    instruction_adherence_changed: Optional[bool] = None
    editing_quality_changed: Optional[bool] = None
    detail_preservation_changed: Optional[bool] = None
    major_semantic_shift: Optional[bool] = None
    passed: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("pass", "passed"),
        serialization_alias="pass",
    )
    reason: str = ""

    # Traceability (not in the raw validator JSON, attached by the runner).
    biased_id: Optional[str] = None
    base_sample_id: Optional[str] = None
    validator_model: Optional[str] = None
    raw_response_path: Optional[PathField] = None
    parse_success: bool = True
    parse_error: Optional[str] = None
    ssim: Optional[float] = None
