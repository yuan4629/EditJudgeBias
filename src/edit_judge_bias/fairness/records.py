"""Record schema for the fairness track (kept separate from the main §3 schema)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from edit_judge_bias.data.schema import PathField


class AttributeEditRecord(BaseModel):
    """One counterfactual attribute edit of an original image (scheme alpha).

    ``counterfactual_image_path`` is the original with a single demographic
    attribute flipped; the pair of variants for one base sample drives the
    attribute-score-gap metric. ``instruction``/``edit_model`` are carried so the
    downstream paired-edit + judge steps can be run later.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    edit_id: str
    base_sample_id: str
    attribute: str
    variant_label: str
    prompt: str
    original_image_path: PathField
    counterfactual_image_path: PathField
    edit_model: str
    content_category: str
    instruction: str
    editor_model: str = "qwen-image-edit"
    success: bool = True
    message: str = ""


class CounterfactualEditRecord(BaseModel):
    """Step 4: the original instruction re-run on one counterfactual original.

    ``render_index`` is 1 for the study render and 2+ for the editor-noise floor (the same
    prompt on the same image again). gpt-image-2 re-renders the whole frame, so two runs of
    an identical request differ; that spread is D's analogue of `sham` + retest and is the
    floor an attribute gap has to clear. ``render_id`` carries the suffix only from 2 on, so
    adding the null control later does not renumber renders already on disk.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    render_id: str
    edit_id: str
    base_sample_id: str
    attribute: str
    variant_label: str
    render_index: int = 1
    instruction: str
    counterfactual_image_path: PathField
    edited_image_path: PathField
    editor_model: str = "gpt-image-2"


class AttributeValidationResult(BaseModel):
    """Validator verdict on one counterfactual PAIR (D's analogue of §6).

    Keyed by ``pair_key`` = ``{base_sample_id}__{attribute}``, because the unit that passes or
    fails is the pair, not a single image: a scene whose flip did not take, or whose scene
    moved, is unusable from BOTH sides and must be dropped as a pair. Dropping one side would
    leave an unpaired variant that no paired metric can use anyway.

    ``is_control`` marks the false-flip floor rows, where both members are the SAME image and
    a sound validator must answer ``attribute_flipped=False``. Whatever rate it reports there
    is its rubber-stamp floor, and the measured flip rate has to be read against it — the same
    role `sham` plays for the quality validators.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    pair_key: str
    base_sample_id: str
    attribute: str
    label_a: str
    label_b: str
    is_control: bool = False
    person_legible: Optional[bool] = None
    attribute_flipped: Optional[bool] = None
    scene_preserved: Optional[bool] = None
    passed: Optional[bool] = None
    reason: str = ""
    validator_model: str = ""
    raw_response_path: Optional[PathField] = None
    parse_success: bool = True
    parse_error: Optional[str] = None


class ConstructScreenResult(BaseModel):
    """Verdict on ONE ORIGINAL image's construct validity for the skin-lightness arm.

    Keyed by `sample_id`, not by a pair: this screen runs BEFORE any manipulation exists and
    asks only about the source photograph, so the unit is the scene. It answers the four
    questions the geometric pool screens provably cannot -- measured on a seeded random sample
    of 72 in-band OmniEdit scenes, 30 (42%) fail on exactly these grounds while passing every
    numeric threshold: multi-subject frames and collages, non-photographic subjects, theatrical
    face paint, and minors whose instructions never mention a child.

    `is_control` marks the FALSE-REJECT FLOOR rows. Unlike the pair validator's floor (the same
    image twice, where the honest answer is False), the control here is a set of images already
    judged usable by eye, where the honest answer is True -- so what is being measured is how
    often the screen throws away a good scene. A screen that rejects everything would otherwise
    look maximally safe while quietly destroying n, and this study has already had one
    validator (glm-4v, 13.6% false-flag rate) fail exactly that test.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    sample_id: str
    source_dataset: str = ""
    is_control: bool = False
    is_photograph: Optional[bool] = None
    single_subject: Optional[bool] = None
    no_minor: Optional[bool] = None
    skin_visible: Optional[bool] = None
    passed: Optional[bool] = None
    reason: str = ""
    validator_model: str = ""
    raw_response_path: Optional[PathField] = None
    parse_success: bool = True
    parse_error: Optional[str] = None


class AttributeInjectionRecord(BaseModel):
    """One DETERMINISTIC attribute injection, applied to one member of one pair.

    Distinct from `AttributeEditRecord` on purpose. That record describes a *generative*
    counterfactual: an API call, a prompt, one output image per (scene, attribute, variant).
    This one describes a *constructed* manipulation, and it therefore has to carry things a
    generative record has no notion of:

    * `applied_to` -- the same injection is applied to BOTH the original and the dataset's
      edited image, so the pair the judge sees is (original_variant, edited_variant). A
      generative record has only one image.
    * `mask_sha256` -- the auditable proof that the two arms of a scene shared ONE mask, which
      is what makes segmentation error common-mode instead of a difference between arms.
    * `role` -- `study` rows enter the gap; `sham` / `sham_nonskin` are floors and are reported
      as their own rows, never inside the study's BH family.
    * the achieved dose (`achieved_delta_ita`, `achieved_bin`, `clipped_frac`) as opposed to
      the requested one, because the solver is allowed to fail and must say so.
    * `max_outside_diff` -- the integer that replaced the MLLM `scene_preserved` question, on
      which two auditors reached only Cohen's kappa +0.130.

    ⚠️ `attempts_used` exists for the generative arm and is deliberately NOT `render_index`:
    that field means "editor-noise repeat" and `compute_attribute_gaps` branches on
    `render_index != 1`, so an attempt-2 row landing there would be silently dropped or
    double-counted.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    injection_id: str
    base_sample_id: str
    attribute: str
    variant_label: str
    role: str                       # study | sham | sham_nonskin
    applied_to: str                 # original | edited
    source_image_path: PathField
    injected_image_path: PathField
    instruction: str
    source_dataset: str
    edit_type: str
    edit_model: str
    mask_kind: str                  # skin | nonskin | head
    mask_sha256: str
    mask_frac: float
    core_frac: float = 0.0
    person_frac: float = 0.0
    face_frac: float = 0.0
    person_method: str = ""
    requested_delta_ita: Optional[float] = None
    achieved_delta_ita: Optional[float] = None
    ita_before: Optional[float] = None
    ita_after: Optional[float] = None
    target_bin: str = ""
    achieved_bin: str = ""
    clipped_frac: float = 0.0
    delta_e00_mean: Optional[float] = None
    delta_e00_mass: Optional[float] = None
    max_outside_diff: int = 0
    max_inside_diff: int = 0
    outside_mask_identical: bool = True
    #: True when the source had to be resampled to the original's size before injection. The
    #: pair members must share a pixel grid (the mask comes from the original), and 12 of the
    #: 31 D-S scenes ship a 500x500 or 512x512 original against a larger edited image. The
    #: construction gate MUST apply the identical resample before re-verifying locality, or it
    #: fails those rows on a shape mismatch that is not a defect -- measured: exactly 24 rows,
    #: i.e. 12 scenes x 2 study arms, all of them false alarms.
    source_resized: bool = False
    #: Share of the applied (original-derived) mask that is still SKIN on the edited member.
    #: `None` on original-member rows, where it would be 1.0 by construction, and `None` when
    #: no skin mask could be computed on the edited image -- a missing measurement, not a pass.
    #: Median 0.971 over 60 sampled OmniEdit pairs; the tail is what the gate exists for.
    edited_mask_skin_coverage: Optional[float] = None
    forced_lsb: bool = False
    attempts_used: int = 1
    injector: str = ""
    seed: int = 42
    success: bool = True
    message: str = ""
