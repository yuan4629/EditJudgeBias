"""Data layer: pydantic record schemas and JSONL i/o.

Later milestones add: build_from_csv, build_i2ebench, build_pairs, validate_manifest.
"""

from edit_judge_bias.data.schema import (
    BiasAppliedTo,
    BiasedRecord,
    ContentCategory,
    EditType,
    JudgeResult,
    PairRecord,
    QualityPreservation,
    QualityValidationResult,
    SampleMetadata,
    SampleRecord,
    TaskType,
)
from edit_judge_bias.data.io import (
    append_jsonl,
    iter_jsonl,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    # schema
    "BiasAppliedTo",
    "BiasedRecord",
    "ContentCategory",
    "EditType",
    "JudgeResult",
    "PairRecord",
    "QualityPreservation",
    "QualityValidationResult",
    "SampleMetadata",
    "SampleRecord",
    "TaskType",
    # io
    "append_jsonl",
    "iter_jsonl",
    "read_jsonl",
    "write_jsonl",
]
