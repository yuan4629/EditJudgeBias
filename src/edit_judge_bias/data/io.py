"""JSONL read/write helpers for pydantic records.

Every manifest and every result file in the pipeline is JSONL: one JSON object
per line. Serialization always uses
``model_dump(mode="json", by_alias=True)`` so ``pathlib.Path`` fields render as
strings and aliased fields (e.g. ``pass``) round-trip correctly.

`append_jsonl` exists to support the resumable / checkpoint-as-you-go batch jobs
required of every runner: a completed item can be flushed immediately instead of
buffering the whole batch in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

PathLike = Union[str, Path]


def _to_line(record: BaseModel) -> str:
    """Serialize one record to a single JSON line (no trailing newline)."""
    return json.dumps(
        record.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
    )


def write_jsonl(
    path: PathLike,
    records: Iterable[BaseModel],
    *,
    append: bool = False,
) -> int:
    """Write `records` to `path`, one JSON object per line.

    Creates parent directories as needed. Returns the number of records written.
    With ``append=True`` the file is opened in append mode (used for resumable
    batches); otherwise it is truncated/overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(_to_line(record))
            fh.write("\n")
            count += 1
    return count


def append_jsonl(path: PathLike, record: BaseModel) -> None:
    """Append a single record to `path` (checkpoint a completed batch item)."""
    write_jsonl(path, [record], append=True)


def iter_jsonl(path: PathLike, model: Type[T]) -> Iterator[T]:
    """Lazily parse `path`, validating each non-blank line into `model`.

    Blank lines are skipped so trailing newlines / hand-edited files are tolerated.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield model.model_validate(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - add location context, re-raise
                raise ValueError(
                    f"{path}:{line_no}: failed to parse as {model.__name__}: {exc}"
                ) from exc


def read_jsonl(path: PathLike, model: Type[T]) -> List[T]:
    """Eagerly read all records from `path` into a list of `model` instances."""
    return list(iter_jsonl(path, model))
