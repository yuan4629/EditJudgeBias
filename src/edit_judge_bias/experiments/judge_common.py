"""Shared helpers for the judge runners (Milestone 3).

Centralizes the cross-cutting concerns both runners need: stable result ids,
saving the raw model response *before* parsing, resuming from an existing manifest,
and routing parsed results to raw_judgments/ (unbiased) vs biased_judgments/.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Set, TypeVar

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import slug, to_rel_posix
from edit_judge_bias.data.schema import JudgeResult
from edit_judge_bias.judges import build_adapter


def result_slug(result_id: str) -> str:
    """Filesystem-safe filename stem for a `::`-delimited result id."""
    return slug(result_id.replace("::", "__"))


def save_raw_response(
    root: Path, raw_dir: Path, model: str, result_id: str, text: str
) -> str:
    """Persist the raw model text and return its manifest-relative POSIX path."""
    out = root / raw_dir / slug(model) / f"{result_slug(result_id)}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return to_rel_posix(out, root)


def load_done_ids(*manifest_paths: Path) -> Set[str]:
    """Collect result_ids already present across one or more result manifests."""
    done: Set[str] = set()
    for path in manifest_paths:
        if path and Path(path).exists():
            done.update(r.result_id for r in io.iter_jsonl(path, JudgeResult))
    return done


def manifest_for(results_dir: Path, task: str, model: str, *, biased: bool) -> Path:
    """raw_judgments/ for unbiased judgments, biased_judgments/ for biased ones."""
    sub = "biased_judgments" if biased else "raw_judgments"
    return results_dir / sub / f"{task}__{slug(model)}.jsonl"


_T = TypeVar("_T")
_R = TypeVar("_R")


def run_jobs(items: List[_T], work_fn: Callable[[_T], _R], workers: int) -> Iterator[_R]:
    """Apply `work_fn` to each item, yielding results as they complete.

    Serial when `workers <= 1` (deterministic order, matches single-thread runs);
    otherwise a ThreadPoolExecutor overlaps the I/O-bound API calls. Only the
    network call runs in worker threads — callers must do manifest writes on the
    main thread (as they consume this generator) to avoid file races.
    """
    if workers and workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(work_fn, it) for it in items]
            for fut in as_completed(futures):
                yield fut.result()
    else:
        for it in items:
            yield work_fn(it)


def adapter_for(
    judge_cfg: dict, *, use_api: bool, root: Path, dry_run: bool
) -> tuple[Optional[object], str]:
    """Build the adapter, or skip it for an offline dry run. Returns (adapter, model).

    A dry run counts targets and never calls anything, so requiring `--use-api` for
    it would mean asserting an intent to spend in order to find out what spending
    would cost — the wrong way round. `build_adapter` raises PermissionError for a
    real adapter without the flag, so the preview of any real judge was impossible.
    The model label comes from the config, which is where the adapter reads it too,
    so the previewed result_ids are the ones the paid run will use.
    """
    if dry_run and not use_api:
        return None, judge_cfg.get("model_name", "unknown")
    adapter = build_adapter(judge_cfg, use_api=use_api, root=root)
    return adapter, adapter.model_name


def load_failure_log_path(cfg: dict, root: Path) -> Optional[Path]:
    """Resolve an optional `failure_log:` config entry to an absolute path."""
    return (root / cfg["failure_log"]) if cfg.get("failure_log") else None


def log_failure(path: Optional[Path], result_id: str, message: str) -> None:
    """Append a transient API failure to a JSONL log (not the result manifest).

    API failures are NOT written as JudgeResults and NOT marked done, so a
    resumable rerun retries them (unlike parse failures, which are real verdicts).
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"result_id": result_id, "message": message}, ensure_ascii=False) + "\n")
