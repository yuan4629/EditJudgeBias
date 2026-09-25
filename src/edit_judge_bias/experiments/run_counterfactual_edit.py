"""Step 4 of the fairness track: re-run the ORIGINAL editing instruction on each
counterfactual original, so the judge can be asked the same question about two scenes
that differ only in a protected attribute.

    step 2 (run_attribute_edit)   original ---flip attribute---> counterfactual original
    step 4 (THIS FILE)            counterfactual original ---original instruction---> edited

The fairness quantity is then `score(edited | woman variant) - score(edited | man variant)`,
paired within scene. Nothing here touches the main A/B/C grid.

Deliberate deviation, and it must be stated in the paper: **we do not have the source
datasets' own editors**, so the same `gpt-image-2` that flips the attribute is also the
step-4 editor. The fairness comparison is within-editor and paired, so it still holds — but
it is not a claim about the datasets' editors.

★ THE NULL CONTROL AND WHY `repeat` NEEDS THE CACHE OFF
gpt-image-2 re-renders the whole frame, so running the SAME prompt on the SAME image twice
gives two different pictures. That editor noise is D's analogue of `sham` + retest, and it
is the floor an attribute gap has to clear. `--repeat 2` collects it. But
`QwenImageEditClient` caches on the SHA-256 of (model, prompt, size, image bytes)
(`fairness/image_edit_client.py:104-126`), so a cached second render would return the FIRST
image byte for byte and report a noise floor of exactly zero — a fabricated result, not a
measured one. This runner therefore refuses to start when `repeat > 1` and a `cache_dir`
survives, mirroring the guard the scoring retest arm needed
(`experiments/run_scoring_judge.py:261-268`).

    # preview, no network:
    python -m edit_judge_bias.experiments.run_counterfactual_edit \
        --config configs/fairness/counterfactual_edit.yaml --dry-run
    # real renders:
    python -m edit_judge_bias.experiments.run_counterfactual_edit \
        --config configs/fairness/counterfactual_edit.yaml --use-api
    # the editor-noise floor (a SECOND render of the same input; needs cache_dir: null):
    python -m edit_judge_bias.experiments.run_counterfactual_edit \
        --config configs/fairness/counterfactual_null.yaml --use-api
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve, to_rel_posix
from edit_judge_bias.experiments.judge_common import run_jobs
from edit_judge_bias.fairness.image_edit_client import ImageEditError, QwenImageEditClient
from edit_judge_bias.fairness.records import AttributeEditRecord, CounterfactualEditRecord


def _render_id(edit_id: str, render_index: int) -> str:
    """`render_index` 1 is the study render; 2+ are the editor-noise floor.

    The suffix is only added from 2 on, so turning the null control on later does not
    invalidate the render ids already on disk (the same reason the retest arm suffixes
    `::rep2` rather than renumbering the first pass).
    """
    return edit_id if render_index <= 1 else f"{edit_id}::render{render_index}"


def _load_done(manifest_out: Path) -> set:
    if not manifest_out.exists():
        return set()
    return {r.render_id for r in io.iter_jsonl(manifest_out, CounterfactualEditRecord)}


def _log_failure(path: Path, render_id: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"render_id": render_id, "message": message}, ensure_ascii=False) + "\n")


def run(
    config_path: str | Path,
    *,
    root: Optional[Path] = None,
    use_api: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict:
    root = Path(root) if root is not None else default_root()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    attribute_edits = root / cfg["attribute_edits"]
    output_dir = root / cfg["output_dir"]
    manifest_out = root / cfg["manifest_out"]
    repeat = int(cfg.get("repeat", 1))
    cache_cfg = cfg.get("cache_dir")

    # Load-bearing guard — see the module docstring. Checked before anything is spent.
    if repeat > 1 and cache_cfg:
        raise ValueError(
            f"repeat={repeat} with cache_dir={cache_cfg!r}: a cached second render returns the "
            "first image byte for byte and would report an editor-noise floor of exactly zero. "
            "Set `cache_dir: null` in the config for any repeat > 1."
        )

    edits = io.read_jsonl(attribute_edits, AttributeEditRecord)
    only = cfg.get("only_attributes")
    if only:
        edits = [e for e in edits if e.attribute in set(only)]
    only_v = cfg.get("only_variants")
    if only_v:
        edits = [e for e in edits if e.variant_label in set(only_v)]

    jobs = [(e, i) for e in edits for i in range(1, repeat + 1)]
    if limit is not None:
        jobs = jobs[:limit]

    if dry_run or not use_api:
        return {
            "attribute_edits": len(edits),
            "repeat": repeat,
            "planned_renders": len(jobs),
            "use_api": use_api,
            "note": "dry-run" if dry_run else "no --use-api: nothing was called",
        }

    client = QwenImageEditClient(
        base_url=cfg.get("base_url") or os.environ.get("EDITJUDGE_BASE_URL", ""),
        model=cfg.get("model", "gpt-image-2"),
        api_key_env=cfg.get("api_key_env", "EDITJUDGE_API_KEY"),
        size=cfg.get("size"),
        response_format=cfg.get("response_format", "b64_json"),
        timeout=int(cfg.get("timeout", 120)),
        max_retries=int(cfg.get("max_retries", 4)),
        cache_dir=root / cache_cfg if cache_cfg else None,
    )
    done = _load_done(manifest_out)
    stats = {"written": 0, "skipped": 0, "failed": 0}
    pending = [(e, i) for e, i in jobs if _render_id(e.edit_id, i) not in done]
    stats["skipped"] = len(jobs) - len(pending)

    def do_one(item):
        edit, render_index = item
        rid = _render_id(edit.edit_id, render_index)
        src = resolve(edit.counterfactual_image_path, root)
        out_path = output_dir / edit.attribute / f"{rid.replace('::', '__')}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # The prompt is the ORIGINAL editing instruction, unchanged — that is the whole
            # point: the same task is asked of two scenes differing only in the attribute.
            img = client.edit(src, edit.instruction)
            img.save(out_path)
        except Exception as exc:  # noqa: BLE001
            # Broad on purpose — see the twin comment in run_attribute_edit. One item must
            # never be able to abort a batch that is resumable by render_id.
            return ("fail", rid, f"{type(exc).__name__}: {exc}")
        return ("ok", CounterfactualEditRecord(
            render_id=rid,
            edit_id=edit.edit_id,
            base_sample_id=edit.base_sample_id,
            attribute=edit.attribute,
            variant_label=edit.variant_label,
            render_index=render_index,
            instruction=edit.instruction,
            counterfactual_image_path=to_rel_posix(src, root),
            edited_image_path=to_rel_posix(out_path, root),
            editor_model=cfg.get("model", "gpt-image-2"),
        ))

    # Only the API call and the distinctly-named PNG write run in workers; the manifest
    # append stays on the main thread, per run_jobs' contract.
    for outcome in run_jobs(pending, do_one, int(cfg.get("workers", 1))):
        if outcome[0] == "fail":
            _, rid, msg = outcome
            stats["failed"] += 1
            _log_failure(
                root / cfg.get("failure_log", "results/logs/counterfactual_edit_fail.jsonl"),
                rid, msg,
            )
            continue
        io.append_jsonl(manifest_out, outcome[1])
        stats["written"] += 1
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 4: re-run the instruction on counterfactuals.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--use-api", action="store_true", help="actually call the edit endpoint (spends)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, call nothing")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    result = run(a.config, root=a.root, use_api=a.use_api, dry_run=a.dry_run, limit=a.limit)
    prefix = "[dry-run] " if (a.dry_run or not a.use_api) else ""
    print(f"{prefix}counterfactual edit: {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
