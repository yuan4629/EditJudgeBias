"""Quality-preservation validation runner.

Two parts per sampled biased image:
  1. SSIM (automatic, local) between the original edited image and the biased one.
  2. An independent MLLM validator call (§6.2) judging whether the cosmetic bias
     changed the true editing outcome -> a QualityValidationResult with a pass flag.

The validator is a DISTINCT call that never participates in the main scoring /
pairwise experiment. Resumable (skips already-validated biased_ids), fail-soft on
API errors, and gated behind --use-api. Writes per-bias pass-rate + SSIM summary.

    python -m edit_judge_bias.experiments.run_quality_validation \
        --config configs/experiment/quality_validation_pilot.yaml --use-api
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root, resolve
from edit_judge_bias.data.schema import BiasedRecord, QualityValidationResult, SampleRecord
from edit_judge_bias.experiments.judge_common import (
    adapter_for as _adapter_for,
    load_failure_log_path,
    log_failure,
    run_jobs,
    save_raw_response,
)
from edit_judge_bias.judges.base import JudgeRequest
from edit_judge_bias.judges.parser import parse_validation
from edit_judge_bias.metrics.quality_preservation import compute_ssim
from edit_judge_bias.prompts.validator_prompt import build_validator_prompt


@dataclass
class RunStats:
    validated: int = 0
    skipped: int = 0
    parse_failures: int = 0
    api_failures: int = 0
    join_misses: int = 0

    def summary(self) -> str:
        return (
            f"validated={self.validated} skipped={self.skipped} "
            f"parse_failures={self.parse_failures} api_failures={self.api_failures} "
            f"join_misses={self.join_misses}"
        )


def _filter_bias_types(
    biased: List[BiasedRecord], bias_types: Sequence[str] | None
) -> List[BiasedRecord]:
    """Keep only the named `bias_type` buckets.

    Added for WP-A4d, which needs ONE of the four `edit_damage` conditions and must not
    pay for the other three.  Two properties are deliberate:

    * **It raises on a name absent from the manifest.**  A typo that silently selected
      zero rows would make the runner report `validated=0`, which is indistinguishable
      from "this arm is already complete" -- the exact shape of failure this project has
      been bitten by before.
    * **It is refused when `per_bias > 0`** (see `_select`).  `_select` shuffles every
      bucket with ONE `random.Random(seed)` built outside the loop and never reset -- the
      same shared-RNG construction whose divergence WP-A3 traced as the root cause of
      `sham` overlapping each arm's base images by only 6-12%.  Dropping a bucket
      therefore hands the surviving buckets a different RNG state, so a k-prefix taken
      with a filter is NOT the k-prefix taken without one.  With `per_bias <= 0` the
      whole bucket is taken and only its order moves, which selects nothing; with
      `per_bias > 0` it would silently re-draw.  Resetting the RNG per bucket would fix
      it in the abstract and re-draw the PUBLISHED `_select(110, seed=42)` sample in
      practice, which is why the combination is refused instead of repaired.
    """
    if not bias_types:
        return biased
    wanted = list(dict.fromkeys(bias_types))
    present = {b.bias_type for b in biased}
    missing = [bt for bt in wanted if bt not in present]
    if missing:
        raise ValueError(
            f"bias_types names {missing} which the biased manifest does not contain "
            f"(it has {sorted(present)}); refusing to run an arm that would collect "
            "nothing and look finished"
        )
    keep = set(wanted)
    return [b for b in biased if b.bias_type in keep]


def _select(
    biased: List[BiasedRecord],
    per_bias: int,
    seed: int,
    bias_types: Sequence[str] | None = None,
) -> List[BiasedRecord]:
    if bias_types and per_bias > 0:
        raise ValueError(
            "bias_types cannot be combined with per_bias > 0: _select draws every "
            "bucket from one shared RNG, so removing a bucket changes which rows the "
            "surviving buckets draw (the WP-A3 root cause). Use per_bias: 0."
        )
    by_type: Dict[str, List[BiasedRecord]] = defaultdict(list)
    for b in _filter_bias_types(biased, bias_types):
        by_type[b.bias_type].append(b)
    rng = random.Random(seed)
    chosen: List[BiasedRecord] = []
    for bt in sorted(by_type):
        bucket = sorted(by_type[bt], key=lambda b: b.biased_id)
        rng.shuffle(bucket)
        chosen.extend(bucket if per_bias <= 0 else bucket[:per_bias])
    return chosen


def run(
    config_path: PathLike,
    *,
    root: PathLike | None = None,
    dry_run: bool = False,
    use_api: bool = False,
) -> RunStats:
    root = Path(root) if root is not None else default_root()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    by_id = {s.sample_id: s for s in io.read_jsonl(root / cfg["samples"], SampleRecord)}
    biased = io.read_jsonl(root / cfg["biased"], BiasedRecord)
    sampled = _select(
        biased,
        int(cfg.get("per_bias", 0)),
        int(cfg.get("seed", 42)),
        cfg.get("bias_types"),
    )

    adapter, validator_model = _adapter_for(
        _judge_cfg(cfg, root), use_api=use_api, root=root, dry_run=dry_run
    )
    raw_dir = Path(cfg.get("raw_dir", "results/raw_responses_validation"))
    out_manifest = root / cfg["out_manifest"]
    summary_csv = root / cfg.get("summary_csv", "results/metrics/quality_preservation_summary.csv")
    failure_log = load_failure_log_path(cfg, root)
    workers = int(cfg.get("workers", 1))
    # The published grid used the default template and must keep using it; the variant
    # exists only for the WP-A4c ablation.  `build_validator_prompt` raises on an unknown
    # style, so a typo in a config fails at the first call rather than silently
    # collecting a whole arm under the published prompt.
    prompt_style = str(cfg.get("prompt_style", "default"))
    build_validator_prompt("", prompt_style)  # eager validation: fail before spending

    stats = RunStats()
    # Resolve base sample for each sampled biased; drop join-misses.
    items: List[tuple] = []
    for b in sampled:
        base = by_id.get(b.base_sample_id)
        if base is None:
            stats.join_misses += 1
        else:
            items.append((b, base))

    done = {} if dry_run else {
        r.biased_id: r for r in (io.iter_jsonl(out_manifest, QualityValidationResult)
                                 if out_manifest.exists() else [])
    }
    pending = [(b, base) for (b, base) in items if b.biased_id not in done]
    if dry_run:
        stats.validated = len(pending)
        stats.skipped = len(items) - len(pending)
        # WP-F4/P7.  The preview is SSIM-only -- `done` is forced empty under --dry-run, so
        # no validator verdict is in it -- and it used to be written straight over the
        # tracked summary, silently replacing a published pass-rate table with a preview
        # that contains no pass rates.  A preview must never be able to overwrite a
        # result, so it goes to a sidecar and says where.
        preview = summary_csv.with_suffix(".dryrun.csv")
        _write_summary(preview, items, done, by_id, root)
        print(f"[dry-run] SSIM-only preview -> {preview} "
              f"(the tracked {summary_csv.name} was not touched)")
        return stats

    def do_one(item):
        b, base = item
        ssim = compute_ssim(resolve(base.edited_image_path, root),
                            resolve(b.biased_image_path, root))
        prompt = build_validator_prompt(base.instruction, prompt_style)
        images = [resolve(base.original_image_path, root),
                  resolve(base.edited_image_path, root),
                  resolve(b.biased_image_path, root)]
        try:
            raw = adapter.generate(JudgeRequest(prompt, images, "validation"))
        except Exception as exc:  # noqa: BLE001
            return ("api_fail", b.biased_id, f"{type(exc).__name__}: {exc}")
        raw_rel = save_raw_response(root, raw_dir, validator_model, f"val::{b.biased_id}", raw)
        parsed = parse_validation(raw)
        result = QualityValidationResult(
            instruction_adherence_changed=parsed.instruction_adherence_changed,
            editing_quality_changed=parsed.editing_quality_changed,
            detail_preservation_changed=parsed.detail_preservation_changed,
            major_semantic_shift=parsed.major_semantic_shift,
            passed=parsed.passed, reason=parsed.reason,
            biased_id=b.biased_id, base_sample_id=b.base_sample_id,
            validator_model=validator_model, raw_response_path=raw_rel,
            parse_success=parsed.success, parse_error=(parsed.error or None), ssim=ssim,
        )
        return ("ok", result)

    for outcome in run_jobs(pending, do_one, workers):
        if outcome[0] == "api_fail":
            _, bid, msg = outcome
            stats.api_failures += 1
            log_failure(failure_log, bid, msg)
            continue
        _, result = outcome
        if not result.parse_success:
            stats.parse_failures += 1
        io.append_jsonl(out_manifest, result)
        done[result.biased_id] = result
        stats.validated += 1

    stats.skipped = len(items) - len(pending)
    _write_summary(summary_csv, items, done, by_id, root)
    return stats


def _write_summary(summary_csv: Path, items, done, by_id, root: Path) -> None:
    """Per-bias pass rate (over parsed validations) + SSIM stats over `items`."""
    bias_of = {b.biased_id: b.bias_type for b, _ in items}
    agg: Dict[str, dict] = defaultdict(lambda: {"pass": 0, "parsed": 0, "ssims": []})
    for b, base in items:
        bt = b.bias_type
        r = done.get(b.biased_id)
        if r is not None:
            if r.ssim is not None:
                agg[bt]["ssims"].append(r.ssim)
            if r.parse_success:
                agg[bt]["parsed"] += 1
                if r.passed:
                    agg[bt]["pass"] += 1
    rows = []
    for bt in sorted(agg):
        a = agg[bt]
        ssims = a["ssims"]
        rows.append({
            "bias_type": bt,
            "n_validated": a["parsed"],
            "mllm_pass_rate": round(a["pass"] / a["parsed"], 4) if a["parsed"] else "",
            "mean_ssim": round(sum(ssims) / len(ssims), 4) if ssims else "",
            "min_ssim": round(min(ssims), 4) if ssims else "",
        })
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def _judge_cfg(cfg: dict, root: Path) -> dict:
    if cfg.get("judge_config"):
        return yaml.safe_load((root / cfg["judge_config"]).read_text(encoding="utf-8"))
    return cfg.get("judge", {})


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MLLM quality-preservation validation.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--use-api", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    stats = run(args.config, root=args.root, dry_run=args.dry_run, use_api=args.use_api)
    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}quality validation: {stats.summary()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
