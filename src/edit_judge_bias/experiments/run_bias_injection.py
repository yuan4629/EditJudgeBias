"""Batch bias injection over a samples manifest (Milestone 2, §10).

Reads a YAML experiment config, applies each configured bias to every sample's
edited image, writes biased images (never touching originals) and a
`biased_samples.jsonl` manifest of BiasedRecords.

Resumable: completed `biased_id`s are read back from the output manifest and
skipped. Failures are logged (to a JSONL failure log) and do not abort the batch.

    python -m edit_judge_bias.experiments.run_bias_injection \
        --config configs/experiment/bias_injection_full_v2.yaml
    # preview the plan without writing images:
    python -m edit_judge_bias.experiments.run_bias_injection --config ... --dry-run
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel

from edit_judge_bias.bias.registry import get_injector
from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root, resolve, to_rel_posix
from edit_judge_bias.data.schema import BiasedRecord, SampleRecord


# --------------------------------------------------------------------------- #
# Config parsing                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class BiasSpec:
    bias_type: str
    config: Dict[str, Any]


def _load_bias_spec(entry, root: Path) -> BiasSpec:
    """A bias entry is either a path to a per-bias YAML or an inline dict."""
    if isinstance(entry, str):
        data = yaml.safe_load((root / entry).read_text(encoding="utf-8"))
    elif isinstance(entry, dict):
        data = entry
    else:
        raise TypeError(f"bias entry must be str path or dict, got {type(entry)}")
    return BiasSpec(bias_type=data["bias_type"], config=dict(data.get("config") or {}))


class InjectionPlan(BaseModel):
    samples: Path
    output_dir: Path
    manifest_out: Path
    failure_log: Optional[Path] = None
    seed: int = 42


def load_plan(config_path: PathLike, root: Path) -> Tuple[InjectionPlan, List[BiasSpec]]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    plan = InjectionPlan(**{k: cfg[k] for k in cfg if k != "biases"})
    specs = [_load_bias_spec(e, root) for e in cfg.get("biases", [])]
    return plan, specs


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class RunStats:
    written: int = 0
    skipped: int = 0
    failed: int = 0
    per_bias: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.per_bias is None:
            self.per_bias = defaultdict(int)

    def summary(self) -> str:
        per = ", ".join(f"{k}={v}" for k, v in sorted(self.per_bias.items()))
        return (
            f"written={self.written} skipped={self.skipped} failed={self.failed}"
            + (f" | per_bias: {per}" if per else "")
        )


def _biased_id(sample_id: str, bias_type: str) -> str:
    return f"{sample_id}__{bias_type}"


def _load_done(manifest_out: Path) -> set[str]:
    if not manifest_out.exists():
        return set()
    return {rec.biased_id for rec in io.iter_jsonl(manifest_out, BiasedRecord)}


def run(
    config_path: PathLike,
    *,
    root: PathLike | None = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> RunStats:
    root = Path(root) if root is not None else default_root()
    plan, specs = load_plan(config_path, root)
    manifest_out = root / plan.manifest_out
    failure_log = (root / plan.failure_log) if plan.failure_log else None

    samples = io.read_jsonl(root / plan.samples, SampleRecord)
    if limit is not None:
        samples = samples[:limit]

    if overwrite and manifest_out.exists() and not dry_run:
        manifest_out.unlink()
    done = set() if dry_run else _load_done(manifest_out)

    stats = RunStats()
    for spec in specs:
        injector = get_injector(spec.bias_type)
        out_subdir = root / plan.output_dir / spec.bias_type
        for sample in samples:
            bid = _biased_id(sample.sample_id, spec.bias_type)
            if bid in done:
                stats.skipped += 1
                continue
            if dry_run:
                stats.written += 1
                stats.per_bias[spec.bias_type] += 1
                continue

            src = resolve(sample.edited_image_path, root)
            out_path = out_subdir / f"{bid}.png"
            call_config = dict(spec.config)
            if getattr(injector, "needs_original", False):
                # C-class injectors localize the edit against the original image.
                call_config["original_image_path_abs"] = (
                    str(resolve(sample.original_image_path, root))
                    if sample.original_image_path
                    else None
                )
                if sample.metadata.mask_path:
                    call_config["mask_path_abs"] = str(resolve(sample.metadata.mask_path, root))
            result = injector.apply(
                src,
                out_path,
                config=call_config,
                sample=sample.model_dump(mode="json"),
                seed=plan.seed,
            )
            if not result.success:
                stats.failed += 1
                if failure_log is not None:
                    _log_failure(failure_log, bid, sample.sample_id, spec.bias_type, result.message)
                continue

            record = BiasedRecord(
                biased_id=bid,
                base_sample_id=sample.sample_id,
                bias_type=spec.bias_type,
                bias_strength=result.bias_strength,
                biased_image_path=to_rel_posix(result.biased_image_path, root),
                bias_applied_to=result.bias_applied_to,
                bias_params=result.params,
            )
            io.append_jsonl(manifest_out, record)
            done.add(bid)
            stats.written += 1
            stats.per_bias[spec.bias_type] += 1
    return stats


def _log_failure(path: Path, biased_id: str, sample_id: str, bias_type: str, message: str) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(
            {"biased_id": biased_id, "base_sample_id": sample_id,
             "bias_type": bias_type, "message": message},
            ensure_ascii=False,
        ) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-inject biases over a samples manifest.")
    p.add_argument("--config", type=Path, required=True, help="injection experiment YAML")
    p.add_argument("--root", type=Path, default=None, help="project root (default: CWD)")
    p.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    p.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    p.add_argument("--overwrite", action="store_true", help="rebuild manifest from scratch")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    stats = run(
        args.config,
        root=args.root,
        dry_run=args.dry_run,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}bias injection: {stats.summary()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
