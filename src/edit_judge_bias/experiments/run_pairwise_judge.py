"""Run pairwise judgments over pairs (Milestone 3/4).

For each pair the runner builds a pairwise prompt, calls the adapter with
[original, edited_A, edited_B], saves the raw response, parses the winner, and
writes a JudgeResult. Optionally also runs a position-bias pass that swaps the A/B
display order (the 6th MVP bias — a judge-protocol bias, no image change, §4.2/§5.4);
those results carry bias_type="position" and land in results/biased_judgments/.

    python -m edit_judge_bias.experiments.run_pairwise_judge \
        --config configs/experiment/pairwise_v2.yaml --judge-config configs/judge/mock.yaml
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root, resolve
from edit_judge_bias.data.schema import BiasedRecord, JudgeResult, PairRecord
from edit_judge_bias.experiments.judge_common import (
    adapter_for as _adapter_for,
    load_done_ids,
    log_failure,
    manifest_for,
    run_jobs,
    save_raw_response,
)
from edit_judge_bias.experiments.prompt_bias import (
    bandwagon_side,
    load_specs,
    permuted_name_map,
    resolve_model_name,
    spec_bias_type,
)
from edit_judge_bias.judges.parser import parse_pairwise
from edit_judge_bias.prompts import build_pairwise_prompt


@dataclass
class PairwiseTarget:
    result_id: str
    pair_id: str
    instruction: str
    original_image: str
    image_a: str  # shown in slot A
    image_b: str  # shown in slot B
    bias_type: Optional[str]
    biased_side: Optional[str]
    # A-class (prompt-level) injections; all unset for image/protocol conditions.
    bandwagon_target: Optional[str] = None
    model_name_a: Optional[str] = None
    model_name_b: Optional[str] = None
    bias_params: dict = field(default_factory=dict)


@dataclass
class RunStats:
    written: int = 0
    skipped: int = 0
    parse_failures: int = 0
    api_failures: int = 0

    def summary(self) -> str:
        return (
            f"written={self.written} skipped={self.skipped} "
            f"parse_failures={self.parse_failures} api_failures={self.api_failures}"
        )


def _resolve_judge_cfg(cfg: dict, root: Path) -> dict:
    """Judge config inline (`judge:`) or referenced by file (`judge_config:`).

    `judge_overrides:` is merged on top, so one arm config can be run against all
    five judges via `--judge-config` while still pinning arm-specific settings.
    """
    if cfg.get("judge_config"):
        base = yaml.safe_load((root / cfg["judge_config"]).read_text(encoding="utf-8"))
    else:
        base = dict(cfg.get("judge", {}))
    base.update(cfg.get("judge_overrides") or {})
    return base


def _one_sided_targets(
    cfg: dict, root: Path, model: str, style: str, pairs: List[PairRecord]
) -> List[PairwiseTarget]:
    """Pair conditions where exactly ONE side carries an image bias.

    This is the pairwise analogue of the scoring grid: the two edits are unchanged
    in real quality, one of them is dressed up, and the question is whether the
    verdict moves. Which side gets dressed is a seeded function of the pair id, not
    always slot A -- otherwise the effect would be inseparable from the position
    bias we are measuring in the same run.
    """
    spec = cfg.get("one_sided_biases") or {}
    if not spec:
        return []
    wanted = set(spec.get("bias_types") or [])
    seed = int(cfg.get("seed", 42))

    # base_sample_id -> {bias_type: biased image path}
    by_sample: Dict[str, Dict[str, str]] = defaultdict(dict)
    for b in io.read_jsonl(root / spec["biased"], BiasedRecord):
        if not wanted or b.bias_type in wanted:
            by_sample[b.base_sample_id][b.bias_type] = b.biased_image_path.as_posix()

    targets: List[PairwiseTarget] = []
    for p in pairs:
        for bias_type in sorted(wanted or {bt for s in by_sample.values() for bt in s}):
            side = spec.get("side")
            if side not in ("a", "b"):
                side = random.Random(f"{seed}:one_sided:{p.pair_id}").choice(("a", "b"))
            sid = p.sample_id_a if side == "a" else p.sample_id_b
            biased_path = by_sample.get(sid, {}).get(bias_type)
            if biased_path is None:
                continue  # not injected for this member; the runner reports the gap
            image_a = biased_path if side == "a" else p.edited_image_a_path.as_posix()
            image_b = biased_path if side == "b" else p.edited_image_b_path.as_posix()
            targets.append(PairwiseTarget(
                result_id=f"pair::{model}::{style}::{p.pair_id}::{bias_type}::{side}",
                pair_id=p.pair_id, instruction=p.instruction,
                original_image=p.original_image_path.as_posix(),
                image_a=image_a, image_b=image_b,
                bias_type=bias_type, biased_side=side,
                bias_params={"biased_sample_id": sid},
            ))
    return targets


def _build_targets(cfg: dict, root: Path, model: str, style: str) -> List[PairwiseTarget]:
    pairs = io.read_jsonl(root / cfg["pairs"], PairRecord)
    include_swap = bool(cfg.get("include_position_swap", False))
    targets: List[PairwiseTarget] = []
    for p in pairs:
        orig = p.original_image_path.as_posix()
        a, b = p.edited_image_a_path.as_posix(), p.edited_image_b_path.as_posix()
        targets.append(PairwiseTarget(
            result_id=f"pair::{model}::{style}::{p.pair_id}", pair_id=p.pair_id,
            instruction=p.instruction, original_image=orig, image_a=a, image_b=b,
            bias_type=None, biased_side=None,
        ))
        if include_swap:
            targets.append(PairwiseTarget(
                result_id=f"pair::{model}::{style}::{p.pair_id}::swap",
                pair_id=p.pair_id, instruction=p.instruction, original_image=orig,
                image_a=b, image_b=a,  # A/B display order swapped
                bias_type="position", biased_side="swap",
            ))

    # A-class: same images in the same slots, different prompt text.
    specs = load_specs(cfg)
    if specs:
        seed = int(cfg.get("seed", 42))
        roster = [m for p in pairs for m in (p.edit_model_a, p.edit_model_b)]
        name_map = permuted_name_map(roster, seed)
        for spec in specs:
            bias_type = spec_bias_type(spec)
            for p in pairs:
                side = bandwagon_side(spec, p.pair_id, seed)
                name_a = resolve_model_name(spec, p.edit_model_a, name_map)
                name_b = resolve_model_name(spec, p.edit_model_b, name_map)
                params = {k: v for k, v in (
                    ("bandwagon_target", side),
                    ("model_name_a", name_a),
                    ("model_name_b", name_b),
                    ("model_name_source", spec.get("model_name_source")),
                ) if v is not None}
                targets.append(PairwiseTarget(
                    result_id=f"pair::{model}::{style}::{p.pair_id}::{bias_type}",
                    pair_id=p.pair_id, instruction=p.instruction,
                    original_image=p.original_image_path.as_posix(),
                    image_a=p.edited_image_a_path.as_posix(),
                    image_b=p.edited_image_b_path.as_posix(),
                    bias_type=bias_type, biased_side=None,
                    bandwagon_target=side, model_name_a=name_a, model_name_b=name_b,
                    bias_params=params,
                ))

    targets.extend(_one_sided_targets(cfg, root, model, style, pairs))

    # Consistency-Rate arm (§8.2 CR): ask the identical question again. RR alone cannot
    # say whether a low score is manipulation or instability — §8.2's own reading table
    # is (high CR + low RR) = "stable but manipulable" vs (low CR + low RR) = "the model
    # is just noisy, treat the bias conclusion with care". Without CR, qwen3.5-plus's
    # RR of 0.352 has no interpretation.
    repeat = int(cfg.get("repeat", 1))
    if repeat > 1:
        base_targets = list(targets)
        for i in range(2, repeat + 1):
            for t in base_targets:
                targets.append(replace(
                    t,
                    result_id=f"{t.result_id}::rep{i}",
                    bias_params={**t.bias_params, "repeat_index": i},
                ))
    return targets


def build_request(t: PairwiseTarget, *, style: str, root: Path) -> Tuple[str, List[Path]]:
    """The exact (prompt, [original, slot A, slot B]) a pairwise target is asked with.

    Module-level so `fill_runner` asks byte-identical questions to this runner.
    """
    prompt = build_pairwise_prompt(
        t.instruction, style, bandwagon_target=t.bandwagon_target,
        model_name_a=t.model_name_a, model_name_b=t.model_name_b,
    )
    return prompt, [resolve(t.original_image, root), resolve(t.image_a, root),
                    resolve(t.image_b, root)]


def build_result(
    t: PairwiseTarget, raw: str, raw_rel: str, *, model: str, style: str
) -> JudgeResult:
    """Parse `raw` into the JudgeResult row this runner writes for `t`."""
    parsed = parse_pairwise(raw)
    return JudgeResult(
        result_id=t.result_id, judge_model=model, task_type="pairwise",
        prompt_type=f"{style}_pairwise", raw_response_path=raw_rel,
        parse_success=parsed.success, bias_type=t.bias_type, pair_id=t.pair_id,
        bias_params=t.bias_params,
        biased_side=t.biased_side, winner=parsed.winner,
        reason=parsed.reason, parse_error=(parsed.error or None),
    )


def run(
    config_path: PathLike,
    *,
    root: PathLike | None = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    use_api: bool = False,
    judge_config: PathLike | None = None,
) -> RunStats:
    root = Path(root) if root is not None else default_root()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if judge_config is not None:
        cfg = {**cfg, "judge_config": str(judge_config), "judge": None}
    style = cfg.get("prompt_style", "vanilla")
    judge_cfg = _resolve_judge_cfg(cfg, root)
    if int(cfg.get("repeat", 1)) > 1 and judge_cfg.get("cache_dir"):
        # The adapter caches on a hash of the request payload, so a second identical
        # ask would be served from disk and the CR arm would report a consistency of
        # exactly 1.0 — a noise floor manufactured by plumbing.
        raise ValueError(
            "repeat>1 needs the response cache off; set `judge_overrides: "
            "{cache_dir: null}` in this config"
        )
    adapter, model = _adapter_for(judge_cfg, use_api=use_api, root=root, dry_run=dry_run)
    raw_dir = Path(cfg.get("raw_dir", "results/raw_responses"))
    results_dir = root / cfg.get("results_dir", "results")
    failure_log = (root / cfg["failure_log"]) if cfg.get("failure_log") else None

    targets = _build_targets(cfg, root, model, style)
    if limit is not None:
        targets = targets[:limit]

    man_orig = manifest_for(results_dir, "pairwise", model, biased=False)
    man_bias = manifest_for(results_dir, "pairwise", model, biased=True)
    done = set() if dry_run else load_done_ids(man_orig, man_bias)
    workers = int(cfg.get("workers", 1))

    stats = RunStats()
    pending = []
    for t in targets:
        if t.result_id in done:
            stats.skipped += 1
        else:
            pending.append(t)
    if dry_run:
        stats.written = len(pending)
        return stats

    def do_one(t: PairwiseTarget):
        prompt, images = build_request(t, style=style, root=root)
        try:
            raw = adapter.compare(prompt, images)
        except Exception as exc:  # noqa: BLE001 - transient API error: log, retry next run
            return ("api_fail", t, f"{type(exc).__name__}: {exc}", None)
        raw_rel = save_raw_response(root, raw_dir, model, t.result_id, raw)
        result = build_result(t, raw, raw_rel, model=model, style=style)
        return ("ok", t, result, result.parse_success)

    for outcome in run_jobs(pending, do_one, workers):
        if outcome[0] == "api_fail":
            _, t, msg, _ = outcome
            stats.api_failures += 1
            log_failure(failure_log, t.result_id, msg)
            continue
        _, t, result, parse_ok = outcome
        if not parse_ok:
            stats.parse_failures += 1
        io.append_jsonl(man_bias if t.bias_type else man_orig, result)
        done.add(t.result_id)
        stats.written += 1
    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pairwise judgments (mock-first).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--use-api", action="store_true", help="allow real API adapters (M4)")
    p.add_argument(
        "--judge-config", type=Path, default=None,
        help="override the config's `judge_config:` — one arm config, five judges",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    stats = run(args.config, root=args.root, dry_run=args.dry_run,
                limit=args.limit, use_api=args.use_api,
                judge_config=args.judge_config)
    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}pairwise: {stats.summary()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
