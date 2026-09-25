"""Run scoring judgments over original and/or biased edited images (Milestone 3/4).

For each target the runner builds a scoring prompt, calls the judge adapter with
[original_image, edited_image], saves the raw response, parses it, and writes a
JudgeResult. Unbiased results land in results/raw_judgments/, biased ones in
results/biased_judgments/ (§5.2). Mock-first: defaults to MockJudgeAdapter; real
API calls (M4) require a real adapter type and an explicit --use-api.

    python -m edit_judge_bias.experiments.run_scoring_judge \
        --config configs/experiment/scoring_pilot_mock.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root, resolve
from edit_judge_bias.data.schema import BiasedRecord, JudgeResult, SampleRecord
from edit_judge_bias.experiments.judge_common import (
    adapter_for as _adapter_for,
    load_done_ids,
    log_failure,
    manifest_for,
    run_jobs,
    save_raw_response,
)
from edit_judge_bias.experiments.prompt_bias import (
    load_specs,
    permuted_name_map,
    resolve_model_name,
    spec_bias_type,
)
from edit_judge_bias.judges.parser import DEFAULT_SCORE_SCALE, parse_scoring
from edit_judge_bias.prompts import build_scoring_prompt


@dataclass
class ScoringTarget:
    result_id: str
    sample_id: str
    biased_id: Optional[str]
    bias_type: Optional[str]
    instruction: str
    original_image: str
    edited_image: str
    # A-class (prompt-level) injections; both off for image-level conditions.
    bandwagon: bool = False
    model_name: Optional[str] = None
    bias_params: dict = field(default_factory=dict)


@dataclass
class RunStats:
    written: int = 0
    skipped: int = 0
    parse_failures: int = 0
    api_failures: int = 0
    join_misses: int = 0

    def summary(self) -> str:
        return (
            f"written={self.written} skipped={self.skipped} "
            f"parse_failures={self.parse_failures} api_failures={self.api_failures} "
            f"join_misses={self.join_misses}"
        )


def matches_filter(rec: SampleRecord, flt: Dict[str, object]) -> bool:
    """Does `rec` match every key in `flt`? Fields and metadata keys both count.

    The judging subset is one manifest holding several blocks (`subset_block`,
    `anchor_source`), each of which gets a different number of conditions — 14 for
    the balanced breadth block, 5 for an anchor. Splitting them into separate
    manifests would duplicate the samples that belong to both.
    """
    _unset = object()
    for key, want in flt.items():
        got = getattr(rec, key, None)
        if got is None:
            # SampleMetadata declares a handful of fields (has_mask, mask_path,
            # original_width/height) and allows extras for the rest (subset_block,
            # anchor_source, ...). Reading only `model_extra` would make a filter on
            # a declared one match nothing at all, which reads as "that block is
            # empty" rather than as a mistake.
            got = getattr(rec.metadata, key, _unset)
            if got is _unset:
                got = (rec.metadata.model_extra or {}).get(key)
        got = getattr(got, "value", got)  # unwrap enums
        if isinstance(want, (list, tuple, set)):
            if got not in want:
                return False
        elif got != want:
            return False
    return True


def _pick(samples: List[SampleRecord], n: int, how: str) -> List[SampleRecord]:
    """Take `n` samples, either as a prefix or spread evenly across the block.

    The manifest is written stratum by stratum, so a prefix of 4 is 4 EBench `add`
    samples on 512x512 frames — fine for the retest arm, which only needs a stable
    subset, but wrong for the smoke batch: one of the things the smoke batch exists
    to measure is the real per-call price, and price tracks image size. The pool
    runs 512x512 to 1024x1024, so a homogeneous prefix would quote a price for one
    corner of it. `spread` walks the whole block at an even stride instead.
    """
    if n >= len(samples) or how == "prefix":
        return samples[:n]
    if how != "spread":
        raise ValueError(f"sample_pick must be 'prefix' or 'spread', got {how!r}")
    step = len(samples) / n
    return [samples[min(len(samples) - 1, int(i * step))] for i in range(n)]


def _build_targets(cfg: dict, root: Path, model: str, style: str) -> List[ScoringTarget]:
    all_samples = io.read_jsonl(root / cfg["samples"], SampleRecord)
    flt = cfg.get("subset_filter") or {}
    samples = [s for s in all_samples if matches_filter(s, flt)] if flt else all_samples
    # Caps the SAMPLES, not the targets. `--limit` truncates after conditions are
    # expanded, which for a repeat arm would take N baselines and zero repeats.
    # The manifest order is the seeded draw's, so both picks below are reproducible.
    max_samples = cfg.get("max_samples")
    if max_samples:
        samples = _pick(samples, int(max_samples), cfg.get("sample_pick", "prefix"))
    known_ids = {s.sample_id for s in all_samples}
    by_id: Dict[str, SampleRecord] = {s.sample_id: s for s in samples}
    targets: List[ScoringTarget] = []

    if cfg.get("score_original", True):
        for s in samples:
            rid = f"score::{model}::{style}::{s.sample_id}"
            targets.append(ScoringTarget(
                result_id=rid, sample_id=s.sample_id, biased_id=None, bias_type=None,
                instruction=s.instruction,
                original_image=s.original_image_path.as_posix(),
                edited_image=s.edited_image_path.as_posix(),
            ))

    if cfg.get("score_biased", True) and cfg.get("biased"):
        bias_types = cfg.get("bias_types")
        for b in io.read_jsonl(root / cfg["biased"], BiasedRecord):
            if bias_types and b.bias_type not in bias_types:
                continue
            base = by_id.get(b.base_sample_id)
            if base is None:
                # Deliberately filtered out is not the same as dangling: only the
                # latter is a build error worth surfacing as a join miss.
                if b.base_sample_id in known_ids:
                    continue
                targets.append(_MISSING)  # sentinel; filtered by caller
                continue
            rid = f"score::{model}::{style}::{b.biased_id}"
            targets.append(ScoringTarget(
                result_id=rid, sample_id=base.sample_id, biased_id=b.biased_id,
                bias_type=b.bias_type, instruction=base.instruction,
                original_image=base.original_image_path.as_posix(),
                edited_image=b.biased_image_path.as_posix(),
            ))

    # A-class: same edited image, different prompt. One extra condition per spec.
    specs = load_specs(cfg)
    if specs:
        seed = int(cfg.get("seed", 42))
        name_map = permuted_name_map([s.edit_model for s in samples], seed)
        for spec in specs:
            bias_type = spec_bias_type(spec)
            bandwagon = bool(spec.get("bandwagon", False))
            for s in samples:
                model_name = resolve_model_name(spec, s.edit_model, name_map)
                params = {k: v for k, v in (
                    ("bandwagon", bandwagon or None),
                    ("model_name", model_name),
                    ("model_name_source", spec.get("model_name_source")),
                ) if v is not None}
                targets.append(ScoringTarget(
                    result_id=f"score::{model}::{style}::{s.sample_id}::{bias_type}",
                    sample_id=s.sample_id, biased_id=None, bias_type=bias_type,
                    instruction=s.instruction,
                    original_image=s.original_image_path.as_posix(),
                    edited_image=s.edited_image_path.as_posix(),
                    bandwagon=bandwagon, model_name=model_name, bias_params=params,
                ))

    # Test-retest arm: ask the identical question again. At temperature 0 the
    # answer is *supposed* to be identical, and the gap between that expectation
    # and reality is the noise floor every reported effect has to clear.
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


_MISSING = ScoringTarget("", "", None, None, "", "", "")  # join-miss sentinel


def resolve_prompt_style(cfg: dict, judge_cfg: dict, cli: Optional[str]) -> str:
    """CLI > the judge's own `scoring_prompt_style` > the arm's `prompt_style`.

    A judge outranks the arm here for one judge only, and on purpose. `gpt-4o-viescore`
    is not a model, it is *gpt-4o driven by the VIEScore rubric* — the rubric is what
    makes it the §2.1 dedicated vision judge and what made it the pilot's robust
    outlier. Arm configs pin `prompt_style: vanilla` and are run across all five
    judges with `--judge-config`, so without this the fifth judge would silently
    become a second plain gpt-4o, stamped `vanilla_scoring` under a name promising a
    rubric. The key is scoring-specific because `build_pairwise_prompt` has no
    VIEScore branch and would emit a vanilla prompt labelled `viescore_pairwise`.
    """
    return cli or judge_cfg.get("scoring_prompt_style") or cfg.get("prompt_style", "vanilla")


def _resolve_judge_cfg(cfg: dict, root: Path) -> dict:
    """Judge config inline (`judge:`) or referenced by file (`judge_config:`).

    `judge_overrides:` is merged on top, so one arm config can be run against all
    five judges (`--judge-config`) while still pinning arm-specific adapter
    settings — the retest arm has to disable the response cache, for instance.
    """
    if cfg.get("judge_config"):
        base = yaml.safe_load((root / cfg["judge_config"]).read_text(encoding="utf-8"))
    else:
        base = dict(cfg.get("judge", {}))
    base.update(cfg.get("judge_overrides") or {})
    return base


def build_request(
    t: ScoringTarget, *, style: str, score_scale: int, root: Path
) -> Tuple[str, List[Path]]:
    """The exact (prompt, [original, edited]) a scoring target is asked with.

    Module-level so `fill_runner` asks byte-identical questions to this runner: a
    second copy of the prompt call would be a second place for the wording to drift
    away from the published arms.
    """
    prompt = build_scoring_prompt(
        t.instruction, style, bandwagon=t.bandwagon, model_name=t.model_name,
        scale=score_scale,
    )
    return prompt, [resolve(t.original_image, root), resolve(t.edited_image, root)]


def build_result(
    t: ScoringTarget, raw: str, raw_rel: str, *, model: str, style: str, score_scale: int
) -> JudgeResult:
    """Parse `raw` into the JudgeResult row this runner writes for `t`."""
    parsed = parse_scoring(raw, scale=score_scale)
    return JudgeResult(
        result_id=t.result_id, judge_model=model, task_type="scoring",
        prompt_type=f"{style}_scoring", raw_response_path=raw_rel,
        parse_success=parsed.success, bias_type=t.bias_type, biased_id=t.biased_id,
        bias_params=t.bias_params,
        sample_id=t.sample_id, overall_score=parsed.overall_score,
        instruction_adherence=parsed.instruction_adherence,
        editing_quality=parsed.editing_quality,
        detail_preservation=parsed.detail_preservation,
        fine_score=parsed.fine_score, score_scale=score_scale,
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
    prompt_style: Optional[str] = None,
) -> RunStats:
    root = Path(root) if root is not None else default_root()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if judge_config is not None:
        cfg = {**cfg, "judge_config": str(judge_config), "judge": None}
    # One knob drives the prompt, the parser's clamp and the stamped provenance:
    # asking for 1-10 while the parser clamps to 5 would silently truncate every
    # high answer, and a row that does not record its own scale cannot later be
    # kept apart from the 1-5 pilot.
    score_scale = int(cfg.get("score_scale", DEFAULT_SCORE_SCALE))
    judge_cfg = _resolve_judge_cfg(cfg, root)
    style = resolve_prompt_style(cfg, judge_cfg, prompt_style)
    judge_cfg = {"score_scale": score_scale, **judge_cfg}
    if int(cfg.get("repeat", 1)) > 1 and judge_cfg.get("cache_dir"):
        # The adapter caches on a hash of the request payload, so a second identical
        # ask would be served from disk and the retest arm would report a variance
        # of exactly zero — a noise floor manufactured by plumbing.
        raise ValueError(
            "repeat>1 needs the response cache off; set `judge_overrides: "
            "{cache_dir: null}` in this config"
        )
    adapter, model = _adapter_for(judge_cfg, use_api=use_api, root=root, dry_run=dry_run)
    raw_dir = Path(cfg.get("raw_dir", "results/raw_responses"))
    results_dir = root / cfg.get("results_dir", "results")
    failure_log = (root / cfg["failure_log"]) if cfg.get("failure_log") else None

    targets = _build_targets(cfg, root, model, style)
    stats = RunStats()

    man_orig = manifest_for(results_dir, "scoring", model, biased=False)
    man_bias = manifest_for(results_dir, "scoring", model, biased=True)
    done = set() if dry_run else load_done_ids(man_orig, man_bias)
    workers = int(cfg.get("workers", 1))

    stats.join_misses = sum(1 for t in targets if t is _MISSING)
    considered = [t for t in targets if t is not _MISSING]
    if limit is not None:
        considered = considered[:limit]
    pending = []
    for t in considered:
        if t.result_id in done:
            stats.skipped += 1
        else:
            pending.append(t)
    if dry_run:
        stats.written = len(pending)
        return stats

    def do_one(t: ScoringTarget):
        prompt, images = build_request(t, style=style, score_scale=score_scale, root=root)
        try:
            raw = adapter.score(prompt, images)
        except Exception as exc:  # noqa: BLE001 - transient API error: log, retry next run
            return ("api_fail", t, f"{type(exc).__name__}: {exc}", None)
        raw_rel = save_raw_response(root, raw_dir, model, t.result_id, raw)
        result = build_result(t, raw, raw_rel, model=model, style=style, score_scale=score_scale)
        return ("ok", t, result, result.parse_success)

    # Worker threads do the API call; manifest writes stay on this (main) thread.
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
    p = argparse.ArgumentParser(description="Run scoring judgments (mock-first).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--use-api", action="store_true", help="allow real API adapters (M4)")
    p.add_argument(
        "--judge-config", type=Path, default=None,
        help="override the config's `judge_config:` — one arm config, five judges",
    )
    p.add_argument(
        "--prompt-style", type=str, default=None,
        help="override both the arm's `prompt_style:` and the judge's "
             "`scoring_prompt_style:` (vanilla | bias_aware | rubric_first | viescore)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    stats = run(args.config, root=args.root, dry_run=args.dry_run,
                limit=args.limit, use_api=args.use_api,
                judge_config=args.judge_config, prompt_style=args.prompt_style)
    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}scoring: {stats.summary()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
