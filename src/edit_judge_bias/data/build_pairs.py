"""Build pairwise records from a samples manifest (Milestone 1).

Pairs are formed within each (original_image_path, instruction) group, across
*different* edit_model outputs. Deterministic given a seed.

    python -m edit_judge_bias.data.build_pairs \
        --samples data/manifests/samples_pilot.jsonl \
        --out     data/manifests/pairs_pilot.jsonl \
        --config  configs/data/pairs.yaml
"""

from __future__ import annotations

import argparse
import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike
from edit_judge_bias.data.schema import PairRecord, SampleRecord

DEFAULT_CONFIG = {"seed": 42, "max_pairs_per_group": 8, "strategy": "combinations"}


def load_config(config_path: PathLike | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if config_path is not None:
        loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        cfg.update(loaded)
    return cfg


def _group_samples(
    samples: List[SampleRecord],
) -> Dict[Tuple[str, str], List[SampleRecord]]:
    groups: Dict[Tuple[str, str], List[SampleRecord]] = defaultdict(list)
    for rec in samples:
        groups[(rec.original_image_path.as_posix(), rec.instruction)].append(rec)
    return groups


def build_pairs(samples: List[SampleRecord], cfg: dict) -> List[PairRecord]:
    """Form pairwise records across distinct edit_models within each group."""
    seed = int(cfg.get("seed", 42))
    max_per_group = int(cfg.get("max_pairs_per_group", 0))
    rng = random.Random(seed)

    groups = _group_samples(samples)
    pairs: List[PairRecord] = []
    counter = itertools.count(1)

    for key in sorted(groups):  # stable order across runs
        # One sample per edit_model (drop accidental dupes), sorted for determinism.
        by_model: Dict[str, SampleRecord] = {}
        for rec in groups[key]:
            by_model.setdefault(rec.edit_model, rec)
        members = [by_model[m] for m in sorted(by_model)]
        if len(members) < 2:
            continue

        combos = list(itertools.combinations(members, 2))
        if max_per_group and len(combos) > max_per_group:
            rng.shuffle(combos)
            combos = combos[:max_per_group]
            combos.sort(key=lambda ab: (ab[0].edit_model, ab[1].edit_model))

        for a, b in combos:
            pid = f"pair_{next(counter):06d}"
            pairs.append(
                PairRecord(
                    pair_id=pid,
                    sample_id_a=a.sample_id,
                    sample_id_b=b.sample_id,
                    original_image_path=a.original_image_path,
                    instruction=a.instruction,
                    edited_image_a_path=a.edited_image_path,
                    edited_image_b_path=b.edited_image_path,
                    edit_model_a=a.edit_model,
                    edit_model_b=b.edit_model,
                    source_dataset=a.source_dataset,
                    edit_type=a.edit_type,
                )
            )
    return pairs


def build(
    samples_path: PathLike,
    out_path: PathLike,
    *,
    config_path: PathLike | None = None,
    dry_run: bool = False,
) -> List[PairRecord]:
    cfg = load_config(config_path)
    samples = io.read_jsonl(samples_path, SampleRecord)
    pairs = build_pairs(samples, cfg)
    if not dry_run:
        io.write_jsonl(out_path, pairs)
    return pairs


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build pairwise records from samples.")
    p.add_argument("--samples", type=Path, required=True, help="input samples JSONL")
    p.add_argument("--out", type=Path, required=True, help="output pairs JSONL")
    p.add_argument("--config", type=Path, default=None, help="pairs config YAML")
    p.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    pairs = build(args.samples, args.out, config_path=args.config, dry_run=args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(pairs)} pairs -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
