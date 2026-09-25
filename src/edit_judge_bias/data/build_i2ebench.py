"""Ingest the extracted I2EBench / EditBench tree into SampleRecord manifests.

Layout consumed:

    <editbench_root>/EditData/<category>/<category>.json   # per-image instruction + type
    <editbench_root>/EditData/<category>/input/<stem>.*     # original image
    <editbench_root>/EditResult/<category>/<model>/<stem>.* # edited image (ext may differ)

Mapping (category -> edit_type, type -> content_category, instruction field) is
config-driven via configs/data/i2ebench.yaml. Edited images are matched by *stem*
because some models re-encode to a different extension.

    python -m edit_judge_bias.data.build_i2ebench --config configs/data/i2ebench.yaml
    python -m edit_judge_bias.data.build_i2ebench --config configs/data/i2ebench.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import (
    PathLike,
    default_root,
    find_by_stem,
    slug,
    to_rel_posix,
)
from edit_judge_bias.data.schema import SampleRecord


@dataclass
class BuildStats:
    n_categories: int = 0
    n_entries: int = 0
    n_samples: int = 0
    missing_input: List[str] = field(default_factory=list)
    missing_edited: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"categories={self.n_categories} entries={self.n_entries} "
            f"samples={self.n_samples} missing_input={len(self.missing_input)} "
            f"missing_edited={len(self.missing_edited)}"
        )


def load_config(config_path: PathLike) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _discover_models(result_cat_dir: Path) -> List[str]:
    if not result_cat_dir.is_dir():
        return []
    return sorted(p.name for p in result_cat_dir.iterdir() if p.is_dir())


def _content_category(entry: dict, cfg: dict) -> str:
    raw = entry.get("type")
    if raw:
        mapped = cfg["content_category_map"].get(str(raw).lower())
        if mapped:
            return mapped
    return cfg["default_content_category"]


def build_all_samples(
    cfg: dict, root: PathLike | None = None
) -> Tuple[List[SampleRecord], BuildStats]:
    """Walk every configured category and emit one SampleRecord per (image, model)."""
    root = Path(root) if root is not None else default_root()
    # absolute() rather than resolve(): a symlinked tmp_data/ must keep yielding
    # `tmp_data/...` manifest paths (see manifest_utils.to_rel_posix).
    eb_root = (root / cfg["editbench_root"]).absolute()
    edit_data = eb_root / "EditData"
    edit_result = eb_root / "EditResult"
    instr_field = cfg.get("instruction_field", "ori_exp")

    samples: List[SampleRecord] = []
    stats = BuildStats()

    for category, edit_type in cfg["edit_type_map"].items():
        cat_json = edit_data / category / f"{category}.json"
        input_dir = edit_data / category / "input"
        result_cat_dir = edit_result / category
        if not cat_json.is_file():
            continue
        stats.n_categories += 1

        models = cfg.get("edit_models") or _discover_models(result_cat_dir)
        entries = json.loads(cat_json.read_text(encoding="utf-8"))

        for entry in entries.values():
            stats.n_entries += 1
            image_name = entry["image"]
            stem = Path(image_name).stem
            instruction = entry.get(instr_field) or entry.get("ori_exp")
            if not instruction:
                continue
            content_category = _content_category(entry, cfg)

            orig = find_by_stem(input_dir, stem)
            if orig is None:
                stats.missing_input.append(f"{category}/{image_name}")
                continue
            orig_rel = to_rel_posix(orig, root)

            for model in models:
                edited = find_by_stem(result_cat_dir / model, stem)
                if edited is None:
                    stats.missing_edited.append(f"{category}/{model}/{stem}")
                    continue
                sample_id = f"i2e_{category}_{stem}_{slug(model)}"
                samples.append(
                    SampleRecord(
                        sample_id=sample_id,
                        source_dataset=cfg.get("source_dataset", "I2EBench"),
                        edit_type=edit_type,
                        content_category=content_category,
                        original_image_path=orig_rel,
                        instruction=instruction,
                        edit_model=model,
                        edited_image_path=to_rel_posix(edited, root),
                    )
                )

    stats.n_samples = len(samples)
    return samples, stats


def _group_key(rec: SampleRecord) -> Tuple[str, str]:
    """A pilot group is one (original image, instruction) across edit models."""
    return (rec.original_image_path.as_posix(), rec.instruction)


def select_pilot(samples: List[SampleRecord], cfg: dict) -> List[SampleRecord]:
    """Balanced subsample: pick N groups per edit_type, cap models per group.

    Deterministic given `pilot.seed`. Groups are picked per edit_type so all six
    edit tasks are represented even though low-level dominates the raw counts.
    """
    pilot_cfg = cfg.get("pilot", {})
    seed = int(pilot_cfg.get("seed", 42))
    groups_per_type = int(pilot_cfg.get("groups_per_edit_type", 8))
    max_models = int(pilot_cfg.get("max_models_per_group", 10))
    rng = random.Random(seed)

    # group_key -> list of samples; remember insertion order for determinism.
    groups: Dict[Tuple[str, str], List[SampleRecord]] = defaultdict(list)
    group_edit_type: Dict[Tuple[str, str], str] = {}
    for rec in samples:
        key = _group_key(rec)
        groups[key].append(rec)
        group_edit_type[key] = rec.edit_type.value

    # Bucket group keys by edit_type, then seeded-shuffle and take the first N.
    by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for key, et in group_edit_type.items():
        by_type[et].append(key)

    selected: List[SampleRecord] = []
    for et in sorted(by_type):
        keys = sorted(by_type[et])  # stable base order before shuffling
        rng.shuffle(keys)
        for key in keys[:groups_per_type]:
            members = sorted(groups[key], key=lambda r: r.edit_model)
            selected.extend(members[:max_models])
    return selected


def build(
    config_path: PathLike,
    *,
    root: PathLike | None = None,
    dry_run: bool = False,
    overwrite: bool = True,
) -> Tuple[List[SampleRecord], List[SampleRecord], BuildStats]:
    cfg = load_config(config_path)
    root = Path(root) if root is not None else default_root()
    samples, stats = build_all_samples(cfg, root)
    pilot = select_pilot(samples, cfg)

    if not dry_run:
        out_full = root / cfg["output"]["samples_full"]
        out_pilot = root / cfg["output"]["samples_pilot"]
        if overwrite or not out_full.exists():
            io.write_jsonl(out_full, samples)
        if overwrite or not out_pilot.exists():
            io.write_jsonl(out_pilot, pilot)
    return samples, pilot, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build I2EBench sample manifests.")
    p.add_argument("--config", type=Path, required=True, help="data config YAML")
    p.add_argument("--root", type=Path, default=None, help="project root (default: CWD)")
    p.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="skip writing a manifest that already exists (resumable).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    samples, pilot, stats = build(
        args.config,
        root=args.root,
        dry_run=args.dry_run,
        overwrite=not args.no_overwrite,
    )
    print(f"[i2ebench] {stats.summary()}")
    verb = "would write" if args.dry_run else "wrote"
    cfg = load_config(args.config)
    print(f"{verb} full={len(samples)} -> {cfg['output']['samples_full']}")
    print(f"{verb} pilot={len(pilot)} -> {cfg['output']['samples_pilot']}")
    if stats.missing_edited:
        print(f"  note: {len(stats.missing_edited)} missing edited images skipped (logged)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
