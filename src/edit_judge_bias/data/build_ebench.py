"""Ingest EBench-18K (paper: LMM4Edit) into a SampleRecord manifest.

This is the project's human-anchor source for claim B: every edited image carries a
3-dimension MOS from 15 raters, and those three dimensions line up one-to-one with
our judge's three output fields (see `configs/data/ebench18k.yaml`).

Layout consumed:

    <images_root>/sourceimg_{h,l}/<model>/{H,L}_<task>_<idx>.jpg   # byte-identical
    <images_root>/targetimg_{h,l}/<model>/{H,L}_<task>_<idx>.jpg   # per-model output
    <meta_dir>/{train,test}_{v,e,c,yn}.json                        # labels, from GitHub

The label files are JSON arrays of `{query, response, images}` — the instruction is
embedded in the `query` prose and the MOS in the `response` prose, so both are
extracted by pattern. `images` carries absolute paths from the dataset authors' own
machine (an arbitrary prefix, then `.../editing_all/...`), which is why only the tail is
used to join.

The source's own task index is authoritative for `edit_type`; the text rules are
deliberately NOT used here (measured: they mislabel `color` 80% of the time on
these instructions and miss `super-resolution` entirely). Tasks with no honest
target class stay unclassified rather than being force-labelled.

    python -m edit_judge_bias.data.build_ebench --config configs/data/ebench18k.yaml
    python -m edit_judge_bias.data.build_ebench --config configs/data/ebench18k.yaml --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.build_pairs import build_pairs as _build_pairs_generic
from edit_judge_bias.data.edit_type_rules import classify_content_category
from edit_judge_bias.data.manifest_utils import PathLike, default_root, to_rel_posix
from edit_judge_bias.data.schema import PairRecord, SampleRecord

# "With the image editing prompt [Add a nose ring to the woman's nose], the source..."
_INSTRUCTION_RE = re.compile(r"\[([^\]]+)\]")
# "The quality score is 46.35"
_SCORE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
# ".../sourceimg_h/model00/H_00_36.jpg"
_IMAGE_RE = re.compile(r"(sourceimg|targetimg)_(?P<level>[hl])/(?P<model>[^/]+)/(?P<stem>[^/]+)\.\w+$")

MOS_DIMENSIONS = ("mos_quality", "mos_alignment", "mos_preservation")


@dataclass
class EBenchBuildStats:
    n_label_rows: int = 0
    n_items: int = 0              # distinct (level, model, stem) after the join
    n_samples: int = 0
    n_pairs: int = 0
    n_pairs_with_preference: int = 0
    dropped_unmapped: Counter = field(default_factory=Counter)     # task -> n
    dropped_contaminated: Counter = field(default_factory=Counter)
    missing_images: List[str] = field(default_factory=list)
    incomplete_joins: List[str] = field(default_factory=list)
    dup_groups: int = 0

    def summary(self) -> str:
        return (
            f"label_rows={self.n_label_rows} items={self.n_items} samples={self.n_samples} "
            f"pairs={self.n_pairs} (gt={self.n_pairs_with_preference}) "
            f"unmapped={sum(self.dropped_unmapped.values())} "
            f"contaminated={sum(self.dropped_contaminated.values())} "
            f"missing_images={len(self.missing_images)} dup_groups={self.dup_groups}"
        )


def load_config(config_path: PathLike) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _read_label_file(path: Path) -> List[dict]:
    """Upstream ships a JSON array; tolerate JSONL in case a mirror differs."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_instruction(query: str) -> Optional[str]:
    """First bracketed span of the prompt prose (the later bracket is the format hint)."""
    m = _INSTRUCTION_RE.search(query)
    return m.group(1).strip() if m else None


def parse_score(response: str) -> Optional[float]:
    m = _SCORE_RE.search(response)
    return float(m.group(1)) if m else None


def parse_qa(response: str) -> Optional[bool]:
    lowered = response.strip().lower()
    if "yes" in lowered:
        return True
    if "no" in lowered:
        return False
    return None


def parse_image_ref(path: str) -> Optional[Tuple[str, str, str]]:
    """(level, model, stem) from an author-absolute image path, or None."""
    m = _IMAGE_RE.search(path.replace("\\", "/"))
    return (m.group("level"), m.group("model"), m.group("stem")) if m else None


def task_index(stem: str) -> Optional[str]:
    """`H_00_36` -> `H_00`, the key of the task table in the config."""
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 3 else None


def load_labels(cfg: dict, root: Path, stats: EBenchBuildStats) -> Dict[Tuple[str, str, str], dict]:
    """Join every configured label dimension on (level, model, stem)."""
    meta_dir = root / cfg["meta_dir"]
    items: Dict[Tuple[str, str, str], dict] = defaultdict(dict)
    for dim, names in cfg["label_files"].items():
        for name in names:
            for row in _read_label_file(meta_dir / name):
                stats.n_label_rows += 1
                ref = parse_image_ref(row["images"][0])
                if ref is None:
                    continue
                rec = items[ref]
                rec.setdefault("instruction", parse_instruction(row["query"]))
                rec[dim] = (parse_qa(row["response"]) if dim == "human_qa_pass"
                            else parse_score(row["response"]))
    stats.n_items = len(items)

    # Join integrity is fatal by design (protocol §2): a dimension silently missing
    # for some rows would put a hole in exactly the human anchor we are buying.
    required = set(cfg["label_files"]) | {"instruction"}
    for key, rec in items.items():
        missing = [d for d in required if rec.get(d) is None]
        if missing:
            stats.incomplete_joins.append(f"{'/'.join(key)}: missing {','.join(sorted(missing))}")
    return items


def _dup_groups(originals: List[Path], root: Path) -> Dict[str, str]:
    """rel-path -> group id, for originals that are byte-identical to another one.

    EBench reuses 18 photographs across two tasks with different instructions. Our
    pair builder keys on (original, instruction) so it will not mis-pair them, but
    any "same photo, different task" analysis needs to know.
    """
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for p in originals:
        if p.is_file():
            by_hash[hashlib.md5(p.read_bytes()).hexdigest()].append(to_rel_posix(p, root))
    out: Dict[str, str] = {}
    for i, (digest, rels) in enumerate(sorted(by_hash.items())):
        if len(rels) > 1:
            for rel in rels:
                out[rel] = f"dup{i:03d}"
    return out


def build_samples(
    cfg: dict, root: PathLike | None = None, *, stats: Optional[EBenchBuildStats] = None
) -> Tuple[List[SampleRecord], EBenchBuildStats]:
    root = Path(root) if root is not None else default_root()
    stats = stats if stats is not None else EBenchBuildStats()
    images_root = root / cfg["images_root"]
    tasks = cfg["tasks"]
    exclude_contaminated = bool(cfg.get("exclude_contaminated", True))
    global_types = set(cfg.get("global_edit_types") or [])
    prefix = cfg.get("id_prefix", "ebench")
    source = cfg.get("source_dataset", "EBench-18K")

    items = load_labels(cfg, root, stats)

    kept: List[Tuple[Tuple[str, str, str], dict, dict]] = []
    for key, rec in sorted(items.items()):
        level, model, stem = key
        task = tasks.get(task_index(stem) or "")
        if task is None or task.get("edit_type") is None:
            stats.dropped_unmapped[task["name"] if task else stem] += 1
            continue
        if exclude_contaminated and task.get("contaminated"):
            stats.dropped_contaminated[task["name"]] += 1
            continue
        kept.append((key, rec, task))

    # Source images are byte-identical across all 17 model dirs, so every record
    # points at model00's copy — one referenced file instead of seventeen.
    def original_for(level: str, stem: str) -> Path:
        return images_root / f"sourceimg_{level}" / "model00" / f"{stem}.jpg"

    dup_map = _dup_groups(
        sorted({original_for(k[0], k[2]) for k, _, _ in kept}), root
    ) if cfg.get("stamp_dup_groups", True) else {}
    stats.dup_groups = len(set(dup_map.values()))

    samples: List[SampleRecord] = []
    for (level, model, stem), rec, task in kept:
        orig = original_for(level, stem)
        edited = images_root / f"targetimg_{level}" / model / f"{stem}.jpg"
        if not orig.is_file() or not edited.is_file():
            # Fail-soft on a single missing asset (a whole missing dimension is not).
            stats.missing_images.append(f"{level}/{model}/{stem}")
            continue
        edit_type = task["edit_type"]
        instruction = rec["instruction"]
        orig_rel = to_rel_posix(orig, root)
        mos = [rec[d] for d in MOS_DIMENSIONS]
        metadata = {
            "human_score_kind": "human_rating",
            "ebench_task": task["name"],
            "ebench_level": "low" if level == "l" else "high",
            "human_qa_pass": rec.get("human_qa_pass"),
            **{d: rec[d] for d in MOS_DIMENSIONS},
        }
        if orig_rel in dup_map:
            metadata["dup_group"] = dup_map[orig_rel]
        samples.append(SampleRecord(
            sample_id=f"{prefix}_{stem}_{model}",
            source_dataset=source,
            edit_type=edit_type,
            content_category=("global" if edit_type in global_types
                              else classify_content_category(instruction)),
            original_image_path=orig_rel,
            instruction=instruction,
            edit_model=model,
            edited_image_path=to_rel_posix(edited, root),
            # Mean of the three dimensions, rescaled onto the 0-1 range the rest of
            # the pool uses (ImagenHub's human_score is already 0-1). The divisor is
            # the *documented* 0-100 rating scale from the prompt, not the observed
            # range, so the value does not shift when a config excludes some tasks.
            # Without this, `pair_quality_gap` would be silently incomparable across
            # sources and a global "|gap| > 0.2 is decisive" filter would pass every
            # EBench pair while meaning something real on ImagenHub. Raw MOS is kept
            # per dimension in metadata, so nothing is lost.
            human_score=round(sum(mos) / len(mos) / 100.0, 6),
            metadata=metadata,
        ))
    stats.n_samples = len(samples)
    return samples, stats


def build_pairs(
    samples: List[SampleRecord], cfg: dict, *, stats: Optional[EBenchBuildStats] = None
) -> List[PairRecord]:
    """Pairs across editors within each (original, instruction) turn.

    EBench is by far the densest pair source in the pool — 17 editors on every one
    of the 590 originals, so C(17,2)=136 candidate pairs per turn — and because
    every sample carries a MOS, *every* pair gets a human ground-truth preference.
    Grouping/capping is delegated to `build_pairs.build_pairs`; this wrapper re-ids
    into the `pair_eb_*` namespace (the generic builder restarts its counter at 1,
    which would collide with I2EBench's `pair_000001`) and attaches the ground truth,
    exactly as `build_imagenhub.build_pairs` does.
    """
    pairs_cfg = dict(cfg.get("pairs", {}) or {})
    tie_eps = float(pairs_cfg.pop("tie_epsilon", 1e-9))
    raw = _build_pairs_generic(samples, pairs_cfg)

    by_id = {s.sample_id: s for s in samples}
    out: List[PairRecord] = []
    n_with_pref = 0
    for i, pair in enumerate(raw, start=1):
        a, b = by_id[pair.sample_id_a], by_id[pair.sample_id_b]
        preference: Optional[str] = None
        gap: Optional[float] = None
        if a.human_score is not None and b.human_score is not None:
            delta = a.human_score - b.human_score
            gap = round(abs(delta), 6)
            preference = "a" if delta > tie_eps else ("b" if delta < -tie_eps else "tie")
            n_with_pref += 1
        out.append(pair.model_copy(update={
            "pair_id": f"pair_eb_{i:06d}",
            "ground_truth_preference": preference,
            "pair_quality_gap": gap,
        }))
    if stats is not None:
        stats.n_pairs = len(out)
        stats.n_pairs_with_preference = n_with_pref
    return out


def build(
    config_path: PathLike,
    *,
    root: PathLike | None = None,
    dry_run: bool = False,
) -> Tuple[List[SampleRecord], List[PairRecord], EBenchBuildStats]:
    cfg = load_config(config_path)
    root = Path(root) if root is not None else default_root()
    samples, stats = build_samples(cfg, root)
    pairs = build_pairs(samples, cfg, stats=stats)
    if stats.incomplete_joins:
        raise ValueError(
            f"{len(stats.incomplete_joins)} label rows have an incomplete join, e.g. "
            + "; ".join(stats.incomplete_joins[:3])
        )
    if not dry_run:
        io.write_jsonl(root / cfg["output"]["samples_full"], samples)
        if cfg["output"].get("pairs_full"):
            io.write_jsonl(root / cfg["output"]["pairs_full"], pairs)
        # Write-then-validate, same contract as build_genaibench.
        missing = [s.sample_id for s in samples
                   if not (root / s.original_image_path).is_file()
                   or not (root / s.edited_image_path).is_file()]
        if missing:
            raise ValueError(f"{len(missing)} written records point at missing files, "
                             f"e.g. {missing[:3]}")
    return samples, pairs, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the EBench-18K sample manifest.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    samples, pairs, stats = build(args.config, root=args.root, dry_run=args.dry_run)
    print(f"[ebench18k] {stats.summary()}")
    for label, counter in (("unmapped", stats.dropped_unmapped),
                           ("contaminated", stats.dropped_contaminated)):
        if counter:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
            print(f"  dropped ({label}): {detail}")
    by_type = Counter(s.edit_type.value for s in samples)
    print("  by edit_type: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    cfg = load_config(args.config)
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} samples={len(samples)} -> {cfg['output']['samples_full']}")
    if cfg["output"].get("pairs_full"):
        print(f"{verb} pairs={len(pairs)} -> {cfg['output']['pairs_full']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
