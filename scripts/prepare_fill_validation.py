"""FILL v2 P4: the preservation check the fill's NEW pair-member images never had.

WHY.  The fill injected 6,776 new images on 847 pair-member bases
(`data/images/biased_pair_fill/`); the other 239 bases reuse published stimuli.  Every
published validator arm saw breadth-block images only, so on those 847 bases the claim that a
cue preserved the edit rests on a design argument (same injector, same parameters), not on a
measurement.  The FILL rows of `pairwise_one_sided.csv` are built on exactly those images.

DESIGN.  110 bases drawn ONCE (seed 42) from the 847, and all EIGHT conditions on those same
110 bases -- the seven one-sided image cues plus `sham` -- so each cue's pass rate is read
against its own `sham` floor on identical pictures.  Independently drawn buckets mix the cue
with "these pictures are harder" (the WP-A3 root cause), which is why this writes a fixed
manifest for `per_bias: 0` instead of letting `run_quality_validation._select` draw.
880 validations per validator.

    PYTHONPATH=src python scripts/prepare_fill_validation.py            # free: write the manifest
    PYTHONPATH=src python scripts/prepare_fill_validation.py --mock-check  # free: plumbing, temp dir

The paid run is `run_quality_validation --config configs/experiment/quality_validation_fill_v2_p4_*.yaml
--use-api`, one validator process at a time.
Never `--dry-run` it against a published summary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SOURCE = REPO / "data" / "manifests" / "biased_pair_members_fill_v2.jsonl"
OUT = REPO / "data" / "manifests" / "quality_pair_members_fill_v2_p4.jsonl"
NEW_IMAGE_DIR = "biased_pair_fill"
CONDITIONS = ("aesthetic_filter", "detail_caption", "distraction", "region_annotation",
              "saturation", "sham", "watermark", "zoom_inset")
N_BASES = 110
SEED = 42


def select_p4(records: Sequence[dict], *, n_bases: int = N_BASES, seed: int = SEED,
              conditions: Sequence[str] = CONDITIONS,
              new_image_dir: str = NEW_IMAGE_DIR) -> List[dict]:
    """All `conditions` on `n_bases` bases whose every image is newly injected."""
    by_base: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_base[r["base_sample_id"]].append(r)
    wanted = set(conditions)
    eligible = sorted(
        base for base, rows in by_base.items()
        if {r["bias_type"] for r in rows} == wanted
        and len(rows) == len(wanted)
        and all(new_image_dir in r["biased_image_path"].replace("\\", "/") for r in rows)
    )
    if len(eligible) < n_bases:
        raise ValueError(f"only {len(eligible)} bases carry all {len(wanted)} conditions as new "
                         f"images; {n_bases} were asked for")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = set(eligible[:n_bases])
    rows = [r for base in chosen for r in by_base[base]]
    return sorted(rows, key=lambda r: (r["bias_type"], r["biased_id"]))


def _read(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def mock_check(rows: Sequence[dict], per_condition: int = 2) -> int:
    """Run the real validator runner with the mock validator on a few rows, in a temp dir."""
    from edit_judge_bias.experiments.run_quality_validation import run

    picked: List[dict] = []
    seen: Dict[str, int] = defaultdict(int)
    for r in rows:
        if seen[r["bias_type"]] < per_condition:
            picked.append(r)
            seen[r["bias_type"]] += 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = tmp_path / "p4_mock.jsonl"
        _write(manifest, picked)
        cfg = tmp_path / "p4_mock.yaml"
        cfg.write_text("\n".join([
            "samples: data/manifests/samples_full_v2.jsonl",
            f"biased: {manifest.as_posix()}",
            "per_bias: 0",
            f"seed: {SEED}",
            "judge_config: configs/judge/mock.yaml",
            f"raw_dir: {(tmp_path / 'raw').as_posix()}",
            f"out_manifest: {(tmp_path / 'validation__mock.jsonl').as_posix()}",
            f"summary_csv: {(tmp_path / 'summary.csv').as_posix()}",
            f"failure_log: {(tmp_path / 'failures.jsonl').as_posix()}",
            "workers: 1",
        ]) + "\n", encoding="utf-8")
        stats = run(cfg, root=REPO, use_api=False)
        out = tmp_path / "validation__mock.jsonl"
        n_out = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
        print(f"mock check: asked {len(picked)}, stats {stats}, rows written {n_out}")
        return 0 if n_out == len(picked) and not getattr(stats, "join_misses", 0) else 1


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mock-check", action="store_true",
                    help="also run the validator runner with the mock validator on 16 rows")
    args = ap.parse_args(argv)
    rows = select_p4(_read(SOURCE))
    _write(OUT, rows)
    bases = {r["base_sample_id"] for r in rows}
    per = defaultdict(int)
    for r in rows:
        per[r["bias_type"]] += 1
    print(f"wrote {OUT.relative_to(REPO)}: {len(rows)} rows, {len(bases)} bases, "
          + ", ".join(f"{k}={v}" for k, v in sorted(per.items())))
    return mock_check(rows) if args.mock_check else 0


if __name__ == "__main__":
    raise SystemExit(main())
