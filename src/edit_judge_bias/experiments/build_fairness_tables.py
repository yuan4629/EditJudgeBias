"""Freeze the D-class fairness result into `attribute_gaps.csv`.

Reads `results/v2_fairness/raw_judgments/scoring__*.jsonl` (the D arm has its own results tree
— see the header of `configs/experiment/scoring_fairness_v2.yaml` for why) and writes one row
per (judge_model, attribute).

BH family is per attribute, because attributes are reported separately and never pooled
(user decision, 2026-07-30). The `mde_sd` column carries the pre-registered reading of a null:
at n=41 the design resolves ~0.44 SD, so a non-significant row means "no gap detected above
that", not "this judge is fair".

    bash scripts/fairness_dg.sh
    python -m edit_judge_bias.experiments.build_fairness_tables --results-dir results/v2_fairness
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_claim_tables import attach_bh, write_csv
from edit_judge_bias.metrics.fairness_metrics import (
    compute_attribute_gaps,
    keys_from_manifest,
    minimum_detectable_effect,
)


def _load(results_dir: Path) -> List[JudgeResult]:
    out: List[JudgeResult] = []
    for p in sorted((results_dir / "raw_judgments").glob("scoring__*.jsonl")):
        out.extend(io.read_jsonl(p, JudgeResult))
    return out


def build(
    *,
    root: Optional[Path] = None,
    results_dir: str = "results/v2_fairness",
    samples: str = "data/manifests/samples_fairness_judge_v2.jsonl",
    out_name: str = "attribute_gaps.csv",
) -> dict:
    root = Path(root) if root is not None else default_root()
    rdir = root / results_dir
    results = _load(rdir)
    if not results:
        return {"rows": 0, "error": f"no scoring results under {results_dir}/raw_judgments"}

    # The pairing information lives in the manifest, not on the result rows — JudgeResult has
    # no metadata container. Without this join every row would be unkeyed and the table empty,
    # which is why the guard below is an explicit error rather than a silent 0 rows.
    samples_path = root / samples
    if not samples_path.exists():
        return {"rows": 0, "error": f"fairness sample manifest not found: {samples}"}
    keys = keys_from_manifest(io.read_jsonl(samples_path, SampleRecord))
    if not keys:
        return {"rows": 0, "error": f"{samples} carries no fairness metadata (base/attribute/variant)"}

    stats = compute_attribute_gaps(results, keys)
    rows = [s.as_row() for s in stats]
    for r in rows:
        r["mde_sd"] = round(minimum_detectable_effect(r["n"]) or 0.0, 4)

    # One family per attribute: gender and skin_tone are separate hypotheses reported side by
    # side, so correcting them together would be a family the paper never claims.
    for attribute in sorted({r["attribute"] for r in rows}):
        subset = [r for r in rows if r["attribute"] == attribute]
        attach_bh(subset, family=f"fairness_gap:{attribute}", p_key="p_value")

    out = rdir / "metrics" / out_name
    write_csv(out, rows)
    return {
        "rows": len(rows),
        "judges": sorted({r["judge_model"] for r in rows}),
        "attributes": sorted({r["attribute"] for r in rows}),
        "out": str(out.relative_to(root)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the D-class attribute-gap table.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--results-dir", default="results/v2_fairness")
    # The D-S arm has its own results tree AND its own sample manifest; both must move together
    # or the join silently produces an empty table (the keys come from the manifest, not the
    # result rows). Exposed as flags so one module serves both arms.
    ap.add_argument("--samples", default="data/manifests/samples_fairness_judge_v2.jsonl")
    ap.add_argument("--out-name", default="attribute_gaps.csv")
    a = ap.parse_args(argv)
    rep = build(root=Path(a.root) if a.root else None, results_dir=a.results_dir,
                samples=a.samples, out_name=a.out_name)
    print(rep)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
