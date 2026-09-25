"""One table: per cue, how many judges or cells show an effect in each of three layers.

A per-cue summary of the signed-shift tests, not the paper's main invariance table: that
one (per-dimension absolute change against the `sham` and retest noise floors) is built
by `build_noise_floor_table`.

Reads frozen CSVs only, so it runs last in `scripts/06_build_tables.sh` and adds
no estimate of its own.  Every count follows a rule that already governs its source table:

claim A (`claim_a.csv`)
    A judge counts when its cell is BH significant AND outside the `sham` equivalence
    bound: cells inside the bound (the `~` cells) cannot be
    claimed as effects.  `claim_a_effects_ci_crossing_zero` says how many of the counted
    cells still have a mean CI containing 0 (the Wilcoxon test and the bootstrap mean CI
    disagree on those).

pairwise (`pairwise_one_sided.csv`)
    BH-significant `bias_advantage` cells.  The file holds two collections -- the main grid
    and FILL v2 -- and a cue's cells must all come from one; `pairwise_reference` is the
    baseline they were measured against (the rows' own `reference` column).

claim B (`claim_b.csv`)
    Cells whose delta-rho cluster-bootstrap CI excludes 0 -- the rank half, which is what
    claim B's headline rests on.  Same one-collection rule; the accuracy half stays in the
    source table.

The published roster only.  The A5 judge is its own declared family and never enters this
table; a row from any other judge, or a family stamped with a roster tag, is refused.

    python -m edit_judge_bias.experiments.build_main_table --metrics-dir results/v2/metrics
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.experiments.aggregate_results import PUBLISHED_ROSTER
from edit_judge_bias.experiments.build_claim_tables import (
    BASELINE_LABEL,
    CONTROL_BIAS,
    MAIN_COLLECTION,
    write_csv,
)
from edit_judge_bias.experiments.build_fill_tables import NOT_TABULATED

#: The cue injection sites (see docs/CUES.md), minus `position` (pairwise only, no claim A row).
CUE_CLASS: Dict[str, str] = {
    "brightness": "pixel", "saturation": "pixel", "watermark": "pixel",
    "text_overlay": "pixel", "padding": "pixel", "aesthetic_filter": "pixel",
    "zoom_inset": "content", "detail_caption": "content",
    "distraction": "content", "region_annotation": "content",
    "bandwagon": "prompt", "model_name": "prompt",
}
CLASS_ORDER: Tuple[str, ...] = ("pixel", "content", "prompt")


def _read(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; build it with scripts/06_build_tables.sh")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _true(value) -> bool:
    return str(value).strip() == "True"


def _num(value) -> Optional[float]:
    text = "" if value is None else str(value).strip()
    return None if text == "" else float(text)


def _direction(values: Iterable[Optional[float]]) -> str:
    vals = [v for v in values if v is not None]
    up, down = any(v > 0 for v in vals), any(v < 0 for v in vals)
    return "mixed" if up and down else "up" if up else "down" if down else ""


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 4) if vals else None


def _published_only(rows: Sequence[dict], label: str) -> None:
    stray = sorted({r["judge_model"] for r in rows} - set(PUBLISHED_ROSTER))
    if stray:
        raise ValueError(f"{label} holds judges outside the published roster: {stray}")
    tagged = sorted({r.get("family") or "" for r in rows
                     if "roster]" in (r.get("family") or "")})
    if tagged:
        raise ValueError(f"{label} holds rows from another roster's family: {tagged}")


def _one_collection(cue: str, rows: Sequence[dict], label: str) -> Tuple[List[dict], str]:
    """A cue's cells and the reference they were measured against; one collection only."""
    cells = [r for r in rows if r["bias_type"] == cue]
    collections = sorted({r.get("collection") or MAIN_COLLECTION for r in cells})
    if len(collections) > 1:
        raise ValueError(f"{label}: {cue!r} has cells in {collections}; a cue's cells come "
                         "from exactly one collection")
    references = sorted({r.get("reference") or BASELINE_LABEL for r in cells})
    if len(references) > 1:
        raise ValueError(f"{label}: {cue!r} mixes the references {references}")
    return cells, (references[0] if cells else "")


def build_main_table(metrics_dir: Path) -> List[dict]:
    metrics_dir = Path(metrics_dir)
    claim_a = [r for r in _read(metrics_dir / "claim_a.csv") if r["bias_type"] != CONTROL_BIAS]
    pairwise = _read(metrics_dir / "pairwise_one_sided.csv")
    claim_b = _read(metrics_dir / "claim_b.csv")
    for rows, label in ((claim_a, "claim_a.csv"), (pairwise, "pairwise_one_sided.csv"),
                        (claim_b, "claim_b.csv")):
        _published_only(rows, label)

    cues = sorted({r["bias_type"] for r in claim_a})
    unknown = [c for c in cues if c not in CUE_CLASS]
    if unknown:
        raise ValueError(f"no cue class declared for {unknown}")

    out: List[dict] = []
    for cue in cues:
        a = [r for r in claim_a if r["bias_type"] == cue]
        counted = [r for r in a
                   if _true(r["significant_bh"]) and not _true(r["inside_placebo_bound"])]
        row = {
            "cue": cue,
            "cue_class": CUE_CLASS[cue],
            "claim_a_effect_judges": len(counted),
            "claim_a_judges": len({r["judge_model"] for r in a}),
            "claim_a_direction": _direction(_num(r["mean_shift"]) for r in counted),
            "claim_a_median_shift": _median(_num(r["mean_shift"]) for r in a),
            "claim_a_effects_ci_crossing_zero": sum(
                1 for r in counted if _num(r["ci_low"]) <= 0 <= _num(r["ci_high"])),
        }

        pw, pw_ref = _one_collection(cue, pairwise, "pairwise")
        sig = [r for r in pw if _true(r["significant_bh_advantage"])]
        row.update({
            "pairwise_significant_cells": len(sig) if pw else None,
            "pairwise_cells": len(pw) if pw else None,
            "pairwise_direction": _direction(_num(r["bias_advantage"]) for r in sig),
            "pairwise_median_advantage": _median(_num(r["bias_advantage"]) for r in pw),
            "pairwise_reference": pw_ref,
        })

        cb, cb_ref = _one_collection(cue, claim_b, "claim B")
        moved = [r for r in cb if _true(r["rho_ci_excludes_zero"])]
        row.update({
            "claim_b_rank_changed_cells": len(moved) if cb else None,
            "claim_b_cells": len(cb) if cb else None,
            "claim_b_direction": _direction(_num(r["spearman_delta"]) for r in moved),
            "claim_b_median_delta_rho": _median(_num(r["spearman_delta"]) for r in cb),
            "claim_b_reference": cb_ref,
        })

        notes = []
        if not pw:
            notes.append("pairwise: " + NOT_TABULATED.get(cue, "no cells collected"))
        if not cb:
            notes.append("claim B: no cells collected")
        row["note"] = "; ".join(notes)
        out.append(row)

    rank = {c: i for i, c in enumerate(CLASS_ORDER)}
    out.sort(key=lambda r: (rank[r["cue_class"]], -r["claim_a_effect_judges"], r["cue"]))
    return out


def main(argv: list[str] | None = None) -> int:
    root = default_root()
    ap = argparse.ArgumentParser(description="Build the one-table per-cue summary.")
    ap.add_argument("--metrics-dir", type=Path, default=root / "results" / "v2" / "metrics")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rows = build_main_table(args.metrics_dir)
    out = args.out or (args.metrics_dir / "main_table.csv")
    write_csv(out, rows)

    def cell(k, n):
        return "n/a" if n in (None, "") else f"{k}/{n}"

    print(f"{'cue':<18}{'class':<9}{'claim A':<10}{'pairwise':<10}{'claim B':<9}")
    for r in rows:
        print(f"{r['cue']:<18}{r['cue_class']:<9}"
              f"{cell(r['claim_a_effect_judges'], r['claim_a_judges']):<10}"
              f"{cell(r['pairwise_significant_cells'], r['pairwise_cells']):<10}"
              f"{cell(r['claim_b_rank_changed_cells'], r['claim_b_cells']):<9}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
