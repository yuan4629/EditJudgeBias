"""The paper's main invariance table: per-dimension absolute change against two noise floors.

Every cell is one (judge, cue, rating dimension). Its statistic is the mean absolute
paired change in that 1-10 rating between the un-cued and the cued answer, multiplied by
10 (so a cell reads as a share of the scale):

    I = 10 * mean_i |score_cued(i) - score_uncued(i)|

Each judge has two floors per dimension, estimated with the same statistic on changes no
cue caused:

``sham``
    the zero-dose control (JPEG round trip) against the un-cued answer, same items.
``retest``
    the second ask of an identical un-cued question against the first
    (the 200-item test-retest arm; see ``scoring_metrics.retest_pairs``).

Floor = the 20th percentile of the statistic's bootstrap distribution (2000 resamples).
A cue cell's lower bound = the 2.5th percentile of its own bootstrap distribution (the
lower end of the 95% percentile interval). Every bootstrap uses seed 42 and the
generator of ``stats.bootstrap_ci``. The marker counts the floors the lower bound
strictly exceeds:

    bold   both floors
    plain  exactly one
    gray   neither

Cells are measured on the complete-case rows ``claim_a.csv`` and
``claim_a_by_dimension.csv`` use (all three ratings parsed), so the statistic equals the
``asc`` column of ``claim_a_by_dimension.csv`` times 10. The summary counts exclude
``zoom_inset``, which the preservation analysis treats as borderline
(``BORDERLINE_CUES``). Displayed values round half up to one decimal.

    python -m edit_judge_bias.experiments.build_noise_floor_table \\
        --results-dir results/v2 \\
        --samples data/manifests/samples_judge_v2.jsonl \\
        --subset-filter subset_block=breadth
"""

from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.experiments.aggregate_results import (
    PUBLISHED_ROSTER,
    ROSTERS,
    _judges,
    sample_ids_matching,
)
from edit_judge_bias.experiments.build_claim_a_dimensions import DIMENSIONS, _complete_case
from edit_judge_bias.experiments.build_claim_tables import CONTROL_BIAS, _read_judge, write_csv
from edit_judge_bias.metrics.scoring_metrics import compute_score_shifts, retest_pairs
from edit_judge_bias.metrics.stats import bootstrap_means

#: Name of the test-retest floor row (it has no bias_type of its own).
RETEST = "retest"
FLOORS = (CONTROL_BIAS, RETEST)

#: Cues shown in the table but left out of the summary counts: `zoom_inset` falls below
#: the `sham` pass rate on two of three preservation validators.
BORDERLINE_CUES = frozenset({"zoom_inset"})

N_BOOT = 2000
SEED = 42
FLOOR_QUANTILE = 0.20
CI_LOW_QUANTILE = 0.025
SCALE = 10.0

MARKERS = {2: "bold", 1: "plain", 0: "gray"}


def display(value: float) -> str:
    """One decimal, half up (4.55 -> 4.6), from the value's shortest repr."""
    return str(Decimal(repr(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _cell(values: Sequence[float]) -> Optional[dict]:
    """Point statistic and the two bootstrap quantiles the table reads."""
    scaled = np.abs(np.asarray(values, dtype=float)) * SCALE
    means = bootstrap_means(scaled, n_boot=N_BOOT, seed=SEED)
    if means is None:
        return None
    return {
        "n": int(scaled.size),
        "stat": float(scaled.mean()),
        "boot_q20": float(np.quantile(means, FLOOR_QUANTILE)),
        "boot_lo": float(np.quantile(means, CI_LOW_QUANTILE)),
    }


def _changes(originals, biased, dim: str) -> Dict[str, List[float]]:
    """bias_type -> paired changes in one dimension, plus the retest changes."""
    out = {st.bias_type: list(st.shifts)
           for st in compute_score_shifts(originals, biased, score_field=dim)}
    retest: List[float] = []
    for scores in retest_pairs(originals, score_field=dim).values():
        retest.extend(b - a for a, b in zip(scores, scores[1:]))
    if retest:
        out[RETEST] = retest
    return out


def build_noise_floor_table(
    results_dir: Path,
    keep_sample_ids: Optional[set] = None,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    def keep(results):
        if keep_sample_ids is None:
            return results
        return [r for r in results if r.sample_id in keep_sample_ids]

    rows: List[dict] = []
    for model in _judges(results_dir, "scoring", roster):
        originals = _complete_case(
            keep(_read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl")))
        biased = _complete_case(
            keep(_read_judge(results_dir / "biased_judgments" / f"scoring__{model}.jsonl")))
        if not biased:
            continue
        for dim in DIMENSIONS:
            cells = {k: _cell(v) for k, v in _changes(originals, biased, dim).items()}
            missing = [f for f in FLOORS if cells.get(f) is None]
            if missing:
                raise ValueError(f"{model} / {dim}: no {' or '.join(missing)} changes; "
                                 "both floors are needed to mark any cell")
            f_sham = cells[CONTROL_BIAS]["boot_q20"]
            f_retest = cells[RETEST]["boot_q20"]
            for name, c in sorted(cells.items()):
                if c is None:
                    continue
                row = {
                    "judge_model": model,
                    "row": name,
                    "row_kind": "floor" if name in FLOORS else "cue",
                    "dimension": dim,
                    "n": c["n"],
                    "stat_x10": round(c["stat"], 4),
                    "stat_x10_display": display(c["stat"]),
                    "boot_q20": round(c["boot_q20"], 4),
                    "boot_ci_low": round(c["boot_lo"], 4),
                    "sham_floor": round(f_sham, 4),
                    "sham_floor_display": display(f_sham),
                    "retest_floor": round(f_retest, 4),
                    "retest_floor_display": display(f_retest),
                    "floors_cleared": None,
                    "marker": None,
                    "counted": None,
                }
                if name not in FLOORS:
                    cleared = int(c["boot_lo"] > f_sham) + int(c["boot_lo"] > f_retest)
                    row["floors_cleared"] = cleared
                    row["marker"] = MARKERS[cleared]
                    row["counted"] = name not in BORDERLINE_CUES
                rows.append(row)
    return sorted(rows, key=lambda r: (r["row_kind"], r["row"], r["dimension"],
                                       r["judge_model"]))


def summarize(rows: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    """Bold cells per dimension and per cue, over the counted cues."""
    counted = [r for r in rows if r["row_kind"] == "cue" and r["counted"]]
    by_dim = {d: sum(1 for r in counted if r["dimension"] == d and r["marker"] == "bold")
              for d in DIMENSIONS}
    by_dim_n = {d: sum(1 for r in counted if r["dimension"] == d) for d in DIMENSIONS}
    cues = sorted({r["row"] for r in rows if r["row_kind"] == "cue"})
    by_cue = {c: sum(1 for r in rows if r["row"] == c and r["marker"] == "bold")
              for c in cues}
    by_cue_n = {c: sum(1 for r in rows if r["row"] == c) for c in cues}
    return {"bold_by_dimension": by_dim, "cells_by_dimension": by_dim_n,
            "bold_by_cue": by_cue, "cells_by_cue": by_cue_n}


def _print_summary(rows: Sequence[dict]) -> None:
    s = summarize(rows)
    excluded = ", ".join(sorted(BORDERLINE_CUES))
    print(f"=== noise-floor criterion: bold cells (excluding {excluded}) ===")
    for d in DIMENSIONS:
        print(f"  {d:<24}{s['bold_by_dimension'][d]:>4} / {s['cells_by_dimension'][d]}")
    total = sum(s["bold_by_dimension"].values())
    total_n = sum(s["cells_by_dimension"].values())
    print(f"  {'total':<24}{total:>4} / {total_n}")
    print("=== bold cells per cue (all dimensions and judges) ===")
    for c, k in s["bold_by_cue"].items():
        tag = "  (borderline, not counted)" if c in BORDERLINE_CUES else ""
        print(f"  {c:<24}{k:>4} / {s['cells_by_cue'][c]}{tag}")


def main(argv: Optional[List[str]] = None) -> int:
    root = default_root()
    ap = argparse.ArgumentParser(
        description="Build the per-dimension invariance table with sham and retest floors.")
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--samples", type=Path,
                    default=root / "data" / "manifests" / "samples_judge_v2.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--subset-filter", action="append", default=None, metavar="KEY=VALUE")
    ap.add_argument("--roster", choices=sorted(ROSTERS), default="published",
                    help="declared judge family; anything but `published` writes to "
                         "prefixed files")
    args = ap.parse_args(argv)

    flt: Optional[Dict[str, str]] = None
    if args.subset_filter:
        flt = dict(kv.split("=", 1) for kv in args.subset_filter)
    keep = sample_ids_matching(args.samples, flt) if flt else None

    prefix = "" if args.roster == "published" else f"{args.roster}_"
    rows = build_noise_floor_table(args.results_dir, keep, roster=ROSTERS[args.roster])
    out_dir = Path(args.out_dir) if args.out_dir else args.results_dir / "metrics"
    out = out_dir / f"{prefix}invariance_noise_floor.csv"
    write_csv(out, rows)
    _print_summary(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
