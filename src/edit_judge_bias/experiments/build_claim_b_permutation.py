"""WP-A1c: is claim B's "8 of 10 cells" more than the best of four cues looking best?

Claim B's headline is that `region_annotation` is the one cue that reliably reorders:
8 of the 10 (judge x anchor source) cells have a clustered rho interval excluding zero,
against 3 / 2 / 1 for `text_overlay` / `padding` / `brightness`. Every one of those 40
intervals is correct on its own. What was never asked is the selection question: with
four cues in the arm, how often does the *best* of four collect 8 cells when the cue
label carries no information at all?

This module answers it by permutation. The null is exchangeability of the CUE LABEL:

    for each item, randomly relabel that item's four biased scores among the four cues,
    using the SAME relabelling for all five judges.

What the design deliberately preserves, because each one is an alternative explanation
that must survive into the null rather than be destroyed by it:

* **turn structure** — items never move between turns, so the 78 anchor turns and their
  8-editor blocks are identical in every draw;
* **each item's own four scores** — a hard item stays hard, so "some pictures simply
  degrade agreement" is inside the null, not outside it;
* **cross-judge coupling** — one relabelling is shared by all five judges, so a
  coincidence that hits every judge at once stays available to the null. Permuting each
  judge independently would make 8-of-10 look far more impossible than it is, since
  five independent judges rarely agree by accident. That would be a test rigged in the
  claim's favour.

★ The statistic is NOT the published "interval excludes zero". That criterion needs a
1,000-resample cluster bootstrap per cell, i.e. 40,000 resamples per permutation, and
the permutation p-value would then be a p-value about the bootstrap's noise as much as
about the data. The statistic used here needs no interval:

``worst_cue_cells``   in how many of the 10 cells does this cue have the most negative
                      delta-rho of the four
``mean_delta_rho``    the cue's mean delta-rho across the 10 cells

    ★ MEASURED, 2026-08-17. On `claim_b.csv`'s own delta-rho values the rank criterion
    returns **8 of 10** for `region_annotation` — the same count as the published
    interval criterion, on 7 shared cells (`kimi x EBench-18K` is worst-of-four but its
    interval covers zero; `gpt-5.5 x EBench-18K` has an excluding interval but `padding`
    is worse there). On the complete-case items this test must use it returns **7 of
    10**: the one cell that moves is `kimi x EBench-18K`, where `brightness` and
    `region_annotation` are separated by 0.0011 of rho (0.0001 in the published table).
    A knife-edge cell, in both readings, in a cell whose interval covers zero either
    way. All three counts are columns in the output rather than a claim in prose.

Two p-values per statistic, and they answer different questions:

``p_label``   for a cue fixed in advance. Pools all four cues' null draws, since under
              exchangeability they are draws from one distribution.
``p_selected`` for the best-of-four, which is what a reader actually saw. This is the
              honest headline number: the paper did not pre-register `region_annotation`.

Both use the add-one estimator ``(1 + #{as extreme}) / (1 + n_perm)`` so a p-value can
never be reported as exactly 0 from a finite number of draws.

⚠️ Complete cases only. One shared relabelling requires every judge to have every cue on
the item, so items where any judge failed to parse are dropped (612 of 624 survive;
kimi-k2.5 is the whole loss). The per-cell delta-rho therefore differs slightly from
`claim_b.csv`, which uses each cell's own paired items — both numbers are in the output
and a large gap between them would mean the drop was not random.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_claim_tables import (
    ANCHOR_BIASES,
    PUBLISHED_ROSTER,
    _judges,
    main_grid_rows,
    write_csv,
)
from edit_judge_bias.metrics.scoring_metrics import (
    check_single_scale,
    is_retest_repeat,
    resolve_score_field,
)

CUES: Tuple[str, ...] = ANCHOR_BIASES

#: Permutations for the two cheap statistics. 10,000 puts the Monte-Carlo standard
#: error of a p near 0.001 at about 0.0003, which is well inside the precision anyone
#: would quote.
DEFAULT_N_PERM = 10_000


# --------------------------------------------------------------------------- #
# assembling the cells                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    """One (judge, anchor source) cell, as matrices ready to permute."""

    judge_model: str
    anchor_source: str
    #: Row indices into the global item list, so one shared permutation can be sliced.
    item_index: np.ndarray
    #: Ranked human scores, centred and normalised once — the fixed side of every rho.
    human_unit: np.ndarray
    #: (n_items, n_cues) biased scores, columns in `CUES` order.
    scores: np.ndarray
    rho_original: float
    n_items: int = 0
    turns: Tuple[str, ...] = field(default_factory=tuple)


def _rank(values: np.ndarray, axis: int = 0) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(values, axis=axis)


def _unit(vector: np.ndarray) -> np.ndarray:
    centred = vector - vector.mean()
    norm = float(np.linalg.norm(centred))
    if norm == 0.0:
        raise ValueError("a constant column has no ranking to correlate")
    return centred / norm


def spearman_columns(human_unit: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Spearman rho of every column of `scores` against the human ranks.

    Ranks with ties averaged and then correlated — not the 1 - 6*sum(d^2) shortcut,
    which is only valid without ties, and judge scores are integer sums on 3-30 where
    ties are the norm rather than the exception.
    """
    ranks = _rank(scores)
    centred = ranks - ranks.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norms > 0, (human_unit @ centred) / norms, np.nan)


def build_cells(
    samples: Sequence[SampleRecord],
    results_dir: PathLike,
    *,
    cues: Sequence[str] = CUES,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> Tuple[List[Cell], List[str]]:
    """Assemble every (judge, source) cell on the items every judge answered.

    Returns the cells and the global item id list they index into.
    """
    results_dir = Path(results_dir)
    anchor = {
        s.sample_id: s for s in samples
        if getattr(s.metadata, "anchor_source", None) and s.human_score is not None
    }
    # ⚠️ MUST go through the shared guard, not a bare glob.  This module intersects the
    # answered-item sets ACROSS judges (`complete &= ...` below), so a judge with a handful
    # of rows does not add a row here -- it empties the table.  Measured 2026-08-18: the
    # WP-A5b calibration batch put 6 breadth-block rows on disk for each of two new judges,
    # neither of which is an anchor sample, and this table went from 8 published rows to 0
    # while printing "no anchor cells found -- nothing to permute", which reads like a
    # benign message about the data rather than a destroyed headline (§5.5's p_selected
    # 0.0305 / 1.0e-4).
    models = _judges(results_dir, "scoring", roster)

    originals: Dict[str, Dict[str, float]] = {}
    biased: Dict[Tuple[str, str], Dict[str, float]] = {}
    field_name: Optional[str] = None
    for model in models:
        base = io.read_jsonl(results_dir / "raw_judgments" / f"scoring__{model}.jsonl",
                             JudgeResult)
        cued = io.read_jsonl(results_dir / "biased_judgments" / f"scoring__{model}.jsonl",
                             JudgeResult)
        check_single_scale(base + cued, label=f"claim B permutation inputs ({model})")
        field_name = field_name or resolve_score_field(base + cued)
        for r in base:
            value = getattr(r, field_name, None)
            if (r.parse_success and value is not None and not r.bias_type
                    and not is_retest_repeat(r) and r.sample_id in anchor):
                originals.setdefault(model, {})[r.sample_id] = float(value)
        for r in cued:
            value = getattr(r, field_name, None)
            if (r.parse_success and value is not None and r.bias_type in set(cues)
                    and r.sample_id in anchor):
                biased.setdefault((model, r.bias_type), {})[r.sample_id] = float(value)

    complete = set(anchor)
    for model in models:
        complete &= set(originals.get(model, {}))
        for cue in cues:
            complete &= set(biased.get((model, cue), {}))
    items = sorted(complete)
    index = {sid: i for i, sid in enumerate(items)}

    cells: List[Cell] = []
    for model in models:
        for source in sorted({anchor[i].metadata.anchor_source for i in items}):
            ids = [i for i in items if anchor[i].metadata.anchor_source == source]
            human = np.array([anchor[i].human_score for i in ids], dtype=float)
            base = np.array([originals[model][i] for i in ids], dtype=float)
            human_unit = _unit(_rank(human))
            scores = np.array(
                [[biased[(model, cue)][i] for cue in cues] for i in ids], dtype=float
            )
            rho_original = float(spearman_columns(human_unit, base.reshape(-1, 1))[0])
            cells.append(Cell(
                judge_model=model,
                anchor_source=source,
                item_index=np.array([index[i] for i in ids], dtype=int),
                human_unit=human_unit,
                scores=scores,
                rho_original=rho_original,
                n_items=len(ids),
                turns=tuple(sorted({
                    f"{anchor[i].original_image_path.as_posix()}|{anchor[i].instruction}"
                    for i in ids
                })),
            ))
    return cells, items


# --------------------------------------------------------------------------- #
# the statistics                                                               #
# --------------------------------------------------------------------------- #
def cell_deltas(cell: Cell, permutation: Optional[np.ndarray] = None) -> np.ndarray:
    """Delta-rho per cue for one cell, optionally under a relabelling.

    `permutation` is the GLOBAL (n_items, n_cues) label map; the cell slices its own
    rows out of it. Sharing one map across judges is the whole point — see the module
    docstring.
    """
    scores = cell.scores
    if permutation is not None:
        scores = np.take_along_axis(scores, permutation[cell.item_index], axis=1)
    return spearman_columns(cell.human_unit, scores) - cell.rho_original


def statistics_from_deltas(deltas: np.ndarray) -> Dict[str, np.ndarray]:
    """(n_cells, n_cues) delta-rho -> the two per-cue statistics.

    `worst_cue_cells` counts a cell for the cue with the most negative delta. A cell
    whose deltas are not all defined contributes to no cue, which keeps an undefined
    correlation from being read as "this cue won".
    """
    usable = ~np.isnan(deltas).any(axis=1)
    winners = np.argmin(np.where(usable[:, None], deltas, np.inf), axis=1)
    counts = np.zeros(deltas.shape[1], dtype=float)
    for row, winner in enumerate(winners):
        if usable[row]:
            counts[winner] += 1
    return {
        "worst_cue_cells": counts,
        "mean_delta_rho": np.nanmean(deltas, axis=0),
        "n_usable_cells": np.full(deltas.shape[1], float(usable.sum())),
    }


#: For each statistic, which tail is "the cue did more damage".
_EXTREME_SIDE = {"worst_cue_cells": "high", "mean_delta_rho": "low"}


def permutation_null(
    cells: Sequence[Cell],
    n_items: int,
    *,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Draw the null distribution of both statistics. Shape (n_perm, n_cues)."""
    rng = np.random.default_rng(seed)
    n_cues = cells[0].scores.shape[1]
    base = np.tile(np.arange(n_cues), (n_items, 1))
    out = {name: np.empty((n_perm, n_cues)) for name in _EXTREME_SIDE}
    for draw in range(n_perm):
        permutation = rng.permuted(base, axis=1)
        deltas = np.vstack([cell_deltas(c, permutation) for c in cells])
        stats = statistics_from_deltas(deltas)
        for name in out:
            out[name][draw] = stats[name]
    return out


def _p_value(null: np.ndarray, observed: float, side: str) -> float:
    """Add-one permutation p-value over a pooled null."""
    flat = null.reshape(-1)
    if side == "high":
        hits = int((flat >= observed - 1e-12).sum())
    else:
        hits = int((flat <= observed + 1e-12).sum())
    return (1 + hits) / (1 + flat.size)


# --------------------------------------------------------------------------- #
# builder                                                                      #
# --------------------------------------------------------------------------- #
def _published_counts(metrics_dir: PathLike, cues: Sequence[str]) -> Dict[str, dict]:
    """The published criterion, read from `claim_b.csv` — never recomputed here."""
    path = Path(metrics_dir) / "claim_b.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in main_grid_rows(csv.DictReader(fh)) if r["bias_type"] in set(cues)]

    # The SAME rank criterion this test uses, applied to the published per-cell item
    # sets. Reported beside the complete-case count so the two reasons a count can move
    # — the criterion and the item set — can be told apart instead of being confounded.
    won: Dict[str, int] = {cue: 0 for cue in cues}
    by_cell: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        value = r.get("spearman_delta")
        if value not in (None, ""):
            by_cell.setdefault((r["judge_model"], r["anchor_source"]), {})[r["bias_type"]] = (
                float(value)
            )
    for deltas in by_cell.values():
        if len(deltas) == len(cues):
            won[min(deltas, key=lambda c: deltas[c])] += 1

    out: Dict[str, dict] = {}
    for cue in cues:
        sub = [r for r in rows if r["bias_type"] == cue]
        hits = [f"{r['judge_model']}|{r['anchor_source']}" for r in sub
                if str(r["rho_ci_excludes_zero"]).lower() == "true"]
        out[cue] = {
            "n_cells": len(sub),
            "n_excluding_zero": len(hits),
            "cells": sorted(hits),
            "n_worst_of_four": won[cue],
        }
    return out


def build_permutation_rows(
    samples: Sequence[SampleRecord],
    results_dir: PathLike,
    *,
    metrics_dir: Optional[PathLike] = None,
    cues: Sequence[str] = CUES,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 42,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    cells, items = build_cells(samples, results_dir, cues=cues, roster=roster)
    if not cells:
        return []
    deltas = np.vstack([cell_deltas(c) for c in cells])
    observed = statistics_from_deltas(deltas)
    null = permutation_null(cells, len(items), n_perm=n_perm, seed=seed)
    published = _published_counts(metrics_dir or Path(results_dir) / "metrics", cues)

    winners = np.argmin(deltas, axis=1)
    won_by: Dict[str, List[str]] = {cue: [] for cue in cues}
    for row, cell in enumerate(cells):
        won_by[cues[winners[row]]].append(f"{cell.judge_model}|{cell.anchor_source}")

    rows: List[dict] = []
    for name, side in _EXTREME_SIDE.items():
        values = observed[name]
        best = float(values.max() if side == "high" else values.min())
        per_perm_best = null[name].max(axis=1) if side == "high" else null[name].min(axis=1)
        for c, cue in enumerate(cues):
            value = float(values[c])
            p_label = _p_value(null[name], value, side)
            p_selected = _p_value(per_perm_best.reshape(-1, 1), value, side)
            pub = published.get(cue, {})
            same = sorted(set(won_by[cue]) & set(pub.get("cells", [])))
            rows.append({
                "statistic": name,
                "bias_type": cue,
                "observed": round(value, 4),
                "n_cells": len(cells),
                "n_usable_cells": int(observed["n_usable_cells"][c]),
                "is_best_of_four": bool(abs(value - best) < 1e-12),
                # p for a cue named in advance; the null pools all four cues because
                # under label exchangeability they are one distribution.
                "p_label": round(p_label, 6),
                # p for the best of four — what a reader actually selected on.
                "p_selected": round(p_selected, 6),
                "mc_se_p_selected": round(
                    float(np.sqrt(max(p_selected * (1 - p_selected), 0.0) / n_perm)), 6
                ),
                "null_mean": round(float(null[name][:, c].mean()), 4),
                "null_p05": round(float(np.quantile(null[name][:, c], 0.05)), 4),
                "null_p95": round(float(np.quantile(null[name][:, c], 0.95)), 4),
                "n_perm": n_perm,
                "seed": seed,
                "n_items_complete_case": len(items),
                "n_turns": len({t for cell in cells for t in cell.turns}),
                "cells_won": ";".join(won_by[cue]),
                # The published criterion, for the substitution to stay auditable, and
                # THIS criterion on the published item sets, so "the count moved" can be
                # attributed to the criterion or to the complete-case drop, never both.
                "published_n_ci_excludes_zero": pub.get("n_excluding_zero"),
                "published_items_n_worst_of_four": pub.get("n_worst_of_four"),
                "published_cells": ";".join(pub.get("cells", [])),
                "criteria_overlap_cells": len(same),
                "family": "claim_B_cue_label_permutation",
            })
    return rows


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def _summarise(rows: Sequence[dict]) -> None:
    if not rows:
        print("no anchor cells found — nothing to permute")
        return
    head = rows[0]
    print(f"=== WP-A1c: cue-label permutation over {head['n_perm']} draws ===")
    print(f"  complete-case items {head['n_items_complete_case']} in "
          f"{head['n_turns']} turns; {head['n_cells']} (judge x source) cells")
    for name in _EXTREME_SIDE:
        print(f"\n  --- {name} ---")
        print(f"    {'cue':<20}{'observed':>9}{'null mean':>11}{'null p05/p95':>16}"
              f"{'p_label':>10}{'p_selected':>12}   published CI count")
        for r in [x for x in rows if x["statistic"] == name]:
            star = "*" if r["is_best_of_four"] else " "
            print(f"  {star} {r['bias_type']:<20}{r['observed']:>9}{r['null_mean']:>11}"
                  f"{str(r['null_p05']) + '/' + str(r['null_p95']):>16}"
                  f"{r['p_label']:>10}{r['p_selected']:>12}"
                  f"   {r['published_n_ci_excludes_zero']} of {r['n_cells']}"
                  f" (overlap {r['criteria_overlap_cells']})")


def main(argv: list[str] | None = None) -> int:
    root = default_root()
    manifests = root / "data" / "manifests"
    ap = argparse.ArgumentParser(description="WP-A1c claim B cue-label permutation test.")
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--samples", type=Path, default=manifests / "samples_judge_v2.jsonl")
    ap.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    samples = io.read_jsonl(args.samples, SampleRecord)
    rows = build_permutation_rows(
        samples, args.results_dir, n_perm=args.n_perm, seed=args.seed,
    )
    out_dir = Path(args.out_dir) if args.out_dir else args.results_dir / "metrics"
    write_csv(out_dir / "claim_b_permutation.csv", rows)
    _summarise(rows)
    print(f"\nwrote -> {out_dir / 'claim_b_permutation.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
