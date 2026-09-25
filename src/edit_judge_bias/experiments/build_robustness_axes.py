"""WP-A1b: the three-axis robustness profile, recomputed under two declared operators.

§5.4's headline is that robustness is not a scalar property of a judge — the same five
judges rank differently on the absolute-score, ranking and protocol layers, and the
lines cross. §7.2 generalises it and prints three Spearman correlations between the
three rankings. **Neither number came from a table.** The paper's §5.4 cell for the
ranking axis already carries two different operators side by side ("worst magnitude"
and "count of cells whose clustered interval excludes zero") and says in prose that
they disagree; nothing recomputed the *ranking* under each, and the correlations in
§7.2 have no recorded operationalisation at all.

    ★ MEASURED HERE, 2026-08-17: **§5.4's table and §5.4's figure do not use the same
    operator, and the section never says so.** The table ranks the absolute axis by the
    `sham` drift and the ranking axis by the *worst* cell; figure 5 —
    `visualization/plot_robustness_layers.py`, which is also where §7.2's three
    correlations and its 9/20 inversion count are computed — ranks both by the *mean*
    over cells. The two orders differ. Neither is wrong; quoting them in one section as
    one ranking is.

That is the defect this table closes. Three tracks, declared up front, applied to all
three axes, with the full trajectory frozen rather than the endpoint:

``published_mean``
    absolute  mean asc over the 36 non-sham cue x dimension cells
                                                    (`claim_a_by_dimension.csv`)
    ranking   mean |spearman_delta| over that judge's 8 cells       (`claim_b.csv`)
    protocol  1 - RR                                                (`position.csv`)
    The operator figure 5 uses. Computed by calling that figure's own layer
    functions, never by reimplementing them, so the two can never drift.

``worst_magnitude``
    absolute  max asc over the 36 non-sham cue x dimension cells
                                                    (`claim_a_by_dimension.csv`)
    ranking   max |spearman_delta| over that judge's 8 cells        (`claim_b.csv`)
    protocol  1 - RR, the share of pairs whose verdict moves when
              only the display order changes                        (`position.csv`)

``significant_count``
    absolute  # of the 12 non-sham cues significant after BH        (`claim_a.csv`)
    ranking   # of the 8 cells whose clustered rho interval
              excludes zero                                         (`claim_b.csv`)
    protocol  1 - CR                              (`pairwise_consistency.csv`)

★ 2026-09-26 — **the absolute axis of both magnitude tracks now uses the invariance
operator the paper's Method defines**: `asc`, the per-item |score change| on one judged
dimension (1-10 scale), one cell per (cue, dimension). Until then both read
|`mean_shift`| of the 3-30 sum from `claim_a.csv` -- the size of each cue's AVERAGE
signed change, inside which opposite-sign changes cancel (across items, and across the
three dimensions inside the sum). That ranked the systematic shift, not how far the
scores move. The swap moved `gpt-4o-viescore` from absolute rank 1 to 3 (mean) and 2
(worst), and the published track's adjacent-layer inversions from 9/20 to 8/20; the
ranking and protocol axes and the whole `significant_count` track are unchanged. A
per-item |change| also counts a judge's own instability: `gemini-3.5-flash`'s null `sham`
cells (mean asc 1.69) exceed its 36 cue cells (1.29), so its last place on this axis is
about score stability, not cue sensitivity.

⚠️ **The protocol axis has no cell count and the table says so in every row.** It is one
measurement per judge, so "how many cells fail" is 0 or 1 and carries no ranking. The
second track therefore uses the *second published protocol statistic* — CR, the rate at
which the judge repeats its own verdict — which is the substantive alternative §5.4
already reports beside RR and which disagrees with it (`gpt-4o-viescore` is 4th on RR
and 2nd on CR). Substituting a different statistic is not the same move as changing the
operator, and conflating the two would be exactly the kind of quiet operationalisation
drift this module exists to expose.

Sign convention, enforced for every axis: **a larger statistic means LESS robust, and
rank 1 is the most robust judge.** Without it the inversion count is not defined.

Two outputs:

``three_axis_dual_operator.csv``
    One row per (track, judge, axis): the statistic, its denominator, the rank, and the
    judge's whole rank trajectory across the three axes.

``three_axis_operator_summary.csv``
    Per track: the three between-axis Spearman correlations (§7.2's numbers, now with a
    definition attached), and the adjacent-layer inversion count against the
    independence expectation.

Read the correlations the way §7.2 already insists: n=5 judges, so none of them is
distinguishable from zero, the scalar hypothesis (rho = +1, zero inversions) is what
gets refuted, and "anti-correlated" is not supported at any operator.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edit_judge_bias.data.manifest_utils import PathLike, default_root
from edit_judge_bias.experiments.build_claim_tables import main_grid_rows, write_csv

#: The placebo arm, excluded from the absolute axis for the same reason claim A's BH
#: family excludes it: it is the control the axis is read against, not a cue.
CONTROL_BIAS = "sham"

#: Axis order is the one §5.4 plots, and "adjacent" in the inversion count means
#: adjacent in THIS order. It is a reading order, not a claim that the layers are
#: ordinally related.
AXES: Tuple[str, ...] = ("absolute", "ranking", "protocol")

TRACKS: Tuple[str, ...] = ("published_mean", "worst_magnitude", "significant_count")

#: Human-readable operator per (track, axis), written into every row so a q- or
#: rho-value can never be quoted without the definition that produced it.
OPERATORS: Dict[Tuple[str, str], str] = {
    ("published_mean", "absolute"):
        "mean asc (mean_i |delta_i|) over non-sham cue x dimension cells (figure 5)",
    ("published_mean", "ranking"): "mean |spearman_delta| over anchor cells (figure 5)",
    ("published_mean", "protocol"): "1 - RR (display-order robustness)",
    ("worst_magnitude", "absolute"):
        "max asc (mean_i |delta_i|) over non-sham cue x dimension cells",
    ("worst_magnitude", "ranking"): "max |spearman_delta| over anchor cells",
    ("worst_magnitude", "protocol"): "1 - RR (display-order robustness)",
    ("significant_count", "absolute"): "# non-sham cues significant after BH",
    ("significant_count", "ranking"): "# anchor cells whose rho CI excludes 0",
    ("significant_count", "protocol"): "1 - CR (self-consistency; see module docstring)",
}

#: Which rows carry a genuine count and which fall back to a second statistic.
_COUNT_FALLBACK_NOTE = (
    "protocol has ONE cell per judge, so a significant-cell count is undefined; "
    "this track uses the second published protocol statistic (CR) instead"
)


# --------------------------------------------------------------------------- #
# reading the frozen tables                                                    #
# --------------------------------------------------------------------------- #
def _read_csv(path: PathLike) -> List[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() == "true"


def axis_statistics(metrics_dir: PathLike) -> Dict[str, Dict[Tuple[str, str], dict]]:
    """(track, axis) -> {judge: {"value": float, "denominator": int|None, ...}}.

    Every number is lifted from a frozen CSV cell. Nothing here recomputes a claim;
    if this module and the claim tables ever disagree, that is a bug signal, not a
    finding — the same contract `position.csv` has with `pairwise_position.csv`.
    """
    metrics_dir = Path(metrics_dir)
    claim_a = _read_csv(metrics_dir / "claim_a.csv")
    # The absolute axis of the two magnitude tracks: one cell per (cue, judged dimension),
    # `asc` = mean_i |delta_i| on the 1-10 scale.  `claim_a.csv` still feeds the count
    # track, whose BH family is the 12 cues of the summed score.
    claim_a_dim = _read_csv(metrics_dir / "claim_a_by_dimension.csv")
    # The ranking axis is the main grid's 8 cells per judge.  Since 2026-09-15 the file also
    # holds the FILL v2 rows -- another collection, measured against another reference --
    # which would silently turn 8 cells into 24 and move every number below.
    claim_b = main_grid_rows(_read_csv(metrics_dir / "claim_b.csv"))
    position = _read_csv(metrics_dir / "position.csv")
    consistency = _read_csv(metrics_dir / "pairwise_consistency.csv")

    judges = sorted({r["judge_model"] for r in claim_a})
    out: Dict[Tuple[str, str], Dict[str, dict]] = {key: {} for key in OPERATORS}

    # ★ The published track is not reimplemented: it calls figure 5's own layer
    # functions. That figure is where §7.2's correlations and its inversion count come
    # from, so a second implementation here would be a second thing to keep right — and
    # the whole point of this table is that two operators were already circulating
    # unlabelled.
    # Imported lazily: `visualization.style` pulls in matplotlib at import time.
    from edit_judge_bias.visualization.plot_robustness_layers import (
        absolute_score_robustness,
        position_robustness,
        ranking_robustness,
    )

    published = {
        "absolute": absolute_score_robustness(claim_a_dim),
        "ranking": ranking_robustness(claim_b),
        "protocol": position_robustness(position),
    }
    for axis, layer in published.items():
        for judge in judges:
            value = layer.values.get(judge)
            # RR is the one published scalar where higher means MORE robust; every
            # value in this table is stored so that higher means LESS robust.
            flip = axis == "protocol"
            out[("published_mean", axis)][judge] = {
                "value": None if value is None else (1.0 - value if flip else value),
                "raw_value": value,
                "denominator": layer.support.get(judge),
                "unit": layer.unit,
            }

    for judge in judges:
        a_cells = [r for r in claim_a
                   if r["judge_model"] == judge and r["bias_type"] != CONTROL_BIAS]
        a_dim_cells = [r for r in claim_a_dim
                       if r["judge_model"] == judge and r["bias_type"] != CONTROL_BIAS]
        b_cells = [r for r in claim_b if r["judge_model"] == judge]
        pos = next((r for r in position if r["judge_model"] == judge), None)
        con = next((r for r in consistency if r["judge_model"] == judge), None)

        movement = [_f(r["asc"]) for r in a_dim_cells if _f(r["asc"]) is not None]
        rhos = [abs(_f(r["spearman_delta"])) for r in b_cells
                if _f(r["spearman_delta"]) is not None]

        out[("worst_magnitude", "absolute")][judge] = {
            "value": max(movement) if movement else None,
            "denominator": len(a_dim_cells),
            "unit": published["absolute"].unit,
        }
        out[("worst_magnitude", "ranking")][judge] = {
            "value": max(rhos) if rhos else None,
            "denominator": len(b_cells),
            "unit": "Spearman rho",
        }
        out[("worst_magnitude", "protocol")][judge] = {
            "value": None if pos is None else 1.0 - _f(pos["rr"]),
            "raw_value": None if pos is None else _f(pos["rr"]),
            "denominator": None if pos is None else int(pos["n"]),
            "unit": "share of pairs",
        }
        out[("significant_count", "absolute")][judge] = {
            "value": float(sum(1 for r in a_cells if _truthy(r["significant_bh"]))),
            "denominator": len(a_cells),
            "unit": "cells",
        }
        out[("significant_count", "ranking")][judge] = {
            "value": float(sum(1 for r in b_cells if _truthy(r["rho_ci_excludes_zero"]))),
            "denominator": len(b_cells),
            "unit": "cells",
        }
        out[("significant_count", "protocol")][judge] = {
            "value": None if con is None else 1.0 - _f(con["cr"]),
            "raw_value": None if con is None else _f(con["cr"]),
            "denominator": None if con is None else int(con["n"]),
            "unit": "share of repeats",
        }
    return out


# --------------------------------------------------------------------------- #
# ranking, correlation, inversions                                             #
# --------------------------------------------------------------------------- #
def rank_ascending(values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Competition-free average ranks, 1 = smallest value = most robust.

    Ties get the average rank, which is what Spearman needs and what keeps the
    inversion count honest: three judges tied at 9-of-12 significant cells are not
    three orderings, and breaking the tie alphabetically would manufacture two.
    """
    present = sorted((v, k) for k, v in values.items() if v is not None)
    ranks: Dict[str, Optional[float]] = {k: None for k in values}
    i = 0
    while i < len(present):
        j = i
        while j + 1 < len(present) and present[j + 1][0] == present[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        for _, key in present[i:j + 1]:
            ranks[key] = average
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    from edit_judge_bias.metrics.agreement import spearman

    return spearman(list(xs), list(ys))


def count_inversions(
    ranks_a: Dict[str, Optional[float]], ranks_b: Dict[str, Optional[float]]
) -> Tuple[int, int, int]:
    """(inversions, ties, comparisons) over every unordered pair of judges.

    An inversion is a pair whose order REVERSES between the two axes. A pair tied on
    either axis is neither an inversion nor a preservation and is counted separately —
    folding ties into either bucket is how a count operator quietly becomes a claim.
    """
    keys = sorted(k for k in ranks_a if ranks_a[k] is not None and ranks_b.get(k) is not None)
    inversions = ties = comparisons = 0
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            comparisons += 1
            da = ranks_a[x] - ranks_a[y]
            db = ranks_b[x] - ranks_b[y]
            if da == 0 or db == 0:
                ties += 1
            elif da * db < 0:
                inversions += 1
    return inversions, ties, comparisons


# --------------------------------------------------------------------------- #
# builders                                                                     #
# --------------------------------------------------------------------------- #
def build_dual_operator_rows(metrics_dir: PathLike) -> List[dict]:
    stats = axis_statistics(metrics_dir)
    judges = sorted(stats[("worst_magnitude", "absolute")])

    ranks: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {
        key: rank_ascending({j: cells[j]["value"] for j in judges})
        for key, cells in stats.items()
    }

    rows: List[dict] = []
    for track in TRACKS:
        trajectory = {
            j: " -> ".join(
                "?" if ranks[(track, ax)][j] is None else f"{ranks[(track, ax)][j]:g}"
                for ax in AXES
            )
            for j in judges
        }
        for judge in judges:
            present = [ranks[(track, ax)][judge] for ax in AXES
                       if ranks[(track, ax)][judge] is not None]
            span = None if len(present) < 2 else max(present) - min(present)
            for axis in AXES:
                cell = stats[(track, axis)][judge]
                rows.append({
                    "track": track,
                    "axis": axis,
                    "judge_model": judge,
                    "operator": OPERATORS[(track, axis)],
                    "statistic": None if cell["value"] is None else round(cell["value"], 4),
                    # The number as the source table prints it — RR and CR are stored
                    # above as 1-x so that "higher is worse" holds on every axis, and a
                    # reader comparing this file with `position.csv` must not have to
                    # do that subtraction in their head.
                    "raw_statistic": (
                        None if cell.get("raw_value", cell["value"]) is None
                        else round(cell.get("raw_value", cell["value"]), 4)
                    ),
                    "unit": cell["unit"],
                    "denominator": cell["denominator"],
                    "rank": ranks[(track, axis)][judge],
                    "rank_trajectory": trajectory[judge],
                    "rank_span": span,
                    "higher_is_worse": True,
                    "note": (_COUNT_FALLBACK_NOTE
                             if track == "significant_count" and axis == "protocol" else ""),
                })
    return rows


def build_operator_summary(metrics_dir: PathLike) -> List[dict]:
    stats = axis_statistics(metrics_dir)
    judges = sorted(stats[("worst_magnitude", "absolute")])
    rows: List[dict] = []
    for track in TRACKS:
        ranks = {
            axis: rank_ascending({j: stats[(track, axis)][j]["value"] for j in judges})
            for axis in AXES
        }
        for i, first in enumerate(AXES):
            for second in AXES[i + 1:]:
                usable = [j for j in judges
                          if ranks[first][j] is not None and ranks[second][j] is not None]
                rho = spearman_rho(
                    [ranks[first][j] for j in usable], [ranks[second][j] for j in usable]
                )
                adjacent = AXES.index(second) - AXES.index(first) == 1
                inv, tied, comp = count_inversions(ranks[first], ranks[second])
                rows.append({
                    "track": track,
                    "axis_pair": f"{first}~{second}",
                    "adjacent": adjacent,
                    "n_judges": len(usable),
                    "spearman_rho": None if rho is None else round(rho, 4),
                    "n_inversions": inv,
                    "n_tied_pairs": tied,
                    "n_comparisons": comp,
                    "operator_first": OPERATORS[(track, first)],
                    "operator_second": OPERATORS[(track, second)],
                })
        adj = [r for r in rows if r["track"] == track and r["adjacent"]]
        rows.append({
            "track": track,
            "axis_pair": "(adjacent layers, pooled)",
            "adjacent": True,
            "n_judges": len(judges),
            "spearman_rho": None,
            "n_inversions": sum(r["n_inversions"] for r in adj),
            "n_tied_pairs": sum(r["n_tied_pairs"] for r in adj),
            "n_comparisons": sum(r["n_comparisons"] for r in adj),
            "operator_first": "",
            # The independence expectation §7.2 compares against: half of the
            # untied comparisons. Written into the row so the comparison the paper
            # makes travels with the count instead of being asserted next to it.
            "operator_second": (
                "independence expectation = "
                f"{(sum(r['n_comparisons'] for r in adj) - sum(r['n_tied_pairs'] for r in adj)) / 2:g}"
                f" of {sum(r['n_comparisons'] for r in adj) - sum(r['n_tied_pairs'] for r in adj)}"
                " untied comparisons"
            ),
        })
    return rows


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def _summarise(rows: Sequence[dict], summary: Sequence[dict]) -> None:
    judges = sorted({r["judge_model"] for r in rows})
    for track in TRACKS:
        print(f"=== track: {track} ===")
        for axis in AXES:
            print(f"  {axis:<10}{OPERATORS[(track, axis)]}")
        print(f"  {'judge':<20}" + "".join(f"{ax:>22}" for ax in AXES) + "   trajectory")
        for judge in judges:
            cells = {r["axis"]: r for r in rows
                     if r["track"] == track and r["judge_model"] == judge}
            line = f"  {judge:<20}"
            for axis in AXES:
                c = cells[axis]
                line += f"{str(c['statistic']):>10} (rank {str(c['rank']):>4})"
            print(line + f"   {cells[AXES[0]]['rank_trajectory']}")
        for r in summary:
            if r["track"] != track:
                continue
            if r["axis_pair"].startswith("("):
                print(f"  adjacent-layer inversions: {r['n_inversions']}/{r['n_comparisons']}"
                      f"  (tied pairs: {r['n_tied_pairs']}) — {r['operator_second']}")
            else:
                print(f"  rho[{r['axis_pair']:<20}] = {str(r['spearman_rho']):>7}"
                      f"   inversions {r['n_inversions']}/{r['n_comparisons']}"
                      f"  ties {r['n_tied_pairs']}")
        print()


def main(argv: list[str] | None = None) -> int:
    root = default_root()
    ap = argparse.ArgumentParser(description="WP-A1b three-axis dual-operator tables.")
    ap.add_argument("--metrics-dir", type=Path, default=root / "results" / "v2" / "metrics")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.metrics_dir)
    rows = build_dual_operator_rows(args.metrics_dir)
    summary = build_operator_summary(args.metrics_dir)
    write_csv(out_dir / "three_axis_dual_operator.csv", rows)
    write_csv(out_dir / "three_axis_operator_summary.csv", summary)
    _summarise(rows, summary)
    print(f"wrote -> {out_dir / 'three_axis_dual_operator.csv'} ({len(rows)} rows)")
    print(f"wrote -> {out_dir / 'three_axis_operator_summary.csv'} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
