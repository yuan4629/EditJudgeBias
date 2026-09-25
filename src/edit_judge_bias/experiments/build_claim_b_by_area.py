"""Claim B's headline cue, stratified by how much of the frame its box covers.

WHAT THIS FIXES. `region_annotation` draws a box around the estimated edit region, and
for 580 of the anchor arm's 585 cells that region is estimated as the largest connected
component of |edited − original| (`bias_params.region_method == "diff"`). An editor that
re-renders the whole frame therefore gets a box around the whole frame: on the EBench
anchor the median `area_frac` is 0.669 and 32.3% of boxes cover more than 90% of the
picture (ImagenHub 0.495 / 31.7%).

That makes the box's GEOMETRY correlate with the edit's QUALITY — worse edits change
more pixels, so they get bigger boxes — and the two move together in the same cells the
claim is measured on. Two consequences, and only the second is a problem:

1. The effect is still there INSIDE a stratum. Agreement decay on small boxes is at
   least as large as the pooled estimate, so the headline does not depend on the
   confound.
2. The POOLED Δρ mixes "the judge reorders items within a box size" with "different
   items got different box sizes", so it cannot support a statement ordering the cues by
   how close the perturbation sits to the edit region. §7.1's monotone ordering is
   therefore not interpretable, and this table is the evidence for saying so.

⚠️ WHAT THIS TABLE IS NOT. It is not a replacement estimate for claim B. Stratifying
cuts turns in half — a turn is 8 editors of one image and they land in different strata
— so every cell here is measured on partial clusters and its interval is wider than the
published one for reasons that have nothing to do with the effect. `claim_b.csv` remains
the estimate; this is the sensitivity analysis beside it. Every row carries its cluster
count and how many of its clusters are whole turns, so that cost is visible per cell.

⚠️ AND WHAT IT STILL CANNOT SEPARATE. "Bigger box" and "worse edit" are one variable
here, in both directions. Separating them needs a control this study does not have: a
FIXED-size box in a RANDOM position, so the geometry stops carrying information about
the edit. That is a paid arm (A8/E2), and the limitation stands until it runs.

Two tables under `<results-dir>/metrics/`:

``anchor_area_frac_distribution.csv``
    Per anchor source: the box-area distribution the paper never reported, and the
    rank correlation between box area and the human score — the confound itself,
    measured. Its interval is a TURN-CLUSTERED bootstrap; the unclustered p is
    reported beside it and named `_unclustered`, exactly as `mcnemar_p_unclustered`
    is, because the 8 editors of one turn share an original image.

``claim_b_by_area_stratum.csv``
    Claim B for `region_annotation` within box-area strata, in two schemes (a
    four-way split and a binary one) plus the unstratified row in the same units, so
    a reader can see what stratification cost before reading what it showed.

    python -m edit_judge_bias.experiments.build_claim_b_by_area \\
        --results-dir results/v2 \\
        --samples data/manifests/samples_judge_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.aggregate_results import (
    PUBLISHED_ROSTER,
    ROSTERS,
    _judges,
)
from edit_judge_bias.experiments.build_claim_tables import (
    _read_judge,
    attach_bh,
    write_csv,
)
from edit_judge_bias.metrics.agreement import (
    by_turn,
    cluster_bootstrap_ci,
    compute_agreement,
    spearman,
)

#: The cue this table is about. It is the only one whose injector estimates a region,
#: so it is the only one whose geometry can co-vary with edit quality.
CUE = "region_annotation"

#: Both stratifications are reported. The four-way one shows the gradient; the binary
#: one keeps enough turns per cell for the interval to mean anything. Publishing only
#: whichever came out better would be choosing the cut after seeing the answer.
SCHEMES: Dict[str, Sequence[Tuple[str, float, float]]] = {
    "quartet": (("<0.25", 0.0, 0.25), ("0.25-0.5", 0.25, 0.5),
                ("0.5-0.9", 0.5, 0.9), (">=0.9", 0.9, 1.01)),
    "binary": (("<0.5", 0.0, 0.5), (">=0.5", 0.5, 1.01)),
}


def load_area_fracs(manifest: Path) -> Tuple[Dict[str, float], Counter]:
    """`base_sample_id -> area_frac` for the region_annotation rows, plus method mix.

    Read from the biased manifest rather than recomputed from the images: this is the
    geometry the injector actually used, and re-deriving it would measure today's
    estimator against yesterday's stimuli.
    """
    area: Dict[str, float] = {}
    methods: Counter = Counter()
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("bias_type") != CUE:
                continue
            params = rec.get("bias_params") or {}
            value = params.get("area_frac")
            if value is None:
                continue
            area[rec["base_sample_id"]] = float(value)
            methods[params.get("region_method") or "unknown"] += 1
    return area, methods


def _anchor_subsets(
    samples: Sequence[SampleRecord], area: Dict[str, float]
) -> Dict[str, List[SampleRecord]]:
    """Anchor samples that carry a box, grouped by source.

    Filtered on `metadata.anchor_source` and never on `subset_block`, for the same
    reason `build_claim_b` is: 39 anchor samples are also in the breadth block.
    """
    out: Dict[str, List[SampleRecord]] = {}
    for s in samples:
        source = getattr(s.metadata, "anchor_source", None)
        if source and s.sample_id in area:
            out.setdefault(source, []).append(s)
    return out


def build_area_distribution(
    samples: Sequence[SampleRecord],
    area: Dict[str, float],
    methods: Counter,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> List[dict]:
    """The box-area distribution, and the confound itself as a number."""
    rows: List[dict] = []
    for source, subset in sorted(_anchor_subsets(samples, area).items()):
        vals = [area[s.sample_id] for s in subset]
        rated = [s for s in subset if s.human_score is not None]
        rho = p_unclustered = None
        ci: Optional[Tuple[float, float]] = None
        if len(rated) >= 3:
            xs = [area[s.sample_id] for s in rated]
            ys = [float(s.human_score) for s in rated]
            rho = spearman(xs, ys)
            from scipy.stats import spearmanr
            p_unclustered = float(spearmanr(xs, ys).pvalue)
            # ⚠️ Clustered by turn, because the 8 editors of one turn share an original
            # image AND (for `diff`) an estimation procedure. The unclustered p is kept
            # beside it and NAMED, the same way `mcnemar_p_unclustered` is -- this
            # project has already published one test that ignored these clusters.
            clusters = by_turn([
                _AreaItem(area[s.sample_id], float(s.human_score),
                          (s.original_image_path.as_posix(), s.instruction))
                for s in rated
            ])
            ci = cluster_bootstrap_ci(
                clusters,
                lambda items: spearman([i.area for i in items], [i.human for i in items]),
                n_boot=n_boot, seed=seed,
            )
        rows.append({
            "anchor_source": source,
            "cue": CUE,
            "n": len(vals),
            "n_human_rated": len(rated),
            "median_area_frac": round(statistics.median(vals), 4),
            "mean_area_frac": round(statistics.fmean(vals), 4),
            "p10_area_frac": round(_quantile(vals, 0.10), 4),
            "p90_area_frac": round(_quantile(vals, 0.90), 4),
            "share_gt_0.9": round(sum(v > 0.9 for v in vals) / len(vals), 4),
            "share_lt_0.05": round(sum(v < 0.05 for v in vals) / len(vals), 4),
            "spearman_area_vs_human": None if rho is None else round(rho, 4),
            "spearman_ci_low": None if ci is None else round(ci[0], 4),
            "spearman_ci_high": None if ci is None else round(ci[1], 4),
            "spearman_ci_excludes_zero": (
                None if ci is None else bool(ci[0] > 0 or ci[1] < 0)
            ),
            "spearman_p_unclustered": (
                None if p_unclustered is None else round(p_unclustered, 6)
            ),
            "region_method_mix": ";".join(f"{k}={v}" for k, v in sorted(methods.items())),
        })
    return rows


class _AreaItem:
    """Minimal item for the clustered bootstrap over (box area, human score)."""

    __slots__ = ("area", "human", "turn")

    def __init__(self, area: float, human: float, turn: Tuple[str, str]) -> None:
        self.area, self.human, self.turn = area, human, turn


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def build_area_strata(
    samples: Sequence[SampleRecord],
    results_dir: Path,
    area: Dict[str, float],
    *,
    n_boot: int = 1000,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    """Claim B for `region_annotation` inside box-area strata, plus the pooled row."""
    rows: List[dict] = []
    for source, subset in sorted(_anchor_subsets(samples, area).items()):
        ids = {s.sample_id for s in subset}
        # How big each turn is BEFORE stratification, so every stratified cell can say
        # how much of a turn it actually kept.
        full_size = Counter(
            (s.original_image_path.as_posix(), s.instruction) for s in subset
        )
        for model in _judges(results_dir, "scoring", roster):
            originals = [
                r for r in _read_judge(
                    results_dir / "raw_judgments" / f"scoring__{model}.jsonl")
                if r.sample_id in ids
            ]
            biased = [
                r for r in _read_judge(
                    results_dir / "biased_judgments" / f"scoring__{model}.jsonl")
                if r.sample_id in ids and r.bias_type == CUE
            ]
            if not biased:
                continue
            plan: List[Tuple[str, str, List[SampleRecord]]] = [
                ("none", "all", list(subset))
            ]
            for scheme, bands in SCHEMES.items():
                for name, lo, hi in bands:
                    plan.append((scheme, name, [
                        s for s in subset if lo <= area[s.sample_id] < hi
                    ]))
            for scheme, name, members in plan:
                if not members:
                    continue
                st = compute_agreement(
                    members, originals, biased,
                    judge_model=model, bias_type=CUE, n_boot=n_boot,
                )
                row = st.as_row()
                sizes = [len(c) for c in by_turn(st.items)]
                kept_full = sum(
                    1 for c in by_turn(st.items)
                    if len(c) == full_size[c[0].turn]
                )
                vals = [area[s.sample_id] for s in members]
                lo, hi = row["spearman_delta_ci_low"], row["spearman_delta_ci_high"]
                rows.append({
                    "judge_model": model,
                    "anchor_source": source,
                    "bias_type": CUE,
                    "scheme": scheme,
                    "stratum": name,
                    "n_members": len(members),
                    "median_area_frac": round(statistics.median(vals), 4),
                    **{k: v for k, v in row.items()
                       if k not in ("judge_model", "bias_type")},
                    # ⚠️ Stratifying cuts turns apart. These two columns are the price:
                    # a cell whose clusters are mostly fragments has a wider interval
                    # than the pooled row for reasons that are not about the effect.
                    "mean_cluster_size": round(statistics.fmean(sizes), 3) if sizes else None,
                    "n_whole_turns": kept_full,
                    "share_whole_turns": (
                        round(kept_full / len(sizes), 4) if sizes else None
                    ),
                    "rho_ci_excludes_zero": (
                        None if lo is None or hi is None else bool(lo > 0 or hi < 0)
                    ),
                })

    # One BH family per scheme, and the unstratified rows are NOT in any of them: they
    # restate `claim_b.csv`'s hypothesis, which that file already corrects. Same rule
    # that keeps the ensemble table's member rows out of its family.
    for scheme in list(SCHEMES) + ["none"]:
        in_scheme = [r for r in rows if r["scheme"] == scheme]
        if scheme == "none":
            for r in in_scheme:
                r["family"] = "reference (claim_b.csv)"
                r["q_value"] = None
                r["significant_bh"] = None
            continue
        attach_bh(in_scheme, family=f"claim_B_area_{scheme}_acc",
                  p_key="accuracy_p_cluster")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    root = default_root()
    ap = argparse.ArgumentParser(
        description="Freeze the box-area distribution and claim B stratified by it."
    )
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--samples", type=Path,
                    default=root / "data" / "manifests" / "samples_judge_v2.jsonl")
    ap.add_argument("--biased-manifest", type=Path,
                    default=root / "data" / "manifests" / "biased_samples_full_v2.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--roster", choices=sorted(ROSTERS), default="published")
    args = ap.parse_args(argv)

    if not args.biased_manifest.exists():
        print(f"skip claim_b_by_area: {args.biased_manifest} not built")
        return 0
    area, methods = load_area_fracs(args.biased_manifest)
    samples = io.read_jsonl(args.samples, SampleRecord)
    out_dir = Path(args.out_dir) if args.out_dir else args.results_dir / "metrics"
    prefix = "" if args.roster == "published" else f"{args.roster}_"

    dist = build_area_distribution(samples, area, methods, n_boot=args.n_boot)
    write_csv(out_dir / f"{prefix}anchor_area_frac_distribution.csv", dist)
    strata = build_area_strata(samples, args.results_dir, area,
                               n_boot=args.n_boot, roster=ROSTERS[args.roster])
    write_csv(out_dir / f"{prefix}claim_b_by_area_stratum.csv", strata)

    print("=== region_annotation box area, per anchor source ===")
    for r in dist:
        print(f"{r['anchor_source']:<12} n={r['n']:<5} median={r['median_area_frac']:.3f} "
              f">0.9={r['share_gt_0.9']:.3f}  rho(area, human)="
              f"{r['spearman_area_vs_human']}  CI=[{r['spearman_ci_low']}, "
              f"{r['spearman_ci_high']}]")
    print("\n=== claim B within box-area strata (binary scheme) ===")
    print(f"{'judge':<18}{'source':<12}{'stratum':<10}{'n':>5}{'dRho':>9}"
          f"{'ci_lo':>9}{'ci_hi':>9}{'turns':>7}{'whole':>7}")
    for r in strata:
        if r["scheme"] not in ("binary", "none"):
            continue
        print(f"{r['judge_model']:<18}{r['anchor_source']:<12}{r['stratum']:<10}"
              f"{r['n_paired']:>5}{str(r['spearman_delta']):>9}"
              f"{str(r['spearman_delta_ci_low']):>9}{str(r['spearman_delta_ci_high']):>9}"
              f"{r['n_clusters']:>7}{str(r['share_whole_turns']):>7}")
    print(f"\nwrote {len(dist)} + {len(strata)} rows -> {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
