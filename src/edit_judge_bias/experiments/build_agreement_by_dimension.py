"""The paper's main-table agreement markers: judge-human rank agreement per rating dimension.

Every row is one (judge, cue, rating dimension) on an anchor benchmark that carries a human
score for each dimension. Only EBench-18K does. ImagenHub's human ratings are semantic
consistency and perceptual quality, with no preservation score, so it has no row here (its
summed-score cells, read against the mean of the two, are in ``claim_b.csv``). The human
label of each dimension (``configs/data/ebench18k.yaml``):

    instruction_adherence   metadata.mos_alignment
    editing_quality         metadata.mos_quality
    detail_preservation     metadata.mos_preservation

The judge side is the same dimension's 1-10 rating (never the summed score). Items are the
complete case -- all three ratings parsed on both arms -- so the three dimensions of one
(judge, cue) share one item set, and only items with both arms enter a correlation.

Reference arm, by collection (the design ``claim_b.csv`` already uses):

``unbiased baseline``
    ``padding``, ``text_overlay``, ``brightness``, ``region_annotation`` (main grid,
    ``results/v2``): the cued answer against the un-cued answer to the same item.
``sham (same collection)``
    the eight FILL v2 anchor cues (``results/v2_fill``): the cued answer against the fill's
    own ``sham`` on the same item. The fill re-asked no un-cued baseline, and two judges'
    answers drifted between the collections (``fill_collection_drift.csv``). The
    same-collection ``sham`` cancels that between-collection drift; it does not cancel
    drift within the fill run, and ``sham`` is a zero-dose control, not a dose-matched one.

    delta_rho = rho_S(cued rating, human label) - rho_S(reference rating, human label)

Interval: 95% percentile interval over 1,000 resamples of whole editing turns (source image
+ instruction; ``cluster_bootstrap_ci``, seed 42) -- the resampling unit is the turn, not
the item, because the editors of one turn share an image and an instruction.

``marker`` is the published rule, and it is UNCORRECTED:

    down   the interval lies entirely below zero
    up     the interval lies entirely above zero
    ''     the interval covers zero

No multiple-testing correction enters it: at 95% per cell, a table of 180 cells expects
about nine exclusions of zero from noise alone.

Supplementary Benjamini-Hochberg columns (they do NOT replace ``marker``):

``p_boot``
    two-sided bootstrap p-value from the same 1,000 resamples as the interval
    (``bootstrap_p_two_sided``; smallest attainable value 2/1001).
``family`` / ``q_value`` / ``significant_bh`` / ``marker_bh``
    BH over every cell of the table, the family the markers are displayed in.
``family_by_collection`` / ``q_value_by_collection`` / ...
    sensitivity: BH within each collection, the split ``claim_b.csv`` declares for its
    accuracy half (main grid / ``[fill]``).

The summary counts exclude ``zoom_inset`` (borderline under the preservation analysis, as in
``invariance_noise_floor.csv``); its rows stay in the file and in both BH families.

    python -m edit_judge_bias.experiments.build_agreement_by_dimension \\
        --results-dir results/v2 --fill-dir results/v2_fill \\
        --samples data/manifests/samples_judge_v2.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.aggregate_results import PUBLISHED_ROSTER, ROSTERS, _judges
from edit_judge_bias.experiments.build_claim_a_dimensions import DIMENSIONS, _complete_case
from edit_judge_bias.experiments.build_claim_tables import (
    ANCHOR_BIASES,
    BASELINE_LABEL,
    MAIN_COLLECTION,
    _read_judge,
    write_csv,
)
from edit_judge_bias.experiments.build_fill_tables import (
    ANCHOR_CUES as FILL_ANCHOR_CUES,
    COLLECTION as FILL_COLLECTION,
    REFERENCE as FILL_REFERENCE,
    REFERENCE_LABEL as FILL_REFERENCE_LABEL,
    _fill_rows,
)
from edit_judge_bias.experiments.build_main_table import CUE_CLASS
from edit_judge_bias.experiments.build_noise_floor_table import BORDERLINE_CUES
from edit_judge_bias.metrics.agreement import (
    _rho,
    bootstrap_p_two_sided,
    build_items,
    by_turn,
    cluster_bootstrap_values,
)
from edit_judge_bias.metrics.stats import benjamini_hochberg

#: The human label that scores the same property as each judge dimension.
HUMAN_LABEL: Dict[str, str] = {
    "instruction_adherence": "mos_alignment",
    "editing_quality": "mos_quality",
    "detail_preservation": "mos_preservation",
}

N_BOOT = 1000
SEED = 42
ALPHA = 0.05
FAMILY = "delta_rho_by_dimension"
MARKERS = ("down", "up", "")


def _marker(lo: Optional[float], hi: Optional[float]) -> Optional[str]:
    if lo is None or hi is None:
        return None
    return "down" if hi < 0 else "up" if lo > 0 else ""


def labelled_sources(samples: Sequence[SampleRecord]) -> Dict[str, List[SampleRecord]]:
    """Anchor sources whose every sample carries all three per-dimension human labels."""
    by_source: Dict[str, List[SampleRecord]] = defaultdict(list)
    for s in samples:
        source = getattr(s.metadata, "anchor_source", None)
        if source:
            by_source[source].append(s)
    return {
        src: subset for src, subset in sorted(by_source.items())
        if all(getattr(s.metadata, f, None) is not None
               for s in subset for f in HUMAN_LABEL.values())
    }


def delta_rho_cell(
    samples: Sequence[SampleRecord],
    reference: Sequence[JudgeResult],
    cued: Sequence[JudgeResult],
    *,
    judge_model: str,
    bias_type: str,
    dimension: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """One cell: rho before/after on the paired items, the turn-cluster interval, p_boot."""
    label = HUMAN_LABEL[dimension]
    relabelled = [s.model_copy(update={"human_score": getattr(s.metadata, label)})
                  for s in samples]
    joined = build_items(relabelled, reference, cued, judge_model=judge_model,
                         bias_type=bias_type, score_field=dimension)
    items = [it for it in joined if it.judge_biased is not None]
    clusters = by_turn(items)
    before, after = _rho(items, biased=False), _rho(items, biased=True)
    delta = None if before is None or after is None else after - before

    def rho_delta(sub):
        b, a = _rho(sub, biased=False), _rho(sub, biased=True)
        return None if b is None or a is None else a - b

    values = (cluster_bootstrap_values(clusters, rho_delta, n_boot=n_boot, seed=seed)
              if delta is not None else [])
    lo = hi = None
    if len(values) >= 2:
        lo, hi = (float(v) for v in np.quantile(values, [ALPHA / 2, 1 - ALPHA / 2]))
    return {
        "human_label": label,
        "n_items": len(joined),
        "n_paired": len(items),
        "n_clusters": len(clusters),
        "spearman_reference": before,
        "spearman_cued": after,
        "spearman_delta": delta,
        "spearman_delta_ci_low": lo,
        "spearman_delta_ci_high": hi,
        "n_boot_defined": len(values),
        "marker": _marker(lo, hi),
        "p_boot": bootstrap_p_two_sided(values) if len(values) >= 2 else None,
    }


def _in(ids: set, results: Sequence[JudgeResult]) -> List[JudgeResult]:
    """The complete-case rows of these samples."""
    return _complete_case([r for r in results if r.sample_id in ids])


def _stamp_bh(rows: List[dict], members: List[int], *, family: str, suffix: str) -> None:
    qs = benjamini_hochberg([rows[i]["p_boot"] for i in members])
    for i, q in zip(members, qs):
        row = rows[i]
        row[f"family{suffix}"] = family
        row[f"q_value{suffix}"] = None if q is None else round(q, 6)
        sig = None if q is None else bool(q < ALPHA)
        row[f"significant_bh{suffix}"] = sig
        row[f"marker_bh{suffix}"] = None if sig is None else (row["marker"] if sig else "")


def build_agreement_by_dimension(
    samples: Sequence[SampleRecord],
    results_dir: Path,
    fill_dir: Path,
    *,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
    roster_tag: str = "",
    n_boot: int = N_BOOT,
) -> List[dict]:
    sources = labelled_sources(samples)
    if not sources:
        raise ValueError("no anchor source carries the three per-dimension human labels")
    rows: List[dict] = []
    for source, subset in sources.items():
        ids = {s.sample_id for s in subset}
        for model in _judges(results_dir, "scoring", roster):
            baseline = _in(ids, _read_judge(
                results_dir / "raw_judgments" / f"scoring__{model}.jsonl"))
            main_cued = _in(ids, _read_judge(
                results_dir / "biased_judgments" / f"scoring__{model}.jsonl"))
            fill = _in(ids, _fill_rows(fill_dir, "scoring", model))
            fill_sham = [r for r in fill if r.bias_type == FILL_REFERENCE]
            arms = [(cue, baseline, main_cued, BASELINE_LABEL, MAIN_COLLECTION)
                    for cue in ANCHOR_BIASES]
            arms += [(cue, fill_sham, fill, FILL_REFERENCE_LABEL, FILL_COLLECTION)
                     for cue in FILL_ANCHOR_CUES]
            for cue, reference, cued, ref_label, collection in arms:
                cued_rows = [r for r in cued if r.bias_type == cue]
                if not cued_rows:
                    raise ValueError(f"{model} / {source}: no complete-case `{cue}` rows")
                for dim in DIMENSIONS:
                    cell = delta_rho_cell(subset, reference, cued_rows, judge_model=model,
                                          bias_type=cue, dimension=dim, n_boot=n_boot)
                    rows.append({
                        "judge_model": model,
                        "anchor_source": source,
                        "bias_type": cue,
                        "site": CUE_CLASS[cue],
                        "dimension": dim,
                        "reference": ref_label,
                        "collection": collection,
                        **cell,
                        "counted": cue not in BORDERLINE_CUES,
                    })
    rows.sort(key=lambda r: (r["anchor_source"], r["judge_model"], r["bias_type"],
                             DIMENSIONS.index(r["dimension"])))

    _stamp_bh(rows, list(range(len(rows))), family=FAMILY + roster_tag, suffix="")
    for collection, tag in ((MAIN_COLLECTION, " [main grid]"), (FILL_COLLECTION, " [fill]")):
        members = [i for i, r in enumerate(rows) if r["collection"] == collection]
        _stamp_bh(rows, members, family=FAMILY + tag + roster_tag, suffix="_by_collection")

    for r in rows:
        for k in ("spearman_reference", "spearman_cued", "spearman_delta",
                  "spearman_delta_ci_low", "spearman_delta_ci_high"):
            if r[k] is not None:
                r[k] = round(r[k], 6)
        if r["p_boot"] is not None:
            r["p_boot"] = round(r["p_boot"], 6)
    return rows


def summarize(rows: Sequence[dict]) -> Dict[str, Dict]:
    """Marker counts over the counted cues: published (uncorrected) and under each BH."""
    counted = [r for r in rows if r["counted"]]

    def count(key: str) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for group, sel in (("all", lambda r: True),
                           *((f"site={s}", (lambda s: lambda r: r["site"] == s)(s))
                             for s in ("prompt", "pixel", "content")),
                           *((f"dim={d}", (lambda d: lambda r: r["dimension"] == d)(d))
                             for d in DIMENSIONS)):
            sub = [r for r in counted if sel(r)]
            out[group] = {"cells": len(sub),
                          "down": sum(1 for r in sub if r.get(key) == "down"),
                          "up": sum(1 for r in sub if r.get(key) == "up")}
        return out

    return {"marker": count("marker"), "marker_bh": count("marker_bh"),
            "marker_bh_by_collection": count("marker_bh_by_collection")}


def _print_summary(rows: Sequence[dict]) -> None:
    s = summarize(rows)
    excluded = ", ".join(sorted(BORDERLINE_CUES))
    names = {"marker": "published marker (95% turn-cluster interval, UNCORRECTED)",
             "marker_bh": f"BH over the whole table ({len(rows)} cells)",
             "marker_bh_by_collection": "BH within each collection (sensitivity)"}
    for key, title in names.items():
        print(f"=== {title}; counts exclude {excluded} ===")
        for group, c in s[key].items():
            print(f"  {group:<32}{c['down'] + c['up']:>4} / {c['cells']:<4}"
                  f"  down {c['down']:>3}  up {c['up']:>3}")
    lost = [r for r in rows if r["marker"] and not r["marker_bh"]]
    gained = [r for r in rows if r["marker_bh"] and not r["marker"]]
    print(f"=== difference: {len(lost)} marker(s) not kept under whole-table BH, "
          f"{len(gained)} gained (all cells, {excluded} included) ===")
    for r in lost:
        print(f"  - {r['judge_model']:<18}{r['bias_type']:<20}{r['dimension']:<24}"
              f"d={r['spearman_delta']:+.4f}  p={r['p_boot']:.4f}  q={r['q_value']:.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    root = default_root()
    ap = argparse.ArgumentParser(description=(
        "Build the per-dimension judge-human agreement table (the main table's markers)."))
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--fill-dir", type=Path, default=root / "results" / "v2_fill")
    ap.add_argument("--samples", type=Path,
                    default=root / "data" / "manifests" / "samples_judge_v2.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--roster", choices=sorted(ROSTERS), default="published",
                    help="declared judge family; anything but `published` writes to "
                         "prefixed files")
    args = ap.parse_args(argv)

    samples = io.read_jsonl(args.samples, SampleRecord)
    published = args.roster == "published"
    rows = build_agreement_by_dimension(
        samples, args.results_dir, args.fill_dir, roster=ROSTERS[args.roster],
        roster_tag="" if published else f" [{args.roster} roster]", n_boot=args.n_boot)
    out_dir = Path(args.out_dir) if args.out_dir else args.results_dir / "metrics"
    out = out_dir / f"{'' if published else args.roster + '_'}agreement_by_dimension.csv"
    write_csv(out, rows)
    _print_summary(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
