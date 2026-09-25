"""WP-A1f: §6.5(a)'s dilution algebra, moved out of prose and into a frozen table.

The D-S skin-tone null is reported next to a measured construct-validity rate of
17/20, so the first question a reader asks is whether the null is simply the valid
scenes' effect diluted by invalid ones. §6.5(a) answers it with arithmetic — assume an
invalid scene contributes 0 to the directional gap, so `observed = (1 - phi) x true`,
invert, and ask how large `phi` would have to be before the observed gap could reach
the MDE. The answer (phi = 0.579, far outside the failure rate's own confidence bound)
is one of the load-bearing sentences of that section.

**None of it was in a table.** Every number was computed in the prose, which is the
same defect that produced this project's SD-denominator error: the fairness section's
"4-8x below the MDE" mixed SD(paired difference) with SD(score) precisely because no
table carried the SD. This module recomputes the algebra from the frozen CSVs, per
judge, and writes both the arithmetic and its measured inputs to disk.

    ★ MEASURED HERE, 2026-08-17 — two corrections to §6.5(a), both in the direction
    that makes the argument STRONGER, and neither one silent:

    1. "the measured dark->light ITA separation has a **median of 61.3 deg**" — 61.3 is
       the **mean** (61.263). The median is **61.196**. A small slip, but the sentence
       names the statistic.
    2. "only **0.27%** fall below 10 deg" — measured on the 743 judged scenes the
       smallest separation is **55.04 deg**, so the share below 10 deg is **0 of 743**,
       and it is zero **by construction**: `run_construction_gate` floors every arm at
       |achieved_delta_ita| >= 25 deg (`ita.MIN_CROSSING_DELTA`), so a scene below
       10 deg cannot reach the judged set. The published 0.27% is not reproducible from
       any frozen artefact and should be replaced by the constructive argument, which
       needs no measurement at all.

Two outputs:

``dilution_algebra.csv``
    Per judge: the observed gap in SD(paired difference) units, its MDE, the implied
    true effect under the observed and upper-bound failure rates, and the break-even
    phi. Plus one summary row for the LARGEST observed gap, which is the case §6.5(a)
    actually argues — using the mean gap would understate what dilution has to explain.

``dilution_algebra_inputs.json``
    The measured inputs, so the arithmetic is auditable without re-deriving them: the
    construct-validity counts with Wilson intervals, the ITA dose distribution, the
    gate floor that makes the 10-degree question moot, and the two corrections above.

⚠️ This is only argument (a). Argument (b) — that dilution cancels in the ratio between
the directional contrast and the dose contrast, because both run on the same scenes
with the same mask — is stronger and needs no algebra at all. The table exists so that
(a) can be checked, not so that (a) can carry the section alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from edit_judge_bias.data.manifest_utils import PathLike, default_root

#: The D-S contrast the null is about. `dose_control` is the positive control and has
#: no dilution question — it fires.
GAP_FAMILY = "skin_tone"

#: `construct_validity.csv`'s row for the corpus the judged scenes were drawn from.
CONSTRUCT_SOURCE = "ds_v4"

#: Below this the "is the dose perceptible" question is asked; kept as a named constant
#: because §6.5(a) quotes it.
WEAK_DOSE_DEGREES = 10.0


def _read_csv(path: PathLike) -> List[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(value) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


# --------------------------------------------------------------------------- #
# measured inputs                                                              #
# --------------------------------------------------------------------------- #
def construct_rates(metrics_dir: PathLike, *, source: str = CONSTRUCT_SOURCE) -> dict:
    """Pass / patch / weak counts with Wilson intervals, from the frozen table."""
    from edit_judge_bias.metrics.construct_validity import wilson_interval

    rows = [r for r in _read_csv(Path(metrics_dir) / "construct_validity.csv")
            if r["source"] == source and r["stratum"] == "all"]
    if not rows:
        raise ValueError(f"no construct_validity row for source={source!r}")
    row = rows[0]
    n = int(row["n_annotated"])
    counts = {
        key: int(row[f"count_{key}"] or 0)
        for key in ("pass", "patch", "weak", "figure")
        if row.get(f"count_{key}") not in (None, "")
    }
    n_fail = n - counts.get("pass", 0)
    lo, hi = wilson_interval(n_fail, n)
    patch_lo, patch_hi = wilson_interval(counts.get("patch", 0), n)
    return {
        "source": source,
        "n_annotated": n,
        "counts": counts,
        "pass_rate": round(counts.get("pass", 0) / n, 6),
        # phi is the FAILURE rate: the share of scenes assumed to contribute nothing.
        "phi_observed": round(n_fail / n, 6),
        "phi_wilson_low": lo,
        "phi_wilson_high": hi,
        # The residual construct risk after argument (b) absorbs the `weak` half.
        "patch_rate": round(counts.get("patch", 0) / n, 6),
        "patch_wilson_low": patch_lo,
        "patch_wilson_high": patch_hi,
    }


def ita_dose(manifest: PathLike, *, weak_degrees: float = WEAK_DOSE_DEGREES) -> dict:
    """The measured dark->light ITA separation over the judged scenes."""
    from edit_judge_bias.fairness import ita as ita_module

    by_scene: Dict[str, Dict[str, Optional[float]]] = {}
    with Path(manifest).open(encoding="utf-8") as fh:
        for line in fh:
            meta = json.loads(line)["metadata"]
            by_scene.setdefault(meta["base_sample_id"], {})[meta["variant_label"]] = (
                meta.get("achieved_delta_ita")
            )
    separations = sorted(
        abs(v["light"] - v["dark"]) for v in by_scene.values()
        if v.get("dark") is not None and v.get("light") is not None
    )
    if not separations:
        raise ValueError("no scene carries both a dark and a light achieved_delta_ita")
    below = sum(1 for s in separations if s < weak_degrees)
    return {
        "n_scenes": len(separations),
        "separation_min": round(separations[0], 4),
        "separation_median": round(statistics.median(separations), 4),
        "separation_mean": round(statistics.fmean(separations), 4),
        "separation_max": round(separations[-1], 4),
        "weak_threshold_degrees": weak_degrees,
        "n_below_weak_threshold": below,
        "share_below_weak_threshold": round(below / len(separations), 6),
        "gate_floor_per_arm_degrees": float(ita_module.MIN_CROSSING_DELTA),
        "why_zero_by_construction": (
            "run_construction_gate rejects any arm with |achieved_delta_ita| below "
            f"{ita_module.MIN_CROSSING_DELTA} deg on the ORIGINAL, so a two-sided "
            f"separation below {2 * ita_module.MIN_CROSSING_DELTA} deg cannot reach "
            "the judged set at all"
        ),
    }


# --------------------------------------------------------------------------- #
# the algebra                                                                  #
# --------------------------------------------------------------------------- #
def dilution_rows(
    metrics_dir: PathLike, *, family: str = GAP_FAMILY, construct: Optional[dict] = None
) -> List[dict]:
    """`observed = (1 - phi) x true`, inverted, per judge and for the largest gap."""
    construct = construct or construct_rates(metrics_dir)
    phi_obs = construct["phi_observed"]
    phi_hi = construct["phi_wilson_high"]

    gaps = [r for r in _read_csv(Path(metrics_dir) / "attribute_gaps.csv")
            if r["attribute"] == family]
    if not gaps:
        raise ValueError(f"no attribute_gaps rows for attribute={family!r}")

    def one(label: str, gap: float, mde: float, n: int, extra: dict) -> dict:
        def implied(phi: Optional[float]) -> Optional[float]:
            if phi is None or phi >= 1.0:
                return None
            return gap / (1.0 - phi)

        at_obs, at_hi = implied(phi_obs), implied(phi_hi)
        return {
            "judge_model": label,
            "attribute": family,
            "n": n,
            **extra,
            "abs_gap_in_sd_units": round(gap, 6),
            "mde_sd": round(mde, 6),
            # How far below the detection threshold the observed gap sits, before any
            # dilution correction. Units on both sides: SD of the PAIRED DIFFERENCE.
            "mde_over_gap": round(mde / gap, 4) if gap else None,
            "phi_observed": phi_obs,
            "implied_true_at_phi_observed": None if at_obs is None else round(at_obs, 6),
            "mde_over_implied_at_phi_observed": (
                None if not at_obs else round(mde / at_obs, 4)
            ),
            "phi_wilson_high": phi_hi,
            "implied_true_at_phi_high": None if at_hi is None else round(at_hi, 6),
            "mde_over_implied_at_phi_high": None if not at_hi else round(mde / at_hi, 4),
            # The number §6.5(a) leads with: the failure rate at which the diluted
            # observation would just reach the MDE.
            "phi_breakeven": round(1.0 - gap / mde, 6) if mde else None,
            "breakeven_outside_wilson_high": (
                None if phi_hi is None or not mde else bool((1.0 - gap / mde) > phi_hi)
            ),
            "family": "dilution_algebra",
        }

    rows: List[dict] = []
    for row in sorted(gaps, key=lambda r: r["judge_model"]):
        signed = _f(row["gap_in_sd_units"])
        mde = _f(row["mde_sd"])
        if signed is None or mde is None:
            continue
        rows.append(one(
            row["judge_model"], abs(signed), mde, int(row["n"]),
            {"scope": "per_judge",
             "mean_gap": _f(row["mean_gap"]),
             "sd_gap": _f(row["sd_gap"]),
             "gap_in_sd_units": signed},
        ))

    binding = max(rows, key=lambda r: r["abs_gap_in_sd_units"])
    rows.append(one(
        f"(largest |gap|: {binding['judge_model']})",
        binding["abs_gap_in_sd_units"], binding["mde_sd"], binding["n"],
        {"scope": "binding_case",
         "mean_gap": binding["mean_gap"],
         "sd_gap": binding["sd_gap"],
         "gap_in_sd_units": binding["gap_in_sd_units"]},
    ))
    return rows


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def _summarise(rows: Sequence[dict], construct: dict, dose: dict) -> None:
    print("=== A1f: §6.5(a) dilution algebra, from the frozen tables ===")
    print(f"  construct validity ({construct['source']}): "
          f"{construct['counts'].get('pass')}/{construct['n_annotated']} pass  -> "
          f"phi = {construct['phi_observed']:.4f}  "
          f"Wilson [{construct['phi_wilson_low']:.4f}, {construct['phi_wilson_high']:.4f}]")
    print(f"  residual construct risk (patch): {construct['patch_rate']:.4f}  "
          f"Wilson [{construct['patch_wilson_low']:.4f}, {construct['patch_wilson_high']:.4f}]")
    print()
    print(f"  {'judge':<28}{'|gap| SD':>10}{'MDE':>9}{'MDE/gap':>9}"
          f"{'true@phi':>10}{'MDE/true':>10}{'true@hi':>9}{'MDE/hi':>8}{'phi*':>8}")
    for r in rows:
        print(f"  {r['judge_model']:<28}{r['abs_gap_in_sd_units']:>10.4f}"
              f"{r['mde_sd']:>9.4f}{r['mde_over_gap']:>9.2f}"
              f"{r['implied_true_at_phi_observed']:>10.4f}"
              f"{r['mde_over_implied_at_phi_observed']:>10.2f}"
              f"{r['implied_true_at_phi_high']:>9.4f}"
              f"{r['mde_over_implied_at_phi_high']:>8.2f}{r['phi_breakeven']:>8.4f}")
    print()
    print(f"  dark->light ITA separation over {dose['n_scenes']} judged scenes: "
          f"median {dose['separation_median']:.3f} deg, mean {dose['separation_mean']:.3f}, "
          f"min {dose['separation_min']:.3f}")
    print(f"  below {dose['weak_threshold_degrees']:.0f} deg: "
          f"{dose['n_below_weak_threshold']} of {dose['n_scenes']} "
          f"({100 * dose['share_below_weak_threshold']:.2f}%) — and the gate floors each "
          f"arm at {dose['gate_floor_per_arm_degrees']:.0f} deg, so this is 0 by construction")


def main(argv: list[str] | None = None) -> int:
    from edit_judge_bias.experiments.build_claim_tables import write_csv

    root = default_root()
    ds = root / "results" / "v2_fairness_ds"
    ap = argparse.ArgumentParser(description="WP-A1f dilution algebra for §6.5(a).")
    ap.add_argument("--metrics-dir", type=Path, default=ds / "metrics")
    ap.add_argument("--manifest", type=Path,
                    default=root / "data" / "manifests"
                    / "samples_fairness_ds_judge_v4.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    construct = construct_rates(args.metrics_dir)
    dose = ita_dose(args.manifest)
    rows = dilution_rows(args.metrics_dir, construct=construct)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.metrics_dir)
    write_csv(out_dir / "dilution_algebra.csv", rows)
    (out_dir / "dilution_algebra_inputs.json").write_text(
        json.dumps({
            "generated_by": "edit_judge_bias.experiments.build_dilution_algebra",
            "reads": "§6.5(a) of PAPER_ZH.md — the arithmetic answer to 'is the null "
                     "just dilution?'",
            "construct_validity": construct,
            "ita_dose": dose,
            "corrections_to_section_6_5_a": {
                "separation_statistic": (
                    "the published 'median 61.3 deg' is the MEAN "
                    f"({dose['separation_mean']}); the median is "
                    f"{dose['separation_median']}"
                ),
                "share_below_10_deg": (
                    "the published 0.27% is not reproducible from any frozen artefact; "
                    f"measured {dose['n_below_weak_threshold']} of {dose['n_scenes']} "
                    f"(min separation {dose['separation_min']} deg), and it is zero by "
                    "construction of the dose floor"
                ),
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _summarise(rows, construct, dose)
    print(f"\nwrote -> {out_dir / 'dilution_algebra.csv'} ({len(rows)} rows)")
    print(f"wrote -> {out_dir / 'dilution_algebra_inputs.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
