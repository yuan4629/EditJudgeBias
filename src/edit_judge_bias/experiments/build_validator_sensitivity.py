"""WP-A4b — the validator SENSITIVITY table, paired against its own specificity floor.

WHY THIS TABLE EXISTS
---------------------
`sham` measures a validator's SPECIFICITY: with nothing wrong, how often does it cry
wolf?  A validator can score perfectly on that and still be worthless, by the simple
expedient of never flagging anything.  WP-A3 turned that from a hypothetical into a live
worry: gemini's false-flag floor came out at 3/780 = 0.38%, and the new §5.7 headline
(`zoom_inset` significantly below its floor) rests entirely on that validator being an
instrument rather than a rubber stamp.

This table measures the other half.  The control images have their edit region replaced
with the ORIGINAL pixels, so the instruction is *demonstrably* not carried out and the
ground truth needs no rater.  A validator that does not flag those is blind.

THE DESIGN POINT: ONE IMAGE SET, BOTH HALVES
--------------------------------------------
The control was rendered on the `sham` arm's published 110 base images on purpose.  That
makes sensitivity and specificity two readings of the SAME instrument on the SAME
pictures, rather than two rates from two samples -- the identical argument as WP-A3's
matched floor, one level up.  So every row carries both, plus a paired McNemar over the
two verdicts on each image, which is the actual "is this an instrument" test:

    b = flagged the damaged version, passed the sham version   (discriminating)
    c = passed the damaged version, flagged the sham version   (anti-discriminating)

THE TRAP THIS MODULE IS BUILT AROUND
------------------------------------
17 of the 110 base images have an edit so small (edit regions of 0.01%-2% of the frame)
that reverting it *entirely* changes the picture by less than 1/255.  On those, the
damaged and the edited version are indistinguishable, so a validator that says
"preserved" is RIGHT, not blind.  Counting them as misses would manufacture exactly the
insensitivity this arm exists to measure.  Every rate is therefore reported on the
evaluable subset, with the excluded count printed beside it.

Two denominators are emitted, because for the weaker rungs of the ladder they differ and
the difference is itself informative:

  * `n_evaluable_fixed`  -- the images evaluable at severity 1.0 (the same set for all
    four conditions, so the threshold curve is comparable rung to rung);
  * `n_evaluable_own`    -- images whose measured damage at THIS condition clears the
    threshold (fair to the validator: sub-threshold damage is genuinely invisible).

Reporting only the first understates sensitivity at 0.25; reporting only the second makes
the curve incomparable.  The headline condition (severity 1.0) is identical under both.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edit_judge_bias.metrics.construct_validity import wilson_interval
from edit_judge_bias.metrics.stats import benjamini_hochberg, mcnemar_pvalue

# Matches scripts/prepare_edit_damage_control.py: a mean absolute pixel change below this is not a
# perceptible change, so the image cannot test sensitivity in either direction.
WEAK_DAMAGE = 1.0

# The ladder, strongest first so the headline row reads first.
CONDITIONS: Tuple[str, ...] = (
    "edit_damage_100",
    "edit_damage_50",
    "edit_damage_25",
    "edit_damage_blur",
)

# The condition whose evaluability defines the fixed denominator.
REFERENCE_CONDITION = "edit_damage_100"

FIELDS: Tuple[str, ...] = (
    "validator_model",
    "condition",
    "mode",
    "severity",
    "mean_abs_change_median",
    "n_judged",
    "n_weak_excluded",
    "n_evaluable_fixed",
    "n_flagged_fixed",
    "sensitivity_fixed",
    "sensitivity_ci_low",
    "sensitivity_ci_high",
    "n_evaluable_own",
    "n_flagged_own",
    "sensitivity_own",
    "n_flagged_among_weak",
    "n_sham_flagged_on_same_images",
    "false_flag_rate_same_images",
    "discrimination",
    "mcnemar_b",
    "mcnemar_c",
    "mcnemar_p",
    "mcnemar_q",
    "discriminates",
)


def _read_control_manifest(path: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    """{condition: {base_sample_id: bias_params}}."""
    out: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["bias_type"]][row["base_sample_id"]] = row.get("bias_params") or {}
    return out


def _read_verdicts(path: Path, *, suffix: Optional[str] = None) -> Dict[str, bool]:
    """{base_sample_id: preserved?} for rows whose cue matches `suffix` (or any).

    Only rows that parsed are returned.  A row that failed to parse carries no verdict,
    and treating it as either value would invent data.
    """
    out: Dict[str, bool] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            biased_id = row.get("biased_id") or ""
            base, _, cue = biased_id.rpartition("__")
            if suffix is not None and cue != suffix:
                continue
            if not row.get("parse_success", True):
                continue
            verdict = row.get("pass", row.get("passed"))
            if verdict is None:
                continue
            out[base] = bool(verdict)
    return out


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_validator_sensitivity(
    control_manifest: Path,
    control_results_dir: Path,
    published_results_dir: Path,
) -> List[Dict[str, object]]:
    """One row per (validator, condition).  See the module docstring for the design."""
    manifest = _read_control_manifest(Path(control_manifest))
    if REFERENCE_CONDITION not in manifest:
        raise ValueError(
            f"control manifest has no {REFERENCE_CONDITION!r} rows; the fixed denominator "
            "is undefined and the threshold curve would not be comparable"
        )

    # The fixed evaluable set: images where a FULL revert is perceptible at all.
    fixed_evaluable = {
        base
        for base, params in manifest[REFERENCE_CONDITION].items()
        if float(params.get("mean_abs_change") or 0.0) >= WEAK_DAMAGE
    }

    rows: List[Dict[str, object]] = []
    control_dir = Path(control_results_dir)
    for control_file in sorted(control_dir.glob("validation__*.jsonl")):
        validator = control_file.stem.split("__", 1)[1]
        sham_file = Path(published_results_dir) / f"validation__{validator}.jsonl"
        sham = _read_verdicts(sham_file, suffix="sham") if sham_file.exists() else {}

        per_condition: List[Dict[str, object]] = []
        for condition in CONDITIONS:
            damage = _read_verdicts(control_file, suffix=condition)
            if not damage:
                continue
            params = manifest.get(condition, {})

            own_evaluable = {
                base
                for base, p in params.items()
                if float(p.get("mean_abs_change") or 0.0) >= WEAK_DAMAGE
            }
            judged = set(damage)
            fixed = sorted(judged & fixed_evaluable)
            own = sorted(judged & own_evaluable)
            weak = sorted(judged - fixed_evaluable)

            n_flag_fixed = sum(1 for b in fixed if not damage[b])
            n_flag_own = sum(1 for b in own if not damage[b])
            n_flag_weak = sum(1 for b in weak if not damage[b])

            sens = n_flag_fixed / len(fixed) if fixed else None
            lo, hi = wilson_interval(n_flag_fixed, len(fixed)) if fixed else (None, None)

            # Paired against the SAME images' sham verdicts.
            shared = [b for b in fixed if b in sham]
            n_sham_flag = sum(1 for b in shared if not sham[b])
            ff = n_sham_flag / len(shared) if shared else None
            mb = sum(1 for b in shared if not damage[b] and sham[b])
            mc = sum(1 for b in shared if damage[b] and not sham[b])
            p = mcnemar_pvalue(mb, mc)

            sample = manifest.get(condition, {})
            first = next(iter(sample.values()), {}) if sample else {}
            per_condition.append(
                {
                    "validator_model": validator,
                    "condition": condition,
                    "mode": first.get("mode"),
                    "severity": first.get("severity"),
                    "mean_abs_change_median": _median(
                        [float(p2.get("mean_abs_change") or 0.0) for p2 in sample.values()]
                    ),
                    "n_judged": len(judged),
                    "n_weak_excluded": len(weak),
                    "n_evaluable_fixed": len(fixed),
                    "n_flagged_fixed": n_flag_fixed,
                    "sensitivity_fixed": sens,
                    "sensitivity_ci_low": lo,
                    "sensitivity_ci_high": hi,
                    "n_evaluable_own": len(own),
                    "n_flagged_own": n_flag_own,
                    "sensitivity_own": (n_flag_own / len(own)) if own else None,
                    "n_flagged_among_weak": n_flag_weak,
                    "n_sham_flagged_on_same_images": n_sham_flag if shared else None,
                    "false_flag_rate_same_images": ff,
                    "discrimination": (sens - ff) if (sens is not None and ff is not None) else None,
                    "mcnemar_b": mb,
                    "mcnemar_c": mc,
                    "mcnemar_p": p,
                }
            )

        # BH within this validator, across its conditions.
        qs = benjamini_hochberg([r["mcnemar_p"] for r in per_condition])
        for r, q in zip(per_condition, qs):
            r["mcnemar_q"] = q
            # "Discriminates" requires BOTH significance AND the right sign: a validator
            # that flags the SHAM more than the damage is anti-discriminating, and a
            # bare q < 0.05 cannot tell the two apart.
            r["discriminates"] = (
                bool(q is not None and q < 0.05 and r["mcnemar_b"] > r["mcnemar_c"])
                if q is not None
                else None
            )
        rows.extend(per_condition)
    return rows


def write_validator_sensitivity(rows: Sequence[Dict[str, object]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDS})
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--control-manifest",
                    default="data/manifests/biased_samples_editdamage_v2.jsonl")
    ap.add_argument("--control-results-dir", default="results/v2_control/quality")
    ap.add_argument("--published-results-dir", default="results/v2/quality")
    ap.add_argument("--out", default="results/v2_control/metrics/validator_sensitivity_control.csv")
    args = ap.parse_args(argv)

    rows = build_validator_sensitivity(
        Path(args.control_manifest),
        Path(args.control_results_dir),
        Path(args.published_results_dir),
    )
    path = write_validator_sensitivity(rows, Path(args.out))
    print(f"wrote -> {path} ({len(rows)} rows)")
    for r in rows:
        sens = r["sensitivity_fixed"]
        ff = r["false_flag_rate_same_images"]
        print(
            f"  {r['validator_model']:18s} {r['condition']:18s} "
            f"sens {r['n_flagged_fixed']:3d}/{r['n_evaluable_fixed']:3d}"
            f"{'' if sens is None else f' = {sens:.4f}'}   "
            f"false-flag {'' if ff is None else f'{ff:.4f}'}   "
            f"b/c {r['mcnemar_b']}/{r['mcnemar_c']}  q={r['mcnemar_q']}  "
            f"discriminates={r['discriminates']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
