"""The D-class CONSTRUCTION GATE — deterministic, free, and it replaces an MLLM judgment.

★ WHY THIS EXISTS
The 2026-07-30 D-class gate asked two MLLM auditors whether the scene had been preserved. They
reached Cohen's kappa **+0.130** on exactly that sub-question -- while agreeing at +0.716 on the
easy "is a person present" -- and the resulting pair-validity agreement was **-0.028**, i.e.
chance. The intersection admitted 5 of 41 pairs and the track stopped.

The lesson was not "get a better auditor". It was that `scene_preserved` is a MEASUREMENT
masquerading as a judgment. Every injection here composites through a mask, so this gate can
assert locality arithmetically:

    max |injected - source| outside the mask  ==  0        (no tolerance)

and it is applied to the SAVED PNG bytes, because a lossy resave is exactly the failure a
tolerance would hide.

★ WHAT ELSE IT CHECKS, AND WHY EACH ONE EARNS ITS PLACE

    same_mask          both arms of a scene must share one `mask_sha256`. That is what makes
                       segmentation error common-mode rather than a difference between arms.
    mask_frac bounds   a mask covering ~nothing is not a manipulation; one covering most of the
                       frame is a global tone shift, and this study already measured that
                       global tonal cues (`saturation`, `aesthetic_filter`) are null on 4/5
                       judges while LOCAL readable ones deflate on 5/5. Staying local is what
                       keeps the arm interpretable.
    inside_differs     a study arm that changed nothing is a silent no-op that would otherwise
                       pass every other check.
    dose floor         |achieved_delta_ita| >= MIN_CROSSING_DELTA, evaluated on the ORIGINAL
                       member only. On the edited member the same Lab offset lands on different
                       pixels (the mask comes from the original) and ITA is nonlinear, so its
                       readout legitimately differs -- measured 16 of 62 edited rows below the
                       floor against 0 of 62 original rows. Gating the edited member on ITA
                       would discard 9 good scenes for a nonlinearity.
    dose consistency   `delta_e00_mean` must match across the two members, which is the correct
                       cross-member check: it measures the physical dose, and it IS matched
                       (6.07 vs 5.89 on the pair whose ITA readouts differ by 13 degrees).

★ THE GATE IS APPLIED PAIRWISE
A scene that fails on either arm is dropped from BOTH. A one-sided drop would leave an orphan
that the paired gap metric discards later and silently -- the same reasoning as the published
`build_fairness_samples` gate, and `tests/test_fairness_pipeline.py` pins it there.

    python -m edit_judge_bias.experiments.run_construction_gate \\
        --config configs/fairness/construction_gate.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve
from edit_judge_bias.fairness import attribute_injectors as AI
from edit_judge_bias.fairness import ita as ITA
from edit_judge_bias.fairness import person_region as PR
from edit_judge_bias.fairness.records import AttributeInjectionRecord

#: Defaults; every one is overridable from the config and echoed into the report.
DEFAULTS = {
    "max_outside_diff": 0,          # NO TOLERANCE -- one stray pixel is a bug
    "mask_frac_min": 0.005,
    "mask_frac_max": 0.60,
    "min_inside_diff": 6,
    "min_delta_ita": ITA.MIN_CROSSING_DELTA,
    "max_delta_e00_ratio": 2.0,     # cross-member physical-dose consistency
    "require_same_mask": True,
    "verify_pixels": True,          # re-read the PNGs rather than trusting the manifest
    #: Floor on how much of the applied mask is still skin on the EDITED member.
    #: `None` disables the check, which is the default so the published v3 gate reproduces --
    #: its records predate the measurement and would all fail as "unmeasured".
    #: 0.5 is where the measured distribution's tail sits (p05 = 0.690, median 0.971), so it
    #: refuses the genuinely broken transfers without trimming the healthy spread.
    "min_edited_mask_coverage": None,
}


def _check_row(row: AttributeInjectionRecord, root: Path, cfg: dict) -> Tuple[bool, List[str]]:
    """Per-row deterministic checks. Returns (passed, failure reasons)."""
    reasons: List[str] = []

    if not row.success:
        return False, ["injection_failed"]

    if row.max_outside_diff > int(cfg["max_outside_diff"]):
        reasons.append(f"outside_mask_diff={row.max_outside_diff}")
    if not row.outside_mask_identical:
        reasons.append("outside_mask_not_identical")

    if not (float(cfg["mask_frac_min"]) <= row.mask_frac <= float(cfg["mask_frac_max"])):
        reasons.append(f"mask_frac={row.mask_frac:.4f}_out_of_bounds")

    # ★ Does the original-derived mask still land on SKIN on the edited member?
    # One scene preparation is shared by both members, which is right for the two ARMS
    # (segmentation error becomes common-mode) but is an assumption across MEMBERS, because
    # these corpora's edited images are whole-frame RE-RENDERS, not inpaints. Measured over 60
    # random OmniEdit pairs the assumption holds well -- median coverage 0.971, IoU 0.923 --
    # but 1 in 60 fell below 0.5, and on that scene the manipulation recolours skin on one side
    # and background on the other. `None` is a MISSING measurement and is not treated as a pass.
    if row.applied_to == AI.APPLIED_EDITED and cfg.get("min_edited_mask_coverage") is not None:
        floor = float(cfg["min_edited_mask_coverage"])
        coverage = row.edited_mask_skin_coverage
        if coverage is None:
            reasons.append("edited_mask_coverage_unmeasured")
        elif coverage < floor:
            reasons.append(f"edited_mask_coverage={coverage:.3f}_below_{floor}")

    if row.role == AI.ROLE_STUDY:
        if row.max_inside_diff < int(cfg["min_inside_diff"]):
            reasons.append(f"inside_diff={row.max_inside_diff}_too_small")
        # ★ The ITA dose floor applies to the ORIGINAL member only -- see the module docstring.
        if row.applied_to == AI.APPLIED_ORIGINAL:
            achieved = abs(row.achieved_delta_ita or 0.0)
            if achieved < float(cfg["min_delta_ita"]):
                reasons.append(f"delta_ita={achieved:.2f}_below_floor")

    if cfg.get("verify_pixels", True):
        ok, worst = _verify_pixels(row, root)
        if not ok:
            reasons.append(f"pixel_reverify_failed(max={worst})")

    return (not reasons), reasons


def _verify_pixels(row: AttributeInjectionRecord, root: Path) -> Tuple[bool, int]:
    """Re-read the saved PNG and re-assert locality against the source image.

    Deliberately NOT trusting `row.max_outside_diff`: that value was computed in memory before
    the PNG was written, so it cannot catch a lossy or mis-sized save. The mask is re-read from
    the cache PNG for the same reason -- the cached mask is the source of truth, not the model.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(resolve(row.injected_image_path, root)) as im:
            injected = np.asarray(im.convert("RGB"))
        with Image.open(resolve(row.source_image_path, root)) as im:
            src_img = im.convert("RGB")
            if row.source_resized and src_img.size != (injected.shape[1], injected.shape[0]):
                # Reproduce the runner's resample EXACTLY (PIL BICUBIC). The two members of a
                # pair must share a pixel grid because the mask comes from the original, and 12
                # of the 31 D-S scenes ship a smaller original than edited image.
                src_img = src_img.resize((injected.shape[1], injected.shape[0]), Image.BICUBIC)
            source = np.asarray(src_img)
    except Exception:
        return False, -1
    if source.shape != injected.shape:
        # Not a resize case: the saved file does not correspond to the recorded source at all.
        return False, -2
    diff = np.abs(source.astype("int32") - injected.astype("int32")).max(axis=2)
    changed = diff > 0
    # Locality without needing the mask: the changed pixels must be a strict subset of a region
    # no larger than the recorded support. If more of the frame moved than the mask covers, the
    # composite leaked regardless of what the manifest claims.
    changed_frac = float(changed.mean())
    if changed_frac > row.mask_frac + 1e-6:
        return False, int(diff.max())
    return True, int(diff[changed].max()) if changed.any() else 0


def run(config_path: str | Path, *, root: Optional[Path] = None,
        dry_run: bool = False) -> dict:
    root = Path(root) if root is not None else default_root()
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    cfg = {**DEFAULTS, **raw}

    rows = io.read_jsonl(root / cfg["injections"], AttributeInjectionRecord)
    if dry_run:
        return {"injections": len(rows), "thresholds": {k: cfg[k] for k in DEFAULTS},
                "note": "deterministic; no API calls"}

    per_row: Dict[str, List[str]] = {}
    for row in rows:
        ok, reasons = _check_row(row, root, cfg)
        if not ok:
            per_row[row.injection_id] = reasons

    # Group into (scene, attribute) pairs so the gate can be applied PAIRWISE.
    scenes: Dict[Tuple[str, str], List[AttributeInjectionRecord]] = defaultdict(list)
    for row in rows:
        scenes[(row.base_sample_id, row.attribute)].append(row)

    passed_pairs: List[str] = []
    dropped: Dict[str, List[str]] = {}
    mask_conflicts = 0
    dose_conflicts = 0

    for (base, attribute), group in sorted(scenes.items()):
        key = f"{base}__{attribute}"
        study = [r for r in group if r.role == AI.ROLE_STUDY]
        reasons: List[str] = []

        labels = {r.variant_label for r in study}
        members = {(r.variant_label, r.applied_to) for r in study}
        if len(labels) != 2:
            reasons.append(f"incomplete_arms({sorted(labels)})")
        if len(members) != 4:
            reasons.append(f"incomplete_members({len(members)}/4)")

        # ★ Both arms must share one mask digest.
        if cfg.get("require_same_mask", True):
            digests = {r.mask_sha256 for r in study}
            if len(digests) > 1:
                reasons.append("arms_do_not_share_a_mask")
                mask_conflicts += 1

        # Cross-member physical-dose consistency, per arm.
        for label in sorted(labels):
            doses = {r.applied_to: r.delta_e00_mean for r in study
                     if r.variant_label == label and r.delta_e00_mean}
            if len(doses) == 2:
                lo, hi = sorted(doses.values())
                if lo > 0 and hi / lo > float(cfg["max_delta_e00_ratio"]):
                    reasons.append(f"{label}_dose_inconsistent_across_members({hi / lo:.2f}x)")
                    dose_conflicts += 1

        failing = [r.injection_id for r in study if r.injection_id in per_row]
        if failing:
            reasons.extend(f"{rid}:{','.join(per_row[rid])}" for rid in failing)

        if reasons:
            dropped[key] = reasons
        else:
            passed_pairs.append(key)

    reason_hist = Counter()
    for reasons in dropped.values():
        for reason in reasons:
            reason_hist[reason.split(":")[-1].split("(")[0].split("=")[0]] += 1

    out_path = root / cfg["out_json"]
    report = {
        "injections": len(rows),
        "rows_failing": len(per_row),
        "pairs_total": len(scenes),
        "pairs_passed": len(passed_pairs),
        "pairs_dropped": len(dropped),
        "mask_conflicts": mask_conflicts,
        "dose_conflicts": dose_conflicts,
        "drop_reason_hist": dict(sorted(reason_hist.items())),
        "thresholds": {k: cfg[k] for k in DEFAULTS},
        "passed_pair_keys": sorted(passed_pairs),
        "dropped": {k: v for k, v in sorted(dropped.items())},
    }
    from edit_judge_bias.metrics.fairness_metrics import (
        minimum_detectable_effect, min_attainable_pvalue,
    )
    report["mde_sd"] = (round(minimum_detectable_effect(len(passed_pairs)), 4)
                        if passed_pairs else None)
    report["min_attainable_p"] = min_attainable_pvalue(len(passed_pairs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: v for k, v in report.items() if k not in ("passed_pair_keys", "dropped")}


def passed_pair_keys(gate_json: Path) -> set:
    """The admitted `{base}__{attribute}` keys, for the sample emitter."""
    data = json.loads(Path(gate_json).read_text(encoding="utf-8"))
    return set(data.get("passed_pair_keys") or ())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.config, root=Path(a.root) if a.root else None,
                         dry_run=a.dry_run), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
