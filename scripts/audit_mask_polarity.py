"""Audit the C-class edit-region mask-polarity defect (free, no API).

★ WHY THIS EXISTS
`bias/edit_region.py::_load` used to flatten a shipped edit mask with
`Image.open(...).convert("RGB")`, which **drops the alpha channel**. MagicBrush -- the only
source in the pool that ships masks -- encodes the edited region as **alpha <= 127**, and
the surviving RGB is bright almost everywhere, so `point(p > 0).getbbox()` returned the
WHOLE FRAME. The three region-dependent C-class injectors (`region_annotation`,
`zoom_inset`, `distraction`) all consume that estimate, so every masked sample got a
full-frame region instead of the real edit region.

This script quantifies the blast radius in the ALREADY-COLLECTED main grid, so the defect
can be disclosed with numbers instead of re-spending on a re-run. It writes
`results/v2/metrics/mask_polarity_audit.json`.

Run:
    PYTHONPATH=src python scripts/audit_mask_polarity.py
    PYTHONPATH=src python scripts/audit_mask_polarity.py --samples data/manifests/samples_full_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from edit_judge_bias.bias.edit_region import Bbox, estimate_edit_region

#: The C-class injectors that call `estimate_edit_region`, i.e. the cues whose images
#: inherit the defect. Kept as data so the multiplier in the report is auditable rather
#: than a number typed into prose.
REGION_DEPENDENT_CUES = ("region_annotation", "zoom_inset", "distraction")

#: bbox IoU below which we call the region "materially changed". Generous on purpose: the
#: interesting claim is that the change is large, so the threshold should not be doing the
#: work. Measured 2026-07-30, every affected sample lands far below it (max IoU 0.88).
MATERIAL_IOU = 0.90


def _iou(a: Bbox, b: Bbox) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def audit(
    samples_path: Path,
    *,
    root: Optional[Path] = None,
    material_iou: float = MATERIAL_IOU,
) -> Dict:
    """Recompute every masked sample's edit region both ways and compare."""
    root = Path(root or ".")
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    masked = [
        r for r in rows
        if (r.get("metadata") or {}).get("has_mask") and (r.get("metadata") or {}).get("mask_path")
    ]

    per_sample: List[Dict] = []
    unreadable = 0
    for r in masked:
        mask_path = root / r["metadata"]["mask_path"]
        edited, original = root / r["edited_image_path"], root / r["original_image_path"]
        new = estimate_edit_region(edited, original, mask=mask_path)
        old = estimate_edit_region(edited, original, mask=mask_path, mask_polarity="luma_high")
        if new.method != "mask" or old.method != "mask":
            # A mask that neither path could read is a different problem; count it rather
            # than letting it average into the IoU as a zero.
            unreadable += 1
            continue
        iou = _iou(new.bbox, old.bbox)
        per_sample.append({
            "sample_id": r["sample_id"],
            "source_dataset": r["source_dataset"],
            "mask_polarity_used": new.mask_polarity,
            "bbox_fixed": list(new.bbox),
            "bbox_published": list(old.bbox),
            "area_frac_fixed": round(new.area_frac, 4),
            "area_frac_published": round(old.area_frac, 4),
            "bbox_iou": round(iou, 4),
            "materially_changed": iou < material_iou,
        })

    changed = [s for s in per_sample if s["materially_changed"]]
    ious = [s["bbox_iou"] for s in per_sample]
    return {
        "samples_manifest": str(samples_path),
        "judged_samples": len(rows),
        "samples_shipping_a_mask": len(masked),
        "samples_compared": len(per_sample),
        "masks_unreadable_by_either_path": unreadable,
        "material_iou_threshold": material_iou,
        "materially_changed_samples": len(changed),
        "materially_changed_by_source": dict(Counter(s["source_dataset"] for s in changed)),
        "bbox_iou_mean": round(st.mean(ious), 4) if ious else None,
        "bbox_iou_median": round(st.median(ious), 4) if ious else None,
        "bbox_iou_max": round(max(ious), 4) if ious else None,
        "area_frac_published_mean": (
            round(st.mean(s["area_frac_published"] for s in per_sample), 4) if per_sample else None
        ),
        "area_frac_fixed_mean": (
            round(st.mean(s["area_frac_fixed"] for s in per_sample), 4) if per_sample else None
        ),
        "region_dependent_cues": list(REGION_DEPENDENT_CUES),
        "affected_biased_images": len(changed) * len(REGION_DEPENDENT_CUES),
        "share_of_each_cue_rows": (
            round(len(changed) / len(rows), 4) if rows else None
        ),
        "per_sample": per_sample,
    }


def _print_report(rep: Dict) -> None:
    print(f"manifest                      {rep['samples_manifest']}")
    print(f"judged samples                {rep['judged_samples']}")
    print(f"  shipping a mask             {rep['samples_shipping_a_mask']}")
    print(f"  compared                    {rep['samples_compared']}"
          f"   (unreadable: {rep['masks_unreadable_by_either_path']})")
    print(f"bbox IoU published-vs-fixed   mean {rep['bbox_iou_mean']}  "
          f"median {rep['bbox_iou_median']}  max {rep['bbox_iou_max']}")
    print(f"mean area_frac  published     {rep['area_frac_published_mean']}   "
          f"fixed {rep['area_frac_fixed_mean']}")
    print(f"materially changed (IoU<{rep['material_iou_threshold']})   "
          f"{rep['materially_changed_samples']} / {rep['samples_compared']}"
          f"   {rep['materially_changed_by_source']}")
    print()
    print(f"=> affected already-judged biased images: {rep['materially_changed_samples']}"
          f" x {len(rep['region_dependent_cues'])} cues = {rep['affected_biased_images']}")
    print(f"   cues: {', '.join(rep['region_dependent_cues'])}")
    print(f"   share of each cue's rows: {rep['share_of_each_cue_rows']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default="data/manifests/samples_judge_v2.jsonl")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/v2/metrics/mask_polarity_audit.json")
    ap.add_argument("--material-iou", type=float, default=MATERIAL_IOU)
    args = ap.parse_args()

    rep = audit(Path(args.samples), root=Path(args.root), material_iou=args.material_iou)
    _print_report(rep)

    out = Path(args.root) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
