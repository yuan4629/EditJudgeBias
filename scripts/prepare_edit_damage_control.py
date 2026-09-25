#!/usr/bin/env python
"""WP-A4a — stage the validator SENSITIVITY control.

    python scripts/prepare_edit_damage_control.py manifest   # the 110-sample subset (free)
    ...  run_bias_injection with the editdamage config   # renders 440 images (free)
    python scripts/prepare_edit_damage_control.py verify     # did the damage actually land?

★ WHICH 110 IMAGES, AND WHY THAT CHOICE IS THE POINT.
The subset is exactly the `sham` arm's published draw — `_select(110, seed=42)`. `sham`
measures each validator's SPECIFICITY (what it flags when nothing changed) and this
measures its SENSITIVITY (what it misses when the edit is provably destroyed). Putting
both on the SAME 110 pictures makes them a matched pair rather than two numbers from two
samples, which is the identical argument WP-A3 makes for the matched floor — and it
means the ROC-style reading ("this validator flags 1.8% of nulls and catches X% of real
damage") is about one instrument on one set of images, not an average over two draws.

★ AND WHY `verify` EXISTS.
`distraction`'s spec violation was found because rebuilt canvas geometry was believed
over pixels that never landed: 22 reported, 21 real. A sensitivity control whose damage
silently failed to land would be worse than no control — it would report that the
validators are blind when in fact they were shown undamaged pictures. So the acceptance
check reads the rendered files back off disk and measures the change.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validator_draw import published_draw  # noqa: E402  (one implementation of `_select`)

JUDGE_SAMPLES = REPO / "data" / "manifests" / "samples_judge_v2.jsonl"
OUT_SAMPLES = REPO / "data" / "manifests" / "samples_editdamage_v2.jsonl"
OUT_MANIFEST = REPO / "data" / "manifests" / "biased_samples_editdamage_v2.jsonl"
CONTROL_BIAS = "sham"
EXPECT = 110
#: One entry per `configs/bias/edit_damage_*.yaml`.
CONDITIONS = ("edit_damage_100", "edit_damage_50", "edit_damage_25", "edit_damage_blur")

#: Mean |Δ| (0-255, whole frame) below which the control is NOT POSITIVE on that image.
#:
#: ★ MEASURED 2026-08-17, and it is a fact about the corpus rather than about this
#: injector: on 17 of these 110 published scenes, UNDOING THE EDIT ENTIRELY moves the
#: image by less than one grey level. The estimated edit region for those is 0.01%-2% of
#: the frame ("change the frisbee into a ball" -> 0.0001; "Remove the noise points" ->
#: 0.0003, a low-level restoration whose per-pixel change is under the diff threshold
#: everywhere). Some of that is an editor that barely did the job and some is an edit
#: that is genuinely tiny (cat's eyes are small), and the two are not separable here.
#:
#: What matters for A4b is that those images CANNOT test sensitivity in either direction:
#: the "damaged" version is indistinguishable from the "edited" one, so a validator that
#: passes it is correct, not blind. Counting them as misses would manufacture exactly the
#: insensitivity the arm is trying to measure. Sensitivity is therefore reported on the
#: subset where the control is positive, with this count disclosed beside it — the same
#: discipline as "a null is worth what its positive control is worth", applied to the
#: positive control itself.
WEAK_DAMAGE = 1.0


def cmd_manifest(_args) -> int:
    want = published_draw()[CONTROL_BIAS]
    print(f"sham published draw: {len(want)} base samples")
    if len(want) != EXPECT:
        print(f"REFUSING: expected {EXPECT}")
        return 1
    kept: List[str] = []
    with JUDGE_SAMPLES.open(encoding="utf-8") as fh, \
         OUT_SAMPLES.open("w", encoding="utf-8", newline="") as out:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["sample_id"] in want:
                out.write(line if line.endswith("\n") else line + "\n")
                kept.append(r["sample_id"])
    print(f"wrote {OUT_SAMPLES.relative_to(REPO)}  rows={len(kept)}")
    missing = sorted(want - set(kept))
    if missing:
        print(f"⚠ {len(missing)} sham base sample(s) absent from the judge subset: {missing[:3]}")
        return 1
    return 0


def cmd_verify(_args) -> int:
    if not OUT_MANIFEST.exists():
        print(f"  FAIL {OUT_MANIFEST.name} missing — run the injection first")
        return 1
    by_cond: Dict[str, List[dict]] = defaultdict(list)
    for line in OUT_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_cond[r["bias_type"]].append(r)

    ok = True
    print(f"{'condition':22}{'n':>5}{'mean|Δ|':>10}{'changed_frac':>14}{'area_frac':>11}  methods")
    for cond in CONDITIONS:
        rows = by_cond.get(cond, [])
        if not rows:
            print(f"  {cond:20} MISSING")
            ok = False
            continue
        chg = [r["bias_params"]["mean_abs_change"] for r in rows]
        frac = [r["bias_params"]["changed_frac"] for r in rows]
        area = [r["bias_params"]["region_area_frac"] for r in rows]
        methods = sorted({r["bias_params"]["region_method"] for r in rows})
        print(f"{cond:22}{len(rows):>5}{sum(chg)/len(chg):>10.2f}"
              f"{sum(frac)/len(frac):>14.4f}{sum(area)/len(area):>11.4f}  {methods}")
        weak = [r["biased_id"] for r in rows
                if r["bias_params"]["mean_abs_change"] < WEAK_DAMAGE]
        if cond == "edit_damage_100":
            # The anchor rung is the only HARD requirement: at severity 1.0 the region
            # is the original's pixels, so a zero here would mean the renderer failed.
            dead = [r["biased_id"] for r in rows if r["bias_params"]["changed_frac"] <= 0.0]
            if dead:
                print(f"      FAIL {len(dead)} image(s) with zero changed pixels: {dead[:3]}")
                ok = False
        print(f"      {len(weak)} image(s) below mean|Δ|={WEAK_DAMAGE} — the control is not "
              "positive there and they cannot test sensitivity")
        if "fallback" in methods:
            print("      FAIL a fallback region reached the control (see edit_damage.py guard 2)")
            ok = False

    # Monotonicity is the property that turns a yes/no control into a threshold.
    means = {c: sum(r["bias_params"]["mean_abs_change"] for r in by_cond.get(c, [])) /
                max(1, len(by_cond.get(c, []))) for c in ("edit_damage_25", "edit_damage_50",
                                                          "edit_damage_100")}
    ladder = [means["edit_damage_25"], means["edit_damage_50"], means["edit_damage_100"]]
    print(f"  severity ladder (25/50/100): {[round(x, 2) for x in ladder]}")
    if ladder != sorted(ladder):
        print("      FAIL the ladder is not monotone")
        ok = False
    print("VERIFY: " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


#: Where the C-class dose audit is frozen (same tree as the mask-polarity audit).
DOSE_AUDIT = REPO / "results" / "v2" / "metrics" / "edit_region_dose_audit.json"
JUDGE_MANIFEST = REPO / "data" / "manifests" / "biased_samples_full_v2.jsonl"


def cmd_region_dose(_args) -> int:
    """How big is the estimated edit region across the PUBLISHED breadth grid?

    Staging A4a surfaced this and it is not about A4a. The three C-class cues size
    themselves to `estimate_edit_region`, so their dose is that region's area — and on
    17 of the 110 control scenes, reverting the edit entirely moved the image by under
    one grey level because the region was 0.01%-2% of the frame. If that tail is
    material in the published grid, then some published C-class rows carry a cue that
    annotated, magnified or avoided a region of almost nothing.

    A small region is NOT automatically an estimator failure — a cat's eyes really are
    a small part of the frame, and a correct box around a small edit is correct. The two
    cannot be separated without a human read, so this freezes the DISTRIBUTION rather
    than a defect count. The direction is favourable and should be stated: a cue whose
    dose is near zero on part of the sample ATTENUATES the measured effect, so the
    published C-class numbers are a lower bound rather than an inflated one.
    """
    breadth = set()
    with (REPO / "data" / "manifests" / "samples_judge_v2.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if (r.get("metadata") or {}).get("subset_block") == "breadth":
                    breadth.add(r["sample_id"])

    per_cue: Dict[str, List[dict]] = defaultdict(list)
    with JUDGE_MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("bias_params") or {}
            # ONLY `region_annotation`. All three C-class cues call the same
            # `estimate_edit_region` with the same inputs, so the region is identical
            # across them for a given sample — but only this cue stores that region's
            # area in `area_frac`. `distraction` stores the STICKER's area there (a
            # constant 0.0397), and reading it as a region would say every breadth
            # sample has a sub-5% edit region, which is false.
            if r["bias_type"] == "region_annotation" and "area_frac" in p \
                    and r["base_sample_id"] in breadth:
                per_cue[r["bias_type"]].append(
                    {"sample_id": r["base_sample_id"], "area_frac": p["area_frac"],
                     "region_method": p.get("region_method")}
                )

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    out = {
        "n_breadth": len(breadth),
        "thresholds": thresholds,
        "measured_on": "region_annotation.area_frac",
        "applies_to": ["region_annotation", "zoom_inset", "distraction"],
        "note": (
            "All three C-class cues call estimate_edit_region with identical inputs, so "
            "the region is the same for a given sample; only region_annotation records "
            "its area. distraction's area_frac is the STICKER's area (constant 0.0397) "
            "and is NOT a region measurement."
        ),
        "per_cue": {},
    }
    for cue, rows in sorted(per_cue.items()):
        a = sorted(r["area_frac"] for r in rows)
        n = len(a)
        below = {str(t): sum(1 for x in a if x < t) for t in thresholds}
        methods = collections_counter(r["region_method"] for r in rows)
        out["per_cue"][cue] = {
            "n": n, "mean": round(sum(a) / n, 4), "median": round(a[n // 2], 4),
            "p05": round(a[int(n * 0.05)], 4), "min": round(a[0], 6), "max": round(a[-1], 4),
            "n_below": below, "region_method": methods,
            "smallest_10": [r["sample_id"] for r in
                            sorted(rows, key=lambda r: r["area_frac"])[:10]],
        }
        print(f"{cue}: n={n} median={a[n // 2]:.4f} mean={sum(a) / n:.4f} methods={methods}")
        for t in thresholds:
            print(f"    area_frac < {t:<6}: {below[str(t)]:3}/{n} ({below[str(t)] / n:.1%})")

    DOSE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    DOSE_AUDIT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {DOSE_AUDIT.relative_to(REPO)}")
    return 0


def collections_counter(it) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for x in it:
        c[str(x)] = c.get(str(x), 0) + 1
    return dict(sorted(c.items()))


def cmd_subset(args) -> int:
    """Write a single-condition slice of the control manifest.

    WHY THIS EXISTS.  `run_quality_validation` has `per_bias` (how many per bias) but no
    way to say *which* bias, so running one condition means handing it a manifest that
    holds only that condition.  WP-A4b uses it to buy glm-4v's sensitivity at the
    strongest damage only: glm-4v costs ~24x gpt-4o-mini and ~1.6x gemini per call
    (measured) and it does NOT gate,
    so paying for its full threshold curve would consume the arm's budget on the one
    validator whose verdicts never enter the QC subset.  One condition still answers the
    question its 10.3% false-flag floor raises: does it at least catch total damage?
    """
    want = args.condition
    if want not in CONDITIONS:
        print(f"unknown condition {want!r}; expected one of {CONDITIONS}", file=sys.stderr)
        return 2
    rows = [line for line in OUT_MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["bias_type"] == want]
    if not rows:
        print(f"no rows for {want} in {OUT_MANIFEST}", file=sys.stderr)
        return 1
    out = OUT_MANIFEST.with_name(f"{OUT_MANIFEST.stem}_{want.rsplit('_', 1)[-1]}.jsonl")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows ({want}) -> {out}")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest", help="write the 110-sample subset").set_defaults(fn=cmd_manifest)
    sub.add_parser("verify", help="measure that the damage landed").set_defaults(fn=cmd_verify)
    sub.add_parser("region-dose-audit",
                   help="freeze the C-class dose distribution over the breadth grid"
                   ).set_defaults(fn=cmd_region_dose)
    p_sub = sub.add_parser("subset", help="write a one-condition slice of the manifest")
    p_sub.add_argument("condition", choices=list(CONDITIONS))
    p_sub.set_defaults(fn=cmd_subset)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
