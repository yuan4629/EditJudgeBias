"""Sensitivity analyses for the headline claims (WP-A1).

Three of the four simulated reviewers converged on the same complaint: the headline
numbers are reported at exactly one analytic choice, so a reader cannot tell a robust
effect from one that lives or dies on that choice. These tables answer that by
recomputing the same claims under *declared alternatives* and shipping the whole
trajectory, not the endpoint.

``distraction_spec_violation_audit.json`` (A1a, step 1)
    A frozen, MEASURED record of which published `distraction` stimuli violate the
    injector's own specification. It has to be frozen now because WP-A2 overwrites the
    pixels in place: after the re-injection this number is 0 by construction, and
    "0 after the fix" is only meaningful next to the pre-fix count.

``claim_a_sensitivity_exclusion.csv`` (A1a, step 2)
    Claim A's three region-dependent cues recomputed with the affected samples
    excluded, at two nested levels.

``claim_a_by_mos_stratum.csv`` (A1d)
    Claim A's six headline cues recomputed inside human-MOS strata. Answers the
    extrapolation question a reviewer asks about any benchmark built on 2023-2024
    editors: *does the judge still deflate on the edits humans rated highly, or is the
    effect carried by bad edits where a deduction is cheap?* Tertiles are taken WITHIN
    source, because the two breadth sources with human labels are on scales that are
    not comparable — EBench-18K's MOS is z-scored and continuous (160 distinct values
    in 160 samples), ImagenHub's is a coarse 13-level scale — so a pooled tertile would
    be a source selector wearing a quality label.

    ⚠️ Every row carries `mde_sd` and `detectable`. A tertile is a third of the human-
    labelled samples, so the smallest headline effects are BELOW the stratum's own
    minimum detectable effect and "not significant here" says nothing about them. This
    is the `zoom_inset` trap in a new place, and the columns exist so a reader cannot
    fall into it.

The defect being probed is documented in the re-injection design notes: 91 judged samples
ship an edit mask whose polarity was misread, so the "edit region" handed to the three
C-class injectors was the whole frame. For `distraction` that is worse than a dose
problem — the injector promises to place its sticker *outside* the edit region, and
with the region reading as the full frame all 8 candidate positions tie at the same
overlap, so the preference collapses to an RNG pick.

    ★ MEASURED HERE, 2026-08-16: **21 of the 91**, not the 22 recorded in
    the re-injection design notes. They reconstructed the sticker's *canvas*
    rectangle and intersected it with the true region. But the assets are drawn on a
    transparent RGBA square (a star fills roughly half of its own bounding box), so a
    canvas can overlap a region that no drawn pixel touches. On
    `mb_362944_1_human_annotator` exactly that happens. Counting CHANGED PIXELS instead
    of the canvas gives 21. The canvas figure is retained in the frozen JSON so the
    correction is auditable rather than a silent renumbering.

Two exclusion levels, both reported against the published baseline so the reader sees
the trajectory:

``published``           nothing dropped (the number currently in the paper)
``drop_spec_viol``      the 21 samples whose sticker provably altered pixels inside
                        the true edit region
``drop_mask_affected``  all 91 samples whose mask polarity was misread

Read `drop_spec_viol` on the other two cues as a CONTROL, not as a test: those 21
samples are not special for `region_annotation` / `zoom_inset`, so a cell that moves
there is telling you about sampling noise at n=590, not about the defect.

⚠️ This is a sensitivity analysis on the PUBLISHED pixels. It is not a substitute for
the re-injection (WP-A2), which produces corrected pixels. It answers a different and
strictly weaker question: "does the headline survive if we simply delete the suspect
samples?" A headline that survives here can only be refined by the re-injection, not
overturned by it.

Every level re-runs the *whole* claim A builder, so the BH family stays the declared
`claim_A` family (5 judges x 12 cues) computed on that level's sample — dropping
samples moves every cell, including ones not extracted here, and correcting only the
three extracted cues would be a smaller family than the one the paper declares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from edit_judge_bias.bias.distraction import _candidate_positions, _intersection
from edit_judge_bias.data.manifest_utils import PathLike, default_root
from edit_judge_bias.experiments.aggregate_results import (
    sample_ids_matching,
)
from edit_judge_bias.experiments.build_claim_tables import (
    PUBLISHED_ROSTER,
    _judges,
    attach_bh,
    build_claim_a,
    write_csv,
)

#: The three injectors that consume `estimate_edit_region`, and therefore the three
#: whose stimuli the mask-polarity defect touched
#: (`scripts/audit_mask_polarity.py::REGION_DEPENDENT_CUES`).
REGION_CUES: Tuple[str, ...] = ("region_annotation", "zoom_inset", "distraction")

#: Per-channel difference above which a pixel counts as changed. The stickers are
#: opaque and saturated, so the measurement is nowhere near this threshold's knee;
#: it exists only to ignore PNG round-trip noise.
CHANGED_PIXEL_THRESHOLD = 8

#: `configs/bias/distraction.yaml`. Only used by the geometry cross-check, which is a
#: control on the pixel measurement rather than the measurement itself.
DISTRACTION_MARGIN_FRAC = 0.02


# --------------------------------------------------------------------------- #
# geometry — a cross-check on the pixel measurement, not the measurement       #
# --------------------------------------------------------------------------- #
def sticker_rect(
    bias_params: Dict[str, Any], *, margin_frac: float = DISTRACTION_MARGIN_FRAC
) -> Tuple[int, int, int, int]:
    """Rebuild the pasted sticker's CANVAS rectangle from a `distraction` manifest row.

    Mirrors `DistractionInjector.transform`: the same `_candidate_positions` table and
    the same clamp. The frame size is recovered from `avoided_bbox`, which on every
    affected row is the full frame (that being the defect).

    ⚠️ This is the transparent square the asset is drawn on, NOT the drawn pixels. The
    difference is the whole reason the published count was 22 and the measured one is
    21 — see the module docstring.
    """
    x0, y0, x1, y1 = bias_params["avoided_bbox"]
    w, h = x1 - x0, y1 - y0
    sw, sh = bias_params["sticker_size"]
    margin = max(1, int(min(w, h) * margin_frac))
    px, py = _candidate_positions(w, h, sw, sh, margin)[bias_params["corner"]]
    px, py = max(0, min(px, w - sw)), max(0, min(py, h - sh))
    return (px, py, px + sw, py + sh)


def minimum_candidate_overlap(
    bias_params: Dict[str, Any],
    region: Sequence[int],
    frame_size: Tuple[int, int],
    *,
    margin_frac: float = DISTRACTION_MARGIN_FRAC,
) -> int:
    """Smallest edit-region overlap ANY of the injector's 8 candidate slots can reach.

    `DistractionInjector` sorts its eight candidate positions by overlap with the edit
    region and takes the minimum, breaking ties with the RNG. So a positive value here
    means **no placement satisfies the specification**: the region is large enough that
    every corner and edge slot touches it, and the injector is already doing the best
    the design allows.

    ⚠️ Unlike :func:`sticker_rect`, the frame size is taken from the IMAGE, not from
    `avoided_bbox`. That shortcut was only ever valid before the WP-A2 fix, when the
    recorded region was the whole frame by definition; afterwards `avoided_bbox` is the
    (small) real region and reusing it would compute the candidates on the wrong canvas.
    """
    from edit_judge_bias.bias.distraction import _candidate_positions

    width, height = frame_size
    sw, sh = bias_params["sticker_size"]
    margin = max(1, int(min(width, height) * margin_frac))
    return min(
        _intersection((x, y, x + sw, y + sh), tuple(region))
        for x, y in _candidate_positions(width, height, sw, sh, margin).values()
    )


def changed_pixel_bbox(
    edited_path: PathLike, biased_path: PathLike, *, threshold: int = CHANGED_PIXEL_THRESHOLD
):
    """Bounding box of the pixels the injector actually altered, or None."""
    from PIL import Image, ImageChops

    a = Image.open(edited_path).convert("RGB")
    b = Image.open(biased_path).convert("RGB")
    if a.size != b.size:
        raise ValueError(f"{edited_path} is {a.size} but {biased_path} is {b.size}")
    mask = ImageChops.difference(a, b).convert("L").point(lambda p: 255 if p > threshold else 0)
    return mask.getbbox()


# --------------------------------------------------------------------------- #
# A1a step 1 — measure and freeze the spec violations                          #
# --------------------------------------------------------------------------- #
def _load_audit(audit_path: PathLike) -> Dict[str, List[int]]:
    """sample_id -> the CORRECTED edit-region bbox, from the mask-polarity audit."""
    data = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    return {r["sample_id"]: r["bbox_fixed"] for r in data["per_sample"]}


def _distraction_rows(biased_manifest: PathLike, wanted: Set[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with Path(biased_manifest).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("bias_type") == "distraction" and r.get("base_sample_id") in wanted:
                out[r["base_sample_id"]] = r
    return out


def measure_spec_violations(
    *,
    audit_path: PathLike,
    biased_manifest: PathLike,
    samples_manifest: PathLike,
    root: Optional[Path] = None,
    threshold: int = CHANGED_PIXEL_THRESHOLD,
) -> dict:
    """Measure, from the pixels on disk, which stickers landed on the true edit region.

    Returns the record that gets frozen to `distraction_spec_violation_audit.json`.
    Also carries the canvas-geometry count, so the discrepancy that corrects
    the design notes travels with the number instead of being asserted in prose.
    """
    from PIL import Image, ImageChops

    root = Path(root) if root else default_root()
    fixed = _load_audit(audit_path)
    rows = _distraction_rows(biased_manifest, set(fixed))
    edited: Dict[str, str] = {}
    with Path(samples_manifest).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            edited[r["sample_id"]] = r["edited_image_path"]

    per_sample: List[dict] = []
    canvas_hits: Set[str] = set()
    contained = 0
    for sid, row in sorted(rows.items()):
        a = Image.open(root / edited[sid]).convert("RGB")
        b = Image.open(root / row["biased_image_path"]).convert("RGB")
        if a.size != b.size:
            raise ValueError(f"{sid}: edited {a.size} vs biased {b.size}")
        mask = ImageChops.difference(a, b).convert("L").point(
            lambda p: 255 if p > threshold else 0
        )
        region = tuple(fixed[sid])
        total = sum(1 for p in mask.get_flattened_data() if p)
        inside = sum(1 for p in mask.crop(region).get_flattened_data() if p)

        canvas = sticker_rect(row["bias_params"])
        if _intersection(canvas, region) > 0:
            canvas_hits.add(sid)
        ink = mask.getbbox()
        if ink and (ink[0] >= canvas[0] and ink[1] >= canvas[1]
                    and ink[2] <= canvas[2] and ink[3] <= canvas[3]):
            contained += 1

        # Could ANY of the eight slots have avoided the region? A positive minimum
        # means the specification is unsatisfiable on this sample, which is a
        # different fact from "the injector placed it badly" and has to be told
        # apart from it — after the WP-A2 fix every surviving violation is of this
        # kind, and reporting them as failures would misdescribe the repair.
        floor = minimum_candidate_overlap(row["bias_params"], region, a.size)
        per_sample.append({
            "sample_id": sid,
            "biased_id": row["biased_id"],
            "asset": row["bias_params"].get("asset"),
            "corner": row["bias_params"].get("corner"),
            "true_edit_region": list(region),
            "region_frac_of_frame": round(
                ((region[2] - region[0]) * (region[3] - region[1]))
                / float(a.size[0] * a.size[1]), 6
            ),
            "changed_px_total": total,
            "changed_px_inside_region": inside,
            "frac_of_sticker_inside_region": round(inside / total, 6) if total else None,
            "violates_spec": bool(inside > 0),
            "min_possible_overlap_px": floor,
            "violation_unavoidable": bool(inside > 0 and floor > 0),
        })

    viol = [r for r in per_sample if r["violates_spec"]]
    fracs = sorted(r["frac_of_sticker_inside_region"] for r in viol)
    mid = len(fracs) // 2
    return {
        "generated_by": "edit_judge_bias.experiments.build_sensitivity_tables",
        "method": (
            "pixel diff of edited vs published biased image (per-channel > "
            f"{threshold}/255), intersected with mask_polarity_audit.bbox_fixed"
        ),
        "why_frozen": (
            "WP-A2 overwrites these pixels in place; after the re-injection the count "
            "is 0 by construction, and '0 after the fix' only means something next to "
            "the pre-fix count measured here"
        ),
        "changed_pixel_threshold": threshold,
        "n_candidates": len(per_sample),
        "n_violating": len(viol),
        "n_violating_canvas_geometry": len(canvas_hits),
        "canvas_only_false_alarms": sorted(canvas_hits - {r["sample_id"] for r in viol}),
        "canvas_geometry_note": (
            "the assets are drawn on a transparent RGBA square, so the canvas can "
            "overlap a region no drawn pixel touches; an earlier design note "
            "counted canvases and reported 22, the measured count is 21"
        ),
        "reconstruction_control": {
            "ink_bbox_inside_reconstructed_canvas": f"{contained}/{len(per_sample)}",
            "reads": (
                "validates sticker_rect against the pixels: every measured ink box must "
                "sit inside the canvas the reconstruction predicts"
            ),
        },
        "frac_inside_min": fracs[0] if fracs else None,
        "frac_inside_median": fracs[mid] if fracs else None,
        "frac_inside_mean": round(sum(fracs) / len(fracs), 6) if fracs else None,
        "frac_inside_max": fracs[-1] if fracs else None,
        "n_over_50pct_inside": sum(1 for f in fracs if f > 0.5),
        "n_over_90pct_inside": sum(1 for f in fracs if f > 0.9),
        "changed_px_inside_region_total": sum(
            r["changed_px_inside_region"] for r in per_sample
        ),
        "n_violating_unavoidable": sum(1 for r in viol if r["violation_unavoidable"]),
        "unavoidable_note": (
            "a violation is 'unavoidable' when all eight of the injector's candidate "
            "slots intersect the edit region, i.e. no placement satisfies the spec. "
            "Before the WP-A2 fix the region was the whole frame, so every violation "
            "was trivially of this kind; after it, a surviving violation means the "
            "dataset's own mask really does cover most of the picture"
        ),
        "per_sample": per_sample,
    }


def spec_violating_ids(violation_audit: PathLike) -> Dict[str, float]:
    """sample_id -> fraction of the sticker inside the true edit region, for violators."""
    data = json.loads(Path(violation_audit).read_text(encoding="utf-8"))
    return {
        r["sample_id"]: r["frac_of_sticker_inside_region"]
        for r in data["per_sample"] if r["violates_spec"]
    }


def mask_affected_ids(audit_path: PathLike) -> Set[str]:
    """Every judged sample whose shipped mask polarity was misread (all 91)."""
    return set(_load_audit(audit_path))


# --------------------------------------------------------------------------- #
# A1a step 2 — claim A under two exclusion levels                              #
# --------------------------------------------------------------------------- #
def build_exclusion_sensitivity(
    results_dir: PathLike,
    *,
    samples_path: PathLike,
    subset_filter: Optional[dict],
    audit_path: PathLike,
    violation_audit: PathLike,
    cues: Sequence[str] = REGION_CUES,
) -> List[dict]:
    """Claim A's region cues at `published` / `drop_spec_viol` / `drop_mask_affected`."""
    if not subset_filter:
        raise ValueError(
            "refusing to build the sensitivity table without a subset filter: claim A "
            "is the breadth block, and pooling the anchor block moves every cell "
            "(the measured n=1,196 accident)"
        )
    base_ids = sample_ids_matching(samples_path, subset_filter)

    spec = set(spec_violating_ids(violation_audit))
    affected = mask_affected_ids(audit_path)
    if not spec <= affected:
        raise ValueError("spec violations are not a subset of the mask-affected samples")

    levels = [
        ("published", set()),
        ("drop_spec_viol", spec),
        ("drop_mask_affected", affected),
    ]

    published: Dict[Tuple[str, str], dict] = {}
    rows: List[dict] = []
    for level, drop in levels:
        keep = base_ids - drop
        for cell in build_claim_a(Path(results_dir), keep):
            key = (cell["judge_model"], cell["bias_type"])
            if level == "published":
                published[key] = cell
            if cell["bias_type"] not in cues:
                continue
            base = published.get(key, {})
            shift, base_shift = cell["mean_shift"], base.get("mean_shift")
            rows.append({
                "level": level,
                "judge_model": cell["judge_model"],
                "bias_type": cell["bias_type"],
                "n_samples_dropped": len(base_ids) - len(keep),
                "n": cell["n"],
                "score_field": cell["score_field"],
                "mean_shift": shift,
                "ci_low": cell["ci_low"],
                "ci_high": cell["ci_high"],
                "p_value": cell["p_value"],
                "q_value": cell["q_value"],
                "significant_bh": cell["significant_bh"],
                "inside_placebo_bound": cell.get("inside_placebo_bound"),
                "family": cell["family"],
                # --- trajectory columns: what this level did to the headline ---
                "published_mean_shift": base_shift,
                "shift_delta_vs_published": (
                    None if shift is None or base_shift is None
                    else round(shift - base_shift, 4)
                ),
                "published_significant_bh": base.get("significant_bh"),
                "sign_preserved": (
                    None if shift is None or base_shift is None
                    else bool(shift * base_shift > 0)
                ),
                # ⚠️ `lost_significance` is defined ONLY on cells that were significant
                # in the published table, and is None elsewhere. A cell that was never
                # significant cannot "stop surviving", and counting it as a loss
                # inflates the apparent fragility — `zoom_inset x gpt-4o-viescore`
                # (published +0.11, q=0.91) is exactly that trap.
                "lost_significance": (
                    None if not base.get("significant_bh")
                    else not bool(cell["significant_bh"] and shift is not None
                                  and base_shift is not None and shift * base_shift > 0)
                ),
            })
    return rows


# --------------------------------------------------------------------------- #
# A1d — claim A inside human-MOS strata                                        #
# --------------------------------------------------------------------------- #
#: The six cells §5.1 leads with: five cues that deflate on 5/5 judges, plus the one
#: that inflates on 5/5. Claim A is still rebuilt WHOLE inside every stratum — the BH
#: family stays the declared 5 x 12, and only these six are extracted.
HEADLINE_CUES: Tuple[str, ...] = (
    "distraction", "text_overlay", "padding", "region_annotation", "brightness",
    "bandwagon",
)

#: Stratum order, published first so every row can carry its own displacement.
MOS_STRATA: Tuple[str, ...] = (
    "published", "has_human", "mos_tertile_low", "mos_tertile_mid", "mos_tertile_high",
)


def within_source_tertiles(
    samples: Sequence, keep_ids: Set[str], *, sources: Optional[Set[str]] = None
) -> Dict[str, Set[str]]:
    """`{stratum: sample_ids}` with the tertile cut taken inside each source.

    Ties are assigned by the cut, not by the count: with a 13-level scale a third of
    the rows cannot be split exactly, and forcing equal sizes would put two identical
    human scores in different quality strata. The realised sizes are reported instead.
    """
    with_human = [
        s for s in samples
        if s.sample_id in keep_ids and s.human_score is not None
        and (sources is None or s.source_dataset in sources)
    ]
    strata: Dict[str, Set[str]] = {
        "published": set(keep_ids),
        "has_human": {s.sample_id for s in with_human},
        "mos_tertile_low": set(), "mos_tertile_mid": set(), "mos_tertile_high": set(),
    }
    by_source: Dict[str, List] = {}
    for s in with_human:
        by_source.setdefault(s.source_dataset, []).append(s)
    for group in by_source.values():
        values = sorted(float(s.human_score) for s in group)
        lo = values[len(values) // 3]
        hi = values[(2 * len(values)) // 3]
        for s in group:
            score = float(s.human_score)
            key = ("mos_tertile_low" if score < lo
                   else "mos_tertile_high" if score >= hi else "mos_tertile_mid")
            strata[key].add(s.sample_id)
    return strata


def _shift_dispersion(
    results_dir: Path, keep_sample_ids: Set[str]
) -> Dict[Tuple[str, str], Tuple[Optional[float], int]]:
    """(judge, bias) -> (SD of the per-sample shift, n).

    Needed because `mean_shift` alone cannot be read against an MDE: the MDE is in
    units of SD(paired difference), and mixing that denominator with SD(score) has
    already doubled one published ratio in this project.
    """
    import statistics

    from edit_judge_bias.metrics.scoring_metrics import compute_score_shifts

    out: Dict[Tuple[str, str], Tuple[Optional[float], int]] = {}
    for model in _discover_scoring_models(results_dir):
        originals = [r for r in _read_results(results_dir / "raw_judgments"
                                              / f"scoring__{model}.jsonl")
                     if r.sample_id in keep_sample_ids]
        biased = [r for r in _read_results(results_dir / "biased_judgments"
                                           / f"scoring__{model}.jsonl")
                  if r.sample_id in keep_sample_ids]
        if not biased:
            continue
        for st in compute_score_shifts(originals, biased):
            sd = (statistics.stdev(st.shifts) if len(st.shifts) > 1 else None)
            out[(st.judge_model, st.bias_type)] = (sd, st.n)
    return out


def _discover_scoring_models(
    results_dir: Path,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[str]:
    """Roster first, coverage second -- see `build_claim_tables._judges`.

    Coverage decides "is this judge finished"; the roster decides "is this judge in this
    published table".  Both are needed: WP-A5's sixth judge is finished on the breadth
    arm and still must not appear here.
    """
    return _judges(Path(results_dir), "scoring", roster)


def _read_results(path: Path):
    from edit_judge_bias.data import io
    from edit_judge_bias.data.schema import JudgeResult

    return io.read_jsonl(path, JudgeResult) if path.exists() else []


def build_mos_stratum_sensitivity(
    results_dir: PathLike,
    *,
    samples_path: PathLike,
    subset_filter: Optional[dict],
    cues: Sequence[str] = HEADLINE_CUES,
    scopes: Sequence[Optional[Set[str]]] = (None, {"EBench-18K"}),
) -> List[dict]:
    """Claim A's headline cues inside human-MOS strata, one scope at a time.

    `scopes` is a list of source restrictions: `None` keeps every breadth source that
    carries a human label, `{"EBench-18K"}` keeps only the source whose MOS is
    continuous. Both are reported because they answer the same question with different
    trade-offs — pooling buys n and costs scale comparability.
    """
    if not subset_filter:
        raise ValueError(
            "refusing to build the MOS stratum table without a subset filter: claim A "
            "is the breadth block, and pooling the anchor block moves every cell"
        )
    from edit_judge_bias.data import io
    from edit_judge_bias.data.schema import SampleRecord
    from edit_judge_bias.metrics.stats import minimum_detectable_effect

    results_dir = Path(results_dir)
    samples = io.read_jsonl(Path(samples_path), SampleRecord)
    base_ids = sample_ids_matching(samples_path, subset_filter)
    by_id = {s.sample_id: s for s in samples}
    wanted = set(cues)

    rows: List[dict] = []
    for scope in scopes:
        label = "all_human_sources" if scope is None else "+".join(sorted(scope))
        strata = within_source_tertiles(samples, base_ids, sources=scope)
        published: Dict[Tuple[str, str], dict] = {}
        for stratum in MOS_STRATA:
            keep = strata[stratum]
            if not keep:
                continue
            dispersion = _shift_dispersion(results_dir, keep)
            sources: Dict[str, int] = {}
            for sid in keep:
                sources[by_id[sid].source_dataset] = sources.get(
                    by_id[sid].source_dataset, 0) + 1
            for cell in build_claim_a(results_dir, keep):
                key = (cell["judge_model"], cell["bias_type"])
                if stratum == "published":
                    published[key] = cell
                if cell["bias_type"] not in wanted:
                    continue
                sd, _ = dispersion.get(key, (None, 0))
                shift = cell["mean_shift"]
                mde_sd = minimum_detectable_effect(cell["n"])
                in_sd = None if not sd else shift / sd
                base = published.get(key, {})
                base_shift = base.get("mean_shift")
                rows.append({
                    "source_scope": label,
                    "stratum": stratum,
                    "judge_model": cell["judge_model"],
                    "bias_type": cell["bias_type"],
                    "n_samples_in_stratum": len(keep),
                    "n_by_source": ";".join(f"{k}:{v}" for k, v in sorted(sources.items())),
                    "n": cell["n"],
                    "score_field": cell["score_field"],
                    "mean_shift": shift,
                    "ci_low": cell["ci_low"],
                    "ci_high": cell["ci_high"],
                    "p_value": cell["p_value"],
                    "q_value": cell["q_value"],
                    "significant_bh": cell["significant_bh"],
                    "inside_placebo_bound": cell.get("inside_placebo_bound"),
                    # --- power, so a null inside a stratum is readable ---
                    "sd_shift": None if sd is None else round(sd, 4),
                    "shift_in_sd_units": None if in_sd is None else round(in_sd, 4),
                    "mde_sd": None if mde_sd is None else round(mde_sd, 4),
                    # ⚠️ `detectable` is about POWER, not about the effect: False means
                    # this stratum could not have found an effect that size even if it
                    # were real, so the cell's q-value carries no information either way.
                    "detectable": (None if in_sd is None or mde_sd is None
                                   else bool(abs(in_sd) >= mde_sd)),
                    # --- trajectory against the published cell ---
                    "published_mean_shift": base_shift,
                    "shift_delta_vs_published": (
                        None if shift is None or base_shift is None
                        else round(shift - base_shift, 4)
                    ),
                    "published_significant_bh": base.get("significant_bh"),
                    "sign_preserved": (
                        None if shift is None or base_shift is None
                        else bool(shift * base_shift > 0)
                    ),
                    "family": cell["family"],
                })
    return rows


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def _summarise_mos(rows: Sequence[dict]) -> None:
    scopes = []
    for r in rows:
        if r["source_scope"] not in scopes:
            scopes.append(r["source_scope"])
    for scope in scopes:
        print(f"=== A1d: claim A inside human-MOS strata — scope {scope} ===")
        for stratum in MOS_STRATA:
            sub = [r for r in rows if r["source_scope"] == scope and r["stratum"] == stratum]
            if not sub:
                continue
            head = sub[0]
            print(f"  --- {stratum}: {head['n_samples_in_stratum']} samples "
                  f"({head['n_by_source']}), judge n={head['n']} ---")
            for cue in HEADLINE_CUES:
                cells = [r for r in sub if r["bias_type"] == cue]
                if not cells:
                    continue
                sig = sum(1 for r in cells if r["significant_bh"])
                same = sum(1 for r in cells if r["sign_preserved"])
                powered = sum(1 for r in cells if r["detectable"])
                shifts = [r["mean_shift"] for r in cells if r["mean_shift"] is not None]
                print(f"    {cue:<19}shift {min(shifts):+.2f}..{max(shifts):+.2f}   "
                      f"BH {sig}/5   sign kept {same}/5   powered {powered}/5")
        print()


def _summarise(rows: Sequence[dict], audit: dict) -> None:
    print("=== A1a: spec violations, measured from the published pixels ===")
    print(f"  candidates (mask-polarity affected)      : {audit['n_candidates']}")
    print(f"  violating (>=1 changed pixel in region)  : {audit['n_violating']}")
    print(f"  canvas-geometry count (original spec)   : {audit['n_violating_canvas_geometry']}"
          f"   -> false alarms: {audit['canvas_only_false_alarms']}")
    print(f"  sticker fraction inside the region       : "
          f"min {audit['frac_inside_min']:.4f}  median {audit['frac_inside_median']:.4f}  "
          f"max {audit['frac_inside_max']:.4f}")
    print(f"  >50% inside: {audit['n_over_50pct_inside']}   >90% inside: {audit['n_over_90pct_inside']}")
    print(f"  reconstruction control                   : "
          f"{audit['reconstruction_control']['ink_bbox_inside_reconstructed_canvas']}")
    print()
    for level in ("published", "drop_spec_viol", "drop_mask_affected"):
        sub = [r for r in rows if r["level"] == level]
        if not sub:
            continue
        print(f"  --- {level}: dropped {sub[0]['n_samples_dropped']}, n={sub[0]['n']} ---")
        for r in sorted(sub, key=lambda r: (r["bias_type"], r["judge_model"])):
            flag = "*" if r["significant_bh"] else "."
            delta = ("" if r["shift_delta_vs_published"] is None
                     else f"  d={r['shift_delta_vs_published']:+.3f}")
            print(f"    {flag} {r['bias_type']:<19}{r['judge_model']:<18}"
                  f"shift={r['mean_shift']:>7}  q={r['q_value']}{delta}")
        if level != "published":
            tested = [r for r in sub if r["lost_significance"] is not None]
            lost = [r for r in tested if r["lost_significance"]]
            print(f"    -> of the {len(tested)} cells that were significant when "
                  f"published, {len(lost)} lost significance"
                  + (f": {[(r['bias_type'], r['judge_model']) for r in lost]}" if lost
                     else " — ALL SURVIVE"))
        print()



# --------------------------------------------------------------------------- #
# WP-F1d — the zero-difference policy as a declared analysis degree of freedom #
# --------------------------------------------------------------------------- #
def build_zero_policy_sensitivity(
    results_dir: PathLike,
    *,
    samples_path: PathLike,
    subset_filter: Optional[dict],
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
    control_bias: str = "sham",
) -> List[dict]:
    """Claim A under both Wilcoxon zero-difference policies, per (judge, cue).

    WHY THIS IS A TABLE AND NOT A SENTENCE. The published p-values use
    `zero_method="wilcox"`, which DISCARDS items whose score did not move; between 20%
    and 84% of items in a cell are such zeros, so the policy is a real analysis
    degree of freedom and the paper says so. It then quoted a sensitivity count that
    lived in no file — and the count was measured before WP-A2 re-judged 273 images.
    A claim about robustness that cannot itself be rebuilt is the weakest sentence in
    a robustness section.

    `pratt` keeps the zeros in the ranking (they cannot contribute signed ranks, but
    they inflate the ranks of everything else), so it is the conservative policy of the
    two. Both p-values get their own BH family of the same shape as `claim_A`, with the
    control held out — a q from one policy has no meaning against the other's.
    """
    from scipy.stats import wilcoxon

    from edit_judge_bias.metrics.scoring_metrics import compute_score_shifts

    results_dir = Path(results_dir)
    keep = sample_ids_matching(Path(samples_path), subset_filter) if subset_filter else None

    def _keep(rows):
        return rows if keep is None else [r for r in rows if r.sample_id in keep]

    rows: List[dict] = []
    for model in _discover_scoring_models(results_dir, roster):
        originals = _keep(_read_results(
            results_dir / "raw_judgments" / f"scoring__{model}.jsonl"))
        biased = _keep(_read_results(
            results_dir / "biased_judgments" / f"scoring__{model}.jsonl"))
        if not biased:
            continue
        for st in compute_score_shifts(originals, biased):
            shifts = list(st.shifts)
            n_zero = sum(1 for x in shifts if x == 0)
            nonzero = [x for x in shifts if x != 0]

            def _p(method: str, data) -> Optional[float]:
                if not data:
                    return None
                try:
                    return float(wilcoxon(data, zero_method=method,
                                          alternative="two-sided").pvalue)
                except ValueError:
                    return None

            rows.append({
                "judge_model": model,
                "bias_type": st.bias_type,
                "n": st.n,
                "n_zero_diff": n_zero,
                "zero_share": round(n_zero / st.n, 4) if st.n else None,
                "mean_shift": round(st.mean_shift, 4),
                # `wilcox` on the non-zero subset is exactly what `stats.wilcoxon_pvalue`
                # does, so this column must reproduce `claim_a.csv`'s p to the digit --
                # that equality is what makes the pratt column comparable to it.
                "p_wilcox": None if (v := _p("wilcox", nonzero)) is None else round(v, 6),
                "p_pratt": None if (v := _p("pratt", shifts)) is None else round(v, 6),
            })

    for key, family in (("p_wilcox", "claim_A"), ("p_pratt", "claim_A_pratt")):
        tested = [r for r in rows if r["bias_type"] != control_bias]
        attach_bh(tested, family=family, p_key=key)
        for r in tested:
            r[f"q_{key.split('_')[1]}"] = r.pop("q_value")
            r[f"significant_{key.split('_')[1]}"] = r.pop("significant_bh")
            r.pop("family", None)
    for r in rows:
        for k in ("q_wilcox", "significant_wilcox", "q_pratt", "significant_pratt"):
            r.setdefault(k, None)
        r["family"] = ("control" if r["bias_type"] == control_bias
                       else "claim_A (wilcox) + claim_A_pratt")
        r["policy_changes_verdict"] = (
            None if r["significant_wilcox"] is None or r["significant_pratt"] is None
            else bool(r["significant_wilcox"] != r["significant_pratt"])
        )
    return sorted(rows, key=lambda r: (r["bias_type"], r["judge_model"]))

def main(argv: list[str] | None = None) -> int:
    root = default_root()
    manifests = root / "data" / "manifests"
    metrics = root / "results" / "v2" / "metrics"
    ap = argparse.ArgumentParser(description="WP-A1a exclusion sensitivity tables.")
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--samples", type=Path, default=manifests / "samples_judge_v2.jsonl")
    ap.add_argument("--subset-filter", action="append", default=None, metavar="KEY=VALUE")
    ap.add_argument("--audit", type=Path, default=metrics / "mask_polarity_audit.json")
    ap.add_argument("--violation-audit", type=Path,
                    default=metrics / "distraction_spec_violation_audit.json")
    ap.add_argument("--biased-manifest", type=Path,
                    default=manifests / "biased_samples_full_v2.jsonl")
    ap.add_argument("--remeasure", action="store_true",
                    help="re-measure the spec violations from the images on disk. "
                         "⚠️ Do NOT pass this after WP-A2 re-injects: it would "
                         "overwrite the frozen pre-fix record with the post-fix zero.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tables", action="append", default=None,
                    choices=["exclusion", "mos", "zero_policy"],
                    help="which sensitivity tables to build (default: all)")
    args = ap.parse_args(argv)
    wanted_tables = set(args.tables or ["exclusion", "mos", "zero_policy"])

    flt: Dict[str, str] = {}
    for item in args.subset_filter or ["subset_block=breadth"]:
        key, _, value = item.partition("=")
        if not _:
            ap.error(f"--subset-filter expects KEY=VALUE, got {item!r}")
        flt[key] = value

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / "metrics"

    if "zero_policy" in wanted_tables:
        zp_rows = build_zero_policy_sensitivity(
            args.results_dir, samples_path=args.samples, subset_filter=flt,
        )
        write_csv(out_dir / "claim_a_zero_policy_sensitivity.csv", zp_rows)
        flips = [r for r in zp_rows if r["policy_changes_verdict"]]
        tested = [r for r in zp_rows if r["policy_changes_verdict"] is not None]
        print(f"=== zero-difference policy: {len(flips)} of {len(tested)} claim A cells "
              f"change verdict under Pratt ===")
        for r in flips:
            print(f"  {r['judge_model']:<18}{r['bias_type']:<18}"
                  f"zeros={r['zero_share']}  wilcox q={r['q_wilcox']} -> "
                  f"pratt q={r['q_pratt']}")
        print(f"wrote -> {out_dir / 'claim_a_zero_policy_sensitivity.csv'} "
              f"({len(zp_rows)} rows)\n")
        if not (wanted_tables - {"zero_policy"}):
            return 0

    if "mos" in wanted_tables:
        mos_rows = build_mos_stratum_sensitivity(
            args.results_dir, samples_path=args.samples, subset_filter=flt,
        )
        write_csv(out_dir / "claim_a_by_mos_stratum.csv", mos_rows)
        _summarise_mos(mos_rows)
        print(f"wrote -> {out_dir / 'claim_a_by_mos_stratum.csv'} ({len(mos_rows)} rows)\n")
        if "exclusion" not in wanted_tables:
            return 0

    if args.remeasure or not args.violation_audit.exists():
        audit = measure_spec_violations(
            audit_path=args.audit,
            biased_manifest=args.biased_manifest,
            samples_manifest=args.samples,
            root=root,
        )
        args.violation_audit.parent.mkdir(parents=True, exist_ok=True)
        args.violation_audit.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[freeze] measured and wrote {args.violation_audit}")
    else:
        audit = json.loads(args.violation_audit.read_text(encoding="utf-8"))

    rows = build_exclusion_sensitivity(
        args.results_dir,
        samples_path=args.samples,
        subset_filter=flt,
        audit_path=args.audit,
        violation_audit=args.violation_audit,
    )
    write_csv(out_dir / "claim_a_sensitivity_exclusion.csv", rows)
    _summarise(rows, audit)
    print(f"wrote -> {out_dir / 'claim_a_sensitivity_exclusion.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
