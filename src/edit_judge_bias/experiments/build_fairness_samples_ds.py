"""Turn DETERMINISTIC attribute injections into `SampleRecord`s the scoring runner can judge.

This is the D-S analogue of `build_fairness_samples`, and it is a separate module because the
two arms produce different records for different reasons. The generative arm emits one
`CounterfactualEditRecord` per rendered image and the judged "original" is that render. Here the
manipulation is applied to BOTH members of an already-existing edit, so one judged item is
assembled from TWO injection rows -- `applied_to=original` and `applied_to=edited` -- that share
a scene, an arm, and (by construction) a single mask digest.

★ FOUR ARMS, TWO CONTRAST FAMILIES, AND NO CHANGE TO THE METRIC LAYER
`compute_attribute_gaps` groups on `(judge_model, attribute)` and requires exactly two
`variant_label`s per group, because a gap is a paired difference. The injector produces four
arms, so the arms are mapped onto two families that are each already a pair:

    role=study        dark / light          -> attribute `skin_tone`     (the fairness gap)
    role=sham*        sham / sham_nonskin   -> attribute `dose_control`  (the specificity floor)

Labels sort alphabetically inside the metric, so the reported sign is fixed and can be stated
once instead of being rediscovered per table:

    skin_tone     gap = mean(dark)  - mean(light)          positive => the darker-skinned
                                                           variant of the same scene scored higher
    dose_control  gap = mean(sham)  - mean(sham_nonskin)    positive => moving a matched physical
                                                           dose OFF the skin lowered the score

`dose_control` is what stops the primary result being read as "any local recolour moves the
judge". `sham` carries dose 0 through the skin mask and `sham_nonskin` carries a full-magnitude
dose through an area-matched non-skin region of the SAME person.

⚠️ **So this pair differs in dose AND in region, not in region alone.** It is a valid positive
control — it measures whether a perturbation of this kind is detectable at all, on these scenes
and these judges — and it is NOT a location contrast: a sentence like "recolouring clothing
costs more than recolouring skin" does not follow from it, because the skin side of the
subtraction carries no dose. `build_dose_control_decomposition` splits it into the two
one-variable contrasts (`study - sham`, `study - sham_nonskin`) and measures how far
`sham_nonskin`'s physical dose actually is from the study arm's (median 1.29x per-pixel dE00);
read that table before writing any mechanism sentence. Both families get their own BH family in `build_fairness_tables`
(`fairness_gap:{attribute}`), which is the reason they must be separate `attribute` values rather
than four labels under one.

★ THE CONSTRUCTION GATE IS APPLIED HERE, AND PAIRWISE
`run_construction_gate` admits a `{base}__{attribute}` key only when both study arms exist on
both members, share one `mask_sha256`, and clear locality and dose checks. A scene that failed is
dropped from every family -- including the sham arms, whose own rows might individually be fine.
That is deliberate: the sham floor is only interpretable for the scenes the study arms actually
used, and a floor computed over a different scene set is not that arm's floor.

⚠️ A row is admitted only if `success` and `outside_mask_identical` and `max_outside_diff == 0`.
The gate already asserts this for study rows; the sham arms are not part of its pairwise check,
so it is asserted here rather than assumed.

    python -m edit_judge_bias.experiments.build_fairness_samples_ds --report
    python -m edit_judge_bias.experiments.build_fairness_samples_ds
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.data.schema import ContentCategory, SampleRecord
from edit_judge_bias.experiments.run_construction_gate import passed_pair_keys
from edit_judge_bias.fairness.records import AttributeInjectionRecord

#: `role` -> the `attribute` the paired metric groups on. See the module docstring for why the
#: four arms become two families rather than one four-label group.
ROLE_TO_FAMILY = {
    "study": "skin_tone",
    "sham": "dose_control",
    "sham_nonskin": "dose_control",
}

#: Members of one judged item, in the order the judge sees them.
APPLIED_ORIGINAL = "original"
APPLIED_EDITED = "edited"


def _row_ok(r: AttributeInjectionRecord) -> Optional[str]:
    """Why this injection row cannot be judged, or None if it can."""
    if not r.success:
        return "injection_failed"
    if not r.outside_mask_identical or r.max_outside_diff != 0:
        # Locality is the whole claim. A row that leaked outside its mask is not a weaker
        # version of the manipulation, it is a different manipulation.
        return "not_byte_local"
    return None


def build(
    *,
    root: Optional[Path] = None,
    pool: str = "data/manifests/samples_fairness_ds_v4.jsonl",
    injections: str = "data/manifests/attribute_injections_v4.jsonl",
    gate_json: Optional[str] = "results/v2_fairness/metrics/construction_gate_v4.json",
    out_path: str = "data/manifests/samples_fairness_ds_judge_v4.jsonl",
    write: bool = True,
) -> dict:
    root = Path(root) if root is not None else default_root()
    base_by_id = {s.sample_id: s for s in io.read_jsonl(root / pool, SampleRecord)}

    inj_path = root / injections
    if not inj_path.exists():
        # A legitimate "injection has not run yet" state, not a bug.
        return {
            "injections_in": 0,
            "samples_out": 0,
            "error": f"injection manifest not found: {injections}. "
                     "Run experiments.run_attribute_inject first.",
            "out_path": None,
        }
    rows = io.read_jsonl(inj_path, AttributeInjectionRecord)

    gate: Optional[set] = None
    if gate_json and (root / gate_json).exists():
        gate = passed_pair_keys(root / gate_json)

    # (base, injection-attribute, variant_label) -> {applied_to: row}
    arms: Dict[tuple, Dict[str, AttributeInjectionRecord]] = collections.defaultdict(dict)
    roles: Dict[tuple, str] = {}
    dropped = collections.Counter()
    for r in rows:
        reason = _row_ok(r)
        if reason:
            dropped[reason] += 1
            continue
        if r.role not in ROLE_TO_FAMILY:
            dropped[f"unknown_role({r.role})"] += 1
            continue
        key = (r.base_sample_id, r.attribute, r.variant_label)
        if r.applied_to in arms[key]:
            # Two rows for one (scene, arm, member) means the manifest was appended to twice
            # without the resume filter. Last-write-wins would silently pick one; say so.
            dropped["duplicate_member_row"] += 1
            continue
        arms[key][r.applied_to] = r
        roles[key] = r.role

    # Group the complete arms into contrast families, so pair completeness is checked before
    # anything is emitted.
    families: Dict[tuple, Dict[str, tuple]] = collections.defaultdict(dict)
    for key, members in arms.items():
        base, attr, label = key
        missing = {APPLIED_ORIGINAL, APPLIED_EDITED} - set(members)
        if missing:
            dropped[f"incomplete_members({sorted(missing)[0]}_absent)"] += len(members)
            continue
        families[(base, attr, ROLE_TO_FAMILY[roles[key]])][label] = (
            members[APPLIED_ORIGINAL], members[APPLIED_EDITED],
        )

    records: List[SampleRecord] = []
    for (base, attr, family), by_label in sorted(families.items()):
        if base not in base_by_id:
            dropped["base_sample_missing_from_pool"] += len(by_label)
            continue
        # ⚠️ The gate key is the INJECTION's attribute (`skin_tone`), not the contrast family:
        # one gate verdict covers the whole scene, and the sham arms inherit it on purpose.
        if gate is not None and f"{base}__{attr}" not in gate:
            dropped["failed_construction_gate"] += len(by_label)
            continue
        if len(by_label) != 2:
            # A one-armed family cannot produce a paired difference. Dropped here, where it is
            # counted, rather than in the metric, where it would vanish silently.
            dropped[f"incomplete_pair:{family}({sorted(by_label)})"] += len(by_label)
            continue
        base_rec = base_by_id[base]
        for label, (orig_row, edit_row) in sorted(by_label.items()):
            rec = SampleRecord(
                sample_id=f"{base}__{attr}__{label}",
                source_dataset=base_rec.source_dataset,
                edit_type=base_rec.edit_type,
                # The detector measured a person here; the corpus's own `content_category`
                # label is unreliable on OmniEdit ("global" for a frame with person_frac 0.41).
                content_category=ContentCategory.HUMAN,
                original_image_path=orig_row.injected_image_path,
                instruction=orig_row.instruction,
                edit_model=orig_row.edit_model,
                edited_image_path=edit_row.injected_image_path,
            )
            rec.metadata.subset_block = "fairness_ds"
            rec.metadata.base_sample_id = base
            rec.metadata.attribute = family
            rec.metadata.variant_label = label
            # Deterministic construction: there is no editor to re-run, so there is no
            # editor-noise repeat. Index 1 everywhere keeps every row inside the gap and
            # leaves `compute_editor_noise` correctly empty rather than fabricating a floor.
            rec.metadata.render_index = 1
            rec.metadata.arm_role = roles[(base, attr, label)]
            rec.metadata.is_null_control = family != "skin_tone"
            rec.metadata.mask_sha256 = orig_row.mask_sha256
            rec.metadata.mask_kind = orig_row.mask_kind
            rec.metadata.achieved_delta_ita = orig_row.achieved_delta_ita
            rec.metadata.delta_e00_mean = orig_row.delta_e00_mean
            records.append(rec)

    by_family = collections.Counter(r.metadata.attribute for r in records)
    report = {
        "injections_in": len(rows),
        "samples_out": len(records),
        "gate_applied": gate is not None,
        "gate_passed_pairs": (len(gate) if gate is not None else None),
        "gate_json": gate_json if gate is not None else None,
        # Counted in JUDGED ITEMS (scene x arm), not scenes and not families, because that is
        # the unit that costs money and the unit `samples_out` is in. A drop of 4 here is one
        # scene's four arms, and reading it as four scenes would overstate the loss 4x.
        "dropped_unit": "judged items (scene x arm)",
        "dropped": dict(sorted(dropped.items())),
        "by_family": dict(sorted(by_family.items())),
        "scenes_by_family": {
            fam: len({r.metadata.base_sample_id for r in records
                      if r.metadata.attribute == fam})
            for fam in sorted(by_family)
        },
        "by_label": dict(
            sorted(collections.Counter(r.metadata.variant_label for r in records).items())
        ),
        "out_path": out_path if write else None,
    }
    # The number that decides whether the arm is worth paying for, printed where the decision
    # is made rather than left to a later table.
    from edit_judge_bias.metrics.fairness_metrics import (
        min_attainable_pvalue, minimum_detectable_effect,
    )
    n_study = report["scenes_by_family"].get("skin_tone", 0)
    report["mde_sd"] = round(minimum_detectable_effect(n_study), 4) if n_study else None
    report["min_attainable_p"] = min_attainable_pvalue(n_study)
    report["panel_passes"] = len(records)
    report["est_usd_at_panel_pass_0_03676"] = round(len(records) * 0.03676, 2)

    # ★★ THE TWO FAMILIES DO NOT COVER THE SAME SCENES, AND THE DIFFERENCE IS NOT RANDOM.
    # `sham_nonskin` needs an area-matched non-skin region on the same person, i.e. a component
    # of `person AND NOT skin` reaching 0.8x the skin area. That is arithmetically impossible
    # once skin exceeds 1/1.8 = 55.6% of the person, and fragmentation costs more on top.
    # Measured on the 20-scene pilot: 7 fail the area bound, 3 fail on fragmentation, 10 pass --
    # and the two groups differ ONLY in skin share (skin/person 0.656 dropped vs 0.304 kept;
    # `person_frac` 0.346 vs 0.345, i.e. not "small subject").
    #
    # So `dose_control` exists only on the systematically MORE CLOTHED half. If the artefact
    # penalty scales with recoloured area -- the whole premise of a dose control -- then a
    # control measured on smaller skin regions UNDERSTATES the penalty on the full sample, and
    # subtracting it inflates the residual one would attribute to skin tone. That is the error
    # in the unsafe direction, and it is the same argument the injector's own docstring uses to
    # reject a background control.
    #
    # → The controlled contrast must be read on `common_subset_scenes`, where both families
    # exist on the SAME scenes. `skin_tone` over all its scenes stays reportable as the
    # secondary, UNCONTROLLED number -- clearly labelled as such, never as the headline.
    scenes_of = {
        fam: {r.metadata.base_sample_id for r in records if r.metadata.attribute == fam}
        for fam in by_family
    }
    common = set.intersection(*scenes_of.values()) if len(scenes_of) > 1 else set()
    report["mde_by_family"] = {
        fam: round(minimum_detectable_effect(len(s)), 4) if s else None
        for fam, s in sorted(scenes_of.items())
    }
    report["common_subset_scenes"] = len(common)
    report["common_subset_mde_sd"] = (
        round(minimum_detectable_effect(len(common)), 4) if common else None
    )
    report["common_subset_note"] = (
        "the controlled contrast (skin_tone minus dose_control) is only interpretable here; "
        "dose_control is missing exactly on the high-skin-share scenes, so the two families "
        "otherwise describe different populations"
    )

    if write:
        io.write_jsonl(root / out_path, records)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--pool", default="data/manifests/samples_fairness_ds_v4.jsonl")
    ap.add_argument("--injections", default="data/manifests/attribute_injections_v4.jsonl")
    ap.add_argument("--gate-json",
                    default="results/v2_fairness/metrics/construction_gate_v4.json")
    ap.add_argument("--no-gate", action="store_true",
                    help="smoke tests only — study numbers must be gated")
    ap.add_argument("--out", default="data/manifests/samples_fairness_ds_judge_v4.jsonl")
    ap.add_argument("--report", action="store_true", help="print counts, write nothing")
    a = ap.parse_args(argv)
    rep = build(
        root=Path(a.root) if a.root else None,
        pool=a.pool,
        injections=a.injections,
        gate_json=(None if a.no_gate else a.gate_json),
        out_path=a.out,
        write=not a.report,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
