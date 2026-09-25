"""D-S injection: the deterministic skin-lightness arm. ZERO API CALLS.

Runs `SkinToneITAInjector` over the D-S pool and writes, for every scene:

    dark / light        the two study arms, +-delta around the subject's OWN median ITA
    sham                the same code path at delta = 0        -> false-positive floor
    sham_nonskin        the same dose on the same person's non-skin -> artefact-vs-attribute

x 2 members (the original AND the dataset's edited image), because the pair the judge sees is
(original_variant, edited_variant).

★ THE ORDER OF OPERATIONS IS THE WHOLE POINT
The stopped 2026-07-30 design flipped the attribute FIRST and then re-ran the instruction on
each counterfactual, so generative churn entered twice and correlated with the arm label (the
"woman" render also gained nail polish). Here the edit happens ONCE -- it is the dataset's own
edited image, shared by both arms -- and the attribute transform is applied to that fixed pair.
The only difference between the two arms is the depicted skin lightness.

★ WHAT MAKES THIS RUNNABLE WITHOUT AN AUDITOR
`scene_preserved` is no longer asked of an MLLM (two auditors reached only Cohen's kappa
+0.130 on it). Every injection composites through a mask, so outside that mask the output is
the input's bytes verbatim, and `max_outside_diff` is written on every row as the proof.

    python -m edit_judge_bias.experiments.run_attribute_inject \\
        --config configs/fairness/skin_tone_inject.yaml --dry-run
    python -m edit_judge_bias.experiments.run_attribute_inject \\
        --config configs/fairness/skin_tone_inject.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve, to_rel_posix
from edit_judge_bias.data.schema import SampleRecord
from edit_judge_bias.fairness import attribute_injectors as AI
from edit_judge_bias.fairness import person_region as PR
from edit_judge_bias.fairness.records import AttributeInjectionRecord

#: Members of a pair, in the order the judge will see them.
MEMBERS = (AI.APPLIED_ORIGINAL, AI.APPLIED_EDITED)


def _injection_id(base_sample_id: str, attribute: str, label: str, applied_to: str) -> str:
    """Stable id. `applied_to` is part of it because both members are separate rows."""
    return f"{base_sample_id}__{attribute}__{label}__{applied_to}"


def _load_done(manifest_out: Path) -> set:
    if not manifest_out.exists():
        return set()
    return {r.injection_id for r in io.iter_jsonl(manifest_out, AttributeInjectionRecord)}


def _log_failure(path: Optional[Path], injection_id: str, message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"injection_id": injection_id, "message": message},
                            ensure_ascii=False) + "\n")


def _member_paths(sample: SampleRecord, root: Path) -> Dict[str, Path]:
    return {
        AI.APPLIED_ORIGINAL: resolve(sample.original_image_path, root),
        AI.APPLIED_EDITED: resolve(sample.edited_image_path, root),
    }


def edited_mask_skin_coverage(edited_rgb, person, skin) -> Optional[float]:
    """Share of the APPLIED mask that is still skin on the edited member.

    ★ WHY THIS IS MEASURED RATHER THAN ASSUMED. One scene preparation derives the mask from the
    ORIGINAL and both members are manipulated through it. Across the two ARMS that sharing is
    essential -- it makes segmentation error common-mode, so a clipped ear is clipped identically
    in dark and light and cannot create a difference between them. Across MEMBERS the argument
    does not carry, and OmniEdit's edited images are whole-frame RE-RENDERS at a different
    resolution, not inpaints. If the person moved, the mask recolours skin on the original side
    and background on the edited side, and the judge is shown a pair whose two members were
    manipulated in different places.

    Measured over 60 random pairs: median coverage **0.971**, IoU 0.923, and resized pairs
    (0.941) barely worse than same-size ones (0.984) -- so the shared mask is sound for this
    corpus. But 1 in 60 fell below 0.5, which is exactly the kind of small silent defect this
    project keeps finding after the fact, so it rides on every edited-member row and the
    construction gate can refuse it.

    Returns None when no skin mask can be computed on the edited member; the caller records
    that as a missing measurement rather than as a passing one.
    """
    import numpy as np

    from edit_judge_bias.fairness import person_region as pr

    applied = np.asarray(skin).astype(bool)
    if not applied.any():
        return None
    try:
        edited_skin, _stats = pr.skin_mask(edited_rgb, person, require_person=True)
    except pr.RegionError:
        return None
    return float((applied & np.asarray(edited_skin).astype(bool)).sum() / applied.sum())


def _aligned_pair(sample: SampleRecord, root: Path):
    """Load both members as RGB arrays, resizing the edited one to the original if needed.

    A size mismatch is resized and REPORTED rather than skipped: 8 of the 41 published scenes
    had a 500x500 original against a 1024x1024 edited image, and dropping them would lose real
    data. The injector itself refuses a mismatch, which is what forces the decision to be made
    here, visibly, instead of somewhere downstream by accident.
    """
    from PIL import Image

    paths = _member_paths(sample, root)
    original = PR._as_rgb_array(paths[AI.APPLIED_ORIGINAL])
    height, width = original.shape[:2]
    with Image.open(paths[AI.APPLIED_EDITED]) as im:
        edited_img = im.convert("RGB")
        resized = edited_img.size != (width, height)
        if resized:
            edited_img = edited_img.resize((width, height), Image.BICUBIC)
        import numpy as np

        edited = np.asarray(edited_img)
    return original, edited, resized


def run(
    config_path: str | Path,
    *,
    root: Optional[Path] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict:
    root = Path(root) if root is not None else default_root()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}

    pool_path = cfg["pool"]
    out_dir = Path(cfg["output_dir"])
    manifest_out = root / cfg["manifest_out"]
    seed = int(cfg.get("seed", 42))
    delta_ita = float(cfg.get("delta_ita", 30.0))
    region_cfg = cfg.get("region_config") or None
    if isinstance(region_cfg, str):
        region_cfg = yaml.safe_load((root / region_cfg).read_text(encoding="utf-8"))
    cache_dir = (root / cfg["cache_dir"]) if cfg.get("cache_dir") else None
    failure_log = (root / cfg["failure_log"]) if cfg.get("failure_log") else None
    only_roles = set(cfg.get("only_roles") or ())

    samples = io.read_jsonl(root / pool_path, SampleRecord)
    if limit:
        samples = samples[:limit]

    injector = AI.get_attribute_injector(
        cfg.get("attribute", "skin_tone"),
        delta_ita=delta_ita,
        include_sham=bool(cfg.get("include_sham", True)),
        include_sham_nonskin=bool(cfg.get("include_sham_nonskin", True)),
    )
    arms = [a for a in injector.arms() if not only_roles or a.role in only_roles]

    if dry_run:
        return {
            "pool": pool_path,
            "scenes": len(samples),
            "arms": [a.label for a in arms],
            "members": list(MEMBERS),
            "planned_injections": len(samples) * len(arms) * len(MEMBERS),
            "delta_ita": delta_ita,
            "note": "deterministic; no API calls at any point",
        }

    done = _load_done(manifest_out)
    stats = Counter()
    written: List[AttributeInjectionRecord] = []

    for sample in samples:
        try:
            original, edited, resized = _aligned_pair(sample, root)
        except PR.RegionError as exc:
            stats["scene_unreadable"] += 1
            _log_failure(failure_log, sample.sample_id, f"unreadable: {exc}")
            continue
        if resized:
            stats["scenes_resized"] += 1

        try:
            regions = PR.region_set(resolve(sample.original_image_path, root),
                                    cfg=region_cfg, cache_dir=cache_dir)
        except PR.RegionError as exc:
            stats["scene_region_error"] += 1
            _log_failure(failure_log, sample.sample_id, f"region_error: {exc}")
            continue

        # ★ ONE scene preparation: one mask, one solved dose per arm, shared by both members.
        scene = injector.prepare_scene(original, base_sample_id=sample.sample_id,
                                       regions=regions, cfg=region_cfg)

        # How well that original-derived mask transfers to the re-rendered edited member.
        # Computed once per scene and stamped on the edited rows; see the function's docstring.
        coverage = edited_mask_skin_coverage(edited, regions.person, scene.skin)
        if coverage is not None:
            stats["coverage_measured"] += 1
        else:
            stats["coverage_unmeasurable"] += 1

        for spec in arms:
            for member in MEMBERS:
                injection_id = _injection_id(sample.sample_id, spec.attribute,
                                             spec.label, member)
                if injection_id in done:
                    stats["skipped"] += 1
                    continue
                base = original if member == AI.APPLIED_ORIGINAL else edited
                result = injector.apply(base, scene, spec, applied_to=member)
                if not result.success:
                    stats[f"dropped_{spec.label}"] += 1
                    _log_failure(failure_log, injection_id, result.message)
                    continue

                rel = out_dir / spec.attribute / spec.label / f"{injection_id}.png"
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                from PIL import Image

                Image.fromarray(result.image).save(path)

                measured, params = result.measured, result.params
                record = AttributeInjectionRecord(
                    injection_id=injection_id,
                    base_sample_id=sample.sample_id,
                    attribute=spec.attribute,
                    variant_label=spec.label,
                    role=spec.role,
                    applied_to=member,
                    source_image_path=to_rel_posix(
                        _member_paths(sample, root)[member], root
                    ),
                    injected_image_path=rel.as_posix(),
                    instruction=sample.instruction or "",
                    source_dataset=sample.source_dataset,
                    edit_type=str(getattr(sample.edit_type, "value", sample.edit_type)),
                    edit_model=sample.edit_model,
                    mask_kind=spec.mask_kind,
                    mask_sha256=result.mask_sha256,
                    mask_frac=round(float(measured.get("mask_frac", 0.0)), 5),
                    core_frac=round(float(measured.get("core_frac", 0.0)), 5),
                    person_frac=round(float(params.get("person_frac", 0.0)), 5),
                    face_frac=round(float(params.get("face_frac", 0.0)), 5),
                    person_method=str(params.get("person_method", "")),
                    requested_delta_ita=params.get("requested_delta_ita"),
                    achieved_delta_ita=_finite(measured.get("delta_ita")),
                    ita_before=_finite(measured.get("ita_before")),
                    ita_after=_finite(measured.get("ita_after")),
                    target_bin=str(params.get("bin_after", "")),
                    achieved_bin=str(measured.get("bin_after", "")),
                    clipped_frac=round(float(measured.get("clipped_frac", 0.0)), 5),
                    delta_e00_mean=_round(measured.get("delta_e00_mean")),
                    delta_e00_mass=_round(measured.get("delta_e00_mass")),
                    max_outside_diff=int(measured.get("max_outside_diff", 0)),
                    max_inside_diff=int(measured.get("max_inside_diff", 0)),
                    outside_mask_identical=bool(measured.get("outside_mask_identical", False)),
                    forced_lsb=bool(result.forced_lsb),
                    source_resized=bool(resized and member == AI.APPLIED_EDITED),
                    # Only meaningful on the edited member -- on the original the mask is the
                    # one that was derived from it, so the answer would be 1.0 by construction
                    # and would dilute the gate's view of the number that matters.
                    edited_mask_skin_coverage=(
                        coverage if member == AI.APPLIED_EDITED else None),
                    injector=type(injector).__name__,
                    seed=seed,
                )
                io.append_jsonl(manifest_out, record)
                written.append(record)
                done.add(injection_id)
                stats["written"] += 1

    report = {
        "pool": pool_path,
        "scenes": len(samples),
        "arms": [a.label for a in arms],
        "manifest_out": cfg["manifest_out"],
        **dict(stats),
    }
    if written:
        report["outside_mask_identical_all"] = all(r.outside_mask_identical for r in written)
        report["max_outside_diff_max"] = max(r.max_outside_diff for r in written)
        report["by_role"] = dict(sorted(Counter(r.role for r in written).items()))
        report["by_member"] = dict(sorted(Counter(r.applied_to for r in written).items()))
    return report


def _finite(value) -> Optional[float]:
    """None for NaN, so the manifest never carries a non-JSON float."""
    if value is None:
        return None
    value = float(value)
    return None if value != value else round(value, 4)


def _round(value) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    report = run(a.config, root=Path(a.root) if a.root else None,
                 dry_run=a.dry_run, limit=a.limit)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
