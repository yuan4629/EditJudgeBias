"""Build the D-class fairness sample pool: one record per CONTENT-distinct human scene.

Why this exists as its own builder instead of a filter inside the runner
---------------------------------------------------------------------------
`run_attribute_edit` dedupes on `original_image_path`, which is a *string*. That is not
enough here, and the gap is large enough to change the study's power. Measured
2026-07-30 over `samples_full_v2.jsonl`:

    human samples                       2,046
    - rejected: attribute words         1,378   (woman 502 / man 421 / hair 176 / beard 85 / ...)
    - rejected: minor words               169   (ethics hard-stop)
    = eligible samples                    502
    -> distinct `original_image_path`      70
    -> distinct CONTENT (ORB+RANSAC)       41   <-- the real pool

The 70 -> 41 collapse is not noise. GenAI-Bench ships hash-named copies of
ImagenHub/MagicBrush base images, and ImagenHub itself stores one photo under several
paths (one per instruction). ImagenHub goes 24 -> 2 representatives. Of the 50
duplicate pairs found, 22 exceed 477 RANSAC inliers, i.e. they are the same photo
pixel for pixel. The collapse is threshold-robust: 41 scenes anywhere in
[30, 96] inliers (the observed gap is 96 -> 43), and still 47 at a punitive >=400.

Judging a pseudo-replicate twice would inflate n without adding information — the
fairness gap is a *paired within-scene* quantity, so duplicated scenes are duplicated
pairs, and every CI and Wilcoxon p would be optimistic.

    python -m edit_judge_bias.experiments.build_fairness_pool            # writes the manifest
    python -m edit_judge_bias.experiments.build_fairness_pool --report   # counts only, no write

Every collapsed member is recorded in `metadata.duplicate_members` so the write-up can
state the collapse instead of quietly reporting 41 as if it were the eligible count.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve
from edit_judge_bias.data.overlap import orb_descriptors, ransac_inliers
from edit_judge_bias.data.schema import SampleRecord
from edit_judge_bias.experiments.run_attribute_edit import _eligible, instruction_ok

# Validated on this pool: true duplicates land at 96..500 inliers, the next value down is
# 43, and an ORB signature matched against itself saturates at 440..500. Anything in the
# gap gives the same 41 components, so the exact cut is not load-bearing.
DUPLICATE_INLIERS = 60

#: An ORB signature matched against ITSELF saturates at 440..500 inliers. If a self-match
#: comes back below this, the instrument is broken and every "0 duplicates" it reports is a
#: lie. `validate_orb_instrument` RAISES on that -- this is the ORB lesson turned into code:
#: a first attempt at the 70 -> 41 collapse silently reported "0 duplicates" because it
#: called `ransac_inliers(path, path)` (the function takes signatures) and the TypeError was
#: swallowed by an `except Exception: continue`.
#: A self-match must recover at least this FRACTION of the signature's own keypoints.
#: Relative, not absolute: `orb_descriptors` caps at 500 features so a real photo self-matches
#: at 440..500, but a low-texture frame may hold only 37 keypoints in total and could never
#: clear an absolute floor however well the matcher works. Measured on this pool: real photos
#: self-match at 88..100%.
ORB_SELF_MATCH_FRAC = 0.8
ORB_MIN_KEYPOINTS = 8

#: Detector screen defaults. `face_frac` is measured against the whole frame.
#:
#: ★ Measured 2026-07-30 over 1,336 distinct originals: face_frac > 0.01 -> 216 originals,
#: > 0.02 -> 139, > 0.04 -> 92, > 0.08 -> 58. The published 41-scene pool contained only
#: **14** originals with ANY detectable face, which is why it was full of gulls, squirrels,
#: zebras and dolls. 0.02 of a 512^2 frame is roughly a 37x37 px face.
DEFAULT_MIN_FACE_FRAC = 0.02
DEFAULT_MIN_PERSON_FRAC = 0.05

#: ★ THE SCREEN IS ARM-AWARE, and getting this wrong cost a whole pool once.
#:
#: A first detector pass gated everything on a Haar face box and returned 16 scenes. That was
#: a design error, not a pool ceiling: the deterministic skin-lightness arm recolours SKIN and
#: does not need a detectable face at all. Measured on the 1,156 candidates, 38 originals carry
#: a clearly segmented person (person_frac up to 0.997, skin_frac up to 0.698) for which Haar
#: finds ZERO faces even with a profile cascade added -- backs turned, occluded, or simply
#: 512px-blurry. Those are excellent skin-recolor subjects and terrible head-composite ones.
#:
#: So: `require_face=True` (the generative D-G arm, which composites a head and therefore
#: needs one) gates on `min_face_frac`; `require_face=False` (the deterministic D-S arm) gates
#: on `min_skin_frac`. Measured funnels at person_frac >= 0.05:
#:     D-S  skin >= 0.02 -> 95   >= 0.04 -> 75   >= 0.06 -> 51   >= 0.10 -> 27
#:     D-G  face >= 0.01 -> 24   >= 0.02 -> 19   >= 0.04 -> 14
DEFAULT_MIN_SKIN_FRAC = 0.04

#: ★★ THE USABLE FRAMING BAND -- an UPPER bound on `face_frac`, and it is as load-bearing as
#: the lower one. Measured 2026-07-31 across corpora:
#:
#:     tight face crop (FairFace / CelebA)  face_frac ~0.35  -> no scene left to edit
#:     head-and-shoulders                   face_frac ~0.12  -> USABLE
#:     half-body                            face_frac ~0.04  -> USABLE
#:     full-body / environmental            face_frac ~0.012 -> manipulation imperceptible
#:     group / distant                      face_frac ~0.003 -> not a single subject
#:
#: A lower bound alone admits portrait crops on which the editing instruction has nothing to
#: act; an upper bound alone admits crowds. `None` disables the check, which is the default so
#: the published v3 pools reproduce unchanged.
DEFAULT_MAX_FACE_FRAC = None

#: ★★ CONSTRUCT-VALIDITY SCREENS, added 2026-07-30 after a HUMAN LOOK at the staged sheets.
#:
#: The deterministic injection passed its arithmetic gate on 31 of 31 scenes -- byte-exact
#: locality, matched dose, one shared mask digest -- and the manipulation was still not a
#: single-subject skin-tone counterfactual on most of them. Two failure modes, both invisible
#: to any numeric check that does not look at WHAT the mask covers:
#:
#:   * `mb_44437_2` is a stadium CROWD. `person_frac` 0.691 is honest (the stands really are
#:     full of people) and `skin_frac` 0.246 then recolours dozens of background spectators plus
#:     the tan uniforms. "The depicted person's skin tone" is not even well defined there, and a
#:     quarter-frame tonal change is closer to `aesthetic_filter` -- which this study measured
#:     as NULL on 4/5 judges -- than to a local attribute edit.
#:   * `ebench_H_02_08` is a pair of hands on a wooden guitar. DeepLab segments guitar+hands as
#:     one "person", wood passes the skin-chroma rule, and the guitar body gets recoloured with
#:     the hands.
#:
#: Measured over the 31-scene D-S pool: 21 have NO face box (no single-subject anchor), 15 have
#: a skin mask over 8% of frame, 3 are crowds -- and exactly **1** scene is clean on all three.
#: n=1 is an MDE of 2.79 SD, i.e. not a study. These screens exist so that finding is ENFORCED
#: rather than remembered.
DEFAULT_MAX_SKIN_FRAC = 0.08
DEFAULT_MAX_PERSON_COMPONENTS = 2
DEFAULT_MIN_COMPONENT_FRAC = 0.01

#: Reported alongside the chosen cut so the n-versus-perceptibility trade-off is explicit
#: rather than buried in one config value.
REPORTED_FACE_TIERS = (0.01, 0.02, 0.04, 0.08)
REPORTED_SKIN_TIERS = (0.02, 0.04, 0.06, 0.10, 0.15)

#: Face-box IoU between the original and a candidate edited image, above which that editor's
#: output counts as GEOMETRICALLY ALIGNED with the original.
#:
#: ★ This screen is only needed by the generative arm (D-G), which composites one generated
#: patch into both members and therefore needs the head in the same place in both. The
#: deterministic arm (D-S) recolours each member through its own mask and needs no alignment
#: at all. Measured: 276 of 609 (original, edited) pairs clear IoU >= 0.5, covering 50 of 61
#: candidate originals -- versus only 16 of 61 if byte-level quietness in the head region is
#: demanded instead, because these corpora's "edited" images are re-renders, not inpaints.
DEFAULT_ALIGN_IOU = 0.5

#: Cap on how many of an original's editors get the alignment check. EBench ships 17 per
#: original and the D-G arm needs only ONE aligned editor, so an exhaustive scan would pay
#: 17 segmentations per scene for no extra information. `n_align_checked` rides on every row
#: next to `n_aligned_editors`, so a capped scan is never mistaken for an exhaustive one --
#: the "no silent caps" rule.
DEFAULT_ALIGN_MAX_CHECKS = 6


def eligible_originals(samples: Sequence[SampleRecord], root: Path, *,
                       require_human_category: bool = True) -> List[SampleRecord]:
    """Eligible samples, one per distinct `original_image_path` that exists on disk.

    `require_human_category=True` reproduces the published 41-scene pool exactly. Set it
    False for the detector-first pool: the label is measurably unreliable here (52 of the
    139 originals with a face >2% of frame were tagged global/object/scenery), and the
    detector screen is a better test of the same thing. The instruction screens always apply.
    """
    seen: set[str] = set()
    out: List[SampleRecord] = []
    for s in samples:
        keep = _eligible(s) if require_human_category else instruction_ok(s.instruction)
        if not keep:
            continue
        key = str(s.original_image_path)
        if key in seen:
            continue
        if not resolve(s.original_image_path, root).exists():
            continue
        seen.add(key)
        out.append(s)
    return out


@dataclass
class PoolScreen:
    """One candidate original and every measurement the screens made on it."""

    sample_id: str
    source_dataset: str
    original_image_path: str
    content_category: str
    instruction: str
    edit_type: str
    person_frac: float = 0.0
    face_frac: float = 0.0
    person_method: str = ""
    skin_frac: float = 0.0
    n_person_components: int = 0
    n_aligned_editors: int = 0
    n_align_checked: int = 0
    best_align_iou: float = 0.0
    aligned_sample_ids: List[str] = field(default_factory=list)
    passed_detector: bool = False
    passed_alignment: bool = False
    reason: str = ""

    def as_row(self) -> dict:
        row = dict(self.__dict__)
        row["aligned_sample_ids"] = list(self.aligned_sample_ids)
        return row


def validate_orb_instrument(signatures: Sequence, *,
                            min_self_match_frac: float = ORB_SELF_MATCH_FRAC,
                            min_keypoints: int = ORB_MIN_KEYPOINTS) -> dict:
    """Prove the ORB matcher works by matching a signature against ITSELF. Raises if not.

    ★ This exists because the instrument has already failed silently once. A first attempt
    at the duplicate collapse reported "0 duplicates" -- a clean, plausible, completely wrong
    answer -- because it passed file paths where signatures were expected and the resulting
    TypeError was swallowed by `except Exception: continue`.

    The floor is a FRACTION of the signature's own keypoints, not an absolute inlier count.
    An absolute floor silently assumes image richness: `orb_descriptors` caps at 500 features,
    so a real photo self-matches at 440..500, but a low-texture frame may only have 37
    keypoints in total and could never reach an absolute 400 no matter how well the matcher
    works. The property actually being checked is "matching a thing against itself finds
    (nearly) all of it", which is scale-free.
    """
    usable = [s for s in signatures if s is not None]
    if not usable:
        raise RuntimeError(
            "ORB produced no usable signatures at all; cannot verify the duplicate "
            "collapse. Check that opencv is installed and the image paths resolve."
        )
    # Validate against the richest signature available: a keypoint-poor image makes a weak
    # control, and picking the best one keeps the check as sharp as the data allows.
    richest = max(usable, key=lambda sig: len(sig[0]))
    n_keypoints = len(richest[0])
    measured = int(ransac_inliers(richest, richest))
    required = max(min_keypoints, int(round(min_self_match_frac * n_keypoints)))
    if measured < required:
        raise RuntimeError(
            f"ORB self-match recovered {measured} of {n_keypoints} keypoints "
            f"({measured / max(1, n_keypoints):.1%}), below the "
            f"{min_self_match_frac:.0%} floor. The matcher is not matching, so every "
            "'no duplicate' verdict it gives is void. Fix the instrument before trusting "
            "any pool count."
        )
    return {"self_match_inliers": measured, "self_match_keypoints": n_keypoints,
            "self_match_frac": round(measured / max(1, n_keypoints), 4),
            "usable_signatures": len(usable),
            "unreadable_signatures": len(signatures) - len(usable)}


def _person_components(person, *, min_component_frac: float) -> int:
    """How many separate people-sized blobs the person mask holds.

    A crowd scene reports many. `person_frac` alone cannot distinguish "one large subject" from
    "a full grandstand", and the difference decides whether "the depicted person's skin tone" is
    a well-defined quantity at all.
    """
    import numpy as np

    if person is None:
        # No mask to measure. Zero, not "unknown": a caller with no person mask has already
        # failed the person screen, and guessing a component count here would let a stubbed or
        # degraded region set slip past the crowd check.
        return 0
    cv2 = __import__("edit_judge_bias.fairness.person_region", fromlist=["_cv2"])._cv2()
    mask = np.asarray(person).astype(np.uint8)
    if not mask.any():
        return 0
    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    area = float(mask.size)
    return sum(1 for i in range(1, count)
               if stats[i, cv2.CC_STAT_AREA] / area > min_component_frac)


def _face_box_iou(a, b) -> float:
    ax, ay, ar, ab = a
    bx, by, br, bb = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ar, br), min(ab, bb)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (ar - ax) * (ab - ay)
    area_b = (br - bx) * (bb - by)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def screen_originals(
    candidates: Sequence[SampleRecord],
    samples_by_original: Dict[str, List[SampleRecord]],
    root: Path,
    *,
    region_cfg: Optional[dict] = None,
    cache_dir: Optional[Path] = None,
    min_face_frac: float = DEFAULT_MIN_FACE_FRAC,
    max_face_frac: Optional[float] = DEFAULT_MAX_FACE_FRAC,
    min_person_frac: float = DEFAULT_MIN_PERSON_FRAC,
    min_skin_frac: float = DEFAULT_MIN_SKIN_FRAC,
    max_skin_frac: float = DEFAULT_MAX_SKIN_FRAC,
    max_person_components: int = DEFAULT_MAX_PERSON_COMPONENTS,
    require_face: bool = True,
    require_skin: Optional[bool] = None,
    align_iou: float = DEFAULT_ALIGN_IOU,
    check_alignment: bool = True,
    align_max_checks: int = DEFAULT_ALIGN_MAX_CHECKS,
) -> List[PoolScreen]:
    """Measure every candidate: is there a legible person, and which editors keep it in place.

    The alignment check is deliberately **attribute-blind** -- it runs on the unmodified
    original and edited images, before any manipulation exists -- so which editor a scene
    contributes cannot depend on the arm.

    ★ `require_face` and `require_skin` are SEPARATE because the two D arms need different
    things, and conflating them has already cost one pool. D-G composites a head, so it needs
    a face box and does not care about skin area. D-S recolours skin, so it needs skin. A
    deterministic arm run over a *portrait* corpus needs BOTH -- the face box to place the
    subject in the framing band, the skin to have something to recolour -- which is a
    combination the old `require_skin = not require_face` coupling could not express.
    `require_skin=None` keeps that historical coupling, so both published v3 pools reproduce.
    """
    from edit_judge_bias.fairness import person_region as pr

    if require_skin is None:
        require_skin = not require_face

    screens: List[PoolScreen] = []
    for sample in candidates:
        rel = str(sample.original_image_path)
        screen = PoolScreen(
            sample_id=sample.sample_id,
            source_dataset=sample.source_dataset,
            original_image_path=rel,
            content_category=sample.content_category.value,
            instruction=sample.instruction or "",
            edit_type=getattr(sample.edit_type, "value", str(sample.edit_type)),
        )
        try:
            regions = pr.region_set(resolve(sample.original_image_path, root),
                                    cfg=region_cfg, cache_dir=cache_dir)
        except pr.RegionError as exc:
            # A measurement failure is not evidence of anything; record and move on.
            screen.reason = f"region_error: {exc}"
            screens.append(screen)
            continue

        screen.n_person_components = _person_components(
            regions.person, min_component_frac=DEFAULT_MIN_COMPONENT_FRAC
        )
        screen.person_frac = round(regions.person_frac, 5)
        screen.face_frac = round(regions.face_frac, 5)
        screen.person_method = regions.person_method
        screen.skin_frac = round(regions.skin_frac, 5)

        if regions.person_frac < min_person_frac:
            screen.reason = f"person_frac {regions.person_frac:.4f} < {min_person_frac}"
        elif require_face and regions.face_box is None:
            screen.reason = "no_face_box_on_person"
        elif require_face and regions.face_frac < min_face_frac:
            screen.reason = f"face_frac {regions.face_frac:.4f} < {min_face_frac}"
        elif max_face_frac is not None and regions.face_frac > max_face_frac:
            # Above the band: a portrait crop, with no scene left for the instruction to act on.
            screen.reason = f"face_frac {regions.face_frac:.4f} > {max_face_frac}"
        elif require_skin and regions.skin_frac < min_skin_frac:
            screen.reason = f"skin_frac {regions.skin_frac:.4f} < {min_skin_frac}"
        elif regions.skin_frac > max_skin_frac:
            # An over-inclusive mask means the "skin" region has swallowed wood, tan clothing
            # or a crowd -- see DEFAULT_MAX_SKIN_FRAC.
            screen.reason = f"skin_frac {regions.skin_frac:.4f} > {max_skin_frac}"
        elif screen.n_person_components > max_person_components:
            screen.reason = f"person_components {screen.n_person_components} > {max_person_components}"
        elif regions.face_box is None and not require_face:
            # D-S does not need a face to APPLY the recolor, but without one there is no
            # single-subject anchor and no way to say whose skin tone was manipulated.
            screen.reason = "no_face_anchor_for_single_subject"
        else:
            screen.passed_detector = True

        if screen.passed_detector and check_alignment:
            # EBench ships 17 editors per original, so cap the checks: one aligned editor is
            # all the D-G arm needs, and the count is reported next to how many were tried so
            # a capped scan can never be mistaken for an exhaustive one.
            siblings = sorted(samples_by_original.get(rel, []), key=lambda s: s.sample_id)
            screen.n_align_checked = min(len(siblings), align_max_checks)
            for sibling in siblings[:align_max_checks]:
                edited = resolve(sibling.edited_image_path, root)
                if not edited.exists():
                    continue
                try:
                    edited_regions = pr.region_set(edited, cfg=region_cfg, cache_dir=cache_dir)
                except pr.RegionError:
                    continue
                if edited_regions.face_box is None:
                    continue
                iou = _face_box_iou(regions.face_box, edited_regions.face_box)
                screen.best_align_iou = max(screen.best_align_iou, round(iou, 4))
                if iou >= align_iou:
                    screen.n_aligned_editors += 1
                    screen.aligned_sample_ids.append(sibling.sample_id)
            screen.passed_alignment = screen.n_aligned_editors > 0
            if not screen.passed_alignment and not screen.reason:
                screen.reason = f"no_editor_aligned (best IoU {screen.best_align_iou:.2f})"
        screens.append(screen)
    return screens


def collapse_duplicates(
    picked: Sequence[SampleRecord],
    root: Path,
    *,
    threshold: int = DUPLICATE_INLIERS,
    screen_degenerate: bool = False,
) -> List[List[SampleRecord]]:
    """Group `picked` into content-distinct components via ORB+RANSAC.

    Samples whose image cannot produce an ORB signature are kept as singletons rather
    than dropped — an unreadable signature is a measurement failure, not evidence of
    distinctness, and silently dropping would shrink the pool for the wrong reason.

    ★ `screen_degenerate` rejects a homography that is "verified" but meaningless (see
    `overlap.DEGENERATE_MIN_HULL_FRAC`). It matters here in a direction that is easy to miss:
    a spurious match MERGES two distinct scenes, so the pool silently loses n — the same
    direction as a real duplicate, which is why it cannot be spotted from the count alone.
    Off by default so the published 41-scene pool reproduces byte-for-byte.
    """
    sigs = [orb_descriptors(resolve(s.original_image_path, root)) for s in picked]
    if picked:
        # ★ Prove the matcher works before believing any verdict it gives. See
        # `validate_orb_instrument` for the incident this prevents.
        collapse_duplicates.last_instrument_check = validate_orb_instrument(sigs)
    parent = list(range(len(picked)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in itertools.combinations(range(len(picked)), 2):
        if sigs[i] is None or sigs[j] is None:
            continue
        if ransac_inliers(sigs[i], sigs[j],
                          screen_degenerate=screen_degenerate) >= threshold:
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b

    groups: Dict[int, List[SampleRecord]] = collections.defaultdict(list)
    for i in range(len(picked)):
        groups[find(i)].append(picked[i])
    # Deterministic order so a rebuild is byte-identical.
    return sorted(
        (sorted(g, key=lambda s: (s.source_dataset, s.sample_id)) for g in groups.values()),
        key=lambda g: (g[0].source_dataset, g[0].sample_id),
    )


def build(
    *,
    root: Optional[Path] = None,
    samples_path: str = "data/manifests/samples_full_v2.jsonl",
    samples_paths: Optional[Sequence[str]] = None,
    out_path: str = "data/manifests/samples_fairness_v2.jsonl",
    threshold: int = DUPLICATE_INLIERS,
    orb_screen_degenerate: bool = False,
    write: bool = True,
    # --- detector-first options; all off by default so the published pool reproduces ---
    detector: bool = False,
    require_human_category: bool = True,
    min_face_frac: float = DEFAULT_MIN_FACE_FRAC,
    max_face_frac: Optional[float] = DEFAULT_MAX_FACE_FRAC,
    min_person_frac: float = DEFAULT_MIN_PERSON_FRAC,
    min_skin_frac: float = DEFAULT_MIN_SKIN_FRAC,
    max_skin_frac: float = DEFAULT_MAX_SKIN_FRAC,
    max_person_components: int = DEFAULT_MAX_PERSON_COMPONENTS,
    require_face: bool = True,
    require_skin: Optional[bool] = None,
    align_iou: float = DEFAULT_ALIGN_IOU,
    check_alignment: bool = True,
    align_max_checks: int = DEFAULT_ALIGN_MAX_CHECKS,
    exclude_edit_types: Sequence[str] = (),
    exclude_tasks: Sequence[str] = (),
    exclude_id_substrings: Sequence[str] = (),
    include_ids: Optional[str] = None,
    k: Optional[int] = None,
    seed: int = 42,
    region_config: Optional[dict] = None,
    cache_dir: Optional[str] = None,
    screens_out: Optional[str] = None,
) -> dict:
    """Build the pool.

    Two modes, and the default is the OLD one so the published 41-scene manifest can still be
    reproduced byte-for-byte:

    * `detector=False` -- the original screen (`content_category == "human"` + instruction
      words) then ORB collapse.
    * `detector=True` -- the instruction screen, then a *measured* person/face screen, then
      optionally an attribute-blind geometric-alignment screen, then ORB collapse.
    """
    root = Path(root) if root is not None else default_root()
    paths = list(samples_paths) if samples_paths else [samples_path]

    samples: List[SampleRecord] = []
    seen_ids: set[str] = set()
    for rel in paths:
        for s in io.read_jsonl(root / rel, SampleRecord):
            # Several manifests overlap (samples_full_v2 already contains an EBench prefix),
            # so dedupe by sample_id before any counting.
            if s.sample_id in seen_ids:
                continue
            seen_ids.add(s.sample_id)
            samples.append(s)

    if exclude_edit_types:
        drop = {str(x) for x in exclude_edit_types}
        samples = [s for s in samples
                   if str(getattr(s.edit_type, "value", s.edit_type)) not in drop]

    human = [s for s in samples if s.content_category.value == "human"]
    picked = eligible_originals(samples, root,
                               require_human_category=require_human_category)

    report: dict = {
        "samples_manifests": paths,
        "total_samples": len(samples),
        "human_samples": len(human),
        "eligible_samples": sum(
            1 for s in samples
            if (_eligible(s) if require_human_category else instruction_ok(s.instruction))
        ),
        "path_distinct_originals": len(picked),
        "excluded_edit_types": list(exclude_edit_types),
        "require_human_category": require_human_category,
        "detector": detector,
    }

    # Resolved once so the report states what was actually applied rather than what was asked
    # for -- `require_skin=None` means "inherit the historical coupling", which is exactly the
    # kind of implicit default a reader of the report should not have to reconstruct.
    effective_require_skin = (not require_face) if require_skin is None else require_skin

    screens: List[PoolScreen] = []
    if detector:
        by_original: Dict[str, List[SampleRecord]] = collections.defaultdict(list)
        for s in samples:
            by_original[str(s.original_image_path)].append(s)
        screens = screen_originals(
            picked, by_original, root,
            region_cfg=region_config,
            cache_dir=Path(root / cache_dir) if cache_dir else None,
            min_face_frac=min_face_frac, max_face_frac=max_face_frac,
            min_person_frac=min_person_frac,
            min_skin_frac=min_skin_frac, max_skin_frac=max_skin_frac,
            max_person_components=max_person_components, require_face=require_face,
            require_skin=require_skin,
            align_iou=align_iou, check_alignment=check_alignment,
            align_max_checks=align_max_checks,
        )
        keep_ids = {
            sc.sample_id for sc in screens
            if sc.passed_detector and (sc.passed_alignment or not check_alignment)
        }
        # Report the tier table so the n-versus-perceptibility trade-off is explicit.
        report["face_tiers"] = {
            f">{tier}": sum(1 for sc in screens if sc.face_frac > tier)
            for tier in REPORTED_FACE_TIERS
        }
        report["skin_tiers"] = {
            f">{tier}": sum(1 for sc in screens
                            if sc.skin_frac > tier and sc.person_frac >= min_person_frac)
            for tier in REPORTED_SKIN_TIERS
        }
        report["detector_screen"] = {
            "candidates_measured": len(screens),
            "passed_detector": sum(1 for sc in screens if sc.passed_detector),
            "passed_alignment": sum(1 for sc in screens if sc.passed_alignment),
            "kept": len(keep_ids),
            "min_face_frac": min_face_frac if require_face else None,
            "min_person_frac": min_person_frac,
            "max_face_frac": max_face_frac,
            "min_skin_frac": min_skin_frac if effective_require_skin else None,
            "max_skin_frac": max_skin_frac,
            "max_person_components": max_person_components,
            "require_face": require_face,
            "require_skin": effective_require_skin,
            "align_iou": align_iou,
            "check_alignment": check_alignment,
            "align_max_checks": align_max_checks,
            "drop_reasons": dict(sorted(collections.Counter(
                sc.reason.split(" ")[0] for sc in screens
                if sc.sample_id not in keep_ids and sc.reason
            ).items())),
            # The `content_category` miss rate is itself a dataset-quality finding.
            "kept_by_declared_category": dict(sorted(collections.Counter(
                sc.content_category for sc in screens if sc.sample_id in keep_ids
            ).items())),
        }
        picked = [s for s in picked if s.sample_id in keep_ids]
        report["screened_originals"] = len(picked)

    # ★ OPERATOR-COLLISION EXCLUSION on the dataset's OWN task label (policy B, user decision
    # 2026-07-31). A skin-lightness manipulation is a photometric operator; if the instruction
    # is also a whole-frame photometric operator the judge is asked two questions at once and a
    # score shift cannot be attributed. Measured on OmniEdit, edit region vs skin mask:
    # `style` median overlap 0.994, `env` 0.981, `attribute_modification` only 0.172.
    #
    # ⚠️ This screens `metadata.raw_task` -- the SOURCE dataset's label -- not our derived
    # `edit_type`, and the distinction matters: `env` reaches us mapped to `background`, so an
    # `exclude_edit_types` rule would either miss it or take the whole `background` family with
    # it. Both sides of the exclusion are reported; a policy that removes a fifth of the pool
    # has to show its work or the next reader cannot tell a corpus limit from a screening choice.
    if exclude_tasks:
        drop_tasks = {str(t) for t in exclude_tasks}
        task_of = {s.sample_id: getattr(s.metadata, "raw_task", None) for s in samples}
        dropped = collections.Counter()
        kept: List[SampleRecord] = []
        for s in picked:
            task = task_of.get(s.sample_id)
            if task in drop_tasks:
                dropped[str(task)] += 1
            else:
                kept.append(s)
        picked = kept
        report["exclude_tasks"] = sorted(drop_tasks)
        report["dropped_by_task_exclusion"] = dict(sorted(dropped.items()))
        report["after_task_exclusion"] = len(picked)

    # ★★ THE SAME COLLISION, ONE LEVEL DOWN, AND IT IS THE BIGGER HALF.
    # OmniEdit's `task` column is coarse: it labels 293 of the 1,022 in-band scenes
    # `attribute_modification`, and the finer sub-task in `omni_edit_id` shows they are
    # `attr_mod_facial` -- "make him be angry", "let him be excited", "make the dryad feel
    # amused". The instruction acts on the FACE, which is exactly where a skin-lightness
    # manipulation acts, so the judge is asked two questions at once. Policy B could not see
    # them, and a contact sheet of the in-band scenes did.
    #
    # ⚠️ This is NOT the plan's "exclude attribute_modification" rule, which would also take
    # `attr_mod_color` (281 scenes: "turn the color of scroll to black") -- a disjoint object,
    # measured skin overlap 0.172, no collision. Excluding the family wholesale costs 281
    # scenes for no validity gain; excluding only its facial half costs 293 and buys the
    # clean causal reading. The measurement is what separates the two, not the label.
    if exclude_id_substrings:
        needles = [str(x) for x in exclude_id_substrings]
        dropped_ids = collections.Counter()
        kept_ids: List[SampleRecord] = []
        for s in picked:
            hit = next((n for n in needles if n in s.sample_id), None)
            if hit:
                dropped_ids[hit] += 1
            else:
                kept_ids.append(s)
        picked = kept_ids
        report["exclude_id_substrings"] = needles
        report["dropped_by_id_exclusion"] = dict(sorted(dropped_ids.items()))
        report["after_id_exclusion"] = len(picked)

    # ★ THE CONSTRUCT SCREEN'S PASS LIST -- the one screen that is not a measurement.
    #
    # The build runs TWICE on purpose, and the loop is not accidental:
    #   pass 1  k unset, no include_ids  -> the candidate pool
    #   screen  run_construct_screen over that pool                (paid, floor-calibrated)
    #   pass 2  include_ids = its pass list, k set -> the final pool
    #
    # It lives here rather than as a post-hoc filter on the manifest so that the ORB collapse
    # and the seeded prefix both run over the SCREENED set. Filtering afterwards would take a
    # prefix of a contaminated list and then delete from it, which is a different -- and
    # smaller -- sample than taking a prefix of the clean one.
    #
    # ⚠️ A scene whose ORB representative failed the screen takes its whole group with it, even
    # if a collapsed member might have passed. That member was never screened, so admitting it
    # would mean gating on an unmeasured verdict. Conservative, and stated rather than silent.
    if include_ids:
        allowed = json.loads((root / include_ids).read_text(encoding="utf-8"))
        allowed = set(allowed if isinstance(allowed, list) else allowed.get("passed", []))
        before = len(picked)
        picked = [s for s in picked if s.sample_id in allowed]
        report["include_ids"] = include_ids
        report["include_ids_listed"] = len(allowed)
        report["dropped_by_include_ids"] = before - len(picked)
        report["after_include_ids"] = len(picked)

    groups = collapse_duplicates(picked, root, threshold=threshold,
                                 screen_degenerate=orb_screen_degenerate)
    instrument = getattr(collapse_duplicates, "last_instrument_check", None)

    aligned_by_id = {sc.sample_id: sc for sc in screens}
    records: List[SampleRecord] = []
    for g in groups:
        rep = g[0].model_copy(deep=True)
        # `extra="allow"` on SampleMetadata is how subset_block / anchor_source ride along;
        # the collapsed members go the same way so the paper can cite them.
        rep.metadata.subset_block = "fairness"
        rep.metadata.duplicate_members = [s.sample_id for s in g]
        rep.metadata.duplicate_sources = sorted({s.source_dataset for s in g})
        screen = aligned_by_id.get(g[0].sample_id)
        if screen is not None:
            rep.metadata.person_frac = screen.person_frac
            rep.metadata.face_frac = screen.face_frac
            rep.metadata.person_method = screen.person_method
            rep.metadata.skin_frac = screen.skin_frac
            # Which editors keep the head in place -- the D-G arm picks from these, and the
            # list was computed BEFORE any manipulation existed, so it is attribute-blind.
            rep.metadata.aligned_sample_ids = screen.aligned_sample_ids
            rep.metadata.best_align_iou = screen.best_align_iou
        records.append(rep)

    # ★ SEEDED-SHUFFLE-THEN-TAKE-PREFIX (project red line R-A, seed 42, k never lowered).
    # Applied AFTER the ORB collapse so it samples content-distinct SCENES, not near-duplicate
    # rows -- sampling before the collapse would let one photo occupy several draws.
    #
    # The superset property (raising k keeps every earlier scene, so injected images and paid
    # judge calls stay valid) holds for a FIXED candidate set. Adding a corpus shard changes
    # the list the permutation runs over and therefore breaks it. That is the reason to decide
    # the source shards once, up front, rather than topping them up later.
    scenes_before_k = len(records)
    if k is not None and k < len(records):
        order = list(range(len(records)))
        random.Random(f"{seed}:fairness_pool").shuffle(order)
        records = [records[i] for i in order[:k]]

    report.update({
        "content_distinct_scenes": len(records),
        "scenes_before_k": scenes_before_k,
        "k": k,
        "seed": seed,
        "duplicate_inlier_threshold": threshold,
        "orb_screen_degenerate": orb_screen_degenerate,
        "orb_instrument": instrument,
        "component_sizes": dict(sorted(collections.Counter(len(g) for g in groups).items())),
        "by_source_representative": dict(
            sorted(collections.Counter(r.source_dataset for r in records).items())
        ),
        "by_source_path_distinct": dict(
            sorted(collections.Counter(s.source_dataset for s in picked).items())
        ),
        "out_path": out_path if write else None,
    })
    # G1 reads this: the smallest two-sided Wilcoxon p reachable at n is 2/2**n, so a tiny
    # pool cannot reach significance at all regardless of effect size.
    from edit_judge_bias.metrics.fairness_metrics import minimum_detectable_effect
    report["mde_sd"] = (
        round(minimum_detectable_effect(len(records)), 4) if records else None
    )

    if write:
        io.write_jsonl(root / out_path, records)
        if screens_out and screens:
            path = root / screens_out
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for sc in screens:
                    fh.write(json.dumps(sc.as_row(), ensure_ascii=False) + "\n")
            report["screens_out"] = screens_out
    return report


#: Config keys `build()` accepts from a YAML file. Anything else in the file is a typo and
#: raises, rather than being silently ignored -- a silently-ignored `min_face_frac` would
#: build a different pool than the one the config claims to describe.
_CONFIG_KEYS = {
    "samples_path", "samples_paths", "out_path", "threshold", "orb_screen_degenerate",
    "detector",
    "require_human_category", "min_face_frac", "max_face_frac", "min_person_frac",
    "min_skin_frac", "require_face", "require_skin", "max_skin_frac",
    "max_person_components", "align_iou",
    "check_alignment", "align_max_checks", "exclude_edit_types", "exclude_tasks",
    "exclude_id_substrings", "include_ids", "k", "seed", "region_config", "cache_dir",
    "screens_out",
}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(cfg) - _CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"unknown keys in {path}: {sorted(unknown)}; known keys: {sorted(_CONFIG_KEYS)}"
        )
    region = cfg.get("region_config")
    if isinstance(region, str):
        # A path to a region config, so the same file feeds the injectors and the pool.
        cfg["region_config"] = yaml.safe_load(
            (path.parent.parent.parent / region).read_text(encoding="utf-8")
        ) if not Path(region).exists() else yaml.safe_load(
            Path(region).read_text(encoding="utf-8")
        )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="YAML config; CLI flags override it")
    ap.add_argument("--root", default=None)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=int, default=None)
    ap.add_argument("--detector", action="store_true", help="detector-first screen")
    ap.add_argument("--screens-out", default=None)
    # ★ The two-pass build (see `build()`): pass 1 with neither flag produces the candidate
    # pool, `run_construct_screen` measures it, pass 2 replays the SAME config with the screen's
    # pass list and a k. Flags rather than a second YAML so the two passes cannot drift apart in
    # any screening parameter -- the only difference between them is on this command line.
    ap.add_argument("--include-ids", default=None,
                    help="JSON list (or {passed: [...]}) of sample_ids the construct screen passed")
    ap.add_argument("--k", type=int, default=None,
                    help="seeded prefix size; raising it later yields a strict superset")
    ap.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    ap.add_argument("--report", action="store_true", help="alias for --dry-run")
    a = ap.parse_args(argv)

    kwargs: dict = {}
    if a.config:
        kwargs.update(load_config(Path(a.config)))
    if a.samples:
        kwargs["samples_path"] = a.samples
        kwargs.pop("samples_paths", None)
    if a.out:
        kwargs["out_path"] = a.out
    if a.threshold is not None:
        kwargs["threshold"] = a.threshold
    if a.detector:
        kwargs["detector"] = True
    if a.screens_out:
        kwargs["screens_out"] = a.screens_out
    if a.include_ids:
        kwargs["include_ids"] = a.include_ids
    if a.k is not None:
        kwargs["k"] = a.k

    rep = build(
        root=Path(a.root) if a.root else None,
        write=not (a.dry_run or a.report),
        **kwargs,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
