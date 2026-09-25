"""Decompose the D-S `dose_control` contrast into its two one-variable components.

★ WHY THIS EXISTS: `dose_control` IS A VALID POSITIVE CONTROL BUT NOT A LOCATION CONTRAST.

`attribute_gaps.csv` reports `dose_control = sham - sham_nonskin`, and that row does the job it
was built for: it proves, on the SAME scenes / pipeline / judges as the skin-tone null, that a
shift of 0.47-0.98 `fine_score` points is detectable, 5/5. Nothing here weakens that.

What it is NOT is "the same dose in two different places". The two arms are

    sham          delta_ita = 0.0   through the SKIN mask       (a placebo: pipeline noise only)
    sham_nonskin  delta_ita = -D    through an area-matched NON-SKIN region of the same person

(`attribute_injectors.SkinToneITAInjector.arms`). So the pair differs in **dose magnitude AND
location at once**, and a sentence like "recolouring clothing costs more than recolouring skin"
does not follow from it -- the skin side of that subtraction carries no dose at all.

This module computes the two contrasts underneath it, on the scenes where a judge answered all
four arms:

    dose_vs_placebo:{dark,light}   study - sham           SAME MASK, dose D vs dose 0
    nonskin_vs_skin:{dark,light}   study - sham_nonskin   dose D on skin vs on non-skin

and they satisfy an exact identity that is asserted in the tests:

    dose_control = (study - sham_nonskin) - (study - sham) = nonskin_vs_skin - dose_vs_placebo

i.e. the frozen positive control is the SUM of a skin-dose penalty and a region effect, both
pointing the same way. Reporting it undecomposed is fine as a control and wrong as a mechanism.

★ THE TWO COMPONENTS ARE NOT EQUALLY CLEAN, AND THE TABLE SAYS SO IN A COLUMN.

`dose_vs_placebo` is the tight one: `sham` runs the identical code path at `delta_ita = 0`
through the SAME `mask_sha256` as the study arm, so the pair differs in dose and in nothing
else -- no region, no area, no segmentation difference. It is therefore the positive control
that belongs next to the skin-tone null: it proves the judges register the very manipulation
whose *direction* the null is about, on the very mask the null is measured over.

`nonskin_vs_skin` is the loose one: `nonskin_same_person_region` matches AREA (median 0.90x the
skin mask), not physical dose, and the construction gate's `max_delta_e00_ratio` only checks
original-vs-edited within a study arm -- never study-vs-`sham_nonskin`. Measured over the 992
scenes that have one: the non-skin arm carries a MEDIAN 1.29x the study arm's per-pixel mean
dE00 and 1.47x its dE00 mass (p90 2.14x / 2.61x). The extra dose pushes the same way as the
claimed region effect, so `nonskin_vs_skin` is an UPPER BOUND on "recolouring clothing costs
more than recolouring skin", not an estimate of it. `--injections` stamps that ratio onto every
`nonskin_vs_skin` row so the caveat travels with the number instead of living in prose.

★ BOTH STUDY ARMS ARE REPORTED, ON PURPOSE. `dark` and `light` carry the same |dose| and differ
only in sign, so their rows must agree; that they do is the same fact as the headline null
(`skin_tone` gap ~ 0). If they ever disagreed, the null would be the thing in doubt, so the
duplication is a cross-check rather than padding.

★ COMPLETE-CASE ON ALL FOUR ARMS, PER JUDGE. The three contrasts are only comparable on one
scene set. 34/12,250 calls were permanently refused, and a scene refused on one arm would
otherwise leave each contrast standing on a slightly different population -- the same discipline
that makes the main grid report `gpt-4o-viescore` complete-case.

    python -m edit_judge_bias.experiments.build_dose_control_decomposition \
        --results-dir results/v2_fairness_ds \
        --samples data/manifests/samples_fairness_ds_judge_v4.jsonl
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_claim_tables import attach_bh, write_csv
from edit_judge_bias.fairness.records import AttributeInjectionRecord
from edit_judge_bias.metrics.fairness_metrics import (
    FairnessKey,
    compute_attribute_gaps,
    keys_from_manifest,
    minimum_detectable_effect,
)
from edit_judge_bias.metrics.scoring_metrics import resolve_score_field

#: The four arms `SkinToneITAInjector` emits. All must be present for a scene to be used.
ALL_ARMS: FrozenSet[str] = frozenset({"dark", "light", "sham", "sham_nonskin"})

#: contrast family -> (study arm, reference arm). Labels sort alphabetically inside
#: `compute_attribute_gaps`, and every study arm sorts before every reference arm here, so the
#: reported gap is always `study - reference` and the sign can be stated once.
CONTRASTS: Dict[str, tuple] = {
    "dose_vs_placebo:dark": ("dark", "sham"),
    "dose_vs_placebo:light": ("light", "sham"),
    "nonskin_vs_skin:dark": ("dark", "sham_nonskin"),
    "nonskin_vs_skin:light": ("light", "sham_nonskin"),
}

#: Families whose reference arm is `sham_nonskin`, i.e. the ones carrying the area-matched-but-
#: not-dose-matched caveat. Used to decide which rows get the measured dose-ratio columns.
NONSKIN_FAMILIES = tuple(f for f, (_, ref) in CONTRASTS.items() if ref == "sham_nonskin")


def _load(results_dir: Path) -> List[JudgeResult]:
    out: List[JudgeResult] = []
    for p in sorted((results_dir / "raw_judgments").glob("scoring__*.jsonl")):
        out.extend(io.read_jsonl(p, JudgeResult))
    return out


def complete_scenes_by_judge(
    results: Sequence[JudgeResult],
    keys: Dict[str, FairnessKey],
    *,
    score_field: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """judge -> base_sample_ids for which that judge answered all four arms.

    Keyed per judge rather than globally: a refusal is a property of (judge, image), so a
    global intersection would punish every judge for one judge's content filter.

    ⚠️ Completeness is judged on the ANALYSIS VARIABLE, not on `parse_success`. `kimi-k2.5`
    returns answers that parse but omit a dimension, so `fine_score` is None on a handful of
    rows; counting those as answered put its four contrasts on two different scene sets
    (476 vs 474), which is precisely the drift this function exists to prevent.
    """
    rows = list(results)
    field_name = score_field or resolve_score_field(rows)
    seen: Dict[str, Dict[str, Set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    for r in rows:
        if not r.parse_success or getattr(r, field_name, None) is None:
            continue
        key = keys.get(r.sample_id or "")
        if key is None or key.render_index != 1:
            continue
        seen[r.judge_model][key.base_sample_id].add(key.variant_label)
    return {
        judge: {base for base, arms in per_scene.items() if ALL_ARMS <= arms}
        for judge, per_scene in seen.items()
    }


def nonskin_dose_ratios(
    injections: Sequence[AttributeInjectionRecord],
    *,
    study_arm: str,
    scenes: Optional[Set[str]] = None,
) -> dict:
    """How much bigger the `sham_nonskin` dose actually is than `study_arm`'s, per scene.

    Returns medians and p90s of the two ratios that matter, plus the share of scenes inside a
    2x band. Reported rather than corrected: there is no defensible way to rescale a judge's
    score by a dose ratio, so the honest move is to publish the confound's size and read
    `nonskin_vs_skin` as an upper bound.
    """
    study: Dict[str, AttributeInjectionRecord] = {}
    control: Dict[str, AttributeInjectionRecord] = {}
    for r in injections:
        if not r.success:
            continue
        if scenes is not None and r.base_sample_id not in scenes:
            continue
        if r.variant_label == study_arm:
            study[r.base_sample_id] = r
        elif r.variant_label == "sham_nonskin":
            control[r.base_sample_id] = r

    mean_ratio, mass_ratio = [], []
    for base, ctl in control.items():
        ref = study.get(base)
        if ref is None:
            continue
        if ref.delta_e00_mean and ctl.delta_e00_mean:
            mean_ratio.append(ctl.delta_e00_mean / ref.delta_e00_mean)
        if ref.delta_e00_mass and ctl.delta_e00_mass:
            mass_ratio.append(ctl.delta_e00_mass / ref.delta_e00_mass)

    def _q(values: List[float], frac: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round(frac * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "n_dose_pairs": len(mean_ratio),
        "dose_mean_ratio_median": _q(mean_ratio, 0.5),
        "dose_mean_ratio_p90": _q(mean_ratio, 0.9),
        "dose_mass_ratio_median": _q(mass_ratio, 0.5),
        "dose_mass_ratio_p90": _q(mass_ratio, 0.9),
        "dose_within_2x_frac": (
            None if not mean_ratio
            else sum(0.5 <= r <= 2.0 for r in mean_ratio) / len(mean_ratio)
        ),
    }


def _remap(keys: Dict[str, FairnessKey], arms: Sequence[str], family: str) -> Dict[str, FairnessKey]:
    """Keep only `arms` and relabel their attribute to `family`.

    `compute_attribute_gaps` needs exactly two labels per (judge, attribute) group; the four
    arms live under two attributes already, so a contrast that crosses them (study vs sham) is
    unreachable without this remap.
    """
    wanted = set(arms)
    return {
        sid: key._replace(attribute=family)
        for sid, key in keys.items()
        if key.variant_label in wanted
    }


def build(
    *,
    root: Optional[Path] = None,
    results_dir: str = "results/v2_fairness_ds",
    samples: str = "data/manifests/samples_fairness_ds_judge_v4.jsonl",
    injections: Optional[str] = "data/manifests/attribute_injections_v4.jsonl",
    out_name: str = "dose_control_decomposition.csv",
) -> dict:
    root = Path(root) if root is not None else default_root()
    rdir = root / results_dir
    results = _load(rdir)
    if not results:
        return {"rows": 0, "error": f"no scoring results under {results_dir}/raw_judgments"}

    samples_path = root / samples
    if not samples_path.exists():
        return {"rows": 0, "error": f"fairness sample manifest not found: {samples}"}
    keys = keys_from_manifest(io.read_jsonl(samples_path, SampleRecord))
    if not keys:
        return {"rows": 0, "error": f"{samples} carries no fairness metadata (base/attribute/variant)"}

    complete = complete_scenes_by_judge(results, keys)
    by_judge: Dict[str, List[JudgeResult]] = collections.defaultdict(list)
    for r in results:
        by_judge[r.judge_model].append(r)

    inj: List[AttributeInjectionRecord] = []
    if injections:
        inj_path = root / injections
        if inj_path.exists():
            inj = list(io.read_jsonl(inj_path, AttributeInjectionRecord))

    rows: List[dict] = []
    # ★ ONE BH FAMILY PER CONTRAST TYPE, NOT PER ARM. `dose_vs_placebo:dark` and
    # `:light` are the same manipulation at opposite sign, on the same 471-482 scenes, and
    # they are reported together as one finding — so correcting them as two families of 5
    # instead of one family of 10 is the least conservative split available and would let a
    # borderline cell pass because its twin was filed separately.
    family_rows_by_contrast: Dict[str, List[dict]] = collections.defaultdict(list)
    for family, (study, reference) in CONTRASTS.items():
        family_rows: List[dict] = []
        for judge in sorted(by_judge):
            scenes = complete.get(judge, set())
            if not scenes:
                continue
            stats = compute_attribute_gaps(
                by_judge[judge],
                _remap(keys, (study, reference), family),
                include_base_sample_ids=scenes,
            )
            for s in stats:
                row = s.as_row()
                row["mde_sd"] = round(minimum_detectable_effect(row["n"]) or 0.0, 4)
                # The area-matched-not-dose-matched caveat is a column, not a footnote: these
                # rows are an upper bound on a region effect by exactly this much.
                if inj and family in NONSKIN_FAMILIES:
                    ratios = nonskin_dose_ratios(inj, study_arm=study, scenes=scenes)
                    row.update({
                        k: (None if v is None else round(v, 4)) for k, v in ratios.items()
                        if k != "n_dose_pairs"
                    })
                    row["n_dose_pairs"] = ratios["n_dose_pairs"]
                family_rows.append(row)
        family_rows_by_contrast[family.split(":", 1)[0]].extend(family_rows)

    for contrast, contrast_rows in sorted(family_rows_by_contrast.items()):
        attach_bh(contrast_rows, family=f"fairness_contrast:{contrast}", p_key="p_value")
        rows.extend(contrast_rows)

    out = rdir / "metrics" / out_name
    write_csv(out, rows)
    return {
        "rows": len(rows),
        "judges": sorted(by_judge),
        "families": sorted({r["attribute"] for r in rows}),
        "scenes_by_judge": {j: len(s) for j, s in sorted(complete.items())},
        "out": str(out.relative_to(root)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Decompose D-S `dose_control` into dose-vs-placebo and location components."
    )
    ap.add_argument("--root", default=None)
    ap.add_argument("--results-dir", default="results/v2_fairness_ds")
    ap.add_argument("--samples", default="data/manifests/samples_fairness_ds_judge_v4.jsonl")
    ap.add_argument("--injections", default="data/manifests/attribute_injections_v4.jsonl",
                    help="measures how far `sham_nonskin`'s physical dose is from the study arm's")
    ap.add_argument("--out-name", default="dose_control_decomposition.csv")
    a = ap.parse_args(argv)
    rep = build(root=Path(a.root) if a.root else None, results_dir=a.results_dir,
                samples=a.samples, injections=a.injections, out_name=a.out_name)
    print(rep)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
