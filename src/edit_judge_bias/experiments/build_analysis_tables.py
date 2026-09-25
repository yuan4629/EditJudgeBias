"""Build the write-up analysis tables (§6 quality, §9.4 breakdowns, positive control).

Local, no-network post-hoc analyses over already-collected results. The headline
claims live in `build_claim_tables.py`; this module holds the supporting slices.

- **§6 combined quality table** (`quality_combined.csv`): per `(validator_model,
  bias_type)`, fold the independent quality-preservation signals into one row —
  automatic SSIM and the MLLM validator pass-rate (from
  `<results>/quality/validation__*.jsonl`) plus, when they exist, the human
  No/Slightly/Yes counts. Each validator's own pass rate on the `sham` arm is carried
  on its rows as `control_pass_rate`: sham is a visually null re-encode, so whatever it
  fails is that validator's FALSE-FLAG FLOOR, and a bias that passes at the same rate
  has not been shown to change anything.

  **One row PER VALIDATOR, never pooled.** Pooling was the original behaviour and it is
  wrong in a way that silently moves the headline: the validators disagree about which
  cue is the boundary case (gemini flags `zoom_inset` 0.891 and passes `text_overlay`
  1.000; gpt-4o-mini is the other way round, 0.964 / 0.900), and a third validator with
  a ~21% false-flag floor (`glm-4v`, sham 0.786 on the n=14 calibration) would drag both
  the pass rates AND the pooled sham floor — so `above_control_floor`, the single fact
  §6 rests on, could flip because of which validators happened to be on disk. Each
  validator is therefore scored against ITS OWN floor, and `--validators` names which
  ones to emit (mirroring `aggregate_results.py`'s gating flag, which takes the
  intersection for the QC subset). The human columns are validator-independent and are
  repeated on each validator's row so a row stays self-contained.

  **The floor is tested twice, and the second test is the honest one.**
  `floor_p_value` is Fisher exact, which asks whether two INDEPENDENT rates differ — but
  the two rates are not measured on the same pictures. `run_quality_validation._select`
  builds one `random.Random(seed)` outside its loop and shuffles each bias bucket
  without resetting it, and all 11 buckets are the same 1,196 base images, so the
  un-reset RNG is the only thing that made the draws differ: `sham`'s 110 images overlap
  each arm's 110 by 7-13 (6-12%). Every Fisher verdict therefore mixes the arm's effect
  with "these pictures are harder". The `*_matched` columns pair each arm image against
  the `sham` verdict on the SAME base image and run McNemar over the discordants
  (`n_discordant_b` = the arm failed where the null perturbation passed). Read
  `matched_design_complete` first: while it is False the matched test rests on whatever
  the two draws happen to share, which is the confound rather than its repair. WP-A3
  tops `sham` up to the union of the ten arms (670 calls per validator) to make it True
  without touching one published arm number; both tests then ship side by side.

- **§9.4 score-shift breakdowns** (`scoring_shift_by_edit_type.csv`,
  `scoring_shift_by_content_category.csv`): the headline per (judge, bias_type) shift
  sliced by the `edit_type` / `content_category` of the base sample (joined back from
  the sample manifest, since `JudgeResult` does not carry them). Two MARGINALS, on
  purpose — the binding write-up commitment is that the full interaction is not
  reported as if it were populated, because it is not: `edit_type_content_coverage.csv`
  records which of the 30 cells exist, and four are structurally empty for semantic
  reasons (there is no `low-level x human` instruction).

  **These two tables are stamped `family="uncorrected — exploratory"` and carry NO
  q-value, deliberately.** 390 + 325 raw p-values with no label is how a bare p gets
  quoted as if it were corrected, but manufacturing a family here would be worse than
  saying so: these cells are SLICES of claim A's rows, and the table is read in both
  directions (down a cue across 6 edit types, and across a slice's 13 cues), so any
  within-slice BH would leave the choice of WHICH of the 30 (edit_type) / 25 (content)
  slicings to quote uncorrected — the multiplicity that actually bites. §9.4's binding
  commitment is that these are descriptive marginals and that no claim rests on a
  single cell; the label says exactly that, in the file.

- **Positive control** (`positive_control.csv`): `low-level x {brightness, saturation,
  aesthetic_filter}` is an *operator collision* — the instruction is already "enhance
  the brightness", so applying the brightness cue re-does the edit rather than dressing
  it up. The judge SHOULD move there, which makes those cells the positive control the
  study otherwise lacks; and since the perturbation is not quality-preserving on them,
  the same table reports each affected bias with the colliding samples removed, so the
  headline can be checked against a clean subset.

  **BH family = one per `subset`** (`positive_control_collision` / `_clean` / `_pooled`,
  15 tests each = 5 judges x 3 colliding cues), matching `claim_A`'s precedent of one
  family per block of hypotheses with the judges pooled. Why not all 45 in one family:
  `pooled` is the deterministic union of `collision` and `clean` over the same judge x
  cue cells, so a single family would count most observations twice and make `m` a
  function of how many nested views the table happens to print, whereas each subset is
  one sample population and the 15 cells inside it are what a reader scans as a block
  ("the collision cells are null for 4/5 judges").

Reuses `metrics.scoring_metrics.compute_score_shifts` so every cell is computed exactly
like the headline table — a breakdown cell is just the headline metric over the subset of
biased results whose base sample falls in that group.

    python -m edit_judge_bias.experiments.build_analysis_tables \\
        --results-dir results/v2 \\
        --manifest data/manifests/samples_judge_v2.jsonl \\
        --subset-filter subset_block=breadth
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root
from edit_judge_bias.data.schema import (
    JudgeResult,
    QualityValidationResult,
    SampleRecord,
)
from edit_judge_bias.experiments.aggregate_results import sample_ids_matching
from edit_judge_bias.metrics.scoring_metrics import compute_score_shifts
from edit_judge_bias.metrics.stats import (
    benjamini_hochberg,
    fisher_exact_pvalue,
    mcnemar_pvalue,
)

# The §2.1 roster (excludes the mock control and the exploratory gpt-5.4-nano mini
# run) — the breakdown tables only make sense for real judges that ran a full grid.
ROSTER = (
    "gpt-5.5",
    "gemini-3.5-flash",
    "gpt-4o-viescore",
    "qwen3.5-plus",
    "kimi-k2.5",
)

# Stable column order for the breakdown tables.
HUMAN_LABELS = ("No", "Slightly", "Yes")

#: The placebo arm — its pass rate is the validator's false-flag floor, not a result.
CONTROL_BIAS = "sham"

#: The operator collision. `Enhance the brightness` IS the brightness injector, so on
#: these samples the cue is not a quality-preserving perturbation and a score change is
#: the correct behaviour rather than a bias.
COLLISION_EDIT_TYPE = "low-level"
COLLISION_BIASES = ("brightness", "saturation", "aesthetic_filter")

#: Stamped on the two §9.4 marginals. They publish a `p_value` column and no q, and the
#: label is what stops a bare p being read as a corrected one. See the module docstring
#: for why a within-slice family would be less honest here, not more.
EXPLORATORY_FAMILY = "uncorrected — exploratory"


# --------------------------------------------------------------------------- #
# shared IO helpers                                                           #
# --------------------------------------------------------------------------- #
def _read_judge(path: Path) -> List[JudgeResult]:
    return io.read_jsonl(path, JudgeResult) if path.exists() else []


def _discover_models(results_dir: Path, task: str) -> List[str]:
    raw = results_dir / "raw_judgments"
    prefix = f"{task}__"
    return sorted(p.stem[len(prefix):] for p in raw.glob(f"{prefix}*.jsonl"))


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# §6 combined quality table                                                   #
# --------------------------------------------------------------------------- #
def _read_human_labels(path: Path) -> Dict[str, Counter]:
    """bias_type -> Counter of No/Slightly/Yes human labels (utf-8-sig: strips BOM)."""
    out: Dict[str, Counter] = {}
    for row in _human_rows(path):
        bias, label = row[0], row[2]
        out.setdefault(bias, Counter())[label] += 1
    return out


def _human_rows(path: Path) -> List[Tuple[str, str, str]]:
    """(bias_type, biased_id, label) triples from one human-label CSV.

    `biased_id` may be empty: the published 150-item package carries it, and any package
    that does not simply cannot be checked for overlap with the validator leg — which is
    itself the finding (see `n_human_matched`).
    """
    out: List[Tuple[str, str, str]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            bias = (row.get("bias_type") or "").strip()
            label = (row.get("quality_changed (No/Slightly/Yes)") or "").strip()
            if not bias or not label:
                continue
            out.append((bias, (row.get("biased_id") or "").strip(), label))
    return out


def _read_human_packages(paths: Sequence[Path]) -> Tuple[
    Dict[str, Counter], Dict[str, Set[str]], Dict[str, List[str]]
]:
    """Fold several human-label packages into per-cue counts, ids, and provenance.

    Several, because the two packages this project has cover DISJOINT cues and were
    drawn from different manifests. The published 150 came from the pilot manifests
    while the validators ran v2, and the legs were aligned only by the `bias_type`
    string — so 150 human labels and 1,210 validator verdicts share exactly 5 pictures
    (3.3%). WP-A6a's package draws from the validators' own published draw instead. Both
    are reported in the same `n_human` column, so the row also carries which package the
    labels came from and how many of them the validator actually saw.
    """
    counts: Dict[str, Counter] = {}
    ids: Dict[str, Set[str]] = {}
    packages: Dict[str, List[str]] = {}
    for path in paths:
        rows = _human_rows(path)
        if not rows:
            continue
        for bias, biased_id, label in rows:
            counts.setdefault(bias, Counter())[label] += 1
            if biased_id:
                ids.setdefault(bias, set()).add(biased_id)
            names = packages.setdefault(bias, [])
            if path.stem not in names:
                names.append(path.stem)
    return counts, ids, packages


#: §13 gate: a bias whose quality-preservation pass rate falls below this cannot carry
#: a claim, because the score change may simply be a correct quality penalty.
PASS_GATE = 0.85


def _matched_floor(
    arm: Dict[str, bool], control: Dict[str, bool]
) -> Dict[str, Optional[float]]:
    """Pair one arm against `sham` on shared base images; McNemar on the discordants.

    ``b`` = the arm failed where the null perturbation passed — the only direction that
    reads as "this cue damaged something". ``c`` = the reverse. Returned as a dict so an
    empty intersection (n=0) is still a row rather than a hole: `n=0` is what "these two
    draws share no pictures" looks like, and it is a fact about the DESIGN, which is
    exactly what this arm exists to expose.

    Why McNemar and not Fisher on the same images: pairing removes the per-image
    difficulty that Fisher has to absorb as noise. `metrics/stats.py:87-98` returns None
    when there are no discordant pairs — an undefined test, not a null result, and the
    write-up must say so rather than printing a blank as if it were "not significant".
    """
    shared = sorted(set(arm) & set(control))
    b = sum(1 for s in shared if not arm[s] and control[s])
    c = sum(1 for s in shared if arm[s] and not control[s])
    return {"n": len(shared), "b": b, "c": c, "p": mcnemar_pvalue(b, c)}


def build_quality_combined(
    results_dir: Path,
    human_csv: Path | Sequence[Path],
    *,
    validators: Optional[Sequence[str]] = None,
) -> List[dict]:
    """One row per (validator_model, bias_type): SSIM + MLLM pass + human + own floor.

    ``validators`` names which `validation__<model>.jsonl` files to emit; ``None`` emits
    every one found. Validators are NEVER pooled — see the module docstring for why that
    was a headline-moving defect rather than a cosmetic one.
    """
    quality_dir = results_dir / "quality"
    wanted = set(validators) if validators else None
    # validator -> bias -> samples
    ssim_by: Dict[str, Dict[str, List[float]]] = {}
    pass_by: Dict[str, Dict[str, List[bool]]] = {}
    # validator -> bias -> base_sample_id -> passed. Same verdicts as `pass_by`, keyed so
    # an arm can be paired against `sham` on the SAME base image (see the matched block).
    pass_ids: Dict[str, Dict[str, Dict[str, bool]]] = {}
    for path in sorted(quality_dir.glob("validation__*.jsonl")):
        validator = path.stem[len("validation__"):]
        if wanted is not None and validator not in wanted:
            continue
        ssim_by.setdefault(validator, {})
        pass_by.setdefault(validator, {})
        pass_ids.setdefault(validator, {})
        for r in io.iter_jsonl(path, QualityValidationResult):
            if not r.parse_success or not r.biased_id:
                continue
            base, _, bias = r.biased_id.rpartition("__")
            if r.ssim is not None:
                ssim_by[validator].setdefault(bias, []).append(r.ssim)
            if r.passed is not None:
                pass_by[validator].setdefault(bias, []).append(bool(r.passed))
                pass_ids[validator].setdefault(bias, {})[base] = bool(r.passed)

    human_paths = [Path(human_csv)] if isinstance(human_csv, (str, Path)) else [
        Path(p) for p in human_csv
    ]
    human, human_ids, human_pkgs = _read_human_packages(human_paths)
    rows: List[dict] = []
    # p-values are collected per validator so BH corrects within one validator's own
    # family of 11 cues, never across validators (they are different instruments).
    p_index: Dict[str, List[int]] = {}
    for validator in sorted(set(ssim_by) | set(pass_by)):
        by_bias_ssim = ssim_by.get(validator, {})
        by_bias_pass = pass_by.get(validator, {})
        # This validator's OWN sham arm. Left blank when it never reached sham (glm-4v
        # stopped at 507/1210), which is the honest reading: its floor is unmeasured, so
        # no `above_control_floor` verdict can be issued for it at all.
        control = by_bias_pass.get(CONTROL_BIAS, [])
        control_rate = sum(control) / len(control) if control else None
        control_ids = pass_ids.get(validator, {}).get(CONTROL_BIAS, {})

        for bias in sorted(set(by_bias_ssim) | set(by_bias_pass) | set(human)):
            ssims = by_bias_ssim.get(bias, [])
            passes = by_bias_pass.get(bias, [])
            rate = sum(passes) / len(passes) if passes else None
            arm_ids = pass_ids.get(validator, {}).get(bias, {})
            matched = _matched_floor(arm_ids, control_ids) if bias != CONTROL_BIAS else None
            h = human.get(bias, Counter())
            h_total = sum(h.get(k, 0) for k in HUMAN_LABELS)
            # "preserved" = human did NOT see a quality change strong enough to flip
            # ranking: No + Slightly (plan §6.3). "strict" = No only.
            preserved = h.get("No", 0) + h.get("Slightly", 0)
            rows.append({
                "validator_model": validator,
                "bias_type": bias,
                "n_ssim": len(ssims),
                "mean_ssim": round(sum(ssims) / len(ssims), 4) if ssims else "",
                "min_ssim": round(min(ssims), 4) if ssims else "",
                "n_mllm": len(passes),
                "mllm_pass_rate": round(rate, 4) if rate is not None else "",
                # This validator's own false-flag floor. A bias at or above it has not
                # been shown to change anything THIS validator can see.
                "control_pass_rate": (
                    round(control_rate, 4) if control_rate is not None else ""
                ),
                "above_control_floor": (
                    "" if rate is None or control_rate is None else bool(rate >= control_rate)
                ),
                # `above_control_floor` alone over-flags: gpt-4o-mini's floor is 109/110,
                # so 9 of its 11 arms sit "below" it on a one- or two-image difference.
                # This asks whether the gap is larger than sampling noise (Fisher exact
                # on pass/fail x arm/sham, BH-corrected within this validator).
                "floor_p_value": (
                    fisher_exact_pvalue(
                        sum(passes), len(passes) - sum(passes),
                        sum(control), len(control) - sum(control),
                    ) if passes and control and bias != CONTROL_BIAS else ""
                ),
                "floor_q_value": "",
                "below_floor_significant": "",
                # --- the MATCHED floor (WP-A3) --------------------------------- #
                # Fisher above asks whether two INDEPENDENT rates differ, but the two
                # halves are not independent draws of the same thing — they are two
                # perturbations of DIFFERENT pictures, because `_select` shuffles each
                # bias bucket with an un-reset RNG. These columns pair each arm image
                # against the `sham` verdict on the SAME base image and run McNemar.
                # Read `matched_design_complete` first: while it is False the pairing
                # rests on the 7-13 images the two draws happen to share, which is the
                # confound itself, not its repair.
                "n_matched": "" if matched is None else matched["n"],
                "matched_design_complete": (
                    "" if matched is None or not passes
                    else bool(matched["n"] == len(passes))
                ),
                "n_discordant_b": "" if matched is None else matched["b"],
                "n_discordant_c": "" if matched is None else matched["c"],
                "floor_p_matched": "" if matched is None or matched["p"] is None else matched["p"],
                "floor_q_matched": "",
                "below_floor_significant_matched": "",
                "passes_85_gate": "" if rate is None else bool(rate >= PASS_GATE),
                "n_human": h_total,
                # How many of those human labels sit on a picture THIS validator judged.
                # The published 150-item package was drawn from the pilot manifests while
                # the validators ran v2, and the two legs were aligned only by cue name —
                # so "triple validation" is three instruments on three different samples,
                # 5 shared images out of 150. That was prose in §5.7; here it is a number
                # per cell, and WP-A6a's package (drawn from the validators' own draw)
                # makes it 30/30 on the six cues that had no human leg at all.
                "n_human_matched": (
                    len({b.rpartition("__")[0] for b in human_ids.get(bias, set())}
                        & set(arm_ids))
                    if h_total else ""
                ),
                "human_package": "|".join(human_pkgs.get(bias, [])) if h_total else "",
                "human_no": h.get("No", 0),
                "human_slightly": h.get("Slightly", 0),
                "human_yes": h.get("Yes", 0),
                "human_preserved_rate": round(preserved / h_total, 4) if h_total else "",
                "human_strict_no_rate": round(h.get("No", 0) / h_total, 4) if h_total else "",
            })
            p_index.setdefault(validator, []).append(len(rows) - 1)

    for validator, idxs in p_index.items():
        ps = [rows[i]["floor_p_value"] for i in idxs]
        qs = benjamini_hochberg([p if isinstance(p, float) else None for p in ps])
        # The matched family is the SAME family: one validator's own cues, sham excluded
        # by construction (its p is None). Correcting the two sets jointly would double
        # the multiplicity for what is one hypothesis per cell asked two ways.
        ps_m = [rows[i]["floor_p_matched"] for i in idxs]
        qs_m = benjamini_hochberg([p if isinstance(p, float) else None for p in ps_m])
        for i, q, q_m in zip(idxs, qs, qs_m):
            r = rows[i]
            r["floor_p_value"] = round(r["floor_p_value"], 6) if isinstance(
                r["floor_p_value"], float) else ""
            r["floor_q_value"] = round(q, 6) if q is not None else ""
            # Direction comes from the discordant counts, not from the rates: b is the
            # arm losing an image the null perturbation kept, so b > c is the only way
            # "this arm damaged something" can be the reading. b == c with a tiny p is
            # impossible, but b < c with a significant q means the arm passed MORE than
            # its own floor, which is not evidence of damage in either direction.
            r["below_floor_significant_matched"] = (
                "" if q_m is None or r["n_discordant_b"] == ""
                else bool(q_m < 0.05 and r["n_discordant_b"] > r["n_discordant_c"])
            )
            r["floor_p_matched"] = round(r["floor_p_matched"], 6) if isinstance(
                r["floor_p_matched"], float) else ""
            r["floor_q_matched"] = round(q_m, 6) if q_m is not None else ""
            # "This arm damaged something the validator can see" requires BOTH a lower
            # rate than the floor AND a gap that survives multiplicity. Numerically
            # below but not significant = not shown to differ from a null perturbation.
            r["below_floor_significant"] = (
                "" if q is None or r["above_control_floor"] == ""
                else bool(q < 0.05 and not r["above_control_floor"])
            )
    return rows


# --------------------------------------------------------------------------- #
# §9.4 score-shift breakdowns                                                 #
# --------------------------------------------------------------------------- #
def _load_sample_groups(manifest: Path) -> Dict[str, Dict[str, str]]:
    """sample_id -> {'edit_type': ..., 'content_category': ...} from the manifest."""
    out: Dict[str, Dict[str, str]] = {}
    for s in io.iter_jsonl(manifest, SampleRecord):
        out[s.sample_id] = {
            "edit_type": s.edit_type.value,
            "content_category": s.content_category.value,
        }
    return out


def _check_single_score_field(rows: Sequence[dict], label: str) -> None:
    """Same guard as the headline table: one column, one analysis variable."""
    fields = {r["score_field"] for r in rows if r.get("score_field")}
    if len(fields) > 1:
        raise ValueError(
            f"{label} rows mix analysis variables {sorted(fields)}; "
            "mean_shift would not be on one scale across judges"
        )


def _breakdown_rows(
    results_dir: Path,
    sample_group: Dict[str, Dict[str, str]],
    group_key: str,
    *,
    roster: Iterable[str] = ROSTER,
    keep_sample_ids: Optional[set] = None,
) -> List[dict]:
    """Per (judge, bias_type, <group_key>) score-shift, computed exactly as the headline.

    For each group value we restrict the *biased* results to base samples in that group
    and re-run `compute_score_shifts` against the full unbiased baseline (pairing is by
    sample_id, so passing all originals is correct and only the in-group biased rows pair).

    `keep_sample_ids` restricts to one block of the judging subset for the same reason
    the headline table does: the three blocks were judged under different numbers of
    conditions and pooling them changes a cell without changing its heading.
    """
    roster = set(roster)
    rows: List[dict] = []
    for model in _discover_models(results_dir, "scoring"):
        if model not in roster:
            continue
        originals = _keep(
            _read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl"),
            keep_sample_ids,
        )
        biased = _keep(
            _read_judge(results_dir / "biased_judgments" / f"scoring__{model}.jsonl"),
            keep_sample_ids,
        )

        # Partition biased results by the base sample's group value.
        by_group: Dict[str, List[JudgeResult]] = {}
        for r in biased:
            grp = sample_group.get(r.sample_id or "", {}).get(group_key)
            if grp is None:
                continue
            by_group.setdefault(grp, []).append(r)

        for grp in sorted(by_group):
            for st in compute_score_shifts(originals, by_group[grp]):
                row = st.as_row()
                rows.append({
                    "judge_model": row["judge_model"],
                    "bias_type": row["bias_type"],
                    group_key: grp,
                    "score_field": row["score_field"],
                    "n": row["n"],
                    "mean_shift": row["mean_shift"],
                    "sir": row["sir"],
                    "p_value": row["p_value"],
                    # No q-value is emitted with it, on purpose — the label IS the
                    # correction status, and a p-value column with neither is what
                    # lets an uncorrected cell be quoted as a corrected finding.
                    "family": EXPLORATORY_FAMILY,
                })
    _check_single_score_field(rows, f"§9.4 by {group_key}")
    return rows


def _keep(results: List[JudgeResult], keep_sample_ids: Optional[set]) -> List[JudgeResult]:
    if keep_sample_ids is None:
        return results
    return [r for r in results if r.sample_id in keep_sample_ids]


def build_cell_coverage(
    samples: Sequence[SampleRecord], keep_sample_ids: Optional[set] = None
) -> List[dict]:
    """Every (edit_type, content_category) cell and how many samples it holds.

    Written out rather than inferred from the breakdown tables so the EMPTY cells are
    on the page. Four of the 30 are structurally empty on this subset — there is no
    `low-level x human` or `background x animal` instruction to draw — and the binding
    commitment is to say so instead of reporting an interaction that is mostly holes.
    """
    pool = [
        s for s in samples
        if keep_sample_ids is None or s.sample_id in keep_sample_ids
    ]
    counts = Counter((s.edit_type.value, s.content_category.value) for s in pool)
    edit_types = sorted({s.edit_type.value for s in pool})
    contents = sorted({s.content_category.value for s in pool})
    return [
        {
            "edit_type": e,
            "content_category": c,
            "n": counts.get((e, c), 0),
            "empty": counts.get((e, c), 0) == 0,
        }
        for e in edit_types
        for c in contents
    ]


def build_positive_control(
    results_dir: Path,
    sample_group: Dict[str, Dict[str, str]],
    *,
    roster: Iterable[str] = ROSTER,
    keep_sample_ids: Optional[set] = None,
) -> List[dict]:
    """The operator-collision cells, and the same biases with them removed.

    Three rows per (judge, colliding bias):

    ``collision``   the `low-level` samples, where the cue re-applies the instruction's
                    own operator. A judge that does NOT move here is failing to notice a
                    real change — this is the study's positive control.
    ``clean``       every other edit_type, i.e. claim A with the collision excluded.
    ``pooled``      the headline cell, for reference; if `clean` and `pooled` agree, the
                    collision is not what is driving the reported effect.

    Each subset is its own BH family of 5 judges x 3 cues — never one family of 45, in
    which `pooled` would double-count the very rows `collision` and `clean` partition.
    See the module docstring.
    """
    roster = set(roster)
    rows: List[dict] = []
    for model in _discover_models(results_dir, "scoring"):
        if model not in roster:
            continue
        originals = _keep(
            _read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl"),
            keep_sample_ids,
        )
        biased = _keep(
            _read_judge(results_dir / "biased_judgments" / f"scoring__{model}.jsonl"),
            keep_sample_ids,
        )
        biased = [r for r in biased if r.bias_type in COLLISION_BIASES]
        if not biased:
            continue

        def edit_type_of(r: JudgeResult) -> Optional[str]:
            return sample_group.get(r.sample_id or "", {}).get("edit_type")

        subsets = {
            "collision": [r for r in biased if edit_type_of(r) == COLLISION_EDIT_TYPE],
            "clean": [
                r for r in biased
                if edit_type_of(r) not in (None, COLLISION_EDIT_TYPE)
            ],
            "pooled": biased,
        }
        for subset_name, subset in subsets.items():
            for st in compute_score_shifts(originals, subset):
                row = st.as_row()
                rows.append({
                    "judge_model": row["judge_model"],
                    "bias_type": row["bias_type"],
                    "subset": subset_name,
                    "score_field": row["score_field"],
                    "n": row["n"],
                    "mean_shift": row["mean_shift"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "sir": row["sir"],
                    "p_value": row["p_value"],
                    "family": f"positive_control_{subset_name}",
                    "q_value": "",
                    "significant_bh": "",
                })
    _check_single_score_field(rows, "positive control")

    # BH within each subset — one sample population per family, judges pooled exactly
    # as `claim_A` pools them. 23 of these 45 p-values are below 0.05 and the table is
    # quoted cell by cell, so shipping it with no correction at all was the defect.
    by_family: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_family.setdefault(r["family"], []).append(i)
    for idxs in by_family.values():
        qs = benjamini_hochberg(
            [rows[i]["p_value"] if isinstance(rows[i]["p_value"], float) else None
             for i in idxs]
        )
        for i, q in zip(idxs, qs):
            rows[i]["q_value"] = "" if q is None else round(q, 6)
            rows[i]["significant_bh"] = "" if q is None else bool(q < 0.05)
    return sorted(rows, key=lambda r: (r["judge_model"], r["bias_type"], r["subset"]))


BREAKDOWN_FIELDS_EDIT = [
    "judge_model", "bias_type", "edit_type", "score_field", "n",
    "mean_shift", "sir", "p_value", "family",
]
BREAKDOWN_FIELDS_CONTENT = [
    "judge_model", "bias_type", "content_category", "score_field", "n",
    "mean_shift", "sir", "p_value", "family",
]
QUALITY_FIELDS = [
    "validator_model",
    "bias_type", "n_ssim", "mean_ssim", "min_ssim", "n_mllm", "mllm_pass_rate",
    "control_pass_rate", "above_control_floor",
    "floor_p_value", "floor_q_value", "below_floor_significant",
    "n_matched", "matched_design_complete", "n_discordant_b", "n_discordant_c",
    "floor_p_matched", "floor_q_matched", "below_floor_significant_matched",
    "passes_85_gate",
    "n_human", "n_human_matched", "human_package",
    "human_no", "human_slightly", "human_yes",
    "human_preserved_rate", "human_strict_no_rate",
]
COVERAGE_FIELDS = ["edit_type", "content_category", "n", "empty"]
POSITIVE_CONTROL_FIELDS = [
    "judge_model", "bias_type", "subset", "score_field", "n",
    "mean_shift", "ci_low", "ci_high", "sir", "p_value",
    "family", "q_value", "significant_bh",
]


def build_all(
    results_dir: PathLike,
    *,
    manifest: PathLike | None = None,
    human_csv: PathLike | None = None,
    out_dir: PathLike | None = None,
    subset_filter: Optional[dict] = None,
    validators: Optional[Sequence[str]] = None,
) -> dict:
    results_dir = Path(results_dir)
    root = default_root()
    manifest = Path(manifest) if manifest else root / "data" / "manifests" / "samples_pilot.jsonl"
    # BOTH human packages by default, and it has to be both: they cover DISJOINT cues
    # (the published 150 are five B-class cues; WP-A6a's are the six that had none), so
    # naming one silently zeroes `n_human` for the other's cues. A package that has not
    # been produced or not been labelled yet simply contributes nothing.
    human_csv = [Path(p) for p in (human_csv if isinstance(human_csv, (list, tuple)) else [human_csv])] \
        if human_csv else [
            root / "data" / "human_validation" / "human_validation_labels.csv",
            root / "data" / "human_validation_v2" / "human_validation_labels_v2.csv",
        ]
    out_dir = Path(out_dir) if out_dir else results_dir / "metrics"
    keep_ids = sample_ids_matching(manifest, subset_filter) if subset_filter else None

    quality = build_quality_combined(results_dir, human_csv, validators=validators)
    samples = io.read_jsonl(manifest, SampleRecord)
    sample_group = _load_sample_groups(manifest)
    by_edit = _breakdown_rows(results_dir, sample_group, "edit_type", keep_sample_ids=keep_ids)
    by_content = _breakdown_rows(
        results_dir, sample_group, "content_category", keep_sample_ids=keep_ids
    )
    coverage = build_cell_coverage(samples, keep_ids)
    control = build_positive_control(results_dir, sample_group, keep_sample_ids=keep_ids)

    _write_csv(out_dir / "quality_combined.csv", quality, QUALITY_FIELDS)
    _write_csv(out_dir / "scoring_shift_by_edit_type.csv", by_edit, BREAKDOWN_FIELDS_EDIT)
    _write_csv(out_dir / "scoring_shift_by_content_category.csv", by_content, BREAKDOWN_FIELDS_CONTENT)
    _write_csv(out_dir / "edit_type_content_coverage.csv", coverage, COVERAGE_FIELDS)
    _write_csv(out_dir / "positive_control.csv", control, POSITIVE_CONTROL_FIELDS)
    return {
        "quality": quality, "by_edit": by_edit, "by_content": by_content,
        "coverage": coverage, "positive_control": control,
    }


def _print_summary(tables: dict) -> None:
    print("=== §6 combined quality (per validator x bias) ===")
    seen_validator = None
    for r in tables["quality"]:
        if r["validator_model"] != seen_validator:
            seen_validator = r["validator_model"]
            print(f"-- validator: {seen_validator}")
        floor = "  <- BELOW the sham floor (q<0.05)" if r["below_floor_significant"] is True \
            else ("  (numerically below, n.s.)" if r["above_control_floor"] is False else "")
        gate = "" if r["passes_85_gate"] in ("", True) else "  <- FAILS the 85% gate"
        print(f"   {r['bias_type']:<18} ssim={r['mean_ssim']} (min {r['min_ssim']})  "
              f"mllm_pass={r['mllm_pass_rate']} (control {r['control_pass_rate']})"
              f"{floor}{gate}")

    empty = [r for r in tables["coverage"] if r["empty"]]
    print(f"\n=== §9.4 by edit_type: {len(tables['by_edit'])} cells, "
          f"by content_category: {len(tables['by_content'])} cells ===")
    print(f"edit_type x content_category: {len(tables['coverage']) - len(empty)} of "
          f"{len(tables['coverage'])} cells populated; empty = "
          f"{[(r['edit_type'], r['content_category']) for r in empty]}")
    sig = [r for r in tables["by_edit"]
           if isinstance(r["p_value"], float) and r["p_value"] < 0.05]
    print(f"significant edit_type cells (p<0.05, uncorrected): {len(sig)} — both §9.4 "
          f"marginals are stamped family={EXPLORATORY_FAMILY!r} and carry no q-value")

    print("\n=== positive control: low-level x the colliding operators ===")
    fam_sizes = Counter(r["family"] for r in tables["positive_control"])
    print(f"BH families (one per subset): "
          f"{', '.join(f'{k} n={v}' for k, v in sorted(fam_sizes.items()))}")
    for r in tables["positive_control"]:
        print(f"  {r['judge_model']:<18}{r['bias_type']:<18}{r['subset']:<10}"
              f"n={r['n']:<5}shift={r['mean_shift']:>8}  p={r['p_value']}"
              f"  q={r['q_value']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build §6 quality + §9.4 breakdown tables.")
    ap.add_argument("--results-dir", type=Path, default=default_root() / "results")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--human-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--subset-filter", action="append", default=None, metavar="KEY=VALUE",
                    help="restrict to one block of the judging subset, e.g. "
                         "subset_block=breadth (repeatable)")
    ap.add_argument("--validators", action="append", default=None, metavar="MODEL",
                    help="emit quality_combined rows only for these validators "
                         "(repeatable); default is every validator on disk. Validators "
                         "are reported per-row, never pooled.")
    ap.add_argument("--quality-only", action="store_true",
                    help="write only quality_combined.csv and join no human package unless "
                         "--human-csv names one: for a validator arm on its OWN images "
                         "(results/v2_fill, FILL v2 P4), where the default packages were "
                         "drawn from other pictures and would join by cue name alone")
    args = ap.parse_args(argv)
    if args.quality_only:
        # No breakdowns, no positive control: a validator-only tree has no judgments for
        # them, and building them would write empty tables beside a real one.
        human = [args.human_csv] if args.human_csv else []
        rows = build_quality_combined(args.results_dir, human, validators=args.validators)
        if not rows:
            ap.error(f"no validation__*.jsonl under {args.results_dir / 'quality'}; "
                     "refusing to write an empty quality table")
        out = (args.out_dir or (args.results_dir / "metrics")) / "quality_combined.csv"
        _write_csv(out, rows, QUALITY_FIELDS)
        print(f"wrote {out} ({len(rows)} rows)")
        return 0

    flt = None
    if args.subset_filter:
        flt = {}
        for item in args.subset_filter:
            key, _, value = item.partition("=")
            if not _:
                ap.error(f"--subset-filter expects KEY=VALUE, got {item!r}")
            flt[key] = value

    tables = build_all(
        args.results_dir,
        manifest=args.manifest,
        human_csv=args.human_csv,
        out_dir=args.out_dir,
        subset_filter=flt,
        validators=args.validators,
    )
    _print_summary(tables)
    print(f"\nwrote 5 tables -> {args.out_dir or (args.results_dir / 'metrics')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
