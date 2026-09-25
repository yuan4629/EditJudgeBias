"""Freeze the headline claims into CSVs (claim A / claim B / pairwise / retest).

Everything here was previously computed by one-off scratchpad scripts, which is fine
for a first look and not fine for a paper: the numbers in the write-up have to come
from a file anyone can regenerate with one command, and the multiple-comparison
correction has to be applied to a *declared* family rather than to whatever happened
to be on screen.

Six tables, all under `<results-dir>/metrics/`:

``claim_a.csv``
    Invariance. Per (judge, bias) on the breadth block: the raw shift against the
    unbiased baseline (the headline, and the conservative reading), the paired
    contrast against the `sham` placebo (a robustness column — see
    `scoring_metrics.compute_placebo_contrast` for why it is not the headline), and
    the placebo's own interval as an equivalence bound.

``claim_b.csv``
    Validity. Per (judge, anchor source, bias): judge-human rank correlation and
    derived pairwise accuracy before and after, with cluster-bootstrap intervals.
    Answers "isn't the deflation a *justified* deduction?" — if the perturbation
    pushed the judge away from human judgement, it was not a correction.

``pairwise_one_sided.csv``
    The same question in the deployed format: one member of the pair is dressed up,
    and the verdict is asked whether it moves toward the cue, and whether that
    movement costs agreement with a real human vote.

``position.csv`` / ``position_joint.csv``
    Slot robustness, and the displayed-letter joint distribution that separates a
    slot-sticky judge from a merely noisy one. `position.csv` holds the same numbers
    as `aggregate_results`' `pairwise_position.csv` — both call
    `compute_position_flips`, so the two files disagreeing is a bug signal rather than
    a finding. It is repeated here so the claim set is self-contained.

``retest.csv``
    The judge's own noise floor: the same question asked twice with the cache off.

Multiplicity: p-values are corrected with Benjamini-Hochberg, and every row carries
the `family` it was corrected in, so a reader never has to guess what `q_value` was
computed over. The families are declared at the :func:`attach_bh` call sites. Claim
B's *rank* half has no p-value — a rho difference is tested by its cluster-bootstrap
interval — so it reports `rho_ci_excludes_zero` and only the accuracy half is
BH-corrected.

    python -m edit_judge_bias.experiments.build_claim_tables \\
        --results-dir results/v2 \\
        --samples data/manifests/samples_judge_v2.jsonl \\
        --subset-filter subset_block=breadth
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root
from edit_judge_bias.data.schema import JudgeResult, PairRecord, SampleRecord
from edit_judge_bias.experiments.aggregate_results import (
    A5_ROSTER,
    MIN_JUDGE_COVERAGE,
    PUBLISHED_ROSTER,
    ROSTERS,
    _judges,
    PARTIAL_JUDGE_LOG,
    _discover_models,
    sample_ids_matching,
)
from edit_judge_bias.metrics.agreement import compute_agreement_grid
from edit_judge_bias.metrics.pairwise_metrics import (
    compute_consistency,
    compute_one_sided_bias,
    compute_position_flips,
    decisive_human_prefs,
    position_joint,
)
from edit_judge_bias.metrics.scoring_metrics import (
    compute_placebo_contrast,
    compute_retest_stats,
    compute_score_shifts,
)
from edit_judge_bias.metrics.stats import benjamini_hochberg

#: The placebo arm. Excluded from claim A's BH family: it is the control the family
#: is measured against, not one of the hypotheses being tested.
CONTROL_BIAS = "sham"

#: Which biases the human-anchor arm asked (`configs/experiment/scoring_anchor_v2.yaml`).
#: Claim B can only speak about these.
ANCHOR_BIASES = ("padding", "text_overlay", "brightness", "region_annotation")

#: Since 2026-09-15 `claim_b.csv` and `pairwise_one_sided.csv` (and their `a5_` copies) hold
#: two collections: the main grid's cells, measured against the unbiased baseline, and the
#: FILL v2 cells (`build_fill_tables`), measured against the fill's own `sham`.  Every row
#: names both in `collection` / `reference`, and the two collections keep apart BH families.
MAIN_COLLECTION = "main grid (results/v2)"
BASELINE_LABEL = "unbiased baseline"


# --------------------------------------------------------------------------- #
# IO helpers                                                                   #
# --------------------------------------------------------------------------- #
def _read_judge(path: Path) -> List[JudgeResult]:
    return io.read_jsonl(path, JudgeResult) if path.exists() else []


# The partial-judge guard now lives in `aggregate_results` because BOTH builders glob the
# same directory and both had to be protected -- see that module for why, and note that
# this file already imports from it (`sample_ids_matching`), so the dependency runs one
# way only.  Re-exported here because this is where the guard was born and where the
# tests reach for it.




def write_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write rows with the union of their keys, in first-seen order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def with_provenance(row: dict, *, reference: str, collection: str) -> dict:
    """A copy of `row` with `reference` and `collection` placed just before `family`."""
    out: dict = {}
    for key, value in row.items():
        if key == "family":
            out["reference"], out["collection"] = reference, collection
        out[key] = value
    out.setdefault("reference", reference)
    out.setdefault("collection", collection)
    return out


def main_grid_rows(rows: Iterable[dict]) -> List[dict]:
    """The main grid's rows of a merged claim table; a row with no `collection` is one.

    A consumer whose estimand was defined on the main grid's cells -- the robustness axes,
    figure 5, the claim B forest, the permutation check -- reads these and nothing else.
    The FILL rows in the same file are another collection measured against another
    reference, and reading them would silently turn "8 cells per judge" into 24.
    """
    return [r for r in rows if (r.get("collection") or MAIN_COLLECTION) == MAIN_COLLECTION]


def attach_bh(rows: List[dict], *, family: str, p_key: str) -> None:
    """Stamp `family` / `q_value` / `significant_bh` onto every row, in place.

    The family is written into the row rather than being implied by the file, because
    one file can hold two families (claim B corrects rho and accuracy separately) and
    a q-value with no stated family is not interpretable.
    """
    qs = benjamini_hochberg([r.get(p_key) for r in rows])
    for row, q in zip(rows, qs):
        row["family"] = family
        row["q_value"] = None if q is None else round(q, 6)
        row["significant_bh"] = None if q is None else bool(q < 0.05)


def _single_score_field(rows: Iterable[dict], label: str) -> None:
    """Refuse to write a table whose judges disagree on the analysis variable.

    Same guard as `aggregate_results`: `mean_shift` holding both 3-30 dimension sums
    and 1-10 overall scores reads as one column and is two, and the odd judge looks an
    order of magnitude more robust than it is.
    """
    fields = {r["score_field"] for r in rows if r.get("score_field")}
    if len(fields) > 1:
        raise ValueError(
            f"{label} rows mix analysis variables {sorted(fields)}; "
            "the column would not be on one scale across judges"
        )


# --------------------------------------------------------------------------- #
# claim A — invariance                                                        #
# --------------------------------------------------------------------------- #
def build_claim_a(results_dir: Path, keep_sample_ids: Optional[set] = None,
                  roster: Optional[Sequence[str]] = PUBLISHED_ROSTER) -> List[dict]:
    """Raw shift + placebo contrast + equivalence bound, per (judge, bias)."""
    def keep(results: List[JudgeResult]) -> List[JudgeResult]:
        if keep_sample_ids is None:
            return results
        return [r for r in results if r.sample_id in keep_sample_ids]

    rows: List[dict] = []
    for model in _judges(results_dir, "scoring", roster):
        originals = keep(_read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl"))
        biased = keep(_read_judge(results_dir / "biased_judgments" / f"scoring__{model}.jsonl"))
        if not biased:
            continue
        shifts = {st.bias_type: st.as_row() for st in compute_score_shifts(originals, biased)}
        placebo = {
            st.bias_type: st.as_row()
            for st in compute_placebo_contrast(biased, control_bias=CONTROL_BIAS)
        }
        control = shifts.get(CONTROL_BIAS)
        for bias, shift in sorted(shifts.items()):
            pc = placebo.get(bias, {})
            row = {
                "judge_model": model,
                "bias_type": bias,
                "score_field": shift["score_field"],
                "n": shift["n"],
                "mean_original": shift["mean_original"],
                "mean_biased": shift["mean_biased"],
                # --- headline: the raw shift against the unbiased baseline ---
                "mean_shift": shift["mean_shift"],
                "ci_low": shift["ci_low"],
                "ci_high": shift["ci_high"],
                "sir": shift["sir"],
                "asc": shift["asc"],
                "p_value": shift["p_value"],
                # --- robustness: the same cell contrasted with the placebo arm ---
                "n_vs_placebo": pc.get("n"),
                "shift_vs_placebo": pc.get("mean_contrast"),
                "placebo_ci_low": pc.get("ci_low"),
                "placebo_ci_high": pc.get("ci_high"),
                "placebo_p_value": pc.get("p_value"),
            }
            # --- the equivalence bound the placebo arm exists to provide ---
            if control and bias != CONTROL_BIAS:
                lo, hi = control["ci_low"], control["ci_high"]
                row["control_shift"] = control["mean_shift"]
                row["control_ci_low"] = lo
                row["control_ci_high"] = hi
                row["inside_placebo_bound"] = (
                    None if lo is None or hi is None
                    else bool(lo <= shift["mean_shift"] <= hi)
                )
            rows.append(row)

    _single_score_field(rows, "claim A")
    # The control is not a hypothesis; correcting it alongside them would inflate the
    # family by one test and make the bound itself subject to the bound.
    attach_bh(
        [r for r in rows if r["bias_type"] != CONTROL_BIAS],
        family="claim_A", p_key="p_value",
    )
    for row in rows:
        row.setdefault("family", "control")
        row.setdefault("q_value", None)
        row.setdefault("significant_bh", None)
    return rows


# --------------------------------------------------------------------------- #
# claim B — validity                                                          #
# --------------------------------------------------------------------------- #
def build_claim_b(
    samples: Sequence[SampleRecord],
    results_dir: Path,
    *,
    biases: Sequence[str] = ANCHOR_BIASES,
    n_boot: int = 1000,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    """Agreement before vs after, per (judge, anchor source, bias).

    Filtered on `metadata.anchor_source` and NEVER on `subset_block`: 39 of the 624
    anchor samples were also drawn into the breadth block, and filtering by block
    would silently drop them from the source they belong to.
    """
    wanted = set(biases)
    by_source: Dict[str, List[SampleRecord]] = {}
    for s in samples:
        source = getattr(s.metadata, "anchor_source", None)
        if source:
            by_source.setdefault(source, []).append(s)

    rows: List[dict] = []
    for source, subset in sorted(by_source.items()):
        ids = {s.sample_id for s in subset}
        for model in _judges(results_dir, "scoring", roster):
            originals = [
                r for r in _read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl")
                if r.sample_id in ids
            ]
            biased = [
                r for r in _read_judge(results_dir / "biased_judgments" / f"scoring__{model}.jsonl")
                if r.sample_id in ids and r.bias_type in wanted
            ]
            if not biased:
                continue
            for st in compute_agreement_grid(subset, originals, biased, n_boot=n_boot):
                row = st.as_row()
                row["anchor_source"] = source
                # The rank half is tested by its cluster-bootstrap interval, not by a
                # p-value, so it gets a flag and is NOT BH-corrected below. Read it
                # with that in mind: 40 cells at 95% expect ~2 spurious exclusions of
                # zero on their own, which is why the claim rests on region_annotation
                # appearing in 8 of its 10 cells rather than on any single one.
                lo, hi = row["spearman_delta_ci_low"], row["spearman_delta_ci_high"]
                row["rho_ci_excludes_zero"] = (
                    None if lo is None or hi is None else bool(lo > 0 or hi < 0)
                )
                rows.append(row)

    # Reorder so the identifying columns lead.
    lead = ["judge_model", "anchor_source", "bias_type"]
    rows = [{**{k: r[k] for k in lead}, **r} for r in rows]
    # ★ BH runs on the CLUSTER-level p, not on the pair-level McNemar. The derived pairs are
    # within-turn comparisons (8 editors -> ~18 comparisons, each item in up to 7 of them), so
    # the McNemar's independence assumption fails by a measured design effect of up to 4.3.
    # The tell that this was not hypothetical: under the old key, 10 of the 13 BH-significant
    # cells had an `accuracy_delta` CI — computed by a correctly clustered bootstrap in the
    # very same row — that straddled zero.
    attach_bh(rows, family="claim_B_acc", p_key="accuracy_p_cluster")
    return rows


# --------------------------------------------------------------------------- #
# pairwise                                                                     #
# --------------------------------------------------------------------------- #
def build_pairwise(
    results_dir: Path, *, decisive_prefs: Optional[Dict[str, str]] = None,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> Dict[str, List[dict]]:
    """Position robustness, slot stickiness, and the one-sided grid."""
    position: List[dict] = []
    joint: List[dict] = []
    one_sided: List[dict] = []
    consistency: List[dict] = []
    for model in _judges(results_dir, "pairwise", roster):
        originals = _read_judge(results_dir / "raw_judgments" / f"pairwise__{model}.jsonl")
        biased = _read_judge(results_dir / "biased_judgments" / f"pairwise__{model}.jsonl")
        position += [st.as_row() for st in compute_position_flips(originals, biased)]
        joint += [st.as_row() for st in position_joint(originals, biased)]
        consistency += [st.as_row() for st in compute_consistency(originals)]
        one_sided += [
            st.as_row()
            for st in compute_one_sided_bias(originals, biased, decisive_prefs=decisive_prefs)
        ]

    # Two families in one file: "did the verdict move toward the cue" and "did that
    # cost agreement with the humans". They are different claims and are corrected
    # apart; the accuracy family uses the cue-on-winner arm, since the pooled delta is
    # null by construction (see compute_one_sided_bias).
    attach_bh(one_sided, family="pairwise_one_sided_advantage", p_key="mcnemar_p")
    q_adv = {(r["judge_model"], r["bias_type"]): (r["q_value"], r["significant_bh"])
             for r in one_sided}
    attach_bh(one_sided, family="pairwise_one_sided_accuracy", p_key="acc_p_cue_on_winner")
    for r in one_sided:
        q, sig = q_adv[(r["judge_model"], r["bias_type"])]
        r["family"] = "pairwise_one_sided_advantage + _accuracy"
        r["q_value_advantage"], r["significant_bh_advantage"] = q, sig
        r["q_value_accuracy"] = r.pop("q_value")
        r["significant_bh_accuracy"] = r.pop("significant_bh")
    return {
        "position": position, "position_joint": joint,
        "one_sided": one_sided, "consistency": consistency,
    }


# --------------------------------------------------------------------------- #
# retest                                                                       #
# --------------------------------------------------------------------------- #
def build_retest(results_dir: Path,
                 roster: Optional[Sequence[str]] = PUBLISHED_ROSTER) -> List[dict]:
    rows: List[dict] = []
    for model in _judges(results_dir, "scoring", roster):
        results = _read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl")
        rows += [st.as_row() for st in compute_retest_stats(results)]
    _single_score_field(rows, "retest")
    return rows


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def build_all(
    results_dir: PathLike,
    *,
    samples_path: PathLike,
    subset_filter: Optional[dict] = None,
    full_samples: PathLike | None = None,
    full_pairs: PathLike | None = None,
    judged_pairs: PathLike | None = None,
    out_dir: PathLike | None = None,
    n_boot: int = 1000,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
    prefix: str = "",
    fill_dir: PathLike | None = None,
) -> Dict[str, List[dict]]:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "metrics"
    samples = io.read_jsonl(Path(samples_path), SampleRecord)

    keep_ids = (
        sample_ids_matching(samples_path, subset_filter) if subset_filter else None
    )

    decisive = None
    if full_samples and full_pairs:
        # ⚠️ The FULL pool, never the judging subset — see decisive_human_prefs.
        decisive = decisive_human_prefs(
            io.read_jsonl(Path(full_samples), SampleRecord),
            io.read_jsonl(Path(full_pairs), PairRecord),
        )
        if judged_pairs:
            judged = {p.pair_id for p in io.read_jsonl(Path(judged_pairs), PairRecord)}
            decisive = {k: v for k, v in decisive.items() if k in judged}

    tables = {
        "claim_a": build_claim_a(results_dir, keep_ids, roster=roster),
        "claim_b": build_claim_b(samples, results_dir, n_boot=n_boot, roster=roster),
        "retest": build_retest(results_dir, roster=roster),
    }
    tables.update(build_pairwise(results_dir, decisive_prefs=decisive, roster=roster))

    # FILL v2 (collected 2026-09-13/14 into results/v2_fill) goes INTO the two files that
    # hold the same kind of cell -- user decision, 2026-09-15: one table per estimand, no
    # side files.  The merge moves nothing in the main grid's rows: each gains only
    # `reference` / `collection`, and the fill keeps its own BH families (`... [fill]`),
    # so no main-grid q-value is re-corrected.  It happens here, in the one writer of
    # these files, so that no partial rebuild can leave them without their fill rows.
    if fill_dir is not None:
        # Imported here because build_fill_tables imports this module.
        from edit_judge_bias.experiments.build_fill_tables import (
            build_collection_drift,
            build_fill_claim_b,
            build_fill_pairwise,
        )
        fill_dir = Path(fill_dir)
        if not fill_dir.exists():
            raise FileNotFoundError(
                f"{fill_dir} does not exist; this tree's claim tables carry its rows")
        main = {"reference": BASELINE_LABEL, "collection": MAIN_COLLECTION}
        tables["claim_b"] = (
            [with_provenance(r, **main) for r in tables["claim_b"]]
            + build_fill_claim_b(samples, results_dir, fill_dir, n_boot=n_boot, roster=roster))
        tables["one_sided"] = (
            [with_provenance(r, **main) for r in tables["one_sided"]]
            + build_fill_pairwise(results_dir, fill_dir, decisive_prefs=decisive, roster=roster))
        tables["fill_collection_drift"] = build_collection_drift(
            samples, results_dir, fill_dir, decisive_prefs=decisive, roster=roster)

    # `prefix` keeps a non-published roster in its OWN files.  Writing a second roster's
    # rows over `claim_a.csv` would be the very corruption the roster exists to prevent,
    # just with an extra step.
    #
    # ...and the FAMILY STAMP has to say so too.  The `family` column exists so a reader
    # can tell which multiple-comparison correction a q belongs to; before 2026-08-19 both
    # `claim_a.csv` and `a5_claim_a.csv` stamped the bare string `claim_A`, so two
    # INDEPENDENT corrections carried one label and the column answered "which test is
    # this" instead of "which correction is this".  Separate files make it fine in
    # practice and wrong the moment anyone concatenates them -- which is exactly the
    # operation the a5 arm's declared-independence rule forbids.  One place, applied after
    # every builder, so no call site can be missed.
    if prefix:
        tag = f" [{prefix.rstrip('_')} roster]"
        for rows in tables.values():
            for row in rows:
                fam = row.get("family")
                if fam and not fam.endswith(tag):
                    row["family"] = fam + tag

    write_csv(out_dir / f"{prefix}claim_a.csv", tables["claim_a"])
    write_csv(out_dir / f"{prefix}claim_b.csv", tables["claim_b"])
    write_csv(out_dir / f"{prefix}pairwise_one_sided.csv", tables["one_sided"])
    write_csv(out_dir / f"{prefix}position.csv", tables["position"])
    write_csv(out_dir / f"{prefix}position_joint.csv", tables["position_joint"])
    write_csv(out_dir / f"{prefix}pairwise_consistency.csv", tables["consistency"])
    write_csv(out_dir / f"{prefix}retest.csv", tables["retest"])
    if "fill_collection_drift" in tables:
        write_csv(out_dir / f"{prefix}fill_collection_drift.csv", tables["fill_collection_drift"])
    return tables


def _print_summary(tables: Dict[str, List[dict]], decisive_n: Optional[int]) -> None:
    a = tables["claim_a"]
    real = [r for r in a if r["bias_type"] != CONTROL_BIAS and r["significant_bh"]]
    print(f"=== claim A: {len(a)} cells, {len(real)} significant after BH ===")
    for r in sorted(a, key=lambda r: (r["judge_model"], r["bias_type"])):
        flag = "*" if r.get("significant_bh") else ("." if r["bias_type"] != CONTROL_BIAS else "C")
        bound = "" if r.get("inside_placebo_bound") is not True else "  [inside placebo bound]"
        print(f"  {flag} {r['judge_model']:<18}{r['bias_type']:<19}n={r['n']:<5}"
              f"shift={r['mean_shift']:>7}  vs_placebo={r['shift_vs_placebo']!s:>7}"
              f"  q={r['q_value']}{bound}")

    b = tables["claim_b"]
    reorder = [r for r in b if r.get("rho_ci_excludes_zero")]
    print(f"\n=== claim B: {len(b)} cells, {len(reorder)} with a rho CI excluding 0 ===")
    for r in reorder:
        print(f"  {r['judge_model']:<18}{r['anchor_source']:<12}{r['bias_type']:<19}"
              f"n={r['n_paired']:<5}drho={r['spearman_delta']:>8}  "
              f"dacc={r['accuracy_delta']!s:>8}  McN q={r['q_value']}")

    print(f"\n=== pairwise one-sided ({len(tables['one_sided'])} cells, "
          f"{decisive_n} decisive human votes) ===")
    for r in tables["one_sided"]:
        print(f"  {r['judge_model']:<18}{r['bias_type']:<14}"
              f"adv={r['bias_advantage']:>8}  q={r['q_value_advantage']!s:>10}  "
              f"dacc|cue-on-winner={r['acc_delta_cue_on_winner']!s:>8} "
              f"(n={r['n_cue_on_winner']}, q={r['q_value_accuracy']})")

    print(f"\n=== position / retest ===")
    joint = {r["judge_model"]: r for r in tables["position_joint"]}
    cr = {r["judge_model"]: r for r in tables["consistency"]}
    for r in tables["position"]:
        model = r["judge_model"]
        same = joint.get(model, {}).get("same_slot_rate")
        c = cr.get(model)
        # §8.2's reading table: RR alone does not have an interpretation.
        verdict = ""
        if c:
            hi_cr, hi_rr = c["cr"] >= 0.8, r["rr"] >= 0.8
            verdict = {
                (True, True): "robust",
                (True, False): "STABLE but slot-manipulable",
                (False, False): "noisy — read the bias claim with care",
                (False, True): "rare: check the data or the parser",
            }[(hi_cr, hi_rr)]
        print(f"  {model:<18}RR={r['rr']:<8}strict={r['strict_flip_rate']:<8}"
              f"same_slot={str(same):<8}CR={str(c['cr']) if c else '(not run)':<8}{verdict}")
    for r in tables["retest"]:
        print(f"  {r['judge_model']:<18}n={r['n']:<5}|d|={r['mean_abs_delta']:<7}"
              f"identical={r['identical_rate']:<8}drift={r['mean_delta']:>7} p={r['p_value']}")


def main(argv: list[str] | None = None) -> int:
    root = default_root()
    manifests = root / "data" / "manifests"
    ap = argparse.ArgumentParser(description="Freeze the claim tables into CSVs.")
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--samples", type=Path, default=manifests / "samples_judge_v2.jsonl")
    ap.add_argument("--subset-filter", action="append", default=None, metavar="KEY=VALUE",
                    help="restrict CLAIM A to one block, e.g. subset_block=breadth. "
                         "Claim B always filters on anchor_source instead.")
    ap.add_argument("--full-samples", type=Path, default=manifests / "samples_full_v2.jsonl",
                    help="the FULL pool; the decisive-pair threshold is measured on it")
    ap.add_argument("--full-pairs", type=Path, default=manifests / "pairs_full_v2.jsonl")
    ap.add_argument("--judged-pairs", type=Path, default=manifests / "pairs_judge_v2.jsonl")
    ap.add_argument("--fill-dir", type=Path, default=None,
                    help="the FILL v2 tree whose rows claim_b.csv and pairwise_one_sided.csv "
                         "carry (results/v2_fill; required for results/v2)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="cluster-bootstrap resamples for claim B (lower = faster draft)")
    ap.add_argument("--roster", choices=sorted(ROSTERS), default="published",
                    help="WHICH JUDGES FORM THIS TABLE'S BH FAMILY. `published` is the "
                         "five whose numbers are in the paper and must stay byte-stable; "
                         "`a5` is the open-weights replication, reported as its own "
                         "declared family (user decision, 2026-08-18) and written to "
                         "a5_*.csv; `all` is for exploration and must never be quoted.")
    args = ap.parse_args(argv)
    if args.fill_dir is None and args.results_dir.resolve() == (root / "results" / "v2").resolve():
        ap.error("results/v2's claim_b.csv and pairwise_one_sided.csv carry the FILL v2 rows "
                 "(merged 2026-09-15); pass --fill-dir results/v2_fill. Rebuilding without "
                 "them would silently drop cells that main_table.csv counts.")
    if args.fill_dir is not None and not args.fill_dir.exists():
        ap.error(f"{args.fill_dir} does not exist")

    flt = None
    if args.subset_filter:
        flt = {}
        for item in args.subset_filter:
            key, _, value = item.partition("=")
            if not _:
                ap.error(f"--subset-filter expects KEY=VALUE, got {item!r}")
            flt[key] = value
        n = len(sample_ids_matching(args.samples, flt))
        if not n:
            ap.error(f"no samples in {args.samples} match {flt}")
        print(f"[subset] claim A restricted to {flt} -> {n} samples")
    else:
        # Loud, because the failure is silent: the manifest holds three blocks judged
        # under different numbers of conditions, and pooling them changes a claim A
        # number without changing the column heading. MEASURED: gpt-5.5's padding cell
        # moved from n=611 / -1.75 to n=1,196 / -1.04 the moment the anchor arm landed.
        blocks = {
            getattr(s.metadata, "subset_block", None)
            for s in io.read_jsonl(args.samples, SampleRecord)
        }
        if len(blocks) > 1:
            print(f"[WARNING] {args.samples.name} holds blocks {sorted(map(str, blocks))} "
                  "and no --subset-filter was given: claim A will POOL them. "
                  "Pass --subset-filter subset_block=breadth for the headline table.")

    have_pairs = args.full_samples.exists() and args.full_pairs.exists()
    tables = build_all(
        args.results_dir,
        samples_path=args.samples,
        subset_filter=flt,
        full_samples=args.full_samples if have_pairs else None,
        full_pairs=args.full_pairs if have_pairs else None,
        judged_pairs=args.judged_pairs if have_pairs else None,
        out_dir=args.out_dir,
        n_boot=args.n_boot,
        roster=ROSTERS[args.roster],
        prefix="" if args.roster == "published" else f"{args.roster}_",
        fill_dir=args.fill_dir,
    )
    decisive_n = max((r["n_decisive"] for r in tables["one_sided"]), default=0)
    _print_summary(tables, decisive_n)
    print(f"\nwrote 6 tables -> {args.out_dir or (args.results_dir / 'metrics')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
