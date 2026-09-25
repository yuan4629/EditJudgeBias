"""Aggregate judge result manifests into metric tables (Milestone 5).

Discovers judge models from results/raw_judgments/, computes scoring score-shift
stats and pairwise position-flip stats, and writes CSV summaries under
results/metrics/. Pure stdlib + the metrics layer; no plotting here (see
visualization/ and scripts/07_plot_figures.sh).

    python -m edit_judge_bias.experiments.aggregate_results --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root
from edit_judge_bias.data.schema import JudgeResult, QualityValidationResult, SampleRecord
from edit_judge_bias.experiments.run_scoring_judge import matches_filter
from edit_judge_bias.metrics.pairwise_metrics import compute_position_flips
from edit_judge_bias.metrics.scoring_metrics import compute_score_shifts


def _read(path: Path) -> List[JudgeResult]:
    return io.read_jsonl(path, JudgeResult) if path.exists() else []


def sample_ids_matching(samples_path: PathLike, flt: dict) -> set:
    """The sample_ids of one block of the judging subset.

    WHY this exists. The subset is one manifest holding three blocks, and each arm
    judges a different one with a different number of conditions: 14 for the balanced
    breadth block, 5 for the two human anchors. The result manifests are per judge,
    not per arm, so once the anchor arm has run, a bias that both arms cover has rows
    from both blocks in the same file. MEASURED: gpt-5.5's `padding` cell went from
    n=611 (breadth) to n=1,196 and its mean_shift from -1.75 to -1.04 the moment the
    anchor arm landed — the anchor blocks are single-source and deliberately
    unbalanced, so pooling them into claim A's cross-source table changes the number
    without changing the column heading. Claim A filters `subset_block == breadth`.
    """
    samples = io.read_jsonl(Path(samples_path), SampleRecord)
    return {s.sample_id for s in samples if matches_filter(s, flt)}


# A judge whose result file holds less than this share of the leading judge's rows is a
# COLLECTION IN PROGRESS, not a judge.  See `_discover_models`.
MIN_JUDGE_COVERAGE = 0.5
PARTIAL_JUDGE_LOG = "partial_judges_excluded.json"


def _emit(msg: str) -> None:
    """Print a warning that survives a non-UTF-8 console.

    ⚠️ MEASURED 2026-08-18, THE FIRST TIME THE GUARD BELOW EVER FIRED.  This project's
    Windows console is GBK; the house warning marker is not encodable in it, so
    `scripts/05_compute_metrics.sh` died with a UnicodeEncodeError *inside the warning* --
    i.e. the guard failed loudest at exactly the moment it had something important to say,
    and it had gone unexercised since it was written because no partial judge had existed.

    Degrading the marker is the right trade: the message is the payload, the glyph is
    decoration, and `metrics/partial_judges_excluded.json` is the durable record either way.
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))


def _discover_models(results_dir: Path, task: str, *,
                     include_partial: bool = False) -> List[str]:
    """Judges on disk, EXCLUDING any whose collection is still in progress.

    ⚠️ WHY THE FILTER EXISTS -- this is a headline-moving silent corruption, caught
    2026-08-17 while designing WP-A5b's calibration batch.

    Discovery is by glob, so a judge appears in every table the moment its FIRST row
    lands.  The A5b plan is to calibrate a new judge's per-call price on the 84-call smoke
    arm, which is a deterministic subset of the breadth arm and therefore writes into the
    published tree on purpose (so the real run resumes over it and nothing is paid twice).
    Rebuilding the tables at that moment would add a 6th judge with n≈6 per cell -- and
    because BH runs per declared family, claim A's family would go from 15 cells to 18 and
    **every published q-value would change**.  Nothing would error; the numbers would just
    quietly stop matching the paper.

    The semantics that make staged collection safe: a judge does not enter a published
    table until it is (nearly) complete, so the five published judges' tables are
    byte-stable across the whole of A5b's collection.  Excluding is therefore right and
    raising is not -- raising would block every rebuild for days while a new judge fills.
    But an exclusion must never be silent, so it is printed AND written to
    `metrics/partial_judges_excluded.json`.

    ⚠️ IT LIVES HERE, NOT IN `build_claim_tables`, BECAUSE THE GUARD HAD A SECOND DOOR.
    It was written for the claim tables and only for them; `aggregate_results` globbed the
    same directory unguarded, so `scoring_shift.csv` and `scoring_shift_qc.csv` -- the
    legacy pair that the write-up still quotes QC-filtered effect sizes from -- would have
    grown two n=6 judge rows the moment the calibration batch landed, and the frozen-table
    acceptance check ("33 CSVs unchanged after a full rebuild") would have failed on tables
    nobody was watching.  Caught 2026-08-17 before the calibration was paid for.
    `build_analysis_tables` was checked at the same time and is safe by a different
    mechanism: its two builders filter against a hardcoded five-name `ROSTER`.

    ⚠️ AND A THIRD DOOR, FOUND 2026-08-18 -- BY IMPORTING THIS FUNCTION, NOT BY SKIPPING
    IT.  `build_mitigation_tables.build_swap_average` called it directly, which made the
    2026-08-17 audit read it as guarded.  But coverage answers "is this judge finished?",
    and mid-collection `qwen3-vl-32b-instruct` answered yes at 578 of 616 base-order
    pairs: it entered `mitigation_swap_average.csv`, took the BH family from 20 rows to
    25, and moved five PUBLISHED q-values while every conclusion stayed the same.  The
    lesson generalises past this codebase: **a call site that uses the guard is not
    automatically a call site the guard protects** -- you have to ask whether the
    question it answers is the question that site needs answered.  That builder now goes
    through `_judges(..., roster)` like every other published table.
    """
    raw = results_dir / "raw_judgments"
    prefix = f"{task}__"
    found = {p.stem[len(prefix):]: sum(1 for _ in p.open(encoding="utf-8"))
             for p in raw.glob(f"{prefix}*.jsonl")}
    if not found or include_partial:
        return sorted(found)
    lead = max(found.values())
    keep = sorted(m for m, n in found.items() if n >= MIN_JUDGE_COVERAGE * lead)
    dropped = {m: n for m, n in sorted(found.items()) if m not in keep}
    if dropped:
        _emit(f"  ⚠️ {task}: excluding {len(dropped)} judge(s) with <"
              f"{MIN_JUDGE_COVERAGE:.0%} of the leading judge's {lead} rows "
              f"(collection in progress): "
              + ", ".join(f"{m} n={n}" for m, n in dropped.items()))
    # WP-F4/P7.  Written on EVERY pass, not only when something was excluded.  Keyed by
    # task and overwritten in place, the old conditional meant an exclusion recorded while
    # an arm was mid-collection stayed in the file forever once that arm finished -- the
    # record would say a judge is being held out long after it had been admitted, and the
    # only way to notice was to re-derive the row counts by hand.  An empty `excluded` is
    # a positive statement ("checked this task, held nobody out"), which is the thing a
    # reader actually needs; absence of a key now means only that the task was not built.
    log = results_dir / "metrics" / PARTIAL_JUDGE_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
    prior[task] = {"leading_rows": lead, "threshold": MIN_JUDGE_COVERAGE,
                   "excluded": dropped, "included": keep}
    log.write_text(json.dumps(prior, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    return keep


#: The five judges the paper's claims are declared over.  Moved here 2026-08-18 from
#: `build_claim_tables` so that "who counts as a judge" has ONE home: this module already
#: owns `_discover_models`, and the legacy tables below could not import the roster from a
#: module that imports *them* without a cycle.  `build_claim_tables` re-exports these names,
#: so every existing import keeps working.
PUBLISHED_ROSTER: Tuple[str, ...] = (
    "gpt-5.5",
    "gemini-3.5-flash",
    "gpt-4o-viescore",
    "qwen3.5-plus",
    "kimi-k2.5",
)

#: WP-A5's open-weights replication, reported as a separate family.
#:
#: ⚠️ ONE NAME, NOT TWO.  `llama-4-scout` was the other A5 candidate and was DROPPED by the
#: user on 2026-08-18 after the calibration batch (see `CANDIDATE_JUDGES`).  It sat in this
#: tuple until 2026-08-19 doing nothing visible, because the 84 rows of its calibration
#: batch are held out by `_discover_models`' coverage guard.  That is exactly the shape of
#: the fourth-door defect documented in `_judges`: the guard answers "has this judge
#: finished?", so the moment anyone ran the full arm the name would have passed the guard
#: honestly, walked into the a5 tables, taken the BH family from 13 cells to 26, rewritten
#: every published a5 q-value -- and `--roster a5` would not have raised a thing.
#: A roster is a DECLARATION; a name that the paper does not declare does not belong in it.
A5_ROSTER: Tuple[str, ...] = (
    "qwen3-vl-32b-instruct",
)

#: Judges that were evaluated and NOT adopted.  Deliberately not a roster and deliberately
#: not in `ROSTERS`: nothing may build a table from this tuple.  It exists so that deleting
#: a name from a roster does not also delete the record of why.
CANDIDATE_JUDGES: Dict[str, str] = {
    "llama-4-scout": (
        "A5 candidate, dropped 2026-08-18 after the 84-call calibration batch: 54% of its "
        "scores sat on the scale ceiling (so it cannot register inflation at all, and "
        "`bandwagon` -- one of claim A's two headline directions -- pinned at 30.00), and "
        "2.4% (2/84) of its calls returned a well-formed, successfully-parsed verdict whose "
        "reason was 'Edited image not provided.'  Those rows score the floor, enter "
        "`mean_shift` at -27, and every pipeline check passes them.  See RESULTS.md 20."
    ),
}

ROSTERS: Dict[str, Optional[Tuple[str, ...]]] = {
    "published": PUBLISHED_ROSTER,
    "a5": A5_ROSTER,
    "all": None,          # every complete judge on disk; for exploration, never for a paper table
}


def _judges(results_dir: Path, task: str,
            roster: Optional[Sequence[str]],
            *, include_partial: bool = False) -> List[str]:
    """Complete judges on disk, restricted to a DECLARED roster.

    Two filters, and they answer different questions.  `_discover_models` asks "is this
    judge finished?" and exists so a table stays byte-stable during collection.  The
    roster asks "is this judge in this table's declared family?" and exists because that
    is an editorial decision the code must not make by itself.

    ⚠️ FOURTH DOOR, FOUND 2026-08-18 BY REBUILDING RATHER THAN BY READING.  The legacy
    tables written by `aggregate` used `_discover_models` alone.  While WP-A5 was still
    collecting, the coverage filter hid the problem; the moment that arm FINISHED, the
    replication judge passed the coverage test honestly and walked into
    `scoring_shift.csv`, `scoring_shift_qc.csv`, `pilot_summary.csv` and
    `pairwise_position.csv` -- tables the paper describes as a five-judge panel.  The
    acceptance check that was supposed to catch this ("N CSVs unchanged after a full
    rebuild") had been RUN AT A MOMENT WHEN THE COVERAGE GUARD WAS STILL MASKING IT, so it
    passed for a reason that expired a few hours later.  ⇒ A verification is only as good
    as the state it was taken in; re-run the rebuild check after the thing you were
    collecting is done, not while it is running.
    """
    found = _discover_models(results_dir, task, include_partial=include_partial)
    if roster is None:
        return found
    keep = [m for m in found if m in set(roster)]
    missing = [m for m in roster if m not in found]
    if missing:
        print(f"  note: roster names {missing} which is not (yet) complete on disk")
    if found and not keep:
        # An empty table is the quietest possible wrong answer: every downstream check
        # reads "no rows" as "nothing to report" rather than "everything was filtered" --
        # the same shape as `build_claim_b_permutation` printing a neutral sentence while
        # it emptied a published headline table.
        #
        # ⚠️ But this WARNS rather than raises, and the reason is the same one that made
        # the coverage filter exclude instead of raise: "this roster has not been collected
        # in this tree yet" is a legitimate state (building the a5_* tables before the A5
        # arm has run), and it is indistinguishable here from a mistyped roster. Raising
        # would block that legitimate path; a silent empty table would hide the mistake.
        # So: say it loudly, name both sides, and let the caller see an empty table it was
        # told about.
        _emit(f"  ⚠️ {task}: roster {sorted(roster)} matched NONE of the judges on disk "
              f"({found}) — this table will be EMPTY. Use roster=None / --roster all to "
              "aggregate whatever is present.")
    return keep


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Union of keys (rows may be heterogeneous, e.g. scoring + pairwise summary).
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _passed_biased_ids(
    results_dir: Path, validators: Optional[Iterable[str]] = None
) -> Optional[set]:
    """biased_ids that passed quality validation under EVERY validator counted.

    INTERSECTION, not union. The QC subset means "images a validator could not tell
    apart from their unperturbed originals", so a second validator can only ever
    *remove* an image from it. Unioning would make the subset GROW as validators are
    added — an image one model flagged would be readmitted by another — which is the
    wrong direction for a gate whose whole job is to be strict.

    `validators` names the model files to count, without the `validation__` prefix or
    the `.jsonl` suffix; None counts every manifest present. Name them explicitly when
    a validator is on disk for cross-checking but is not calibrated enough to gate on:
    measured 2026-07-29, glm-4v passes a visually null `sham` re-encode only 78.6% of
    the time, so intersecting with it would discard ~1 image in 5 at random.

    The `validation__` prefix is load-bearing: an earlier arm wrote
    `validation_full_v2__...`, which this glob does not match, and the v2 QC table was
    silently computed against the PILOT's passed ids.
    """
    quality_dir = results_dir / "quality"
    if not quality_dir.is_dir():
        return None
    wanted = set(validators) if validators is not None else None
    passed: Optional[set] = None
    for path in sorted(quality_dir.glob("validation__*.jsonl")):
        model = path.stem[len("validation__"):]
        if wanted is not None and model not in wanted:
            continue
        here = {
            r.biased_id for r in io.iter_jsonl(path, QualityValidationResult)
            if r.parse_success and r.passed and r.biased_id
        }
        passed = here if passed is None else (passed & here)
    return passed


def aggregate(
    results_dir: PathLike,
    out_dir: PathLike | None = None,
    *,
    keep_sample_ids: Optional[set] = None,
    validators: Optional[Iterable[str]] = None,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
    include_partial: bool = False,
) -> dict:
    """Metric tables from the result manifests.

    `keep_sample_ids` restricts every scoring row to one block of the judging subset
    (see :func:`sample_ids_matching`); None uses whatever is on disk. `validators`
    names which quality validators gate the QC table (see :func:`_passed_biased_ids`).
    `roster` is the declared family these legacy tables report; None means every complete
    judge on disk, which is an exploration mode and never a paper table.

    `include_partial` switches OFF the coverage guard, and exists for exactly one tree.
    The 1-5 pilot archive under `results/` is deliberately ragged -- `gpt-5.4-nano` ran a
    96-sample mini batch against everyone else's 480 -- so 50%-of-the-leader drops it, and
    the three TRACKED pilot tables stopped being reproducible by any documented command
    the day that guard landed (2026-08-18, for the v2 tree's benefit). On the pilot tree
    the ragged roster IS the declared family; on `results/v2` this flag would readmit
    every half-collected judge and is never correct. `scripts/05_compute_metrics.sh`
    passes it on the pilot branch only, and a test pins that.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "metrics"
    passed_ids = _passed_biased_ids(results_dir, validators)

    def keep(results: List[JudgeResult]) -> List[JudgeResult]:
        if keep_sample_ids is None:
            return results
        return [r for r in results if r.sample_id in keep_sample_ids]

    scoring_rows: List[dict] = []
    qc_rows: List[dict] = []
    for model in _judges(results_dir, "scoring", roster,
                         include_partial=include_partial):
        originals = keep(_read(results_dir / "raw_judgments" / f"scoring__{model}.jsonl"))
        biased = keep(_read(results_dir / "biased_judgments" / f"scoring__{model}.jsonl"))
        for st in compute_score_shifts(originals, biased):
            scoring_rows.append(st.as_row())
        if passed_ids is not None:
            for st in compute_score_shifts(originals, biased, include_biased_ids=passed_ids):
                qc_rows.append(st.as_row())

    # One table, one analysis variable. The shifts are resolved per model (each judge
    # is joined against its own baselines), so a judge whose answers were missing a
    # dimension could resolve to `overall_score` while the rest stayed on `fine_score`
    # — a `mean_shift` column holding both 3-30 sums and 1-10 overalls, in which the
    # odd judge reads as an order of magnitude more robust. That happened.
    fields = {r["score_field"] for r in scoring_rows if r.get("score_field")}
    if len(fields) > 1:
        raise ValueError(
            f"scoring rows mix analysis variables {sorted(fields)}; "
            "mean_shift would not be on one scale across judges"
        )

    pairwise_rows: List[dict] = []
    for model in _judges(results_dir, "pairwise", roster,
                         include_partial=include_partial):
        originals = _read(results_dir / "raw_judgments" / f"pairwise__{model}.jsonl")
        position = _read(results_dir / "biased_judgments" / f"pairwise__{model}.jsonl")
        for st in compute_position_flips(originals, position):
            pairwise_rows.append(st.as_row())

    _write_csv(out_dir / "scoring_shift.csv", scoring_rows)
    _write_csv(out_dir / "pairwise_position.csv", pairwise_rows)
    _write_csv(out_dir / "pilot_summary.csv", scoring_rows + _summary_padding(pairwise_rows))
    if qc_rows:
        _write_csv(out_dir / "scoring_shift_qc.csv", qc_rows)
    return {"scoring": scoring_rows, "pairwise": pairwise_rows, "qc": qc_rows}


def _summary_padding(pairwise_rows: List[dict]) -> List[dict]:
    """Fold pairwise rows into a scoring-shaped summary (sparse, key fields only)."""
    folded = []
    for r in pairwise_rows:
        folded.append({
            "judge_model": r["judge_model"],
            "bias_type": r["bias_type"],
            "n": r["n"],
            "mean_shift": "",
            "sir": "",
            "rr": r.get("rr", ""),
            "strict_flip_rate": r.get("strict_flip_rate", ""),
            "p_value": r.get("p_value", ""),
        })
    return folded


def _print_summary(stats: dict) -> None:
    print("=== scoring score-shift (mean_shift, SIR, p) ===")
    print(f"{'model':<22} {'bias':<13} {'n':>4} {'mean_shift':>10} {'SIR':>6} {'p':>9}")
    for r in stats["scoring"]:
        p = "" if r["p_value"] is None else f"{r['p_value']:.4f}"
        print(f"{r['judge_model']:<22} {r['bias_type']:<13} {r['n']:>4} "
              f"{r['mean_shift']:>10} {r['sir']:>6} {p:>9}")
    if stats["pairwise"]:
        print("\n=== pairwise position (RR, strict_flip, posA) ===")
        for r in stats["pairwise"]:
            print(f"{r['judge_model']:<22} RR={r['rr']} strict_flip={r['strict_flip_rate']} "
                  f"posA_orig={r['posA_rate_original']} posA_swap={r['posA_rate_swapped']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate judge results into metric CSVs.")
    ap.add_argument("--results-dir", type=Path, default=default_root() / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--samples", type=Path, default=None,
                    help="judging-subset manifest; required by --subset-filter")
    ap.add_argument("--subset-filter", action="append", default=None, metavar="KEY=VALUE",
                    help="restrict scoring rows to one block, e.g. subset_block=breadth "
                         "(repeatable). Without it the table pools every block the "
                         "judges have answered — see sample_ids_matching().")
    ap.add_argument("--validators", action="append", default=None, metavar="MODEL",
                    help="which quality validators gate the QC table, by model name "
                         "(repeatable). Default: every validator on disk, intersected. "
                         "Name them when one is present for cross-checking but is too "
                         "noisy to gate on — see _passed_biased_ids().")
    ap.add_argument("--roster", choices=sorted(ROSTERS), default="published",
                    help="which declared judge family these tables report. 'published' is "
                         "the five-judge panel the paper's claims are declared over; 'a5' "
                         "is the open-weights replication, reported separately; 'all' takes "
                         "every complete judge on disk and is for exploration only.")
    ap.add_argument("--include-partial", action="store_true",
                    help="switch OFF the 50%%-of-the-leader coverage guard. For the 1-5 "
                         "pilot archive ONLY, where the ragged roster (gpt-5.4-nano ran a "
                         "96-sample mini batch against everyone else's 480) is the declared "
                         "family. On results/v2 this readmits half-collected judges into "
                         "published tables and is never correct.")
    args = ap.parse_args(argv)
    if args.include_partial and Path(args.results_dir).name.startswith("v2"):
        ap.error("--include-partial on a v2 tree would readmit half-collected judges into "
                 "published tables; it exists for the results/ pilot archive only")

    keep_ids = None
    if args.subset_filter:
        if not args.samples:
            ap.error("--subset-filter needs --samples <manifest>")
        flt = {}
        for item in args.subset_filter:
            key, _, value = item.partition("=")
            if not _:
                ap.error(f"--subset-filter expects KEY=VALUE, got {item!r}")
            flt[key] = value
        keep_ids = sample_ids_matching(args.samples, flt)
        if not keep_ids:
            ap.error(f"no samples in {args.samples} match {flt}")
        print(f"[subset] {flt} -> {len(keep_ids)} samples")
    if args.validators:
        print(f"[QC] gated on validators {args.validators} (intersection)")

    stats = aggregate(args.results_dir, args.out_dir, keep_sample_ids=keep_ids,
                      validators=args.validators, roster=ROSTERS[args.roster],
                      include_partial=args.include_partial)
    _print_summary(stats)
    print(f"\nwrote metrics -> {args.out_dir or (args.results_dir / 'metrics')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
