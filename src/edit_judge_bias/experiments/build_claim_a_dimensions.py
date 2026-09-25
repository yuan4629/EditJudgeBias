"""Claim A decomposed into the three dimensions the judge was actually asked for.

WHY THIS TABLE EXISTS. The single question every reviewer puts to this paper is
**"is that deflation a bias, or is it a justified deduction?"**, and the summed
score cannot answer it: `fine_score` is the SUM of three dimensions
(`data/schema.py:210-212`), so a cue that genuinely degrades the picture and a cue
that merely distracts the judge both show up as one negative number.

The three dimensions are not symmetric with respect to that question, and the asymmetry
is written into the instruments themselves:

- The judge rubric asks, verbatim, for *"Detail preservation: whether irrelevant
  regions are preserved"* (`prompts/scoring_prompt.py:31`). An overlay IS a change to
  an irrelevant region. So a deduction on **DP** — and to a lesser extent on **editing
  quality** — is a defensible reading of the rubric as written, not obviously an error.
- **Instruction adherence** is different. The cue is applied AFTER the edit, it never
  touches the instruction, and the quality-preservation protocol verified that the
  instruction's effect is intact — the validator template (`prompts/validator_prompt.py:25-26`)
  and the human annotators (`data/human_validation_v2/INSTRUCTIONS.txt`) were both told
  to IGNORE the overlay. So IA is the dimension on which, BY CONSTRUCTION, the correct
  answer cannot move. A judge that lowers IA because a caption was pasted on is not
  making a defensible deduction; it is answering a different question from the one it
  was asked.

⚠️ Note what that same asymmetry costs: the validated construct is the AUTHORS'
definition of preserved quality (ignore the overlay), which is not the definition the
judge rubric states (preserve irrelevant regions). That gap is a limitation to report,
not something this table repairs. What the table does is separate the part of the
effect that survives the gap from the part that does not.

SAME SAMPLE AS CLAIM A. Every cell is computed on the SAME complete-case
subset `claim_a.csv` uses — rows where all three dimensions parsed, i.e. exactly the
rows that have a `fine_score` — so the three `mean_shift` values ADD UP to the
`claim_a.csv` headline for that cell, to the last decimal, and
`tests/test_claim_a_dimensions.py` pins the identity so it cannot quietly stop holding.
The per-dimension `asc` column is the statistic of the paper's main invariance table;
`build_noise_floor_table` reads it against the `sham` and retest noise floors.

BH: one declared family per dimension (`claim_A_instruction_adherence`, ...), each of
the same shape as `claim_A` — 12 cues x the roster, with `sham` held out as the control
it is. Three families, never one over all 195 rows: the three dimensions are three
different questions, and pooling them would correct each one for the other two.

    python -m edit_judge_bias.experiments.build_claim_a_dimensions \\
        --results-dir results/v2 \\
        --samples data/manifests/samples_judge_v2.jsonl \\
        --subset-filter subset_block=breadth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.data.schema import JudgeResult
from edit_judge_bias.experiments.aggregate_results import (
    PUBLISHED_ROSTER,
    ROSTERS,
    _judges,
    sample_ids_matching,
)
from edit_judge_bias.experiments.build_claim_tables import (
    CONTROL_BIAS,
    _read_judge,
    _single_score_field,
    attach_bh,
    write_csv,
)
from edit_judge_bias.metrics.scoring_metrics import (
    compute_placebo_contrast,
    compute_score_shifts,
)

#: The three fields every scoring answer carries, in the order the rubric asks them.
#: `fine_score` is their sum, which is why a row missing any one of them has no
#: `fine_score` and is outside claim A's sample to begin with.
DIMENSIONS = ("instruction_adherence", "editing_quality", "detail_preservation")


def _complete_case(results: Sequence[JudgeResult]) -> List[JudgeResult]:
    """The rows claim A is measured on: all three dimensions present.

    Filtering on `fine_score` rather than on the three fields separately is not a
    shortcut — it is the definition (`judges/parser.py:173` sets `fine_score` to the
    sum iff every dimension parsed). Doing it per dimension instead would give each
    dimension its own item set, and "IA moved on 5 judges but DP on 4" would then be
    partly a statement about which answers happened to parse.
    """
    return [r for r in results if r.fine_score is not None]


def build_claim_a_dimensions(
    results_dir: Path,
    keep_sample_ids: Optional[set] = None,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    """Per (judge, cue, dimension): the same estimator claim A uses, one field at a time."""
    def keep(results: List[JudgeResult]) -> List[JudgeResult]:
        if keep_sample_ids is None:
            return results
        return [r for r in results if r.sample_id in keep_sample_ids]

    rows: List[dict] = []
    for model in _judges(results_dir, "scoring", roster):
        originals = _complete_case(
            keep(_read_judge(results_dir / "raw_judgments" / f"scoring__{model}.jsonl"))
        )
        biased = _complete_case(
            keep(_read_judge(results_dir / "biased_judgments" / f"scoring__{model}.jsonl"))
        )
        if not biased:
            continue
        for dim in DIMENSIONS:
            shifts = {
                st.bias_type: st.as_row()
                for st in compute_score_shifts(originals, biased, score_field=dim)
            }
            placebo = {
                st.bias_type: st.as_row()
                for st in compute_placebo_contrast(
                    biased, control_bias=CONTROL_BIAS, score_field=dim
                )
            }
            control = shifts.get(CONTROL_BIAS)
            for bias, shift in sorted(shifts.items()):
                pc = placebo.get(bias, {})
                row = {
                    "judge_model": model,
                    "bias_type": bias,
                    "dimension": dim,
                    "score_field": shift["score_field"],
                    "n": shift["n"],
                    "mean_original": shift["mean_original"],
                    "mean_biased": shift["mean_biased"],
                    "mean_shift": shift["mean_shift"],
                    "ci_low": shift["ci_low"],
                    "ci_high": shift["ci_high"],
                    "sir": shift["sir"],
                    "asc": shift["asc"],
                    "p_value": shift["p_value"],
                    "n_vs_placebo": pc.get("n"),
                    "shift_vs_placebo": pc.get("mean_contrast"),
                    "placebo_ci_low": pc.get("ci_low"),
                    "placebo_ci_high": pc.get("ci_high"),
                    "placebo_p_value": pc.get("p_value"),
                }
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

    # One family per dimension, and the guard that a column holds one variable applied
    # per dimension too -- across dimensions `score_field` is SUPPOSED to differ, which
    # is exactly why the global version of this check cannot be used here.
    for dim in DIMENSIONS:
        in_dim = [r for r in rows if r["dimension"] == dim]
        if not in_dim:
            continue
        _single_score_field(in_dim, f"claim A ({dim})")
        attach_bh(
            [r for r in in_dim if r["bias_type"] != CONTROL_BIAS],
            family=f"claim_A_{dim}", p_key="p_value",
        )
    for row in rows:
        row.setdefault("family", "control")
        row.setdefault("q_value", None)
        row.setdefault("significant_bh", None)
    return sorted(rows, key=lambda r: (r["bias_type"], r["dimension"], r["judge_model"]))


def _print_summary(rows: List[dict]) -> None:
    """Signed judge counts per (cue, dimension) — the shape the write-up quotes.

    ⚠️ Reported as `down/up`, never as one "significant" count. Three cues in this
    table are significant in BOTH directions across the panel (`aesthetic_filter`,
    `detail_caption`, `model_name`), so a single count would be a composite rate
    mixing two behaviours — the defect this project has already published once and
    corrected ("bandwagon is the only cue that inflates" -> "the only cue that
    inflates on 5/5").
    """
    cues = sorted({r["bias_type"] for r in rows})
    judges = sorted({r["judge_model"] for r in rows})
    print(f"=== claim A by dimension ({len(judges)} judges, BH within each dimension) ===")
    print(f"{'cue':<20}" + "".join(f"{d.split('_')[0][:4]:>22}" for d in DIMENSIONS))
    for cue in cues:
        cells = ""
        for dim in DIMENSIONS:
            sel = [r for r in rows if r["bias_type"] == cue and r["dimension"] == dim]
            dn = sum(1 for r in sel if r["significant_bh"] and r["mean_shift"] < 0)
            up = sum(1 for r in sel if r["significant_bh"] and r["mean_shift"] > 0)
            lo = min((r["mean_shift"] for r in sel), default=0.0)
            hi = max((r["mean_shift"] for r in sel), default=0.0)
            cells += f"{dn}v{up}^/{len(sel)} [{lo:+.2f},{hi:+.2f}]".rjust(22)
        tag = "  (control)" if cue == CONTROL_BIAS else ""
        print(f"{cue:<20}{cells}{tag}")


def main(argv: Optional[List[str]] = None) -> int:
    root = default_root()
    ap = argparse.ArgumentParser(
        description="Freeze claim A decomposed into the three judged dimensions."
    )
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--samples", type=Path,
                    default=root / "data" / "manifests" / "samples_judge_v2.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--subset-filter", action="append", default=None, metavar="KEY=VALUE")
    ap.add_argument("--roster", choices=sorted(ROSTERS), default="published",
                    help="declared judge family; anything but `published` writes to "
                         "prefixed files so the published table cannot be overwritten")
    args = ap.parse_args(argv)

    flt: Optional[Dict[str, str]] = None
    if args.subset_filter:
        flt = dict(kv.split("=", 1) for kv in args.subset_filter)
    keep = sample_ids_matching(args.samples, flt) if flt else None

    roster = ROSTERS[args.roster]
    prefix = "" if args.roster == "published" else f"{args.roster}_"
    rows = build_claim_a_dimensions(args.results_dir, keep, roster=roster)
    out_dir = Path(args.out_dir) if args.out_dir else args.results_dir / "metrics"
    write_csv(out_dir / f"{prefix}claim_a_by_dimension.csv", rows)
    _print_summary(rows)
    print(f"\nwrote {len(rows)} rows -> {out_dir / f'{prefix}claim_a_by_dimension.csv'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
