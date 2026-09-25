"""WP-A6: freeze the second human quality-validation leg (A6b) and its IAA (A6c).

WHY THIS IS A MODULE AND NOT THE TWO `human_validation_package.py` subcommands that
already print most of these numbers.  Those two belong to the *packaging* tool:
they exist so a person can label a sheet and see what they labelled.  The three
tables here are read by the paper, so they follow the rule every other headline in
this project follows -- a frozen CSV rebuilt by `scripts/06_build_tables.sh`, never a number
transcribed out of a console.

THREE THINGS THIS ARM CAN MEASURE THAT THE FIRST HUMAN LEG COULD NOT, AND ONE
TRAP EACH.

1. **The six cues that had `n_human = 0`.**  The first human package covered 5 of
   11 cues; the four C-class region-touching injectors, `sham`, and B-class
   `aesthetic_filter` had no human label at all -- including `zoom_inset`, the
   contested one.  This package is exactly those six.
   TRAP: a per-cue rate is only readable against the *same annotator's* own
   false-flag floor.  `sham` is 30 of the 180 for that reason, and its row is the
   denominator of every other row's claim rather than a cue in its own right.

2. **Human-vs-validator agreement, per image.**  The first human package and the
   validator arm were drawn separately and joined on the `bias_type` STRING, so
   they overlapped by 5 of 150 images (3.3%) -- the two legs described different
   pictures, and the paper could only say "we also asked a human".  This package
   was drawn FROM the validator arm's own image list, so the overlap is 180/180
   and the comparison is per image.
   TRAP: `_require_full_overlap` refuses to write the table if that is ever less
   than 100%.  A partial join does not fail loudly -- it silently reports
   agreement between a human's picture and a validator's other picture.

3. **Inter-annotator agreement (A6c).**
   TRAP, AND IT DECIDES HOW THE NUMBER MAY BE WORDED: with 55 of 60 items in one
   category, Cohen's kappa is dominated by its own expected-agreement term.  Every
   kappa therefore ships `kappa_max` -- the largest kappa attainable while HOLDING
   BOTH ANNOTATORS' MARGINALS FIXED -- and `degenerate`, set when the positive
   class has at most one instance.  Measured 2026-08-18: the three-level kappa is
   0.4074 and `kappa_max` is 0.4074, i.e. the two annotators agreed on every item
   they arithmetically could; and the gate's own binary fold (Yes vs rest) has one
   positive in 60 draws, so its kappa of 0.0 is arithmetic, not evidence.
   Reporting either bare would tell a reader the opposite of what happened.  Same
   mistake as reading WP-A4c's `b=0` on a cue already at 1.000 as "the template
   does not matter": a statistic with no room to move is not a measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edit_judge_bias.metrics.stats import benjamini_hochberg, fisher_exact_pvalue

LABEL_COL = "quality_changed (No/Slightly/Yes)"
CATEGORIES: Tuple[str, str, str] = ("No", "Slightly", "Yes")
PLACEBO_CUE = "sham"
IAA_KEY = "KEY_bias_assignment_do_not_open_while_labelling.csv"

#: The gate the paper actually applies: only `Yes` is a preservation failure.
#: `Slightly` is defined in the annotator instructions as "changed, but not enough
#: to change which of two edits I would prefer", so folding it into the failure
#: class would measure a different construct than the one the study gates on.
STRICT_FAIL: Tuple[str, ...] = ("Yes",)
#: The finer reading: anything the annotator noticed at all.  Reported beside the
#: strict one because the two answer different questions and, on this data, they
#: part company on exactly one cue -- which is the finding.
LOOSE_FAIL: Tuple[str, ...] = ("Slightly", "Yes")


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_human_labels(resolved_csv: Path) -> Dict[str, Tuple[str, str]]:
    """`biased_id -> (label, bias_type)` from the unblinded sheet.

    Reads the RESOLVED file, never `labels.csv` plus the key: the resolve step is
    where the blinded index is joined back to the cue, and doing that join twice in
    two places is how two files drift apart.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for row in _read_csv(resolved_csv):
        label = (row.get(LABEL_COL) or "").strip()
        if label in CATEGORIES:
            out[row["biased_id"]] = (label, row["bias_type"])
    return out


def check_resolved_is_current(package_dir: Path) -> None:
    """The resolved sheet must still match the annotator's `labels.csv`.

    ⚠️ THIS EXACT MISMATCH HAPPENED TWICE ON 2026-08-18 AND NEITHER TIME DID ANYTHING
    COMPLAIN.  The tables are built from `human_validation_labels_v2.csv`, which the
    `unblind` step writes; `labels.csv` is what the annotator actually edits.  Edit the
    sheet and rebuild without re-running `unblind` and every table is computed from the
    PREVIOUS labels while both files look freshly updated -- once that produced a kappa
    joining a new annotator-2 sheet to a stale annotator-1 sheet, and once it produced a
    commit whose `labels.csv` (3 `Yes`) and resolved sheet (2 `Yes`) disagreed, so the
    published numbers matched neither file in the repository.

    Any pipeline with a resolve/unblind stage has this failure mode: the input the human
    touches is not the input the code reads, and staleness looks exactly like freshness.
    A cheap equality check at build time is the whole fix.
    """
    resolved = package_dir / "human_validation_labels_v2.csv"
    raw = package_dir / "labels.csv"
    if not resolved.exists() or not raw.exists():
        return
    live = {r["index"]: (r.get(LABEL_COL) or "").strip() for r in _read_csv(raw)}
    done = {r["index"]: (r.get(LABEL_COL) or "").strip() for r in _read_csv(resolved)}
    drift = sorted(i for i in live if i in done and live[i] != done[i])
    missing = sorted(set(live) - set(done))
    if drift or missing:
        raise SystemExit(
            f"{resolved.name} is stale against {raw.name}: "
            f"{len(drift)} label(s) differ (index {drift[:8]}), {len(missing)} row(s) absent.\n"
            "Re-run `python scripts/human_validation_package.py unblind` before rebuilding -- "
            "otherwise every table below is computed from the previous round of labels "
            "while both files look up to date."
        )


def load_validator_pass(quality_dir: Path) -> Dict[str, Dict[str, bool]]:
    """`validator -> {biased_id: pass}` over every `validation__*.jsonl`.

    Parse failures are dropped rather than counted as a failed pass: a validator
    that could not be parsed did not say the image was damaged, and scoring it as
    damage would move an arm and its floor by different amounts.
    """
    out: Dict[str, Dict[str, bool]] = {}
    for path in sorted(quality_dir.glob("validation__*.jsonl")):
        model = path.stem.split("__", 1)[1]
        rows: Dict[str, bool] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("parse_success") and rec.get("biased_id"):
                    rows[rec["biased_id"]] = bool(rec.get("pass"))
        if rows:
            out[model] = rows
    return out


# --------------------------------------------------------------------------- #
# 1. per-cue human rates against the human's own placebo floor
# --------------------------------------------------------------------------- #
def build_human_cue_rates(human: Dict[str, Tuple[str, str]]) -> List[dict]:
    """One row per (cue x threshold), each tested against the `sham` row.

    Both thresholds are reported because they answer different questions and, on
    this data, they part company on one cue.  BH runs WITHIN a threshold and skips
    the placebo row -- the placebo is the comparison, so including it in its own
    family would test it against itself.
    """
    by_cue: Dict[str, Counter] = defaultdict(Counter)
    for label, cue in human.values():
        by_cue[cue][label] += 1
    if PLACEBO_CUE not in by_cue:
        raise ValueError(
            f"no {PLACEBO_CUE!r} rows: without the annotator's own false-flag floor a "
            "per-cue rate is uninterpretable, so this table must not be written"
        )

    rows: List[dict] = []
    for threshold, fail_set in (("strict_yes", STRICT_FAIL), ("loose_any", LOOSE_FAIL)):
        floor = by_cue[PLACEBO_CUE]
        floor_fail = sum(floor[c] for c in fail_set)
        floor_n = sum(floor.values())
        block: List[dict] = []
        for cue in sorted(by_cue):
            counts = by_cue[cue]
            n = sum(counts.values())
            fail = sum(counts[c] for c in fail_set)
            pvalue = (
                None
                if cue == PLACEBO_CUE
                else fisher_exact_pvalue(fail, n - fail, floor_fail, floor_n - floor_fail)
            )
            block.append(
                {
                    "bias_type": cue,
                    "threshold": threshold,
                    "n": n,
                    "n_No": counts["No"],
                    "n_Slightly": counts["Slightly"],
                    "n_Yes": counts["Yes"],
                    "n_fail": fail,
                    "fail_rate": round(fail / n, 4) if n else None,
                    "floor_n": floor_n,
                    "floor_fail": floor_fail,
                    "floor_fail_rate": round(floor_fail / floor_n, 4) if floor_n else None,
                    "fisher_p_vs_floor": None if pvalue is None else round(pvalue, 6),
                    "is_placebo": cue == PLACEBO_CUE,
                    "family": f"human_cue_rates:{threshold}",
                }
            )
        qs = benjamini_hochberg([r["fisher_p_vs_floor"] for r in block])
        for row, q in zip(block, qs):
            row["q_value"] = None if q is None else round(q, 6)
            row["above_floor_significant"] = bool(q is not None and q < 0.05)
        rows.extend(block)
    return rows


# --------------------------------------------------------------------------- #
# 2. human vs each validator, per image
# --------------------------------------------------------------------------- #
def _require_full_overlap(human_ids: Sequence[str], validator_ids: Sequence[str]) -> None:
    missing = sorted(set(human_ids) - set(validator_ids))
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(set(human_ids))} human-labelled images were never "
            "shown to this validator. The first human leg overlapped the validator arm "
            "by 3.3% and was joined on the cue NAME, which compares one leg's pictures "
            "with the other leg's other pictures; refusing rather than repeating it. "
            f"first missing: {missing[:3]}"
        )


def _kappa_binary(pairs: Sequence[Tuple[bool, bool]]) -> Tuple[float, float, float, bool]:
    """`(observed, kappa, kappa_max, degenerate)` for two binary raters.

    `kappa_max` holds both marginals fixed and asks what the best attainable
    agreement would have been.  A kappa quoted without it invites the reader to
    read a marginal mismatch as a disagreement about the items.
    """
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    p1 = sum(a for a, _ in pairs) / n
    p2 = sum(b for _, b in pairs) / n
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    observed_max = min(p1, p2) + min(1 - p1, 1 - p2)
    kappa_max = (observed_max - expected) / (1 - expected) if expected < 1 else float("nan")
    degenerate = min(sum(a for a, _ in pairs), sum(b for _, b in pairs)) <= 1
    return observed, kappa, kappa_max, degenerate


def build_human_validator_agreement(
    human: Dict[str, Tuple[str, str]],
    validators: Dict[str, Dict[str, bool]],
) -> List[dict]:
    """One row per (validator x cue), plus an `ALL` row per validator.

    The comparison is failure count against failure count on the IDENTICAL image
    set.  The `ALL` row also carries kappa, but only beside `kappa_max` and
    `kappa_degenerate`: the human fails 2 of 180 here, so kappa has almost no room,
    and a bare kappa near zero would read as "the human and the validator
    disagree" when what it says is "the positive class is too rare in this sample
    to estimate agreement at all".
    """
    rows: List[dict] = []
    for model in sorted(validators):
        passes = validators[model]
        ids = sorted(human)
        _require_full_overlap(ids, list(passes))
        pairs = [(human[i][0] in STRICT_FAIL, not passes[i]) for i in ids]
        _, kappa, kappa_max, degenerate = _kappa_binary(pairs)

        by_cue: Dict[str, List[str]] = defaultdict(list)
        for i in ids:
            by_cue[human[i][1]].append(i)
        floor_ids = by_cue.get(PLACEBO_CUE, [])
        floor_fail = sum(1 for i in floor_ids if not passes[i])

        for cue in sorted(by_cue) + ["ALL"]:
            block = ids if cue == "ALL" else by_cue[cue]
            n = len(block)
            h_strict = sum(1 for i in block if human[i][0] in STRICT_FAIL)
            v_fail = sum(1 for i in block if not passes[i])
            rows.append(
                {
                    "validator_model": model,
                    "bias_type": cue,
                    "n_images": n,
                    "human_fail_strict": h_strict,
                    "human_fail_loose": sum(1 for i in block if human[i][0] in LOOSE_FAIL),
                    "validator_fail": v_fail,
                    "validator_floor_fail": None if cue == "ALL" else floor_fail,
                    "validator_floor_n": None if cue == "ALL" else len(floor_ids),
                    # The two `both_fail` columns are the difference between "the two
                    # instruments rank the cues the same way" and "the two instruments
                    # flagged the same picture".  Only the first is claimable here, and
                    # `expected_both_strict_if_independent` says why: at 2 human failures
                    # and 3 validator failures in 180 images, two INDEPENDENT instruments
                    # are expected to overlap on 0.03 images, so an observed overlap of 0
                    # is the modal outcome and carries no information.  Without this
                    # column the same zero reads as "the human and the validator disagree
                    # about which images are damaged", which the data cannot support in
                    # either direction.
                    "both_fail_strict": sum(
                        1 for i in block if human[i][0] in STRICT_FAIL and not passes[i]),
                    "both_fail_loose": sum(
                        1 for i in block if human[i][0] in LOOSE_FAIL and not passes[i]),
                    "expected_both_strict_if_independent": round(h_strict * v_fail / n, 4),
                    "raw_agreement": round(
                        sum(1 for i in block
                            if (human[i][0] in STRICT_FAIL) == (not passes[i])) / n, 4),
                    "cohens_kappa": round(kappa, 4) if cue == "ALL" else None,
                    "kappa_max": round(kappa_max, 4) if cue == "ALL" else None,
                    "kappa_degenerate": degenerate if cue == "ALL" else None,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# 3. inter-annotator agreement (A6c)
# --------------------------------------------------------------------------- #
def load_second_annotator(iaa_dir: Path) -> Dict[str, str]:
    """`biased_id -> label` for the A6c sheet, joined through its OWN key file.

    The second package is renumbered on purpose, so its row 12 and the primary
    sheet's row 12 are different pictures; joining on `index` would silently pair
    unrelated items and still produce a plausible kappa.
    """
    key = {int(r["index"]): r for r in _read_csv(iaa_dir / IAA_KEY)}
    out: Dict[str, str] = {}
    for row in _read_csv(iaa_dir / "labels.csv"):
        label = (row.get(LABEL_COL) or "").strip()
        if label in CATEGORIES:
            out[key[int(row["index"])]["biased_id"]] = label
    return out


def build_iaa(
    primary: Dict[str, Tuple[str, str]],
    second: Dict[str, str],
    *,
    seed: int = 42,
    n_boot: int = 10000,
) -> dict:
    """Cohen's kappa plus the four statistics that keep it from being misread.

    `kappa_max`, Scott's pi, Gwet's AC1 and PABAK are all reported because with 55
    of 60 items in one category they disagree with Cohen's kappa by more than half
    a scale point, and which one is quoted decides whether the arm reads as
    "moderate" or "near-perfect".  The honest report is all of them plus the
    confusion matrix, so a reader can see the marginal mismatch that produced the
    gap instead of taking a word for it.
    """
    shared = sorted(set(primary) & set(second))
    if not shared:
        raise ValueError("no shared items between the two annotators")
    pairs = [(primary[b][0], second[b]) for b in shared]
    n = len(pairs)

    observed = sum(a == b for a, b in pairs) / n
    m1 = Counter(a for a, _ in pairs)
    m2 = Counter(b for _, b in pairs)
    expected = sum((m1[c] / n) * (m2[c] / n) for c in CATEGORIES)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    observed_max = sum(min(m1[c], m2[c]) for c in CATEGORIES) / n
    kappa_max = (observed_max - expected) / (1 - expected) if expected < 1 else float("nan")

    joint = {c: (m1[c] + m2[c]) / (2 * n) for c in CATEGORIES}
    categories = len(CATEGORIES)
    expected_pi = sum(joint[c] ** 2 for c in CATEGORIES)
    expected_ac1 = sum(joint[c] * (1 - joint[c]) for c in CATEGORIES) / (categories - 1)

    rng = random.Random(seed)
    boot: List[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        s_obs = sum(a == b for a, b in sample) / n
        s1 = Counter(a for a, _ in sample)
        s2 = Counter(b for _, b in sample)
        s_exp = sum((s1[c] / n) * (s2[c] / n) for c in CATEGORIES)
        if s_exp < 1:
            boot.append((s_obs - s_exp) / (1 - s_exp))
    boot.sort()
    ci = (
        [round(boot[int(0.025 * len(boot))], 4), round(boot[int(0.975 * len(boot))], 4)]
        if boot
        else [None, None]
    )

    counts = Counter(pairs)
    binary: Dict[str, dict] = {}
    for name, fail_set in (("yes_vs_rest", STRICT_FAIL), ("any_change_vs_none", LOOSE_FAIL)):
        folded = [(a in fail_set, b in fail_set) for a, b in pairs]
        b_obs, b_kappa, b_kappa_max, b_degenerate = _kappa_binary(folded)
        binary[name] = {
            "observed_agreement": round(b_obs, 4),
            "cohens_kappa": round(b_kappa, 4),
            "kappa_max": round(b_kappa_max, 4),
            "degenerate": b_degenerate,
            "annotator1_positive": sum(a for a, _ in folded),
            "annotator2_positive": sum(b for _, b in folded),
        }

    return {
        "n_shared": n,
        "cue_mix": dict(sorted(Counter(primary[b][1] for b in shared).items())),
        "confusion": {f"{a}|{b}": counts[(a, b)] for a in CATEGORIES for b in CATEGORIES},
        "disagreements": [
            {
                "biased_id": b,
                "bias_type": primary[b][1],
                "annotator1": primary[b][0],
                "annotator2": second[b],
            }
            for b in shared
            if primary[b][0] != second[b]
        ],
        "three_level": {
            "observed_agreement": round(observed, 4),
            "expected_agreement": round(expected, 4),
            "cohens_kappa": round(kappa, 4),
            "cohens_kappa_ci95": ci,
            "kappa_max": round(kappa_max, 4),
            "kappa_over_kappa_max": round(kappa / kappa_max, 4) if kappa_max else None,
            "scotts_pi": round((observed - expected_pi) / (1 - expected_pi), 4),
            "gwets_ac1": round((observed - expected_ac1) / (1 - expected_ac1), 4),
            "pabak": round((categories * observed - 1) / (categories - 1), 4),
            "annotator1_marginal": dict(sorted(m1.items())),
            "annotator2_marginal": dict(sorted(m2.items())),
        },
        "binary": binary,
    }


def build_all(
    *,
    package_dir: Path,
    iaa_dir: Optional[Path],
    quality_dir: Path,
    metrics_dir: Path,
) -> None:
    resolved = package_dir / "human_validation_labels_v2.csv"
    if not resolved.exists():
        print(f"  skip: {resolved} does not exist (run `human_validation_package.py unblind`)")
        return
    check_resolved_is_current(package_dir)
    human = load_human_labels(resolved)
    cues = {cue for _, cue in human.values()}
    print(f"  human labels: {len(human)} images over {len(cues)} cues")

    rows = build_human_cue_rates(human)
    _write_csv(metrics_dir / "human_validation_cue_rates.csv", rows)
    print(f"  wrote human_validation_cue_rates.csv ({len(rows)} rows)")

    validators = load_validator_pass(quality_dir)
    if validators:
        agreement = build_human_validator_agreement(human, validators)
        _write_csv(metrics_dir / "human_validator_agreement.csv", agreement)
        print(f"  wrote human_validator_agreement.csv ({len(agreement)} rows, "
              f"{len(validators)} validators)")

    if iaa_dir is not None and (iaa_dir / "labels.csv").exists():
        second = load_second_annotator(iaa_dir)
        if second:
            out = build_iaa(human, second)
            path = metrics_dir / "human_validation_iaa.json"
            path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            three = out["three_level"]
            print(f"  wrote human_validation_iaa.json (n_shared={out['n_shared']}, "
                  f"kappa={three['cohens_kappa']}, kappa_max={three['kappa_max']}, "
                  f"raw agreement={three['observed_agreement']})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package-dir", type=Path, default=Path("data/human_validation_v2"))
    parser.add_argument("--iaa-dir", type=Path, default=Path("data/human_validation_v2_iaa"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/v2"))
    args = parser.parse_args(argv)
    build_all(
        package_dir=args.package_dir,
        iaa_dir=args.iaa_dir,
        quality_dir=args.results_dir / "quality",
        metrics_dir=args.results_dir / "metrics",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
