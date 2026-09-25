"""Turn the D-G human adjudication into a frozen, paper-ready table (zero API).

    python scripts/analyze_dg_verdicts.py            # writes the CSV
    python scripts/analyze_dg_verdicts.py --dry-run  # prints the report, writes nothing

Reads, STRICTLY READ-ONLY:
  * `data/human_validation_fairness/pairs.jsonl` -- 82 adjudicated D-G pairs (the human).
  * `results/v2_fairness/quality/attribute_validation__gpt-4o-mini.jsonl` -- auditor A, raw,
    incl. the 82 identical-image CONTROL rows and the three SUB-judgements
    (`person_legible` / `attribute_flipped` / `scene_preserved`) that `pairs.jsonl` collapses.
  * `results/v2_fairness/quality/attribute_validation__gpt-5.5.jsonl` -- auditor B, same shape.
  * `results/v2_fairness_ds/metrics/attribute_gaps.csv` -- D-S's measured `gap_in_sd_units`,
    the only real effect size this project has for a fairness gap, used as the yardstick the
    D-G power calculation is read against.

Writes exactly one file: `results/v2_fairness/metrics/dg_human_adjudication.csv`, long format
with a `block` column. It never writes a verdict and never touches an input.

★ WHY THE SUB-JUDGEMENTS ARE RE-JOINED FROM THE RAW FILES.
`pairs.jsonl` carries `auditor_{A,B}_passed` and `{A,B}_scene_preserved` only, so from it the
two auditors' disagreement is a brute fact with no mechanism. The raw validation rows carry
all three sub-questions, and `passed = legible AND flipped AND preserved` -- which is what
makes the collapse legible: A's binding constraint is `attribute_flipped` (36/82) on an axis
the human says almost never fails (1/82 `noflip`), B's is `scene_preserved` (35/82) on the
axis the human says fails 25/82. Same pass/fail interface, two different questions answered.

★ KAPPA MEASURES PATTERN, McNEMAR MEASURES RATE, AND HERE THEY DISAGREE ON PURPOSE.
The headline risk in this table is reading "the human's pattern matches auditor B" as "the
human endorses the strict auditor". It does not: the human passes 42/82 and B passes 24/82,
a marginal difference McNemar puts at p=1e-4. Every confusion row therefore carries BOTH a
`kappa` (pattern, marginal-free) and a `mcnemar_p_marginal` (rate), and neither is reported
without the other.

★ THE DISAGREEMENT STRATUM IS A SELECTED SUBSET AND THE TEST ON IT IS CONDITIONAL.
On those 40 pairs `auditor_A_passed == not auditor_B_passed` by construction, so
kappa(H,A) and kappa(H,B) there are ONE fact with mechanically opposite signs, not two
findings -- `cohens_kappa` gives kappa_B = -kappa_A * (1-pe)/pe. The defensible statement is
a single binomial: the human's verdict matches B on 28 of 40. `fisher_p_one_sided_less` is
reported for the same table conditioned on its margins. Neither licenses "A is
anti-correlated with the truth": on the full 82, kappa(H,A) = +0.076, i.e. a coin flip.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # importable without an install, like the test suite
    sys.path.insert(0, str(_SRC))

from edit_judge_bias.metrics.construct_validity import cohens_kappa, wilson_interval  # noqa: E402
from edit_judge_bias.metrics.fairness_metrics import minimum_detectable_effect  # noqa: E402
from edit_judge_bias.metrics.stats import mcnemar_pvalue  # noqa: E402

PAIRS = "data/human_validation_fairness/pairs.jsonl"
AUDITOR_A = "results/v2_fairness/quality/attribute_validation__gpt-4o-mini.jsonl"
AUDITOR_B = "results/v2_fairness/quality/attribute_validation__gpt-5.5.jsonl"
DS_GAPS = "results/v2_fairness_ds/metrics/attribute_gaps.csv"
DEFAULT_OUT = "results/v2_fairness/metrics/dg_human_adjudication.csv"

#: `human_verdict` == this is the only construct pass. Kept as a constant so the definition
#: is in one place: three different failure labels must never be silently pooled into "fail"
#: anywhere except where a binary is explicitly required.
PASS_LABEL = "valid"
FAILURE_LABELS = ("moved", "noflip", "noperson")

#: Which of the three auditor sub-questions each human failure label denies. `valid` denies
#: nothing. Used to score the auditors' sub-judgements against the human on the same axis.
DENIES = {"noperson": "person_legible", "noflip": "attribute_flipped", "moved": "scene_preserved"}

SOURCE_PREFIXES = (("ebench_", "EBench-18K"), ("gb_", "GenAI-Bench"),
                   ("mb_", "MagicBrush"), ("ih_", "ImagenHub"))

#: mb / gb / ih are ONE content pool: the source audit records ImagenHub's rated set as exactly a
#: subset of MagicBrush dev and GenAI-Bench's prompts as 96.6% MagicBrush dev. ORB-collapsing
#: the D-G pool removed duplicate *images*, not this shared provenance, so survivor counts are
#: reported against both the 4 nominal sources and the 2 content pools.
CONTENT_POOL = {"EBench-18K": "EBench-18K", "GenAI-Bench": "MagicBrush-derived",
                "MagicBrush": "MagicBrush-derived", "ImagenHub": "MagicBrush-derived"}

FIELDS = [
    "block", "scope", "attribute", "stratum", "source", "comparator", "n",
    "n_human_pass", "human_pass_rate", "n_comparator_pass", "comparator_pass_rate",
    "wilson_low", "wilson_high",
    "tp", "fp", "fn", "tn", "accuracy", "sensitivity", "specificity", "ppv", "npv",
    "ppv_lift_over_base_rate",
    "kappa", "kappa_ci_low", "kappa_ci_high",
    "fisher_p_two_sided", "fisher_p_one_sided_less", "mcnemar_p_marginal", "binomial_p",
    "count_valid", "count_moved", "count_noflip", "count_noperson",
    "mde_sd", "ds_reference_gap_sd", "mde_over_ds_gap",
    "note",
]


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def source_of(pair_key: str) -> str:
    for prefix, name in SOURCE_PREFIXES:
        if pair_key.startswith(prefix):
            return name
    return "other"


def base_scene(pair_key: str) -> str:
    """`{base_sample_id}__{attribute}` -> `{base_sample_id}`.

    Split from the RIGHT: base ids contain `__` (`ebench_H_00_05_model00`), so a left split
    would regroup scenes instead of failing.
    """
    return pair_key.rsplit("__", 1)[0]


def r4(x: Optional[float]) -> Optional[float]:
    """Six SIGNIFICANT digits, not six decimals.

    `round(5.68e-14, 6)` is `0.0`, and this table's most decisive cells are p-values around
    1e-14 (auditor A denying an attribute flip the human says happened). A p printed as 0.0
    is a p a reader cannot check.
    """
    return None if x is None else float(f"{float(x):.6g}")


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def fisher(tp: int, fn: int, fp: int, tn: int, alternative: str) -> Optional[float]:
    """Fisher exact p for [[tp, fn], [fp, tn]] (rows = human pass/fail, cols = auditor).

    The project's `stats.fisher_exact_pvalue` is two-sided only; below-chance agreement is a
    directional question (odds ratio < 1), so `alternative='less'` is needed and the table
    orientation stops being free -- a transpose leaves two-sided invariant but flips 'less'.
    """
    if (tp + fn) <= 0 or (fp + tn) <= 0 or (tp + fp) <= 0 or (fn + tn) <= 0:
        return None
    from scipy.stats import fisher_exact

    return float(fisher_exact([[tp, fn], [fp, tn]], alternative=alternative)[1])


def binomial_p(successes: int, n: int, p: float = 0.5) -> Optional[float]:
    if n <= 0:
        return None
    from scipy.stats import binomtest

    return float(binomtest(successes, n, p, alternative="two-sided").pvalue)


def kappa_bootstrap_ci(
    a: Sequence[bool], b: Sequence[bool], *, n_boot: int = 2000, seed: int = 42
):
    """Percentile bootstrap CI for Cohen's kappa, resampling ITEMS with replacement.

    Asymptotic kappa SEs are unusable here -- the disagreement stratum has a cell of size 1 --
    and the project already standardises on a seeded percentile bootstrap (`stats.bootstrap_ci`,
    seed 42). Resamples where kappa is undefined (a constant rater) are dropped rather than
    imputed; `n_boot_valid` would be reported if that ever bit, and the count is asserted below.
    """
    import numpy as np

    n = len(a)
    if n < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    aa, bb = list(a), list(b)
    draws = []
    for idx in rng.integers(0, n, size=(n_boot, n)):
        k = cohens_kappa([aa[i] for i in idx], [bb[i] for i in idx])
        if k is not None:
            draws.append(k)
    if len(draws) < n_boot // 2:  # too degenerate to interval-estimate honestly
        return (None, None)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return (round(float(lo), 6), round(float(hi), 6))


def confusion(human: Sequence[bool], other: Sequence[bool]) -> Dict[str, int]:
    tp = sum(1 for h, o in zip(human, other) if h and o)
    fp = sum(1 for h, o in zip(human, other) if (not h) and o)
    fn = sum(1 for h, o in zip(human, other) if h and (not o))
    tn = sum(1 for h, o in zip(human, other) if (not h) and (not o))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# --------------------------------------------------------------------------- #
# row builders
# --------------------------------------------------------------------------- #
def blank() -> dict:
    return {k: None for k in FIELDS}


def marginal_row(scope: str, rows: List[dict], *, attribute=None, stratum=None,
                 source=None, note=None) -> dict:
    n = len(rows)
    n_pass = sum(1 for r in rows if r["human_verdict"] == PASS_LABEL)
    lo, hi = wilson_interval(n_pass, n)
    out = blank()
    out.update(
        block="marginal", scope=scope, attribute=attribute, stratum=stratum, source=source,
        comparator="human", n=n, n_human_pass=n_pass,
        human_pass_rate=r4(n_pass / n) if n else None, wilson_low=lo, wilson_high=hi,
        count_valid=sum(1 for r in rows if r["human_verdict"] == "valid"),
        count_moved=sum(1 for r in rows if r["human_verdict"] == "moved"),
        count_noflip=sum(1 for r in rows if r["human_verdict"] == "noflip"),
        count_noperson=sum(1 for r in rows if r["human_verdict"] == "noperson"),
        note=note,
    )
    return out


def confusion_row(scope: str, rows: List[dict], auditor_field: str, comparator: str,
                  *, attribute=None, stratum=None, note=None) -> dict:
    human = [r["human_verdict"] == PASS_LABEL for r in rows]
    other = [bool(r[auditor_field]) for r in rows]
    c = confusion(human, other)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = len(rows)
    base_rate = sum(human) / n if n else None
    ppv = tp / (tp + fp) if (tp + fp) else None
    lo, hi = kappa_bootstrap_ci(human, other)
    out = blank()
    out.update(
        block="confusion", scope=scope, attribute=attribute, stratum=stratum,
        comparator=comparator, n=n,
        n_human_pass=sum(human), human_pass_rate=r4(base_rate),
        n_comparator_pass=sum(other), comparator_pass_rate=r4(sum(other) / n) if n else None,
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=r4((tp + tn) / n) if n else None,
        sensitivity=r4(tp / (tp + fn)) if (tp + fn) else None,
        specificity=r4(tn / (tn + fp)) if (tn + fp) else None,
        ppv=r4(ppv), npv=r4(tn / (tn + fn)) if (tn + fn) else None,
        ppv_lift_over_base_rate=(
            None if (ppv is None or base_rate is None) else r4(ppv - base_rate)
        ),
        kappa=cohens_kappa(human, other), kappa_ci_low=lo, kappa_ci_high=hi,
        fisher_p_two_sided=r4(fisher(tp, fn, fp, tn, "two-sided")),
        fisher_p_one_sided_less=r4(fisher(tp, fn, fp, tn, "less")),
        # b = human pass & auditor fail, c = human fail & auditor pass: the marginal question,
        # deliberately reported beside kappa because the two answer different questions.
        mcnemar_p_marginal=r4(mcnemar_pvalue(fn, fp)),
        note=note,
    )
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build(root: Path, out: str = DEFAULT_OUT, dry_run: bool = False) -> dict:
    pairs = read_jsonl(root / PAIRS)
    raw_a = read_jsonl(root / AUDITOR_A)
    raw_b = read_jsonl(root / AUDITOR_B)

    unlabelled = [r for r in pairs if not r.get("human_verdict")]
    if unlabelled:
        raise SystemExit(
            f"{len(unlabelled)} of {len(pairs)} pairs have no human_verdict — this analysis "
            "reports a census, not a sample. Fill the sheet or narrow the input."
        )
    bad = sorted({r["human_verdict"] for r in pairs} - {PASS_LABEL, *FAILURE_LABELS})
    if bad:
        raise SystemExit(f"unknown human_verdict label(s): {bad}")

    A = {r["pair_key"]: r for r in raw_a}
    B = {r["pair_key"]: r for r in raw_b}
    missing = [r["pair_key"] for r in pairs if r["pair_key"] not in A or r["pair_key"] not in B]
    if missing:
        raise SystemExit(f"{len(missing)} adjudicated pairs have no raw auditor row: {missing[:3]}")

    for r in pairs:
        k = r["pair_key"]
        # The sheet's collapsed flags must agree with the raw rows they were built from; a
        # mismatch would mean the two files disagree about what an auditor said.
        assert bool(r["auditor_A_passed"]) == bool(A[k]["passed"]), k
        assert bool(r["auditor_B_passed"]) == bool(B[k]["passed"]), k
        r["_source"] = source_of(k)
        r["_pool"] = CONTENT_POOL.get(r["_source"], "other")
        r["_base"] = base_scene(k)

    rows: List[dict] = []
    keys_all = [r["pair_key"] for r in pairs]
    verdict = {r["pair_key"]: r["human_verdict"] for r in pairs}
    agree = [r for r in pairs if r["auditors_agree"]]
    disagree = [r for r in pairs if not r["auditors_agree"]]

    # ---- Q0/Q2: marginals and failure-mode composition ---------------------- #
    rows.append(marginal_row("all", pairs, note="82 pairs = 41 content-distinct scenes x 2 attributes"))
    for attr in ("gender", "skin_tone"):
        rows.append(marginal_row(f"attribute={attr}", [r for r in pairs if r["attribute"] == attr],
                                 attribute=attr))
    rows.append(marginal_row("auditors_agree", agree, stratum="agree"))
    rows.append(marginal_row("auditors_disagree", disagree, stratum="disagree",
                             note="A == not B on all 40 by construction"))
    for _, name in SOURCE_PREFIXES:
        sub = [r for r in pairs if r["_source"] == name]
        if sub:
            rows.append(marginal_row(f"source={name}", sub, source=name,
                                     note=f"content pool: {CONTENT_POOL[name]}"))
    for pool in ("EBench-18K", "MagicBrush-derived"):
        sub = [r for r in pairs if r["_pool"] == pool]
        if sub:
            rows.append(marginal_row(f"content_pool={pool}", sub, source=pool,
                                     note="mb/gb/ih share MagicBrush dev base images"))
    for attr in ("gender", "skin_tone"):
        for _, name in SOURCE_PREFIXES:
            sub = [r for r in pairs if r["attribute"] == attr and r["_source"] == name]
            if sub:
                rows.append(marginal_row(f"attribute={attr} x source={name}", sub,
                                         attribute=attr, source=name))

    # ---- Q1: human vs each auditor, pattern AND rate ------------------------ #
    for scope, sub, stratum in (("all", pairs, None), ("auditors_agree", agree, "agree"),
                                ("auditors_disagree", disagree, "disagree")):
        for field, comp in (("auditor_A_passed", "auditor_A(gpt-4o-mini)"),
                            ("auditor_B_passed", "auditor_B(gpt-5.5)")):
            note = None
            if stratum == "disagree":
                note = ("selected stratum: A == not B, so the two kappas here are one fact "
                        "with mechanically opposite signs")
            rows.append(confusion_row(scope, sub, field, comp, stratum=stratum, note=note))
    for attr in ("gender", "skin_tone"):
        sub = [r for r in pairs if r["attribute"] == attr]
        for field, comp in (("auditor_A_passed", "auditor_A(gpt-4o-mini)"),
                            ("auditor_B_passed", "auditor_B(gpt-5.5)")):
            rows.append(confusion_row(f"attribute={attr}", sub, field, comp, attribute=attr))

    # The tie-break stated as ONE binomial rather than two mirrored kappas.
    matches_b = sum(1 for r in disagree
                    if (r["human_verdict"] == PASS_LABEL) == bool(r["auditor_B_passed"]))
    row = blank()
    row.update(block="tiebreak", scope="auditors_disagree", stratum="disagree",
               comparator="human matches auditor_B", n=len(disagree), tp=matches_b,
               accuracy=r4(matches_b / len(disagree)),
               binomial_p=r4(binomial_p(matches_b, len(disagree))),
               note=("A and B are exact complements here, so matching B == not matching A; "
                     "one binomial, not two independent kappas"))
    rows.append(row)
    for attr in ("gender", "skin_tone"):
        sub = [r for r in disagree if r["attribute"] == attr]
        m = sum(1 for r in sub if (r["human_verdict"] == PASS_LABEL) == bool(r["auditor_B_passed"]))
        row = blank()
        row.update(block="tiebreak", scope=f"auditors_disagree x attribute={attr}",
                   attribute=attr, stratum="disagree", comparator="human matches auditor_B",
                   n=len(sub), tp=m, accuracy=r4(m / len(sub)) if sub else None,
                   binomial_p=r4(binomial_p(m, len(sub))))
        rows.append(row)

    # Auditor A vs auditor B, for the record: pattern at chance, rates not far apart.
    a_pass = [bool(r["auditor_A_passed"]) for r in pairs]
    b_pass = [bool(r["auditor_B_passed"]) for r in pairs]
    c = confusion(a_pass, b_pass)
    lo, hi = kappa_bootstrap_ci(a_pass, b_pass)
    row = blank()
    row.update(block="confusion", scope="all", comparator="auditor_A vs auditor_B",
               n=len(pairs), n_comparator_pass=sum(b_pass),
               comparator_pass_rate=r4(sum(b_pass) / len(pairs)),
               tp=c["tp"], fp=c["fp"], fn=c["fn"], tn=c["tn"],
               accuracy=r4((c["tp"] + c["tn"]) / len(pairs)),
               kappa=cohens_kappa(a_pass, b_pass), kappa_ci_low=lo, kappa_ci_high=hi,
               fisher_p_two_sided=r4(fisher(c["tp"], c["fn"], c["fp"], c["tn"], "two-sided")),
               mcnemar_p_marginal=r4(mcnemar_pvalue(c["fn"], c["fp"])),
               note="rows = auditor A pass/fail; the published kappa = -0.028")
    rows.append(row)

    # ---- Q2/Q3: the three sub-questions, where the collapse actually is ----- #
    keys = keys_all
    for sub_q in ("person_legible", "attribute_flipped", "scene_preserved"):
        human = [verdict[k] != next(v for v, f in DENIES.items() if f == sub_q) for k in keys]
        av = [bool(A[k][sub_q]) for k in keys]
        bv = [bool(B[k][sub_q]) for k in keys]
        for comp, other in (("auditor_A(gpt-4o-mini)", av), ("auditor_B(gpt-5.5)", bv),
                            ("auditor_A vs auditor_B", None)):
            x, y = (human, other) if other is not None else (av, bv)
            cc = confusion(x, y)
            lo, hi = kappa_bootstrap_ci(x, y)
            row = blank()
            row.update(block="subquestion", scope=sub_q, comparator=comp, n=len(keys),
                       n_human_pass=sum(human), human_pass_rate=r4(sum(human) / len(keys)),
                       n_comparator_pass=sum(y), comparator_pass_rate=r4(sum(y) / len(keys)),
                       tp=cc["tp"], fp=cc["fp"], fn=cc["fn"], tn=cc["tn"],
                       accuracy=r4((cc["tp"] + cc["tn"]) / len(keys)),
                       kappa=cohens_kappa(x, y), kappa_ci_low=lo, kappa_ci_high=hi,
                       mcnemar_p_marginal=r4(mcnemar_pvalue(cc["fn"], cc["fp"])),
                       note=("passed = legible AND flipped AND preserved; the human's implied "
                             "sub-label is 'denied by exactly one failure verdict'"))
            rows.append(row)

    # ---- Q2: compliance vs preservation, on the pairs where it is ASSESSABLE - #
    # `noperson` makes the flip unassessable, so a raw 25/82 understates the preservation
    # failure and 1/82 understates nothing but invites the wrong denominator. The editor's
    # two obligations are scored on the 68 pairs where a person's attribute is discernible.
    for attr in (None, "gender", "skin_tone"):
        sub = [r for r in pairs if attr is None or r["attribute"] == attr]
        disc = [r for r in sub if r["human_verdict"] != "noperson"]
        flipped = [r for r in disc if r["human_verdict"] != "noflip"]
        moved = [r for r in disc if r["human_verdict"] == "moved"]
        lo, hi = wilson_interval(len(flipped), len(disc))
        row = blank()
        row.update(block="editor_behaviour", scope="attribute flip succeeded (compliance)",
                   attribute=attr or "both", n=len(disc), n_human_pass=len(flipped),
                   human_pass_rate=r4(len(flipped) / len(disc)) if disc else None,
                   wilson_low=lo, wilson_high=hi,
                   count_noflip=len(disc) - len(flipped),
                   note="denominator = pairs with a discernible person, NOT all 82")
        rows.append(row)
        lo, hi = wilson_interval(len(moved), len(disc))
        row = blank()
        row.update(block="editor_behaviour", scope="scene NOT preserved (preservation failure)",
                   attribute=attr or "both", n=len(disc), n_human_pass=len(moved),
                   human_pass_rate=r4(len(moved) / len(disc)) if disc else None,
                   wilson_low=lo, wilson_high=hi, count_moved=len(moved),
                   note="same denominator; this is the failure the D-G stop was about")
        rows.append(row)

    # Is one auditor's `scene_preserved` a strict subset of the other's? If so the two are
    # not disagreeing symmetrically -- one is uniformly stricter -- and "the auditors
    # disagree" understates what happened.
    a_only = sum(1 for k in keys_all if A[k]["scene_preserved"] and not B[k]["scene_preserved"])
    b_only = sum(1 for k in keys_all if B[k]["scene_preserved"] and not A[k]["scene_preserved"])
    row = blank()
    row.update(block="editor_behaviour", scope="scene_preserved nesting", n=len(keys_all),
               comparator="auditor_A vs auditor_B", tp=a_only, fp=b_only,
               mcnemar_p_marginal=r4(mcnemar_pvalue(a_only, b_only)),
               note=("tp = A-preserved-only, fp = B-preserved-only. fp == 0 means B's "
                     "preserved set is a STRICT SUBSET of A's: uniformly stricter, not "
                     "symmetrically disagreeing"))
    rows.append(row)

    # ---- Q3: the `noperson` stratum in detail ------------------------------- #
    nop = [r for r in pairs if r["human_verdict"] == "noperson"]
    row = blank()
    row.update(block="noperson", scope="all", n=len(nop),
               n_comparator_pass=sum(1 for r in nop if r["auditor_A_passed"]),
               comparator="auditor_A passes among human-noperson",
               tp=sum(1 for r in nop if r["auditors_agree"]),
               fp=sum(1 for r in nop if not r["auditors_agree"]),
               count_noperson=len(nop),
               note=("tp/fp columns reused: pairs in the AGREE / DISAGREE stratum. "
                     f"auditor_B passes {sum(1 for r in nop if r['auditor_B_passed'])} of them"))
    rows.append(row)
    for keyfn, label in ((lambda r: r["attribute"], "attribute"),
                         (lambda r: r["_source"], "source"),
                         (lambda r: "agree" if r["auditors_agree"] else "disagree", "stratum")):
        for value in sorted({keyfn(r) for r in nop}):
            sub = [r for r in nop if keyfn(r) == value]
            row = blank()
            row.update(block="noperson", scope=f"{label}={value}", n=len(sub),
                       attribute=value if label == "attribute" else None,
                       source=value if label == "source" else None,
                       stratum=value if label == "stratum" else None,
                       count_noperson=len(sub))
            rows.append(row)

    # ---- auditor false-flip floors (what makes any of this readable) -------- #
    for name, raw in (("auditor_A(gpt-4o-mini)", raw_a), ("auditor_B(gpt-5.5)", raw_b)):
        ctl = [r for r in raw if r.get("is_control")]
        false_pass = sum(1 for r in ctl if r["passed"])
        lo, hi = wilson_interval(false_pass, len(ctl))
        row = blank()
        row.update(block="floor", scope="identical-image controls", comparator=name,
                   n=len(ctl), n_comparator_pass=false_pass,
                   comparator_pass_rate=r4(false_pass / len(ctl)) if ctl else None,
                   wilson_low=lo, wilson_high=hi,
                   note=("false-flip floor: label_a == label_b, so a sound auditor must "
                         "refuse. 41 controls per attribute"))
        rows.append(row)

    # ---- Q4: what survives, and what n it buys ----------------------------- #
    ds_gaps = []
    ds_path = root / DS_GAPS
    if ds_path.exists():
        with ds_path.open(encoding="utf-8") as fh:
            ds_gaps = [r for r in csv.DictReader(fh) if r["attribute"] == "skin_tone"]
    ds_abs = sorted(abs(float(r["gap_in_sd_units"])) for r in ds_gaps) or [None]
    ds_min, ds_max = ds_abs[0], ds_abs[-1]

    survivors = {a: [r for r in pairs if r["attribute"] == a and r["human_verdict"] == PASS_LABEL]
                 for a in ("gender", "skin_tone")}
    for attr, sub in survivors.items():
        n = len(sub)
        mde = minimum_detectable_effect(n)
        for ref, tag in ((ds_max, "largest"), (ds_min, "smallest")):
            row = blank()
            row.update(block="power", scope=f"human-valid only, attribute={attr}",
                       attribute=attr, n=n, mde_sd=r4(mde),
                       ds_reference_gap_sd=r4(ref),
                       mde_over_ds_gap=r4(mde / ref) if (ref and mde) else None,
                       note=(f"MDE is in SD(PAIRED DIFFERENCE) units; ref = {tag} |gap| the "
                             "D-S arm measured for skin_tone across its 5 judges"))
            rows.append(row)
    for n, scope in ((41, "as-designed, all pairs, per attribute"),
                     (27, "human-valid, scenes valid on EITHER attribute (pooling refused)"),
                     (15, "human-valid, scenes valid on BOTH attributes"),
                     (743, "D-S skin_tone actual (gpt-5.5)"),
                     (611, "claim A breadth block reference")):
        row = blank()
        row.update(block="power", scope=scope, n=n, mde_sd=r4(minimum_detectable_effect(n)),
                   ds_reference_gap_sd=r4(ds_max),
                   mde_over_ds_gap=r4(minimum_detectable_effect(n) / ds_max) if ds_max else None,
                   note="reference row")
        rows.append(row)

    # survivor composition -- the selected sample is not the designed one
    for attr, sub in survivors.items():
        for _, name in SOURCE_PREFIXES:
            k = sum(1 for r in sub if r["_source"] == name)
            if k:
                row = blank()
                row.update(block="survivors", scope=f"attribute={attr} x source={name}",
                           attribute=attr, source=name, n=k,
                           note=f"content pool: {CONTENT_POOL[name]}")
                rows.append(row)
        for pool in ("EBench-18K", "MagicBrush-derived"):
            k = sum(1 for r in sub if r["_pool"] == pool)
            row = blank()
            row.update(block="survivors", scope=f"attribute={attr} x content_pool={pool}",
                       attribute=attr, source=pool, n=k,
                       note="content-pool provenance, not the nominal source label")
            rows.append(row)

    # ---- could an AUTOMATABLE gate have replaced the human read? ------------ #
    # The human verdict is the ground truth here, so every auditor-only gating policy can be
    # scored against it exactly. If some policy hit both usable purity and usable n, the human
    # read would be a convenience; none does, which is what makes it load-bearing.
    n_valid = sum(1 for r in pairs if r["human_verdict"] == PASS_LABEL)
    policies: List[tuple] = [
        ("auditor_A only", lambda r: bool(r["auditor_A_passed"])),
        ("auditor_B only", lambda r: bool(r["auditor_B_passed"])),
        ("A AND B (intersection)", lambda r: bool(r["auditor_A_passed"] and r["auditor_B_passed"])),
        ("A OR B (union)", lambda r: bool(r["auditor_A_passed"] or r["auditor_B_passed"])),
        ("human read (ground truth)", lambda r: r["human_verdict"] == PASS_LABEL),
    ]
    for name, keep in policies:
        for attr in (None, "gender", "skin_tone"):
            sub = [r for r in pairs if attr is None or r["attribute"] == attr]
            kept = [r for r in sub if keep(r)]
            good = sum(1 for r in kept if r["human_verdict"] == PASS_LABEL)
            denom = sum(1 for r in sub if r["human_verdict"] == PASS_LABEL)
            mde = minimum_detectable_effect(len(kept)) if kept else None
            lo, hi = wilson_interval(good, len(kept))
            row = blank()
            row.update(block="gate_policy", scope=name, attribute=attr or "both",
                       n=len(kept), n_human_pass=good,
                       ppv=r4(good / len(kept)) if kept else None,
                       wilson_low=lo, wilson_high=hi,
                       sensitivity=r4(good / denom) if denom else None,
                       mde_sd=r4(mde),
                       note=("n = pairs the policy retains; ppv = purity against the human; "
                             "sensitivity = recall of the human's valid set; mde is what that "
                             "n buys, in SD(paired difference)"))
            rows.append(row)
    assert n_valid == 42, n_valid  # the ground-truth ceiling any policy is scored against

    # ---- pipeline state: what exists on disk vs what a restart needs -------- #
    for label, rel, expected in (
        ("step2_attribute_flip_renders", "data/images/counterfactual", 164),
        ("step4_instruction_rerun_renders", "data/images/counterfactual_edited", 164),
        ("editor_noise_null_control", "data/manifests/counterfactual_edits.jsonl", 41),
    ):
        path = root / rel
        if path.is_dir():
            present = sum(1 for _ in path.rglob("*.png"))
        elif path.is_file():
            present = len(read_jsonl(path))
        else:
            present = 0
        row = blank()
        row.update(block="pipeline_state", scope=label, n=present,
                   comparator=rel,
                   note=(f"expected {expected} for a complete D-G arm; "
                         f"{'PRESENT' if present else 'NEVER RUN'}"))
        rows.append(row)

    if not dry_run:
        out_path = root / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    return {"rows": len(rows), "out": str(root / out), "dry_run": dry_run,
            "n_pairs": len(pairs), "n_scenes": len({r["_base"] for r in pairs})}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    report = build(root, out=args.out, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
