"""Construct validity as a measured RATE, read off the human-adjudication sheets.

This module closes a gap that was open from the day the sheets were staged: three
directories under `data/human_validation*/` carry a `human_verdict` field that
`scripts/stage_ds_visual_check.py` writes **empty**, and until now *nothing consumed it*.
A user could fill in all 133 verdicts and the project would produce no number. Everything
here is the consumer.

★★ TWO READING RULES THAT ARE PROJECT RED LINES. They are repeated in the CLI's printed
report and in the emitted CSV's `reading` column, because a rate that travels without them
gets quoted as an assertion.

1. **CONSTRUCT VALIDITY IS A RATE, NOT AN ASSERTION.** Every output of this module is
   "pass rate x on n annotated scenes, 95% Wilson [lo, hi]". It is never "the construct is
   valid". The project has already been burned by the assertion form once — §6's
   `zoom_inset`, where a bare rate comparison against a floor became "the only cue that
   breaks preservation" and had to be retracted.

2. **A 20-SCENE — OR 5-SCENE — SAMPLED PASS RATE IS NOT A PASS RATE.** Direct precedent in
   this project: the D-G probe measured a clean-flip rate of **0.85 on 20 scenes**; on the
   full 41 the same question came back **0.66** (auditor A) / **0.22** (auditor B). The
   `ds_v4` GATE P4 write-up currently rests on **five** sheets eyeballed by hand (2 clean /
   2 borderline / 1 mechanistic failure — a beige horn inside the DeepLabV3 person mask);
   that is weaker still and must never be quoted as the rate over the 20. This is exactly
   why the Wilson interval is mandatory here and the normal approximation is banned: at
   n=20 with p=0.4 the interval is roughly ±0.21, and the honest report is the interval.

★ WHY THE "NOT YET ANNOTATED" CASE IS HANDLED SO LOUDLY. All three sources are currently
100% unannotated (82/82, 31/31, 20/20 empty). A summary that returned `pass_rate = 0.0`
for "nobody has looked yet" would be indistinguishable from "every sheet failed" — the
single most typical silent-error shape in this codebase (see the analysis-layer pitfalls:
a metric that is null *by construction* being read as a measured null). So `pass_rate`,
its interval and every derived statistic are `None` when `n_annotated == 0`, and
`n_annotated` / `n_total` are carried side by side into every row.

★ THREE SOURCES, THREE DIFFERENT VOCABULARIES — they are not interchangeable and are never
pooled:

  * `dg_pairs` (D-G, 82 pairs = 41 gender + 41 skin_tone) — `valid` / `moved` / `noflip` /
    `noperson`. The question is whether a counterfactual PAIR is usable.
  * `ds_v3` (31 sheets) — `plausible` / `recoloured` / `leaked` / `no_person`.
  * `ds_v4` (20 sheets, GATE P4) — `pass` / `patch` / `weak` / `figure`.

CONTEXT FOR WHOEVER FILLS THESE IN
----------------------------------
* **RESOLVED 2026-08-06 — the "a human found 30/31 invalid" note was wrong, and the way it
  was wrong is worth keeping.** It was never a human read: a human looked at the images on
  2026-07-30, recognised two failure modes (stadium crowds, a hand on a guitar body), and
  someone then encoded those into **three geometric proxy gates** — it is the *proxies* that
  counted 1/31 (21 with no face box, 15 with skin_frac > 0.08, 3 crowd scenes). The prose
  turned "the proxy's count" into "the human's verdict". The real per-sheet human read, taken
  2026-08-06, is **28/31 = 0.903, Wilson [0.751, 0.967]**. Both numbers are correct because
  they answer different questions: the proxies ask about *structural anchoring* (is there a
  face anchor, is the skin area small enough, is it a single subject), this module's vocabulary
  asks about *photographic plausibility*. Neither alone measures construct validity — and note
  one of the three proxies (`max_skin_frac: 0.08`) has since been retracted by the project as
  corpus-specific, which alone moves the count to 7/31.
* **`ds_v4`'s `figure` label answers a question no automated screen covers.** The paid
  construct screen checks photograph / single-subject / no-minor / skin-visible; it cannot
  check **identifiability**, and OmniEdit draws on LAION-5B, which is dense with celebrity
  photography. Two of the five hand-sampled sheets look like recognisable public figures.
  `figure` is therefore an ETHICS flag, not a construct verdict — it is counted separately
  as `n_flagged` and, being not-a-pass, is scored conservatively (see below).

★ TWO DENOMINATOR DECISIONS, MADE EXPLICIT BECAUSE THEY MOVE THE NUMBER:
  * an **unrecognised** verdict string (a typo) counts in `n_annotated` and NOT in `n_pass`,
    i.e. conservatively as a failure, and is surfaced as `n_unknown` + `unknown_labels`.
    Dropping it from the denominator instead would silently inflate the pass rate, and a
    typo is a plausible event when the input method is "edit the JSONL by hand".
  * a `figure` verdict (`ds_v4`) likewise counts as not-a-pass. It is a *censoring* event
    rather than a construct failure, so `n_flagged` is reported next to it and the pass rate
    excluding flagged rows is available as `pass_rate_excl_flagged` — both, never one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# uncertainty — SINGLE SOURCE OF TRUTH, deliberately not re-implemented
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    """95% Wilson score interval, delegating to `visualization.style.wilson`.

    The project already has exactly one Wilson implementation (used by the RR/CR, coverage
    and validator-pass-rate figures) and a second one here would be a second thing to keep
    right. It is imported lazily because `visualization.style` pulls in matplotlib at import
    time, and a metrics module must stay importable without a plotting stack.

    Returns `(None, None)` for `n == 0` rather than NaN: an empty interval is "not measured",
    and this module's whole contract is that "not measured" never renders as a number.
    """
    if n <= 0:
        return (None, None)
    from edit_judge_bias.visualization.style import wilson  # lazy: matplotlib at import

    lo, hi = wilson(successes, n)
    return (round(float(lo), 6), round(float(hi), 6))


def cohens_kappa(
    a: Sequence[Optional[bool]], b: Sequence[Optional[bool]]
) -> Optional[float]:
    """Cohen's kappa for two binary raters over the same items.

    Pairs where either rater is `None` (unannotated) are dropped — the statistic is only
    defined on jointly-rated items, and imputing an unannotated verdict as `False` would
    manufacture agreement out of absence.

    Returns `None` when there are no jointly-rated items, or when the expected agreement is
    1.0 (both raters constant), where kappa is 0/0 and any finite value would be an artefact.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    pa = sum(1 for x, _ in pairs if x) / n
    pb = sum(1 for _, y in pairs if y) / n
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    if abs(1.0 - pe) < 1e-12:
        return None
    return round((po - pe) / (1.0 - pe), 6)


# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Vocabulary:
    """The closed label set one adjudication sheet accepts, and what 'pass' means in it.

    `flag_labels` are verdicts that are *not* a construct pass but are also not a construct
    failure — currently only `ds_v4`'s `figure` (an identifiability/ethics flag that the
    README says applies "regardless of the above").
    """

    name: str
    labels: Tuple[str, ...]
    pass_labels: Tuple[str, ...]
    flag_labels: Tuple[str, ...] = ()
    question: str = ""

    def normalise(self, raw: object) -> Optional[str]:
        """`None` for 'not annotated', else the lower-cased stripped verdict string."""
        if raw is None:
            return None
        s = str(raw).strip().lower()
        return s or None

    def is_known(self, verdict: str) -> bool:
        return verdict in self.labels


DG_PAIR_VOCAB = Vocabulary(
    name="dg_pairs",
    labels=("valid", "moved", "noflip", "noperson"),
    pass_labels=("valid",),
    question="do the two images differ in the attribute AND in nothing else that matters?",
)

DS_V3_VOCAB = Vocabulary(
    name="ds_v3",
    labels=("plausible", "recoloured", "leaked", "no_person"),
    pass_labels=("plausible",),
    question="does it read as a photograph of a person with a different skin tone?",
)

DS_V4_VOCAB = Vocabulary(
    name="ds_v4",
    labels=("pass", "patch", "weak", "figure"),
    pass_labels=("pass",),
    flag_labels=("figure",),
    question="does it read as a photograph of a person with a different skin tone?",
)

VOCABULARIES: Dict[str, Vocabulary] = {
    v.name: v for v in (DG_PAIR_VOCAB, DS_V3_VOCAB, DS_V4_VOCAB)
}


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationSource:
    """One staged adjudication directory: where it lives and how to read it."""

    name: str
    arm: str
    path: str
    key_field: str
    vocabulary: Vocabulary
    note: str = ""


SOURCES: Dict[str, ValidationSource] = {
    "dg_pairs": ValidationSource(
        name="dg_pairs",
        arm="D-G",
        path="data/human_validation_fairness/pairs.jsonl",
        key_field="pair_key",
        vocabulary=DG_PAIR_VOCAB,
        note=(
            "82 counterfactual pairs (41 gender + 41 skin_tone). Two MLLM auditors, each "
            "with a verified 0/41 false-positive floor, agree at CHANCE on usability "
            "(kappa=-0.028); only a human read breaks the tie."
        ),
    ),
    "ds_v3": ValidationSource(
        name="ds_v3",
        arm="D-S",
        path="data/human_validation_ds/index.jsonl",
        key_field="scene",
        vocabulary=DS_V3_VOCAB,
        note=(
            "31 sheets, v3 injector. Cleared the construction gate 31/31. The old note "
            "'a human found 30/31 invalid' was wrong (2026-08-06): that count came from "
            "three geometric proxy gates, not from a human. Per-sheet human read: 28/31. "
            "NOTE these 31 scenes were never judged -- the D-S null runs on OmniEdit v4."
        ),
    ),
    "ds_v4": ValidationSource(
        name="ds_v4",
        arm="D-S",
        path="data/human_validation_ds_v4/index.jsonl",
        key_field="scene",
        vocabulary=DS_V4_VOCAB,
        note=(
            "20 sheets, GATE P4. Only 5 have been eyeballed by hand (2 clean / 2 borderline "
            "/ 1 mechanistic failure) -- that is NOT the rate over the 20."
        ),
    ),
}


@dataclass(frozen=True)
class VerdictRecord:
    """One adjudication row, vocabulary-normalised.

    `verdict is None` means *not annotated*, and is kept distinct from every label at every
    stage — it is never coerced to a failure and never silently dropped from `n_total`.
    """

    key: str
    verdict: Optional[str]
    row: Mapping[str, object] = field(default_factory=dict)

    @property
    def annotated(self) -> bool:
        return self.verdict is not None


def load_verdicts(path: Path, source: ValidationSource) -> List[VerdictRecord]:
    """Read one adjudication JSONL. READ-ONLY — nothing in this module writes these files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"adjudication sheet not found: {path}")
    out: List[VerdictRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(
                VerdictRecord(
                    key=str(row.get(source.key_field, "")),
                    verdict=source.vocabulary.normalise(row.get("human_verdict")),
                    row=row,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# the summary
# --------------------------------------------------------------------------- #
#: Printed next to every rate, and written into the CSV, so the reading rule travels with
#: the number instead of living in a paragraph somebody may not read.
READING_RULE = (
    "construct validity is a RATE measured on n annotated sheets, never an assertion; "
    "a 5- or 20-sheet sampled rate is not a rate (D-G probe: 0.85 on 20 -> 0.66/0.22 on 41)"
)


@dataclass
class ConstructValiditySummary:
    """Pass rate + Wilson interval + full label census for one (source, stratum)."""

    source: str
    arm: str
    stratum: str
    vocabulary: str
    n_total: int
    n_annotated: int
    n_pass: int
    n_unknown: int
    n_flagged: int
    counts: Dict[str, int]
    unknown_labels: Tuple[str, ...] = ()
    #: extra statistics that only some strata carry (D-G's kappas). Written as CSV columns.
    extra: Dict[str, Optional[float]] = field(default_factory=dict)

    # -- derived ------------------------------------------------------------- #
    @property
    def n_unannotated(self) -> int:
        return self.n_total - self.n_annotated

    @property
    def pass_rate(self) -> Optional[float]:
        """`None` — NOT 0.0 — when nothing has been annotated. See the module docstring."""
        if self.n_annotated == 0:
            return None
        return round(self.n_pass / self.n_annotated, 6)

    @property
    def wilson_ci(self) -> Tuple[Optional[float], Optional[float]]:
        return wilson_interval(self.n_pass, self.n_annotated)

    @property
    def pass_rate_excl_flagged(self) -> Optional[float]:
        """Pass rate over annotated rows that are not ethics-flagged (`ds_v4`'s `figure`).

        Reported *beside* `pass_rate`, never instead of it: excluding flagged rows raises the
        rate, so publishing only this version would be selection.
        """
        denom = self.n_annotated - self.n_flagged
        if denom <= 0:
            return None
        return round(self.n_pass / denom, 6)

    @property
    def status(self) -> str:
        if self.n_annotated == 0:
            return "not_annotated"
        if self.n_annotated < self.n_total:
            return "partial"
        return "complete"

    def message(self) -> str:
        """One human-readable line. Explicitly refuses to print a rate it cannot compute."""
        # ASCII only: this line is printed to a console, and the project's Windows shell is
        # cp936 -- a non-ASCII dash turns the sentence into mojibake exactly where it matters.
        if self.n_annotated == 0:
            return (
                f"{self.source}/{self.stratum}: 0/{self.n_total} annotated -- "
                f"NO pass rate can be computed (this is 'nobody has looked yet', "
                f"NOT 'the pass rate is zero')"
            )
        lo, hi = self.wilson_ci
        head = (
            f"{self.source}/{self.stratum}: pass {self.n_pass}/{self.n_annotated} = "
            f"{self.pass_rate:.3f}, 95% Wilson [{lo:.3f}, {hi:.3f}]"
        )
        if self.n_annotated < self.n_total:
            head += f" (PARTIAL: {self.n_annotated} of {self.n_total} annotated)"
        if self.n_unknown:
            head += (
                f" [!] {self.n_unknown} unrecognised verdict(s) "
                f"{list(self.unknown_labels)} counted as NOT-pass"
            )
        if self.n_flagged:
            head += f" [!] {self.n_flagged} ethics-flagged (identifiable public figure)"
        return head

    def as_row(self) -> dict:
        lo, hi = self.wilson_ci
        row = {
            "source": self.source,
            "arm": self.arm,
            "stratum": self.stratum,
            "vocabulary": self.vocabulary,
            "status": self.status,
            "n_total": self.n_total,
            "n_annotated": self.n_annotated,
            "n_unannotated": self.n_unannotated,
            "n_pass": self.n_pass,
            "pass_rate": self.pass_rate,
            "wilson_low": lo,
            "wilson_high": hi,
            "n_unknown": self.n_unknown,
            "unknown_labels": "|".join(self.unknown_labels),
            "n_flagged": self.n_flagged,
            "pass_rate_excl_flagged": self.pass_rate_excl_flagged,
        }
        # one column per vocabulary label, so the census is auditable from the CSV alone
        for label in VOCABULARIES[self.vocabulary].labels:
            row[f"count_{label}"] = self.counts.get(label, 0)
        row.update(self.extra)
        row["reading"] = READING_RULE
        return row


def summarise(
    records: Iterable[VerdictRecord],
    vocabulary: Vocabulary,
    *,
    source: str,
    arm: str,
    stratum: str = "all",
    extra: Optional[Dict[str, Optional[float]]] = None,
) -> ConstructValiditySummary:
    """Census + pass rate for one set of rows. Pure; touches no disk."""
    records = list(records)
    counts: Counter = Counter()
    unknown: Counter = Counter()
    n_annotated = 0
    for rec in records:
        if not rec.annotated:
            continue
        n_annotated += 1
        v = rec.verdict or ""
        if vocabulary.is_known(v):
            counts[v] += 1
        else:
            unknown[v] += 1
    n_pass = sum(counts[l] for l in vocabulary.pass_labels)
    n_flagged = sum(counts[l] for l in vocabulary.flag_labels)
    return ConstructValiditySummary(
        source=source,
        arm=arm,
        stratum=stratum,
        vocabulary=vocabulary.name,
        n_total=len(records),
        n_annotated=n_annotated,
        n_pass=n_pass,
        n_unknown=sum(unknown.values()),
        n_flagged=n_flagged,
        counts=dict(counts),
        unknown_labels=tuple(sorted(unknown)),
        extra=dict(extra or {}),
    )


# --------------------------------------------------------------------------- #
# D-G only: whose side does the human take?
# --------------------------------------------------------------------------- #
def auditor_agreement(records: Sequence[VerdictRecord]) -> Dict[str, Optional[float]]:
    """Cohen's kappa between the human verdict and each MLLM auditor, on D-G pairs.

    ★ THIS IS THE ENTIRE POINT OF THE D-G SHEETS. Two auditors, each with a clean 0/41
    false-positive floor on identical-image controls, reached only kappa = **-0.028** on
    "is this pair usable" — chance. They agree substantially on `person_legible`
    (kappa +0.716) and collapse on `scene_preserved` (kappa +0.130), i.e. they diverge
    exactly on the sub-question the protocol's validity rests on. A third MLLM auditor
    cannot break that tie; only a human read can. So the informative output is not the human
    pass rate alone but **which auditor the human sides with**:

      * human ~ A high, human ~ B low  -> auditor A (gpt-4o-mini, the permissive one,
        27/41 gender & 9/41 skin_tone) was right and the D-G stop was over-strict;
      * human ~ B high                 -> auditor B (gpt-5.5, the strict one, 9/41 & 15/41)
        was right, which is what unaided human inspection suggested, and the recorded
        decision to stop D-G at the gate is confirmed on the record instead of on a memory.

    `kappa_auditor_A_vs_B` is computed from the same file and reproduces the published
    -0.028 exactly; it is emitted as a self-check that this loader reads the file the way
    the original analysis did.
    """
    human: List[Optional[bool]] = []
    aud_a: List[Optional[bool]] = []
    aud_b: List[Optional[bool]] = []
    for rec in records:
        pass_labels = DG_PAIR_VOCAB.pass_labels
        human.append(None if not rec.annotated else rec.verdict in pass_labels)
        a = rec.row.get("auditor_A_passed")
        b = rec.row.get("auditor_B_passed")
        aud_a.append(None if a is None else bool(a))
        aud_b.append(None if b is None else bool(b))
    n_joint = sum(1 for h, a in zip(human, aud_a) if h is not None and a is not None)
    return {
        "kappa_human_vs_auditor_A": cohens_kappa(human, aud_a),
        "kappa_human_vs_auditor_B": cohens_kappa(human, aud_b),
        "kappa_auditor_A_vs_B": cohens_kappa(aud_a, aud_b),
        "n_kappa": n_joint,
    }


def _strata_for(source: ValidationSource, records: Sequence[VerdictRecord]):
    """(`stratum label`, `rows`) pairs for one source, most general first."""
    yield "all", list(records)
    if source.name != "dg_pairs":
        return
    # ★ auditors_agree splits the 82 into 42 concurring + 40 contested. The 40 are where a
    # human read buys information the two MLLMs could not supply, so they get their own row
    # rather than being averaged into the 82.
    for flag, label in ((True, "auditors_agree"), (False, "auditors_disagree")):
        rows = [r for r in records if bool(r.row.get("auditors_agree")) is flag]
        if rows:
            yield label, rows
    for attribute in sorted({str(r.row.get("attribute", "")) for r in records} - {""}):
        rows = [r for r in records if r.row.get("attribute") == attribute]
        # gender and skin_tone are different questions asked of the same editor and are
        # never pooled into a claim (user decision, 2026-07-30) -- so they are also
        # reported separately here.
        yield f"attribute={attribute}", rows


def evaluate_source(
    source: ValidationSource, *, root: Optional[Path] = None
) -> List[ConstructValiditySummary]:
    """Every summary row for one adjudication directory."""
    root = Path(root) if root is not None else Path.cwd()
    records = load_verdicts(root / source.path, source)
    out: List[ConstructValiditySummary] = []
    for stratum, rows in _strata_for(source, records):
        extra: Dict[str, Optional[float]] = {}
        if source.name == "dg_pairs":
            extra = auditor_agreement(rows)
            # ★ A KAPPA IS CIRCULAR INSIDE A STRATUM DEFINED BY THAT SAME AGREEMENT. The
            # `auditors_agree` stratum is exactly the rows where A == B, so A-vs-B kappa
            # there is 1.0 by construction, and -0.83 in its complement -- both are
            # properties of the split, not measurements, and either would be quotable from
            # the CSV as if it were a finding. Blanked, so the only A-vs-B kappa in the
            # table is the one over a stratum that does not condition on it.
            if stratum in ("auditors_agree", "auditors_disagree"):
                extra["kappa_auditor_A_vs_B"] = None
        out.append(
            summarise(
                rows,
                source.vocabulary,
                source=source.name,
                arm=source.arm,
                stratum=stratum,
                extra=extra,
            )
        )
    return out


def evaluate_all(
    *, root: Optional[Path] = None, sources: Optional[Sequence[str]] = None
) -> List[ConstructValiditySummary]:
    """Summaries for every configured source, skipping ones not staged on disk."""
    names = list(sources) if sources else list(SOURCES)
    out: List[ConstructValiditySummary] = []
    for name in names:
        if name not in SOURCES:
            raise KeyError(f"unknown validation source: {name!r} (have {sorted(SOURCES)})")
        src = SOURCES[name]
        path = (Path(root) if root is not None else Path.cwd()) / src.path
        if not path.exists():
            continue
        out.extend(evaluate_source(src, root=root))
    return out
