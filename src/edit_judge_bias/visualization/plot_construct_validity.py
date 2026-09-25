"""F10 — construct validity as a measured RATE, and the tie-break the auditors could not supply.

Source CSV (frozen; this module recomputes **nothing** except an integrity check):

``results/v2_fairness_ds/metrics/construct_validity.csv``
    7 rows: ``dg_pairs`` in five strata, ``ds_v3``, ``ds_v4``.

Optionally, for the right panel's marginals only (a local read, no API):

``data/human_validation_fairness/pairs.jsonl``
    the D-G adjudication sheet, read **read-only** to count how many pairs each rater
    passed. If it is absent the marginal strip is simply not drawn and the panel says so.

WHY THE FIGURE EXISTS
---------------------
Two things that a table of seven rows cannot make a reader see.

**Left — a pass rate on 20-82 sheets is an INTERVAL, not a number.** This project has
already been burned twice by the point-estimate form: §6's ``zoom_inset`` ("the only cue
that breaks preservation", retracted once the floor got a significance test) and the D-G
probe, whose clean-flip rate was **0.85 on 20 scenes** and **0.66 / 0.22 on the full 41**.
So the point estimate is never drawn alone here, and the interval is drawn *emphatically*.

**Right — the D-G sheets exist because two MLLM auditors tied at chance** (kappa = -0.028
on "is this pair usable"), and the only thing that can break that tie is a human read.
The figure shows which auditor the human's pattern tracks — and, in the same breath, the
fact that would make that sentence misleading on its own (see the red line below).

★ RED LINE 1 — THREE SOURCES, THREE VOCABULARIES, THREE RULERS
--------------------------------------------------------------
The three sources ask three *different questions* with three *different closed label sets*:

===========  =====================================================  ==========================
source       question                                               labels
===========  =====================================================  ==========================
``dg_pairs`` do the two images differ in the attribute AND in        valid / moved / noflip /
             nothing else that matters?  (a PAIR is the unit)        noperson
``ds_v3``    does it read as a photograph of a person with a         plausible / recoloured /
             different skin tone?  (a SHEET is the unit)             leaked / no_person
``ds_v4``    same question, GATE P4's label set                      pass / patch / weak /
                                                                     figure
===========  =====================================================  ==========================

They are all proportions in [0, 1], which is exactly what makes the trap easy: "D-S 0.90
beats D-G 0.51" is a sentence the units permit and the *questions* forbid. So each source
gets its **own axis** with its own spine, its own ticks and its own printed question — the
same device F1 uses for the MDE units and F9 for the three layer scalars — and
:func:`build_blocks` raises if two vocabularies are ever routed onto one axis.

★ RED LINE 2 — "THE HUMAN SIDES WITH AUDITOR B" IS ABOUT PATTERN, NOT STRICTNESS
--------------------------------------------------------------------------------
kappa(human, B) > 0 > kappa(human, A) in every stratum, and on the 40 contested pairs the
human sides with B 28 times against A's 12. Read alone, that invites "so the strict auditor
was right and the corpus is as bad as it said". It is not what the sheets say: the human
passes **42 of 82 — more than auditor A (36) and far more than auditor B (24)**. B's passes
are nearly a *subset* of the human's (only 2 of B's 24 are human rejects, against 16 of A's
36), i.e. B is a conservative filter that still misses 20 of the 42 usable pairs.

So the marginal pass counts are drawn **in the same panel as the kappas**, as counts rather
than as intervals, and the panel says the two sentences together. A figure that drew only
the kappas would be quoted as "the human confirmed the strict auditor", which the same file
refutes.

★ A CONTRADICTION THIS FIGURE SURFACED, AND HOW IT WAS RESOLVED (2026-08-06)
---------------------------------------------------------------------------
``data/human_validation_ds_v4/README_GATE_P4.md`` used to state in prose that v3's injector
"passed the same gate 31/31, and a human then found **30 of those 31 invalid**", while the
``ds_v3`` sheets say **28 of 31 plausible**. The prose was wrong, and the way it was wrong is
worth keeping: **the 1/31 was the count from three geometric proxy gates, not a human read**
(21 sheets with no face box, 15 with ``skin_frac`` > 0.08, 3 crowd scenes). A human did look
at the images on 2026-07-30 and named two failure modes; someone then encoded those into the
three proxies, and the proxies counted. In transmission "the proxy's count" became "the
human's verdict".

Both numbers are correct because **they answer different questions** — the proxies ask about
structural anchoring (is there a face anchor, is the skin area small enough, is it a single
subject), this module's vocabulary asks about photographic plausibility. Neither alone is
construct validity. One of the three proxies (``max_skin_frac: 0.08``) has since been
retracted by the project as corpus-specific, which on its own moves the count to 7/31.

⚠️ Scope, and it is the part that matters for §6: **those 31 scenes were never judged.**
The published D-S null runs on OmniEdit v4's 743 scenes and overlaps them by zero, so the
rate that constrains §6 is ``ds_v4``'s 0.850, not any ds_v3 number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from matplotlib.lines import Line2D

from edit_judge_bias.metrics.construct_validity import VOCABULARIES
from edit_judge_bias.visualization.style import (
    BAND,
    BASE,
    BLUE,
    CRITICAL,
    GRID,
    INK,
    INK2,
    MUTED,
    ORANGE,
    SURFACE,
    f,
    new_figure,
    save,
    wilson,
)

#: Columns the figure reads. A missing one raises rather than drawing a blank row: every
#: one of them is load-bearing (the denominator, the count, the two interval endpoints),
#: and a silently empty row in a 7-row figure is a row a reader will read as "zero".
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "source",
    "arm",
    "stratum",
    "vocabulary",
    "n_total",
    "n_annotated",
    "n_pass",
    "pass_rate",
    "wilson_low",
    "wilson_high",
)

#: How far the stored interval may sit from the Wilson interval recomputed from
#: ``n_pass / n_annotated`` before :func:`estimates` refuses to draw it. This is the guard
#: against the single most likely way this figure could go wrong: a table that had switched
#: to the normal approximation would still populate ``wilson_low``/``wilson_high``, and the
#: two differ by 0.02-0.09 at exactly the n and p this study lives at.
WILSON_TOLERANCE = 5e-3

#: Colour follows the ARM, never the rank — D-G blue, D-S orange (categorical slots 1-2).
ARM_COLOUR: Dict[str, str] = {"D-G": BLUE, "D-S": ORANGE}

#: Short row labels. Anything unmatched falls back to the raw stratum with `attribute=`
#: stripped, so a new stratum shows up readable rather than crashing.
STRATUM_LABEL: Dict[str, str] = {
    "all": "all",
    "auditors_agree": "auditors agree",
    "auditors_disagree": "auditors disagree",
    "attribute=gender": "gender",
    "attribute=skin_tone": "skin_tone",
}

class KappaSeries(NamedTuple):
    """One kappa column and how it is drawn.

    ``offset`` staggers the three series inside their shared row. Without it the
    ``auditors_agree`` stratum draws two marks at the identical x (kappa is +0.485 against
    both auditors there, because A == B by construction) and the figure would show ONE
    mark where the data has two — the reader would count four measurements instead of five.
    """

    column: str
    legend: str
    colour: str
    marker: str
    offset: float
    size: float


#: ``auditor_A`` = gpt-4o-mini (the permissive one), ``auditor_B`` = gpt-5.5 (the strict one).
KAPPA_SERIES: Tuple[KappaSeries, ...] = (
    KappaSeries("kappa_human_vs_auditor_A", "human ~ auditor A", ORANGE, "o", -0.21, 8.0),
    KappaSeries("kappa_human_vs_auditor_B", "human ~ auditor B", BLUE, "s", +0.21, 8.0),
    KappaSeries("kappa_auditor_A_vs_B", "auditor A ~ auditor B", MUTED, "d", 0.0, 6.6),
)

#: A drawn convention, labelled as one on the figure — not a hypothesis test.
NEAR_CHANCE = 0.20


# --------------------------------------------------------------------------- #
# reading the frozen table
# --------------------------------------------------------------------------- #
def check_columns(rows: Sequence[dict]) -> None:
    """Raise unless every required column is present. Never draws a partial figure."""
    if not rows:
        raise ValueError("construct_validity.csv is empty — nothing to draw")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            "construct_validity.csv is missing column(s) "
            f"{missing}; refusing to draw a figure whose rows would be silently blank"
        )


def check_one_vocabulary_per_axis(rows: Sequence[dict], source: str = "") -> str:
    """Return the single vocabulary these rows share, or raise.

    The three sources' rates are all proportions, so nothing about the *units* stops a
    reader comparing them; only the question differs. Putting two vocabularies on one axis
    would silently license that comparison, so it is refused in code rather than warned
    about in a caption.
    """
    vocabs = sorted({str(r.get("vocabulary", "")) for r in rows})
    if len(vocabs) > 1:
        raise ValueError(
            "refusing to draw two vocabularies on one axis"
            + (f" (source {source!r})" if source else "")
            + f": {vocabs}. Each source asks a different question with a different closed "
              "label set; one shared ruler would invite a comparison the labels forbid."
        )
    if not vocabs or not vocabs[0]:
        raise ValueError("rows carry no vocabulary")
    if vocabs[0] not in VOCABULARIES:
        raise ValueError(
            f"unknown vocabulary {vocabs[0]!r} (have {sorted(VOCABULARIES)}) — a new "
            "adjudication vocabulary needs its question written down before it is plotted"
        )
    return vocabs[0]


@dataclass(frozen=True)
class Block:
    """One source = one axis. Carries the question and label set printed above it."""

    source: str
    arm: str
    vocabulary: str
    question: str
    labels: Tuple[str, ...]
    pass_labels: Tuple[str, ...]
    flag_labels: Tuple[str, ...]
    rows: List[dict]

    @property
    def colour(self) -> str:
        return ARM_COLOUR.get(self.arm, MUTED)


def build_blocks(rows: Sequence[dict]) -> List[Block]:
    """Group the table into one block per source, `all` first inside each block."""
    check_columns(rows)
    order: List[str] = []
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        src = str(row.get("source", ""))
        if src not in grouped:
            grouped[src] = []
            order.append(src)
        grouped[src].append(row)

    blocks: List[Block] = []
    for src in order:
        got = grouped[src]
        vocab_name = check_one_vocabulary_per_axis(got, src)
        vocab = VOCABULARIES[vocab_name]
        got = sorted(got, key=lambda r: (str(r.get("stratum")) != "all",))
        blocks.append(
            Block(
                source=src,
                arm=str(got[0].get("arm", "")),
                vocabulary=vocab_name,
                question=vocab.question,
                labels=tuple(vocab.labels),
                pass_labels=tuple(vocab.pass_labels),
                flag_labels=tuple(vocab.flag_labels),
                rows=got,
            )
        )
    return blocks


class Estimate(NamedTuple):
    """One drawn row: a count, a rate, and the two ENDPOINTS of its Wilson interval.

    ``lo``/``hi`` are absolute positions, never a half-width — the interval is asymmetric
    at every n in this table and a symmetric error bar would misplace one end by up to
    0.09 (``ds_v3``: 0.903 sits 0.152 above its lower endpoint and 0.064 below its upper).
    """

    stratum: str
    label: str
    n_pass: int
    n_annotated: int
    n_total: int
    rate: Optional[float]
    lo: Optional[float]
    hi: Optional[float]

    @property
    def annotated(self) -> bool:
        return self.n_annotated > 0


def stratum_label(stratum: str) -> str:
    return STRATUM_LABEL.get(stratum, stratum.replace("attribute=", ""))


def estimates(block: Block) -> List[Estimate]:
    """Rows -> drawable estimates, with two refusals.

    * ``n_annotated == 0`` yields ``rate=None`` and NOT ``0.0``. "Nobody has looked yet"
      and "every sheet failed" are the same pixel otherwise, and this project's own
      pitfall list has that shape at the top.
    * stored endpoints that are not the Wilson interval for ``n_pass / n_annotated``
      raise. See :data:`WILSON_TOLERANCE`.
    """
    out: List[Estimate] = []
    for row in block.rows:
        n_ann = int(float(row.get("n_annotated") or 0))
        n_pass = int(float(row.get("n_pass") or 0))
        n_total = int(float(row.get("n_total") or 0))
        rate = f(row, "pass_rate")
        lo, hi = f(row, "wilson_low"), f(row, "wilson_high")
        if n_ann == 0:
            rate = lo = hi = None
        elif lo is None or hi is None:
            raise ValueError(
                f"{block.source}/{row.get('stratum')}: annotated rows must carry both "
                "wilson_low and wilson_high; a rate without its interval is exactly the "
                "form this figure exists to refuse"
            )
        else:
            want_lo, want_hi = wilson(n_pass, n_ann)
            if abs(want_lo - lo) > WILSON_TOLERANCE or abs(want_hi - hi) > WILSON_TOLERANCE:
                raise ValueError(
                    f"{block.source}/{row.get('stratum')}: stored interval "
                    f"[{lo:.4f}, {hi:.4f}] is not the Wilson interval for {n_pass}/{n_ann} "
                    f"([{want_lo:.4f}, {want_hi:.4f}]). Refusing to draw a symmetric "
                    "normal-approximation interval as though it were Wilson."
                )
        out.append(
            Estimate(
                stratum=str(row.get("stratum", "")),
                label=stratum_label(str(row.get("stratum", ""))),
                n_pass=n_pass,
                n_annotated=n_ann,
                n_total=n_total,
                rate=rate,
                lo=lo,
                hi=hi,
            )
        )
    return out


def census_line(block: Block) -> str:
    """`40 failures: moved 25 · noflip 1 · noperson 14` — read off the `count_*` columns.

    The census is what turns a bare rate into a *mechanism*: on D-G the failures are 62%
    ``moved``, which is the recorded reason D-G stopped (the attribute editor re-renders
    the whole frame instead of editing locally).

    ★ A flag label is printed EVEN WHEN ITS COUNT IS ZERO. ``ds_v4``'s ``figure`` is an
    identifiability flag, and "0 recorded" is a claim about the sheet worth seeing next to
    ``README_GATE_P4.md``'s statement that two of the five hand-sampled scenes look like
    recognisable public figures. A zero that is never printed is a zero nobody checks.
    """
    top = next((r for r in block.rows if str(r.get("stratum")) == "all"), None)
    if top is None:
        return ""
    counts = {l: int(float(top.get(f"count_{l}") or 0)) for l in block.labels}
    fails = [(l, counts[l]) for l in block.labels
             if l not in block.pass_labels and l not in block.flag_labels]
    total = sum(c for _, c in fails)
    parts = [f"{total} failures: " + " · ".join(f"{l} {c}" for l, c in fails if c)
             if total else "no failures recorded"]
    parts += [f"ethics flag “{l}”: {counts[l]} recorded" for l in block.flag_labels]
    return "     ·     ".join(parts)


# --------------------------------------------------------------------------- #
# the kappa marks
# --------------------------------------------------------------------------- #
class KappaMark(NamedTuple):
    stratum: str
    label: str
    series: str
    legend: str
    value: float
    colour: str
    marker: str
    offset: float
    size: float


def kappa_marks(rows: Sequence[dict]) -> List[KappaMark]:
    """Every kappa the table actually carries, in stratum order.

    ★ The blanks are load-bearing and are respected rather than filled. ``metrics/
    construct_validity.py`` deliberately writes an EMPTY ``kappa_auditor_A_vs_B`` on the
    ``auditors_agree`` / ``auditors_disagree`` strata, because those strata are *defined*
    by that agreement: the value is +1 and -1 by construction, a property of the split
    rather than a measurement, and it would be quotable off the figure as though it were a
    finding. Reading `None` as "0" or interpolating it would resurrect exactly that.
    """
    marks: List[KappaMark] = []
    for row in rows:
        stratum = str(row.get("stratum", ""))
        for series in KAPPA_SERIES:
            value = f(row, series.column)
            if value is None:
                continue
            marks.append(
                KappaMark(
                    stratum=stratum,
                    label=stratum_label(stratum),
                    series=series.column,
                    legend=series.legend,
                    value=value,
                    colour=series.colour,
                    marker=series.marker,
                    offset=series.offset,
                    size=series.size,
                )
            )
    return marks


# --------------------------------------------------------------------------- #
# the marginals — counted from the sheet, never taken on trust
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Marginals:
    """Who passed how many of the SAME items, plus the two cross-tab cells that matter.

    All counts are taken over the rows the human has actually annotated, so the three
    marginals share a denominator. ``*_pass_human_reject`` is the cell that separates "a
    conservative filter" from "a different opinion": auditor B contributes 2, auditor A 16.
    """

    n_annotated: int
    n_total: int
    n_human: int
    n_auditor_a: int
    n_auditor_b: int
    contested_n: int
    contested_with_a: int
    contested_with_b: int
    a_pass_human_reject: int
    b_pass_human_reject: int


def auditor_marginals(path: Path,
                      pass_labels: Sequence[str] = ("valid",)) -> Optional[Marginals]:
    """Count the D-G sheet. READ-ONLY; returns ``None`` when the sheet is not on disk."""
    path = Path(path)
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        return None

    def human(row: dict) -> Optional[bool]:
        raw = row.get("human_verdict")
        verdict = None if raw is None else str(raw).strip().lower()
        return None if not verdict else verdict in tuple(pass_labels)

    annotated = [r for r in rows if human(r) is not None]
    contested = [r for r in annotated
                 if bool(r.get("auditor_A_passed")) != bool(r.get("auditor_B_passed"))]
    return Marginals(
        n_annotated=len(annotated),
        n_total=len(rows),
        n_human=sum(1 for r in annotated if human(r)),
        n_auditor_a=sum(1 for r in annotated if r.get("auditor_A_passed")),
        n_auditor_b=sum(1 for r in annotated if r.get("auditor_B_passed")),
        contested_n=len(contested),
        contested_with_a=sum(1 for r in contested
                             if human(r) == bool(r.get("auditor_A_passed"))),
        contested_with_b=sum(1 for r in contested
                             if human(r) == bool(r.get("auditor_B_passed"))),
        a_pass_human_reject=sum(1 for r in annotated
                                if r.get("auditor_A_passed") and not human(r)),
        b_pass_human_reject=sum(1 for r in annotated
                                if r.get("auditor_B_passed") and not human(r)),
    )


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def _bare_axis(fig, rect, *, n_rows: int, x_lo: float, x_hi: float,
               bound_lo: float, bound_hi: float, ticks: Sequence[float],
               tick_labels: Optional[Sequence[str]] = None):
    """One row-oriented axis whose ruler visibly STOPS where its scale stops."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(SURFACE)
    for name, sp in ax.spines.items():
        sp.set_visible(name == "bottom")
    ax.spines["bottom"].set_color(BASE)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["bottom"].set_bounds(bound_lo, bound_hi)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(n_rows - 0.45, -0.75)
    ax.set_xticks(list(ticks))
    if tick_labels is not None:
        ax.set_xticklabels(list(tick_labels))
    ax.tick_params(axis="x", colors=INK2, labelsize=7.6, length=3, color=BASE)
    ax.tick_params(axis="y", length=0)
    for x in ticks:
        ax.plot([x, x], [n_rows - 0.45, -0.75], color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    return ax


def _block_header(ax, block: Block, *, n_rows: int) -> None:
    """Title + the question + the closed label set + the failure census, per axis."""
    tr = ax.transAxes
    unit = "counterfactual PAIRS" if block.source == "dg_pairs" else "contact sheets"
    top = next((r for r in block.rows if str(r.get("stratum")) == "all"), block.rows[0])
    n_total = int(float(top.get("n_total") or 0))
    head = f"{block.arm} · {block.source} — {n_total} {unit}"
    # header lines are placed in axes coords; a 1-row axis is short, so the offsets are
    # expressed per ROW rather than per axis height, or the text would collide.
    step = 0.30 / max(n_rows, 1)
    ax.text(0.0, 1.0 + step * 2.80, head, transform=tr, fontsize=10.2, color=INK,
            fontweight="bold", va="bottom")
    ax.text(0.0, 1.0 + step * 1.72, f"“{block.question}”", transform=tr, fontsize=8.0,
            color=INK, va="bottom")
    ax.text(0.0, 1.0 + step * 0.72,
            "own label set:  " + " / ".join(block.labels) + "     ·     " + census_line(block),
            transform=tr, fontsize=7.2, color=INK2, va="bottom", style="italic")


def _draw_rate_rows(ax, ests: Sequence[Estimate], colour: str, *,
                    x_count: float, x_rate: float) -> None:
    for i, est in enumerate(ests):
        emph = est.stratum == "all"
        if not est.annotated:
            ax.text(0.02, i, f"0 of {est.n_total} annotated — NO rate can be computed",
                    ha="left", va="center", fontsize=7.8, color=CRITICAL,
                    fontweight="bold")
            continue
        # the interval first, drawn as ENDPOINTS with caps: this is the mark, and the
        # point estimate is the decoration on it, not the other way round.
        ax.plot([est.lo, est.hi], [i, i], color=colour, lw=3.4 if emph else 2.2,
                solid_capstyle="butt", alpha=1.0 if emph else 0.75, zorder=3)
        for x in (est.lo, est.hi):
            ax.plot([x, x], [i - 0.20, i + 0.20], color=colour,
                    lw=1.5 if emph else 1.1, zorder=4)
        ax.plot([est.rate], [i], "o", ms=8.5 if emph else 7.0,
                mfc=SURFACE, mec=colour, mew=2.0 if emph else 1.5, zorder=5)
        ax.text(x_count, i, f"{est.n_pass}/{est.n_annotated}", ha="right", va="center",
                fontsize=7.8, color=INK if emph else INK2,
                fontweight="bold" if emph else "normal")
        ax.text(x_rate, i, f"{est.rate:.3f}  [{est.lo:.3f}, {est.hi:.3f}]", ha="right",
                va="center", fontsize=7.8, color=INK if emph else INK2,
                fontweight="bold" if emph else "normal")
    ax.set_yticks(range(len(ests)))
    ax.set_yticklabels(
        [("" if e.stratum == "all" else "   ") + e.label for e in ests],
        fontsize=8.4, color=INK)
    for tick, est in zip(ax.get_yticklabels(), ests):
        tick.set_fontweight("bold" if est.stratum == "all" else "normal")
        tick.set_color(INK if est.stratum == "all" else INK2)


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def plot_construct_validity(
    rows: Sequence[dict],
    out_path: Path,
    *,
    marginals: Optional[Marginals] = None,
    title: Optional[str] = None,
) -> Path:
    """Two panels: the rate with its interval (left), and the tie-break (right)."""
    blocks = build_blocks(rows)
    if not blocks:
        raise ValueError("no source blocks to draw")
    per_block = [(b, estimates(b)) for b in blocks]
    dg = next((pair for pair in per_block if pair[0].source == "dg_pairs"), None)
    marks = kappa_marks(dg[0].rows) if dg else []

    fig = new_figure((15.2, 10.4))

    # Layout: every axis in the figure is laid out from the SAME row height, so a row on
    # the left panel and a row on the right panel are the same size and the eye can carry
    # a horizontal read across. `HEADER_GAP` is the vertical space each block's three
    # header lines need.
    row_h, header_gap = 0.0345, 0.108
    top_y = 0.700

    # ---------------- panel A ------------------------------------------------ #
    left, width = 0.088, 0.372
    x_count, x_rate = 1.10, 1.55
    stack = []
    y_cursor = top_y
    for block, ests in per_block:
        h = row_h * len(ests)
        y_cursor -= h
        stack.append((block, ests, y_cursor, h))
        y_cursor -= header_gap

    for block, ests, y0, h in stack:
        ax = _bare_axis(fig, [left, y0, width, h], n_rows=len(ests),
                        x_lo=-0.012, x_hi=1.62, bound_lo=0.0, bound_hi=1.0,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
                        tick_labels=["0", ".25", ".50", ".75", "1"])
        if len(ests) > 1:
            for i in range(len(ests)):
                if i and i % 2 == 1:
                    ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0)
        _block_header(ax, block, n_rows=len(ests))
        _draw_rate_rows(ax, ests, block.colour, x_count=x_count, x_rate=x_rate)
        ax.text(x_count, -0.72, "pass / n", ha="right", va="center", fontsize=6.9,
                color=INK2, style="italic")
        ax.text(x_rate, -0.72, "rate  [95% Wilson]", ha="right", va="center",
                fontsize=6.9, color=INK2, style="italic")

    fig.text(left, 0.878, "A ·  the rate, with its interval", fontsize=12.6, color=INK,
             fontweight="bold", va="bottom")
    fig.text(left, 0.844,
             "THREE SOURCES · THREE VOCABULARIES · THREE RULERS.  Each block below\n"
             "has its own axis, its own question and its own closed label set.\n"
             "All three are proportions — which is exactly what makes the trap easy:\n"
             "“0.90 beats 0.51” is a sentence the units permit and the questions forbid.\n"
             "Within D-G the four indented strata RE-CUT the same 82 pairs — not\n"
             "independent samples, and not independent intervals.",
             fontsize=8.2, color=INK2, va="top", linespacing=1.55)

    # ---------------- panel B ------------------------------------------------ #
    rleft, rwidth = 0.578, 0.348
    kx_count = 1.32
    strata: List[str] = []
    for mark in marks:
        if mark.stratum not in strata:
            strata.append(mark.stratum)
    kappa_bottom = top_y - row_h * max(len(strata), 1)
    if marks:
        n_k = len(strata)
        kax = _bare_axis(fig, [rleft, kappa_bottom, rwidth, row_h * n_k],
                         n_rows=n_k, x_lo=-0.70, x_hi=1.46, bound_lo=-0.6, bound_hi=0.8,
                         ticks=[-0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
                         tick_labels=["-.50", "-.25", "0", ".25", ".50", ".75"])
        kax.axvspan(-NEAR_CHANCE, NEAR_CHANCE, color=BAND, zorder=0)
        kax.axvline(0.0, color=BASE, lw=1.4, zorder=2)
        index = {s: i for i, s in enumerate(strata)}
        for i in range(n_k):
            if i % 2 == 1:
                kax.axhspan(i - 0.5, i + 0.5, color=BAND, alpha=0.55, zorder=0)
        for mark in marks:
            y = index[mark.stratum] + mark.offset
            kax.plot([0.0, mark.value], [y, y], color=mark.colour, lw=1.0, alpha=0.55,
                     zorder=3)
            kax.plot([mark.value], [y], mark.marker, ms=mark.size, mfc=mark.colour,
                     mec=mark.colour, mew=1.4, zorder=5)
        for stratum, y in index.items():
            got = {m.series: m.value for m in marks if m.stratum == stratum}
            a = got.get("kappa_human_vs_auditor_A")
            b = got.get("kappa_human_vs_auditor_B")
            txt = (f"A {a:+.2f}   B {b:+.2f}" if a is not None and b is not None
                   else "")
            kax.text(kx_count, y, txt, ha="right", va="center", fontsize=7.8, color=INK2)
        kax.set_yticks(range(n_k))
        kax.set_yticklabels(
            [("" if s == "all" else "   ") + stratum_label(s) for s in strata],
            fontsize=8.4, color=INK)
        for tick, s in zip(kax.get_yticklabels(), strata):
            tick.set_fontweight("bold" if s == "all" else "normal")
            tick.set_color(INK if s == "all" else INK2)
        kax.text(0.0, -0.66, "chance", ha="center", va="center", fontsize=6.9,
                 color=INK2, style="italic")
        kax.text(kx_count, -0.66, "κ vs A / vs B", ha="right", va="center", fontsize=6.9,
                 color=INK2, style="italic")

        khandles = [Line2D([], [], marker=s.marker, ls="none", color=s.colour,
                           mfc=s.colour, mec=s.colour, ms=7.4, label=s.legend)
                    for s in KAPPA_SERIES]
        kax.legend(handles=khandles, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                   ncol=3, frameon=False, fontsize=7.6, labelcolor=INK2,
                   handletextpad=0.4, columnspacing=1.6, borderaxespad=0.0)

    fig.text(rleft, 0.878, "B ·  whose side does the human take?", fontsize=12.6,
             color=INK, fontweight="bold", va="bottom")
    fig.text(rleft, 0.844,
             "Cohen's κ between the recorded human verdict and each MLLM auditor, over the\n"
             "same pairs.  Positive = the human's pass/fail PATTERN tracks that auditor's.\n"
             "κ(A, B) is drawn only where the stratum is not defined by that agreement "
             "(±1 there by construction).\n"
             f"Shaded band |κ| ≤ {NEAR_CHANCE:g} is a drawn convention for “at or near "
             "chance”, not a test.",
             fontsize=8.2, color=INK2, va="top", linespacing=1.55)

    # ---- the marginals, in COUNTS, as the corrective ------------------------ #
    if marginals is not None:
        m = marginals
        mrows = [("human (the sheet)", m.n_human, INK),
                 ("auditor A · gpt-4o-mini", m.n_auditor_a, ORANGE),
                 ("auditor B · gpt-5.5", m.n_auditor_b, BLUE)]
        y0 = kappa_bottom - header_gap - row_h * len(mrows)
        mx = _bare_axis(fig, [rleft, y0, rwidth, row_h * len(mrows)], n_rows=len(mrows),
                        x_lo=-0.012, x_hi=1.62, bound_lo=0.0, bound_hi=1.0,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
                        tick_labels=["0", ".25", ".50", ".75", "1"])
        for i, (label, count, colour) in enumerate(mrows):
            share = count / m.n_annotated if m.n_annotated else 0.0
            mx.barh([i], [share], height=0.46, color=colour, alpha=0.85, zorder=3)
            mx.text(x_count, i, f"{count}/{m.n_annotated}", ha="right", va="center",
                    fontsize=7.8, color=INK if i == 0 else INK2,
                    fontweight="bold" if i == 0 else "normal")
            mx.text(x_rate, i, f"{share:.3f}", ha="right", va="center", fontsize=7.8,
                    color=INK if i == 0 else INK2,
                    fontweight="bold" if i == 0 else "normal")
        mx.set_yticks(range(len(mrows)))
        mx.set_yticklabels([r[0] for r in mrows], fontsize=8.4, color=INK2)
        mx.get_yticklabels()[0].set_color(INK)
        mx.get_yticklabels()[0].set_fontweight("bold")
        step = 0.30 / len(mrows)
        mx.text(0.0, 1.0 + step * 2.80,
                "…but the human is the MOST PERMISSIVE of the three",
                transform=mx.transAxes, fontsize=10.2, color=INK, fontweight="bold",
                va="bottom")
        mx.text(0.0, 1.0 + step * 1.72,
                "pass counts over the SAME items — a census, not an estimate, so no "
                "interval is drawn",
                transform=mx.transAxes, fontsize=8.0, color=INK, va="bottom")
        mx.text(0.0, 1.0 + step * 0.72,
                f"auditor B passes only {m.b_pass_human_reject} pair(s) the human rejects "
                f"(auditor A passes {m.a_pass_human_reject}) — B is nearly a SUBSET of the "
                "human read",
                transform=mx.transAxes, fontsize=7.2, color=INK2, va="bottom",
                style="italic")
        mx.text(x_count, -0.72, "pass / n", ha="right", va="center", fontsize=6.9,
                color=INK2, style="italic")
        mx.text(x_rate, -0.72, "share", ha="right", va="center", fontsize=6.9,
                color=INK2, style="italic")

        # ---- the tie-break itself, in counts ------------------------------- #
        if m.contested_n:
            cy = y0 - header_gap
            cx = fig.add_axes([rleft, cy, rwidth * (1.0 / 1.62), 0.030])
            cx.set_facecolor(SURFACE)
            for sp in cx.spines.values():
                sp.set_visible(False)
            cx.set_xticks([])
            cx.set_yticks([])
            cx.set_xlim(0, m.contested_n)
            cx.set_ylim(0, 1)
            cx.barh([0.5], [m.contested_with_b], height=1.0, color=BLUE, zorder=3)
            cx.barh([0.5], [m.contested_with_a], left=m.contested_with_b, height=1.0,
                    color=ORANGE, zorder=3)
            cx.text(m.contested_with_b / 2, 0.5, f"with B  {m.contested_with_b}",
                    ha="center", va="center", fontsize=8.4, color=SURFACE,
                    fontweight="bold")
            cx.text(m.contested_with_b + m.contested_with_a / 2, 0.5,
                    f"with A  {m.contested_with_a}", ha="center", va="center",
                    fontsize=8.4, color=SURFACE, fontweight="bold")
            fig.text(rleft, cy + 0.072,
                     f"the tie-break, in counts — the {m.contested_n} CONTESTED pairs",
                     fontsize=10.2, color=INK, fontweight="bold", va="bottom")
            fig.text(rleft, cy + 0.063,
                     "where A and B disagreed, so exactly one of them matches the human on "
                     "each pair\n(A ≡ ¬B here: ONE measurement, mirrored — not two)",
                     fontsize=7.2, color=INK2, va="top", style="italic",
                     linespacing=1.5)
    else:
        fig.text(rleft, 0.40,
                 "marginal pass counts not drawn:\n"
                 "data/human_validation_fairness/pairs.jsonl is not on disk, so κ cannot "
                 "be read next to how STRICT each rater was.",
                 fontsize=8.4, color=CRITICAL, va="top", linespacing=1.5)

    # ---------------- chrome ------------------------------------------------- #
    fig.suptitle(title or "Construct validity is a RATE with an interval — and a tie-break "
                          "two MLLM auditors could not supply",
                 fontsize=15.0, color=INK, x=0.006, ha="left", y=0.988,
                 fontweight="bold")
    fig.text(0.006, 0.946,
             "133 hand-adjudicated sheets from three staged directories.  n = 20–82, so "
             "THE INTERVAL IS THE RESULT and no point estimate is quoted alone.\n"
             "Precedent this figure exists to obey: the D-G probe measured a clean-flip "
             "rate of 0.85 on 20 scenes; on the full 41 the same question came back 0.66 "
             "(auditor A) / 0.22 (auditor B).",
             fontsize=8.6, color=INK2, ha="left", va="top", linespacing=1.55)

    read = _read_in_3s(per_block, marginals)
    fig.text(0.006, 0.182, read, fontsize=8.4, color=INK, ha="left", va="top",
             linespacing=1.62)
    return save(fig, out_path)


def _read_in_3s(per_block, marginals: Optional[Marginals]) -> str:
    """The caption, built from the drawn numbers so it cannot drift from the figure."""
    lines: List[str] = []
    if marginals is not None and marginals.contested_n:
        m = marginals
        missed = m.n_human - (m.n_auditor_b - m.b_pass_human_reject)
        lines.append(
            f"Read in 3 s:   On the {m.contested_n} contested pairs the human sides with "
            f"the STRICT auditor B {m.contested_with_b}–{m.contested_with_a}, and "
            "κ(human, B) > 0 > κ(human, A) in every stratum —"
        )
        lines.append(
            "                     the tie the two auditors could not break IS broken, on "
            "the record, and the decision to stop D-G stands on an artefact rather than "
            "on a memory."
        )
        lines.append(
            f"BUT the human passes {m.n_human}/{m.n_annotated} — MORE than auditor A "
            f"({m.n_auditor_a}) and far more than auditor B ({m.n_auditor_b}). B's passes "
            f"are nearly a SUBSET of the human's ({m.b_pass_human_reject} of "
            f"{m.n_auditor_b} are human rejects, against {m.a_pass_human_reject} of "
            f"{m.n_auditor_a}),"
        )
        lines.append(
            f"so B is a conservative filter that still misses {missed} of the "
            f"{m.n_human} usable pairs.  “Sides with B” is a statement about PATTERN, not "
            "about strictness — and the corpus is about HALF usable, not the 5-of-41 per "
            "attribute that the A ∩ B intersection implied."
        )
    dg = next((p for p in per_block if p[0].source == "dg_pairs"), None)
    if dg is not None:
        lines.append("Mechanism, straight off the label census: " + census_line(dg[0])
                     + " — “moved” dominates, which is the recorded reason D-G stopped "
                       "(the attribute editor re-renders the whole frame instead of "
                       "editing locally).")
    v3 = next((p for p in per_block if p[0].source == "ds_v3"), None)
    if v3 is not None:
        top = next((e for e in v3[1] if e.stratum == "all"), None)
        if top is not None and top.annotated:
            lines.append(
                f"RESOLVED 2026-08-06: ds_v3 is annotated {top.n_pass}/{top.n_annotated} "
                "here, against an older prose note of “a human found 30 of those 31 "
                "invalid”. That 1/31 was the count from THREE GEOMETRIC PROXY GATES, not a "
                "human read"
            )
            lines.append(
                "                     (21 no face box · 15 skin_frac>0.08 · 3 crowd scenes); "
                "the proxies ask about structural anchoring, this sheet asks about "
                "photographic plausibility. Both are right, about different questions."
            )
            lines.append(
                "                     Scope: these 31 scenes were NEVER judged — the "
                "published D-S null runs on OmniEdit v4 and overlaps them by zero, so the "
                "rate that constrains it is ds_v4's 0.850."
            )
    lines.append(
        "Caveats: construct validity is a RATE on n annotated sheets, never an assertion. "
        "Wilson intervals are asymmetric and are drawn as ENDPOINTS, never as ± error "
        "bars. The three blocks are never differenced, ranked or pooled."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI — appended to scripts/07_plot_figures.sh; zero API, idempotent
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse
    import csv

    from edit_judge_bias.data.manifest_utils import default_root
    from edit_judge_bias.experiments.plot_results import figure_prefix

    root = default_root()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics-dir", type=Path,
                    default=root / "results" / "v2_fairness_ds" / "metrics")
    ap.add_argument("--figures-dir", type=Path,
                    default=root / "results" / "v2_fairness_ds" / "figures")
    ap.add_argument("--pairs", type=Path,
                    default=root / "data" / "human_validation_fairness" / "pairs.jsonl",
                    help="D-G adjudication sheet, read READ-ONLY for the marginals only")
    args = ap.parse_args(argv)

    path = Path(args.metrics_dir) / "construct_validity.csv"
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        print("skip construct_validity.png (needs construct_validity.csv)")
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("skip construct_validity.png (construct_validity.csv has no rows)")
        return 0

    marginals = auditor_marginals(Path(args.pairs))
    if marginals is None:
        print(f"note: {args.pairs} not found — drawing without the marginal panel")
    # The pilot tree's `pilot_` prefix rule applies here too.
    name = f"{figure_prefix(Path(args.metrics_dir))}construct_validity.png"
    out = plot_construct_validity(rows, Path(args.figures_dir) / name,
                                  marginals=marginals)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
