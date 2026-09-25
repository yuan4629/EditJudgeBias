"""F2 / F3 — the two claim forests.

``plot_claim_a_forest``
    Source CSV: ``results/v2/metrics/claim_a.csv``.
    NOT ``scoring_shift.csv`` — that table carries only the point estimate. It has no
    CI, no BH q, no ``inside_placebo_bound`` and no per-judge n, so a figure drawn from
    it could not distinguish "significant deflation" from "inside the sham equivalence
    bound", which is the single distinction claim A rests on. ``claim_a.csv`` is the
    frozen table with all four.

``plot_claim_b_forest``
    Source CSV: ``results/v2/metrics/claim_b.csv``.
    Grouped by ``bias_type`` rather than by judge, because the finding is about the
    CUE: ``region_annotation`` is the one perturbation that reorders across judges and
    across both anchors, while ``text_overlay`` / ``brightness`` deflate *order-
    preservingly*. A judge-major layout scatters that pattern over five blocks.

    ★ Significance styling, and why it is written defensively. The rank panel is driven
    by ``rho_ci_excludes_zero`` (a cluster bootstrap, correct since it was written). The
    accuracy panel used to be driven by the CI alone, because the ``significant_bh``
    stored beside it came from a McNemar over ~18 within-turn comparisons per turn and
    ignored that clustering — under it, **10 of 13 "significant" cells contradicted
    their own clustered CI**. That recompute landed 2026-08-03: BH now runs on
    ``accuracy_p_cluster`` (per-turn net discordance, Wilcoxon over the 48/30 turns),
    so ``q_value`` under ``family="claim_B_acc"`` IS the cluster-level q and is what the
    panel reads. The CI rule remains as the fallback for any table that predates it, and
    a row whose ``family`` says something else falls back rather than trusting a q from
    a different family. The subtitle prints which rule was used, so the figure can never
    silently claim more than the table supports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from matplotlib.lines import Line2D

from edit_judge_bias.visualization.style import (
    BAND,
    BASE,
    BLUE,
    GRID,
    INK,
    INK2,
    MUTED,
    ORANGE,
    ROSTER_NOTE,
    SURFACE,
    f,
    judge_sort_key,
    marker_for,
    new_figure,
    save,
    short,
    truthy,
)

#: `sham` is a control arm, not a cue. It carries an empty `q_value` BY DESIGN — it is
#: the perturbation that defines the equivalence bound, so it is not tested against
#: it. Rendering it in the main strip would show five blank-q rows that read as "not
#: significant", i.e. as evidence of robustness that was never measured.
CONTROL_CUES = ("sham",)

#: The cue whose sign is the opposite of every other cue's — pixel artefacts are
#: punished, fabricated social proof is rewarded. Emphasis colour, one entity.
INFLATING_CUE = "bandwagon"


def split_control_rows(
    rows: Sequence[dict], control_cues: Sequence[str] = CONTROL_CUES
) -> Tuple[List[dict], List[dict]]:
    """(claim rows, control rows). The control strip is drawn below a rule."""
    control = [r for r in rows if r.get("bias_type") in set(control_cues)]
    claim = [r for r in rows if r.get("bias_type") not in set(control_cues)]
    return claim, control


def _cue_order(rows: Sequence[dict], key: str) -> List[str]:
    """Cues ordered by mean |effect| across judges, largest first."""
    agg: Dict[str, List[float]] = {}
    for r in rows:
        v = f(r, key)
        if v is not None:
            agg.setdefault(r["bias_type"], []).append(abs(v))
    return sorted(agg, key=lambda c: -float(np.mean(agg[c])))


class ClaimALayout(NamedTuple):
    """Row geometry of the claim-A forest, separated out so it can be asserted on.

    ``rule_y`` is the horizontal rule; every control row sits below it and every
    claim row above it. That separation is the whole reason the layout is a named
    thing rather than a loop body.
    """

    judges: List[str]
    cues: List[str]
    bands: List[Tuple[str, float, float, bool]]   # cue, y_lo, y_hi, is_control
    y_by_key: Dict[Tuple[str, str], float]        # (judge, cue) -> y
    rule_y: Optional[float]
    y_max: float


def claim_a_layout(
    rows: Sequence[dict], control_cues: Sequence[str] = CONTROL_CUES
) -> ClaimALayout:
    """Lay the forest out: cue bands by |effect|, control cues last, below a rule."""
    claim_rows, control_rows = split_control_rows(rows, control_cues)
    if not claim_rows:
        raise ValueError("claim_a.csv held no non-control rows")
    judges = sorted({r["judge_model"] for r in claim_rows}, key=judge_sort_key)
    cues = _cue_order(claim_rows, "mean_shift")
    per = len(judges)

    bands: List[Tuple[str, float, float, bool]] = []
    y_by_key: Dict[Tuple[str, str], float] = {}
    y = 0.0
    for cue in cues:
        for ji, judge in enumerate(judges):
            y_by_key[(judge, cue)] = y + ji
        bands.append((cue, y - 0.5, y + per - 0.5, False))
        y += per

    rule_y: Optional[float] = None
    if control_rows:
        rule_y = y - 0.5
        base = y + 0.55
        for cue in sorted({r["bias_type"] for r in control_rows}):
            for ji, judge in enumerate(judges):
                y_by_key[(judge, cue)] = base + ji
            bands.append((cue, base - 0.5, base + per - 0.5, True))
            base += per
        y = base + 0.55
    return ClaimALayout(judges, cues, bands, y_by_key, rule_y, y)


def plot_claim_a_forest(
    rows: Sequence[dict], out_path: Path, *, title: Optional[str] = None
) -> Path:
    """One interval per (cue, judge); cues ordered by |mean shift|; sham in its own strip."""
    layout = claim_a_layout(rows)
    judges, cues = layout.judges, layout.cues
    by_key = {(r["judge_model"], r["bias_type"]): r for r in rows}

    per = len(judges)
    n_rows = len(layout.y_by_key)
    height = 0.152 * n_rows + 3.05
    top_pad = 1.62 / height
    fig = new_figure((9.6, height))
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    lows = [f(r, "ci_low") for r in rows if f(r, "ci_low") is not None]
    highs = [f(r, "ci_high") for r in rows if f(r, "ci_high") is not None]
    x_lo, x_hi = min(lows) - 0.20, max(highs) + 0.20
    x_q = x_hi + (x_hi - x_lo) * 0.150
    x_n = x_q - (x_hi - x_lo) * 0.076

    tick_pos: List[float] = []
    tick_lab: List[str] = []

    def draw(cue: str, colour: str) -> None:
        for judge in judges:
            row = by_key.get((judge, cue))
            if row is None:
                continue
            y = layout.y_by_key[(judge, cue)]
            tick_pos.append(y)
            tick_lab.append(short(judge))
            v, lo, hi = f(row, "mean_shift"), f(row, "ci_low"), f(row, "ci_high")
            sig = truthy(row, "significant_bh")
            placebo = truthy(row, "inside_placebo_bound")
            solid = bool(sig) and not placebo
            ax.plot([lo, hi], [y, y], color=colour, lw=2.0 if solid else 1.1,
                    solid_capstyle="round", zorder=4)
            ax.plot([v], [y], marker_for(judge), ms=6.6 if solid else 5.8,
                    mfc=colour if solid else SURFACE, mec=colour,
                    mew=1.6 if placebo else 1.2, zorder=5)
            n = f(row, "n")
            ax.text(x_n, y, "—" if n is None else f"{int(n)}", ha="right", va="center",
                    fontsize=6.3, color=MUTED)
            q = f(row, "q_value")
            label = "—" if q is None else ("<.001" if q < 0.001 else f"{q:.3f}".lstrip("0"))
            ax.text(x_q, y, label, ha="right", va="center", fontsize=6.3,
                    color=INK if solid else MUTED,
                    fontweight="bold" if solid else "normal")

    for i, (cue, lo_y, hi_y, is_control) in enumerate(layout.bands):
        colour = (MUTED if is_control
                  else (ORANGE if cue == INFLATING_CUE else BLUE))
        draw(cue, colour)
        if not is_control and i % 2 == 1:
            ax.axhspan(lo_y, hi_y, color=BAND, zorder=0)

    rule_y = layout.rule_y
    ax.axvline(0, color=BASE, linewidth=1.2, zorder=2)
    if rule_y is not None:
        ax.axhline(rule_y + 0.3, color=BASE, linewidth=1.0, zorder=3)

    ax.set_ylim(-0.9, layout.y_max - 0.75)
    ax.invert_yaxis()
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=6.1, color=MUTED)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlim(x_lo, x_hi)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5, length=3, color=BASE)
    ax.set_xlabel("mean score shift, biased − original  (fine_score points, 3–30)",
                  fontsize=9, color=INK2)

    # Cue names sit in the outer gutter, anchored to the figure's own left edge so a
    # long name can never be clipped no matter how the margins move.
    fig.subplots_adjust(left=0.190, right=0.862, top=1 - top_pad,
                        bottom=0.74 / height + 0.030)
    box = ax.get_position()
    x_cue = (0.006 - box.x0) / box.width
    trans = ax.get_yaxis_transform()
    for cue, lo_y, hi_y, _is_control in layout.bands:
        emph = cue == INFLATING_CUE
        ax.text(x_cue, (lo_y + hi_y) / 2, cue, transform=trans, ha="left",
                va="center", fontsize=8.0, color=ORANGE if emph else INK,
                fontweight="bold" if emph else "normal")

    ax.text(x_n, -0.95, "n", ha="right", va="center", fontsize=6.6, color=MUTED,
            style="italic")
    ax.text(x_q, -0.95, "q (BH)", ha="right", va="center", fontsize=6.6, color=MUTED,
            style="italic")
    if rule_y is not None:
        # The sham band's left half is empty by construction (a null perturbation sits
        # on zero), so the caption goes there rather than in the gutter.
        ax.text(0.012, rule_y + (len(judges) + 1) / 2.0,
                "the control arm is a visually null JPEG round-trip.\n"
                "It DEFINES the equivalence bound, so it carries no q by design —\n"
                "an empty q here is NOT “not significant”.",
                transform=trans, ha="left", va="center", fontsize=7.4, color=MUTED,
                linespacing=1.5)

    fig.suptitle(title or "Claim A — the judges are not invariant to quality-preserving "
                          "perturbation",
                 fontsize=14, color=INK, x=0.006, ha="left", y=0.995, fontweight="bold")
    fig.text(0.006, 1 - 0.030,
             "611 breadth samples, 5 judges, 95% bootstrap CI on the paired shift.  "
             "Marker shape = judge (fixed roster order, top to bottom in every band).\n"
             "Filled = BH-significant AND outside the sham equivalence bound.  Hollow "
             "with a heavy ring = inside the sham bound → not an effect, whatever q says.",
             fontsize=8.3, color=INK2, ha="left", va="top", linespacing=1.5)

    handles = [
        Line2D([], [], marker="o", ls="-", lw=2.0, color=BLUE, mfc=BLUE, mec=BLUE,
               ms=6.6, label="deflation, significant"),
        Line2D([], [], marker="o", ls="-", lw=1.1, color=BLUE, mfc=SURFACE, mec=BLUE,
               ms=5.8, mew=1.2, label="not significant"),
        Line2D([], [], marker="o", ls="none", color=BLUE, mfc=SURFACE, mec=BLUE,
               ms=5.8, mew=1.6, label="inside the sham bound"),
        Line2D([], [], marker="o", ls="-", lw=2.0, color=ORANGE, mfc=ORANGE,
               mec=ORANGE, ms=6.6, label="bandwagon — the only cue that INFLATES, 5/5"),
    ] + [
        Line2D([], [], marker=marker_for(j), ls="none", color=INK2, mfc=SURFACE,
               mec=INK2, ms=5.8, label=short(j)) for j in judges
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.004, 1 - 0.100),
               ncol=5, frameon=False, fontsize=7.6, labelcolor=INK2, handlelength=2.0,
               columnspacing=1.4, handletextpad=0.5)

    fig.text(0.006, 0.012,
             "Read in 3 s:  distraction, zoom_inset, text_overlay, region_annotation "
             "and padding deflect the score on 5/5 judges; bandwagon is the lone "
             "inflater.\n" + ROSTER_NOTE,
             fontsize=8.0, color=INK, ha="left", va="bottom", linespacing=1.6)
    return save(fig, out_path)


# --------------------------------------------------------------------------- #
# F3
# --------------------------------------------------------------------------- #
SOURCE_STYLE = {
    "EBench-18K": (BLUE, "o"),
    "ImagenHub": (ORANGE, "s"),
}


def _acc_significant(row: dict) -> bool:
    """Prefer the cluster-aware q; else the CI rule.

    ★ The recompute landed 2026-08-03 and it is `q_value`, BH-attached to
    `accuracy_p_cluster` under `family="claim_B_acc"` — so `q_value` here IS the
    cluster-level q, and `significant_bh` is now safe to trust. Both the old
    speculative name and the real one are accepted, and a row whose `family` says
    otherwise falls back to the CI rule rather than trusting a q from another family.
    """
    q = f(row, "accuracy_q_cluster")
    if q is None and (row.get("family") or "").strip() == "claim_B_acc":
        q = f(row, "q_value")
    if q is not None:
        return q < 0.05
    lo, hi = f(row, "accuracy_delta_ci_low"), f(row, "accuracy_delta_ci_high")
    if lo is None or hi is None:
        return False
    return lo > 0 or hi < 0


def plot_claim_b_forest(
    rows: Sequence[dict], out_path: Path, *, title: Optional[str] = None
) -> Path:
    """Two panels — Δρ and Δ derived pairwise accuracy — grouped by cue, not by judge."""
    usable = [r for r in rows if f(r, "spearman_delta") is not None]
    if not usable:
        raise ValueError("claim_b.csv held no rows with a spearman_delta")
    sources = sorted({r.get("anchor_source", "") for r in usable})
    judges = sorted({r["judge_model"] for r in usable}, key=judge_sort_key)

    # Bands ordered by how often the rho CI excludes zero: the headline cue floats up
    # by its own evidence, not by hand.
    def excl_count(cue: str) -> Tuple[int, float]:
        got = [r for r in usable if r["bias_type"] == cue]
        n = sum(1 for r in got if truthy(r, "rho_ci_excludes_zero"))
        return (-n, float(np.mean([f(r, "spearman_delta") or 0.0 for r in got])))

    cues = sorted({r["bias_type"] for r in usable}, key=excl_count)
    by_key = {(r["judge_model"], r.get("anchor_source", ""), r["bias_type"]): r
              for r in usable}

    per = len(judges) * len(sources)
    n_rows = len(cues) * per
    height = 0.165 * n_rows + 3.15
    fig = new_figure((11.4, height))
    ax_rho, ax_acc = fig.subplots(1, 2, sharey=True,
                                  gridspec_kw=dict(wspace=0.055, width_ratios=[1, 1]))

    cluster_aware = any(
        f(r, "accuracy_q_cluster") is not None
        or ((r.get("family") or "").strip() == "claim_B_acc" and f(r, "q_value") is not None)
        for r in usable
    )

    rows_drawn: List[Tuple[float, str]] = []
    y = 0.0
    for i, cue in enumerate(cues):
        for si, src in enumerate(sources):
            for ji, judge in enumerate(judges):
                row = by_key.get((judge, src, cue))
                if row is None:
                    continue
                colour, marker = SOURCE_STYLE.get(src, (BLUE, "o"))
                yy = y + si * len(judges) + ji
                rows_drawn.append((yy, short(judge)))
                # panel 1 — rho, styled by its own cluster-bootstrap CI
                d = f(row, "spearman_delta")
                lo, hi = f(row, "spearman_delta_ci_low"), f(row, "spearman_delta_ci_high")
                excl = bool(truthy(row, "rho_ci_excludes_zero"))
                ax_rho.plot([lo, hi], [yy, yy], color=colour, lw=2.0 if excl else 1.1,
                            solid_capstyle="round", zorder=4)
                ax_rho.plot([d], [yy], marker, ms=6.4 if excl else 5.6,
                            mfc=colour if excl else SURFACE, mec=colour, mew=1.2,
                            zorder=5)
                # panel 2 — accuracy, in percentage points
                a = f(row, "accuracy_delta")
                alo, ahi = f(row, "accuracy_delta_ci_low"), f(row, "accuracy_delta_ci_high")
                asig = _acc_significant(row)
                if a is not None:
                    ax_acc.plot([alo * 100, ahi * 100], [yy, yy], color=colour,
                                lw=2.0 if asig else 1.1, solid_capstyle="round", zorder=4)
                    ax_acc.plot([a * 100], [yy], marker, ms=6.4 if asig else 5.6,
                                mfc=colour if asig else SURFACE, mec=colour, mew=1.2,
                                zorder=5)
        if i % 2 == 1:
            for ax in (ax_rho, ax_acc):
                ax.axhspan(y - 0.5, y + per - 0.5, color=BAND, zorder=0)
        y += per

    for ax, label in ((ax_rho, "Δ Spearman ρ (judge vs human), after − before"),
                      (ax_acc, "Δ derived pairwise accuracy (percentage points)")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvline(0, color=BASE, linewidth=1.2, zorder=2)
        ax.set_xlabel(label, fontsize=9, color=INK2)
        ax.tick_params(axis="x", colors=INK2, labelsize=8.3, length=3, color=BASE)
        ax.tick_params(axis="y", length=0)

    ax_rho.set_ylim(-0.8, y - 0.3)
    ax_rho.invert_yaxis()
    ax_rho.set_yticks([yy for yy, _ in rows_drawn])
    ax_rho.set_yticklabels([lbl for _, lbl in rows_drawn], fontsize=6.4, color=MUTED)
    ax_rho.tick_params(axis="y", pad=1.5)

    # Two-level left gutter, laid out from the axes' real position so that neither the
    # cue name nor the source name can be clipped by the figure edge.
    fig.subplots_adjust(left=0.205, right=0.988, top=1 - 1.72 / height,
                        bottom=0.98 / height + 0.028)
    box = ax_rho.get_position()
    trans = ax_rho.get_yaxis_transform()
    x_cue = (0.006 - box.x0) / box.width          # flush with the figure's left edge
    x_src = (0.118 - box.x0) / box.width          # between cue name and tick labels
    for i, cue in enumerate(cues):
        emph = i == 0
        ax_rho.text(x_cue, i * per + (per - 1) / 2, cue, transform=trans,
                    ha="left", va="center", fontsize=9.2,
                    color=INK if emph else INK2,
                    fontweight="bold" if emph else "normal")
        for si, src in enumerate(sources):
            colour = SOURCE_STYLE.get(src, (BLUE, "o"))[0]
            lo_y = i * per + si * len(judges)
            ax_rho.text(x_src, lo_y + (len(judges) - 1) / 2, src, transform=trans,
                        rotation=90, ha="center", va="center", fontsize=6.4,
                        color=colour)

    fig.suptitle(title or "Claim B — only one cue reorders; the rest deflate "
                          "order-preservingly",
                 fontsize=14, color=INK, x=0.006, ha="left", y=0.995, fontweight="bold")
    counts = {s: (max((f(r, "n_items") or 0) for r in usable
                      if r.get("anchor_source") == s),
                  max((f(r, "n_clusters") or 0) for r in usable
                      if r.get("anchor_source") == s),
                  max((f(r, "n_pairs") or 0) for r in usable
                      if r.get("anchor_source") == s))
              for s in sources}
    counts_note = "  ·  ".join(
        f"{s}: {int(it)} items / {int(cl)} turns / {int(pr)} derived pairs"
        for s, (it, cl, pr) in counts.items())
    fig.text(0.006, 1 - 0.030,
             "Human-anchored subset, two anchors of different label kinds.  "
             + counts_note + ".\n"
             "95% CLUSTER bootstrap resampling whole turns.  Filled = the interval "
             "excludes zero.\n"
             + ("Accuracy panel: BH q on the CLUSTER-level test (per-turn net discordance, "
                "Wilcoxon over turns) < .05."
                if cluster_aware else
                "Accuracy panel is styled from its CI, not from significant_bh — the "
                "stored McNemar test ignores within-turn clustering."),
             fontsize=8.3, color=INK2, ha="left", va="top", linespacing=1.5)

    handles = [Line2D([], [], marker=SOURCE_STYLE.get(s, (BLUE, "o"))[1], ls="-",
                      lw=2.0, color=SOURCE_STYLE.get(s, (BLUE, "o"))[0],
                      mfc=SOURCE_STYLE.get(s, (BLUE, "o"))[0],
                      mec=SOURCE_STYLE.get(s, (BLUE, "o"))[0], ms=6.4,
                      label=f"{s} — CI excludes 0") for s in sources]
    handles += [Line2D([], [], marker=SOURCE_STYLE.get(s, (BLUE, "o"))[1], ls="-",
                       lw=1.1, color=SOURCE_STYLE.get(s, (BLUE, "o"))[0], mfc=SURFACE,
                       mec=SOURCE_STYLE.get(s, (BLUE, "o"))[0], ms=5.6,
                       label=f"{s} — straddles 0") for s in sources]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.004, 1 - 1.06 / height),
               ncol=4, frameon=False, fontsize=7.8, labelcolor=INK2, handlelength=2.0,
               columnspacing=1.6, handletextpad=0.5)

    fig.text(0.006, 0.010,
             "Read in 3 s:  region_annotation is the one cue whose ρ interval clears "
             "zero on 8 of 10 (judge × anchor) cells — it changes the ORDER.\n"
             "text_overlay and brightness mostly straddle zero: they move the number "
             "without moving the ranking.\n" + ROSTER_NOTE,
             fontsize=8.0, color=INK, ha="left", va="bottom", linespacing=1.6)
    return save(fig, out_path)


# --------------------------------------------------------------------------- #
# F13 — claim A decomposed into the three judged dimensions (WP-F1b)          #
# --------------------------------------------------------------------------- #
#: Panels in the order the rubric asks the dimensions, which is also the order that
#: matters for reading the figure: the leftmost is the dimension the cue cannot
#: legitimately move, so an interval that clears zero THERE is the form of this effect
#: that cannot be read as a justified deduction.
DIMENSION_PANELS = (
    ("instruction_adherence", "instruction adherence"),
    ("editing_quality", "editing quality"),
    ("detail_preservation", "detail preservation"),
)

#: What each panel's title says about how far the rubric licenses a deduction there.
DIMENSION_NOTE = {
    "instruction_adherence": "cue is applied after the edit → cannot legitimately move",
    "editing_quality": "partly licensed by the rubric",
    "detail_preservation": "rubric asks for “irrelevant regions preserved”",
}


def dimension_shared_xlim(rows: Sequence[dict]) -> Tuple[float, float]:
    """The ONE x range all three panels of F13 use, over every dimension's intervals.

    A pure function rather than four lines inside the plotter because it is the
    figure's load-bearing property, not a styling detail: the instruction-adherence
    effects are the smallest of the three, so per-panel autoscaling would redraw them
    at the size of the detail-preservation effects and invert what the figure says.
    """
    lows = [f(r, "ci_low") for r in rows if f(r, "ci_low") is not None]
    highs = [f(r, "ci_high") for r in rows if f(r, "ci_high") is not None]
    span = max(highs) - min(lows)
    return min(lows) - 0.05 * span, max(highs) + 0.05 * span


def plot_claim_a_dimension_forest(
    rows: Sequence[dict], out_path: Path, *, title: Optional[str] = None
) -> Path:
    """F13. Source CSV: ``results/v2/metrics/claim_a_by_dimension.csv``.

    Three panels, ONE cue order and ONE x-axis across all three. Both are load-bearing,
    not cosmetic:

    - Ordering each panel by its own effect size would put a different cue at the top of
      each, and this figure exists to be read ACROSS panels ("this cue moves detail
      preservation but not instruction adherence"). The shared order is claim A's own —
      cues by the summed effect, which is exactly ``claim_a.csv``'s ``mean_shift``.
    - Letting each panel autoscale would delete the finding. The instruction-adherence
      effects are the smallest of the three, and a per-panel axis would redraw them the
      size of the detail-preservation effects, which is the opposite of what the data
      says.

    ``sham`` is drawn below a rule in every panel, as in F2: it is the equivalence
    bound, not a cue, and it carries no q by design.
    """
    summed: Dict[Tuple[str, str], float] = {}
    for r in rows:
        v = f(r, "mean_shift")
        if v is not None:
            key = (r["judge_model"], r["bias_type"])
            summed[key] = summed.get(key, 0.0) + v
    # Layout built from the SUMMED rows, so the cue order is claim A's own rather than
    # a fourth ordering nobody declared.
    layout = claim_a_layout([
        {"judge_model": j, "bias_type": c, "mean_shift": v}
        for (j, c), v in sorted(summed.items())
    ])
    judges = layout.judges
    by_key = {(r["judge_model"], r["bias_type"], r["dimension"]): r for r in rows}

    x_lo, x_hi = dimension_shared_xlim(rows)

    height = 0.130 * len(layout.y_by_key) + 3.1
    fig = new_figure((12.6, height))
    axes = fig.subplots(1, len(DIMENSION_PANELS), sharey=True)

    def sig_down(cue: str, dim: str) -> int:
        n = 0
        for j in judges:
            row = by_key.get((j, cue, dim))
            if row is not None and truthy(row, "significant_bh"):
                v = f(row, "mean_shift")
                if v is not None and v < 0:
                    n += 1
        return n

    for ax, (dim, label) in zip(axes, DIMENSION_PANELS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

        for i, (cue, lo_y, hi_y, is_control) in enumerate(layout.bands):
            if not is_control and i % 2 == 1:
                ax.axhspan(lo_y, hi_y, color=BAND, zorder=0)
            colour = (MUTED if is_control
                      else (ORANGE if cue == INFLATING_CUE else BLUE))
            for judge in judges:
                row = by_key.get((judge, cue, dim))
                if row is None:
                    continue
                y = layout.y_by_key[(judge, cue)]
                v, lo, hi = f(row, "mean_shift"), f(row, "ci_low"), f(row, "ci_high")
                sig = bool(truthy(row, "significant_bh"))
                ax.plot([lo, hi], [y, y], color=colour, lw=2.0 if sig else 1.0,
                        solid_capstyle="round", zorder=4)
                ax.plot([v], [y], marker_for(judge), ms=6.2 if sig else 5.4,
                        mfc=colour if sig else SURFACE, mec=colour, mew=1.2, zorder=5)

        ax.axvline(0, color=BASE, linewidth=1.2, zorder=2)
        if layout.rule_y is not None:
            ax.axhline(layout.rule_y + 0.3, color=BASE, linewidth=1.0, zorder=3)
        ax.set_xlim(x_lo, x_hi)
        ax.tick_params(axis="x", colors=INK2, labelsize=8.0, length=3, color=BASE)
        ax.tick_params(axis="y", length=0)   # sharey still draws stubs on panels 2-3
        ax.set_title(label, fontsize=10.0, color=INK, fontweight="bold", pad=13)
        ax.text(0.5, 1.006, DIMENSION_NOTE[dim], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.4, color=INK2)

    ax0 = axes[0]
    ax0.set_ylim(-0.9, layout.y_max - 0.75)
    ax0.invert_yaxis()
    # Ticks driven by the layout's own y positions. Rebuilding the label order from
    # `cues x judges` would agree today and silently disagree the moment the layout
    # inserts a band anywhere but the end.
    ordered = sorted(layout.y_by_key.items(), key=lambda kv: kv[1])
    ax0.set_yticks([y for _k, y in ordered])
    ax0.set_yticklabels([short(j) for (j, _cue), _y in ordered],
                        fontsize=6.0, color=MUTED)
    ax0.tick_params(axis="y", length=0, pad=1.5)

    fig.subplots_adjust(left=0.150, right=0.995, top=1 - 1.62 / height,
                        bottom=0.78 / height + 0.030, wspace=0.055)

    box = ax0.get_position()
    x_cue = (0.006 - box.x0) / box.width
    trans = ax0.get_yaxis_transform()
    for cue, lo_y, hi_y, _is_control in layout.bands:
        emph = cue == INFLATING_CUE
        ax0.text(x_cue, (lo_y + hi_y) / 2, cue, transform=trans, ha="left",
                 va="center", fontsize=8.0, color=ORANGE if emph else INK,
                 fontweight="bold" if emph else "normal")

    bottom_frac = 0.78 / height + 0.030
    fig.text(0.5, bottom_frac * 0.50,
             "mean score shift, biased − original, in points of that single dimension "
             "(1–10 each; the three sum to fine_score)",
             fontsize=9, color=INK2, ha="center", va="center")

    # ★ The caption is COMPUTED from the table, never asserted. This project has already
    # shipped a figure whose hardcoded subtitle had been refuted by a later arm.
    ia_all = [c for c in layout.cues
              if sig_down(c, "instruction_adherence") == len(judges)]
    ia_most = [c for c in layout.cues
               if len(judges) > sig_down(c, "instruction_adherence") >= len(judges) - 1]
    # The case §7 turns on: the cue deflates the SUM on the whole panel, but almost
    # none of it is on the dimension the cue cannot legitimately move.
    ia_spared = [c for c in layout.cues
                 if sig_down(c, "instruction_adherence") <= 1
                 and sig_down(c, "editing_quality") >= len(judges) - 1]
    fig.suptitle(
        title or "Claim A by dimension — separating the deduction the rubric licenses "
                 "from the one it does not",
        fontsize=14, color=INK, x=0.006, ha="left", y=0.995, fontweight="bold",
    )
    fig.text(
        0.006, 1 - 0.034,
        "Same 611 breadth samples, same complete-case cells as claim_a.csv — the "
        "three dimensions ADD UP to its fine_score shift.\n95% bootstrap CI; "
        "filled = BH-significant within that dimension's own family."
        + (f"\nInstruction adherence deflated on {len(judges)}/{len(judges)} judges: "
           + ", ".join(ia_all) + ";" if ia_all else "")
        + (f" on {len(judges) - 1}/{len(judges)}: " + ", ".join(ia_most) + "."
           if ia_most else "")
        + (f"\nOn ≤1/{len(judges)} while editing quality falls on "
           f"≥{len(judges) - 1}/{len(judges)}: " + ", ".join(ia_spared)
           + " — for those the deduction has a defensible reading."
           if ia_spared else ""),
        fontsize=8.3, color=INK2, ha="left", va="top", linespacing=1.5,
    )
    fig.text(0.006, 0.012, ROSTER_NOTE, fontsize=7.6, color=MUTED, ha="left",
             va="bottom")
    return save(fig, out_path)
