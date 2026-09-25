"""F5 / F7 — the two protocol-level figures.

``plot_rr_vs_cr``
    Source CSVs: ``results/v2/metrics/position.csv`` + ``pairwise_consistency.csv``.
    The point of the pair is that they answer different questions: CR is "does the
    judge give the same answer to the SAME question twice", RR is "does it give the
    same answer when only the display slot moves". A judge can be high on one and low
    on the other, and qwen3.5-plus is — CR 0.867 against RR 0.352 — which is why the
    earlier inference "qwen is simply noisy" was wrong. Both axes are proportions with
    a known denominator, so both get Wilson intervals; bare dots would hide that
    qwen's n is smaller than everyone else's.

``plot_mitigation_tradeoff``
    Source CSV: ``results/v2/metrics/mitigation_swap_average.csv``.
    Swap-and-average is the only mitigation level in the study that works, and it
    works by *declining to answer*: it buys agreement among the pairs it still decides
    and pays in coverage. A connected dot plot is the form that makes a trade visible
    — the arrow's direction IS the finding. qwen's arrow runs almost the width of the
    plot and lands below chance; it is not clipped, because it is the point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from matplotlib.lines import Line2D

from edit_judge_bias.visualization.style import (
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
    wilson_from_rate,
)


def _rr_chance(row: dict) -> Optional[float]:
    """Chance RR for a judge that answers the two orders independently.

    In the swapped order the slots are flipped, so the *content* winner is unchanged
    only when the judge picks display-A in one order and display-B in the other::

        RR_chance = pA_orig (1 - pA_swap) + (1 - pA_orig) pA_swap

    Computed per judge from that judge's own slot-A rates, which is the only honest
    baseline — a judge with a strong slot-A pull has a different chance level from one
    without.
    """
    a, b = f(row, "posA_rate_original"), f(row, "posA_rate_swapped")
    if a is None or b is None:
        return None
    return a * (1 - b) + (1 - a) * b


def _cr_chance(row: dict) -> Optional[float]:
    """Chance CR for two independent asks of the same question, ties ignored.

    ``pA^2 + (1-pA)^2``. Ties are not in ``pairwise_consistency.csv``, and ignoring
    them makes this an UPPER bound on chance agreement — the conservative direction
    for a figure whose claim is "qwen is above chance on CR".
    """
    a = f(row, "posA_rate_original")
    if a is None:
        return None
    return a * a + (1 - a) * (1 - a)


def plot_rr_vs_cr(
    position_rows: Sequence[dict],
    consistency_rows: Sequence[dict],
    out_path: Path,
    *,
    emphasise: str = "qwen3.5-plus",
    title: Optional[str] = None,
) -> Path:
    """Slot robustness (RR) against self-consistency (CR), with Wilson intervals."""
    pos = {r["judge_model"]: r for r in position_rows}
    con = {r["judge_model"]: r for r in consistency_rows}
    judges = sorted(set(pos) & set(con), key=judge_sort_key)
    if not judges:
        raise ValueError("no judge appears in both position.csv and pairwise_consistency.csv")

    rr_chance = [c for c in (_rr_chance(pos[j]) for j in judges) if c is not None]
    cr_chance = [c for c in (_cr_chance(pos[j]) for j in judges) if c is not None]
    rr_line = sum(rr_chance) / len(rr_chance) if rr_chance else 0.5
    cr_line = sum(cr_chance) / len(cr_chance) if cr_chance else 0.5

    fig = new_figure((9.4, 6.9))
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(BASE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    x_lo, x_hi, y_lo, y_hi = 0.44, 1.02, 0.20, 1.03
    # the quadrant that names the study's sharpest finding
    ax.add_patch(plt_rect(cr_line, y_lo, x_hi - cr_line, rr_line - y_lo))
    ax.axhline(rr_line, color=BASE, linewidth=1.0, zorder=2)
    ax.axvline(cr_line, color=BASE, linewidth=1.0, zorder=2)

    n_full = max((f(pos[j], "n") or 0) for j in judges)
    for judge in judges:
        p, c = pos[judge], con[judge]
        rr, n_rr = f(p, "rr"), f(p, "n")
        cr, n_cr = f(c, "cr"), f(c, "n")
        if rr is None or cr is None:
            continue
        emph = judge == emphasise
        reduced = (n_rr or 0) < n_full or (n_cr or 0) < n_full
        colour = BLUE if emph else MUTED
        rr_lo, rr_hi = wilson_from_rate(rr, n_rr or 0)
        cr_lo, cr_hi = wilson_from_rate(cr, n_cr or 0)
        ax.plot([cr, cr], [rr_lo, rr_hi], color=colour, lw=1.4, solid_capstyle="round",
                zorder=4, alpha=0.85)
        ax.plot([cr_lo, cr_hi], [rr, rr], color=colour, lw=1.4, solid_capstyle="round",
                zorder=4, alpha=0.85)
        ax.plot([cr], [rr], marker_for(judge), ms=11, mfc=colour, mec=SURFACE, mew=2.0,
                zorder=6)
        label = f"{short(judge)}\nRR {rr:.3f} · CR {cr:.3f}"
        if reduced:
            label += (f"\nn = {int(n_rr or 0)} / {int(n_cr or 0)} of {int(n_full)}  "
                      "— unmappable \"winner\":\"C\" rows dropped")
        dy = 0.030 if rr < 0.9 else -0.040
        ax.text(cr, rr + dy, label,
                ha="center", va="bottom" if dy > 0 else "top", fontsize=8.0,
                color=INK if emph else INK2, linespacing=1.4,
                fontweight="bold" if emph else "normal")

    ax.text(x_hi - 0.008, y_hi - 0.010, "consistent  AND  slot-robust",
            ha="right", va="top", fontsize=9.0, color=INK2)
    ax.text(x_hi - 0.008, y_lo + 0.012,
            "consistent BUT captured by the slot\n"
            "— answers the same way 87% of the time,\nand still flips 65% of pairs "
            "when the order swaps",
            ha="right", va="bottom", fontsize=9.0, color=INK, linespacing=1.4,
            fontweight="bold")
    ax.text(x_lo + 0.008, y_hi - 0.010,
            "noisy but slot-robust\n(empty — nobody is here)",
            ha="left", va="top", fontsize=8.6, color=MUTED, linespacing=1.4)
    ax.text(x_lo + 0.008, y_lo + 0.012, "noisy AND slot-driven",
            ha="left", va="bottom", fontsize=8.6, color=MUTED)

    ax.text(cr_line - 0.005, y_lo + 0.11, f"chance CR ≈ {cr_line:.2f}", rotation=90,
            ha="right", va="bottom", fontsize=7.2, color=MUTED)
    ax.text(x_lo + 0.008, rr_line + 0.008,
            f"chance RR ≈ {rr_line:.2f}  (independent answers, each judge's own "
            "slot-A rate)", ha="left", va="bottom", fontsize=7.2, color=MUTED)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("CR — consistency rate: same question asked twice, same verdict",
                  fontsize=9.5, color=INK2)
    ax.set_ylabel("RR — robustness rate: same content winner after the A/B swap",
                  fontsize=9.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, color=BASE)

    fig.suptitle(title or "A judge disagreeing with humans does not mean it disagrees "
                          "with itself",
                 fontsize=14, color=INK, x=0.006, ha="left", y=0.985, fontweight="bold")
    fig.text(0.006, 0.940,
             "616 human-anchored pairs, each judged in both display orders and asked a "
             "second time unchanged.  Bars are 95% Wilson intervals.\n"
             "One judge is in the accent hue because it is the finding; the rest are "
             "de-emphasised grey, with identity carried by the printed label and the "
             "marker shape.",
             fontsize=8.4, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.006, 0.014,
             "Read in 3 s:  four judges sit top-right.  qwen3.5-plus sits bottom-right "
             "— high self-agreement, chance-level slot robustness.  That is a STABLE "
             "bias, not noise.\n" + ROSTER_NOTE,
             fontsize=8.1, color=INK, ha="left", va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=0.088, right=0.988, top=0.845, bottom=0.135)
    return save(fig, out_path)


def plt_rect(x, y, w, h):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, facecolor="#f4f3ef", edgecolor="none", zorder=1)


# --------------------------------------------------------------------------- #
# F7
# --------------------------------------------------------------------------- #
def plot_mitigation_tradeoff(
    rows: Sequence[dict],
    out_path: Path,
    *,
    start_arm: str = "base_order_only",
    end_arm: str = "reconciled_strict",
    emphasise: str = "qwen3.5-plus",
    title: Optional[str] = None,
) -> Path:
    """Connected dot plot: what swap-and-average buys, and what it costs."""
    by: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        by.setdefault(r["judge_model"], {})[r["arm"]] = r
    judges = sorted([j for j in by if start_arm in by[j] and end_arm in by[j]],
                    key=judge_sort_key)
    if not judges:
        raise ValueError(f"no judge has both {start_arm!r} and {end_arm!r}")

    fig = new_figure((10.4, 6.8))
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(BASE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0.5, color=BASE, linewidth=1.0, zorder=2)
    ax.text(0.005, 0.505, "chance (0.50)", ha="left", va="bottom", fontsize=7.4,
            color=MUTED, transform=ax.get_yaxis_transform())

    # Head labels are hand-anchored: the four working judges land inside a 0.04-tall
    # band, so an automatic offset piles them on top of each other. The Δ numbers move
    # to the table block below rather than riding the marks.
    label_at = {
        "kimi-k2.5": (0.0, +0.022, "center", "bottom"),
        "gemini-3.5-flash": (-0.022, 0.0, "right", "center"),
        "gpt-5.5": (-0.006, -0.020, "right", "top"),
        "gpt-4o-viescore": (0.0, -0.020, "center", "top"),
        "qwen3.5-plus": (0.0, -0.022, "center", "top"),
    }
    deltas = []
    for judge in judges:
        a, b = by[judge][start_arm], by[judge][end_arm]
        emph = judge == emphasise
        colour = ORANGE if emph else MUTED
        x0, y0 = f(a, "coverage"), f(a, "acc_on_decided")
        x1, y1 = f(b, "coverage"), f(b, "acc_on_decided")
        for row, x, y in ((a, x0, y0), (b, x1, y1)):
            lo, hi = wilson_from_rate(y, f(row, "n_decided") or 0)
            ax.plot([x, x], [lo, hi], color=colour, lw=1.3, alpha=0.8,
                    solid_capstyle="round", zorder=4)
            clo, chi = wilson_from_rate(f(row, "coverage"), f(row, "n_pairs") or 0)
            ax.plot([clo, chi], [y, y], color=colour, lw=1.3, alpha=0.8,
                    solid_capstyle="round", zorder=4)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                    lw=2.4 if emph else 1.6, shrinkA=7.5, shrinkB=9.5,
                                    mutation_scale=15), zorder=5)
        ax.plot([x0], [y0], marker_for(judge), ms=8.5, mfc=SURFACE, mec=colour,
                mew=1.8, zorder=6)
        ax.plot([x1], [y1], marker_for(judge), ms=9.5, mfc=colour, mec=SURFACE,
                mew=1.8, zorder=6)
        dx, dy, ha, va = label_at.get(judge, (0.0, -0.022, "center", "top"))
        ax.text(x1 + dx, y1 + dy, short(judge), ha=ha, va=va,
                fontsize=9.2 if emph else 8.4, color=INK if emph else INK2,
                fontweight="bold" if emph else "normal", zorder=7)
        deltas.append((judge, (x1 - x0) * 100, (y1 - y0) * 100, emph))

    # The numbers, once, in the empty middle-left — direct labels stay sparing.
    tx, ty = 0.275, 0.845
    ax.text(tx, ty, "what the trade costs and buys", fontsize=8.4, color=INK,
            fontweight="bold", ha="left", va="top")
    ax.text(tx, ty - 0.033, f"{'':<12}{'coverage':>11}{'agreement':>12}",
            fontsize=7.4, color=MUTED, ha="left", va="top", family="monospace")
    for k, (judge, d_cov, d_acc, emph) in enumerate(
            sorted(deltas, key=lambda t: t[1])):
        ax.text(tx, ty - 0.062 - 0.030 * k,
                f"{short(judge):<12}{d_cov:>+9.1f} pp{d_acc:>+9.1f} pp",
                fontsize=7.4, color=INK if emph else INK2, ha="left", va="top",
                family="monospace", fontweight="bold" if emph else "normal")

    q = by.get(emphasise, {}).get(end_arm)
    if q is not None:
        ax.annotate(
            "reconciliation cannot repair a judge that never tracked\n"
            "human preference — it only exposes it: coverage 0.98 → 0.31\n"
            "and agreement lands BELOW chance",
            xy=(f(q, "coverage"), f(q, "acc_on_decided")),
            xytext=(0.275, 0.598), fontsize=8.5, color=INK, linespacing=1.5,
            ha="left", va="top",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                            connectionstyle="arc3,rad=0.16"), zorder=7)

    ax.set_xlim(0.24, 1.03)
    ax.set_ylim(0.28, 1.00)
    ax.set_xlabel("coverage — share of decisive human-labelled pairs the protocol "
                  "still answers", fontsize=9.5, color=INK2)
    ax.set_ylabel("agreement with humans, among the pairs it answers",
                  fontsize=9.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, color=BASE)

    handles = [
        Line2D([], [], marker="o", ls="none", color=MUTED, mfc=SURFACE, mec=MUTED,
               ms=8.5, mew=1.8, label=f"start: {start_arm.replace('_', ' ')}"),
        Line2D([], [], marker="o", ls="none", color=MUTED, mfc=MUTED, mec=SURFACE,
               ms=9.5, mew=1.8, label=f"end: {end_arm.replace('_', ' ')}"),
        Line2D([], [], ls="-", lw=1.3, color=MUTED, label="95% Wilson interval"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.0,
              labelcolor=INK2, handlelength=1.8, handletextpad=0.6,
              borderaxespad=0.4)

    fig.suptitle(title or "The one mitigation that works, works by declining to answer",
                 fontsize=14, color=INK, x=0.006, ha="left", y=0.985, fontweight="bold")
    fig.text(0.006, 0.940,
             "438 decisive human-labelled pairs. Every pair was judged in both display "
             "orders, so reconciling the two verdicts costs no extra API call.\n"
             "A pairwise row carries only a winner, so this is verdict RECONCILIATION, "
             "not averaging: the strict rule discards any pair the two orders did not "
             "agree on.",
             fontsize=8.4, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.006, 0.014,
             "Read in 3 s:  four arrows go up-and-left — a few points of agreement "
             "bought with 7–31 pp of coverage.  qwen's arrow crosses the whole plot "
             "and lands below chance.\n" + ROSTER_NOTE,
             fontsize=8.1, color=INK, ha="left", va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=0.074, right=0.988, top=0.845, bottom=0.135)
    return save(fig, out_path)
