"""F12 — what a quality-preserving cue does to an editor leaderboard.

Source CSV: ``results/v2/metrics/leaderboard_simulation.csv`` (WP-A1e).

Two panels, because the table holds two different threat models and averaging them
would hide the one that matters:

**left — one editor carries the cue alone.** Each row is a (source, cue); each marker
is a judge's *mean* places lost across that board's editors, with a hairline out to the
*worst* editor's loss. A filled marker means the board's leader lost first place. This
is the deployment sentence: an editor that captions, pads or annotates its own outputs
falls, and nothing about its edits changed.

**right — every editor carries the cue.** Kendall tau between the clean board and the
cued board. tau = 1 is the no-op a uniform deduction is supposed to produce, so the
distance from the right-hand edge IS the finding: uniform deflation is *not* order-
preserving at board level, even though it is close to order-preserving item by item.

Deliberately NOT drawn: the change in agreement with the *human* board. It is in the
table with a cluster-bootstrap interval, and **every one of its 40 intervals covers
zero** — 48 turns cannot resolve a tau shift of 0.05. Drawing forty near-zero dots
would invite a reader to find a pattern in noise. The count is computed from the rows
and printed in the footnote, so if it ever stops being all forty the figure says so
instead of the caption being wrong.

    ⚠️ That is the whole reading of this figure and it must not be softened in either
    direction: the board is **unstable** under a cue that changed no edit, and the data
    do **not** show it becoming less accurate. Instability is the finding; invalidity
    at board level is not measurable at this n.

⚠️ The two boards are not the same size — EBench-18K ranks 17 editors, ImagenHub 8 —
so a place lost is worth more on the smaller board. The axis is places, not fractions,
because "fell 9 places" is the quantity a leaderboard reader recognises; the editor
count is printed on every row label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from matplotlib.lines import Line2D

from edit_judge_bias.visualization.style import (
    BAND,
    BASE,
    CRITICAL,
    GRID,
    INK,
    INK2,
    MUTED,
    ROSTER_NOTE,
    SURFACE,
    f,
    judge_sort_key,
    marker_for,
    new_figure,
    save,
    short,
    style_axes,
    truthy,
)

#: Cue order: most damaging first, so the eye reads the panel top-down as a severity
#: list rather than alphabetically.
CUE_ORDER = ("text_overlay", "region_annotation", "padding", "brightness")


def _rows_by(rows: Sequence[dict], scenario: str) -> List[dict]:
    return [r for r in rows if r.get("scenario") == scenario]


def _row_keys(rows: Sequence[dict]) -> List[tuple]:
    """(source, cue) rows, ordered by source then by the declared cue order."""
    seen = {(r["anchor_source"], r["bias_type"]) for r in rows}
    order = {cue: i for i, cue in enumerate(CUE_ORDER)}
    return sorted(seen, key=lambda k: (k[0], order.get(k[1], 99)))


def plot_leaderboard_simulation(
    rows: Sequence[dict], out_path: Path
) -> Optional[Path]:
    single = _rows_by(rows, "single_editor")
    uniform = _rows_by(rows, "all_editors")
    if not single or not uniform:
        return None

    keys = _row_keys(single)
    judges = sorted({r["judge_model"] for r in single}, key=judge_sort_key)
    editors_by_source: Dict[str, int] = {
        r["anchor_source"]: int(f(r, "n_editors") or 0) for r in single
    }

    height = 1.05 + 0.42 * len(keys)
    fig = new_figure((12.4, height + 1.5))
    left = fig.add_axes([0.255, 0.185, 0.375, 0.70])
    right = fig.add_axes([0.700, 0.185, 0.275, 0.70])
    for ax in (left, right):
        style_axes(ax, xgrid=True)
        ax.set_ylim(len(keys) - 0.5, -0.5)
        ax.set_yticks(range(len(keys)))

    # alternating bands per source block, so the two boards read as two blocks
    for i, (source, _cue) in enumerate(keys):
        if source == keys[0][0]:
            for ax in (left, right):
                ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0)

    jitter = {j: (k - (len(judges) - 1) / 2) * 0.135 for k, j in enumerate(judges)}

    # --- left: one editor cued alone ---------------------------------------- #
    for i, key in enumerate(keys):
        for judge in judges:
            row = next((r for r in single
                        if (r["anchor_source"], r["bias_type"]) == key
                        and r["judge_model"] == judge), None)
            if row is None:
                continue
            mean = f(row, "mean_rank_drop")
            worst = f(row, "max_rank_drop")
            y = i + jitter[judge]
            if mean is not None and worst is not None:
                left.plot([mean, worst], [y, y], color=BASE, linewidth=0.9, zorder=2)
                left.plot([worst], [y], marker="|", color=MUTED, markersize=5, zorder=3)
            if mean is None:
                continue
            filled = truthy(row, "leader_loses_top1")
            left.plot(
                [mean], [y], marker=marker_for(judge), markersize=6.0,
                markerfacecolor=(CRITICAL if filled else SURFACE),
                markeredgecolor=(CRITICAL if filled else INK2),
                markeredgewidth=1.0, linestyle="none", zorder=4,
            )
    left.axvline(0, color=INK, linewidth=0.9, zorder=1)
    left.set_xlabel("places lost by the cued editor  (mean → worst editor on that board)",
                    fontsize=8.5, color=INK2)
    left.set_title("one editor carries the cue alone", fontsize=10, color=INK,
                   loc="left", pad=8)

    # --- right: everybody cued ---------------------------------------------- #
    for i, key in enumerate(keys):
        for judge in judges:
            row = next((r for r in uniform
                        if (r["anchor_source"], r["bias_type"]) == key
                        and r["judge_model"] == judge), None)
            tau = None if row is None else f(row, "kendall_tau_clean_vs_biased")
            if tau is None:
                continue
            changed = truthy(row, "top1_changed")
            right.plot(
                [tau], [i + jitter[judge]], marker=marker_for(judge), markersize=6.0,
                markerfacecolor=(CRITICAL if changed else SURFACE),
                markeredgecolor=(CRITICAL if changed else INK2),
                markeredgewidth=1.0, linestyle="none", zorder=4,
            )
    right.axvline(1.0, color=INK, linewidth=0.9, zorder=1)
    right.set_xlim(0.35, 1.06)
    right.set_xlabel("Kendall τ, clean board vs cued board", fontsize=8.5, color=INK2)
    right.set_title("every editor carries the cue", fontsize=10, color=INK,
                    loc="left", pad=8)

    labels = [f"{cue}" for _src, cue in keys]
    left.set_yticklabels(labels, fontsize=8.5, color=INK)
    right.set_yticklabels([])
    # source brackets down the far left, one per block
    for source in dict.fromkeys(src for src, _ in keys):
        idx = [i for i, (s, _) in enumerate(keys) if s == source]
        mid = (min(idx) + max(idx)) / 2
        left.text(-0.36, mid, f"{source}\n{editors_by_source.get(source, 0)} editors",
                  transform=left.get_yaxis_transform(), ha="center", va="center",
                  fontsize=8.5, color=INK2, linespacing=1.35)
        left.plot([-0.245, -0.245], [min(idx) - 0.34, max(idx) + 0.34],
                  transform=left.get_yaxis_transform(), color=BASE, linewidth=1.0,
                  clip_on=False)

    handles = [
        Line2D([], [], marker=marker_for(j), linestyle="none", markersize=6,
               markerfacecolor=SURFACE, markeredgecolor=INK2, label=short(j))
        for j in judges
    ]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=CRITICAL, markeredgecolor=CRITICAL,
                          label="filled = top-1 editor changes"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.055), handletextpad=0.35,
               columnspacing=1.4, labelcolor=INK2)
    # Counted from the rows, never remembered: if the human-agreement result ever
    # stops being "all of them cover zero", this sentence changes with it.
    n_excl = sum(1 for r in uniform if truthy(r, "delta_tau_ci_excludes_zero"))
    covered = ("all " if n_excl == 0 else f"{len(uniform) - n_excl} of ")
    fig.text(0.5, 0.012,
             "leaderboards are turn-centred means of fine_score; the cue changes no "
             "edit, only its presentation.  Agreement with the human board is in "
             f"leaderboard_simulation.csv: {covered}{len(uniform)} cluster-bootstrap "
             "intervals cover zero.\n" + ROSTER_NOTE,
             ha="center", fontsize=7.2, color=MUTED, linespacing=1.5)
    return save(fig, out_path)


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse
    import csv

    from edit_judge_bias.data.manifest_utils import default_root
    from edit_judge_bias.experiments.plot_results import figure_prefix

    root = default_root()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics-dir", type=Path,
                    default=root / "results" / "v2" / "metrics")
    ap.add_argument("--figures-dir", type=Path,
                    default=root / "results" / "v2" / "figures")
    args = ap.parse_args(argv)

    path = Path(args.metrics_dir) / "leaderboard_simulation.csv"
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        print("skip leaderboard_simulation.png (needs leaderboard_simulation.csv)")
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    name = f"{figure_prefix(Path(args.metrics_dir))}leaderboard_simulation.png"
    out = plot_leaderboard_simulation(rows, Path(args.figures_dir) / name)
    print(f"wrote {out}" if out else "skip leaderboard_simulation.png (no usable rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
