"""F9 — robustness at one layer does not imply robustness at another.

Source CSVs (all frozen; this module recomputes nothing):

``results/v2/metrics/claim_a_by_dimension.csv``
    absolute-score layer (``asc``, one cell per cue × judged dimension).
``results/v2/metrics/claim_b.csv``
    ranking layer.
``results/v2/metrics/position.csv``
    protocol layer.

Why the figure exists
---------------------
§5.4's finding — *a judge that is robust at one layer is not thereby robust at
another* — is the one result in this study with no counterpart in the competing
work, and until now it lived only as three lines of prose. ``gemini-3.5-flash``
has the largest per-item score change of the five judges (last on **absolute
scores**) and the verdicts that best survive a display swap (first on
**protocol**, RR 0.899); ``gpt-4o-viescore`` is third on absolute scores and the
**worst** judge on **ranking** (``region_annotation`` Δρ −0.141 / −0.213, derived
pairwise accuracy −9.0pp). 8 of the 20 adjacent-layer orderings invert.

★ 2026-09-26 — the absolute layer now uses the invariance operator the paper's
Method defines: the per-item |score change| on each judged dimension (``asc`` in
``claim_a_by_dimension.csv``), averaged over the 36 cue × dimension cells. Until
then it was the mean over the 12 cues of |``mean_shift``| in ``claim_a.csv``, the
size of each cue's AVERAGE signed change of the 3-30 sum. Opposite-sign changes
cancel inside that average (across items, and across the three dimensions inside
the sum), so it measured the systematic shift rather than how far the scores
move. It put ``gpt-4o-viescore`` first on absolute scores (``sham`` +0.007), and
the figure read "a rubric buys absolute-score stability, not ranking or protocol
stability"; that reading holds for the signed shift only. The inversion count
went from 9/20 to 8/20.

The form is a parallel-coordinates plot **on rank**, because the finding is
ordinal: the claim is "the ordering of the five judges is not the same ordering
at each layer", and a crossing line is exactly that statement. The three layer
scalars are not commensurable and are never drawn against one shared numeric
scale — see the red line below.

★ THE RED LINE: THREE UNITS, NEVER ONE RULER
--------------------------------------------
The three axis scalars are measured in three different units:

======================  ========================================  ==============
layer                   scalar                                    unit
======================  ========================================  ==============
absolute score          mean ``asc`` (per-item |Δ|) over the 36    rating points
                        claim-A cue × dimension cells              (1-10 scale)
ranking                 mean |spearman_delta| over the claim-B     Spearman ρ
                        (cue × anchor) cells                       (unitless)
protocol / position     RR                                         proportion
======================  ========================================  ==============

Normalising the three onto a common 0–1 axis would manufacture a comparison the
data cannot support (0.73 rating points and Δρ 0.062 are not "the same size"
under any defensible transform), so the top panel carries **rank only** and the
three raw scalars are shown underneath on **three independent numeric axes**,
one per layer, each with its own tick range and its own "more robust →"
direction. This mirrors how F1 draws the D-S MDE ticks in units of SD(paired
difference) with the legend stating out loud that it is NOT SD(score).

Reading the panels together is also what keeps the ordinal top panel honest: the
bottom strips show the *spacing*, so a rank gap that is really a hair (qwen vs
viescore on absolute scores, 0.714 vs 0.732; kimi vs qwen on ranking, 0.027 vs
0.035) cannot be mistaken for qwen's protocol collapse (RR 0.352 against
everyone else's 0.69–0.90).

Sign conventions, spelled out because two of the three invert
-------------------------------------------------------------
* absolute score — **lower is more robust** (a smaller mean per-item |score change|);
* ranking — **lower is more robust** (a smaller mean absolute Δρ);
* protocol — **higher is more robust** (RR is a *retention* rate).

``rank_judges`` takes the direction explicitly for that reason; there is no
default, so a new layer cannot silently inherit the wrong one.

Scope caveats that travel with the scalars
------------------------------------------
* The absolute-score scalar is a mean over the **36 cells of the 12 claim-A cues
  × 3 judged dimensions**; ``sham`` carries ``family="control"`` and is excluded,
  because it is the perturbation that *defines* the equivalence bound and
  averaging it in would reward a judge for drifting on the placebo.
* Being a per-item |change|, the absolute-score scalar counts a judge's own
  instability against it, with no directional bias required:
  ``gemini-3.5-flash``'s own ``sham`` cells average 1.69 points, more than its 36
  cue cells (1.29). Its last place on this layer is a fact about how stable its
  scores are, not evidence that the cues move it most. The figure prints both
  numbers.
* The ranking scalar averages **8 cells** (4 cues × 2 anchors) — a much smaller
  and differently-chosen cue set than the absolute-score scalar's 12 cues. The two
  axes therefore rank the judges over different perturbation sets by
  construction; that is a property of the collected grid, not a defect, but it
  is the reason the axes are never differenced or summed.
* The protocol scalar is a single number per judge with no cue dimension at all.
* **The ranking axis depends on which claim-B scalar is chosen, and the figure
  says so out loud.** ``mean |Δρ|`` is a MAGNITUDE scalar and puts
  ``gpt-4o-viescore`` last (0.062), which is the reading §5.4 makes — its
  ``region_annotation`` Δρ (−0.141 / −0.213) and derived accuracy delta (−9.0pp)
  are the largest in the whole claim-B table. Counting instead the cells whose ρ
  interval excludes zero puts ``gemini-3.5-flash`` last (5 of 8) with viescore
  joint-second (3 of 8) — that scalar answers "on how many cells is a reordering
  *detectable*", not "how big is the reordering". Both are defensible and the
  layers cross under either; the footnote prints the alternative so the choice
  is never hidden.

None of the three is a "robustness score" for a judge. Each is a per-layer
summary whose only job is to induce an ordering within its own layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

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
)

#: ``sham`` is the placebo arm, not a cue. ``claim_a_by_dimension.csv`` stamps it
#: ``family="control"`` and every cue cell ``family="claim_A_<dimension>"``; the
#: absolute-score scalar keeps only the cue cells.
CLAIM_A_FAMILY = "claim_A"

#: The two judges whose trajectories carry the finding: gemini-3.5-flash (last on
#: absolute scores, first on protocol) and gpt-4o-viescore (last on ranking). Two
#: series hues, everything else in the de-emphasis ink — the rule in ``style``.
EMPHASIS: Dict[str, str] = {
    "gpt-4o-viescore": ORANGE,
    "gemini-3.5-flash": BLUE,
}

LOWER_IS_BETTER = "lower"
HIGHER_IS_BETTER = "higher"


class Layer(NamedTuple):
    """One robustness layer: its scalar per judge, its unit, and its direction."""

    key: str
    title: str
    scalar: str          # what the number IS, printed on the figure
    unit: str            # the unit it is in, printed on the figure
    better: str          # LOWER_IS_BETTER / HIGHER_IS_BETTER
    values: Dict[str, float]
    support: Dict[str, int]   # how many cells each judge's scalar averages
    fmt: str = "{:.3f}"


# --------------------------------------------------------------------------- #
# the three layer scalars
# --------------------------------------------------------------------------- #
def absolute_score_robustness(claim_a_dim_rows: Sequence[dict]) -> Layer:
    """Mean ``asc`` over the claim-A cue × dimension cells, in rating points (1-10).

    One cell is one (cue, judged dimension) row of ``claim_a_by_dimension.csv``, and
    its ``asc`` is the mean over items of |cued score − un-cued score| — the
    invariance cell of the paper's Method. Lower = more robust. ``family="control"``
    rows (``sham``) are dropped: the placebo defines the equivalence bound and is not
    one of the cues being resisted.

    A row without a ``dimension`` column is refused rather than read: ``claim_a.csv``
    has an ``asc`` column too, but on the 3-30 sum, where changes on different
    dimensions cancel before the absolute value is taken.
    """
    values: Dict[str, List[float]] = {}
    for row in claim_a_dim_rows:
        if "dimension" not in row:
            raise ValueError(
                "absolute_score_robustness reads claim_a_by_dimension.csv (one row per "
                "cue x dimension); got a row with no `dimension` column")
        if not (row.get("family") or "").startswith(CLAIM_A_FAMILY + "_"):
            continue
        cell = f(row, "asc")
        if cell is None:
            continue
        values.setdefault(row["judge_model"], []).append(cell)
    return Layer(
        key="absolute",
        title="1 · absolute score",
        scalar="mean per-item |Δ| over 36 cue × dimension cells",
        unit="rating points (1-10 scale)",
        better=LOWER_IS_BETTER,
        values={j: sum(v) / len(v) for j, v in values.items() if v},
        support={j: len(v) for j, v in values.items() if v},
        fmt="{:.3f}",
    )


def ranking_robustness(claim_b_rows: Sequence[dict]) -> Layer:
    """Mean |Δρ| over the claim-B (cue × anchor) cells. Unitless Spearman ρ.

    Lower = more robust. The absolute value matters: a cue that *raises* agreement
    has also moved the ranking, and a signed mean would let two cues cancel and
    report a reordering judge as a stable one.
    """
    values: Dict[str, List[float]] = {}
    for row in claim_b_rows:
        delta = f(row, "spearman_delta")
        if delta is None:
            continue
        values.setdefault(row["judge_model"], []).append(abs(delta))
    return Layer(
        key="ranking",
        title="2 · ranking (human agreement)",
        scalar="mean |Δρ| over the 8 claim-B cue × anchor cells",
        unit="Spearman ρ (unitless)",
        better=LOWER_IS_BETTER,
        values={j: sum(v) / len(v) for j, v in values.items() if v},
        support={j: len(v) for j, v in values.items() if v},
        fmt="{:.3f}",
    )


def position_robustness(position_rows: Sequence[dict]) -> Layer:
    """RR — the share of pairs whose content winner survives the display swap.

    Higher = more robust. Taken verbatim from ``position.csv``; nothing is
    recomputed here.
    """
    values: Dict[str, float] = {}
    support: Dict[str, int] = {}
    for row in position_rows:
        rr = f(row, "rr")
        if rr is None:
            continue
        values[row["judge_model"]] = rr
        n = f(row, "n")
        support[row["judge_model"]] = int(n) if n is not None else 0
    return Layer(
        key="protocol",
        title="3 · protocol (display order)",
        scalar="RR — verdict survives an A/B slot swap",
        unit="proportion of judged pairs",
        better=HIGHER_IS_BETTER,
        values=values,
        support=support,
        fmt="{:.3f}",
    )


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def rank_judges(values: Dict[str, float], better: str) -> Dict[str, int]:
    """1 = most robust … n = least. Exact ties share the lower (better) rank.

    ``better`` is mandatory — two of this figure's three layers invert, and a
    default would be a silent way to plot one of them upside down.
    """
    if better not in (LOWER_IS_BETTER, HIGHER_IS_BETTER):
        raise ValueError(f"better must be {LOWER_IS_BETTER!r} or {HIGHER_IS_BETTER!r}")
    sign = 1.0 if better == LOWER_IS_BETTER else -1.0
    ordered = sorted(values, key=lambda j: (sign * values[j], judge_sort_key(j)))
    ranks: Dict[str, int] = {}
    for i, judge in enumerate(ordered):
        if i and values[judge] == values[ordered[i - 1]]:
            ranks[judge] = ranks[ordered[i - 1]]
        else:
            ranks[judge] = i + 1
    return ranks


def rank_table(layers: Sequence[Layer]) -> Dict[str, List[Optional[int]]]:
    """``{judge: [rank per layer]}`` — the data behind every line in the figure."""
    per_layer = [rank_judges(layer.values, layer.better) for layer in layers]
    judges = sorted({j for layer in layers for j in layer.values}, key=judge_sort_key)
    return {j: [r.get(j) for r in per_layer] for j in judges}


def crossings(layers: Sequence[Layer]) -> int:
    """How many (judge pair × adjacent layer) rank orders actually invert.

    The figure's whole claim is that lines cross, so the count is printed on the
    figure. If it ever comes back 0, the narrative is not in the data and the
    caption must say so rather than the figure implying it.
    """
    table = rank_table(layers)
    judges = list(table)
    total = 0
    for a_i in range(len(judges)):
        for b_i in range(a_i + 1, len(judges)):
            a, b = table[judges[a_i]], table[judges[b_i]]
            for k in range(len(layers) - 1):
                if None in (a[k], b[k], a[k + 1], b[k + 1]):
                    continue
                if (a[k] - b[k]) * (a[k + 1] - b[k + 1]) < 0:
                    total += 1
    return total


def _colour(judge: str) -> str:
    return EMPHASIS.get(judge, MUTED)


def _noise_note(claim_a_dim_rows: Sequence[dict],
                judge: str = "gemini-3.5-flash") -> str:
    """A judge's own ``sham`` cells against its cue cells (mean ``asc``), at draw time.

    Layer 1 is a per-item |change|, so instability counts against a judge with no
    directional bias at all; printing the placebo beside the cues keeps that visible.
    """
    def mean_asc(keep) -> Optional[float]:
        cells = [f(r, "asc") for r in claim_a_dim_rows
                 if r.get("judge_model") == judge and keep(r.get("family") or "")]
        cells = [v for v in cells if v is not None]
        return sum(cells) / len(cells) if cells else None

    sham = mean_asc(lambda fam: fam == "control")
    cues = mean_asc(lambda fam: fam.startswith(CLAIM_A_FAMILY + "_"))
    if sham is None or cues is None:
        return ""
    return f"{judge}: sham {sham:.2f} vs cues {cues:.2f}"


def _viescore_notes(claim_b_rows: Sequence[dict],
                    judge: str = "gpt-4o-viescore") -> str:
    """The ranking-layer numbers behind viescore's last place, read at draw time."""
    bits: List[str] = []
    rho = sorted(v for v in (f(r, "spearman_delta") for r in claim_b_rows
                             if r.get("judge_model") == judge
                             and r.get("bias_type") == "region_annotation")
                 if v is not None)
    if rho:
        bits.append("region_annotation Δρ "
                    + " / ".join(f"{v:+.3f}" for v in rho))
    acc = sorted(v for v in (f(r, "accuracy_delta") for r in claim_b_rows
                             if r.get("judge_model") == judge
                             and r.get("bias_type") == "region_annotation")
                 if v is not None)
    if acc:
        bits.append(f"derived pairwise accuracy {acc[0] * 100:+.1f}pp")
    return "  ·  ".join(bits)


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def plot_robustness_layers(
    claim_a_dim_rows: Sequence[dict],
    claim_b_rows: Sequence[dict],
    position_rows: Sequence[dict],
    out_path: Path,
    *,
    title: Optional[str] = None,
) -> Path:
    """F9. Rank parallel coordinates over three layers + three own-unit strips."""
    layers = [
        absolute_score_robustness(claim_a_dim_rows),
        ranking_robustness(claim_b_rows),
        position_robustness(position_rows),
    ]
    judges = sorted({j for layer in layers for j in layer.values}, key=judge_sort_key)
    if not judges:
        raise ValueError("no judge carries a scalar on any layer")
    complete = [j for j in judges if all(j in layer.values for layer in layers)]
    if not complete:
        raise ValueError("no judge appears on all three layers")

    table = rank_table(layers)
    n_cross = crossings(layers)
    n_j = len(judges)

    fig = new_figure((11.4, 9.2))
    ax = fig.add_axes([0.300, 0.455, 0.520, 0.375])
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.06, 2.06)
    ax.set_ylim(n_j + 0.62, 0.38)          # rank 1 at the top
    ax.set_xticks([])
    ax.set_yticks(range(1, n_j + 1))
    ax.set_yticklabels([str(r) for r in range(1, n_j + 1)],
                       fontsize=8.8, color=INK2)
    ax.tick_params(length=0, colors=INK2, pad=3.0)

    # alternating rank bands, so a horizontal read is easy
    for r in range(1, n_j + 1):
        if r % 2 == 0:
            ax.axhspan(r - 0.5, r + 0.5, color=BAND, zorder=0)
    for x in range(len(layers)):
        ax.axvline(x, color=BASE, linewidth=1.0, zorder=1)

    for judge in judges:
        ranks = table[judge]
        xs = [x for x, r in enumerate(ranks) if r is not None]
        ys = [r for r in ranks if r is not None]
        if not xs:
            continue
        colour = _colour(judge)
        emph = judge in EMPHASIS
        ax.plot(xs, ys, "-", color=colour, linewidth=2.6 if emph else 1.4,
                alpha=1.0 if emph else 0.85, zorder=4 if emph else 3,
                solid_capstyle="round")
        ax.plot(xs, ys, linestyle="none", marker=marker_for(judge),
                markersize=9.0 if emph else 7.0, color=colour,
                markerfacecolor=colour if emph else SURFACE,
                markeredgecolor=colour, markeredgewidth=1.5, zorder=5)
        # a label at BOTH ends: the two orders differ, which is the finding.
        # The left label clears the whole rank-tick column (-30 pt), so the row
        # reads "viescore — 1 — marker" rather than piling name onto digit.
        ax.annotate(short(judge), (xs[0], ys[0]), xytext=(-30, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=9.0 if emph else 8.4,
                    fontweight="bold" if emph else "normal",
                    color=INK if emph else INK2)
        ax.annotate(short(judge), (xs[-1], ys[-1]), xytext=(14, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9.0 if emph else 8.4,
                    fontweight="bold" if emph else "normal",
                    color=INK if emph else INK2)

    for x, layer in enumerate(layers):
        ax.text(x, 1.055, layer.title, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10.6, color=INK,
                fontweight="bold")

    ax.text(0.5, -0.115,
            f"{n_cross} of the {n_j * (n_j - 1) // 2 * (len(layers) - 1)} "
            "(judge pair × adjacent layer) orderings INVERT — the crossings are the "
            "finding.\nVertical position is RANK ONLY (1 = most robust within that "
            "layer); the numbers live in the three panels below.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
            color=INK2, linespacing=1.5)

    # ---- the three own-unit strips ----------------------------------------- #
    strip_w, gap = 0.235, 0.055
    left0 = 0.075
    for x, layer in enumerate(layers):
        sx = fig.add_axes([left0 + x * (strip_w + gap), 0.185, strip_w, 0.115])
        sx.set_facecolor(SURFACE)
        for name, sp in sx.spines.items():
            sp.set_visible(name == "bottom")
            sp.set_color(BASE)
            sp.set_linewidth(0.8)
        sx.grid(axis="x", color=GRID, linewidth=0.6)
        sx.set_axisbelow(True)
        sx.tick_params(colors=INK2, labelsize=7.6, length=3, color=BASE)
        sx.set_yticks([])
        sx.set_ylim(len(judges) - 0.4, -0.6)

        vals = [layer.values[j] for j in judges if j in layer.values]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        pad = span * 0.20 or (abs(hi) * 0.1 or 0.1)
        sx.set_xlim(lo - pad, hi + pad)

        for i, judge in enumerate(judges):
            v = layer.values.get(judge)
            if v is None:
                continue
            colour = _colour(judge)
            emph = judge in EMPHASIS
            sx.plot([v], [i], marker=marker_for(judge), linestyle="none",
                    markersize=7.4 if emph else 6.0, color=colour,
                    markerfacecolor=colour if emph else SURFACE,
                    markeredgecolor=colour, markeredgewidth=1.4, zorder=4)
            # Flip the label to the inside once the mark sits in the right half,
            # so a value label can never run off the panel and get clipped.
            right_half = span > 0 and (v - lo) / span > 0.5
            sx.annotate(f"{short(judge)}  {layer.fmt.format(v)}", (v, i),
                        xytext=(-8 if right_half else 8, 0),
                        textcoords="offset points",
                        ha="right" if right_half else "left", va="center",
                        fontsize=7.2, fontweight="bold" if emph else "normal",
                        color=INK if emph else INK2)

        arrow = "←  more robust" if layer.better == LOWER_IS_BETTER \
            else "more robust  →"
        sx.set_title(f"{layer.scalar}\n{layer.unit}   ·   {arrow}",
                     fontsize=7.8, color=INK2, loc="left", pad=6.0,
                     linespacing=1.45)

    fig.suptitle(title or "Robustness at one layer does NOT imply robustness at "
                          "another",
                 fontsize=14.6, color=INK, x=0.006, ha="left", y=0.992,
                 fontweight="bold")
    fig.text(0.006, 0.955,
             "Five judges, three layers, three DIFFERENT units — so the top panel "
             "plots RANK ONLY and the three raw scalars sit below on three "
             "independent axes.\n"
             "The three are never normalised onto one ruler: "
             + "   ".join(f"[{i + 1}] {lay.unit}" for i, lay in enumerate(layers))
             + ".",
             fontsize=8.4, color=INK2, ha="left", va="top", linespacing=1.55)

    handles = [Line2D([], [], marker=marker_for(j), linestyle="-",
                      linewidth=2.4 if j in EMPHASIS else 1.4, color=_colour(j),
                      markerfacecolor=_colour(j) if j in EMPHASIS else SURFACE,
                      markeredgecolor=_colour(j), markersize=7.0, label=short(j))
               for j in judges]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.988, 0.968),
               ncol=1, frameon=False, fontsize=7.8, labelcolor=INK2,
               handlelength=2.2, handletextpad=0.6, borderaxespad=0.0)

    def trip(judge: str) -> str:
        ranks = table.get(judge)
        if ranks is None:
            return "?"
        return " → ".join("?" if r is None else str(r) for r in ranks)

    notes = _viescore_notes(claim_b_rows)
    noise = _noise_note(claim_a_dim_rows)
    fig.text(0.006, 0.108,
             "Read in 3 s:  gemini-3.5-flash is the LEAST robust judge on absolute "
             f"scores and the MOST robust on protocol ({trip('gemini-3.5-flash')}); "
             "gpt-4o-viescore is the LEAST robust on ranking "
             f"({trip('gpt-4o-viescore')}).\n"
             + (f"viescore on ranking: {notes}\n" if notes else "")
             + "Caveats: the scalars average different cell sets (36 cue × dimension / "
               "8 / none), so they order judges WITHIN a layer and are never differenced."
               "  Layer 1 is a per-item |change|:\n"
               "a judge's own instability counts against it"
             + (f" ({noise})" if noise else "")
             + "; ranking by |mean_shift|, each cue's AVERAGE signed change, would put "
               "viescore first.  Layer 2 is a\n"
               "MAGNITUDE scalar (mean |Δρ|); counting cells whose ρ interval excludes 0 "
               "instead would put gemini-3.5-flash last there (5 of 8) and viescore "
               "joint-second (3 of 8).\n"
             + ROSTER_NOTE,
             fontsize=8.0, color=INK, ha="left", va="top", linespacing=1.62)
    return save(fig, out_path)


# --------------------------------------------------------------------------- #
# CLI — appended to scripts/07_plot_figures.sh; zero API, idempotent
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse
    import csv

    from edit_judge_bias.data.manifest_utils import default_root
    from edit_judge_bias.experiments.build_claim_tables import main_grid_rows
    from edit_judge_bias.experiments.plot_results import figure_prefix

    root = default_root()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics-dir", type=Path,
                    default=root / "results" / "v2" / "metrics")
    ap.add_argument("--figures-dir", type=Path,
                    default=root / "results" / "v2" / "figures")
    args = ap.parse_args(argv)

    def read(name: str) -> List[dict]:
        path = Path(args.metrics_dir) / name
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            return []
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    # The ranking layer is defined on the main grid's claim B cells; the FILL rows merged into
    # the same file on 2026-09-15 are another collection with another reference.
    claim_a_dim = read("claim_a_by_dimension.csv")
    claim_b = main_grid_rows(read("claim_b.csv"))
    position = read("position.csv")
    if not (claim_a_dim and claim_b and position):
        print("skip robustness_layers.png "
              "(needs claim_a_by_dimension.csv + claim_b.csv + position.csv)")
        return 0
    # The pilot tree's `pilot_` prefix rule applies here too — a 1-5-scale,
    # mock-judge-containing tree must never be able to write a v2 basename.
    name = f"{figure_prefix(Path(args.metrics_dir))}robustness_layers.png"
    out = plot_robustness_layers(
        claim_a_dim, claim_b, position, Path(args.figures_dir) / name)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
