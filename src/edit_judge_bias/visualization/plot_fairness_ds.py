"""F1 — the D-S skin-tone null, drawn next to the control that makes it readable.

Source CSVs
    ``results/v2_fairness_ds/metrics/attribute_gaps.csv``            (frozen, 10 rows)
    ``results/v2_fairness_ds/metrics/dose_control_decomposition.csv`` (20 rows)
optionally, for the MDE reference only:
    ``results/v2_fairness_ds/raw_judgments/scoring__*.jsonl``
    ``data/manifests/samples_fairness_ds_judge_v4.jsonl``

FORM: forest / dot-and-interval small multiples. The data's job is an *interval
estimate against a reference of zero*; a bar chart of the point estimates would hide
that every interval in panel A straddles zero, which IS the result.

THE ONE RULE THIS MODULE ENFORCES IN CODE
    Benjamini-Hochberg families are never pooled onto one axis. Each panel draws one
    *contrast* (optionally split into its ``dark``/``light`` arms, which are corrected
    separately and get separate marker shapes); asking a panel to hold two different
    contrasts raises :class:`ValueError`. Pooling them would let a reader read the
    null and the control off one ruler as if a single BH correction had covered both,
    and the whole argument of this figure is that they are two measurements.

AXIS UNITS — the trap this figure exists to avoid
    ``mde_sd`` in the frozen tables is in SD of the **paired difference**, not SD of
    the score. Drawing ``mde_sd x SD(score)`` would put the detection threshold about
    3.5x too far out and make the null look far better established than it is. The MDE
    ticks here are ``mde_sd x SD(paired difference)``, measured from the raw rows by
    :func:`paired_diff_sd`; if that measurement is unavailable the ticks are simply
    not drawn, and the legend says so.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from matplotlib.lines import Line2D

from edit_judge_bias.visualization.style import (
    BASE,
    BLUE,
    GRID,
    INK,
    INK2,
    JUDGE_ORDER,
    MUTED,
    ORANGE,
    SURFACE,
    f,
    new_figure,
    qtext,
    save,
    truthy,
)

#: One panel per contrast. ``arms`` are the ``attribute`` values to draw; ``mde``
#: says whether a detection threshold is meaningful for that contrast.
PANELS: List[dict] = [
    dict(
        key="skin_tone",
        arms=[("skin_tone", "")],
        colour=BLUE,
        mde=True,
        title="A   the result: skin-tone direction",
        sub=("dark − light",
             "same scene · one edit · same mask · symmetric ±30° ITA"),
    ),
    dict(
        key="dose_vs_placebo",
        arms=[("dose_vs_placebo:dark", "dark"), ("dose_vs_placebo:light", "light")],
        colour=ORANGE,
        mde=True,
        title="B   the control that makes A readable",
        sub=("study − sham   (dose D vs dose 0)",
             "same mask, same location — only the dose differs"),
    ),
    dict(
        key="nonskin_vs_skin",
        arms=[("nonskin_vs_skin:dark", "dark"), ("nonskin_vs_skin:light", "light")],
        colour=MUTED,
        mde=False,
        title="C   UPPER BOUND, not an estimate",
        sub=("study − sham_nonskin   (skin vs non-skin)",
             "area-matched only — non-skin carries 1.29× the dose"),
    ),
    dict(
        key="dose_control",
        arms=[("dose_control", "")],
        colour=MUTED,
        mde=False,
        title="D   the frozen control  ( = C − B )",
        sub=("sham − sham_nonskin",
             "mixes dose AND location — positive control only"),
    ),
]

ARM_MARKER = {"": "o", "dark": "o", "light": "s"}

#: The ``:dark`` / ``:light`` suffix marks the two ARMS of one contrast, not two
#: contrasts. Everything before it is the family *group* a panel may hold.
_ARM_SUFFIXES = (":dark", ":light")


def family_group(family: str) -> str:
    """``fairness_contrast:dose_vs_placebo:dark`` -> ``fairness_contrast:dose_vs_placebo``."""
    for suffix in _ARM_SUFFIXES:
        if family.endswith(suffix):
            return family[: -len(suffix)]
    return family


def check_one_family_per_panel(rows: Sequence[dict], panel_key: str = "") -> str:
    """Return the panel's single family group, or raise.

    A panel is allowed to carry the two *arms* of one contrast (separately BH-corrected,
    separately shaped). It is never allowed to carry two contrasts: a shared axis would
    invite reading the null and its control off one ruler as though one correction had
    covered both.
    """
    groups = sorted({family_group(str(r.get("family", ""))) for r in rows})
    if len(groups) > 1:
        raise ValueError(
            "refusing to pool two BH families onto one axis"
            + (f" (panel {panel_key!r})" if panel_key else "")
            + f": {groups}. One panel per contrast; the dark/light arms of a single "
              "contrast are the only split allowed."
        )
    return groups[0] if groups else ""


# --------------------------------------------------------------------------- #
# SD of the paired difference — the unit the MDE is actually defined in
# --------------------------------------------------------------------------- #
def paired_diff_sd(
    manifest_path: Path, raw_dir: Path
) -> Dict[Tuple[str, str], float]:
    """``(judge, attribute) -> SD of the within-scene paired difference``.

    Read straight off the raw judgments because the frozen tables do not carry it.
    Purely local file reads — no API, no spend. Returns ``{}`` (and the caller simply
    omits the MDE ticks) if either input is missing.
    """
    manifest_path, raw_dir = Path(manifest_path), Path(raw_dir)
    if not manifest_path.exists() or not raw_dir.exists():
        return {}

    keys: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        meta = rec.get("metadata") or {}
        keys[rec["sample_id"]] = (meta.get("base_sample_id"), meta.get("variant_label"))

    scores: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(raw_dir.glob("scoring__*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("parse_success") or row.get("fine_score") is None:
                continue
            base, variant = keys.get(row.get("sample_id") or "", (None, None))
            if base:
                scores[row["judge_model"]][base][variant] = float(row["fine_score"])

    all_arms = ("dark", "light", "sham", "sham_nonskin")
    contrasts = {
        "skin_tone": ("dark", "light"),
        "dose_vs_placebo:dark": ("dark", "sham"),
        "dose_vs_placebo:light": ("light", "sham"),
        "nonskin_vs_skin:dark": ("dark", "sham_nonskin"),
        "nonskin_vs_skin:light": ("light", "sham_nonskin"),
        "dose_control": ("sham", "sham_nonskin"),
    }
    out: Dict[Tuple[str, str], float] = {}
    for judge, scenes in scores.items():
        # The controlled contrasts live on `common_subset_scenes` only, so their SD
        # must be measured there too.
        complete = {b for b, per in scenes.items() if all(a in per for a in all_arms)}
        for name, (a, b) in contrasts.items():
            pool = scenes if name == "skin_tone" else {k: scenes[k] for k in complete}
            diffs = [p[a] - p[b] for p in pool.values() if a in p and b in p]
            if len(diffs) > 1:
                out[(judge, name)] = st.stdev(diffs)
    return out


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def plot_ds_null_vs_control(
    rows: Sequence[dict],
    out_path: Path,
    *,
    paired_sd: Optional[Mapping[Tuple[str, str], float]] = None,
    judges: Sequence[str] = tuple(JUDGE_ORDER),
    title: Optional[str] = None,
) -> Path:
    """Four panels: the skin-tone null, its two decompositions, and the frozen control.

    `rows` is ``attribute_gaps.csv`` + ``dose_control_decomposition.csv`` concatenated.
    """
    paired_sd = dict(paired_sd or {})
    by_key = {(r["judge_model"], r["attribute"]): r for r in rows}
    judges = [j for j in judges
              if any((j, a) in by_key for panel in PANELS for a, _ in panel["arms"])]

    panels = []
    for spec in PANELS:
        got = [by_key[(j, a)] for a, _ in spec["arms"] for j in judges
               if (j, a) in by_key]
        if not got:
            continue
        # THE GUARD: one contrast per axis, always.
        check_one_family_per_panel(got, spec["key"])
        families = sorted({str(r.get("family", "")) for r in got})
        panels.append((spec, got, families))
    if not panels:
        raise ValueError("no rows matched any D-S panel")

    x_lo, x_hi = -0.80, 1.70
    x_n, x_q = 1.40, 1.68

    fig = new_figure((3.7 * len(panels) + 0.9, 6.2))
    axes = fig.subplots(1, len(panels), sharex=True, sharey=True,
                        gridspec_kw=dict(wspace=0.07))
    axes = [axes] if len(panels) == 1 else list(axes)

    drew_mde = False
    for ax, (spec, got, families) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvline(0, color=BASE, linewidth=1.3, zorder=2)
        ax.tick_params(axis="y", length=0)

        ns: List[int] = []
        offsets = [0.0] if len(spec["arms"]) == 1 else [+0.16, -0.16]
        for ai, (attr, arm) in enumerate(spec["arms"]):
            for ji, judge in enumerate(judges):
                row = by_key.get((judge, attr))
                if row is None:
                    continue
                y = ji + offsets[ai]
                value, lo, hi = f(row, "mean_gap"), f(row, "ci_low"), f(row, "ci_high")
                sig = bool(truthy(row, "significant_bh"))
                n = f(row, "n")
                if n is not None:
                    ns.append(int(n))
                ax.plot([lo, hi], [y, y], color=spec["colour"],
                        lw=2.2 if sig else 1.3, solid_capstyle="round", zorder=4)
                ax.plot([value], [y], ARM_MARKER[arm], ms=8.5 if sig else 7.5,
                        mfc=spec["colour"] if sig else SURFACE,
                        mec=spec["colour"], mew=1.8, zorder=5)
                sd = paired_sd.get((judge, attr))
                mde_sd = f(row, "mde_sd")
                if spec["mde"] and sd and mde_sd:
                    drew_mde = True
                    for xx in (-mde_sd * sd, mde_sd * sd):
                        # zorder above the interval: on the control panel every
                        # interval covers its own MDE tick, and a threshold you cannot
                        # see is a threshold the reader has to take on trust.
                        ax.plot([xx, xx], [y - 0.21, y + 0.21], color=INK, lw=0.9,
                                ls=(0, (2, 1.6)), zorder=6)
                if n is not None:
                    ax.text(x_n, y, f"{int(n)}", ha="right", va="center",
                            fontsize=7.4, color=INK2)
                ax.text(x_q, y, qtext(f(row, "q_value")), ha="right", va="center",
                        fontsize=7.4, color=INK if sig else INK2,
                        fontweight="bold" if sig else "normal")

        tr = ax.transAxes
        ax.text(0.0, 1.245, spec["title"], transform=tr, fontsize=11, color=INK,
                fontweight="bold", va="bottom")
        ax.text(0.0, 1.153, spec["sub"][0], transform=tr, fontsize=8.1, color=INK,
                va="bottom")
        ax.text(0.0, 1.084, spec["sub"][1], transform=tr, fontsize=7.5, color=INK2,
                va="bottom")
        fam = " | ".join(families) if len(families) < 3 else f"{len(families)} families"
        span = (f"n = {min(ns)}" if ns and min(ns) == max(ns)
                else (f"n = {min(ns)}–{max(ns)}" if ns else "n = —"))
        ax.text(0.0, 1.016, f"own BH family: {fam}   ·   {span}", transform=tr,
                fontsize=6.9, color=INK2, va="bottom", style="italic")

    axes[0].set_ylim(-0.75, len(judges) - 0.25 + 0.03)
    axes[0].invert_yaxis()
    axes[0].set_yticks(range(len(judges)))
    axes[0].set_yticklabels(judges, fontsize=9.3, color=INK)
    axes[0].set_xlim(x_lo, x_hi)
    for ax in axes:
        ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
        ax.tick_params(axis="x", colors=INK2, labelsize=8.5, length=3, color=BASE)
        ax.text(x_n, -0.70, "n", ha="right", va="center", fontsize=7.2,
                color=INK2, style="italic")
        ax.text(x_q, -0.70, "q", ha="right", va="center", fontsize=7.2,
                color=INK2, style="italic")

    fig.suptitle(title or "A null is only readable next to a control that fires",
                 fontsize=14.5, color=INK, x=0.006, ha="left", y=0.986,
                 fontweight="bold")
    fig.text(0.006, 0.944,
             "5 judges × the same content-distinct OmniEdit scenes.  Paired "
             "within-scene difference in fine_score (sum of 3 dimensions, 3–30), "
             "95% bootstrap CI.\nFilled marker + thick line = significant after "
             "Benjamini–Hochberg WITHIN ITS OWN FAMILY.  The four families are never "
             "pooled onto one ruler.",
             fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.5)

    fig.text(0.5, 0.104, "paired difference in fine_score  (points)",
             fontsize=9, color=INK2, ha="center")

    handles = [
        Line2D([], [], marker="o", ls="-", lw=2.2, color=INK2, mfc=INK2, mec=INK2,
               ms=8.5, label="BH-significant"),
        Line2D([], [], marker="o", ls="-", lw=1.3, color=INK2, mfc=SURFACE, mec=INK2,
               ms=7.5, label="not significant"),
        Line2D([], [], marker="o", ls="none", color=INK2, mfc=SURFACE, mec=INK2,
               ms=7.5, label="circle = dark arm"),
        Line2D([], [], marker="s", ls="none", color=INK2, mfc=SURFACE, mec=INK2,
               ms=7.5, label="square = light arm"),
    ]
    if drew_mde:
        handles.append(Line2D([], [], ls=(0, (2, 1.6)), lw=0.9, color=INK2,
                              label="±MDE at 80% power, drawn as mde_sd × SD(paired "
                                    "difference) — NOT SD(score)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.004, 0.006),
               ncol=5, frameon=False, fontsize=8.1, labelcolor=INK2,
               handlelength=2.2, columnspacing=1.6, handletextpad=0.6)

    fig.text(0.006, 0.055,
             "Read in 3 s:   A sits on zero, inside the detection threshold.   "
             "B — same mask, same location, dose the only difference — clears zero on "
             "9 of 10 arms.   So “no gap detected above 0.1025 SD” is a measurement, "
             "not a failure to measure.",
             fontsize=8.8, color=INK, ha="left", va="bottom")

    fig.subplots_adjust(left=0.088, right=0.995, top=0.715, bottom=0.185)
    return save(fig, out_path)
