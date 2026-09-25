"""F8 — quality preservation, each cue against ITS OWN validator's false-flag floor.

Source CSV: ``results/v2/metrics/quality_combined.csv``.

Why one panel per validator and never a pooled panel: the three validators do not
share a floor. ``gpt-4o-mini`` passes a visually null JPEG round-trip 109/110, so on a
bare comparison 9 of its 11 cues sit "below floor" on a one- or two-image difference;
``glm-4v`` passes the same control at 95/110, a 13.6% false-flag rate. A figure that
drew all three against one reference line would manufacture exactly the artefact that
overturned the ``zoom_inset`` headline. The floor line in each panel is that
validator's own ``control_pass_rate``, and the sham row that produced it is drawn on
the floor line rather than in the cue list.

The 85% gate is drawn as a dashed threshold in every panel. Exactly one cell in the
study fails it — ``glm-4v x zoom_inset = 0.718`` — and it is marked with the reserved
status colour plus a printed label, never colour alone. ``glm-4v`` does not gate the
QC subset, and its panel says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.lines import Line2D

from edit_judge_bias.visualization.style import (
    BASE,
    BLUE,
    CRITICAL,
    GRID,
    INK,
    INK2,
    MUTED,
    SURFACE,
    f,
    new_figure,
    save,
    truthy,
    wilson_from_rate,
)

GATE = 0.85
CONTROL_CUE = "sham"

#: Validators that gate the QC subset. `glm-4v` is reported for ordering only — a 13.6%
#: false-flag floor would discard one image in seven at random, not one bad image.
GATING = ("gemini-3.5-flash", "gpt-4o-mini")

#: Panel order: the main validator, then the independent gate, then the non-gating
#: cross-check. Anything unrecognised is appended alphabetically.
PANEL_ORDER = ["gemini-3.5-flash", "gpt-4o-mini", "glm-4v"]


def _validator_key(name: str):
    try:
        return (PANEL_ORDER.index(name), "")
    except ValueError:
        return (len(PANEL_ORDER), name)


def significant_cells(rows: Sequence[dict]) -> List[tuple]:
    """(validator, cue) pairs that are significantly below their own floor, MATCHED.

    Matched, not Fisher: the unmatched test compares an arm's 110 images against a floor
    measured on 780, so part of its power comes from images the arm never ran on.  And
    never `rate < floor` — a bare comparison marked 9 of gpt-4o-mini's 11 cues "below
    floor" on one- or two-image differences.
    """
    return [(r["validator_model"], r["bias_type"]) for r in rows
            if truthy(r, "below_floor_significant_matched") is True]


def quality_headline(rows: Sequence[dict]) -> str:
    """The figure's title, DERIVED from the table.

    This exists as a function so a test can pin it.  Until 2026-08-17 the title was the
    hardcoded string "No cue is significantly below its own validator's false-flag
    floor"; WP-A3's matched floor falsified that, and a hardcoded title would have kept
    asserting it inside the figure while the CSV beside it said the opposite.  A figure
    that states a conclusion must compute it.
    """
    sig = significant_cells(rows)
    if not sig:
        return ("No cue is significantly below its own validator's false-flag floor "
                "(matched design)")
    named = sorted({c for _, c in sig})
    who = named[0] if len(named) == 1 else ", ".join(named)
    n_validators = len({r["validator_model"] for r in rows})
    return (f"{who} falls significantly below its own floor on "
            f"{len({v for v, _ in sig})} of {n_validators} validators")


def plot_quality_vs_floor(
    rows: Sequence[dict], out_path: Path, *, title: Optional[str] = None
) -> Path:
    """One panel per validator; every cue against that validator's own control floor."""
    validators = sorted({r["validator_model"] for r in rows}, key=_validator_key)
    if not validators:
        raise ValueError("quality_combined.csv held no validator rows")

    # Cue order is shared across panels (otherwise the panels cannot be compared),
    # ranked by the mean pass rate so the boundary cases sit at the bottom.
    agg: Dict[str, List[float]] = {}
    for r in rows:
        if r["bias_type"] == CONTROL_CUE:
            continue
        v = f(r, "mllm_pass_rate")
        if v is not None:
            agg.setdefault(r["bias_type"], []).append(v)
    cues = sorted(agg, key=lambda c: -float(np.mean(agg[c])))
    by_key = {(r["validator_model"], r["bias_type"]): r for r in rows}

    fig_h = 0.30 * len(cues) + 4.35
    fig = new_figure((4.35 * len(validators) + 1.15, fig_h))
    axes = fig.subplots(1, len(validators), sharey=True,
                        gridspec_kw=dict(wspace=0.10))
    axes = [axes] if len(validators) == 1 else list(axes)

    x_lo, x_hi = 0.60, 1.02
    failed_gate = False
    any_significant = False
    for ax, validator in zip(axes, validators):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

        control = by_key.get((validator, CONTROL_CUE))
        floor = f(control or {}, "control_pass_rate")
        if floor is None:
            floor = f(next((r for r in rows if r["validator_model"] == validator), {}),
                      "control_pass_rate")
        gating = validator in GATING

        if floor is not None:
            ax.axvspan(floor, x_hi, color="#f4f3ef", zorder=0)
            ax.axvline(floor, color=INK2, linewidth=1.3, zorder=3)
        ax.axvline(GATE, color=MUTED, linewidth=1.1, ls=(0, (4, 2.4)), zorder=3)

        for i, cue in enumerate(cues):
            row = by_key.get((validator, cue))
            if row is None:
                continue
            rate, n = f(row, "mllm_pass_rate"), f(row, "n_mllm")
            lo, hi = wilson_from_rate(rate, n or 0)
            below_floor = floor is not None and rate < floor
            fails_gate = truthy(row, "passes_85_gate") is False
            # ★ Significance comes from the MATCHED test (WP-A3), never from
            # `rate < floor`.  A bare comparison marked 9 of gpt-4o-mini's 11 cues as
            # "below floor" on one- or two-image differences, which is the defect the
            # significance columns exist to fix; and the unmatched Fisher test borrows
            # power from 670 images the arm never ran on, so it is not the primary
            # judgement either.
            sig = truthy(row, "below_floor_significant_matched") is True
            q_m = f(row, "floor_q_matched")
            colour = CRITICAL if (fails_gate or sig) else BLUE
            failed_gate = failed_gate or fails_gate
            any_significant = any_significant or sig
            ax.plot([lo, hi], [i, i], color=colour, lw=1.9 if (fails_gate or sig) else 1.4,
                    solid_capstyle="round", zorder=4)
            # Filled = significantly below after matching; hollow = point estimate under
            # the floor but NOT significant.  The two must not look alike: before A3 the
            # published reading of this figure was that no cell was significant at all.
            ax.plot([rate], [i], "o", ms=7.6,
                    mfc=colour if sig else (SURFACE if below_floor else colour),
                    mec=colour, mew=1.7, zorder=5)
            # Direct-label selectively: only the cells a reader must be able to read
            # off exactly — the ones under their own floor, and the gate failure.
            if below_floor or fails_gate:
                ax.text(lo - 0.008, i, f"{rate:.3f}", ha="right", va="center",
                        fontsize=7.0, color=CRITICAL if (fails_gate or sig) else MUTED,
                        fontweight="bold" if (fails_gate or sig) else "normal", zorder=6)
            if sig and q_m is not None:
                b, c = f(row, "n_discordant_b"), f(row, "n_discordant_c")
                bc = "" if b is None or c is None else f", {b:.0f} vs {c:.0f} discordant"
                # Below the marker, centred — an annotation placed to the RIGHT of the
                # interval is clipped by the panel edge exactly when the cell sits near
                # 1.0, i.e. on the highest-floor validator, which is where the finding
                # matters most.  Never anchor a label to a coordinate that can leave frame.
                ax.text(rate, i - 0.42, f"matched q={q_m:.4f}{bc}",
                        ha="center", va="bottom", fontsize=6.9, color=CRITICAL,
                        fontweight="bold", zorder=6)
            if fails_gate:
                ax.text(rate, i + 0.55,
                        "▲ the study's only 85%-gate failure\n"
                        "(and this validator does not gate)",
                        ha="center", va="top", fontsize=7.2, color=CRITICAL,
                        fontweight="bold", linespacing=1.4, zorder=6)

        ax.text(0.0, 1.135, f"{validator}", transform=ax.transAxes, fontsize=11,
                color=INK, fontweight="bold", va="bottom")
        ax.text(0.0, 1.078, "gates the QC subset" if gating
                else "cross-check only — DOES NOT GATE",
                transform=ax.transAxes, fontsize=8.0, va="bottom",
                color=INK2 if gating else CRITICAL,
                fontweight="normal" if gating else "bold")
        if floor is not None:
            ax.text(0.0, 1.006,
                    f"own sham floor = {floor:.3f}\n"
                    f"flags {(1 - floor) * 100:.1f}% of a visually null re-encode",
                    transform=ax.transAxes, fontsize=7.3, color=INK2, va="bottom",
                    style="italic", linespacing=1.45)

        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0])
        ax.tick_params(axis="x", colors=INK2, labelsize=8.3, length=3, color=BASE)
        ax.tick_params(axis="y", length=0)

    axes[0].set_ylim(-0.9, len(cues) + 0.35)
    axes[0].invert_yaxis()
    axes[0].set_yticks(range(len(cues)))
    axes[0].set_yticklabels(cues, fontsize=8.6, color=INK)

    for ax in axes:
        ax.text(GATE - 0.007, -0.75, "0.85 gate", rotation=90, ha="right",
                va="top", fontsize=7.0, color=MUTED)

    # ★ The headline is READ OFF THE TABLE, never hardcoded — see `quality_headline`.
    sig_cells = significant_cells(rows)
    headline = title or quality_headline(rows)

    n_floor = None
    for v in validators:
        ctrl = by_key.get((v, CONTROL_CUE))
        if ctrl is not None:
            n_floor = f(ctrl, "n_mllm")
            break
    # The MODAL arm n, not the first one found: one cell is 109 (a single image failed to
    # validate) and taking row order would print that as the design.
    arm_ns = [f(r, "n_mllm") for r in rows if r["bias_type"] != CONTROL_CUE]
    arm_ns = [x for x in arm_ns if x is not None]
    n_arm = max(set(arm_ns), key=arm_ns.count) if arm_ns else None

    fig.suptitle(headline, fontsize=14, color=INK, x=0.006, ha="left", y=0.988,
                 fontweight="bold")
    fig.text(0.006, 0.948,
             f"{n_arm:.0f} images per cue; the floor is measured on {n_floor:.0f} — the UNION of "
             "every arm's images, so each cue is paired image-by-image with its own control.\n"
             "Filled red = significantly below that floor after matching (McNemar + BH within "
             "the validator).  Hollow = point estimate below the floor but NOT significant."
             if (n_arm and n_floor) else
             "Bars are 95% Wilson intervals; the shaded region is at-or-above THAT "
             "validator's own floor.",
             fontsize=8.4, color=INK2, ha="left", va="top", linespacing=1.5)

    handles = [
        Line2D([], [], marker="o", ls="-", lw=1.4, color=BLUE, mfc=BLUE, mec=BLUE,
               ms=7.6, label="at or above its own floor"),
        Line2D([], [], marker="o", ls="-", lw=1.4, color=BLUE, mfc=SURFACE, mec=BLUE,
               ms=7.6, mew=1.7, label="below the floor, NOT significant"),
        Line2D([], [], ls="-", lw=1.3, color=INK2, label="that validator's sham floor"),
        Line2D([], [], ls=(0, (4, 2.4)), lw=1.1, color=MUTED, label="0.85 gate"),
    ]
    if any_significant:
        handles.insert(2, Line2D([], [], marker="o", ls="-", lw=1.9, color=CRITICAL,
                                 mfc=CRITICAL, mec=CRITICAL, ms=7.6,
                                 label="significantly below the floor (matched)"))
    if failed_gate:
        handles.append(Line2D([], [], marker="o", ls="-", lw=1.9, color=CRITICAL,
                              mfc=SURFACE, mec=CRITICAL, ms=7.6, mew=1.7,
                              label="fails the 0.85 gate"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.004, 0.052),
               ncol=5, frameon=False, fontsize=7.9, labelcolor=INK2, handlelength=2.0,
               columnspacing=1.5, handletextpad=0.6)

    # The 3-second read, also computed rather than asserted.
    floors = []
    for v in validators:
        ctrl = by_key.get((v, CONTROL_CUE))
        fv = f(ctrl or {}, "control_pass_rate")
        if fv is not None:
            floors.append(f"{fv:.4f}")
    n_gate_ok = sum(1 for r in rows if truthy(r, "passes_85_gate") is not False)
    fig.text(0.006, 0.014,
             f"Read in 3 s:  the floors are {' / '.join(floors)} — different rulers, and "
             f"the two gating validators are NOT in the order 110 images suggested.  "
             + (f"{len(sig_cells)} of {len(rows) - len(validators)} cells are significantly "
                f"below their own floor after matching.  " if sig_cells else
                "No cell is significantly below its own floor after matching.  ")
             + f"{n_gate_ok} of {len(rows)} cells clear the gate.",
             fontsize=8.1, color=INK, ha="left", va="bottom")

    fig.subplots_adjust(left=0.115, right=0.985, top=1 - 1.80 / fig_h,
                        bottom=1.05 / fig_h)
    return save(fig, out_path)
