"""Figures for the frozen claim tables (claim B, slot stickiness, retest).

The §9.2 heatmaps already cover claim A. These three cover the parts of the story a
heatmap cannot carry:

``plot_agreement_forest``
    Claim B is a set of *intervals*, not point estimates — the whole argument is which
    (judge x source x bias) cells have a rho change whose CI excludes zero. A forest
    plot is the only honest rendering; a bar chart of the deltas would hide that most
    of them are indistinguishable from no change.

``plot_position_joint``
    The displayed-letter joint distribution. Stacked so the diagonal (same slot twice)
    reads as one block: a content-fair judge shows almost none of it, a slot-sticky
    judge is mostly it.

``plot_retest``
    The judge's own noise floor next to zero, so a reader can see immediately whether
    a reported effect clears it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    if v in (None, ""):
        return None
    return float(v)


def plot_agreement_forest(
    rows: Sequence[dict], out_path: Path, *, title: Optional[str] = None
) -> Path:
    """One line per (judge, source, bias): rho change with its cluster-bootstrap CI."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    usable = [
        r for r in rows
        if _f(r, "spearman_delta") is not None
        and _f(r, "spearman_delta_ci_low") is not None
    ]
    usable.sort(key=lambda r: (r.get("anchor_source", ""), r["bias_type"], r["judge_model"]))

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.28 * len(usable) + 1.2)))
    for i, r in enumerate(usable):
        d = _f(r, "spearman_delta")
        lo, hi = _f(r, "spearman_delta_ci_low"), _f(r, "spearman_delta_ci_high")
        # Excluding zero is the claim; colour carries it so the eye finds them.
        excludes = lo is not None and hi is not None and (lo > 0 or hi < 0)
        colour = "#b2182b" if excludes else "#999999"
        ax.plot([lo, hi], [i, i], color=colour, linewidth=2 if excludes else 1)
        ax.plot([d], [i], "o", color=colour, markersize=5 if excludes else 3.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(usable)))
    ax.set_yticklabels(
        [f"{r['judge_model']} · {r.get('anchor_source', '')} · {r['bias_type']}"
         for r in usable], fontsize=7)
    ax.set_xlabel("Spearman rho after − before (judge vs human)")
    ax.set_title(title or "Claim B: agreement change, cluster-bootstrap 95% CI",
                 fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_position_joint(
    rows: Sequence[dict], out_path: Path, *, title: Optional[str] = None
) -> Path:
    """Stacked bars of (displayed winner before, after), same-slot cells grouped."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts = ("A", "B", "Tie")
    # Same-slot transitions first so the stickiness block sits at the bottom.
    cells = [(a, b) for a in verdicts for b in verdicts if a == b]
    cells += [(a, b) for a in verdicts for b in verdicts if a != b]
    models = [r["judge_model"] for r in rows]

    fig, ax = plt.subplots(figsize=(1.4 * len(models) + 3, 4.2))
    bottom = np.zeros(len(rows))
    cmap = plt.get_cmap("tab20")
    for k, (a, b) in enumerate(cells):
        vals = np.array([
            float(r.get(f"count_{a}_{b}", 0) or 0) / max(float(r["n"]), 1) for r in rows
        ])
        same_slot = a == b
        ax.bar(models, vals, bottom=bottom, label=f"{a}→{b}",
               color=cmap(k), hatch="//" if same_slot else None,
               edgecolor="black" if same_slot else "none",
               linewidth=0.4 if same_slot else 0)
        bottom += vals
    ax.set_ylabel("share of pairs")
    ax.set_title(title or "Displayed verdict before vs after the A/B swap "
                          "(hatched = same slot twice = slot-sticky)", fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_retest(
    rows: Sequence[dict], out_path: Path, *, title: Optional[str] = None
) -> Path:
    """Absolute movement per item, and the signed drift with its CI, per judge."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: -float(r["mean_abs_delta"]))
    models = [r["judge_model"] for r in rows]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.bar(x, [float(r["mean_abs_delta"]) for r in rows], color="#4575b4")
    ax1.set_xticks(x, models, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("mean |Δ| per item, same question twice")
    ax1.set_title("Test-retest noise floor")

    drift = [float(r["mean_delta"]) for r in rows]
    lo = [d - float(r["ci_low"]) for d, r in zip(drift, rows)]
    hi = [float(r["ci_high"]) - d for d, r in zip(drift, rows)]
    ax2.bar(x, drift, color="#d73027", yerr=[lo, hi], capsize=4)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x, models, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("signed drift (2nd ask − 1st)")
    # A positive drift is not a threat here, it is a conservatism argument: the biased
    # conditions were asked after the baselines, so it shrinks the reported deflation.
    ax2.set_title("Systematic drift (95% CI)")
    fig.suptitle(title or "")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
