"""Grouped bar chart of mean score shift per bias, with bootstrap CIs (§9.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_bias_effect(
    rows: Sequence[dict],
    out_path: Path,
    *,
    value_key: str = "mean_shift",
    title: Optional[str] = None,
) -> Path:
    """Bars of `value_key` grouped by bias, one cluster of models per bias."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({r["judge_model"] for r in rows})
    biases = sorted({r["bias_type"] for r in rows})
    lookup = {(r["judge_model"], r["bias_type"]): r for r in rows}

    x = np.arange(len(biases))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(1.6 * len(biases) + 2, 4))
    for k, model in enumerate(models):
        vals, lo_err, hi_err = [], [], []
        for b in biases:
            r = lookup.get((model, b))
            v = float(r[value_key]) if r and r.get(value_key) not in (None, "") else np.nan
            vals.append(v)
            lo = r.get("ci_low") if r else None
            hi = r.get("ci_high") if r else None
            lo_err.append(v - float(lo) if r and lo not in (None, "") else 0.0)
            hi_err.append(float(hi) - v if r and hi not in (None, "") else 0.0)
        ax.bar(x + k * width, vals, width, label=model,
               yerr=[lo_err, hi_err], capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(models) - 1) / 2, biases, rotation=30, ha="right")
    ax.set_ylabel(value_key)
    ax.set_title(title or f"{value_key} by bias")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
