"""Model x bias heatmaps for a scalar metric."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _pivot(
    rows: Sequence[dict], value_key: str
) -> Tuple[List[str], List[str], np.ndarray]:
    models = sorted({r["judge_model"] for r in rows})
    biases = sorted({r["bias_type"] for r in rows})
    mat = np.full((len(models), len(biases)), np.nan)
    mi = {m: i for i, m in enumerate(models)}
    bi = {b: i for i, b in enumerate(biases)}
    for r in rows:
        v = r.get(value_key)
        if v is None or v == "":
            continue
        mat[mi[r["judge_model"]], bi[r["bias_type"]]] = float(v)
    return models, biases, mat


def plot_metric_heatmap(
    rows: Sequence[dict],
    value_key: str,
    out_path: Path,
    *,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    center: Optional[float] = 0.0,
    fmt: str = "{:.2f}",
) -> Path:
    """Render a model x bias heatmap of `value_key` to `out_path` (PNG)."""
    models, biases, mat = _pivot(rows, value_key)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(1.4 * len(biases) + 2, 0.8 * len(models) + 1.5))
    if center is not None and np.isfinite(mat).any():
        vmax = np.nanmax(np.abs(mat - center)) or 1.0
        im = ax.imshow(mat, cmap=cmap, vmin=center - vmax, vmax=center + vmax, aspect="auto")
    else:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(biases)), biases, rotation=30, ha="right")
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(biases)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(title or value_key)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
