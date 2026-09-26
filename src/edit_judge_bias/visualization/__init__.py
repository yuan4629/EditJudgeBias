"""Visualization layer (Milestone 5 + the paper figure set).

All plotters use the non-interactive Agg backend and write PNGs to disk, so they
run headless (CI, servers) without a display.

``style`` holds the one palette and the one uncertainty helper the whole set shares —
read its docstring before adding a figure.
"""

from edit_judge_bias.visualization.plot_heatmap import plot_metric_heatmap
from edit_judge_bias.visualization.plot_bias_effect import plot_bias_effect
from edit_judge_bias.visualization.plot_claims import (
    plot_agreement_forest,
    plot_position_joint,
    plot_retest,
)
from edit_judge_bias.visualization.plot_forests import (
    plot_claim_a_dimension_forest,
    plot_claim_a_forest,
    plot_claim_b_forest,
    split_control_rows,
)
from edit_judge_bias.visualization.plot_protocol import (
    plot_mitigation_tradeoff,
    plot_rr_vs_cr,
)
from edit_judge_bias.visualization.plot_quality import plot_quality_vs_floor

# NOTE: `plot_robustness_layers` (F9) is deliberately NOT re-exported here. It is the
# one plotter with its own ``python -m`` entry point (it joins three frozen tables and
# is driven straight from scripts/07_plot_figures.sh); importing it eagerly here makes
# runpy emit a "found in sys.modules before execution" RuntimeWarning on every run.
# Import it by module path: `from ...visualization.plot_robustness_layers import ...`.

__all__ = [
    "plot_metric_heatmap",
    "plot_bias_effect",
    "plot_agreement_forest",
    "plot_position_joint",
    "plot_retest",
    "plot_claim_a_dimension_forest",
    "plot_claim_a_forest",
    "plot_claim_b_forest",
    "split_control_rows",
    "plot_mitigation_tradeoff",
    "plot_rr_vs_cr",
    "plot_quality_vs_floor",
]
