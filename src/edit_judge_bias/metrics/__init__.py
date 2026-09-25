"""Metrics layer.

Scoring metrics (score shift / SIR / ASC), pairwise metrics (flip rate / RR),
robustness, and statistics (bootstrap CI, Wilcoxon, McNemar) computed over the
JudgeResult manifests produced by the judge runners.
"""

from edit_judge_bias.metrics.scoring_metrics import (
    PlaceboContrastStats,
    RetestStats,
    ScoringShiftStats,
    compute_placebo_contrast,
    compute_retest_stats,
    compute_score_shifts,
)
from edit_judge_bias.metrics.pairwise_metrics import (
    ConsistencyStats,
    OneSidedBiasStats,
    PositionFlipStats,
    PositionJointStats,
    compute_consistency,
    compute_one_sided_bias,
    compute_position_flips,
    decisive_human_prefs,
    position_joint,
)
from edit_judge_bias.metrics.agreement import (
    AgreementStats,
    compute_agreement,
    compute_agreement_grid,
)

__all__ = [
    "ScoringShiftStats",
    "compute_score_shifts",
    "PlaceboContrastStats",
    "compute_placebo_contrast",
    "RetestStats",
    "compute_retest_stats",
    "PositionFlipStats",
    "compute_position_flips",
    "PositionJointStats",
    "position_joint",
    "OneSidedBiasStats",
    "compute_one_sided_bias",
    "ConsistencyStats",
    "compute_consistency",
    "decisive_human_prefs",
    "AgreementStats",
    "compute_agreement",
    "compute_agreement_grid",
]
