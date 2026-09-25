#!/usr/bin/env bash
# Stage 6 -- build every analysis table from the judgments. Free, local, a few minutes
# (almost all of it the permutation and bootstrap steps).
#
#   bash scripts/06_build_tables.sh
#
# Run after stage 5 and before stage 7. Reads results/v2 (main collection) and
# results/v2_fill (fill collection); writes results/*/metrics/*.csv.
#
# The two judge rosters are separate multiple-testing families and are never pooled:
# the five-judge panel writes the plain tables, the replication judge writes `a5_*` tables.
# Fill rows are written into claim_b.csv / pairwise_one_sided.csv with their own
# `collection` / `reference` columns and their own BH family (`[fill]`): each fill cell is
# measured against the SAME collection's `sham`, so no main-grid value or q-value moves.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working

R=results/v2
FILL=results/v2_fill
S=data/manifests/samples_judge_v2.jsonl
BREADTH=(--subset-filter subset_block=breadth)
# The decisive-pair threshold is an SD of each source's own human scores, so it is computed
# on the FULL pool, not on the judged subset.
FULL=(--full-samples data/manifests/samples_full_v2.jsonl
      --full-pairs   data/manifests/pairs_full_v2.jsonl)

echo ">> headline claims, quality fold-in, per-dimension decomposition, mitigation"
python -m edit_judge_bias.experiments.build_claim_tables \
  --results-dir "$R" --samples "$S" "${BREADTH[@]}" --fill-dir "$FILL"
python -m edit_judge_bias.experiments.build_analysis_tables \
  --results-dir "$R" --manifest "$S" "${BREADTH[@]}"
python -m edit_judge_bias.experiments.build_claim_a_dimensions \
  --results-dir "$R" --samples "$S" "${BREADTH[@]}"
python -m edit_judge_bias.experiments.build_mitigation_tables \
  --results-dir "$R" --samples "$S" "${BREADTH[@]}" "${FULL[@]}"

echo ">> main invariance table: per-dimension |change| against the sham and retest floors"
python -m edit_judge_bias.experiments.build_noise_floor_table \
  --results-dir "$R" --samples "$S" "${BREADTH[@]}"

echo ">> sensitivity analyses (read the tables above)"
# The exclusion analysis reads the mask-polarity audit; rebuild it if absent.
[ -f "$R/metrics/mask_polarity_audit.json" ] || python scripts/audit_mask_polarity.py
python -m edit_judge_bias.experiments.build_sensitivity_tables --results-dir "$R" --samples "$S"
python -m edit_judge_bias.experiments.build_robustness_axes --metrics-dir "$R/metrics"
python -m edit_judge_bias.experiments.build_claim_b_permutation --results-dir "$R" --samples "$S"
python -m edit_judge_bias.experiments.build_leaderboard_simulation --results-dir "$R" --samples "$S"
python -m edit_judge_bias.experiments.build_claim_b_by_area --results-dir "$R" --samples "$S"

echo ">> human validation leg (prints 'skip' while the label sheet is absent)"
python -m edit_judge_bias.experiments.build_human_validation_tables --results-dir "$R"

echo ">> replication judge: its own roster, its own a5_* tables"
python -m edit_judge_bias.experiments.build_claim_tables \
  --results-dir "$R" --samples "$S" "${BREADTH[@]}" --fill-dir "$FILL" --roster a5
python -m edit_judge_bias.experiments.build_claim_a_dimensions \
  --results-dir "$R" --samples "$S" "${BREADTH[@]}" --roster a5
python -m edit_judge_bias.experiments.build_mitigation_tables \
  --results-dir "$R" --samples "$S" "${BREADTH[@]}" "${FULL[@]}" --roster a5

echo ">> fill: preservation of the newly rendered pair images (against their own sham)"
python -m edit_judge_bias.experiments.build_analysis_tables --results-dir "$FILL" --quality-only

echo ">> validator controls (skipped when their verdicts are absent)"
if [ -d results/v2_control/quality ]; then
  python -m edit_judge_bias.experiments.build_validator_sensitivity
fi
if [ -d results/v2_ablation/quality ]; then
  python -m edit_judge_bias.experiments.build_prompt_ablation
fi
if [ -d results/v2_ablation_control/quality ]; then
  python -m edit_judge_bias.experiments.build_validator_sensitivity \
    --control-results-dir results/v2_ablation_control/quality \
    --out results/v2_ablation_control/metrics/validator_sensitivity_a4d.csv
fi

echo ">> one-table summary (reads the CSVs above, so it runs last)"
python -m edit_judge_bias.experiments.build_main_table --metrics-dir "$R/metrics"
