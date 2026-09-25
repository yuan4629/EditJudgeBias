#!/usr/bin/env bash
# Fairness track, D-G: generative counterfactual attribute edits (gpt-image-2).
# Exploratory: not part of the experiments reported in the submitted paper.
#
#   bash scripts/fairness_dg.sh
#
# This track stopped at its construct-validity gate: the two MLLM auditors disagreed on
# pair validity at chance level, so no attribute-gap claim is made from it. It is kept so
# that the gate result (scripts/construct_validity.sh) can be reproduced.
#
#   1  pool      content-distinct human scenes (ORB near-duplicates collapsed)       free
#   2  flip      gpt-image-2 flips one attribute on the ORIGINAL image              PAID
#   3  gate      validate each pair: the flip took AND the scene survived           PAID
#   4  edit      re-run the ORIGINAL instruction on each counterfactual             PAID
#   4b null      a second render of the same request = editor-noise floor           PAID
#   5  emit      SampleRecords for the scoring runner                               free
#   6  judge     the panel, one judge per invocation                                PAID
#   7  table     attribute_gaps.csv                                                 free
#
# Step 3 runs before step 4 because the gate is far cheaper than the renders it authorises.
# Step 4b needs `cache_dir: null` (the runner enforces it): a cached second render would
# report an editor-noise floor of exactly zero.
#
# This script runs the free steps plus dry-runs of the paid ones, and prints the paid
# commands for you to run one at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working

echo "=== 1. build the content-distinct pool (free) ==="
python -m edit_judge_bias.experiments.build_fairness_pool

echo
echo "=== dry-run the paid steps (no network) ==="
python -m edit_judge_bias.experiments.run_attribute_edit \
  --config configs/fairness/qwen_image_edit.yaml --dry-run
python -m edit_judge_bias.experiments.run_attribute_validation \
  --config configs/fairness/attribute_validation.yaml --dry-run
python -m edit_judge_bias.experiments.run_counterfactual_edit \
  --config configs/fairness/counterfactual_edit.yaml --dry-run
python -m edit_judge_bias.experiments.run_counterfactual_edit \
  --config configs/fairness/counterfactual_null.yaml --dry-run

echo
echo "=== 5. emit SampleRecords (free; needs steps 2-4 on disk) ==="
python -m edit_judge_bias.experiments.build_fairness_samples || true

echo
echo "=== 7. build the table (free; needs step 6 on disk) ==="
python -m edit_judge_bias.experiments.build_fairness_tables || true

cat <<'MSG'

--- PAID STEPS, one at a time, each with --use-api ---
python -m edit_judge_bias.experiments.run_attribute_edit        --config configs/fairness/qwen_image_edit.yaml       --use-api
python -m edit_judge_bias.experiments.run_attribute_validation  --config configs/fairness/attribute_validation.yaml  --use-api
python -m edit_judge_bias.experiments.run_counterfactual_edit   --config configs/fairness/counterfactual_edit.yaml   --use-api
python -m edit_judge_bias.experiments.run_counterfactual_edit   --config configs/fairness/counterfactual_null.yaml   --use-api
python -m edit_judge_bias.experiments.run_scoring_judge --config configs/experiment/scoring_fairness_v2.yaml --judge-config configs/judge/<judge>.yaml --use-api
MSG
