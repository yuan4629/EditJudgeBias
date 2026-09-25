#!/usr/bin/env bash
# Stage 5 -- aggregate the judgments into per-(judge, cue) metric tables. Free, local.
#
#   bash scripts/05_compute_metrics.sh [results_dir]     # default: results/v2
#
# Why the subset filter: result files are per judge, not per arm, so once the anchor arm
# has run, cues covered by both arms have rows from two blocks in one file. The cross-source
# tables read the balanced breadth block only; anchor-based tables filter on
# `anchor_source` instead.
#
# Why --validators names two of the three validators: the QC subset is an INTERSECTION over
# validators, so a noisy validator would remove images at random rather than bad ones.
# glm-4v is reported as a cross-check against its own floor, not used as a gate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working

RESULTS_DIR="${1:-results/v2}"
python -m edit_judge_bias.experiments.aggregate_results \
  --results-dir "$RESULTS_DIR" \
  --samples data/manifests/samples_judge_v2.jsonl \
  --subset-filter subset_block=breadth \
  --validators gemini-3.5-flash --validators gpt-4o-mini
