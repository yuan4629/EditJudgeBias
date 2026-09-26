#!/usr/bin/env bash
# Stage 7 -- render every figure from the metric tables. Free, local, idempotent.
#
#   bash scripts/07_plot_figures.sh                # results/v2
#   bash scripts/07_plot_figures.sh <results_dir>  # another results tree
#
# Run after stages 5 and 6 (it reads tables from both). Each entry point prints "skip ..."
# and exits 0 on a tree that lacks the tables it needs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working

render() {
  local base="$1"
  if [ ! -d "$base/metrics" ]; then
    echo "skip $base (no metrics dir)"
    return 0
  fi
  python -m edit_judge_bias.experiments.plot_results \
    --metrics-dir "$base/metrics" --figures-dir "$base/figures"
  python -m edit_judge_bias.visualization.plot_robustness_layers \
    --metrics-dir "$base/metrics" --figures-dir "$base/figures"
  python -m edit_judge_bias.visualization.plot_leaderboard \
    --metrics-dir "$base/metrics" --figures-dir "$base/figures"
}

if [ "$#" -ge 1 ]; then
  render "$1"
else
  render results/v2
fi
