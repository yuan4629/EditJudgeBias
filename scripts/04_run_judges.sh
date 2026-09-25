#!/usr/bin/env bash
# Stage 4 -- judge evaluation (PAID).
#
#   bash scripts/04_run_judges.sh <judge> --use-api     # the main collection, one judge
#   bash scripts/04_run_judges.sh fill --use-api        # the fill collection, all judges
#
#   <judge> = gpt5_5 | gemini_flash_lite | kimi_k25 | gpt4o_viescore | qwen35_plus
#             (published panel) | qwen3vl_32b_instruct (replication judge)
#
# Main collection, per judge (all rows go to results/v2):
#   scoring_breadth_v2   baseline + 11 image cues + 2 prompt cues on the breadth block
#   scoring_anchor_v2    4 cues on the human-anchor block (EBench-18K, ImagenHub)
#   scoring_retest_v2    200 baselines asked twice with the cache off (self-noise floor)
#   pairwise_v2          baseline + position swap + 3 one-sided cues on 616 pairs
#   pairwise_cr_v2       616 pairs asked twice with the cache off (pairwise self-noise)
#
# The fill collection completes the anchor and one-sided pairwise grids for the remaining
# cues (configs/experiment/fill_v2.yaml). It must run after the main collection: any
# request already answered in results/v2 is skipped, and rows go to results/v2_fill.
#
# Run one judge per invocation. Every runner resumes by result_id and caches identical
# requests, so re-running the same command only asks what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working

JUDGE="${1:-}"
USE_API="${2:-}"
if [ -z "$JUDGE" ] || [ "$USE_API" != "--use-api" ]; then
  echo "usage: $0 <judge|fill> --use-api   (see the header of this script)" >&2
  echo "Export EDITJUDGE_API_KEY and EDITJUDGE_BASE_URL first (see .env.example)." >&2
  exit 2
fi

if [ "$JUDGE" = "fill" ]; then
  python -m edit_judge_bias.experiments.fill_runner plan
  python -m edit_judge_bias.experiments.fill_runner run --use-api
  python -m edit_judge_bias.experiments.fill_runner verify
  exit 0
fi

JC="configs/judge/$JUDGE.yaml"
[ -f "$JC" ] || { echo "no judge config $JC" >&2; exit 2; }
E=configs/experiment

for arm in scoring_breadth_v2 scoring_anchor_v2 scoring_retest_v2; do
  echo ">> [$JUDGE] $arm"
  python -m edit_judge_bias.experiments.run_scoring_judge \
    --config "$E/$arm.yaml" --judge-config "$JC" --use-api
done
for arm in pairwise_v2 pairwise_cr_v2; do
  echo ">> [$JUDGE] $arm"
  python -m edit_judge_bias.experiments.run_pairwise_judge \
    --config "$E/$arm.yaml" --judge-config "$JC" --use-api
done
echo ">> done ($JUDGE)"
