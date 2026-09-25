#!/usr/bin/env bash
# Stage 3 -- quality-preservation validation (PAID: one MLLM call per image).
#
#   bash scripts/03_validate_preservation.sh <validator> --use-api
#   <validator> = gemini | gpt4o_mini | glm4v
#
# A validator is a separate MLLM call that asks whether the cue changed how well the edit
# was carried out; it never takes part in the judge evaluation. The QC subset is gated on
# the intersection of the two calibrated validators (gemini, gpt4o_mini); glm4v is an
# independent-vendor cross-check reported against its own false-flag floor.
#
# Run one validator per invocation (they share the same image uploads; running them in
# parallel mostly adds transport failures). Every arm is resumable: re-running asks only
# what is missing, so a second pass is the way to mop up transient failures.
#
# Arms, in order:
#   main       110 images per cue (seeded draw)                -> results/v2/quality
#   sham       the matched sham floor: sham on every cue's draw -> results/v2/quality
#   damage     edit-damage positive controls (sensitivity)     -> results/v2_control/quality
#   fill       the fill's newly rendered pair images (P4)       -> results/v2_fill/quality
#              (gating validators only; quality_validation_fill_v2_p4_glm4v.yaml is optional)
#   ablation   C-class template ablation + its control (gemini only)
#              -> results/v2_ablation/quality, results/v2_ablation_control/quality
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working

VALIDATOR="${1:-}"
USE_API="${2:-}"
case "$VALIDATOR" in
  gemini)     SUFFIX="";   P4=gemini ;;
  gpt4o_mini) SUFFIX="_b"; P4=gpt4o_mini ;;
  glm4v)      SUFFIX="_c"; P4="" ;;   # the fill P4 arm was run on the two gating validators only
  *) echo "usage: $0 <gemini|gpt4o_mini|glm4v> --use-api" >&2; exit 2 ;;
esac
if [ "$USE_API" != "--use-api" ]; then
  echo "This stage calls a paid API. Re-run with --use-api after exporting" >&2
  echo "EDITJUDGE_API_KEY and EDITJUDGE_BASE_URL (see .env.example)." >&2
  exit 2
fi

run() { python -m edit_judge_bias.experiments.run_quality_validation --config "$1" --use-api; }
E=configs/experiment

echo ">> [$VALIDATOR] main arm";   run "$E/quality_validation_full_v2${SUFFIX}.yaml"
echo ">> [$VALIDATOR] sham floor"; run "$E/quality_validation_shammatch_v2${SUFFIX}.yaml"
echo ">> [$VALIDATOR] edit damage"; run "$E/quality_validation_editdamage_v2${SUFFIX}.yaml"
if [ -n "$P4" ]; then
  echo ">> [$VALIDATOR] fill (P4)"; run "$E/quality_validation_fill_v2_p4_${P4}.yaml"
fi
if [ "$VALIDATOR" = "gemini" ]; then
  echo ">> [$VALIDATOR] C-class template ablation"
  run "$E/quality_validation_cclass_ablation.yaml"
  run "$E/quality_validation_a4d_ablation_control.yaml"
fi
echo ">> done ($VALIDATOR). Re-run the same command once to mop up transient failures."
