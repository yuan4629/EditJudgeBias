#!/usr/bin/env bash
# Fairness track, D-S: deterministic skin-lightness counterfactuals on the OmniEdit pool.
#
#   bash scripts/fairness_ds.sh
#
# No generative model is involved: every injection composites through a person/skin mask,
# so outside the mask the output is the input's bytes verbatim (`max_outside_diff == 0`).
#
#   0  lock    OmniEdit shard set, frozen in data/provenance/omniedit_shard_lock.json   free
#   1a pool    pass 1: the candidate pool (k unset)                                     free
#   1b screen  construct-validity screen, gpt-4o-mini                                   PAID
#   1c pool    pass 2: same config + the screen's pass list (+ optional k)              free
#   2  edited  the dataset's own edited member for those scenes (local parquet)          free
#   3  inject  4 arms x 2 members                                                        free
#   4  gate    deterministic construction gate + contact sheets for a HUMAN look         free
#   5  emit    injections + gate -> judgeable SampleRecords                              free
#   6  judge   the panel, one judge per invocation                                       PAID
#   7  tables                                                                            free
#
# The order of 1a/1b/1c matters: `include_ids` filters BEFORE the seeded shuffle, so a seeded
# prefix taken before the screen is complete is not a subset of the one taken after it.
# The shard set is frozen and must not be topped up: adding a shard re-runs the permutation
# over a longer list and changes which scenes are drawn.
#
# This script runs the free steps and prints the paid ones for you to run deliberately.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working

# K is unset by default, which is the safe state while the screen is incomplete: with k
# unset every passing scene is kept, so topping the screen up only ADDS scenes. Set K once,
# after the screen is complete, and never lower it.
K="${K:-}"

echo "=== 1a. pool, pass 1 -- the candidate set (free) ==="
python -m edit_judge_bias.experiments.build_fairness_pool \
  --config configs/fairness/pool_ds_omniedit.yaml

cat <<'MSG'

=== 1b. construct screen -- PAID, run it yourself ===
python -m edit_judge_bias.experiments.run_construct_screen \
  --config configs/fairness/construct_screen.yaml --use-api

Resumable by sample_id. It writes results/v2_fairness/metrics/construct_screen_passed.json,
which step 1c consumes.
MSG

echo
echo "=== 1c. pool, pass 2 -- screened, collapsed, optional seeded prefix (free) ==="
# --out and --screens-out are mandatory: without them pass 2 would overwrite the candidate
# manifest that the construct screen reads.
if [ -f results/v2_fairness/metrics/construct_screen_passed.json ]; then
  python -m edit_judge_bias.experiments.build_fairness_pool \
    --config configs/fairness/pool_ds_omniedit.yaml \
    --include-ids results/v2_fairness/metrics/construct_screen_passed.json \
    --out data/manifests/samples_fairness_ds_v4_screened.jsonl \
    --screens-out results/v2_fairness/metrics/pool_screens_ds_v4_screened.jsonl \
    ${K:+--k "$K"}
else
  echo "    construct_screen_passed.json absent -- run step 1b first; stopping here"
  exit 0
fi

echo
echo "=== 2. fetch the edited member for the selected scenes (free, local parquet) ==="
python scripts/fetch_edited_images.py \
  --pool data/manifests/samples_fairness_ds_v4_screened.jsonl

echo
echo "=== 3. inject -- prefix pilot first (free) ==="
python -m edit_judge_bias.experiments.run_attribute_inject \
  --config configs/fairness/skin_tone_inject_v4.yaml --limit 20

echo
echo "=== 4. construction gate + contact sheets (free) ==="
python -m edit_judge_bias.experiments.run_construction_gate \
  --config configs/fairness/construction_gate_v4.yaml
python scripts/stage_ds_visual_check.py \
  --injections data/manifests/attribute_injections_v4.jsonl \
  --gate-json results/v2_fairness/metrics/construction_gate_v4.json \
  --out-dir data/human_validation_ds_v4

cat <<'MSG'

--- STOP. The gate ends with a human look at data/human_validation_ds_v4/: ---
Does each sheet look like a photograph of a person with a different skin tone, or like a
recoloured patch? If most are patches, report `sham_nonskin` as a control and make no
fairness claim.

--- After the look passes: the full injection (free, resumes from the pilot) ---
python -m edit_judge_bias.experiments.run_attribute_inject --config configs/fairness/skin_tone_inject_v4.yaml
python -m edit_judge_bias.experiments.run_construction_gate --config configs/fairness/construction_gate_v4.yaml

--- 5. emit the judgeable manifest (free; --report prints n and the MDE first) ---
python -m edit_judge_bias.experiments.build_fairness_samples_ds --report
python -m edit_judge_bias.experiments.build_fairness_samples_ds

--- 6. PAID: one judge per invocation ---
python -m edit_judge_bias.experiments.run_scoring_judge --config configs/experiment/scoring_fairness_v4.yaml --judge-config configs/judge/gpt5_5.yaml --use-api

--- 7. tables (free) ---
python -m edit_judge_bias.experiments.build_fairness_tables \
  --results-dir results/v2_fairness_ds \
  --samples data/manifests/samples_fairness_ds_judge_v4.jsonl
python -m edit_judge_bias.experiments.build_dose_control_decomposition \
  --results-dir results/v2_fairness_ds \
  --samples data/manifests/samples_fairness_ds_judge_v4.jsonl
python -m edit_judge_bias.experiments.build_dilution_algebra \
  --metrics-dir results/v2_fairness_ds/metrics \
  --manifest data/manifests/samples_fairness_ds_judge_v4.jsonl
MSG
