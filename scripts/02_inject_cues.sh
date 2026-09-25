#!/usr/bin/env bash
# Stage 2 -- render the cue images. Free, deterministic, resumable (existing images are
# skipped). Original images are never modified: every injection writes a new file and a new
# JSONL row that points back to its `base_sample_id`.
#
#   bash scripts/02_inject_cues.sh
#
# Outputs:
#   data/images/biased/             main grid (11 image cues incl. `sham`) + 3 one-sided pair cues
#   data/images/biased_pair_fill/   the fill's one-sided pair cues (8 conditions)
#   data/images/control/            edit-damage positive controls for the validators
#   data/manifests/biased_*.jsonl   one row per rendered image
#
# Text-bearing cues render with Arial; see docs/REPRODUCTION.md ("Fonts") before comparing
# pixels across machines. Verify with: python scripts/verify_benchmark.py images
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working
M=data/manifests

echo ">> [1/5] main grid: 11 image cues on the judged subset"
python -m edit_judge_bias.experiments.run_bias_injection \
  --config configs/experiment/bias_injection_full_v2.yaml
python -m edit_judge_bias.data.validate_manifest --biased "$M/biased_samples_full_v2.jsonl" --root .

echo ">> [2/5] pair members: the three one-sided cues of the main pairwise arm"
python -m edit_judge_bias.experiments.run_bias_injection \
  --config configs/experiment/bias_injection_pair_members_v2.yaml

echo ">> [3/5] fill pair members: seed the members that already have main-grid images, then render"
# `snapshot` records data/images/biased so `audit` can prove the fill never wrote there.
# `seed` copies the rows of members that are also scoring samples, so they keep the main
# grid's pixels; it is load-bearing for which bases count as newly rendered downstream.
[ -f results/logs/fill_v2_protected_images_snapshot.json ] || python scripts/prepare_fill_injection.py snapshot
python scripts/prepare_fill_injection.py seed
python -m edit_judge_bias.experiments.run_bias_injection \
  --config configs/experiment/bias_injection_pair_members_fill_v2.yaml
python scripts/prepare_fill_injection.py audit

echo ">> [4/5] edit-damage controls (validator sensitivity), on the sham arm's validator draw"
python scripts/prepare_edit_damage_control.py manifest
python -m edit_judge_bias.experiments.run_bias_injection \
  --config configs/experiment/bias_injection_editdamage_v2.yaml
python scripts/prepare_edit_damage_control.py subset edit_damage_100
python scripts/prepare_edit_damage_control.py verify

echo ">> [5/5] validator manifests: sham matched floor, C-class template ablation, fill P4"
python scripts/validator_draw.py manifest
python scripts/validator_draw.py ablation-manifest
python scripts/prepare_fill_validation.py

echo ">> done"
