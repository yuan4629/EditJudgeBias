#!/usr/bin/env bash
# Stage 1 -- build the benchmark manifests from the upstream corpora.
#
#   bash scripts/01_build_manifests.sh
#
# Free and deterministic (seed 42). Needs the corpora laid out under tmp_data/ as described
# in docs/DATASETS.md. ImagenHub images are fetched from GitHub, and GenAI-Bench / MagicBrush
# from the Hugging Face Hub, on first use.
#
# Outputs (data/manifests/):
#   samples_full_v2.jsonl / pairs_full_v2.jsonl      the full pool
#   samples_judge_v2.jsonl / pairs_judge_v2.jsonl    the judged subset (breadth + anchor blocks)
#   samples_pair_members_v2.jsonl                    every image a judged pair shows
#
# Verify the result with: python scripts/verify_benchmark.py manifests
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working
M=data/manifests

echo ">> [1/8] I2EBench"
python -m edit_judge_bias.data.build_i2ebench --config configs/data/i2ebench_600.yaml
python -m edit_judge_bias.data.build_pairs \
  --samples "$M/samples_i2e600.jsonl" --out "$M/pairs_i2e600.jsonl" --config configs/data/pairs.yaml

echo ">> [2/8] ImagenHub (downloads images on first use)"
python -m edit_judge_bias.data.build_imagenhub --config configs/data/imagenhub.yaml

echo ">> [3/8] GenAI-Bench (Hugging Face)"
python -m edit_judge_bias.data.build_genaibench --config configs/data/genaibench.yaml

echo ">> [4/8] MagicBrush dev (Hugging Face; excludes the turns ImagenHub was built from)"
python -m edit_judge_bias.data.build_from_hf --config configs/data/magicbrush_dev.yaml

echo ">> [5/8] EBench-18K"
python -m edit_judge_bias.data.build_ebench --config configs/data/ebench18k.yaml

echo ">> [6/8] full pool + judged subset"
python -m edit_judge_bias.data.build_full_manifest --config configs/data/full_v2.yaml

echo ">> [7/8] pair members (the images the pairwise arms show)"
python scripts/make_pair_member_manifest.py \
  --pairs "$M/pairs_judge_v2.jsonl" --pool "$M/samples_full_v2.jsonl" \
  --out "$M/samples_pair_members_v2.jsonl"

echo ">> [8/8] validate"
python -m edit_judge_bias.data.validate_manifest \
  --samples "$M/samples_judge_v2.jsonl" --pairs "$M/pairs_judge_v2.jsonl" --root .

echo ">> done"
