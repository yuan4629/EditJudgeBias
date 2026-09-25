#!/usr/bin/env bash
# Fairness track: freeze the human construct-validity verdicts into a table. Free, local.
#
#   bash scripts/construct_validity.sh --dry-run   # coverage report, writes nothing
#   bash scripts/construct_validity.sh             # writes the table
#
# Consumes the `human_verdict` fields that scripts/stage_ds_visual_check.py writes EMPTY into
# the adjudication sheets. The three sheets use three different vocabularies and are never
# pooled:
#   data/human_validation_fairness/pairs.jsonl   D-G pairs      valid | moved | noflip | noperson
#   data/human_validation_ds/index.jsonl         D-S v3 sheets  plausible | recoloured | leaked | no_person
#   data/human_validation_ds_v4/index.jsonl      D-S v4 sheets  pass | patch | weak | figure
#
# A half-filled sheet yields `status=partial` with `n_annotated` beside `n_total`. Report the
# result as "pass rate x on n annotated sheets, 95% Wilson [lo, hi]".
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8   # some builders print non-ASCII; keeps Windows consoles working

python -m edit_judge_bias.experiments.build_construct_validity_table "$@"
# The D-G adjudication: whose side the human takes when the two MLLM auditors disagree.
python scripts/analyze_dg_verdicts.py "$@"
