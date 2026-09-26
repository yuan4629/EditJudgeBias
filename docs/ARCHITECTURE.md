# Architecture

```
upstream corpora ──► manifests ──► cue injection ──► preservation validation ──► judges ──► metrics ──► tables ──► figures
   tmp_data/        data/manifests/   data/images/        results/*/quality/      results/*/   results/*/metrics/  results/*/figures/
```

Each stage reads JSONL written by the previous one and writes JSONL for the next. Every
artefact traces back to an original `sample_id`.

## Invariants

These hold everywhere in the code. Several are enforced by tests.

1. **Originals are immutable.** Each injection writes a new image and a new `BiasedRecord`
   that points to its `base_sample_id`. Nothing overwrites an original or an edited image.
2. **Raw responses are stored before parsing.** A judge's raw text is written to
   `raw_response_path` first. A parse failure keeps the raw text, sets
   `parse_success=false` with a `parse_error`, and does not stop the batch.
3. **Validation is a separate call.** Quality-preservation validators never take part in the
   judge evaluation, and judges never see validator output.
4. **Everything is seeded.** All randomness comes from `seed` (42) through string-keyed
   generators, so results do not depend on processing order or `PYTHONHASHSEED`.
5. **Every batch is resumable.** Runners skip any `result_id` / `biased_id` already on disk
   and log failures instead of aborting. Identical API requests are cached, so a re-run
   never pays twice.
6. **Behaviour lives in YAML.** Every CLI takes `--config`, and most take `--dry-run` to
   report the plan without writing anything.
7. **Network calls are opt-in.** A real judge adapter refuses to run without `--use-api`,
   and the key is read from an environment variable only.
8. **Grouping keys travel with every record.** `edit_type` (add / remove / replace /
   color / background / low-level) and `content_category` (human / animal / object /
   scenery / global) are on every sample, because the per-group breakdowns depend on them.

## Records (`src/edit_judge_bias/data/schema.py`)

| Record | One row per | Key fields |
|---|---|---|
| `SampleRecord` | edited image | `sample_id`, `source_dataset`, `edit_type`, `content_category`, `original_image_path`, `instruction`, `edit_model`, `edited_image_path`, `human_score`, `metadata` |
| `PairRecord` | two edits of the same (original, instruction) | `pair_id`, `sample_id_a/b`, `edited_image_a/b_path`, `ground_truth_preference`, `pair_quality_gap` |
| `BiasedRecord` | rendered cue image | `biased_id` (`<base_sample_id>__<cue>`), `base_sample_id`, `bias_type`, `bias_strength`, `biased_image_path`, `bias_params` |
| `JudgeResult` | judge call | `result_id`, `judge_model`, `task_type` (scoring / pairwise), `bias_type`, the three dimension scores, `fine_score`, `winner`, `biased_side`, `raw_response_path`, `parse_success` |
| `QualityValidationResult` | validator call | `biased_id`, `validator_model`, the `*_changed` flags, `passed`, `raw_response_path` |

All paths in manifests are POSIX paths relative to the repository root.

## Scoring

A scoring judge rates an edit on three dimensions, each on a 1–10 scale:

- instruction adherence;
- editing quality;
- detail preservation.

`fine_score` is the sum of the three dimension ratings (3–30), while
`overall_score` is the judge's separate overall rating. The headline
invariance analysis uses the three dimension ratings individually;
some auxiliary analyses use `fine_score`.

The headline table is `invariance_noise_floor.csv` (`build_noise_floor_table`). Each cell
is 10 × the mean absolute per-item change in one rating. Each judge has two floors per
rating, the `sham` change and the retest change, each taken as the 20th percentile of 2000
bootstrap means. A cell is marked by how many floors its 95% bootstrap lower bound
exceeds (a descriptive rule, not a test). The same table's agreement markers come from
`agreement_by_dimension.csv` (`build_agreement_by_dimension`): per rating, the change in
Spearman ρ against the matching EBench-18K human label, marked when its 95% turn-cluster
interval excludes zero (uncorrected; BH columns beside it).

## Result trees

```
results/
  v2/                      main collection
    raw_judgments/         <task>__<judge>.jsonl   unbiased baseline verdicts
    biased_judgments/      <task>__<judge>.jsonl   verdicts under a cue
    raw_responses/         raw text of every judge call
    quality/               validation__<validator>.jsonl
    metrics/               tables (stages 5-6)
    figures/               figures (stage 7)
  v2_fill/                 the fill collection (same layout; read with --fill-dir)
  v2_control/              validator edit-damage controls
  v2_ablation/             C-class validator template ablation
  v2_ablation_control/     the ablation template on the edit-damage controls
  cache/                   request cache (safe to delete; re-runs then call the API again)
```

`v2` names the benchmark version reported in the paper. Several builders check this name
(for example, `build_claim_tables` requires `--fill-dir` for `results/v2`), so keep it.

## Rosters and multiple testing

Judges are analysed in declared rosters, never in whatever happens to be in a directory:

- `PUBLISHED_ROSTER`: the five-judge panel.
- `A5_ROSTER`: the replication judge (`--roster a5`, tables prefixed `a5_`).

Each roster is its own Benjamini–Hochberg family. Fill rows form their own family as well
(`[fill]`), and each fill cell is measured against the fill's own `sham`. No main-collection
q-value changes when the fill is added.

## Package map

| Module | Responsibility |
|---|---|
| `data/` | schemas, JSONL I/O, one builder per source, the combined pool and judged subset (`build_full_manifest`), pairs, edit-type rules, manifest validation, the source-overlap gate |
| `bias/` | `BiasInjector`, one injector per cue, the edit-region estimate, fonts, registry |
| `prompts/` | scoring, pairwise and validator prompts; prompt styles and prompt-level cues |
| `judges/` | `JudgeAdapter`, `MockJudgeAdapter`, the OpenAI-compatible adapter, the tolerant JSON parser |
| `metrics/` | score shifts, pairwise flip rates, agreement with humans, preservation metrics, statistics (Wilcoxon, McNemar, Benjamini–Hochberg, bootstrap intervals) |
| `experiments/` | runners (`run_*`), the fill orchestrator, and all table builders (`build_*`, `aggregate_results`) |
| `visualization/` | figures |

## Glossary of arm labels

Code comments and some config names refer to experiment arms by short labels:

| Label | Meaning |
|---|---|
| claim A | invariance: does a cue shift the judge's score? (`claim_a*.csv`, `invariance_noise_floor.csv`) |
| claim B | agreement: does a cue change the judge's agreement with human rankings? (`claim_b*.csv`, `agreement_by_dimension.csv`) |
| A- / B- / C-class | prompt-level cues / global pixel cues / cues near the edit region |
| breadth, anchor | the stratified cross-source block, and the block of samples with human ratings |
| fill | the second collection that completed the anchor and one-sided pairwise grids (`results/v2_fill`) |
| FILL P4 | preservation validation of the fill's newly rendered pair images |
| WP-A1 | sensitivity analyses (exclusion, permutation, leaderboard simulation) |
| WP-A2 | re-rendering of the C-class cues after the edit-region mask fix (the released code renders the fixed images directly) |
| WP-A3 | the matched `sham` floor for the validators |
| WP-A4a/b | edit-damage positive controls (validator sensitivity) |
| WP-A4c/d | the C-class validator template ablation, and the same template on the edit-damage controls |
| WP-A5 | the replication judge (`qwen3-vl-32b-instruct`, roster `a5`) |
| WP-A6 | the second human preservation package and its second annotator (inter-annotator agreement) |

## Section references in comments

Some comments cite a section as `§n`, for example "the §9.4 breakdown". These refer to the
project's original design plan or to the paper. Neither document ships with this
repository. The design-plan sections cited most often are:

| Ref | Topic |
|---|---|
| §2.1, §4.2 | the initial scope: judges and cues |
| §3.3, §12.1 | record formats (see `data/schema.py`) |
| §5.4 | per-cue implementation details (see [CUES.md](CUES.md)) |
| §6, §6.1, §6.2 | quality-preservation validation: automatic checks, MLLM validators |
| §7.1–§7.3 | judge prompts: scoring, pairwise, mitigation |
| §8.1–§8.3 | scoring metrics, pairwise metrics, significance testing |
| §9.2–§9.5 | experiments: single-cue effects, cue strength, edit-type and content breakdown, mitigation |
| §13 | pilot decision thresholds |

In the analysis builders, `§5.7` (validator floors and controls) and correlation-related
`§7.2` refer to section numbers of an earlier draft of the paper. The surrounding comment makes clear what is meant.
