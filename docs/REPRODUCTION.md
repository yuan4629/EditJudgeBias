# Reproducing EditJudgeBias

This guide rebuilds the benchmark from the upstream corpora, runs judges on it, and produces
the analysis tables and figures.

What each part guarantees:

| Part | Deterministic? | How to check |
|---|---|---|
| Manifests (samples, pairs, judged subset) | yes, row for row | `verify_benchmark.py manifests` |
| Cue images | yes, pixel for pixel, given the reference fonts and libraries | `verify_benchmark.py images` |
| Preservation validation and judge verdicts | **no**: API models are non-deterministic and drift over time | runners are resumable, and raw responses are kept beside the parsed verdicts |
| Metrics, tables, figures | yes, given the same verdicts | re-run stages 5–7 and compare |

## 1. Environment

```bash
pip install -e ".[ingest,dev]" -c constraints.txt
```

The published fingerprints were produced on this reference environment:

- Windows 11
- Python 3.11.7
- Pillow 12.2.0 (FreeType 2.14.3, libjpeg-turbo 3.1.4.1)
- numpy 1.26.4
- scipy 1.11.4

`data/provenance/benchmark_v2_fingerprints.json` records the full list under `environment`.
Run every command from the repository root; the scripts `cd` there themselves.

### Determinism constraints

- **Corpus location.** Keep the corpora under `tmp_data/` exactly (a symlink or junction is
  fine). Manifest paths such as `tmp_data/EditBench/...` seed part of the sampling, so a
  different location draws a different benchmark.
- **Fonts.** `watermark`, `text_overlay` and `detail_caption` draw text with Arial. The
  reference font is Windows' `arial.ttf`, Arial Regular version 7.01, with sha256
  `baa251526d6862712a58e613ef451d8a2b60482142ec6aab1d47fb8e23e21a7c`.
  - On Windows it is found automatically.
  - Elsewhere, Pillow falls back to DejaVu Sans or its built-in font, and those three cues
    (plus `detail_caption`'s line wrapping) will differ. To match them, point the injectors
    at the same file: add `font_path: /path/to/arial.ttf` under `config:` in the three cue
    configs. `font_path` is not recorded in the manifests, so rows and pixels then match.
    Arial is licensed by Microsoft and is not distributed here.
- **Imaging libraries.**
  - `sham` is a JPEG round trip, so it depends on the libjpeg-turbo that Pillow bundles.
    Other cues depend on Pillow's resampling and FreeType. Use the pinned Pillow.
  - scipy must be installed: without it, the edit-region estimate falls back to a coarser
    method and the region cues (`zoom_inset`, `distraction`, `region_annotation`) move.
- **Seeds.** Every random draw is seeded (seed 42) with string-keyed generators, so results
  do not depend on `PYTHONHASHSEED` or on processing order.

## 2. Build the benchmark (free)

```bash
python scripts/verify_benchmark.py upstream     # check the downloaded corpora first
bash scripts/01_build_manifests.sh              # manifests
bash scripts/02_inject_cues.sh                  # cue images + validator draws
python scripts/verify_benchmark.py all          # compare against the fingerprints
```

Stage 1 downloads ImagenHub images from GitHub and two datasets from the Hugging Face Hub
(pinned revisions). Everything else is local. Stage 2 writes about 25,000 PNG images (on
the order of 10 GB).

### When a group differs

`verify_benchmark.py` reports matches per group: by source (and editor) for manifests and
source images, and by cue for rendered images. Usual causes:

| Differing group | Likely cause |
|---|---|
| `upstream :: tmp_data/...` | a different upstream version; re-download, or check the pinned revision |
| `sources :: I2EBench/fluxkontext`, `I2EBench/qwen_image_edit` | these two editor folders are not in the public I2EBench release (see [DATASETS.md](DATASETS.md#i2ebench)) |
| `sources :: ImagenHub/...` | a different museum revision (see `revision` in `configs/data/imagenhub.yaml`), or an incomplete download |
| every I2EBench / EBench-18K manifest group | the corpora are not under `tmp_data/`, so the paths (and the draws they seed) differ |
| `images :: watermark`, `text_overlay`, `detail_caption` only | a different font (see Fonts above) |
| `images :: sham` only | a different libjpeg-turbo (Pillow version) |
| `images :: zoom_inset`, `distraction`, `region_annotation` | scipy missing, or a different scipy |

A difference in the judged subset propagates to everything downstream. Fix the earliest
differing stage first.

## 3. Preservation validation (paid)

```bash
export EDITJUDGE_API_KEY=...        # see .env.example
export EDITJUDGE_BASE_URL=...
bash scripts/03_validate_preservation.sh gemini --use-api
bash scripts/03_validate_preservation.sh gpt4o_mini --use-api
bash scripts/03_validate_preservation.sh glm4v --use-api
```

Each validator judges, per image, whether the cue changed how well the edit was carried
out. It sees the original image, the unmodified edit and the cue image, and its verdicts
never enter the judge evaluation. The arms are:

- **Main draw:** 110 images per cue, seeded (the same 110 for every validator).
- **Sham floor:** `sham` on the union of the cues' drawn base images, so each cue's pass
  rate is compared with a zero-dose floor on the same pictures.
- **Edit-damage positive controls:** the edit region reverted to the original pixels (25,
  50 and 100%) or blurred, on 110 base images. These measure each validator's sensitivity.
- **Fill P4:** the fill's newly rendered pair images.
- **C-class template ablation** (gemini only).

The result is evidence about each cue, not a certificate for each image: the scoring arms
alone render about 1,200 images per cue (the pairwise arms add their own), and the
validators see 110 of them. The analysis reads a cue's pass rate (`quality_combined.csv`,
gate 0.85) and its paired comparison with the `sham` floor. The human spot checks (see
[Human validation](#human-validation)) label a subset of cue images; those labels are not
distributed.

Run one validator at a time. The payloads are large images, and parallel processes mostly
add transport failures. Re-run a command to fill in transient failures; nothing is asked
twice.

## 4. Judge evaluation (paid)

```bash
bash scripts/04_run_judges.sh gpt5_5 --use-api
bash scripts/04_run_judges.sh gemini_flash_lite --use-api
bash scripts/04_run_judges.sh kimi_k25 --use-api
bash scripts/04_run_judges.sh gpt4o_viescore --use-api
bash scripts/04_run_judges.sh qwen35_plus --use-api
bash scripts/04_run_judges.sh qwen3vl_32b_instruct --use-api   # replication judge
bash scripts/04_run_judges.sh fill --use-api                   # after all of the above
```

- To see how many calls an arm will make before spending anything, pass `--dry-run`
  instead of `--use-api` to the runner. The runner commands are printed in the script
  header.
- The `model_name` values in `configs/judge/*.yaml` are the ids the reference endpoint
  served. Set `api_model:` if your endpoint uses different ids. The results keep the
  `model_name` label.
- Every runner:
  - resumes by `result_id` (re-running asks only what is missing);
  - caches identical requests under `results/cache/`;
  - writes the raw response before parsing it;
  - records parse failures instead of stopping.
- **Collect all of a judge's cells within one short period.** The same model behind the
  same endpoint can drift between days. The analysis compares each cue with the unbiased
  baseline and with `sham`; a sham collected at a different time than the baseline mixes
  drift into the comparison.
- The **fill** completes the anchor and one-sided pairwise grids for the cues the main
  collection did not cover. It writes to `results/v2_fill`, skipping anything already
  answered in `results/v2`. The analysis reads each fill cell against the fill's own
  `sham`, in its own multiple-testing family.

### Collection batches behind the published results

The published verdicts come from two collections through the same OpenAI-compatible
endpoint (an API relay that routes each call to one of several upstream channels). All
times are UTC.

| Collection | When | What it asked | Reference arm |
|---|---|---|---|
| main (`results/v2`) | 2026-07-26 to 07-29; 91 items of each of three C-class cues re-judged on 08-16 after an edit-region mask fix; replication judge 08-17/18 | scoring on the stratified block (13 conditions and `sham`); the anchor block for `padding`, `text_overlay`, `brightness`, `region_annotation`; pairwise `position` and three cues; the retest | the same collection's un-cued baseline |
| fill (`results/v2_fill`) | 2026-09-13 21:04 to 09-14 03:13 | the other eight anchor cues; the other seven one-sided image cues, `bandwagon` and `model_name` on the pairs; `sham` in both arms | the fill's own `sham` |

The main invariance table's values, bold cells and floors use the main collection only.
Its agreement markers for the eight fill cues, and the fill rows of `claim_b.csv` and
`pairwise_one_sided.csv`, use the fill.

Limitations of the fill comparison:

- **Between-collection drift.** Byte-identical requests on the same 585 anchor items scored
  differently in the two collections for two judges. The fill's `sham` minus the main
  baseline, on the summed score, is +1.07 for gemini-3.5-flash and +2.04 for gpt-5.5; the
  other judges stay within ±0.08 (`fill_collection_drift.csv`). The time and the serving
  channel both differ between the collections, so the local records cannot tell whether the
  routing or the upstream model changed. This is why fill cells are read against the fill's
  `sham` rather than the main baseline.
- **What the same-collection `sham` removes.** A shift common to the `sham` and the cue
  arm, such as a collection-wide offset. Whether the shift is one additive constant cannot
  be tested with one `sham` per collection.
- **What it does not remove.** (i) Drift within the fill run: conditions were asked in a
  fixed priority order over about 6 h, and a judge's cue blocks were asked from about 0.5 h
  before to 5 h after its `sham` block. (ii) Channel changes within the run: according to
  the relay's per-call usage records, most of gemini-3.5-flash's cue blocks, and gpt-5.5's
  scoring `aesthetic_filter` and both `model_name` blocks, were served by a different
  upstream channel than the matching `sham`. The other judges' fill cells share their
  `sham`'s channel. (iii) `sham` is a zero-dose control, not a dose-matched one.
- The client sends nothing that selects a channel, and the usage records are not
  distributed. The main collection also has time gaps within it: gpt-5.5's `sham`,
  `bandwagon` and `model_name` cells were asked about 19 h after its baseline.

## 5–7. Metrics, tables, figures (free)

```bash
bash scripts/05_compute_metrics.sh
bash scripts/06_build_tables.sh
bash scripts/07_plot_figures.sh
```

The order matters: stage 6 reads stage 5's output, and stage 7 reads both. Always build
tables through `06_build_tables.sh`, because it sets the subset filters and collection
arguments that a bare call to an individual builder would miss.

Given the same verdicts and the inputs listed under
[Inputs outside this repository](#inputs-outside-this-repository), these stages reproduce
every file they write byte for byte. This was checked against the published results
with the versions in `constraints.txt`: the tables and figures the three scripts write
were identical. `invariance_noise_floor.csv` reproduces every value, floor and marker of
the paper's main invariance table; stage 6 prints its bold-cell counts per dimension.
`agreement_by_dimension.csv` reproduces the same table's agreement markers (all 180 cells);
stage 6 prints its marker counts, uncorrected and under BH.
PNG figures also depend on the matplotlib version and the installed fonts. The validators' per-cue summaries (`quality_preservation_summary*.csv`) are written
by stage 3, not here.

### Reading the statistics

Tests are Benjamini–Hochberg-corrected within the family named in each table's `family`
column (a few exploratory tables say `uncorrected — exploratory` there instead). Three
outputs use a rule that is not a corrected test:

- **Bold cells of the main invariance table** (`invariance_noise_floor.csv`). This is a
  descriptive rule, not a test. A cell is bold when the 2.5% lower bound of its bootstrap
  mean exceeds both of the judge's own floors: the 20th percentile of the bootstrap means of
  its `sham` and of its test–retest differences. It controls no error rate, and a bold cell
  is not "BH-significant".
- **Agreement markers of the same table** (`agreement_by_dimension.csv`, column `marker`).
  A cell is marked down or up when the 95% turn-cluster bootstrap interval of Δρ lies
  entirely below or above zero. This is per cell and uncorrected. Main-grid cues are read
  against the un-cued baseline, fill cues against the fill's own `sham`. Beside `marker`
  the file adds a bootstrap p-value and BH columns, with all 180 cells as one family
  (`q_value`, `marker_bh`) and per collection (`*_by_collection`). They do not change
  `marker`.
- **The rank half of `claim_b.csv`** (`rho_ci_excludes_zero`): the same uncorrected interval
  rule, on the summed score. Its accuracy half is BH-corrected (`claim_B_acc`, and
  `claim_B_acc [fill]` for the fill rows).

The resampling unit of each analysis:

| Output | Interval or test | Unit | Resamples, seed |
|---|---|---|---|
| `invariance_noise_floor.csv` (cells and floors) | percentile bootstrap of the mean | item | 2,000, 42 |
| `agreement_by_dimension.csv`; `claim_b.csv` rank half | percentile bootstrap of Δρ | editing turn (source image + instruction) | 1,000, 42 |
| `claim_b.csv` accuracy half | Wilcoxon on each turn's net discordance over decisive pairs | editing turn | — |
| `claim_a*.csv` | Wilcoxon signed-rank; percentile bootstrap of the mean shift | item | 2,000, 42 |
| `pairwise_one_sided.csv` | exact McNemar | pair (not clustered by turn) | — |
| `quality_combined.csv` | McNemar against `sham` on the same base images (Fisher also reported) | base image | — |
| `claim_b_permutation.csv` | cue labels permuted within each item | item | 10,000, 42 |

## Inputs outside this repository

This repository distributes no images, human labels, raw model responses, judge or
validator verdicts, run logs or API credentials. The published analyses also used the
inputs below, which no code in this repository can regenerate. Without them the dependent
tables are skipped, or rebuilt from what is available (as noted).

| Input | Used by | Without it |
|---|---|---|
| Judge and validator verdicts, with their raw responses (`results/`) | stages 5–7 | nothing to analyse; collect your own (stages 3–4) |
| Human preservation labels (`data/human_validation*/`) | human-validation and inter-annotator tables | those tables print `skip` |
| `results/v2/metrics/distraction_spec_violation_audit.json` | exclusion sensitivity table | re-measured on the current images (the build prints `[freeze] measured and wrote ...`); the published table used a measurement taken on images rendered before an edit-region mask fix, which the current code no longer produces |
| I2EBench `fluxkontext` and `qwen_image_edit` outputs (generated by us; not in any public release) | the I2EBench pool, the I2EBench strata of the judged subset and pairs, and every cue's validator draw | 480 instead of 600 I2EBench rows and a different benchmark from there on (see [DATASETS.md](DATASETS.md#i2ebench)) |

### Human validation

`scripts/human_validation_package.py` builds a blinded labelling package from the images the
validators judged. The annotator sees only index-named items; the key file stays with you.
After labelling, run these in order:

```bash
python scripts/human_validation_package.py unblind
python scripts/human_validation_package.py kappa
bash scripts/06_build_tables.sh
```

Skipping a step leaves agreement statistics computed on a mix of old and new labels.
