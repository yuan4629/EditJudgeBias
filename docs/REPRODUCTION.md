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
pip install -e ".[ingest,dev]" -c constraints.txt     # add ,fairness for the fairness track
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
- **Fairness masks.** The fairness track's person/skin masks use torchvision's DeepLabV3.
  CPU and GPU inference can differ at mask boundaries, and without torch the code falls
  back to a Haar + GrabCut detector.
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
| `sources :: ImagenHub/...` | the museum repository's `main` branch changed |
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

- **Main draw:** 110 images per cue, seeded.
- **Sham floor:** `sham` on every cue's base images, so each cue's pass rate is read against
  a zero-dose floor on the same pictures.
- **Edit-damage positive controls:** the edit region reverted to the original pixels (25,
  50 and 100%) or blurred. These measure each validator's sensitivity.
- **Fill P4:** the fill's newly rendered pair images.
- **C-class template ablation** (gemini only).

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
with the versions in `constraints.txt`: all 64 tables and figures the three scripts write
were identical. PNG figures also depend on the matplotlib version and the installed
fonts. The validators' per-cue summaries (`quality_preservation_summary*.csv`) are written
by stage 3, not here.

## Fairness track

```bash
bash scripts/fairness_ds.sh          # skin-lightness counterfactuals (D-S); prints the paid steps
bash scripts/fairness_dg.sh          # generative attribute edits (D-G); stopped at its gate
bash scripts/construct_validity.sh   # human construct-validity verdicts -> table
```

The D-S pool depends on a paid construct-validity screen, so a fresh run draws its pool
from a new screen. Given the published D-S verdicts, the three table commands that
`fairness_ds.sh` prints as step 7 reproduce the published tables byte for byte.

## Inputs outside this repository

The published analyses also used the inputs below, which no code in this repository can
regenerate. Without them the dependent tables are skipped, or rebuilt from what is
available (as noted).

| Input | Used by | Without it |
|---|---|---|
| Judge and validator verdicts (`results/`) | stages 5–7 | nothing to analyse; collect your own (stages 3–4) |
| Human preservation labels (`data/human_validation*/`) | human-validation and inter-annotator tables | those tables print `skip` |
| Human construct-validity verdicts (`data/human_validation_ds*/`, `data/human_validation_fairness/`) | construct-validity table | `status=partial` rows |
| Construct-screen pass list (`results/v2_fairness/metrics/construct_screen_passed.json`) | D-S pool | produced by the paid screen |
| `results/v2/metrics/distraction_spec_violation_audit.json` | exclusion sensitivity table | re-measured on the current images (the build prints `[freeze] measured and wrote ...`); the published table used a measurement taken on images rendered before an edit-region mask fix, which the current code no longer produces |
| I2EBench `fluxkontext` and `qwen_image_edit` outputs | I2EBench strata | a smaller I2EBench draw |

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
