# EditJudgeBias

A counterfactual benchmark for **bias in multimodal-LLM judges of instruction-based image
editing**.

MLLM judges are used to score and rank image editors. EditJudgeBias asks whether their
verdicts can be moved by cues that have nothing to do with editing quality. It takes real
`(original image, instruction, edited image)` triples and injects one cue, designed to leave
editing quality unchanged, into the edited image or into the judge prompt. It then measures
whether the judge's score or pairwise preference moves when it should not.

Quality preservation is supported by sampled, cue-level evidence. It is not certified image
by image. For each image-side cue, separate MLLM validators check a seeded sample of 110
rendered images; they never take part in the judge evaluation. Each cue's pass rate is read
against a `sham` floor on the same base images and against edit-damage positive controls,
which measure how often the validators catch a damaged edit. Human labels spot-check a
subset of cue images. Most judged cue images were never validated individually. `position`
and the two prompt cues leave the images untouched.

This repository contains the code to build the benchmark, run judges on it, and analyse
the results. **It contains no data**: no images (original, edited or rendered cue images),
no human labels, no raw model responses, no judge or validator verdicts, no logs and no API
credentials. The images come from public editing benchmarks you download from their owners
(see [docs/DATASETS.md](docs/DATASETS.md)), with one exception: exact reconstruction of the
I2EBench component also needs the outputs of two editors (`fluxkontext`, `qwen_image_edit`)
that we generated ourselves and that are not part of the public I2EBench release. Published
fingerprints let you check that your rebuild matches ours ([Reproducing the
benchmark](#reproducing-the-benchmark)). The inputs that a rebuild cannot recover are listed
in [docs/REPRODUCTION.md](docs/REPRODUCTION.md#inputs-outside-this-repository).

## What is measured

Thirteen conditions plus a placebo, grouped by where the cue enters the judge's input
(details in [docs/CUES.md](docs/CUES.md)):

| Site | Cues |
|---|---|
| Protocol | `position` (swap the A/B order; no pixel changes) |
| Pixels, global | `brightness`, `saturation`, `watermark`, `text_overlay`, `padding`, `aesthetic_filter` |
| Content near the edit | `zoom_inset`, `detail_caption`, `distraction`, `region_annotation` |
| Prompt | `bandwagon` (fabricated majority opinion), `model_name` (the editor is named) |
| Placebo | `sham` (JPEG round trip; the zero-dose control every effect is read against) |

The audit reports three complementary dimensions:

- **Invariance:** how much do the cues change the judge's ratings?
- **Agreement:** how do those cues change agreement with human judgments?
- **Stability:** are pairwise preferences preserved under A/B display-order swaps?

The main invariance table (`invariance_noise_floor.csv`, stage 6) reads each judge's
per-dimension rating change against two noise floors of that judge's own: the `sham`
control and a re-query of 200 identical un-cued inputs. Its agreement markers
(`agreement_by_dimension.csv`) give the per-dimension change in rank agreement with human
ratings on EBench-18K; they are uncorrected per-cell intervals (see
[Reading the statistics](docs/REPRODUCTION.md#reading-the-statistics)).

Judges are evaluated in two protocols: single-image **scoring** (three dimensions on a 1–10
scale) and **pairwise** preference.

## Installation

Python 3.10–3.12. Download or clone the repository, then run from its root:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[ingest,dev]" -c constraints.txt
pytest -q                                             # offline; no data or API key needed
```

`constraints.txt` pins the exact versions the published fingerprints were produced with.
The extras are:

- `ingest`: downloading corpora from the Hugging Face Hub.
- `overlap`: the source-admission near-duplicate gate.
- `perceptual`: optional LPIPS/CLIP preservation metrics.

## Repository layout

```
configs/               YAML that drives every step (no behaviour is hard-coded)
  data/                one file per source corpus + the combined pool (full_v2.yaml)
  bias/                one file per cue (injector parameters)
  judge/               one file per judge / validator (model id, decoding settings)
  experiment/          one file per arm (which manifest, which cues, where results go)
data/provenance/       tracked provenance metadata: the benchmark fingerprints
scripts/               the pipeline in stage order, plus helpers (see scripts/README.md)
src/edit_judge_bias/   the package
  data/                record schemas, JSONL I/O, per-source builders, manifest checks
  bias/                deterministic cue injectors and their registry
  judges/              judge adapters (OpenAI-compatible, mock) and a robust JSON parser
  prompts/             scoring, pairwise and validator prompt builders
  metrics/             scoring, pairwise, agreement and preservation metrics, statistics
  experiments/         CLI runners and table builders
  visualization/       figures
tests/                 unit and integration tests (synthetic fixtures only)
docs/                  datasets, reproduction guide, cue catalogue, architecture
```

Everything the pipeline builds (`data/manifests/`, `data/images/`, `results/`) and the
upstream corpora (`tmp_data/`) are git-ignored.

## Reproducing the benchmark

| Stage | Command | Cost |
|---|---|---|
| 1. Build manifests | `bash scripts/01_build_manifests.sh` | free |
| 2. Render cue images | `bash scripts/02_inject_cues.sh` | free |
| 3. Preservation validation | `bash scripts/03_validate_preservation.sh <validator> --use-api` | paid |
| 4. Judge evaluation | `bash scripts/04_run_judges.sh <judge> --use-api`, then `... fill --use-api` | paid |
| 5. Metrics | `bash scripts/05_compute_metrics.sh` | free |
| 6. Analysis tables | `bash scripts/06_build_tables.sh` | free |
| 7. Figures | `bash scripts/07_plot_figures.sh` | free |

Stages 1–2 are deterministic (seed 42). Check them against the published fingerprints:

```bash
python scripts/verify_benchmark.py all      # manifests, source images, rendered cue pixels
```

The paid stages call any OpenAI-compatible `/chat/completions` endpoint. The key and base
URL are read only from the environment (`EDITJUDGE_API_KEY`, `EDITJUDGE_BASE_URL`; see
`.env.example`), and no network call is made without `--use-api`. Every runner is resumable
and caches identical requests. The step-by-step guide covers what is and is not
reproducible byte for byte, and how to diagnose a mismatch:
**[docs/REPRODUCTION.md](docs/REPRODUCTION.md)**.

## Evaluating your own judge

1. Add a judge config, e.g. `configs/judge/my_judge.yaml`:

   ```yaml
   type: openai            # any OpenAI-compatible endpoint; `mock` needs no network
   model_name: my-judge    # label used in results and tables
   api_model: my-model-id  # optional: the model id sent to the endpoint, if different
   api_key_env: EDITJUDGE_API_KEY
   temperature: 0
   max_tokens: 1024
   image_detail: low
   cache_dir: results/cache
   ```

2. Check the plan offline (`--dry-run` makes no calls and writes nothing), then run a small
   paid smoke batch. The batch covers every condition on 6 samples, and the full run resumes
   over it:

   ```bash
   export PYTHONPATH=src
   python -m edit_judge_bias.experiments.run_scoring_judge --dry-run \
     --config configs/experiment/scoring_smoke_v2.yaml --judge-config configs/judge/my_judge.yaml
   python -m edit_judge_bias.experiments.run_scoring_judge --use-api \
     --config configs/experiment/scoring_smoke_v2.yaml --judge-config configs/judge/my_judge.yaml
   ```

3. Run the full grid with `bash scripts/04_run_judges.sh my_judge --use-api`, then stages
   5–7.

   The analysis builders have declared rosters (`--roster`): the five-judge panel and a
   separate replication roster. A new judge is best reported as its own roster, not pooled
   into the panel's multiple-testing family.

For a non-OpenAI API, implement `JudgeAdapter` (`src/edit_judge_bias/judges/base.py`) and
add it to `build_adapter` in `judges/__init__.py`. `MockJudgeAdapter` is a minimal example.

## Citation

If you use EditJudgeBias, please cite it as described in [CITATION.cff](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE). The benchmark is built from
third-party datasets that keep their own licenses and terms of use. This repository
redistributes none of them. See [docs/DATASETS.md](docs/DATASETS.md) before you download or
use them.
