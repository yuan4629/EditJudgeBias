# Datasets

EditJudgeBias ships **no images and no annotations**: no original, edited or rendered cue
images, no human labels, and no upstream metadata files. It is built from five public
image-editing benchmarks. You download each from its owner under that owner's license, and
the builders in `src/edit_judge_bias/data/` turn them into JSONL manifests.

The five public benchmarks provide almost every input. The exception is the I2EBench
component: exact reconstruction also needs the outputs of two editors, `fluxkontext` and
`qwen_image_edit`, that we generated ourselves. They are not part of the upstream I2EBench
release and are not distributed here; see the I2EBench section below.

All corpora live under `tmp_data/` in the repository root (git-ignored). **Keep that exact
name.** Manifest paths are stored relative to the root (`tmp_data/...`), and those path
strings seed part of the benchmark's sampling. A different location would therefore draw a
different benchmark, not just move it. `tmp_data/` may be a symlink or junction to another
disk; the builders keep the paths lexical.

After downloading, check your copies before building:

```bash
python scripts/verify_benchmark.py upstream
```

## Overview

| Source | Role | Obtained by | License / terms |
|---|---|---|---|
| I2EBench | breadth block | manual download | see the I2EBench repository |
| ImagenHub (Text-Guided IE) | breadth block, human-anchor block | fetched by the builder + manual metadata | see the ImagenHub repository |
| GenAI-Bench (image editing) | breadth block | Hugging Face Hub (automatic) | see the dataset card |
| MagicBrush (dev split) | breadth block | Hugging Face Hub (automatic) | see the dataset card |
| EBench-18K (LMM4Edit) | breadth block, human-anchor block | manual download | **no license declared**, see below |

The breadth block is a stratified subset (source × edit type) used for the cross-source
invariance analyses. The human-anchor block consists of samples with human quality ratings,
used for the agreement analyses. Both are drawn with seed 42 by
`build_full_manifest --config configs/data/full_v2.yaml`.

Three of the sources share content. ImagenHub's rated items are a subset of MagicBrush dev,
and most GenAI-Bench prompts also come from MagicBrush dev. The MagicBrush builder therefore
excludes the ImagenHub turns (via `dataset_lookup.csv`), and analyses should treat these
sources as one content pool, not three.

## Required layout

```
tmp_data/
  EditBench/                              I2EBench
    EditData/<Category>/<Category>.json   instructions (14 categories used)
    EditData/<Category>/input/            original images
    EditResult/<Category>/<editor>/       edited images, matched to inputs by file stem
  imagenhub_meta/                         ImagenHub metadata
    dataset_lookup.json
    dataset_lookup.csv
    rater1.tsv  rater2.tsv  rater3.tsv
  imagenhub_museum/tree_main.json         cached GitHub tree listing (created on first run)
  hf_cache/                               Hugging Face cache (created on first run)
  editing_all/                            EBench-18K images
    sourceimg_h/  sourceimg_l/            originals (high-level / low-level edits)
    targetimg_h/  targetimg_l/            edited images, one folder per anonymous editor model00..model16
  ebench_meta/                            EBench-18K labels
    train_v.json train_e.json train_c.json train_yn.json
    test_v.json  test_e.json  test_c.json  test_yn.json
```

The builders write the images they fetch (ImagenHub, GenAI-Bench, MagicBrush) under
`data/images/{original,edited,mask}/<source>/`, and every manifest under `data/manifests/`.

## Sources

### I2EBench

- Download `EditBench/` from the link in the I2EBench repository's README (a Google Drive
  folder). Place it at `tmp_data/EditBench/`.
- Config: `configs/data/i2ebench_600.yaml`. The builder reads each category's
  `<Category>.json`, pairs each input with each editor folder under `EditResult/`, and
  draws 600 samples (seed 42).
- **Two editor folders are not in the upstream release.** The benchmark uses ten editor
  folders: `any2pix`, `hive`, `hqedit`, `iedit`, `instruct-diffusion`, `instructpix2pix`,
  `magicbrush` and `mgie` are upstream. `fluxkontext` and `qwen_image_edit` hold edits of
  the I2EBench inputs that we generated ourselves; they are not part of the public release,
  and no public dataset contains them. The builder uses every editor folder it finds under
  `EditResult/<Category>/`, so without these two:
  - `samples_i2e600.jsonl` has 480 rows instead of 600 (the same 60 image–instruction
    groups, eight editors each);
  - the I2EBench strata of the judged subset and of the judged pairs are drawn from a
    different candidate list, so they differ (106 of the 160 samples and 150 of the 168
    pairs), and so do the cue images rendered from them;
  - each cue's seeded 110-image validator draw keeps its images from the other sources and
    replaces its I2EBench ones (8 to 17 images per cue).

  The judged subset keeps its size (1,196 samples, 616 pairs), and the strata of the other
  sources have their own seeds and are unaffected.
  `verify_benchmark.py sources` reports the missing groups as `I2EBench/fluxkontext` and
  `I2EBench/qwen_image_edit`.
- **Known issue — instruction variant.** Upstream, `EditResult/` holds edits made from each
  item's *diverse* instruction (`div_exp`), and `EditResult_ori/` holds edits made from the
  *original* instruction (`ori_exp`). The benchmark reads images from `EditResult/` but
  pairs them with `ori_exp` (`instruction_field` in the config). Where the two phrasings ask
  for different edits, the judged instruction does not match the upstream editors' image. Within-sample cue
  effects compare the same image and instruction with and without a cue, so the mismatch is
  shared by both arms. It does, however, shift the baseline scores of the affected I2EBench
  items. I2EBench is not a human-anchor source.

### ImagenHub (Text-Guided Image Editing)

- Images are fetched by the builder from the ImagenHub museum site repository
  (`ChromAIca/ChromAIca.github.io`, `Museum/ImagenHub_Text-Guided_IE/`) over
  raw.githubusercontent.com, pinned to commit `d5f553773f57bdb98cfe2ecf77312dd07c818fb6`
  (`revision` in `configs/data/imagenhub.yaml`).
  - The GitHub tree listing is cached at `tmp_data/imagenhub_museum/tree_main.json` on the
    first run, because the unauthenticated API allows 60 requests/hour.
  - `verify_benchmark.py upstream` and `sources` check that you received the same files.
- The instruction lookup and the three raters' scores (`dataset_lookup.json`,
  `dataset_lookup.csv`, `rater{1,2,3}.tsv`) come from ImagenHub's released human-evaluation
  data. Place them in `tmp_data/imagenhub_meta/` and check them with
  `verify_benchmark.py upstream`.
- Config: `configs/data/imagenhub.yaml` (12 editors, the human reference edit, pairs).

### GenAI-Bench (image editing)

- Hugging Face `TIGER-Lab/GenAI-Bench`, file `image_edition/test-00000-of-00001.parquet`,
  pinned to revision `7981a194091d2201447f0c0e9cc4261dd5ad5846` in
  `configs/data/genaibench.yaml`. It is downloaded into `tmp_data/hf_cache/` on the first
  run.
- Sample ids hash the image bytes, so a different revision yields different ids.

### MagicBrush (dev split)

- Hugging Face `osunlp/MagicBrush`, the four `data/dev-*` shards, pinned to revision
  `1d8d4629150d18ca50afab66391866f2085be989` in `configs/data/magicbrush_dev.yaml`. They
  are downloaded on the first run.
- Build ImagenHub's metadata first: MagicBrush excludes the turns ImagenHub was built from,
  using `tmp_data/imagenhub_meta/dataset_lookup.csv`.

### EBench-18K (LMM4Edit)

- Images: download from the link in the LMM4Edit repository (`IntMeGroup/LMM4Edit`) and
  place the `sourceimg_*` / `targetimg_*` folders in `tmp_data/editing_all/`.
- Labels: the eight JSON files under `data/` of the same repository, placed in
  `tmp_data/ebench_meta/`.
- **The repository declares no license.** Obtain permission from the dataset authors before
  using or redistributing it. The editors are anonymous (`model00`–`model16`), so no
  per-editor claim can be made from this source.
- Config: `configs/data/ebench18k.yaml`.

## Inputs that cannot be rebuilt from these datasets

Some published analyses depend on inputs no downloadable dataset provides: the two
self-generated I2EBench editor folders above, the judge and validator verdicts with their
raw responses, the human preservation labels, and one frozen measurement file. The code
runs without them: the dependent tables are skipped, or clearly re-derived. The published
numbers need them, though. See [REPRODUCTION.md](REPRODUCTION.md#inputs-outside-this-repository).
