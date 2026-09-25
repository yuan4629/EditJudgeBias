# Datasets

EditJudgeBias ships **no images and no annotations**. It is built from five public
image-editing benchmarks, plus one more for an exploratory fairness track that is not part
of the reported experiments. You download each from its
owner under that owner's license, and the builders in `src/edit_judge_bias/data/` turn them
into JSONL manifests.

Most benchmark inputs are obtained from the five public source benchmarks.
Exact reconstruction of the reported I2EBench component additionally
requires the author-generated `fluxkontext` and `qwen_image_edit` outputs,
which are not part of the upstream I2EBench release; see the I2EBench
section below.

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
| OmniEdit-Filtered-1.2M | exploratory fairness track only | Hugging Face Hub (manual shards) | see the dataset card |

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
  downloads/                              OmniEdit shards (exploratory fairness track only)
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
  `magicbrush` and `mgie` are upstream. `fluxkontext` and `qwen_image_edit` were generated
  by the benchmark authors and are not part of the public release. Without them the
  I2EBench draw shrinks from 600 to 480 rows and the judged subset changes. `verify_benchmark.py sources` reports the affected groups as
  `I2EBench/fluxkontext` and `I2EBench/qwen_image_edit`.
- **Known issue — instruction variant.** Upstream, `EditResult/` holds edits made from each
  item's *diverse* instruction (`div_exp`), and `EditResult_ori/` holds edits made from the
  *original* instruction (`ori_exp`). The benchmark reads images from `EditResult/` but
  pairs them with `ori_exp` (`instruction_field` in the config). Where the two phrasings ask
  for different edits, the judged instruction does not match the image. Within-sample cue
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

### OmniEdit-Filtered-1.2M (exploratory fairness track only)

- Hugging Face `TIGER-Lab/OmniEdit-Filtered-1.2M`, split `train`. The fairness pool uses a
  fixed set of 15 shards. Their file names, sha256 and row counts are frozen in
  `data/provenance/omniedit_shard_lock.json`. Place those shards in `tmp_data/downloads/`.
- **Do not add shards.** The pool is a seeded shuffle of the candidate set, so any extra
  shard changes which scenes are drawn.

## Inputs that cannot be rebuilt from these datasets

Some published analyses depend on inputs no downloadable dataset provides. The code runs
without them: the dependent tables are skipped, or clearly re-derived. The published
numbers need them, though. See [REPRODUCTION.md](REPRODUCTION.md#inputs-outside-this-repository).
