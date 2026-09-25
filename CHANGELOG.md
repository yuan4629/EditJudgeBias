# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-26

First public release of the benchmark code (benchmark version v2).

### Added

- Builders for the five source corpora, the combined pool and the judged subset.
- Deterministic injectors for the image cues, the prompt-level cues and the controls.
- The quality-preservation validation runner, with sham floors and edit-damage positive
  controls.
- Scoring and pairwise judge runners (OpenAI-compatible and mock adapters) and the
  multi-judge fill orchestrator.
- Metric, analysis-table and figure builders, and the fairness track.
- Pipeline scripts `01`–`07`.
- `scripts/verify_benchmark.py` and the published fingerprints
  (`data/provenance/benchmark_v2_fingerprints.json`), to check a rebuild without
  redistributing any data.
- `constraints.txt` with the reference environment.

### Changed (relative to the code used for the paper)

- Hugging Face datasets are pinned to the revisions the benchmark was built from, and
  MagicBrush's shards are listed explicitly, so no repository listing is needed.
- Manifest paths stay relative when `tmp_data/` is a symlink or junction. Previously they
  became absolute, which silently changed the seeded draw. Unchanged for a regular
  directory.
- The fairness track's image-edit endpoint falls back to `EDITJUDGE_BASE_URL` when a config
  sets no `base_url`.
