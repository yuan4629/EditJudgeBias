# Contributing

Thanks for your interest in EditJudgeBias. Issues and pull requests are welcome, in
particular:

- new judge adapters;
- new cues;
- additional source corpora;
- fixes that make the benchmark easier to rebuild.

## Development setup

```bash
pip install -e ".[ingest,dev]" -c constraints.txt
pytest -q
ruff check src tests scripts
```

The test suite is offline and uses synthetic fixtures only. It must not download data or
call an API. A few tests read artefacts of a full rebuild and skip themselves when those
are absent.

## Conventions

- Python 3.10+, `pathlib` for paths, `pydantic` for every record schema.
- Every manifest and result file is JSONL, and every artefact traces back to a `sample_id`.
- Every CLI takes `--config` (YAML); batch runners take `--dry-run` and are resumable.
- All randomness is seeded. Never change the seed or the size of an existing draw: a
  seeded prefix is only stable while its candidate list is.
- Real API adapters must be gated behind `--use-api` and read keys from environment
  variables only. Never commit a key.
- Behaviour belongs in `configs/`, not in code. New modules come with unit tests.

## Changes that alter the benchmark

A change that alters which samples are drawn, or any rendered pixel, changes the benchmark
itself. That includes:

- sampling;
- the edit-region estimate;
- an injector;
- path handling;
- a dependency pin.

Such changes must say so in the pull request and in `CHANGELOG.md`, bump the benchmark
version, and regenerate `data/provenance/benchmark_v2_fingerprints.json` under a new name.
Verify an unchanged benchmark with `python scripts/verify_benchmark.py all`.

## Data

Do not add images, annotations, judge outputs or any other dataset content to the
repository. Everything under `data/` except `data/provenance/`, and everything under
`results/` and `tmp_data/`, is git-ignored on purpose.
