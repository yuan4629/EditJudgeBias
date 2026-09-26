# data/provenance

Tracked provenance metadata. Everything else under `data/` is built locally and git-ignored.

| File | What it is |
|---|---|
| `benchmark_v2_fingerprints.json` | Counts and SHA-256 digests of the published benchmark: upstream metadata files, every manifest (grouped by source or cue), the judged source images, and the decoded pixels of every rendered cue image. It holds no rows, ids, text or pixels. Checked by `scripts/verify_benchmark.py`; the `environment` block records the library versions it was produced with. |
