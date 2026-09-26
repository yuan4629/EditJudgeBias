# data/provenance

Tracked provenance metadata. Everything else under `data/` is built locally and git-ignored.

| File | What it is |
|---|---|
| `benchmark_v2_fingerprints.json` | Counts and SHA-256 digests of the published benchmark: upstream metadata files, every manifest (grouped by source or cue), the judged source images, and the decoded pixels of every rendered cue image. It holds no rows, ids, text or pixels. Checked by `scripts/verify_benchmark.py`; the `environment` block records the library versions it was produced with. |
| `distraction_spec_violation_audit.json` | The frozen measurement the exclusion sensitivity table reads (stage 6). For each of the 91 MagicBrush `distraction` images whose edit-region mask was later fixed, it records the sticker's corner, its changed pixels inside the true edit region, and whether it covered the region (21 did). It was measured on the images rendered before the fix, which the current code no longer produces. It holds sample ids and pixel counts, no pixels. |
