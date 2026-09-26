# scripts

Run everything from the repository root. Each shell script `cd`s there itself and sets
`PYTHONPATH=src`, so it also works without `pip install`.

## Pipeline (in order)

| Script | Stage | Cost |
|---|---|---|
| `01_build_manifests.sh` | per-source manifests, the combined pool, the judged subset, pair members | free |
| `02_inject_cues.sh` | cue images; edit-damage controls; validator draws | free |
| `03_validate_preservation.sh <validator> --use-api` | quality-preservation validation | paid |
| `04_run_judges.sh <judge>\|fill --use-api` | judge evaluation (main collection, then the fill) | paid |
| `05_compute_metrics.sh` | per-(judge, cue) metric tables | free |
| `06_build_tables.sh` | every analysis table | free |
| `07_plot_figures.sh` | every figure | free |
| `verify_benchmark.py {upstream,manifests,sources,images,all}` | compare a rebuild with the published fingerprints | free |

## Helpers (called by the stage scripts, or run by hand)

| Script | Purpose |
|---|---|
| `make_pair_member_manifest.py` | the samples every judged pair shows, as a samples manifest |
| `prepare_fill_injection.py {snapshot,seed,audit}` | stage the fill's pair-member injection and check it touched nothing else |
| `prepare_fill_validation.py` | the fill's validator draw (110 bases × 8 conditions) |
| `prepare_edit_damage_control.py {manifest,subset,verify,region-dose-audit}` | validator positive controls |
| `validator_draw.py {manifest,ablation-manifest,verify,backup,restore}` | reproduce the seeded validator draw; build the matched `sham` floor and the C-class ablation set |
| `audit_mask_polarity.py` | edit-region mask audit read by the sensitivity tables |
| `human_validation_package.py {build,status,unblind,build-iaa,kappa}` | blinded human preservation labelling |
