# Run2 results and split audit, 2026-10-21

Authoritative artifacts: `artifacts/wbf_cloud_monitor/run2_final/` downloaded
from `baohuynh05/scale-error-wbf-val-test-safe-run2`. Kaggle reports ERROR.
Val completed; Test inference did not start because the split audit rejected
near-duplicate image content. A failed run is not a failed improvement trial.

Val metrics from `ensemble_outputs/validation_tuning_metrics.json`:

| Metric | Percent |
| --- | ---: |
| mAP50:95 | 44.9940 |
| AP50 | 77.4347 |
| AP75 | 43.8979 |
| AP-small | 31.2113 |
| DiVat | 29.2363 |
| DiVatLoiLom | 35.1173 |
| LoiChi | 65.7990 |
| LoiNhua | 50.4858 |
| LoiTray | 44.3318 |

These are tuning-set metrics on 878 images, not an independent generalization
estimate. The historical V9 46.10% used a different evaluation split and cannot
establish a gain or regression against this run.

## Evidence of split contamination

The exception prints at most five pairs; five is not the total overlap count.
A metadata-only audit of `images[].file_name` finds 18 shared export families,
covering 19 Val images and 19 Test images. Its family rule removes the Roboflow
`_jpg.rf.`/`.rf.` suffix; it does not establish pixel identity for every family.
The full list is in `run2_final/filename_family_overlap_audit.json`.

Downloaded the two `MyImage-20250723185242068_jpg.rf.*.jpg` files from the declared
Kaggle dataset's `valid/images` directory. Direct visual inspection shows the
same bottle pose, background, and fine defect structures. Both are 1024x1024;
mean absolute pixel difference is 2.6742/255, maximum 19. File SHA256 differs,
consistent with separately encoded near-identical images. Measurements and
hashes are in `duplicate_review/pixel_comparison.json`. This confirms a real
content overlap for this pair, not merely a filename or dHash suspicion.

No Test scores or labels were used to select a candidate. Dataset files and
the rejection threshold remain unchanged. A defensible independent Test result
requires a separately versioned split with acquisition/export families kept
together and checkpoint training provenance checked. Do not silently remove
the reported Test images or reuse the old Test score as a fresh result.

## Current experiment

Submitted existing `kernel_wbf_routing_candidate` version 1 as
`baohuynh05/scale-error-wbf-routing-val`; confirmed RUNNING. Its default phase
is `tune`, so it does not evaluate Test. Before submission, the routing/cache
and leakage suite passed: 24 tests. The job caches Val streams and compares
the predeclared ten-route bank against a reproduced baseline. The incumbent
is retained unless the declared acceptance checks pass. Later trial families
are specified in `wbf_next_experiments_20261021.md`.
