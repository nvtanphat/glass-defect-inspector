# Three predeclared Val-only ensemble experiments

Date: 2026-10-21. This is a prospective plan, not a claim of gains. Read alongside
`wbf_ensemble_improvement.md` and `scripts/prepare_wbf_routing_candidate.py`.

## Starting evidence and interpretation

Run2 completed 878 Val images: mAP@50:95 44.994%, AP-small 31.21%; class AP in
`CLASS_NAMES` order is DiVat 29.24, DiVatLoiLom 35.12, LoiChi 65.80, LoiNhua
50.49, LoiTray 44.33. These supplied run2 figures should be reconciled against
the exact downloaded metrics before comparing a new run. The weak first two
classes and AP-small justify examining missed small defects; they do not prove
that scale is the cause. Separate missed detections from localization errors
and false-positive ranking with Val-only matched detections and image crops.
The historical 46.10% is not a same-split baseline. Test remains blocked by five
near-duplicate Val/Test pairs; do not relax that check or choose methods using
Test labels.

WBF uses confidence-weighted coordinates from multiple predictions [1]. The
current TTA merge instead takes the first box from each matched view, averages
coordinates equally and boosts the larger score. Its cluster center is not
updated while assigning subsequent boxes. Thus view routing and fusion policy
are concrete ablations, without assuming the implementation is necessarily
wrong. The existing routing candidate already declares ten routes, retains
baseline weights and checks two deterministic Val subsets. It caches original
and merged streams, but not every unmerged flipped/1280 stream: experiment 2
needs fresh inference or a richer cache, not reconstruction from fused boxes.

## Fixed experiment bank

Each numbered family counts as one trial for the user's three-consecutive-
nonimprovement stopping rule. All variants within it must be logged, including
losers. Freeze the bank before reading its scores. Each family starts from the
incumbent accepted configuration, with all other settings held fixed.

| Order | Concrete variants and selection | Why / principal failure mode |
| --- | --- | --- |
| 1. Class-specific route | Execute exactly the ten `ROUTING_BANK` entries already in `prepare_wbf_routing_candidate.py`. Preserve run2 baseline weights, IoU, floor and consensus settings; select independently per class, then evaluate the assembled five-class output under the existing global cap. No new routes or retuning weights during this trial. | WBF supports combining model and augmented-view outputs [1]. Removing a weak/correlated stream may improve localization or ranking; route selection itself can overfit Val. Single-model routes are useful controls, not assumed improvements. |
| 2. Score-weighted TTA | Keep selected model routes fixed. Compare the existing merge with two new variants: at the same matching thresholds (0.55 flip, 0.60 scale), use coordinates `sum(score * box) / sum(score)` for the first matched prediction per view, and fused score either arithmetic mean or maximum **without** the 1.05 boost. Leave unmatched boxes and all outer WBF settings unchanged. Select one variant globally, not separately for each class. | This isolates coordinate weighting plus two explicit ranking policies motivated by WBF [1]; these custom policies are ablations, not an exact reproduction of paper WBF. Wrong matches and confidence bias can move coordinates away from ground truth. Do not invert a previous TTA merge to fabricate the required streams. |
| 3. Soft-NMS after fusion | Compare incumbent with class-wise Gaussian Soft-NMS using only sigma `{0.3, 0.5, 0.7}`. Apply decay `score *= exp(-IoU²/sigma)` after selecting each highest-scoring box; retain boxes above the incumbent output floor and apply the same final global cap. Coordinates, outer WBF and all model streams remain fixed. Choose one sigma globally. | Soft-NMS decays overlapping detection scores and needs no retraining [2]. It may reduce residual duplicates but can also demote real neighboring defects; it cannot unmerge objects already collapsed by WBF. If duplicates are absent, a null/negative result is expected. |

## Acceptance, provenance, and stop

Use the unchanged evaluator/protocol and the exact same Val image identities.
Require assembled mAP@50:95 to improve by at least 0.10 percentage points and
positive improvement in both predeclared deterministic Val subsets, with
sufficient class support. Report every per-class AP, AP-small, AP50, detections
per image and runtime; keep the incumbent on ties/failures. The 0.10 threshold
is an engineering rule, not statistical significance. These subsets are
stability diagnostics, **not** independent cross-validation: existing baseline
hyperparameters were fitted on all Val images. Model selection on finite noisy
samples can overfit [3]. Do not describe a new best Val score as a generalization
gain. Bootstrap paired image/group deltas may describe uncertainty but cannot
remove adaptive-selection bias.

Record script/checkpoint/image/annotation/cache hashes, selected routes, every
candidate result, evaluator settings and the nonimprovement counter. Reject
cache provenance mismatches. Only a measured accepted improvement resets the
counter; after three consecutive rejected families, stop ensemble search and
freeze the incumbent. A blocked or crashed trial is not evidence of lack of
improvement. If a family improves, predeclare the next bank before continuing;
do not silently expand the within-family sweep until one variant wins.

Test is a separate final step after configuration freeze and resolution of the
split-provenance issue. Review duplicate image content and source identities
without Test labels. Any corrected split must have a separate version and must
not silently replace the benchmark. Never delete inconvenient Test examples or
lower the duplicate threshold just to obtain a score.

## Deferred direction

SAHI is a credible later option if Val diagnostics show small-object misses:
it supports sliced inference without fine-tuning [4]. It needs new tile
predictions and extra compute, and aerial-dataset gains do not establish glass
defect gains. It is deliberately outside this compact three-family bank; do
not add a tile-size/overlap search mid-run.

## Primary sources checked

1. Solovyev, Wang & Gabruseva, *Weighted boxes fusion: Ensembling boxes from different object detection models*, 2021. [Paper](https://arxiv.org/abs/1910.13302); [authors' implementation](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py).
2. Bodla et al., *Soft-NMS — Improving Object Detection With One Line of Code*, ICCV 2017. [Paper](https://arxiv.org/abs/1704.04503). Its reported benchmark improvements are not forecasts for this dataset.
3. Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 2010. [Paper](https://www.jmlr.org/papers/v11/cawley10a.html).
4. Akyon, Altinuc & Temizel, *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*, ICIP 2022. [Paper](https://arxiv.org/abs/2202.06934).
