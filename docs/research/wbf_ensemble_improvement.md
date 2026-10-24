# Research note: validation-safe ways to improve the WBF ensemble

Scope: inspect `kaggle/kernel_apex_hypersearch/run_apex_hypersearch.py` and identify evidence-based experiments that can use its frozen checkpoints and prediction streams. This note proposes experiments; it does not claim any will beat the current reported 46.10% mAP@50:95. Reported gains from other datasets are not transferable guarantees.

Treat 46.10% as a historical benchmark lead, not a clean estimate of generalization: earlier experiment code in this project optimized ensemble configurations against the held-out Test annotations. The new Val-tune/Test-once workflow should establish any fresh result; do not use Test feedback to target or confirm the historical number.

## What the current script actually searches

`generate_class_predictions_fast()` always fuses `f_tta`, `d_tta`, and `y_ms_tta`. Per class, `tune_wbf_hyperparameters()` runs deterministic coordinate search over IoU, three model weights, confidence floor, and consensus boost. It does not compare two-model versus three-model routing, original versus flipped views, 960 versus 1280 YOLO, or alternative TTA fusion. Those are promising candidates because the historical benchmark describes class-specific model/scale routing, but that result must be treated as a separate experiment, not as a result reproduced by this script.

The current augmentation merge (`merge_single_model_tta`) averages the first original/flip boxes 50:50 regardless of their scores and sets their fused score to `1.05 * max(score)`. The 960/1280 merge uses this same rule. A useful ablation is to compare this rule with score-weighted coordinates and the WBF paper's score-weighted fusion; retain per-class/view outputs so each choice is testable on Val. Do not interpret agreement boosts as calibrated probabilities: their effect is ranking and must be judged by the target AP protocol.

## Evidence and experiments, in priority order

| Priority | Val-only experiment | Rationale and cautions |
|---|---|---|
| 1 | **Select prediction branch per class:** compare FBDNet/D-FINE/YOLO subsets and original, flip-TTA, 960, 1280, and multi-scale streams. Start with a small predeclared branch bank, then tune WBF only for the selected branch. | WBF was proposed for model ensembles and same-model augmented views; it uses score-weighted boxes rather than retaining an arbitrary single box. The existing 46.10% result is evidence to include morphology-specific routing as a candidate, but not evidence that the current fixed three-stream implementation reproduces it. The branch bank requires caching the unmerged per-model/per-view predictions. [WBF paper](https://arxiv.org/abs/1910.13302) |
| 2 | **Tune TTA fusion per class/view:** compare no flip, flip with score-weighted box averaging, and the current equal-coordinate/max-score rule; tune matching IoU over a modest grid. For scratches/wires, retain the single-view control because augmentation can change confidence ranking and box localization. | WBF explicitly discusses fusing predictions from original and transformed views. Research on detection confidence bias finds confidence depends on box size and that TTA can confound this bias; hence a blanket TTA score boost may reorder boxes in a harmful way. [WBF paper](https://arxiv.org/abs/1910.13302), [box-size confidence-bias paper](https://openaccess.thecvf.com/content/WACV2023/html/Gilg_The_Box_Size_Confidence_Bias_Harms_Your_Object_Detector_WACV_2023_paper.html) |
| 3 | **Try sliced YOLO inference for tiny defects:** use overlapping tiles on the existing ROI image, map boxes back to full-image coordinates, and fuse them with full-frame predictions; first validate DiVat/DiVatLoiLom and the small-object AP slice. | SAHI reports training-free sliced inference gains on VisDrone/xView and describes tile-to-image coordinate mapping plus optional full-image inference. Those are aerial datasets, so this is a reason to test, not a glass-defect gain estimate. It costs substantially more inference and edge tiles/duplicate boxes can increase false positives. [SAHI paper](https://arxiv.org/abs/2202.06934) |
| 4 | **Fit score calibration on a calibration fold, evaluate on a separate Val fold:** test per-model/per-class monotonic calibration or size-conditioned calibration before fusion, then compare AP and reliability diagnostics. | AP depends on score ranking. Detection confidence can be conditionally biased by box size, and the cited work shows this can harm AP and TTA can interact with the bias. Calibration fitted and evaluated on the same images would create another selection path; use grouped, disjoint Val folds and freeze the fitted transform with hashes. [WACV paper](https://openaccess.thecvf.com/content/WACV2023/html/Gilg_The_Box_Size_Confidence_Bias_Harms_Your_Object_Detector_WACV_2023_paper.html) |
| 5 | **Compare Soft-NMS/IoU-aware decay after fusion for dense classes:** sweep a small, fixed set of decay settings on Val against plain WBF, especially where adjacent defects are at risk of being merged or duplicates crowd the top detections. | Soft-NMS decays overlapping detection scores instead of deleting boxes and is training-free; it changes ranking, not box coordinates, and its reported gains are on other datasets/models. Treat it as an ablation after routing/TTA, not as a guaranteed upgrade. [Soft-NMS paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf) |

## Scoring and validation controls

- **Keep the evaluator fixed and explicit.** COCO bbox AP averages thresholds IoU 0.50 through 0.95; the official COCO API defaults to `maxDets=[1,10,100]`, and AP summary reads its third value. This script caps its predictions at 300 but evaluates with `[1,10,100]`, so detections ranked 101–300 cannot affect reported COCO AP. Keep this if the target is standard COCO AP. If the benchmark's official protocol truly defines 300, implement and report that as a separately named protocol, and compare every baseline under the same setting. Do not silently change `maxDets`. [COCO evaluation documentation](https://cocodataset.org/dataset/detection-eval.htm), [official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- **Control search-selection bias.** The current per-class coordinate search is less expensive than a full Cartesian grid, but selecting the best result after many Val comparisons can still overfit a finite validation set. Predeclare a compact search, select using grouped cross-validation or a Val calibration/selection split, and retain the frozen external Test phase for one final evaluation. Group images by acquisition session/capture family where possible; random image splits can put related views or backgrounds on both sides. [Cawley & Talbot, JMLR](https://www.jmlr.org/beta/papers/v11/cawley10a.html)
- **Match the evaluator's per-class cap end to end.** With standard COCO `useCats=1`, `maxDets=100` is applied per image and category, not globally across all five categories. Thus detections 101–300 matter only if a class has over 100 candidates in one image. Separately, this script globally truncates final per-image predictions to 300, while its Val tuner evaluates one category at a time before that combined cap. Log how often either cap binds; if the global 300 cap binds, verify the complete assembled Val predictions under the same cap used at final evaluation. The official API defaults to `[1,10,100]`; keep it for standard COCO AP, or use a separately named 300-detection protocol only if that is the benchmark's explicit definition. [COCO evaluation documentation](https://cocodataset.org/dataset/detection-eval.htm), [official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- **Keep Val/Test identity checks independent of labels.** Preserve exact-image and near-duplicate audits and data-source provenance. Do not load Test annotations, compute Test AP during tuning, or use Test metrics to choose a branch or stop rule. A frozen artifact should include the script, checkpoint, Val image/annotation, evaluator-protocol, and prediction-stream hashes, plus the full search trace.

## Recommended stop rule

Choose the best method by the predeclared grouped-Val estimate of mAP@50:95, with per-class AP and AP-small as diagnostics. Require a repeatable gain across folds (and no material regression in a target class), then freeze the exact code, checkpoints, split identities, and protocol. If the next predeclared experiment does not improve the grouped-Val estimate beyond fold-to-fold variability, stop searching and run Test once. A larger raw Val maximum alone is not sufficient evidence of a better generalizing ensemble, because optimizing a noisy finite-sample criterion can itself overfit. [Cawley & Talbot, JMLR](https://www.jmlr.org/beta/papers/v11/cawley10a.html)

## Prepared first ablation (real-data results pending)

`kaggle/kernel_wbf_routing_candidate/run_wbf_routing.py` is generated separately
from V6 by `scripts/prepare_wbf_routing_candidate.py`. It first reproduces the
V6 tuning baseline, then compares ten predeclared routes without reoptimizing
their weights. Acceptance requires at least 0.10 percentage points more AP,
positive gains in two deterministic Val subsets with sufficient class support,
and at least 0.10 points more assembled mAP after the same global detection cap.
The subsets share the whole-Val-fitted baseline parameters, so this is a
stability diagnostic, not an independent cross-validation estimate. Acquisition
groups are not inferred without reliable metadata. It does not implement the
stronger grouped-Val recommendation above.

The candidate defaults to Val-only tuning and caches all required streams before
search, with image-ID mappings and image/checkpoint/script provenance. Existing
V6 submissions are unchanged. Local tests verify route exclusion, cache mismatch
rejection, Test-GT rejection and an actual COCOeval improvement on synthetic
detections; these are correctness checks and do not establish glass-dataset gains.

## Primary references

1. Solovyev, Wang, Gabruseva, “Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models,” 2019. https://arxiv.org/abs/1910.13302
2. Gilg et al., “The Box Size Confidence Bias Harms Your Object Detector,” WACV 2023. https://openaccess.thecvf.com/content/WACV2023/html/Gilg_The_Box_Size_Confidence_Bias_Harms_Your_Object_Detector_WACV_2023_paper.html
3. Akyon et al., “Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection,” 2022. https://arxiv.org/abs/2202.06934
4. Bodla et al., “Soft-NMS—Improving Object Detection With One Line of Code,” ICCV 2017. https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf
5. Cawley & Talbot, “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” JMLR 2010. https://www.jmlr.org/beta/papers/v11/cawley10a.html
6. Dave et al., “Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details,” 2021. https://arxiv.org/abs/2102.01066
7. COCO Detection Evaluation and official evaluator. https://cocodataset.org/dataset/detection-eval.htm ; https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
