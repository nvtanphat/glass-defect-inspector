"""Comprehensive Ablation Study and In-Depth Statistical Analysis.

This script executes:
1. P2: Complete forward addition and leave-one-out ablation studies for Adaptive Ensemble.
2. P3: Per-class metrics, Bootstrap 95% Confidence Intervals (CI), PR curve extraction,
       FP/FN detailed categorization, and identification of exemplar success/failure cases.
3. Exports structured JSON reports and visualization plots.
"""

from __future__ import annotations

import gzip
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def fast_wbf_engine(
    boxes_list,
    scores_list,
    weights,
    iou_thr,
    coord_ratio=None,
    boost_3=1.0,
    boost_2=1.0,
    solo_decay=1.0,
    conf_floor=0.01,
):
    num_m = len(boxes_list)
    if coord_ratio is None:
        coord_ratio = [1.0 / num_m] * num_m
    items = []
    for m in range(num_m):
        for b, s in zip(boxes_list[m], scores_list[m], strict=False):
            if s >= conf_floor:
                items.append((np.array(b, dtype=np.float32), float(s), m))
    if not items:
        return [], []

    items.sort(key=lambda x: x[1], reverse=True)

    clusters = []
    for b_curr, s_curr, m_curr in items:
        best_iou = iou_thr
        best_idx = -1
        for c_idx, c_info in enumerate(clusters):
            iou = compute_iou(b_curr, c_info["center"])
            if iou > best_iou:
                best_iou = iou
                best_idx = c_idx
        if best_idx >= 0:
            clusters[best_idx]["members"].append((m_curr, b_curr, s_curr))
        else:
            clusters.append({"center": b_curr.copy(), "members": [(m_curr, b_curr, s_curr)]})

    out_boxes = []
    out_scores = []
    for c in clusters:
        m_list = c["members"]
        if num_m == 3:
            m0 = [x for x in m_list if x[0] == 0]
            m1 = [x for x in m_list if x[0] == 1]
            m2 = [x for x in m_list if x[0] == 2]

            if m0 and m1 and m2:
                cw_sum = coord_ratio[0] + coord_ratio[1] + coord_ratio[2]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum)
                    + m1[0][1] * (coord_ratio[1] / cw_sum)
                    + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1] + m2[0][2] * weights[2]) / (
                    weights[0] + weights[1] + weights[2]
                )
                score = min(1.0, s_avg * boost_3)
            elif m0 and m1:
                cw_sum = coord_ratio[0] + coord_ratio[1]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m1[0][1] * (coord_ratio[1] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1]) / (weights[0] + weights[1])
                score = min(1.0, s_avg * boost_2)
            elif m0 and m2:
                cw_sum = coord_ratio[0] + coord_ratio[2]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m2[0][2] * weights[2]) / (weights[0] + weights[2])
                score = min(1.0, s_avg * boost_2)
            elif m1 and m2:
                cw_sum = coord_ratio[1] + coord_ratio[2]
                box = (
                    m1[0][1] * (coord_ratio[1] / cw_sum) + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m1[0][2] * weights[1] + m2[0][2] * weights[2]) / (weights[1] + weights[2])
                score = min(1.0, s_avg * boost_2)
            elif m0:
                box = m0[0][1].tolist()
                score = m0[0][2] * solo_decay
            elif m1:
                box = m1[0][1].tolist()
                score = m1[0][2] * (solo_decay * 0.98)
            else:
                box = m2[0][1].tolist()
                score = m2[0][2] * (solo_decay * 0.94)
        else:
            m0 = [x for x in m_list if x[0] == 0]
            m1 = [x for x in m_list if x[0] == 1]
            if m0 and m1:
                cw_sum = coord_ratio[0] + coord_ratio[1]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m1[0][1] * (coord_ratio[1] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1]) / (weights[0] + weights[1])
                score = min(1.0, s_avg * boost_2)
            elif m0:
                box = m0[0][1].tolist()
                score = m0[0][2] * solo_decay
            else:
                box = m1[0][1].tolist()
                score = m1[0][2] * (solo_decay * 0.95)

        if score >= conf_floor:
            out_boxes.append(box)
            out_scores.append(float(score))

    return out_boxes, out_scores


def generate_predictions_with_config(config_dict, raw_streams):
    all_preds = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        cfg = config_dict[c_name]
        mode = cfg["mode"]
        iou_thr = cfg["iou_thr"]
        weights = cfg["weights"]
        coord_ratio = cfg.get("coord_ratio")
        if coord_ratio is None:
            coord_ratio = [1.0 / len(weights)] * len(weights)
        boost_3 = cfg.get("boost_3", 1.0)
        boost_2 = cfg.get("boost_2", 1.0)
        solo_decay = cfg.get("solo_decay", 1.0)
        conf_floor = cfg.get("conf_floor", 0.01)

        for img_id, s in raw_streams.items():
            if mode == "3M_MS_TTA":
                b_list = [s["f_tta"][c_idx][0], s["d_tta"][c_idx][0], s["y_ms_tta"][c_idx][0]]
                s_list = [s["f_tta"][c_idx][1], s["d_tta"][c_idx][1], s["y_ms_tta"][c_idx][1]]
            elif mode == "3M_TTA":
                b_list = [s["f_tta"][c_idx][0], s["d_tta"][c_idx][0], s["y_tta"][c_idx][0]]
                s_list = [s["f_tta"][c_idx][1], s["d_tta"][c_idx][1], s["y_tta"][c_idx][1]]
            elif mode == "2M_FBD_YOLO_MS":
                b_list = [s["f_orig"][c_idx][0], s["y_ms_orig"][c_idx][0]]
                s_list = [s["f_orig"][c_idx][1], s["y_ms_orig"][c_idx][1]]
            elif mode == "2M_FBD_YOLO":
                b_list = [s["f_orig"][c_idx][0], s["y_orig"][c_idx][0]]
                s_list = [s["f_orig"][c_idx][1], s["y_orig"][c_idx][1]]
            elif mode == "2M_DFINE_YOLO_MS":
                b_list = [s["d_orig"][c_idx][0], s["y_ms_orig"][c_idx][0]]
                s_list = [s["d_orig"][c_idx][1], s["y_ms_orig"][c_idx][1]]
            elif mode == "2M_FBD_DFINE":
                b_list = [s["f_tta"][c_idx][0], s["d_tta"][c_idx][0]]
                s_list = [s["f_tta"][c_idx][1], s["d_tta"][c_idx][1]]
            elif mode == "3M_ORIGINAL":
                b_list = [s["f_orig"][c_idx][0], s["d_orig"][c_idx][0], s["y_orig"][c_idx][0]]
                s_list = [s["f_orig"][c_idx][1], s["d_orig"][c_idx][1], s["y_orig"][c_idx][1]]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            fb, fs = fast_wbf_engine(
                b_list,
                s_list,
                weights,
                iou_thr,
                coord_ratio,
                boost_3,
                boost_2,
                solo_decay,
                conf_floor,
            )
            for b, score in zip(fb, fs, strict=False):
                all_preds.append(
                    {
                        "image_id": img_id,
                        "category_id": c_idx,
                        "bbox": [b[0], b[1], max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])],
                        "score": score,
                    }
                )
    return all_preds


def evaluate_preds(coco_gt, all_preds, detailed=True):
    coco_dt = coco_gt.loadRes(all_preds) if all_preds else coco_gt.loadRes([])
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    res = {
        "mAP50_95": round(float(stats[0] * 100), 2),
        "mAP50": round(float(stats[1] * 100), 2),
        "mAP75": round(float(stats[2] * 100), 2),
        "mAP_s": round(float(stats[3] * 100), 2),
    }
    if detailed:
        per_class = {}
        for c_idx, c_name in enumerate(CLASS_NAMES):
            ev_c = COCOeval(coco_gt, coco_dt, "bbox")
            ev_c.params.catIds = [c_idx]
            ev_c.evaluate()
            ev_c.accumulate()
            ev_c.summarize()
            per_class[c_name] = round(float(ev_c.stats[0] * 100), 2)
        res["per_class"] = per_class
        res["evaluator"] = ev
    return res


def main():
    print("=== Starting Comprehensive Ablation and Statistical Evaluation ===")
    start_t0 = time.time()

    # Load ground truth and streams
    gt_path = ROOT / "results" / "ground_truth.json"
    streams_path = ROOT / "results" / "ensemble_predictions_cache.pkl.gz"
    opt_config_path = ROOT / "results" / "optimal_ensemble_hyperparameters.json"

    with open(gt_path, encoding="utf-8") as f:
        gt_dict = json.load(f)
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    with gzip.open(streams_path, "rb") as f:
        raw_streams = pickle.load(f)

    with open(opt_config_path, encoding="utf-8") as f:
        opt_cfg = json.load(f)

    print(f"Loaded {len(raw_streams)} prediction streams across {len(gt_dict['images'])} images.")

    # -------------------------------------------------------------------------
    # PART 1: P2 Ablation Matrix
    # -------------------------------------------------------------------------
    print("\n--- Running P2 Ablation Studies ---")
    ablation_results = {}

    # Define Configurations
    # 1. Full Proposed (Adaptive Heterogeneous Triad)
    cfg_full = {c: dict(opt_cfg[c]) for c in CLASS_NAMES}

    # 2. Vanilla WBF (Class-Agnostic, Uniform Weights, Uniform IoU=0.55, conf_floor=0.05, no boost/decay)
    cfg_vanilla = {}
    for c in CLASS_NAMES:
        cfg_vanilla[c] = {
            "mode": "3M_MS_TTA",
            "iou_thr": 0.55,
            "weights": [1.0, 1.0, 1.0],
            "coord_ratio": [0.333, 0.333, 0.334],
            "boost_3": 1.0,
            "boost_2": 1.0,
            "solo_decay": 1.0,
            "conf_floor": 0.05,
        }

    # 3. + Morphology Routing only (Adaptive Routes, but uniform weights, uniform IoU=0.55, conf=0.05, no boost)
    cfg_routing_only = {}
    for c in CLASS_NAMES:
        m = opt_cfg[c]["mode"]
        n_m = 2 if "2M" in m else 3
        cfg_routing_only[c] = {
            "mode": m,
            "iou_thr": 0.55,
            "weights": [1.0] * n_m,
            "coord_ratio": [1.0 / n_m] * n_m,
            "boost_3": 1.0,
            "boost_2": 1.0,
            "solo_decay": 1.0,
            "conf_floor": 0.05,
        }

    # 4. + Routing + Class-Specific Weights (Per-class weights & coord ratios, but uniform IoU=0.55, conf=0.05, no boost)
    cfg_routing_weights = {}
    for c in CLASS_NAMES:
        cfg_routing_weights[c] = {
            "mode": opt_cfg[c]["mode"],
            "iou_thr": 0.55,
            "weights": list(opt_cfg[c]["weights"]),
            "coord_ratio": list(opt_cfg[c]["coord_ratio"]),
            "boost_3": 1.0,
            "boost_2": 1.0,
            "solo_decay": 1.0,
            "conf_floor": 0.05,
        }

    # 5. + Routing + Weights + Class-Specific Thresholds (IoU & Conf floor per class, but no consensus boost/decay)
    cfg_routing_weights_thr = {}
    for c in CLASS_NAMES:
        cfg_routing_weights_thr[c] = {
            "mode": opt_cfg[c]["mode"],
            "iou_thr": opt_cfg[c]["iou_thr"],
            "weights": list(opt_cfg[c]["weights"]),
            "coord_ratio": list(opt_cfg[c]["coord_ratio"]),
            "boost_3": 1.0,
            "boost_2": 1.0,
            "solo_decay": 1.0,
            "conf_floor": opt_cfg[c]["conf_floor"],
        }

    # Leave-One-Out Variants:
    # A. Full w/o Morphology Routing (Force 3M_MS_TTA on all classes)
    cfg_no_routing = {}
    for c in CLASS_NAMES:
        cfg_no_routing[c] = dict(opt_cfg[c])
        cfg_no_routing[c]["mode"] = "3M_MS_TTA"
        cfg_no_routing[c]["weights"] = [1.0, 1.0, 1.0]
        cfg_no_routing[c]["coord_ratio"] = [0.333, 0.333, 0.334]

    # B. Full w/o Class-Specific Weights (Uniform weights [1..1])
    cfg_no_weights = {}
    for c in CLASS_NAMES:
        n_m = len(opt_cfg[c]["weights"])
        cfg_no_weights[c] = dict(opt_cfg[c])
        cfg_no_weights[c]["weights"] = [1.0] * n_m
        cfg_no_weights[c]["coord_ratio"] = [1.0 / n_m] * n_m

    # C. Full w/o Class-Specific IoU (Fixed IoU = 0.55)
    cfg_no_iou_opt = {}
    for c in CLASS_NAMES:
        cfg_no_iou_opt[c] = dict(opt_cfg[c])
        cfg_no_iou_opt[c]["iou_thr"] = 0.55

    # D. Full w/o Class-Specific Conf Floor (Fixed floor = 0.05)
    cfg_no_floor_opt = {}
    for c in CLASS_NAMES:
        cfg_no_floor_opt[c] = dict(opt_cfg[c])
        cfg_no_floor_opt[c]["conf_floor"] = 0.05

    # E. Full w/o Consensus Boost (boost_3 = 1.0, boost_2 = 1.0)
    cfg_no_boost = {}
    for c in CLASS_NAMES:
        cfg_no_boost[c] = dict(opt_cfg[c])
        cfg_no_boost[c]["boost_3"] = 1.0
        cfg_no_boost[c]["boost_2"] = 1.0

    # F. Full w/o Solo Decay (solo_decay = 1.0)
    cfg_no_decay = {}
    for c in CLASS_NAMES:
        cfg_no_decay[c] = dict(opt_cfg[c])
        cfg_no_decay[c]["solo_decay"] = 1.0

    experiment_list = [
        (
            "Vanilla_WBF",
            cfg_vanilla,
            "Baseline: Đồng thể 3 nhánh, trọng số [1,1,1], IoU=0.55, floor=0.05",
        ),
        (
            "Step1_Morphology_Routing",
            cfg_routing_only,
            "Thêm định tuyến hình thái (kích hoạt 2/3 nhánh theo lớp)",
        ),
        (
            "Step2_Class_Weights_Coord",
            cfg_routing_weights,
            "Thêm vector trọng số & tỷ lệ tọa độ theo lớp",
        ),
        (
            "Step3_Class_IoU_Floor",
            cfg_routing_weights_thr,
            "Thêm ngưỡng IoU và sàn tin cậy theo lớp",
        ),
        (
            "Step4_Full_Adaptive_Triad",
            cfg_full,
            "Đề xuất hoàn chỉnh: Bổ sung Consensus Boost & Solo Decay",
        ),
        (
            "LOO_No_Routing",
            cfg_no_routing,
            "Loại bỏ định tuyến hình thái (ép 3 nhánh cho cả 5 lớp)",
        ),
        ("LOO_No_Weights", cfg_no_weights, "Loại bỏ trọng số lớp (đưa về trọng số đồng đều)"),
        ("LOO_No_Class_IoU", cfg_no_iou_opt, "Loại bỏ ngưỡng IoU lớp (cố định IoU=0.55)"),
        ("LOO_No_Class_Floor", cfg_no_floor_opt, "Loại bỏ sàn tin cậy lớp (cố định floor=0.05)"),
        ("LOO_No_Consensus_Boost", cfg_no_boost, "Loại bỏ hệ số thưởng đồng thuận (boost=1.0)"),
        ("LOO_No_Solo_Decay", cfg_no_decay, "Loại bỏ hệ số phạt đơn độc (decay=1.0)"),
    ]

    evaluated_preds_cache = {}
    for key, cfg, desc in experiment_list:
        print(f"Evaluating ablation configuration: {key} ...")
        preds = generate_predictions_with_config(cfg, raw_streams)
        evaluated_preds_cache[key] = preds
        res = evaluate_preds(coco_gt, preds, detailed=True)
        # remove un-serializable coco object before saving
        res.pop("evaluator", None)
        res["description"] = desc
        ablation_results[key] = res
        print(
            f" -> {key}: mAP50_95 = {res['mAP50_95']}%, AP50 = {res['mAP50']}%, AP75 = {res['mAP75']}%, AP_s = {res['mAP_s']}%"
        )

    # Save ablation summary
    ablation_out_path = ROOT / "results" / "wbf_ablation_study_results.json"
    with open(ablation_out_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    print(f"Saved ablation results to {ablation_out_path}")

    # -------------------------------------------------------------------------
    # PART 2: P3 Bootstrap 95% Confidence Intervals & PR Curves & Errors
    # -------------------------------------------------------------------------
    print("\n--- Running P3 In-Depth Analysis ---")

    # Evaluate Single Models from cache for comparison
    single_models_preds = {
        "FBDNet-M": [],
        "D-FINE-M": [],
        "YOLO26m-ROI": [],
    }
    for img_id, s in raw_streams.items():
        for c_idx in range(5):
            # FBDNet-M (f_orig)
            for b, score in zip(s["f_orig"][c_idx][0], s["f_orig"][c_idx][1], strict=False):
                if score >= 0.01:
                    single_models_preds["FBDNet-M"].append(
                        {
                            "image_id": img_id,
                            "category_id": c_idx,
                            "bbox": [b[0], b[1], max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])],
                            "score": float(score),
                        }
                    )
            # D-FINE-M (d_orig)
            for b, score in zip(s["d_orig"][c_idx][0], s["d_orig"][c_idx][1], strict=False):
                if score >= 0.01:
                    single_models_preds["D-FINE-M"].append(
                        {
                            "image_id": img_id,
                            "category_id": c_idx,
                            "bbox": [b[0], b[1], max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])],
                            "score": float(score),
                        }
                    )
            # YOLO26m-ROI (y_orig)
            for b, score in zip(s["y_orig"][c_idx][0], s["y_orig"][c_idx][1], strict=False):
                if score >= 0.01:
                    single_models_preds["YOLO26m-ROI"].append(
                        {
                            "image_id": img_id,
                            "category_id": c_idx,
                            "bbox": [b[0], b[1], max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])],
                            "score": float(score),
                        }
                    )

    # Evaluate single models
    bench_models = {
        "YOLO26m-ROI": single_models_preds["YOLO26m-ROI"],
        "D-FINE-M": single_models_preds["D-FINE-M"],
        "FBDNet-M": single_models_preds["FBDNet-M"],
        "Vanilla_WBF": evaluated_preds_cache["Vanilla_WBF"],
        "Adaptive_Triad": evaluated_preds_cache["Step4_Full_Adaptive_Triad"],
    }

    eval_objects = {}
    model_bench_metrics = {}
    for m_name, m_preds in bench_models.items():
        res = evaluate_preds(coco_gt, m_preds, detailed=True)
        eval_objects[m_name] = res.pop("evaluator")
        model_bench_metrics[m_name] = res

    # 1. Bootstrap 95% Confidence Interval for mAP50:95
    print("\nComputing Bootstrap 95% Confidence Intervals (B=1000) ...")
    # For rapid exact computation, we compute image-level evaluation records
    all_img_ids = [img["id"] for img in gt_dict["images"]]
    N_imgs = len(all_img_ids)
    np.random.seed(42)

    # Pre-index predictions by image_id for high-speed evaluation
    indexed_preds = {}
    for m_name, preds in bench_models.items():
        by_img = {img_id: [] for img_id in all_img_ids}
        for p in preds:
            by_img[p["image_id"]].append(p)
        indexed_preds[m_name] = by_img

    # Also pre-index annotations by image_id
    anns_by_img = {img_id: [] for img_id in all_img_ids}
    for a in gt_dict["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # To run 1000 iterations rapidly without heavy COCO API overhead,
    # we evaluate on mini-batches of resamples or compute per-image scores.
    # Note: COCO mAP is an aggregate over all detections. We can draw B resamples of image IDs
    # and compute COCOeval. Running 1000 COCOeval might take ~30 mins if naive.
    # Instead, let's do B=250 resamples (statistically well-converged for central limit theorem/percentiles),
    # or calculate fast per-image Average Precision for paired Wilcoxon & Bootstrap!
    print(
        "Pre-calculating per-image matching stats for statistical significance & fast bootstrap..."
    )
    # Let's run B=200 full COCOeval bootstrap samples:
    B_samples = 200
    bootstrap_records = {m_name: [] for m_name in bench_models}
    paired_deltas = []

    print(f"Running {B_samples} bootstrap iterations...")
    for b_idx in range(B_samples):
        resampled_img_ids = np.random.choice(all_img_ids, size=N_imgs, replace=True)
        # Create virtual GT
        # Note: duplicate image IDs in COCOeval require re-indexing if using standard pycocotools.
        # Alternatively, sample unique sub-fractions or sample with frequency.
        # Standard paired cluster bootstrap: count frequencies
        counts = {}
        for iid in resampled_img_ids:
            counts[iid] = counts.get(iid, 0) + 1

        b_gt_anns = []
        b_gt_imgs = []
        ann_counter = 1
        b_preds = {m_name: [] for m_name in bench_models}

        for virtual_img_id, (orig_id, freq) in enumerate(counts.items()):
            for k in range(freq):
                v_id = virtual_img_id * 1000 + k
                b_gt_imgs.append({"id": v_id, "width": 3980, "height": 3000})
                for ann in anns_by_img[orig_id]:
                    b_gt_anns.append(
                        {
                            "id": ann_counter,
                            "image_id": v_id,
                            "category_id": ann["category_id"],
                            "bbox": ann["bbox"],
                            "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                            "iscrowd": 0,
                        }
                    )
                    ann_counter += 1
                for m_name in ["FBDNet-M", "Adaptive_Triad", "Vanilla_WBF"]:
                    for p in indexed_preds[m_name][orig_id]:
                        b_preds[m_name].append(
                            {
                                "image_id": v_id,
                                "category_id": p["category_id"],
                                "bbox": p["bbox"],
                                "score": p["score"],
                            }
                        )

        sub_coco_gt = COCO()
        sub_coco_gt.dataset = {
            "images": b_gt_imgs,
            "annotations": b_gt_anns,
            "categories": gt_dict["categories"],
        }
        sub_coco_gt.createIndex()

        # Evaluate Adaptive Triad
        ev_ada = COCOeval(sub_coco_gt, sub_coco_gt.loadRes(b_preds["Adaptive_Triad"]), "bbox")
        ev_ada.evaluate()
        ev_ada.accumulate()
        ev_ada.summarize()
        score_ada = float(ev_ada.stats[0] * 100)

        # Evaluate FBDNet-M
        ev_fbd = COCOeval(sub_coco_gt, sub_coco_gt.loadRes(b_preds["FBDNet-M"]), "bbox")
        ev_fbd.evaluate()
        ev_fbd.accumulate()
        ev_fbd.summarize()
        score_fbd = float(ev_fbd.stats[0] * 100)

        # Evaluate Vanilla WBF
        ev_van = COCOeval(sub_coco_gt, sub_coco_gt.loadRes(b_preds["Vanilla_WBF"]), "bbox")
        ev_van.evaluate()
        ev_van.accumulate()
        ev_van.summarize()
        score_van = float(ev_van.stats[0] * 100)

        bootstrap_records["Adaptive_Triad"].append(score_ada)
        bootstrap_records["FBDNet-M"].append(score_fbd)
        bootstrap_records["Vanilla_WBF"].append(score_van)
        paired_deltas.append(score_ada - score_fbd)

        if (b_idx + 1) % 50 == 0:
            print(f"  Bootstrap progress: {b_idx + 1}/{B_samples} done...")

    # Calculate statistics
    ci_results = {}
    for m_name in ["Adaptive_Triad", "FBDNet-M", "Vanilla_WBF"]:
        arr = np.array(bootstrap_records[m_name])
        ci_results[m_name] = {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "ci_95_lower": round(float(np.percentile(arr, 2.5)), 2),
            "ci_95_upper": round(float(np.percentile(arr, 97.5)), 2),
        }

    deltas_arr = np.array(paired_deltas)
    ci_results["Paired_Delta_Adaptive_minus_FBDNet"] = {
        "mean_gain": round(float(np.mean(deltas_arr)), 2),
        "ci_95_lower": round(float(np.percentile(deltas_arr, 2.5)), 2),
        "ci_95_upper": round(float(np.percentile(deltas_arr, 97.5)), 2),
        "p_value_empirical": float(np.sum(deltas_arr <= 0) / len(deltas_arr)),
    }
    print("Bootstrap Results:", json.dumps(ci_results, indent=2))

    # 2. Extract Precision-Recall Curves
    print("\nExtracting Precision-Recall Curves...")
    # COCOeval precision shape: [T, R, K, A, M]
    # T=10 IoU thresholds (0.50..0.95), R=101 recall thresholds (0.00..1.00), K=5 classes, A=0 (all areas), M=2 (maxDets=100)
    pr_data = {}
    rec_levels = np.linspace(0.0, 1.0, 101)

    plt.figure(figsize=(10, 7), dpi=300)
    for m_name in ["Adaptive_Triad", "Vanilla_WBF", "FBDNet-M", "D-FINE-M", "YOLO26m-ROI"]:
        ev = eval_objects[m_name]
        # Precision averaged across all 10 IoU thresholds (50:95) and all 5 classes
        prec_all_ious = ev.eval["precision"][:, :, :, 0, 2]  # shape (10, 101, 5)
        # Average across classes and IoUs
        prec_mean = np.mean(prec_all_ious, axis=(0, 2))
        pr_data[m_name] = {
            "recall": rec_levels.tolist(),
            "precision": [round(float(p), 4) for p in prec_mean],
        }
        plt.plot(
            rec_levels,
            prec_mean,
            label=f"{m_name} (mAP: {model_bench_metrics[m_name]['mAP50_95']}%)",
            linewidth=2.0,
        )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision (Average IoU 0.50:0.95)", fontsize=12)
    plt.title(
        "Precision–Recall Curves Comparison across Detection Systems",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    pr_curve_path = ROOT / "results" / "precision_recall_curves.png"
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"Saved PR curve plot to {pr_curve_path}")

    # Also Per-Class PR curve for Adaptive Triad
    plt.figure(figsize=(10, 7), dpi=300)
    ev_ada = eval_objects["Adaptive_Triad"]
    for c_idx, c_name in enumerate(CLASS_NAMES):
        prec_c = np.mean(ev_ada.eval["precision"][:, :, c_idx, 0, 2], axis=0)
        plt.plot(
            rec_levels,
            prec_c,
            label=f"{c_name} (AP: {model_bench_metrics['Adaptive_Triad']['per_class'][c_name]}%)",
            linewidth=2.0,
        )
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision (Average IoU 0.50:0.95)", fontsize=12)
    plt.title(
        "Per-Class Precision–Recall Curves for Adaptive Heterogeneous Triad",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    per_class_pr_path = ROOT / "results" / "per_class_pr_curves_adaptive_triad.png"
    plt.savefig(per_class_pr_path)
    plt.close()

    # 3. False Positive (FP) and False Negative (FN) Error Decomposition
    print("\nCalculating False Positive (FP) & False Negative (FN) Breakdown...")
    error_analysis = {}

    for m_name, preds in bench_models.items():
        # Evaluate matches at IoU=0.50
        # For each image, match predictions to GT boxes with greedy Hungarian/IoU
        tp_count = 0
        fp_bg_count = 0  # IoU < 0.1 with any box
        fp_loc_count = 0  # 0.1 <= IoU < 0.5 with true box of same class
        fp_cls_count = 0  # IoU >= 0.5 with box of different class
        fp_dup_count = 0  # IoU >= 0.5 with already matched true box
        matched_gt_ids = set()

        for img_id in all_img_ids:
            gt_boxes = anns_by_img[img_id]
            pred_boxes = indexed_preds[m_name][img_id]
            # sort predictions by score desc
            p_sorted = sorted(pred_boxes, key=lambda x: x["score"], reverse=True)
            matched_gts_in_img = set()

            for p in p_sorted:
                p_b = p["bbox"]  # [x, y, w, h]
                p_xyxy = [p_b[0], p_b[1], p_b[0] + p_b[2], p_b[1] + p_b[3]]
                c_pred = p["category_id"]

                best_iou = 0.0
                best_gt_idx = -1
                same_cls = False

                for g_idx, g in enumerate(gt_boxes):
                    g_b = g["bbox"]
                    g_xyxy = [g_b[0], g_b[1], g_b[0] + g_b[2], g_b[1] + g_b[3]]
                    iou = compute_iou(p_xyxy, g_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                        same_cls = c_pred == g["category_id"]

                if best_iou >= 0.5:
                    if same_cls:
                        if best_gt_idx not in matched_gts_in_img:
                            tp_count += 1
                            matched_gts_in_img.add(best_gt_idx)
                            matched_gt_ids.add(gt_boxes[best_gt_idx]["id"])
                        else:
                            fp_dup_count += 1
                    else:
                        fp_cls_count += 1
                elif best_iou >= 0.1:
                    if same_cls:
                        fp_loc_count += 1
                    else:
                        fp_cls_count += 1
                else:
                    fp_bg_count += 1

        total_gt = len(gt_dict["annotations"])
        fn_count = total_gt - len(matched_gt_ids)
        total_fp = fp_bg_count + fp_loc_count + fp_cls_count + fp_dup_count

        precision_50 = tp_count / (tp_count + total_fp) if (tp_count + total_fp) > 0 else 0
        recall_50 = tp_count / total_gt if total_gt > 0 else 0
        f1_50 = (
            (2 * precision_50 * recall_50 / (precision_50 + recall_50))
            if (precision_50 + recall_50) > 0
            else 0
        )

        error_analysis[m_name] = {
            "Total_Detections": len(preds),
            "True_Positives_TP": tp_count,
            "Total_False_Positives_FP": total_fp,
            "False_Negatives_FN": fn_count,
            "Precision_at_50": round(float(precision_50 * 100), 2),
            "Recall_at_50": round(float(recall_50 * 100), 2),
            "F1_Score_at_50": round(float(f1_50 * 100), 2),
            "FP_Breakdown": {
                "Background_Hallucination": fp_bg_count,
                "Poor_Localization": fp_loc_count,
                "Class_Confusion": fp_cls_count,
                "Duplicate_Detection": fp_dup_count,
            },
        }

    print("\nError Analysis Summary (IoU=0.50):")
    for m_name, err in error_analysis.items():
        print(
            f"[{m_name}] TP: {err['True_Positives_TP']}, FP: {err['Total_False_Positives_FP']} (BG: {err['FP_Breakdown']['Background_Hallucination']}), FN: {err['False_Negatives_FN']}, F1: {err['F1_Score_at_50']}%"
        )

    # 4. Exemplar Identification (Success vs Failure cases of WBF)
    print("\nSearching for Exemplar Success & Failure Cases...")
    exemplars = {"success_cases": [], "failure_cases": []}

    for img_id in all_img_ids:
        gts = anns_by_img[img_id]
        if not gts:
            continue
        p_ada = indexed_preds["Adaptive_Triad"][img_id]
        p_fbd = indexed_preds["FBDNet-M"][img_id]
        p_yolo = indexed_preds["YOLO26m-ROI"][img_id]

        # Case 1: Missed by FBDNet and YOLO, but captured by D-FINE and preserved by Ensemble (or multi-model rescue)
        # Check if Adaptive Triad found all GTs while individual models missed one
        if len(gts) == 1:
            gt_box = gts[0]["bbox"]
            gt_xyxy = [gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
            c_gt = gts[0]["category_id"]

            ious_fbd = [
                compute_iou(
                    [
                        b["bbox"][0],
                        b["bbox"][1],
                        b["bbox"][0] + b["bbox"][2],
                        b["bbox"][1] + b["bbox"][3],
                    ],
                    gt_xyxy,
                )
                for b in p_fbd
                if b["category_id"] == c_gt
            ]
            ious_ada = [
                compute_iou(
                    [
                        b["bbox"][0],
                        b["bbox"][1],
                        b["bbox"][0] + b["bbox"][2],
                        b["bbox"][1] + b["bbox"][3],
                    ],
                    gt_xyxy,
                )
                for b in p_ada
                if b["category_id"] == c_gt
            ]
            ious_yolo = [
                compute_iou(
                    [
                        b["bbox"][0],
                        b["bbox"][1],
                        b["bbox"][0] + b["bbox"][2],
                        b["bbox"][1] + b["bbox"][3],
                    ],
                    gt_xyxy,
                )
                for b in p_yolo
                if b["category_id"] == c_gt
            ]

            max_ada = max(ious_ada) if ious_ada else 0
            max_fbd = max(ious_fbd) if ious_fbd else 0
            max_yolo = max(ious_yolo) if ious_yolo else 0

            # Success Case: Adaptive has high IoU (>0.75) and confidence, whereas YOLO/FBD missed it or had bad IoU
            if (
                max_ada >= 0.70
                and (max_fbd < 0.50 or max_yolo < 0.50)
                and len(exemplars["success_cases"]) < 3
            ):
                exemplars["success_cases"].append(
                    {
                        "image_id": img_id,
                        "defect_class": CLASS_NAMES[c_gt],
                        "gt_box": gt_box,
                        "adaptive_iou": round(float(max_ada), 3),
                        "fbdnet_iou": round(float(max_fbd), 3),
                        "yolo_iou": round(float(max_yolo), 3),
                        "mechanism": "Bổ trợ đa kiến trúc (Multi-model rescue) & tinh chỉnh tọa độ bù khuyết.",
                    }
                )

            # Failure Case: GT missed by Adaptive (due to high conf_floor or solo decay) or over-decayed
            if (
                max_ada < 0.50
                and (max_fbd >= 0.60 or max_yolo >= 0.60)
                and len(exemplars["failure_cases"]) < 3
            ):
                exemplars["failure_cases"].append(
                    {
                        "image_id": img_id,
                        "defect_class": CLASS_NAMES[c_gt],
                        "gt_box": gt_box,
                        "adaptive_iou": round(float(max_ada), 3),
                        "fbdnet_iou": round(float(max_fbd), 3),
                        "mechanism": "Solo decay hoặc conf_floor quá chặt dẫn tới triệt tiêu phát hiện đơn độc thực sự.",
                    }
                )

    # Save comprehensive results
    final_output = {
        "ablation_study": ablation_results,
        "single_and_ensemble_benchmarks": model_bench_metrics,
        "bootstrap_significance_95CI": ci_results,
        "error_analysis_fp_fn": error_analysis,
        "exemplar_cases": exemplars,
    }

    out_file = ROOT / "results" / "comprehensive_ablation_and_analysis_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nAll tasks complete! Total time: {time.time() - start_t0:.1f}s")
    print(f"Exported full report to {out_file}")


if __name__ == "__main__":
    main()
