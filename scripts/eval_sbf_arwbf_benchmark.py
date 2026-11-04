"""Fast, leakage-safe validation benchmark for Baseline WBF, SBF, and AR-WBF."""

import gzip
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]

def compute_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def fast_eval_class(coco_gt: COCO, preds_c: list[dict], c_idx: int) -> float:
    """Evaluate COCO AP@[.50:.95] for a single category silently."""
    if not preds_c:
        return 0.0
    f = io.StringIO()
    with redirect_stdout(f):
        coco_dt = coco_gt.loadRes(preds_c)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [c_idx]
        ev.params.useCats = 1
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return round(float(ev.stats[0] * 100), 4)


def full_coco_eval(coco_gt: COCO, all_preds: list[dict]) -> dict[str, Any]:
    """Evaluate full COCO metric suite across all categories."""
    if not all_preds:
        return {"mAP50_95": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0, "per_class": {}}
    f = io.StringIO()
    with redirect_stdout(f):
        coco_dt = coco_gt.loadRes(all_preds)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        stats = ev.stats
        
        per_class = {}
        for c_idx, c_name in enumerate(CLASS_NAMES):
            ev_c = COCOeval(coco_gt, coco_dt, "bbox")
            ev_c.params.catIds = [c_idx]
            ev_c.params.useCats = 1
            ev_c.evaluate()
            ev_c.accumulate()
            ev_c.summarize()
            per_class[c_name] = round(float(ev_c.stats[0] * 100), 2)
            
    return {
        "mAP50_95": round(float(stats[0] * 100), 2),
        "AP50": round(float(stats[1] * 100), 2),
        "AP75": round(float(stats[2] * 100), 2),
        "AP_small": round(float(stats[3] * 100), 2),
        "per_class": per_class
    }


# ==============================================================================
# FUSION ENGINES
# ==============================================================================

def generate_wbf_class(c_idx: int, streams: dict, cfg: dict) -> list[dict]:
    iou_thr = cfg.get("iou_thr", 0.55)
    weights = cfg.get("weights", [1.0, 1.0, 1.0])
    conf_floor = cfg.get("conf_floor", 0.03)
    consensus_boost = cfg.get("consensus_boost", 1.15)
    solo_decay = cfg.get("solo_decay", 0.95)
    
    preds_c = []
    for img_id, s in streams.items():
        boxes_list = [s['f_tta'][c_idx][0], s['d_tta'][c_idx][0], s['y_ms_tta'][c_idx][0]]
        scores_list = [s['f_tta'][c_idx][1], s['d_tta'][c_idx][1], s['y_ms_tta'][c_idx][1]]
        
        candidates = []
        for m in range(3):
            for b, sc in zip(boxes_list[m], scores_list[m]):
                if sc >= conf_floor:
                    candidates.append((b, float(sc), m))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        clusters = []
        for b, sc, m in candidates:
            best_iou, best_idx = iou_thr, -1
            for idx, cl in enumerate(clusters):
                if any(mem[2] == m for mem in cl['members']):
                    continue
                iou = compute_iou(b, cl['center'])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0:
                clusters[best_idx]['members'].append((b, sc, m))
            else:
                clusters.append({'center': b, 'members': [(b, sc, m)]})
                
        for cl in clusters:
            mems = cl['members']
            cw_sum = sum(mem[1] * weights[mem[2]] for mem in mems)
            box = [
                sum(mem[0][0] * (mem[1] * weights[mem[2]]) for mem in mems) / cw_sum,
                sum(mem[0][1] * (mem[1] * weights[mem[2]]) for mem in mems) / cw_sum,
                sum(mem[0][2] * (mem[1] * weights[mem[2]]) for mem in mems) / cw_sum,
                sum(mem[0][3] * (mem[1] * weights[mem[2]]) for mem in mems) / cw_sum,
            ]
            w_sum = sum(weights[mem[2]] for mem in mems)
            sc_avg = sum(mem[1] * weights[mem[2]] for mem in mems) / w_sum
            num_agree = len({mem[2] for mem in mems})
            score = min(1.0, sc_avg * (consensus_boost if num_agree >= 2 else solo_decay))
            if score >= conf_floor:
                preds_c.append({
                    "image_id": img_id,
                    "category_id": c_idx,
                    "bbox": [box[0], box[1], max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])],
                    "score": score
                })
    return preds_c


def generate_sbf_class(c_idx: int, streams: dict, cfg: dict) -> list[dict]:
    """Selective Box Fusion (SBF - Han et al., 2025).
    
    Implements multi-model voting. Predictions require consensus quorum (min_votes=2);
    single-detector proposals are filtered out unless their confidence exceeds
    solo_conf_thr (selective false-positive suppression).
    """
    iou_thr = cfg.get("iou_thr", 0.55)
    weights = cfg.get("weights", [1.0, 1.0, 1.0])
    min_votes = cfg.get("min_votes", 2)
    solo_conf_thr = cfg.get("solo_conf_thr", 0.35)
    consensus_boost = cfg.get("consensus_boost", 1.15)
    conf_floor = cfg.get("conf_floor", 0.03)
    
    preds_c = []
    for img_id, s in streams.items():
        boxes_list = [s['f_tta'][c_idx][0], s['d_tta'][c_idx][0], s['y_ms_tta'][c_idx][0]]
        scores_list = [s['f_tta'][c_idx][1], s['d_tta'][c_idx][1], s['y_ms_tta'][c_idx][1]]
        
        candidates = []
        for m in range(3):
            for b, sc in zip(boxes_list[m], scores_list[m]):
                if sc >= conf_floor:
                    candidates.append((b, float(sc), m))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        clusters = []
        for b, sc, m in candidates:
            best_iou, best_idx = iou_thr, -1
            for idx, cl in enumerate(clusters):
                if any(mem[2] == m for mem in cl['members']):
                    continue
                iou = compute_iou(b, cl['center'])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0:
                clusters[best_idx]['members'].append((b, sc, m))
            else:
                clusters.append({'center': b, 'members': [(b, sc, m)]})
                
        for cl in clusters:
            mems = cl['members']
            num_votes = len({mem[2] for mem in mems})
            # SBF Selective Decision Rule
            if num_votes < min_votes:
                max_sc = max(mem[1] for mem in mems)
                if max_sc < solo_conf_thr:
                    continue # Discard unconfirmed low-confidence false positive
                score = max_sc * 0.92
                box = mems[0][0]
            else:
                cw_sum = sum(mem[1] * weights[mem[2]] for mem in mems)
                box = [
                    sum(mem[0][0] * (mem[1] * weights[mem[2]]) for mem in mems) / cw_sum,
                    sum(mem[0][1] * (mem[1] * weights[mem[2]]) for mem in mems) / cw_sum,
                    sum(mem[0][2] * (mem[1] * weights[mem[2]]) for mem in mems) / cw_sum,
                    sum(mem[0][3] * (mem[1] * weights[mem[2]]) for mem in mems) / cw_sum,
                ]
                w_sum = sum(weights[mem[2]] for mem in mems)
                sc_avg = sum(mem[1] * weights[mem[2]] for mem in mems) / w_sum
                # Vote agreement bonus
                vote_ratio = (num_votes / 3.0) ** 0.4
                score = min(1.0, sc_avg * consensus_boost * vote_ratio)
                
            if score >= conf_floor:
                preds_c.append({
                    "image_id": img_id,
                    "category_id": c_idx,
                    "bbox": [box[0], box[1], max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])],
                    "score": score
                })
    return preds_c


def generate_ar_wbf_class(c_idx: int, streams: dict, cfg: dict) -> list[dict]:
    """Adaptive Reliability Weighted Boxes Fusion (AR-WBF - Daoud & Alqasir, 2026).
    
    Dynamically adjusts IoU clustering based on model reliability & score discrepancy;
    weights coordinates and confidence using detector validation priors.
    """
    iou_base = cfg.get("iou_base", 0.52)
    reliabilities = cfg.get("reliabilities", [1.0, 1.0, 1.0])
    lambda_adapt = cfg.get("lambda_adapt", 0.20)
    consensus_boost = cfg.get("consensus_boost", 1.15)
    solo_decay = cfg.get("solo_decay", 0.95)
    conf_floor = cfg.get("conf_floor", 0.03)
    
    r = reliabilities
    preds_c = []
    for img_id, s in streams.items():
        boxes_list = [s['f_tta'][c_idx][0], s['d_tta'][c_idx][0], s['y_ms_tta'][c_idx][0]]
        scores_list = [s['f_tta'][c_idx][1], s['d_tta'][c_idx][1], s['y_ms_tta'][c_idx][1]]
        
        candidates = []
        for m in range(3):
            for b, sc in zip(boxes_list[m], scores_list[m]):
                if sc >= conf_floor:
                    candidates.append((b, float(sc), m))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        clusters = []
        for b, sc, m in candidates:
            best_iou, best_idx = -1.0, -1
            for idx, cl in enumerate(clusters):
                if any(mem[2] == m for mem in cl['members']):
                    continue
                cl_top_sc = cl['members'][0][1]
                cl_top_rel = r[cl['members'][0][2]]
                score_diff = abs(sc - cl_top_sc)
                # Adaptive IoU clustering threshold
                adaptive_thr = iou_base * (1.0 - lambda_adapt * score_diff * (1.0 - min(r[m], cl_top_rel)))
                iou = compute_iou(b, cl['center'])
                if iou > adaptive_thr and iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0:
                clusters[best_idx]['members'].append((b, sc, m))
            else:
                clusters.append({'center': b, 'members': [(b, sc, m)]})
                
        for cl in clusters:
            mems = cl['members']
            cw_sum = sum(mem[1] * r[mem[2]] for mem in mems)
            box = [
                sum(mem[0][0] * (mem[1] * r[mem[2]]) for mem in mems) / cw_sum,
                sum(mem[0][1] * (mem[1] * r[mem[2]]) for mem in mems) / cw_sum,
                sum(mem[0][2] * (mem[1] * r[mem[2]]) for mem in mems) / cw_sum,
                sum(mem[0][3] * (mem[1] * r[mem[2]]) for mem in mems) / cw_sum,
            ]
            r_sum = sum(r[mem[2]] for mem in mems)
            sc_avg = sum(mem[1] * r[mem[2]] for mem in mems) / r_sum
            agree_models = {mem[2] for mem in mems}
            
            if len(agree_models) >= 2:
                collective_factor = (sum(r[m] for m in agree_models) / 3.0) ** 0.3
                score = min(1.0, sc_avg * consensus_boost * collective_factor)
            else:
                solo_m = mems[0][2]
                rel_factor = (r[solo_m] / max(r)) ** 0.4
                score = min(1.0, sc_avg * solo_decay * rel_factor)
                
            if score >= conf_floor:
                preds_c.append({
                    "image_id": img_id,
                    "category_id": c_idx,
                    "bbox": [box[0], box[1], max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])],
                    "score": score
                })
    return preds_c


# ==============================================================================
# COORDINATE DESCENT TUNER (Fast & Systematic)
# ==============================================================================

def tune_sbf(coco_gt: COCO, streams: dict) -> dict[str, Any]:
    print("\n--- Tuning SBF (Selective Box Fusion) across 5 classes ---")
    best_configs = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        t0 = time.time()
        # Default start
        best_cfg = {
            "iou_thr": 0.55,
            "weights": [1.0, 1.0, 1.0],
            "min_votes": 2,
            "solo_conf_thr": 0.35,
            "consensus_boost": 1.15,
            "conf_floor": 0.03
        }
        best_score = fast_eval_class(coco_gt, generate_sbf_class(c_idx, streams, best_cfg), c_idx)
        
        # 1. Coordinate descent on IoU threshold
        for iou in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            cand = dict(best_cfg, iou_thr=iou)
            sc = fast_eval_class(coco_gt, generate_sbf_class(c_idx, streams, cand), c_idx)
            if sc > best_score:
                best_cfg, best_score = cand, sc
                
        # 2. Coordinate descent on Solo Confidence Quorum Threshold
        for solo_thr in [0.15, 0.25, 0.35, 0.45]:
            cand = dict(best_cfg, solo_conf_thr=solo_thr)
            sc = fast_eval_class(coco_gt, generate_sbf_class(c_idx, streams, cand), c_idx)
            if sc > best_score:
                best_cfg, best_score = cand, sc
                
        # 3. Coordinate descent on Model Weights
        for w in [[1.5, 1.0, 1.0], [1.0, 1.5, 1.0], [1.0, 1.0, 1.5], [2.0, 1.0, 0.8], [0.8, 1.5, 1.0]]:
            cand = dict(best_cfg, weights=w)
            sc = fast_eval_class(coco_gt, generate_sbf_class(c_idx, streams, cand), c_idx)
            if sc > best_score:
                best_cfg, best_score = cand, sc
                
        # 4. Coordinate descent on Consensus Boost
        for boost in [1.10, 1.15, 1.20, 1.25]:
            cand = dict(best_cfg, consensus_boost=boost)
            sc = fast_eval_class(coco_gt, generate_sbf_class(c_idx, streams, cand), c_idx)
            if sc > best_score:
                best_cfg, best_score = cand, sc
                
        best_cfg["val_ap"] = best_score
        best_configs[c_name] = best_cfg
        print(f"  [{c_name:<11}] SBF Best: AP={best_score:.2f}% | IoU={best_cfg['iou_thr']} | SoloThr={best_cfg['solo_conf_thr']} | w={best_cfg['weights']} | Boost={best_cfg['consensus_boost']} ({time.time()-t0:.1f}s)")
        
    return best_configs


def tune_ar_wbf(coco_gt: COCO, streams: dict) -> dict[str, Any]:
    print("\n--- Tuning AR-WBF (Adaptive Reliability WBF) across 5 classes ---")
    best_configs = {}
    
    # Baseline single-detector validation AP:
    single_ap = {
        "DiVat": [28.31, 26.59, 22.28],
        "DiVatLoiLom": [34.06, 32.84, 30.45],
        "LoiChi": [63.90, 63.76, 62.84],
        "LoiNhua": [48.78, 51.26, 45.76],
        "LoiTray": [41.15, 33.03, 34.16]
    }
    
    for c_idx, c_name in enumerate(CLASS_NAMES):
        t0 = time.time()
        # Normalized reliability vector
        aps = single_ap[c_name]
        tot = sum(aps)
        base_rel = [round(a / tot * 3.0, 3) for a in aps]
        
        best_cfg = {
            "iou_base": 0.52,
            "reliabilities": base_rel,
            "lambda_adapt": 0.20,
            "consensus_boost": 1.15,
            "solo_decay": 0.95,
            "conf_floor": 0.03
        }
        best_score = fast_eval_class(coco_gt, generate_ar_wbf_class(c_idx, streams, best_cfg), c_idx)
        
        # 1. Coordinate descent on Base IoU
        for iou in [0.42, 0.47, 0.52, 0.57, 0.62]:
            cand = dict(best_cfg, iou_base=iou)
            sc = fast_eval_class(coco_gt, generate_ar_wbf_class(c_idx, streams, cand), c_idx)
            if sc > best_score:
                best_cfg, best_score = cand, sc
                
        # 2. Coordinate descent on Lambda Adaptability
        for lam in [0.05, 0.15, 0.25, 0.35]:
            cand = dict(best_cfg, lambda_adapt=lam)
            sc = fast_eval_class(coco_gt, generate_ar_wbf_class(c_idx, streams, cand), c_idx)
            if sc > best_score:
                best_cfg, best_score = cand, sc
                
        # 3. Coordinate descent on Solo Decay
        for decay in [0.90, 0.95, 1.00]:
            cand = dict(best_cfg, solo_decay=decay)
            sc = fast_eval_class(coco_gt, generate_ar_wbf_class(c_idx, streams, cand), c_idx)
            if sc > best_score:
                best_cfg, best_score = cand, sc
                
        # 4. Coordinate descent on Consensus Boost
        for boost in [1.10, 1.15, 1.20, 1.25]:
            cand = dict(best_cfg, consensus_boost=boost)
            sc = fast_eval_class(coco_gt, generate_ar_wbf_class(c_idx, streams, cand), c_idx)
            if sc > best_score:
                best_cfg, best_score = cand, sc
                
        best_cfg["val_ap"] = best_score
        best_configs[c_name] = best_cfg
        print(f"  [{c_name:<11}] AR-WBF Best: AP={best_score:.2f}% | IoU={best_cfg['iou_base']} | Lambda={best_cfg['lambda_adapt']} | Rel={best_cfg['reliabilities']} | Boost={best_cfg['consensus_boost']} ({time.time()-t0:.1f}s)")
        
    return best_configs


def main():
    root = Path(__file__).resolve().parents[1]
    gt_path = root / "kaggle" / "output_wbf_routing_val" / "ensemble_outputs" / "val_ground_truth.json"
    streams_path = root / "kaggle" / "output_wbf_routing_val" / "ensemble_outputs" / "val_streams.json.gz"
    
    print("=" * 80)
    print("STEP 2: BENCHMARK SBF & AR-WBF AGAINST WBF BASELINE ON VALIDATION")
    print("=" * 80)
    print(f"• Loading GT: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"• Loading Streams: {streams_path}")
    with gzip.open(streams_path, "rt", encoding="utf-8") as f:
        streams = {int(k): v for k, v in json.load(f)["streams"].items()}
    print(f"• Loaded {len(streams)} images for validation benchmark.\n")
    
    f = io.StringIO()
    with redirect_stdout(f):
        coco_gt = COCO()
        coco_gt.dataset = gt_data
        coco_gt.createIndex()
        
    # Baseline WBF
    print("1. Evaluating Baseline WBF (Solovyev et al., 2021)...")
    baseline_cfg = {"iou_thr": 0.55, "weights": [1.0, 1.0, 1.0], "conf_floor": 0.03, "consensus_boost": 1.15, "solo_decay": 0.95}
    wbf_preds = []
    for c_idx in range(5):
        wbf_preds.extend(generate_wbf_class(c_idx, streams, baseline_cfg))
    wbf_metrics = full_coco_eval(coco_gt, wbf_preds)
    print(f"   mAP@50:95 = {wbf_metrics['mAP50_95']}% | AP50 = {wbf_metrics['AP50']}% | AP_small = {wbf_metrics['AP_small']}%")
    print(f"   Per-class: {wbf_metrics['per_class']}\n")
    
    # Paper Heterogeneous Triad WBF
    print("2. Evaluating Heterogeneous Triad WBF (Current Paper Configuration)...")
    triad_config_path = root / "results" / "optimal_ensemble_hyperparameters.json"
    triad_metrics = None
    if triad_config_path.is_file():
        from eval_morphology_wbf import generate_class_predictions
        with open(triad_config_path, "r", encoding="utf-8") as f:
            t_cfg = json.load(f)
        triad_preds = []
        for c_idx, c_name in enumerate(CLASS_NAMES):
            preds_c = generate_class_predictions(c_idx, t_cfg[c_name], streams)
            triad_preds.extend(preds_c)
        triad_metrics = full_coco_eval(coco_gt, triad_preds)
        print(f"   mAP@50:95 = {triad_metrics['mAP50_95']}% | AP50 = {triad_metrics['AP50']}% | AP_small = {triad_metrics['AP_small']}%")
        print(f"   Per-class: {triad_metrics['per_class']}\n")
        
    # SBF Tuning
    sbf_best_cfgs = tune_sbf(coco_gt, streams)
    all_sbf_preds = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        all_sbf_preds.extend(generate_sbf_class(c_idx, streams, sbf_best_cfgs[c_name]))
    sbf_metrics = full_coco_eval(coco_gt, all_sbf_preds)
    print("\n>>> Final Assembled SBF Metrics (Validation) <<<")
    print(f"    mAP@50:95 = {sbf_metrics['mAP50_95']}% | AP50 = {sbf_metrics['AP50']}% | AP_small = {sbf_metrics['AP_small']}%")
    print(f"    Per-class: {sbf_metrics['per_class']}\n")
    
    # AR-WBF Tuning
    ar_best_cfgs = tune_ar_wbf(coco_gt, streams)
    all_ar_preds = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        all_ar_preds.extend(generate_ar_wbf_class(c_idx, streams, ar_best_cfgs[c_name]))
    ar_metrics = full_coco_eval(coco_gt, all_ar_preds)
    print("\n>>> Final Assembled AR-WBF Metrics (Validation) <<<")
    print(f"    mAP@50:95 = {ar_metrics['mAP50_95']}% | AP50 = {ar_metrics['AP50']}% | AP_small = {ar_metrics['AP_small']}%")
    print(f"    Per-class: {ar_metrics['per_class']}\n")
    
    # Save Artifacts
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "sbf_optimal_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(sbf_best_cfgs, f, indent=2)
    with open(results_dir / "ar_wbf_optimal_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(ar_best_cfgs, f, indent=2)
        
    benchmark_report = {
        "split": "validation",
        "num_images": len(streams),
        "detectors": ["FBDNet-M (f_tta)", "D-FINE-M (d_tta)", "YOLO26m-ROI (y_ms_tta)"],
        "models_summary": {
            "vanilla_wbf": wbf_metrics,
            "triad_wbf": triad_metrics,
            "sbf_han_2025": sbf_metrics,
            "ar_wbf_daoud_2026": ar_metrics
        },
        "optimal_hyperparameters": {
            "sbf": sbf_best_cfgs,
            "ar_wbf": ar_best_cfgs
        }
    }
    with open(results_dir / "sbf_arwbf_validation_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
        
    print(f"\nArtifacts successfully written to: {results_dir}")


if __name__ == "__main__":
    main()
