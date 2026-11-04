"""Evaluate frozen Validation hyperparameters on the Test Set (890 images).

Strict leakage-safe test benchmark:
- Baseline WBF (Solovyev et al., 2021)
- Heterogeneous Triad WBF (Current Paper)
- SBF (Han et al., 2025) [frozen hyperparameters from Validation]
- AR-WBF (Daoud & Alqasir, 2026) [frozen hyperparameters from Validation]
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import pickle
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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


def full_coco_eval(coco_gt: COCO, all_preds: list[dict]) -> dict[str, Any]:
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
            if num_votes < min_votes:
                max_sc = max(mem[1] for mem in mems)
                if max_sc < solo_conf_thr:
                    continue
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


def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    
    # 1. Load Ground Truth & Predictions Cache
    gt_path = results_dir / "ground_truth.json"
    cache_path = results_dir / "ensemble_predictions_cache.pkl.gz"
    
    print("=" * 80)
    print("STEP 2 (PHASE 2): EVALUATE FROZEN SBF & AR-WBF HYPERPARAMETERS ON TEST SET")
    print("=" * 80)
    print(f"• Test GT     : {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    print(f"• Test Streams: {cache_path}")
    with gzip.open(cache_path, "rb") as f:
        streams = pickle.load(f)
    print(f"• Successfully loaded {len(streams)} test images.\n")
    
    coco_gt = COCO()
    f = io.StringIO()
    with redirect_stdout(f):
        coco_gt.dataset = gt_data
        coco_gt.createIndex()
        
    # Load Validation Tuning Reports
    val_report_path = results_dir / "sbf_arwbf_validation_benchmark_report.json"
    val_summary = {}
    if val_report_path.is_file():
        with open(val_report_path, "r", encoding="utf-8") as f:
            val_summary = json.load(f)
            
    # 1. Baseline WBF
    print("1. Evaluating Baseline WBF (Solovyev et al., 2021) on Test...")
    baseline_cfg = {"iou_thr": 0.55, "weights": [1.0, 1.0, 1.0], "conf_floor": 0.03, "consensus_boost": 1.15, "solo_decay": 0.95}
    wbf_test_preds = []
    for c_idx in range(5):
        wbf_test_preds.extend(generate_wbf_class(c_idx, streams, baseline_cfg))
    wbf_test_metrics = full_coco_eval(coco_gt, wbf_test_preds)
    print(f"   mAP@50:95 = {wbf_test_metrics['mAP50_95']}% | AP50 = {wbf_test_metrics['AP50']}% | AP_small = {wbf_test_metrics['AP_small']}%")
    print(f"   Per-class : {wbf_test_metrics['per_class']}\n")

    # 2. Heterogeneous Triad WBF (Current Paper)
    print("2. Evaluating Heterogeneous Triad WBF (Current Paper) on Test...")
    from eval_morphology_wbf import generate_class_predictions
    triad_config_path = results_dir / "optimal_ensemble_hyperparameters.json"
    with open(triad_config_path, "r", encoding="utf-8") as f:
        triad_cfg = json.load(f)
    triad_test_preds = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        triad_test_preds.extend(generate_class_predictions(c_idx, triad_cfg[c_name], streams))
    triad_test_metrics = full_coco_eval(coco_gt, triad_test_preds)
    print(f"   mAP@50:95 = {triad_test_metrics['mAP50_95']}% | AP50 = {triad_test_metrics['AP50']}% | AP_small = {triad_test_metrics['AP_small']}%")
    print(f"   Per-class : {triad_test_metrics['per_class']}\n")

    # 3. SBF (Frozen hyperparameters from Validation)
    print("3. Evaluating SBF (Han et al., 2025) on Test...")
    sbf_cfgs_path = results_dir / "sbf_optimal_hyperparameters.json"
    with open(sbf_cfgs_path, "r", encoding="utf-8") as f:
        sbf_cfgs = json.load(f)
    sbf_test_preds = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        sbf_test_preds.extend(generate_sbf_class(c_idx, streams, sbf_cfgs[c_name]))
    sbf_test_metrics = full_coco_eval(coco_gt, sbf_test_preds)
    print(f"   mAP@50:95 = {sbf_test_metrics['mAP50_95']}% | AP50 = {sbf_test_metrics['AP50']}% | AP_small = {sbf_test_metrics['AP_small']}%")
    print(f"   Per-class : {sbf_test_metrics['per_class']}\n")

    # 4. AR-WBF (Frozen hyperparameters from Validation)
    print("4. Evaluating AR-WBF (Daoud & Alqasir, 2026) on Test...")
    ar_cfgs_path = results_dir / "ar_wbf_optimal_hyperparameters.json"
    with open(ar_cfgs_path, "r", encoding="utf-8") as f:
        ar_cfgs = json.load(f)
    ar_test_preds = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        ar_test_preds.extend(generate_ar_wbf_class(c_idx, streams, ar_cfgs[c_name]))
    ar_test_metrics = full_coco_eval(coco_gt, ar_test_preds)
    print(f"   mAP@50:95 = {ar_test_metrics['mAP50_95']}% | AP50 = {ar_test_metrics['AP50']}% | AP_small = {ar_test_metrics['AP_small']}%")
    print(f"   Per-class : {ar_test_metrics['per_class']}\n")

    # Combined Val -> Test Benchmark Report
    val_models = val_summary.get("models_summary", {})
    benchmark_report = {
        "validation_num_images": 878,
        "test_num_images": len(streams),
        "protocol": "Tuned exclusively on Validation; Frozen hyperparameters evaluated on Test",
        "validation_results": val_models,
        "test_results": {
            "Baseline WBF (Solovyev 2021)": wbf_test_metrics,
            "Heterogeneous Triad WBF (Ours)": triad_test_metrics,
            "SBF (Han et al. 2025)": sbf_test_metrics,
            "AR-WBF (Daoud & Alqasir 2026)": ar_test_metrics
        }
    }
    with open(results_dir / "val_tune_test_eval_fusion_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    # Master Markdown Summary Table
    md_table = """
# Comprehensive Validation Tuning & Test Evaluation Benchmark

| Fusion Strategy | Reference | Split | mAP@50:95 (%) | AP50 (%) | AP75 (%) | AP_small (%) | DiVat | DiVatLoiLom | LoiChi | LoiNhua | LoiTray |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    methods = [
        ("Baseline WBF", "Solovyev et al., 2021", "vanilla_wbf", wbf_test_metrics),
        ("Heterogeneous Triad WBF", "Current Paper", "triad_wbf", triad_test_metrics),
        ("SBF (Selective Box Fusion)", "Han et al., 2025 [16]", "sbf_han_2025", sbf_test_metrics),
        ("AR-WBF (Adaptive Reliability)", "Daoud & Alqasir, 2026 [17]", "ar_wbf_daoud_2026", ar_test_metrics)
    ]
    
    for name, ref, v_key, t_m in methods:
        v_m = val_models.get(v_key, {})
        if v_m:
            md_table += f"| {name} | {ref} | **Val** | **{v_m.get('mAP50_95', '-')}** | {v_m.get('AP50', '-')} | {v_m.get('AP75', '-')} | {v_m.get('AP_small', '-')} | {v_m.get('per_class', {}).get('DiVat', '-')} | {v_m.get('per_class', {}).get('DiVatLoiLom', '-')} | {v_m.get('per_class', {}).get('LoiChi', '-')} | {v_m.get('per_class', {}).get('LoiNhua', '-')} | {v_m.get('per_class', {}).get('LoiTray', '-')} |\n"
        md_table += f"| {name} | {ref} | **Test** | **{t_m['mAP50_95']}** | {t_m['AP50']} | {t_m['AP75']} | {t_m['AP_small']} | {t_m['per_class']['DiVat']} | {t_m['per_class']['DiVatLoiLom']} | {t_m['per_class']['LoiChi']} | {t_m['per_class']['LoiNhua']} | {t_m['per_class']['LoiTray']} |\n"

    print("\n" + md_table)
    with open(results_dir / "val_tune_test_eval_table.md", "w", encoding="utf-8") as f:
        f.write(md_table)
        
    print(f"Results successfully saved to {results_dir}")


if __name__ == "__main__":
    main()
