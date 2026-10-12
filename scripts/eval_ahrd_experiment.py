"""Evaluation and Comparative Analysis Script for RF-DETR-M + AHRD Experiment.

Evaluates trained checkpoints (or performs architecture profiling) against the canonical
test split (890 images, 1,573 ground-truth defect annotations).

Metrics Computed:
1. Standard COCO Metrics:
   - mAP50, mAP75, mAP50:95
   - AP_small (area < 32^2), AP_medium (32^2 <= area <= 96^2), AP_large (area > 96^2)
2. Per-class Breakdown (DiVat, DiVatLoiLom, LoiChi, LoiNhua, LoiTray):
   - AP50:95, AP50, AP75
   - Precision, Recall, False Negatives (FN), False Positives (FP)
3. Specialized Bounding Box Scale Analysis (after resize to 960x960):
   - min(w, h) < 8 px      (Micro tiny defects)
   - 8 <= min(w, h) < 16 px (Small defects)
   - 16 <= min(w, h) < 32 px (Medium defects)
   - min(w, h) >= 32 px    (Large defects)
   For each bin: GT count, TP count, FN count, Miss rate (%), Recall (%).
4. Operational Hardware Metrics:
   - Inference Latency (ms) & Throughput (FPS)
   - Peak VRAM Consumption (MB/GB)
   - Parameter Count (M) & FLOPs (GFLOPs)

Usage:
    python scripts/eval_ahrd_experiment.py \
        --config results/D-FINE/configs/dfine/custom/rfdetr_m_e1_ahrd.yml \
        --weights weights/dfine_m_best.pth \
        --gt-json results/data_roi/test/test.json \
        --img-dir results/data_roi/test \
        --output-json results/evaluation_ahrd_e1.json
"""

import sys
import types
from importlib.machinery import ModuleSpec
_tf_stub = types.ModuleType("tensorflow")
_tf_stub.__version__ = "stub"
_tf_stub.__spec__ = ModuleSpec("tensorflow", loader=None)
_tf_stub.__path__ = []
_tf_stub.io = types.SimpleNamespace(gfile=types.SimpleNamespace(join=lambda *a: "/".join(str(x) for x in a)))
sys.modules["tensorflow"] = _tf_stub

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Setup D-FINE path
ROOT = Path(__file__).resolve().parent.parent
DFINE_DIR = ROOT / "results" / "D-FINE"
if str(DFINE_DIR) not in sys.path:
    sys.path.insert(0, str(DFINE_DIR))

from src.core import YAMLConfig
# Ensure AHRD modules are registered
import src.zoo.dfine.ahrd  # noqa: F401

CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]

SCALE_BINS = {
    "lt_8px": lambda w, h: min(w, h) < 8.0,
    "8_to_16px": lambda w, h: 8.0 <= min(w, h) < 16.0,
    "16_to_32px": lambda w, h: 16.0 <= min(w, h) < 32.0,
    "gte_32px": lambda w, h: min(w, h) >= 32.0,
}


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area1 = max(0.0, w1) * max(0.0, h1)
    area2 = max(0.0, w2) * max(0.0, h2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def measure_model_complexity(model: torch.nn.Module, input_size=(1, 3, 960, 960), device="cpu") -> dict[str, Any]:
    """Measure total params, trainable params, and FLOPs."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gflops = 0.0
    try:
        from torchprofile import profile_macs
        dummy = torch.randn(*input_size).to(device)
        macs = profile_macs(model, dummy)
        gflops = (macs * 2) / 1e9
    except Exception:
        # Fallback estimation if torchprofile is unavailable
        gflops = -1.0

    return {
        "total_params_M": round(total_params / 1e6, 2),
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "gflops": round(gflops, 2) if gflops > 0 else "N/A",
    }


def measure_latency_and_vram(model: torch.nn.Module, input_size=(1, 3, 960, 960), device="cpu", num_warmup=20, num_runs=50) -> dict[str, Any]:
    """Benchmark inference latency and peak VRAM."""
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    dummy_sizes = torch.tensor([[input_size[2], input_size[3]]], dtype=torch.float32).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy, dummy_sizes)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy, dummy_sizes)
            torch.cuda.synchronize()
        end = time.perf_counter()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy, dummy_sizes)
        end = time.perf_counter()
        peak_vram_mb = 0.0

    total_time = end - start
    avg_latency_ms = (total_time / num_runs) * 1000.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def evaluate_bbox_scale_breakdown(gt_json_path: Path, all_predictions: list[dict[str, Any]], iou_th=0.5, conf_th=0.5) -> dict[str, Any]:
    """Compute detection rate and false negatives binned by min(w, h) in pixels."""
    gt_data = json.loads(gt_json_path.read_text(encoding="utf-8"))
    annotations = gt_data.get("annotations", [])

    # Filter predictions above confidence threshold
    valid_preds = [p for p in all_predictions if p.get("score", 0.0) >= conf_th]

    # Group preds by image_id
    preds_by_img: dict[int, list[dict[str, Any]]] = {}
    for p in valid_preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    scale_results = {
        bin_name: {
            "gt_count": 0,
            "tp_count": 0,
            "fn_count": 0,
            "recall": 0.0,
            "fn_rate": 0.0,
            "by_class": {c: {"gt": 0, "tp": 0, "fn": 0} for c in CLASS_NAMES},
        }
        for bin_name in SCALE_BINS
    }

    # Group GT annotations by image
    gt_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, gts in gt_by_img.items():
        preds = preds_by_img.get(img_id, [])
        used_preds = set()

        for gt in gts:
            c_id = gt["category_id"]
            c_name = CLASS_NAMES[c_id] if c_id < len(CLASS_NAMES) else f"cls_{c_id}"
            w, h = gt["bbox"][2], gt["bbox"][3]

            # Determine scale bin
            matched_bin = None
            for bin_name, bin_fn in SCALE_BINS.items():
                if bin_fn(w, h):
                    matched_bin = bin_name
                    break

            if matched_bin is None:
                continue

            scale_results[matched_bin]["gt_count"] += 1
            scale_results[matched_bin]["by_class"][c_name]["gt"] += 1

            # Match with prediction
            matched = False
            best_iou = 0.0
            best_pred_idx = -1
            for pi, p in enumerate(preds):
                if pi in used_preds:
                    continue
                if p["category_id"] == c_id:
                    iou = compute_iou(gt["bbox"], p["bbox"])
                    if iou >= iou_th and iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pi

            if best_pred_idx >= 0:
                used_preds.add(best_pred_idx)
                scale_results[matched_bin]["tp_count"] += 1
                scale_results[matched_bin]["by_class"][c_name]["tp"] += 1
            else:
                scale_results[matched_bin]["fn_count"] += 1
                scale_results[matched_bin]["by_class"][c_name]["fn"] += 1

    # Calculate percentages
    for b_name, b_data in scale_results.items():
        gt_c = b_data["gt_count"]
        tp_c = b_data["tp_count"]
        fn_c = b_data["fn_count"]
        b_data["recall"] = round((tp_c / gt_c) * 100, 2) if gt_c > 0 else 0.0
        b_data["fn_rate"] = round((fn_c / gt_c) * 100, 2) if gt_c > 0 else 0.0

    return scale_results


def evaluate_per_class_fn_fp(gt_json_path: Path, all_predictions: list[dict[str, Any]], iou_th=0.5, conf_th=0.5) -> dict[str, Any]:
    """Compute per-class Precision, Recall, False Negatives, False Positives."""
    gt_data = json.loads(gt_json_path.read_text(encoding="utf-8"))
    annotations = gt_data.get("annotations", [])

    valid_preds = [p for p in all_predictions if p.get("score", 0.0) >= conf_th]

    gt_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)

    preds_by_img: dict[int, list[dict[str, Any]]] = {}
    for p in valid_preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    class_stats = {
        c: {"gt": 0, "tp": 0, "fn": 0, "fp": 0, "precision": 0.0, "recall": 0.0}
        for c in CLASS_NAMES
    }

    all_img_ids = set(gt_by_img.keys()).union(set(preds_by_img.keys()))

    for img_id in all_img_ids:
        img_gts = gt_by_img.get(img_id, [])
        img_preds = preds_by_img.get(img_id, [])

        unmatched_gt = set(range(len(img_gts)))
        unmatched_pred = set(range(len(img_preds)))

        # Match IoU
        matches = []
        for gi, gt in enumerate(img_gts):
            c_gt = gt["category_id"]
            for pi, p in enumerate(img_preds):
                if p["category_id"] == c_gt:
                    iou = compute_iou(gt["bbox"], p["bbox"])
                    if iou >= iou_th:
                        matches.append((iou, gi, pi, c_gt))

        matches.sort(reverse=True, key=lambda x: x[0])
        for iou, gi, pi, c_id in matches:
            if gi in unmatched_gt and pi in unmatched_pred:
                unmatched_gt.remove(gi)
                unmatched_pred.remove(pi)
                c_name = CLASS_NAMES[c_id]
                class_stats[c_name]["tp"] += 1

        for gi in unmatched_gt:
            c_name = CLASS_NAMES[img_gts[gi]["category_id"]]
            class_stats[c_name]["fn"] += 1

        for pi in unmatched_pred:
            c_name = CLASS_NAMES[img_preds[pi]["category_id"]]
            class_stats[c_name]["fp"] += 1

    for c in CLASS_NAMES:
        gt_c = class_stats[c]["tp"] + class_stats[c]["fn"]
        pred_c = class_stats[c]["tp"] + class_stats[c]["fp"]
        class_stats[c]["gt"] = gt_c
        class_stats[c]["recall"] = round((class_stats[c]["tp"] / gt_c) * 100, 2) if gt_c > 0 else 0.0
        class_stats[c]["precision"] = round((class_stats[c]["tp"] / pred_c) * 100, 2) if pred_c > 0 else 0.0

    return class_stats


def run_coco_eval(gt_json_path: Path, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Run pycocotools COCOeval."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_json_path))
    if not predictions:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "AP_s": 0.0, "AP_m": 0.0, "AP_l": 0.0, "per_class": {}}

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    mAP50_95 = round(float(stats[0]) * 100, 2)
    mAP50 = round(float(stats[1]) * 100, 2)
    mAP75 = round(float(stats[2]) * 100, 2)
    AP_s = round(float(stats[3]) * 100, 2)
    AP_m = round(float(stats[4]) * 100, 2)
    AP_l = round(float(stats[5]) * 100, 2)

    # Per class AP
    per_class = {}
    for cat_id, cat_name in enumerate(CLASS_NAMES):
        evaluator.params.catIds = [cat_id]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        cat_stats = evaluator.stats
        per_class[cat_name] = {
            "mAP50_95": round(float(cat_stats[0]) * 100, 2) if cat_stats[0] >= 0 else 0.0,
            "mAP50": round(float(cat_stats[1]) * 100, 2) if cat_stats[1] >= 0 else 0.0,
            "mAP75": round(float(cat_stats[2]) * 100, 2) if cat_stats[2] >= 0 else 0.0,
        }

    return {
        "mAP50_95": mAP50_95,
        "mAP50": mAP50,
        "mAP75": mAP75,
        "AP_s": AP_s,
        "AP_m": AP_m,
        "AP_l": AP_l,
        "per_class": per_class,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR-M / AHRD Experiment")
    parser.add_argument("-c", "--config", type=str, required=True, help="YAML config path")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Model weights checkpoint (.pth)")
    parser.add_argument("--gt-json", type=str, default="results/data_roi/test/test.json", help="Path to test.json ground truth")
    parser.add_argument("--img-dir", type=str, default="results/data_roi/test", help="Path to test ROI images directory")
    parser.add_argument("-o", "--output-json", type=str, default="results/eval_ahrd_report.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf-th", type=float, default=0.5)
    parser.add_argument("--iou-th", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("[*] RF-DETR-M + AHRD SCIENTIFIC EXPERIMENT EVALUATION")
    print("=" * 70)
    print(f"Config:      {args.config}")
    print(f"Weights:     {args.weights}")
    print(f"Ground Truth:{args.gt_json}")
    print(f"Device:      {args.device}")

    gt_path = Path(args.gt_json).resolve()
    img_dir = Path(args.img_dir).resolve()
    assert gt_path.exists(), f"GT file not found: {gt_path}"
    assert img_dir.exists(), f"Image directory not found: {img_dir}"

    cfg = YAMLConfig(args.config)
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))

    # Load state dict
    cfg.model.load_state_dict(state, strict=False)

    class FullModel(torch.nn.Module):
        def __init__(self, m, p):
            super().__init__()
            self.model = m.deploy()
            self.postprocessor = p.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    full_model = FullModel(cfg.model, cfg.postprocessor).to(args.device)
    full_model.eval()

    # Hardware profiling
    print("\n[*] Measuring Model Complexity & Latency...")
    complexity = measure_model_complexity(full_model.model, device=args.device)
    perf = measure_latency_and_vram(full_model, device=args.device)
    print(f"  Params:  {complexity['total_params_M']} M (Trainable: {complexity['trainable_params_M']} M)")
    print(f"  Latency: {perf['latency_ms']} ms ({perf['fps']} FPS)")
    print(f"  VRAM:    {perf['peak_vram_mb']} MB")

    # Run inference over test images
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    images_info = gt_data.get("images", [])
    print(f"\n[*] Running inference on {len(images_info)} test images...")

    tf = T.Compose([T.Resize((960, 960)), T.ToTensor()])
    predictions = []

    for i in range(0, len(images_info), args.batch_size):
        batch = images_info[i : i + args.batch_size]
        tensors, sizes = [], []
        for info in batch:
            im_path = img_dir / info["file_name"]
            im = Image.open(im_path).convert("RGB")
            tensors.append(tf(im))
            sizes.append([im.width, im.height])

        batch_tensor = torch.stack(tensors).to(args.device)
        batch_sizes = torch.tensor(sizes, dtype=torch.float32).to(args.device)

        with torch.no_grad():
            batch_labels, batch_boxes, batch_scores = full_model(batch_tensor, batch_sizes)

        for j, info in enumerate(batch):
            boxes = batch_boxes[j].cpu().numpy()  # [x1, y1, x2, y2]
            scores = batch_scores[j].cpu().numpy()
            labels = batch_labels[j].cpu().numpy()

            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b
                predictions.append({
                    "image_id": info["id"],
                    "category_id": int(l),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                })

    # 1. COCO metrics
    print("\n[+] Computing Standard COCO Metrics...")
    coco_metrics = run_coco_eval(gt_path, predictions)
    print(f"  mAP@50:95: {coco_metrics['mAP50_95']}% | mAP@50: {coco_metrics['mAP50']}% | mAP@75: {coco_metrics['mAP75']}%")
    print(f"  AP_small:  {coco_metrics['AP_s']}% | AP_med: {coco_metrics['AP_m']}% | AP_large: {coco_metrics['AP_l']}%")

    # 2. Per-class FN/FP & Precision/Recall
    print("\n[+] Computing Per-Class Detection & False Negatives...")
    class_stats = evaluate_per_class_fn_fp(gt_path, predictions, iou_th=args.iou_th, conf_th=args.conf_th)
    for c in CLASS_NAMES:
        cs = class_stats[c]
        print(f"  - {c:<12}: P = {cs['precision']:>5.1f}% | R = {cs['recall']:>5.1f}% | FN = {cs['fn']:>4} | FP = {cs['fp']:>4}")

    # 3. Bounding box scale breakdown (after resize)
    print("\n[+] Computing Bounding Box Scale Breakdown (min(w, h) in pixels)...")
    scale_stats = evaluate_bbox_scale_breakdown(gt_path, predictions, iou_th=args.iou_th, conf_th=args.conf_th)
    for bin_name, bdata in scale_stats.items():
        print(f"  - {bin_name:<11}: GT = {bdata['gt_count']:>4} | Recall = {bdata['recall']:>5.1f}% | FN = {bdata['fn_count']:>4} (Miss Rate: {bdata['fn_rate']:>5.1f}%)")

    # Save output report
    report = {
        "metadata": {
            "config": args.config,
            "weights": args.weights,
            "device": args.device,
            "num_test_images": len(images_info),
        },
        "complexity": complexity,
        "performance": perf,
        "coco_metrics": coco_metrics,
        "per_class_stats": class_stats,
        "scale_breakdown": scale_stats,
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[OK] Full evaluation report saved to: {out_file}")


if __name__ == "__main__":
    main()
