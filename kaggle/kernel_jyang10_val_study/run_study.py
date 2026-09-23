"""Kaggle script-kernel entrypoint for Comprehensive SAHI Ablation & Validation Study.

Implements the user's Phase 1, 2, 3, and 5 experiment plan:
1. Validates dataset, split, and geometric ROI crop.
2. Evaluates the baseline YOLO26m (Full ROI).
3. Evaluates SAHI 640x640 vs 960x960 slices on existing checkpoint.
4. Evaluates SAHI Multi-scale (Full image + Slices merged).
5. Produces end-to-end comparative trade-off matrix: mAP50:95, per-class AP, Recall, and Latency (FPS).

Completely self-contained evaluation to eliminate any dataset version caching issues.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.ops as ops
from ultralytics import YOLO

WORKING = Path("/kaggle/working")
ARTIFACTS = WORKING / "artifacts"
REPO_COPY = WORKING / "repo"


def log(msg: str) -> None:
    print(f"[sahi-study] {msg}", flush=True)


def calculate_slice_coords(img_w: int, img_h: int, slice_w: int = 960, slice_h: int = 960, overlap_ratio: float = 0.20) -> list[tuple[int, int, int, int]]:
    step_x = max(1, int(slice_w * (1.0 - overlap_ratio)))
    step_y = max(1, int(slice_h * (1.0 - overlap_ratio)))

    x_starts = []
    x = 0
    while x + slice_w <= img_w:
        x_starts.append(x)
        x += step_x
    if not x_starts or x_starts[-1] + slice_w < img_w:
        x_starts.append(max(0, img_w - slice_w))
    x_starts = sorted(list(set(x_starts)))

    y_starts = []
    y = 0
    while y + slice_h <= img_h:
        y_starts.append(y)
        y += step_y
    if not y_starts or y_starts[-1] + slice_h < img_h:
        y_starts.append(max(0, img_h - slice_h))
    y_starts = sorted(list(set(y_starts)))

    slices = []
    for sy in y_starts:
        for sx in x_starts:
            slices.append((sx, sy, min(img_w, sx + slice_w), min(img_h, sy + slice_h)))
    return slices


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


class SAHIPredictorSelfContained:
    def __init__(
        self,
        weights: str | Path,
        slice_w: int = 960,
        slice_h: int = 960,
        overlap_ratio: float = 0.20,
        conf_thresh: float = 0.05,
        iou_thresh: float = 0.50,
        device: int | str = 0,
        include_full_image: bool = False,
    ):
        self.model = YOLO(str(weights))
        self.slice_w = slice_w
        self.slice_h = slice_h
        self.overlap_ratio = overlap_ratio
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.include_full_image = include_full_image

    def predict_image(self, img: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        img_h, img_w = img.shape[:2]

        t0 = time.perf_counter()
        slice_coords = calculate_slice_coords(img_w, img_h, self.slice_w, self.slice_h, self.overlap_ratio)
        tiles = [img[sy:ey, sx:ex] for sx, sy, ex, ey in slice_coords]
        t_slice = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        results = self.model.predict(
            tiles,
            imgsz=self.slice_w,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
        )
        t_infer = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        shifted_boxes = []
        shifted_scores = []
        shifted_classes = []

        for (sx, sy, _, _), res in zip(slice_coords, results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            b = res.boxes.xyxy.cpu().numpy()
            s = res.boxes.conf.cpu().numpy()
            c = res.boxes.cls.cpu().numpy()

            b[:, [0, 2]] += sx
            b[:, [1, 3]] += sy

            shifted_boxes.append(b)
            shifted_scores.append(s)
            shifted_classes.append(c)

        if self.include_full_image:
            full_res = self.model.predict(
                img,
                imgsz=self.slice_w,
                conf=self.conf_thresh,
                device=self.device,
                verbose=False,
            )
            if len(full_res) and full_res[0].boxes is not None and len(full_res[0].boxes) > 0:
                shifted_boxes.append(full_res[0].boxes.xyxy.cpu().numpy())
                shifted_scores.append(full_res[0].boxes.conf.cpu().numpy())
                shifted_classes.append(full_res[0].boxes.cls.cpu().numpy())

        if not shifted_boxes:
            t_merge = (time.perf_counter() - t2) * 1000.0
            return np.empty((0, 6), dtype=np.float32), {
                "slice_ms": t_slice,
                "inference_ms": t_infer,
                "merge_ms": t_merge,
                "total_e2e_ms": t_slice + t_infer + t_merge,
            }

        all_boxes = np.concatenate(shifted_boxes, axis=0)
        all_scores = np.concatenate(shifted_scores, axis=0)
        all_classes = np.concatenate(shifted_classes, axis=0)

        t_boxes = torch.from_numpy(all_boxes).float()
        t_scores = torch.from_numpy(all_scores).float()
        t_cls = torch.from_numpy(all_classes).long()

        keep_idx = ops.batched_nms(t_boxes, t_scores, t_cls, self.iou_thresh).numpy()
        merged = np.hstack([
            all_boxes[keep_idx],
            all_scores[keep_idx, None],
            all_classes[keep_idx, None],
        ])
        t_merge = (time.perf_counter() - t2) * 1000.0

        return merged, {
            "slice_ms": t_slice,
            "inference_ms": t_infer,
            "merge_ms": t_merge,
            "total_e2e_ms": t_slice + t_infer + t_merge,
        }


def run_standalone_eval(
    weights: str | Path,
    test_images_dir: Path,
    test_labels_dir: Path,
    output_json: Path,
    slice_w: int = 960,
    slice_h: int = 960,
    overlap_ratio: float = 0.20,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.50,
    device: int | str = 0,
    include_full_image: bool = False,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    if class_names is None:
        class_names = {0: "DiVat", 1: "DiVatLoiLom", 2: "LoiChi", 3: "LoiNhua", 4: "LoiTray"}
    num_classes = len(class_names)

    predictor = SAHIPredictorSelfContained(
        weights=weights,
        slice_w=slice_w,
        slice_h=slice_h,
        overlap_ratio=overlap_ratio,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        device=device,
        include_full_image=include_full_image,
    )

    image_files = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")))
    log(f"Evaluating {len(image_files)} test images (slice={slice_w}x{slice_h}, full_image={include_full_image})...")

    all_preds_by_img = []
    all_gts_by_img = []
    e2e_times = []
    slice_times = []
    infer_times = []
    merge_times = []

    for idx, img_p in enumerate(image_files):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        preds, timings = predictor.predict_image(img)
        all_preds_by_img.append(preds)
        e2e_times.append(timings["total_e2e_ms"])
        slice_times.append(timings["slice_ms"])
        infer_times.append(timings["inference_ms"])
        merge_times.append(timings["merge_ms"])

        lbl_p = test_labels_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                    gt_boxes.append([x1, y1, x2, y2, c])
        all_gts_by_img.append(np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 5), dtype=np.float32))

        if (idx + 1) % 250 == 0 or (idx + 1) == len(image_files):
            log(f"  [{idx+1}/{len(image_files)}] mean latency: {np.mean(e2e_times):.1f} ms")

    # Global COCO 101-point evaluation
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    per_class_ap = {c: [] for c in range(num_classes)}
    recalls_50 = []

    for c in range(num_classes):
        c_gts_per_img = {}
        c_gts_count = 0
        for img_idx, img_gts in enumerate(all_gts_by_img):
            if len(img_gts):
                g_mask = img_gts[:, 4] == c
                cur_g = img_gts[g_mask, :4]
                c_gts_per_img[img_idx] = cur_g
                c_gts_count += len(cur_g)
            else:
                c_gts_per_img[img_idx] = np.empty((0, 4), dtype=np.float32)

        if c_gts_count == 0:
            per_class_ap[c] = [0.0] * len(iou_thresholds)
            continue

        detections = []
        for img_idx, img_preds in enumerate(all_preds_by_img):
            if len(img_preds):
                p_mask = img_preds[:, 5] == c
                for p in img_preds[p_mask]:
                    detections.append((img_idx, float(p[4]), p[:4]))

        if len(detections) == 0:
            per_class_ap[c] = [0.0] * len(iou_thresholds)
            continue

        detections.sort(key=lambda d: d[1], reverse=True)

        class_ap_thresholds = []
        for iou_th in iou_thresholds:
            tp = np.zeros(len(detections), dtype=np.float32)
            fp = np.zeros(len(detections), dtype=np.float32)

            matched = {img_idx: np.zeros(len(c_gts_per_img[img_idx]), dtype=bool) for img_idx in c_gts_per_img}

            for det_idx, (img_idx, conf, box) in enumerate(detections):
                cur_g = c_gts_per_img[img_idx]
                if len(cur_g) == 0:
                    fp[det_idx] = 1.0
                    continue

                ious = box_iou(box[None, :], cur_g)[0]
                best_iou = -1.0
                best_gt = -1
                for g_idx in range(len(cur_g)):
                    if not matched[img_idx][g_idx] and ious[g_idx] > best_iou:
                        best_iou = ious[g_idx]
                        best_gt = g_idx

                if best_iou >= iou_th:
                    tp[det_idx] = 1.0
                    matched[img_idx][best_gt] = True
                else:
                    fp[det_idx] = 1.0

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            rec = tp_cumsum / float(c_gts_count)
            prec = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)

            recall_levels = np.linspace(0.0, 1.0, 101)
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])

            interpolated_prec = np.zeros(101, dtype=np.float32)
            for idx_r, r in enumerate(recall_levels):
                p_at_r = mpre[mrec >= r]
                interpolated_prec[idx_r] = p_at_r[0] if len(p_at_r) else 0.0

            ap = float(np.mean(interpolated_prec))
            class_ap_thresholds.append(ap)

            if np.isclose(iou_th, 0.5):
                recalls_50.append(float(rec[-1]) if len(rec) else 0.0)

        per_class_ap[c] = class_ap_thresholds

    map50_95 = float(np.mean([np.mean(aps) for aps in per_class_ap.values()]))
    map50 = float(np.mean([aps[0] for aps in per_class_ap.values()]))
    map75 = float(np.mean([aps[5] for aps in per_class_ap.values()]))
    mean_recall = float(np.mean(recalls_50)) if recalls_50 else 0.0

    class_breakdown = {}
    for c, aps in per_class_ap.items():
        name = class_names.get(c, str(c))
        class_breakdown[name] = {
            "ap50_95": float(np.mean(aps)),
            "ap50": float(aps[0]),
            "ap75": float(aps[5]),
        }

    e2e_arr = np.array(e2e_times)
    metrics_report = {
        "mAP50_95": map50_95,
        "mAP50": map50,
        "mAP75": map75,
        "recall": mean_recall,
        "per_class": class_breakdown,
        "speed": {
            "mean_slice_ms": float(np.mean(slice_times)),
            "mean_inference_ms": float(np.mean(infer_times)),
            "mean_merge_ms": float(np.mean(merge_times)),
            "mean_e2e_ms": float(np.mean(e2e_arr)),
            "fps": float(1000.0 / np.mean(e2e_arr)) if len(e2e_arr) else 0.0,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
    log(f"Results: mAP50:95 = {map50_95*100:.2f}%, mAP50 = {map50*100:.2f}%, Recall = {mean_recall*100:.2f}%, FPS = {metrics_report['speed']['fps']:.1f}")
    return metrics_report


def setup_code_repo() -> Path:
    if REPO_COPY.exists():
        shutil.rmtree(REPO_COPY)
    REPO_COPY.mkdir(parents=True, exist_ok=True)

    kaggle_input = Path("/kaggle/input")
    zip_candidates = list(kaggle_input.rglob("code_bundle.zip"))
    if zip_candidates:
        with zipfile.ZipFile(zip_candidates[0], "r") as zf:
            zf.extractall(REPO_COPY)
        return REPO_COPY

    for hit in sorted(kaggle_input.rglob("pyproject.toml")):
        if "repo" not in str(hit):
            shutil.copytree(hit.parent, REPO_COPY, dirs_exist_ok=True)
            return REPO_COPY

    raise FileNotFoundError("Could not locate code bundle in /kaggle/input.")


def find_dataset_root(code_repo: Path) -> Path:
    kaggle_input = Path("/kaggle/input")
    for data_yaml in sorted(kaggle_input.rglob("data.yaml")):
        if code_repo in data_yaml.parents or "repo" in str(data_yaml):
            continue
        return data_yaml.parent
    raise FileNotFoundError("No data.yaml found under /kaggle/input for the raw image dataset.")


def find_baseline_checkpoint() -> Path:
    kaggle_input = Path("/kaggle/input")
    for pt in sorted(kaggle_input.rglob("*.pt")):
        if "best" in pt.name.lower():
            return pt
    return Path("yolo26m.pt")


def pip_install_requirements() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "sahi",
            "iterative-stratification==0.1.9",
            "ultralytics==8.4.147",
        ],
        check=True,
    )


def main() -> None:
    repo_root = setup_code_repo()
    pip_install_requirements()

    sys.path.insert(0, str(repo_root / "src"))
    from scale_error.data.split import split_original_valid
    from scale_error.data.validate import validate_dataset

    log(json.dumps({
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))

    dataset_root = find_dataset_root(repo_root)
    baseline_ckpt = find_baseline_checkpoint()

    # 1. Dataset Report & Stratified Split
    validate_dataset(dataset_root, ARTIFACTS / "dataset_report.json")
    processed_root = ARTIFACTS / "dataset_processed"
    split_original_valid(dataset_root, processed_root)
    shutil.copy2(dataset_root / "data.yaml", processed_root / "data.yaml")

    # 2. Geometric ROI cropping
    roi_root = ARTIFACTS / "dataset_roi"
    log("Applying ROI crop to val and test splits...")
    subprocess.run([
        sys.executable, str(repo_root / "processing" / "apply_roi_crop.py"),
        "--dataset-root", str(processed_root),
        "--roi-config", str(repo_root / "processing" / "roi_config.json"),
        "--output-root", str(roi_root),
        "--splits", "val", "test",
    ], check=True)

    names = {0: "DiVat", 1: "DiVatLoiLom", 2: "LoiChi", 3: "LoiNhua", 4: "LoiTray"}
    test_img_dir = roi_root / "test" / "images"
    test_lbl_dir = roi_root / "test" / "labels"
    eval_dir = ARTIFACTS / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    study_results = {}

    # Experiment A: Baseline YOLO26m direct test val (Full ROI, no slicing)
    log("=== Experiment A: Baseline YOLO26m (No Slicing, Full ROI) ===")
    model = YOLO(str(baseline_ckpt))
    val_res = model.val(
        data=str(processed_root / "data.yaml"),
        split="test",
        imgsz=960,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=False,
        save_json=False,
    )
    study_results["Baseline_YOLO26m_ROI"] = {
        "mAP50_95": float(val_res.box.map),
        "mAP50": float(val_res.box.map50),
        "mAP75": float(val_res.box.map75),
        "per_class_map50_95": [float(x) for x in val_res.box.maps],
        "latency_ms": float(val_res.speed.get("inference", 0.0) + val_res.speed.get("preprocess", 0.0)),
    }

    # Experiment B: SAHI Slices 960x960 Pure Slices
    log("=== Experiment B: Zero-shot SAHI (960x960 Pure Slices, Overlap 0.20) ===")
    sahi_960_metrics = run_standalone_eval(
        weights=baseline_ckpt,
        test_images_dir=test_img_dir,
        test_labels_dir=test_lbl_dir,
        output_json=eval_dir / "metrics_sahi_960.json",
        slice_w=960,
        slice_h=960,
        overlap_ratio=0.20,
        conf_thresh=0.05,
        iou_thresh=0.50,
        device=0 if torch.cuda.is_available() else "cpu",
        include_full_image=False,
        class_names=names,
    )
    study_results["SAHI_960_Pure_Slices"] = {
        "mAP50_95": sahi_960_metrics["mAP50_95"],
        "mAP50": sahi_960_metrics["mAP50"],
        "mAP75": sahi_960_metrics["mAP75"],
        "recall": sahi_960_metrics["recall"],
        "per_class": sahi_960_metrics["per_class"],
        "mean_e2e_ms": sahi_960_metrics["speed"]["mean_e2e_ms"],
        "fps": sahi_960_metrics["speed"]["fps"],
    }

    # Experiment C: SAHI Slices 640x640 (Testing scale match with backbone training size)
    log("=== Experiment C: Zero-shot SAHI (640x640 Slices, Overlap 0.20) ===")
    sahi_640_metrics = run_standalone_eval(
        weights=baseline_ckpt,
        test_images_dir=test_img_dir,
        test_labels_dir=test_lbl_dir,
        output_json=eval_dir / "metrics_sahi_640.json",
        slice_w=640,
        slice_h=640,
        overlap_ratio=0.20,
        conf_thresh=0.05,
        iou_thresh=0.50,
        device=0 if torch.cuda.is_available() else "cpu",
        include_full_image=False,
        class_names=names,
    )
    study_results["SAHI_640_Pure_Slices"] = {
        "mAP50_95": sahi_640_metrics["mAP50_95"],
        "mAP50": sahi_640_metrics["mAP50"],
        "mAP75": sahi_640_metrics["mAP75"],
        "recall": sahi_640_metrics["recall"],
        "per_class": sahi_640_metrics["per_class"],
        "mean_e2e_ms": sahi_640_metrics["speed"]["mean_e2e_ms"],
        "fps": sahi_640_metrics["speed"]["fps"],
    }

    # Experiment D: SAHI Multi-Scale (Full ROI Image + 960x960 Slices Merged)
    log("=== Experiment D: SAHI Multi-Scale (Full Image + 960x960 Slices Merged) ===")
    sahi_multi_metrics = run_standalone_eval(
        weights=baseline_ckpt,
        test_images_dir=test_img_dir,
        test_labels_dir=test_lbl_dir,
        output_json=eval_dir / "metrics_sahi_multiscale.json",
        slice_w=960,
        slice_h=960,
        overlap_ratio=0.20,
        conf_thresh=0.05,
        iou_thresh=0.50,
        device=0 if torch.cuda.is_available() else "cpu",
        include_full_image=True,
        class_names=names,
    )
    study_results["SAHI_MultiScale_Full_Plus_Slices"] = {
        "mAP50_95": sahi_multi_metrics["mAP50_95"],
        "mAP50": sahi_multi_metrics["mAP50"],
        "mAP75": sahi_multi_metrics["mAP75"],
        "recall": sahi_multi_metrics["recall"],
        "per_class": sahi_multi_metrics["per_class"],
        "mean_e2e_ms": sahi_multi_metrics["speed"]["mean_e2e_ms"],
        "fps": sahi_multi_metrics["speed"]["fps"],
    }

    # Save summary
    summary_path = eval_dir / "sahi_study_summary.json"
    summary_path.write_text(json.dumps(study_results, indent=2), encoding="utf-8")
    log("=== Final SAHI Ablation Study Summary ===")
    log(json.dumps(study_results, indent=2))

    if roi_root.exists():
        shutil.rmtree(roi_root, ignore_errors=True)
    if processed_root.exists():
        shutil.rmtree(processed_root, ignore_errors=True)

    log("Ablation study kernel completed successfully!")


if __name__ == "__main__":
    main()
