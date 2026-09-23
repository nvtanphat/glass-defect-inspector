"""Kaggle script-kernel entrypoint for Comprehensive SAHI Ablation & Validation Study.

Implements the user's Phase 1, 2, 3, and 5 experiment plan:
1. Validates dataset, split, and geometric ROI crop.
2. Evaluates the baseline YOLO26m (Full ROI).
3. Evaluates SAHI 640x640 vs 960x960 slices on existing checkpoint.
4. Evaluates SAHI Multi-scale (Full image + Slices merged).
5. Produces end-to-end comparative trade-off matrix: mAP50:95, per-class AP, Recall, and Latency (FPS).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")
ARTIFACTS = WORKING / "artifacts"
REPO_COPY = WORKING / "repo"


def log(msg: str) -> None:
    print(f"[sahi-study] {msg}", flush=True)


def setup_code_repo() -> Path:
    if REPO_COPY.exists():
        shutil.rmtree(REPO_COPY)
    REPO_COPY.mkdir(parents=True, exist_ok=True)

    kaggle_input = Path("/kaggle/input")
    log(f"Inspecting /kaggle/input: {[p.name for p in kaggle_input.iterdir()]}")

    zip_candidates = list(kaggle_input.rglob("code_bundle.zip"))
    if zip_candidates:
        zip_path = zip_candidates[0]
        log(f"Extracting code bundle from {zip_path} -> {REPO_COPY}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(REPO_COPY)
        return REPO_COPY

    for hit in sorted(kaggle_input.rglob("pyproject.toml")):
        if "repo" not in str(hit):
            log(f"Copying code directory from {hit.parent} -> {REPO_COPY}")
            shutil.copytree(hit.parent, REPO_COPY, dirs_exist_ok=True)
            return REPO_COPY

    raise FileNotFoundError("Could not locate code bundle in /kaggle/input.")


def find_dataset_root(code_repo: Path) -> Path:
    kaggle_input = Path("/kaggle/input")
    for data_yaml in sorted(kaggle_input.rglob("data.yaml")):
        if code_repo in data_yaml.parents or "repo" in str(data_yaml):
            continue
        log(f"Using attached raw dataset at {data_yaml.parent}")
        return data_yaml.parent
    raise FileNotFoundError("No data.yaml found under /kaggle/input for the raw image dataset.")


def find_baseline_checkpoint() -> Path:
    kaggle_input = Path("/kaggle/input")
    for pt in sorted(kaggle_input.rglob("*.pt")):
        if "best" in pt.name.lower():
            log(f"Found baseline checkpoint to evaluate: {pt} ({pt.stat().st_size / 1e6:.1f} MB)")
            return pt
    log("No external checkpoint found, falling back to yolo26m.pt pretrained weights.")
    return Path("yolo26m.pt")


def pip_install_requirements() -> None:
    log("Installing dependencies (sahi, iterative-stratification, ultralytics)...")
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
    from scale_error.evaluation.sahi_evaluator import evaluate_sahi

    import torch
    from ultralytics import YOLO

    log(json.dumps({
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))

    dataset_root = find_dataset_root(repo_root)
    baseline_ckpt = find_baseline_checkpoint()

    # Phase 1: Validate dataset & 50/50 Stratified split
    log("=== Phase 1: Dataset Validation & Stratified Split ===")
    validate_dataset(dataset_root, ARTIFACTS / "dataset_report.json")
    processed_root = ARTIFACTS / "dataset_processed"
    split_original_valid(dataset_root, processed_root)
    shutil.copy2(dataset_root / "data.yaml", processed_root / "data.yaml")

    # Geometric ROI crop for val and test splits
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

    # Experiment A: Baseline YOLO26m standard evaluation (Direct full ROI image)
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
    log(f"Baseline result: mAP50:95 = {val_res.box.map*100:.2f}%, mAP50 = {val_res.box.map50*100:.2f}%")

    # Experiment B: SAHI Slices 960x960 (Pure Tiling)
    log("=== Experiment B: Zero-shot SAHI (960x960 Pure Slices, Overlap 0.20) ===")
    sahi_960_metrics = evaluate_sahi(
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
        num_classes=5,
        class_names=names,
        include_full_image=False,
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
    sahi_640_metrics = evaluate_sahi(
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
        num_classes=5,
        class_names=names,
        include_full_image=False,
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
    sahi_multi_metrics = evaluate_sahi(
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
        num_classes=5,
        class_names=names,
        include_full_image=True,
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

    # Save complete study summary
    summary_path = eval_dir / "sahi_study_summary.json"
    summary_path.write_text(json.dumps(study_results, indent=2), encoding="utf-8")
    log("=== Final SAHI Ablation Study Summary ===")
    log(json.dumps(study_results, indent=2))

    # Clean up heavy processed splits
    if roi_root.exists():
        shutil.rmtree(roi_root, ignore_errors=True)
    if processed_root.exists():
        shutil.rmtree(processed_root, ignore_errors=True)

    log("Ablation study kernel completed successfully!")


if __name__ == "__main__":
    main()
