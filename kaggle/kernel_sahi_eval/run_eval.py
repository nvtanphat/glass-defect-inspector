"""Kernel 3: Sliced Fine-tuning Resume & SAHI Benchmark Evaluation.

Attaches:
  - phtnguyn1ytj/scale-error-yolo26-sahi-train (Sliced fine-tuned weights: best.pt or last.pt)
  - phtnguyn1ytj/scale-error-yolo26-roi-crop-train (Baseline checkpoint & data configs)
  - phtnguyn1ytj/scale-error-yolo26-code (Source code bundle)
  - nguynvntnpht/scale-error-dc-v26-yolov8 (Raw image dataset)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")
ARTIFACTS = WORKING / "artifacts"
REPO_COPY = WORKING / "repo"


def log(msg: str) -> None:
    print(f"[scale-error-sahi-eval] {msg}", flush=True)


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


def find_eval_checkpoint() -> Path:
    kaggle_input = Path("/kaggle/input")
    
    # Check SAHI kernel outputs first
    sahi_hits = []
    for pt in sorted(kaggle_input.rglob("*.pt")):
        if "sahi" in str(pt).lower() and ("best" in pt.name.lower() or "last" in pt.name.lower()):
            sahi_hits.append(pt)
    if sahi_hits:
        chosen = sahi_hits[0]
        log(f"Found Sliced fine-tuned checkpoint from SAHI kernel: {chosen} ({chosen.stat().st_size / 1e6:.1f} MB)")
        return chosen

    # Fallback to any best.pt
    for pt in sorted(kaggle_input.rglob("*.pt")):
        if "best" in pt.name.lower():
            log(f"Found best checkpoint from kernel input: {pt} ({pt.stat().st_size / 1e6:.1f} MB)")
            return pt

    log("Falling back to yolo26m.pt pretrained weights.")
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
    import yaml

    log(json.dumps({
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))

    dataset_root = find_dataset_root(repo_root)
    weights = find_eval_checkpoint()

    # 1. 50/50 Stratified split on valid -> val & test
    processed_root = ARTIFACTS / "dataset_processed"
    split_original_valid(dataset_root, processed_root)
    shutil.copy2(dataset_root / "data.yaml", processed_root / "data.yaml")

    # 2. Apply geometric ROI crop to test split
    roi_root = ARTIFACTS / "dataset_roi"
    log("Applying ROI crop to test split...")
    subprocess.run([
        sys.executable, str(repo_root / "processing" / "apply_roi_crop.py"),
        "--dataset-root", str(processed_root),
        "--roi-config", str(repo_root / "processing" / "roi_config.json"),
        "--output-root", str(roi_root),
        "--splits", "test",
    ], check=True)

    # 3. Run SAHI Sliced Inference & Comprehensive Metric Evaluation
    test_img_dir = roi_root / "test" / "images"
    test_lbl_dir = roi_root / "test" / "labels"
    out_metrics_json = ARTIFACTS / "evaluation" / "metrics_sahi.json"
    out_metrics_json.parent.mkdir(parents=True, exist_ok=True)

    names = {0: "DiVat", 1: "DiVatLoiLom", 2: "LoiChi", 3: "LoiNhua", 4: "LoiTray"}
    log(f"Starting SAHI evaluation with weights: {weights} on {test_img_dir}...")
    
    metrics = evaluate_sahi(
        weights=weights,
        test_images_dir=test_img_dir,
        test_labels_dir=test_lbl_dir,
        output_json=out_metrics_json,
        slice_w=960,
        slice_h=960,
        overlap_ratio=0.20,
        conf_thresh=0.20,
        iou_thresh=0.50,
        device=0 if torch.cuda.is_available() else "cpu",
        num_classes=5,
        class_names=names,
    )

    log("SAHI Evaluation Metrics Summary:")
    log(json.dumps(metrics, indent=2))

    # Clean up heavy images to keep artifact small
    if roi_root.exists():
        shutil.rmtree(roi_root, ignore_errors=True)
    if processed_root.exists():
        shutil.rmtree(processed_root, ignore_errors=True)

    log("Kernel 3 SAHI Evaluation finished successfully!")


if __name__ == "__main__":
    main()
