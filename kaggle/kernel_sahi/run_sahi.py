"""Kaggle script-kernel entrypoint for YOLO26m + ROI + Sliced Fine-tuning + SAHI.

Pipeline:
1. Attaches Roboflow raw dataset, updated code bundle, and baseline YOLO26m+ROI checkpoint.
2. Performs stratified train/val/test splitting and geometric ROI cropping.
3. High-resolution slicing: tiles ROI into 960x960 patches (20% overlap, boundary truncation, negative sampling).
4. Sliced fine-tuning: fine-tunes YOLO26m starting from best.pt on native 960x960 tiles.
5. SAHI sliced inference: evaluates held-out test split with on-the-fly slicing, batched inference,
   NMM merging, and coordinate remapping.
6. Measures complete benchmark: mAP50, mAP75, mAP50-95, Recall, per-class AP, and E2E latency.
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
    print(f"[scale-error-sahi] {msg}", flush=True)


def setup_code_repo() -> Path:
    if REPO_COPY.exists():
        shutil.rmtree(REPO_COPY)
    REPO_COPY.mkdir(parents=True, exist_ok=True)

    kaggle_input = Path("/kaggle/input")
    log(f"Inspecting /kaggle/input: {[p.name for p in kaggle_input.iterdir()]}")

    # Check for code bundle zip
    zip_candidates = list(kaggle_input.rglob("code_bundle.zip"))
    if zip_candidates:
        zip_path = zip_candidates[0]
        log(f"Extracting code bundle from {zip_path} -> {REPO_COPY}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(REPO_COPY)
        return REPO_COPY

    # Check for unpacked pyproject.toml
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
        if "best" in pt.name.lower() or "last" in pt.name.lower():
            log(f"Found baseline checkpoint to fine-tune from: {pt} ({pt.stat().st_size / 1e6:.1f} MB)")
            return pt
    log("No external checkpoint found, falling back to yolo26m.pt pretrained weights.")
    return Path("yolo26m.pt")


def pip_install_requirements(repo_root: Path) -> None:
    # Directly install external dependencies without editable packaging
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
    pip_install_requirements(repo_root)

    sys.path.insert(0, str(repo_root / "src"))
    from scale_error.data.split import split_original_valid
    from scale_error.data.validate import validate_dataset
    from scale_error.evaluation.sahi_evaluator import evaluate_sahi
    from scale_error.training.trainer import train

    import torch
    import yaml

    log(json.dumps({
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))

    dataset_root = find_dataset_root(repo_root)
    baseline_ckpt = find_baseline_checkpoint()

    # 1. Validate & create dataset report
    validate_dataset(dataset_root, ARTIFACTS / "dataset_report.json")

    # 2. 50/50 Stratified split on valid -> val & test
    processed_root = ARTIFACTS / "dataset_processed"
    split_original_valid(dataset_root, processed_root)
    shutil.copy2(dataset_root / "data.yaml", processed_root / "data.yaml")

    # 3. Apply geometric ROI crop
    roi_root = ARTIFACTS / "dataset_roi"
    log("Applying ROI crop to train split...")
    subprocess.run([
        sys.executable, str(repo_root / "processing" / "apply_roi_crop.py"),
        "--dataset-root", str(dataset_root),
        "--roi-config", str(repo_root / "processing" / "roi_config.json"),
        "--output-root", str(roi_root),
        "--splits", "train",
    ], check=True)

    log("Applying ROI crop to val and test splits...")
    subprocess.run([
        sys.executable, str(repo_root / "processing" / "apply_roi_crop.py"),
        "--dataset-root", str(processed_root),
        "--roi-config", str(repo_root / "processing" / "roi_config.json"),
        "--output-root", str(roi_root),
        "--splits", "val", "test",
    ], check=True)

    # 4. Slicing dataset into high-resolution 960x960 tiles
    sliced_root = ARTIFACTS / "dataset_roi_sliced"
    log("Slicing ROI images into 960x960 tiles with 20% overlap, box truncation, and negative sampling...")
    subprocess.run([
        sys.executable, str(repo_root / "processing" / "slice_dataset.py"),
        "--dataset-root", str(roi_root),
        "--output-root", str(sliced_root),
        "--splits", "train", "val",
        "--slice-w", "960",
        "--slice-h", "960",
        "--overlap", "0.20",
        "--min-area-ratio", "0.30",
        "--negative-ratio", "0.20",
    ], check=True)

    # 5. Prepare sliced training configuration
    sliced_cfg_path = repo_root / "configs" / "train_sliced.yaml"
    cfg = yaml.safe_load(sliced_cfg_path.read_text(encoding="utf-8"))
    cfg["model"] = str(baseline_ckpt.resolve()) if baseline_ckpt.exists() else "yolo26m.pt"
    run_cfg_path = ARTIFACTS / "train_sliced_runtime.yaml"
    run_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    run_cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # 6. Sliced Fine-Tuning with Varifocal Loss + Element-wise Class Weights
    log(f"Starting YOLO26m Sliced Fine-tuning using weights from {cfg['model']}...")
    train(
        sliced_root / "data.yaml",
        run_cfg_path,
        ARTIFACTS / "runs",
        run_name="yolo26m_sahi",
        dataset_report=ARTIFACTS / "dataset_report.json",
    )

    # 7. Locate fine-tuned checkpoint
    weights = ARTIFACTS / "runs" / "yolo26m_sahi" / "weights" / "best.pt"
    if not weights.exists():
        candidates = list(ARTIFACTS.rglob("best.pt"))
        weights = candidates[0] if candidates else baseline_ckpt

    log(f"Fine-tuning complete. Evaluating best checkpoint ({weights}) on held-out test split using SAHI...")

    # 8. Run SAHI Sliced Inference & Comprehensive Metric Evaluation
    test_img_dir = roi_root / "test" / "images"
    test_lbl_dir = roi_root / "test" / "labels"
    out_metrics_json = ARTIFACTS / "evaluation" / "metrics_sahi.json"

    names = {0: "DiVat", 1: "DiVatLoiLom", 2: "LoiChi", 3: "LoiNhua", 4: "LoiTray"}
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

    # 9. Clean up intermediate heavy tile folders so output zip remains compact (<200MB)
    log("Cleaning up intermediate sliced tiles to keep output archive compact...")
    for redundant in [sliced_root, processed_root]:
        if redundant.exists():
            shutil.rmtree(redundant, ignore_errors=True)
            log(f"Cleaned up: {redundant}")

    log("YOLO26m + ROI + Sliced Fine-tuning + SAHI experiment finished successfully!")


if __name__ == "__main__":
    main()
