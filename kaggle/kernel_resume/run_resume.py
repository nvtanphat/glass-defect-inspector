"""Kaggle script-kernel entrypoint to RESUME YOLO26m fine-tuning.

Attaches:
  - nguynvntnpht/scale-error-dc-v26-yolov8         (Roboflow export)
  - phtnguyn1ytj/scale-error-yolo26-code           (this repo code)
  - phtnguyn1ytj/scale-error-yolo26-checkpoint     (uploaded last.pt)
  - phtnguyn1ytj/scale-error-yolo26-roi-crop-train (kernel source with pre-cropped dataset)

Flow:
  1. Set up code repo from /kaggle/input and pip install -e
  2. Locate last.pt checkpoint
  3. Locate or build dataset_roi (uses pre-cropped if kernel source is mounted)
  4. Ensure runtime_data.yaml and checkpoint directory structure are aligned
  5. Resume training from epoch 88 to 120
  6. Final held-out test evaluation (NMS=True and NMS=False)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
ARTIFACTS = WORKING / "artifacts"
REPO_COPY = WORKING / "repo"


def log(msg: str) -> None:
    print(f"[scale-error-resume] {msg}", flush=True)


def find_code_repo() -> Path:
    kaggle_input = Path("/kaggle/input")
    log(f"/kaggle/input contains: {[p.name for p in kaggle_input.iterdir()]}")
    for hit in sorted(kaggle_input.rglob("pyproject.toml")):
        if "repo" in hit.parts:
            continue
        log(f"Using code repo at {hit.parent}")
        return hit.parent
    raise FileNotFoundError("Could not find pyproject.toml under /kaggle/input")


def find_checkpoint() -> Path:
    kaggle_input = Path("/kaggle/input")
    for hit in kaggle_input.rglob("last.pt"):
        log(f"Found last.pt checkpoint at {hit}")
        return hit
    for hit in kaggle_input.rglob("best.pt"):
        log(f"Found best.pt checkpoint at {hit}")
        return hit
    raise FileNotFoundError("No last.pt or best.pt found under /kaggle/input")


def copy_code_repo(source: Path) -> Path:
    if REPO_COPY.exists():
        shutil.rmtree(REPO_COPY)
    shutil.copytree(source, REPO_COPY)
    log(f"Copied code repo {source} -> {REPO_COPY}")
    return REPO_COPY


def pip_install_repo(repo_root: Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_root)], check=True)


def setup_roi_dataset(repo_root: Path) -> Path:
    """Find pre-cropped dataset or generate from raw images."""
    kaggle_input = Path("/kaggle/input")
    
    # 1. Check if pre-cropped dataset_roi is available via kernel source
    for d in kaggle_input.rglob("dataset_roi"):
        if (d / "train" / "images").exists() and (d / "val" / "images").exists():
            log(f"Using pre-cropped dataset_roi from {d}")
            return d

    # 2. Fallback: generate ROI crops from raw dataset
    log("Pre-cropped dataset_roi not found mounted. Building from raw dataset...")
    for data_yaml in sorted(kaggle_input.rglob("data.yaml")):
        if repo_root in data_yaml.parents:
            continue
        dataset_root = data_yaml.parent
        break
    else:
        raise FileNotFoundError("No data.yaml found for raw dataset.")

    sys.path.insert(0, str(repo_root / "src"))
    from scale_error.data.split import split_original_valid
    from scale_error.data.validate import validate_dataset

    validate_dataset(dataset_root, ARTIFACTS / "dataset_report.json")
    processed_root = ARTIFACTS / "dataset_processed"
    split_original_valid(dataset_root, processed_root)
    shutil.copy2(dataset_root / "data.yaml", processed_root / "data.yaml")

    roi_root = ARTIFACTS / "dataset_roi"
    subprocess.run([
        sys.executable, str(repo_root / "processing" / "apply_roi_crop.py"),
        "--dataset-root", str(dataset_root),
        "--roi-config", str(repo_root / "processing" / "roi_config.json"),
        "--output-root", str(roi_root),
        "--splits", "train",
    ], check=True)
    subprocess.run([
        sys.executable, str(repo_root / "processing" / "apply_roi_crop.py"),
        "--dataset-root", str(processed_root),
        "--roi-config", str(repo_root / "processing" / "roi_config.json"),
        "--output-root", str(roi_root),
        "--splits", "val", "test",
    ], check=True)

    return roi_root


def main() -> None:
    repo_root = copy_code_repo(find_code_repo())
    pip_install_repo(repo_root)

    sys.path.insert(0, str(repo_root / "src"))
    from scale_error.evaluation.evaluator import evaluate
    from ultralytics import YOLO
    import torch
    import yaml

    log(json.dumps({
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))

    # 1. Setup ROI dataset
    roi_root = setup_roi_dataset(repo_root)

    # 2. Write runtime_data.yaml
    names = {0: "DiVat", 1: "DiVatLoiLom", 2: "LoiChi", 3: "LoiNhua", 4: "LoiTray"}
    runtime_yaml = ARTIFACTS / "dataset" / "runtime_data.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str((roi_root / "train" / "images").resolve()),
                "val": str((roi_root / "val" / "images").resolve()),
                "test": str((roi_root / "test" / "images").resolve()),
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    log(f"Configured runtime_data.yaml at {runtime_yaml}")

    # 3. Setup checkpoint directory structure for resume
    ckpt_source = find_checkpoint()
    run_weights_dir = ARTIFACTS / "runs" / "yolo26m" / "weights"
    run_weights_dir.mkdir(parents=True, exist_ok=True)
    local_last = run_weights_dir / "last.pt"
    if not local_last.exists() or local_last.resolve() != ckpt_source.resolve():
        shutil.copy2(ckpt_source, local_last)
        log(f"Copied checkpoint {ckpt_source} -> {local_last}")

    # 4. Resume training
    log("Initializing YOLO with checkpoint for resume...")
    model = YOLO(str(local_last))
    log("Resuming training to complete remaining epochs...")
    model.train(resume=True)

    # 5. Final held-out test evaluation
    best_weights = run_weights_dir / "best.pt"
    if not best_weights.exists():
        best_weights = local_last

    log(f"Training finished. Evaluating best weights ({best_weights}) on test split...")
    for nms in (True, False):
        log(f"Evaluating test split (nms={nms})...")
        evaluate(
            best_weights,
            runtime_yaml,
            ARTIFACTS / "evaluation" / f"metrics_nms_{nms}.json",
            nms=nms,
        )

    log("All resume training and evaluation tasks completed successfully!")


if __name__ == "__main__":
    main()
