"""Kaggle script-kernel entrypoint for YOLO26m + ROI + Elementwise VFL.

Loads codebase directly from the attached Kaggle code bundle dataset:
  - phtnguyn1ytj/scale-error-yolo26-code
And raw images from:
  - nguynvntnpht/scale-error-dc-v26-yolov8

Runs: validate -> split valid->val/test -> ROI crop -> train -> evaluate on test
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
    print(f"[scale-error] {msg}", flush=True)


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
    raise FileNotFoundError("No data.yaml found under /kaggle/input for the image dataset.")


def pip_install_requirements() -> None:
    log("Installing python dependencies...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "iterative-stratification==0.1.9",
            "ultralytics==8.4.147",
        ],
        check=True,
    )


def find_resume_checkpoint() -> Path | None:
    candidates = list(Path("/kaggle/input").rglob("last.pt"))
    return candidates[0] if candidates else None


def main() -> None:
    repo_root = setup_code_repo()
    pip_install_requirements()

    sys.path.insert(0, str(repo_root / "src"))
    from scale_error.data.split import split_original_valid
    from scale_error.data.validate import validate_dataset
    from scale_error.evaluation.evaluator import evaluate
    from scale_error.training.trainer import train

    import torch
    import yaml

    log(
        json.dumps(
            {
                "python": sys.version,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            indent=2,
        )
    )

    dataset_root = find_dataset_root(repo_root)

    # 1. Validation & dataset_report.json
    validate_dataset(dataset_root, ARTIFACTS / "dataset_report.json")

    # 2. Split original valid -> val & test (50/50 image-level stratified)
    processed_root = ARTIFACTS / "dataset_processed"
    split_original_valid(dataset_root, processed_root)
    shutil.copy2(dataset_root / "data.yaml", processed_root / "data.yaml")

    # 3. Apply geometric ROI crop
    roi_root = ARTIFACTS / "dataset_roi"
    log("Applying ROI crop to train split...")
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "processing" / "apply_roi_crop.py"),
            "--dataset-root",
            str(dataset_root),
            "--roi-config",
            str(repo_root / "processing" / "roi_config.json"),
            "--output-root",
            str(roi_root),
            "--splits",
            "train",
        ],
        check=True,
    )
    log("Applying ROI crop to val and test splits...")
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "processing" / "apply_roi_crop.py"),
            "--dataset-root",
            str(processed_root),
            "--roi-config",
            str(repo_root / "processing" / "roi_config.json"),
            "--output-root",
            str(roi_root),
            "--splits",
            "val",
            "test",
        ],
        check=True,
    )

    # 4. Generate runtime_data.yaml
    names = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
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
    log(f"Configured runtime_data.yaml -> {runtime_yaml}")

    # 5. Fine-tune YOLO26m with elementwise VFL + Class Weights
    log("Starting YOLO26m fine-tuning with elementwise Varifocal Loss + Class Weights...")
    resume_checkpoint = find_resume_checkpoint()
    if resume_checkpoint:
        log(f"Resuming YOLO26m from {resume_checkpoint}")
    else:
        log("No YOLO resume checkpoint attached; starting from pretrained weights")
    train(
        runtime_yaml,
        repo_root / "configs" / "train_yolo26m.yaml",
        ARTIFACTS / "runs",
        dataset_report=ARTIFACTS / "dataset_report.json",
        resume_checkpoint=resume_checkpoint,
    )

    # 6. Locate best checkpoint
    weights = ARTIFACTS / "runs" / "yolo26m" / "weights" / "best.pt"
    if not weights.exists():
        weights = ARTIFACTS / "runs" / "yolo26m_960" / "weights" / "best.pt"
    if not weights.exists():
        hits = list(ARTIFACTS.rglob("best.pt"))
        if hits:
            weights = hits[0]

    log(f"Evaluating best checkpoint: {weights} on held-out test split...")
    for nms in (True, False):
        log(f"Evaluating test split (nms={nms})...")
        evaluate(
            weights,
            runtime_yaml,
            ARTIFACTS / "evaluation" / f"metrics_nms_{nms}.json",
            nms=nms,
        )

    # 7. Clean up intermediate image directories so the final Kaggle artifact zip stays lean (~150MB)
    log("Cleaning up intermediate image directories to keep output compact...")
    for redundant in [roi_root, processed_root, REPO_COPY]:
        if redundant.exists():
            shutil.rmtree(redundant, ignore_errors=True)
            log(f"Removed temporary directory: {redundant}")

    log("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
