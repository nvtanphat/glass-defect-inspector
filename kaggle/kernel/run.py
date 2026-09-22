"""Kaggle script-kernel entrypoint for YOLO26m + ROI + Elementwise VFL.

Clones the latest codebase from GitHub (with fixed elementwise Varifocal Loss
and per-class inverse-frequency weighting) and uses the attached Roboflow dataset:
  - nguynvntnpht/scale-error-dc-v26-yolov8 (Raw images + labels)

Runs: validate -> split valid->val/test -> ROI crop -> train -> evaluate on test
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
ARTIFACTS = WORKING / "artifacts"
REPO_COPY = WORKING / "repo"


def log(msg: str) -> None:
    print(f"[scale-error] {msg}", flush=True)


def find_code_repo() -> Path:
    if REPO_COPY.exists():
        shutil.rmtree(REPO_COPY)
    try:
        log("Cloning latest code from GitHub (commit with elementwise VFL fix)...")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/nvtanphat/-glass-defect-inspector.git",
                str(REPO_COPY),
            ],
            check=True,
        )
        log(f"Cloned repository successfully into {REPO_COPY}")
        return REPO_COPY
    except Exception as e:
        log(f"Git clone failed ({e}), searching /kaggle/input for pyproject.toml...")
        kaggle_input = Path("/kaggle/input")
        for hit in sorted(kaggle_input.rglob("pyproject.toml")):
            log(f"Using code repo from input at {hit.parent}")
            shutil.copytree(hit.parent, REPO_COPY)
            return REPO_COPY
    raise FileNotFoundError("Could not obtain code repository.")


def find_dataset_root(code_repo: Path) -> Path:
    kaggle_input = Path("/kaggle/input")
    for data_yaml in sorted(kaggle_input.rglob("data.yaml")):
        if code_repo in data_yaml.parents:
            continue
        log(f"Using attached raw dataset at {data_yaml.parent}")
        return data_yaml.parent
    raise FileNotFoundError("No data.yaml found under /kaggle/input for the image dataset.")


def pip_install_repo(repo_root: Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_root)], check=True)


def main() -> None:
    repo_root = find_code_repo()
    pip_install_repo(repo_root)

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
    train(
        runtime_yaml,
        repo_root / "configs" / "train.yaml",
        ARTIFACTS / "runs",
        dataset_report=ARTIFACTS / "dataset_report.json",
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

    # 7. Clean up heavy image directories so the final Kaggle artifact zip stays lean (~150MB)
    log("Cleaning up intermediate image directories to keep output compact...")
    for redundant in [roi_root, processed_root, REPO_COPY]:
        if redundant.exists():
            shutil.rmtree(redundant, ignore_errors=True)
            log(f"Removed temporary directory: {redundant}")

    log("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
