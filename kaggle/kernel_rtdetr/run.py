"""Kaggle script-kernel entrypoint: RT-DETR-L on the same ROI-cropped data as
the YOLO26m run, for a two-architecture ensemble (transformer vs CNN detector,
so prediction errors are less correlated than two YOLO variants).

Attaches two Kaggle datasets:
  - nguynvntnpht/scale-error-dc-v26-yolov8   (Roboflow export, raw images+labels)
  - phtnguyn1ytj/scale-error-yolo26-code     (this repo's src/ + processing/ + configs/)

Runs: validate -> split valid->val/test -> ROI crop (train + val/test) -> train RT-DETR-L -> evaluate
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
    print(f"[scale-error-rtdetr] {msg}", flush=True)


def find_code_repo() -> Path:
    kaggle_input = Path("/kaggle/input")
    log(f"/kaggle/input contains: {[p.name for p in kaggle_input.iterdir()]}")
    for hit in sorted(kaggle_input.rglob("pyproject.toml")):
        log(f"Using code repo at {hit.parent}")
        return hit.parent
    raise FileNotFoundError("Could not find pyproject.toml under /kaggle/input (code dataset not attached?)")


def find_dataset_root(code_repo: Path) -> Path:
    kaggle_input = Path("/kaggle/input")
    for data_yaml in sorted(kaggle_input.rglob("data.yaml")):
        if code_repo in data_yaml.parents:
            continue
        log(f"Using attached raw dataset at {data_yaml.parent}")
        return data_yaml.parent
    raise FileNotFoundError("No data.yaml found under /kaggle/input for the image dataset.")


def copy_code_repo(source: Path) -> Path:
    if REPO_COPY.exists():
        shutil.rmtree(REPO_COPY)
    shutil.copytree(source, REPO_COPY)
    log(f"Copied code repo {source} -> {REPO_COPY}")
    return REPO_COPY


def pip_install_repo(repo_root: Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_root)], check=True)


def main() -> None:
    repo_root = copy_code_repo(find_code_repo())
    pip_install_repo(repo_root)

    sys.path.insert(0, str(repo_root / "src"))
    from scale_error.data.split import split_original_valid
    from scale_error.data.validate import validate_dataset
    from scale_error.evaluation.evaluator import evaluate
    from scale_error.training.trainer import train

    import torch

    log(json.dumps({
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))

    dataset_root = find_dataset_root(repo_root)

    validate_dataset(dataset_root, ARTIFACTS / "dataset_report.json")

    processed_root = ARTIFACTS / "dataset_processed"
    split_original_valid(dataset_root, processed_root)
    shutil.copy2(dataset_root / "data.yaml", processed_root / "data.yaml")

    roi_root = ARTIFACTS / "dataset_roi"
    log("Applying ROI crop to train ...")
    subprocess.run(
        [
            sys.executable, str(repo_root / "processing" / "apply_roi_crop.py"),
            "--dataset-root", str(dataset_root),
            "--roi-config", str(repo_root / "processing" / "roi_config.json"),
            "--output-root", str(roi_root),
            "--splits", "train",
        ],
        check=True,
    )
    log("Applying ROI crop to val/test ...")
    subprocess.run(
        [
            sys.executable, str(repo_root / "processing" / "apply_roi_crop.py"),
            "--dataset-root", str(processed_root),
            "--roi-config", str(repo_root / "processing" / "roi_config.json"),
            "--output-root", str(roi_root),
            "--splits", "val", "test",
        ],
        check=True,
    )

    import yaml
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
    log(f"runtime_data.yaml -> {runtime_yaml}")

    log("Starting RT-DETR-L fine-tuning (ROI-cropped data)...")
    train(runtime_yaml, repo_root / "configs" / "train_rtdetr.yaml", ARTIFACTS / "runs", run_name="rtdetr_l_960")

    weights = ARTIFACTS / "runs" / "rtdetr_l_960" / "weights" / "best.pt"
    for nms in (True, False):
        log(f"Evaluating test split (nms={nms})...")
        evaluate(weights, runtime_yaml, ARTIFACTS / "evaluation" / f"metrics_nms_{nms}.json", nms=nms)

    log("Done.")


if __name__ == "__main__":
    main()
