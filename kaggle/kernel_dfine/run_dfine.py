"""Kaggle entrypoint for D-FINE-M on the label-preserving ROI dataset."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")
ARTIFACTS = WORKING / "artifacts"
REPO_COPY = WORKING / "repo"
DFINE_REPO = WORKING / "D-FINE"
DFINE_WEIGHTS = WORKING / "dfine_m_coco.pth"

# Reduce allocator fragmentation on memory-constrained Tesla T4 workers.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def log(message: str) -> None:
    print(f"[scale-error-dfine] {message}", flush=True)


def setup_code_repo() -> Path:
    REPO_COPY.mkdir(parents=True, exist_ok=True)
    kaggle_input = Path("/kaggle/input")
    log(f"Available inputs: {[path.name for path in kaggle_input.iterdir()]}")
    zip_candidates = list(kaggle_input.rglob("code_bundle.zip"))
    if zip_candidates:
        with zipfile.ZipFile(zip_candidates[0], "r") as archive:
            archive.extractall(REPO_COPY)
        return REPO_COPY
    for pyproject in sorted(kaggle_input.rglob("pyproject.toml")):
        source_root = pyproject.parent
        if (source_root / "src").exists():
            shutil.copytree(source_root, REPO_COPY, dirs_exist_ok=True)
            return REPO_COPY
    raise FileNotFoundError(
        "Code dataset is attached but neither code_bundle.zip nor an unpacked src tree was found"
    )


def find_resume_checkpoint(suffix: str) -> Path | None:
    candidates = [
        path
        for path in Path("/kaggle/input").rglob(f"last.{suffix}")
        if "dfine-m-coco" not in str(path).lower()
    ]
    return candidates[0] if candidates else None


def find_dataset_root(code_repo: Path) -> Path:
    for data_yaml in sorted(Path("/kaggle/input").rglob("data.yaml")):
        if code_repo not in data_yaml.parents and "repo" not in str(data_yaml):
            return data_yaml.parent
    raise FileNotFoundError("Raw dataset data.yaml was not found")


def install_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "iterative-stratification==0.1.9",
            "PyYAML>=6.0.2",
            "Pillow>=10.0",
            "pycocotools>=2.0.7",
            "tensorboard>=2.15",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/Peterande/D-FINE.git",
            str(DFINE_REPO),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(DFINE_REPO / "requirements.txt")],
        check=True,
    )


def main() -> None:
    repo_root = setup_code_repo()
    install_dependencies()
    sys.path.insert(0, str(repo_root / "src"))

    import torch
    import yaml

    from scale_error.data.split import split_original_valid
    from scale_error.data.validate import validate_dataset
    from scale_error.evaluation.evaluator_dfine import evaluate_dfine
    from scale_error.training.trainer_dfine import train_dfine

    log(
        json.dumps(
            {"cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0)},
            indent=2,
        )
    )
    dataset_root = find_dataset_root(repo_root)
    report = ARTIFACTS / "dataset_report.json"
    validate_dataset(dataset_root, report)

    processed_root = ARTIFACTS / "dataset_processed"
    split_original_valid(dataset_root, processed_root)
    shutil.copy2(dataset_root / "data.yaml", processed_root / "data.yaml")
    roi_root = ARTIFACTS / "dataset_roi"
    for source_root, splits in ((dataset_root, ["train"]), (processed_root, ["val", "test"])):
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "processing" / "apply_roi_crop.py"),
                "--dataset-root",
                str(source_root),
                "--roi-config",
                str(repo_root / "processing" / "roi_config.json"),
                "--output-root",
                str(roi_root),
                "--splits",
                *splits,
            ],
            check=True,
        )

    raw_cfg = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))
    names = raw_cfg["names"]
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
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
        ),
        encoding="utf-8",
    )

    urllib.request.urlretrieve(
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
        DFINE_WEIGHTS,
    )
    output_dir = ARTIFACTS / "runs" / "dfine_m"
    resume_checkpoint = find_resume_checkpoint("pth")
    if resume_checkpoint:
        log(f"Resuming D-FINE from {resume_checkpoint}")
    else:
        log("No D-FINE resume checkpoint attached; starting from COCO weights")
    train_dfine(
        runtime_yaml,
        DFINE_REPO,
        DFINE_WEIGHTS,
        output_dir,
        epochs=100,
        eval_size=960,
        batch_size=4,
        stop_augment_epoch=70,
        resume_checkpoint=resume_checkpoint,
    )
    best = output_dir / "best.pth"
    if not best.exists():
        raise FileNotFoundError(f"D-FINE best checkpoint not found: {best}")
    evaluate_dfine(
        best,
        DFINE_REPO,
        runtime_yaml,
        ARTIFACTS / "evaluation" / "metrics_dfine_m.json",
        eval_size=960,
        device="0",
    )
    log("D-FINE-M training and held-out evaluation completed")


if __name__ == "__main__":
    main()
