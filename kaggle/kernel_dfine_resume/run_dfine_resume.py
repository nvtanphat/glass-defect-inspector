"""Resume D-FINE-M from the checkpoint that completed epoch 92.

The run executes epoch indices 93 through 99.  The upstream stage-2 rollback
is disabled because it restores the complete stage-1 training state (including
the epoch counter, optimizer, and schedulers) whenever validation stalls.
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
DFINE_REPO = WORKING / "D-FINE"
OUTPUT_DIR = WORKING / "output_dfine_resume_final"

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def log(message: str) -> None:
    print(f"[scale-error-dfine-resume] {message}", flush=True)


def setup_code_repo() -> Path:
    inputs = Path("/kaggle/input")
    log(f"Available inputs: {[path.name for path in inputs.iterdir()]}")

    # Kaggle's dataset uploader expands directory-mode ZIPs into a
    # ``code_bundle/`` directory.  Keep ZIP support as a fallback so this
    # kernel also works when the archive is uploaded as a single file.
    extracted_bundles = [
        path
        for path in sorted(inputs.rglob("code_bundle"))
        if path.is_dir() and (path / "pyproject.toml").is_file()
    ]
    if extracted_bundles:
        shutil.copytree(extracted_bundles[0], REPO_COPY, dirs_exist_ok=True)
        log(f"Copied extracted code bundle from {extracted_bundles[0]}")
        return REPO_COPY

    REPO_COPY.mkdir(parents=True, exist_ok=True)
    bundles = sorted(inputs.rglob("code_bundle.zip"))
    if not bundles:
        raise FileNotFoundError(
            "Neither an extracted code_bundle directory nor code_bundle.zip "
            "was found in attached datasets"
        )
    with zipfile.ZipFile(bundles[0], "r") as archive:
        archive.extractall(REPO_COPY)
    return REPO_COPY


def find_resume_checkpoint() -> Path:
    candidates = [
        path
        for path in Path("/kaggle/input").rglob("resume.pth")
        if "scale-error-dfine-resume-bundle" in path.parts
    ]
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError(
        "resume.pth was not found under the attached resume dataset"
    )


def find_stage1_checkpoint() -> Path:
    candidates = [
        path
        for path in Path("/kaggle/input").rglob("best_stg1.pth")
        if "scale-error-dfine-resume-bundle" in path.parts
    ]
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError(
        "best_stg1.pth was not found under the attached resume dataset"
    )


def find_dataset_root(code_repo: Path) -> Path:
    candidates = [
        path
        for path in Path("/kaggle/input").rglob("data.yaml")
        if "scale-error-dc-v26-yolov8" in path.parts
    ]
    for data_yaml in sorted(candidates):
        if code_repo not in data_yaml.parents and "repo" not in str(data_yaml).lower():
            return data_yaml.parent
    raise FileNotFoundError("Raw Scale Error dataset data.yaml was not found")


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


def patch_solver_source(source: str) -> str:
    """Patch D-FINE for monotonic, recoverable stage-2 resume training."""
    stage1_only_save = (
        "if self.output_dir and epoch < "
        "self.train_dataloader.collate_fn.stop_epoch:"
    )
    if stage1_only_save not in source:
        raise RuntimeError("Expected D-FINE checkpoint condition was not found")
    source = source.replace(stage1_only_save, "if self.output_dir:", 1)

    # Upstream calls load_resume_state(best_stg1.pth) here.  That method loads
    # every stateful object, not just EMA/model weights, so last_epoch and the
    # optimizer/schedulers jump backwards during a stage-2 resume.
    rollback_block = """                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        \"epoch\": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / \"best_stg1.pth\"))
                        print(f\"Refresh EMA at epoch {epoch} with decay {self.ema.decay}\")
"""
    keep_state_block = """                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    print(
                        f\"Keep stage-2 state at epoch {epoch}; \"
                        \"validation did not improve\"
                    )
"""
    if rollback_block not in source:
        raise RuntimeError("Expected D-FINE stage-2 rollback block was not found")
    return source.replace(rollback_block, keep_state_block, 1)


def patch_stage2_resume_behavior() -> None:
    """Make upstream D-FINE save and resume stage 2 monotonically."""
    solver = DFINE_REPO / "src" / "solver" / "det_solver.py"
    source = solver.read_text(encoding="utf-8")
    solver.write_text(patch_solver_source(source), encoding="utf-8")
    log("Patched D-FINE: save stage-2 last.pth and disable stage-1 rollback")


def checkpoint_epoch(path: Path) -> int:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "last_epoch" not in state:
        raise RuntimeError(f"Checkpoint has no last_epoch metadata: {path}")
    return int(state["last_epoch"])


def build_runtime_dataset(repo_root: Path, dataset_root: Path) -> Path:
    import yaml

    from scale_error.data.split import split_original_valid
    from scale_error.data.validate import validate_dataset

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
    return runtime_yaml


def select_best_checkpoint() -> Path:
    candidates = [
        OUTPUT_DIR / "best_stg2.pth",
        OUTPUT_DIR / "last.pth",
        OUTPUT_DIR / "best_stg1.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Training completed without a usable checkpoint")


def main() -> None:
    repo_root = setup_code_repo()
    install_dependencies()
    patch_stage2_resume_behavior()
    sys.path.insert(0, str(repo_root / "src"))

    import torch

    from scale_error.evaluation.evaluator_dfine import evaluate_dfine
    from scale_error.training.trainer_dfine import train_dfine

    log(
        json.dumps(
            {
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            indent=2,
        )
    )

    resume_checkpoint = find_resume_checkpoint()
    last_epoch = checkpoint_epoch(resume_checkpoint)
    if last_epoch != 92:
        raise RuntimeError(
            f"Expected checkpoint last_epoch=92, found {last_epoch}: {resume_checkpoint}"
        )
    log(f"Resuming from {resume_checkpoint} after epoch {last_epoch}")

    stage1_checkpoint = find_stage1_checkpoint()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage1_checkpoint, OUTPUT_DIR / "best_stg1.pth")
    log(f"Seeded stage-1 checkpoint for EMA refresh: {stage1_checkpoint}")

    # The checkpoint being resumed is the validation baseline.  Preserve it as
    # best_stg2 so select_best_checkpoint() cannot fall back to a worse final
    # epoch when none of the resumed epochs improves on the baseline.
    shutil.copy2(resume_checkpoint, OUTPUT_DIR / "best_stg2.pth")
    log(f"Seeded stage-2 validation baseline: {resume_checkpoint}")

    dataset_root = find_dataset_root(repo_root)
    runtime_yaml = build_runtime_dataset(repo_root, dataset_root)

    # The pretrained path is unused when resume_checkpoint is provided.
    train_dfine(
        runtime_yaml,
        DFINE_REPO,
        resume_checkpoint,
        OUTPUT_DIR,
        epochs=100,
        eval_size=960,
        batch_size=4,
        stop_augment_epoch=70,
        resume_checkpoint=resume_checkpoint,
    )

    final_epoch = checkpoint_epoch(OUTPUT_DIR / "last.pth")
    expected_final_epoch = 99
    if final_epoch != expected_final_epoch:
        raise RuntimeError(
            f"Expected final last_epoch={expected_final_epoch}, found {final_epoch}"
        )

    selected = select_best_checkpoint()
    final_checkpoint = WORKING / "best_resume.pth"
    shutil.copy2(selected, final_checkpoint)
    shutil.copy2(OUTPUT_DIR / "last.pth", WORKING / "last_resume.pth")
    log(f"Selected validation checkpoint: {selected}")
    log(f"Final last checkpoint epoch: {final_epoch}")

    metrics = evaluate_dfine(
        selected,
        DFINE_REPO,
        runtime_yaml,
        WORKING / "metrics_dfine_resume_test.json",
        eval_size=960,
        device="0",
    )
    summary = {
        "resume_from_last_epoch": last_epoch,
        "target_epochs": 100,
        "selected_checkpoint": selected.name,
        "last_checkpoint_epoch": final_epoch,
        "held_out_test_metrics": metrics,
    }
    (WORKING / "resume_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log("Resume training and held-out test evaluation completed")


if __name__ == "__main__":
    main()
