"""Resume the corrected D-FINE-M run from epoch 23 through epoch 46."""

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
OUTPUT_DIR = WORKING / "output_dfine_fresh_resume24"

EXPECTED_RESUME_EPOCH = 23
RUN_TARGET_EPOCHS = 47
FINAL_TARGET_EPOCHS = 100
STOP_AUGMENT_EPOCH = 88
TRAIN_TOTAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
ACCUMULATE_STEPS = TRAIN_TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["DFINE_ACCUMULATE_STEPS"] = str(ACCUMULATE_STEPS)


def log(message: str) -> None:
    print(f"[dfine-fresh-resume24] {message}", flush=True)


def setup_code_repo() -> Path:
    inputs = Path("/kaggle/input")
    extracted = [
        path
        for path in sorted(inputs.rglob("code_bundle"))
        if path.is_dir() and (path / "pyproject.toml").is_file()
    ]
    if extracted:
        shutil.copytree(extracted[0], REPO_COPY, dirs_exist_ok=True)
        return REPO_COPY

    bundles = sorted(inputs.rglob("code_bundle.zip"))
    if not bundles:
        raise FileNotFoundError("Code bundle was not found in Kaggle inputs")
    REPO_COPY.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundles[0], "r") as archive:
        archive.extractall(REPO_COPY)
    return REPO_COPY


def find_dataset_root(code_repo: Path) -> Path:
    candidates = [
        path
        for path in Path("/kaggle/input").rglob("data.yaml")
        if "scale-error-dc-v26-yolov8" in path.parts
    ]
    for data_yaml in sorted(candidates):
        if code_repo not in data_yaml.parents:
            return data_yaml.parent
    raise FileNotFoundError("Raw Scale Error dataset data.yaml was not found")


def find_checkpoint(filename: str) -> Path:
    candidates = [
        path
        for path in Path("/kaggle/input").rglob(filename)
        if "scale-error-dfine-fresh-resume-epoch24" in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(f"{filename} was not found in the resume dataset")
    return sorted(candidates)[0]


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


def patch_gradient_accumulation(source: str) -> str:
    """Add single-GPU gradient accumulation to D-FINE's train loop."""
    replacements = {
        "import math\nimport sys": "import math\nimport os\nimport sys",
        """    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
""": """    accumulate_steps = int(os.environ.get(\"DFINE_ACCUMULATE_STEPS\", \"1\"))
    optimizer_steps = math.ceil(len(data_loader) / accumulate_steps)
    optimizer.zero_grad()
    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        optimizer_step = i // accumulate_steps
        global_step = epoch * optimizer_steps + optimizer_step
        metas = dict(
            epoch=epoch,
            step=optimizer_step,
            global_step=global_step,
            epoch_step=optimizer_steps,
        )
        should_step = (i + 1) % accumulate_steps == 0 or i + 1 == len(data_loader)
""",
        """            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
""": """            scaler.scale(loss / accumulate_steps).backward()

            if should_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
""",
        """            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
""": """            loss: torch.Tensor = sum(loss_dict.values())
            (loss / accumulate_steps).backward()

            if should_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad()
""",
        """        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()
""": """        # ema
        # Match one EMA/warmup update to one effective global-batch update.
        if should_step and ema is not None:
            ema.update(model)

        if should_step and lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()
""",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"Expected D-FINE engine fragment not found: {old[:80]!r}")
        source = source.replace(old, new, 1)
    return source


def patch_dfine_engine() -> None:
    engine = DFINE_REPO / "src" / "solver" / "det_engine.py"
    source = engine.read_text(encoding="utf-8")
    engine.write_text(patch_gradient_accumulation(source), encoding="utf-8")
    log(
        f"Enabled gradient accumulation: micro_batch={MICRO_BATCH_SIZE}, "
        f"steps={ACCUMULATE_STEPS}, effective_batch={TRAIN_TOTAL_BATCH_SIZE}"
    )


def patch_reference_augmentation(source: str) -> str:
    """Restore the augmentation recipe recorded in the original run log."""
    replacements = {
        '{"type": "RandomPhotometricDistort", "p": 0.3},': (
            '{"type": "RandomPhotometricDistort", "p": 0.5},\n'
            '                        {"type": "RandomZoomOut", "fill": 0},'
        ),
        '{"type": "RandomIoUCrop", "p": 0.5},': (
            '{"type": "RandomIoUCrop", "p": 0.8},'
        ),
        '                            "RandomPhotometricDistort",\n'
        '                            "RandomIoUCrop",': (
            '                            "RandomPhotometricDistort",\n'
            '                            "RandomZoomOut",\n'
            '                            "RandomIoUCrop",'
        ),
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"Expected trainer recipe fragment not found: {old!r}")
        source = source.replace(old, new, 1)
    return source


def patch_code_recipe(repo_root: Path) -> None:
    trainer = repo_root / "src" / "scale_error" / "training" / "trainer_dfine.py"
    source = trainer.read_text(encoding="utf-8")
    trainer.write_text(patch_reference_augmentation(source), encoding="utf-8")
    log("Restored reference augmentation: photometric=0.5, zoom-out, IoU-crop=0.8")


def build_runtime_dataset(repo_root: Path, dataset_root: Path) -> Path:
    import yaml

    from scale_error.data.split import split_original_valid
    from scale_error.data.validate import validate_dataset

    validate_dataset(dataset_root, ARTIFACTS / "dataset_report.json")
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


def checkpoint_epoch(path: Path) -> int:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "last_epoch" not in state:
        raise RuntimeError(f"Invalid checkpoint: {path}")
    return int(state["last_epoch"])


def export_resume_outputs(resume_from: int) -> None:
    last_checkpoint = OUTPUT_DIR / "last.pth"
    best_checkpoint = OUTPUT_DIR / "best_stg1.pth"
    final_epoch = checkpoint_epoch(last_checkpoint)
    if final_epoch != RUN_TARGET_EPOCHS - 1:
        raise RuntimeError(
            f"Expected resumed last_epoch={RUN_TARGET_EPOCHS - 1}, found {final_epoch}"
        )

    shutil.copy2(last_checkpoint, WORKING / "resume24_last.pth")
    shutil.copy2(best_checkpoint, WORKING / "resume24_best_stg1.pth")
    log_path = OUTPUT_DIR / "log.txt"
    if log_path.exists():
        shutil.copy2(log_path, WORKING / "resume24_log.txt")

    summary = {
        "resume_from_last_epoch": resume_from,
        "phase_epoch_indices": [resume_from + 1, RUN_TARGET_EPOCHS - 1],
        "last_checkpoint_epoch": final_epoch,
        "final_target_epochs": FINAL_TARGET_EPOCHS,
        "stop_augment_epoch": STOP_AUGMENT_EPOCH,
        "train_total_batch_size": TRAIN_TOTAL_BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "accumulate_steps": ACCUMULATE_STEPS,
    }
    (WORKING / "resume24_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def cleanup_large_intermediates() -> None:
    """Keep Kaggle output limited to checkpoints, log, and summary."""
    for path in (ARTIFACTS, OUTPUT_DIR, DFINE_REPO, REPO_COPY):
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    repo_root = setup_code_repo()
    patch_code_recipe(repo_root)
    install_dependencies()
    patch_dfine_engine()
    sys.path.insert(0, str(repo_root / "src"))

    import torch

    from scale_error.training.trainer_dfine import train_dfine

    log(
        json.dumps(
            {
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count(),
                "fresh_start": False,
            },
            indent=2,
        )
    )
    dataset_root = find_dataset_root(repo_root)
    runtime_yaml = build_runtime_dataset(repo_root, dataset_root)

    resume_checkpoint = find_checkpoint("resume.pth")
    best_stage1 = find_checkpoint("best_stg1.pth")
    resume_epoch = checkpoint_epoch(resume_checkpoint)
    if resume_epoch != EXPECTED_RESUME_EPOCH:
        raise RuntimeError(
            f"Expected resume last_epoch={EXPECTED_RESUME_EPOCH}, found {resume_epoch}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_stage1, OUTPUT_DIR / "best_stg1.pth")
    log(f"Resuming after epoch {resume_epoch}: {resume_checkpoint}")
    train_dfine(
        runtime_yaml,
        DFINE_REPO,
        resume_checkpoint,
        OUTPUT_DIR,
        epochs=RUN_TARGET_EPOCHS,
        eval_size=960,
        batch_size=MICRO_BATCH_SIZE,
        stop_augment_epoch=STOP_AUGMENT_EPOCH,
        resume_checkpoint=resume_checkpoint,
    )
    export_resume_outputs(resume_epoch)
    cleanup_large_intermediates()
    log("Resume phase completed")


if __name__ == "__main__":
    main()
