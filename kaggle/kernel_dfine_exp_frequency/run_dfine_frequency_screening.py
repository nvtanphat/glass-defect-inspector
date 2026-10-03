"""Experiment E2: D-FINE-M with Frequency-Aware Feature Refinement (Screening Stage B - 15 Epochs).

Hypothesis:
    Industrial glass bottle images feature high-contrast background frequencies
    (specular reflections, illumination boundaries, molded ring edges). At the same time,
    hairline defects (scratches, cracks) are also high-frequency signals.
    By decomposing shallow feature P3 (stride 8, 384 channels) into low- and high-frequency components
    and applying a gated refinement with learnable residual alpha (initialized at 0.0),
    the model learns to suppress non-defect glass reflections while sharpening defect edges.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")
EXPERIMENTS_ROOT = WORKING / "glass-defect-experiments"
EXP_DIR = EXPERIMENTS_ROOT / "frequency"
ARTIFACTS = WORKING / "artifacts"
REPO_COPY = WORKING / "repo"
DFINE_REPO = WORKING / "D-FINE"

TOTAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
ACCUMULATE_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE
SCREENING_EPOCHS = 15

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["DFINE_ACCUMULATE_STEPS"] = str(ACCUMULATE_STEPS)


def log(msg: str) -> None:
    print(f"[exp-freq-screening] {msg}", flush=True)


def inspect_environment() -> dict:
    log("Inspecting Kaggle Environment...")
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0
    free_disk_gb = round(shutil.disk_usage(WORKING)[2] / (1024**3), 2)

    env_info = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "free_disk_gb": free_disk_gb,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "accumulate_steps": ACCUMULATE_STEPS,
        "effective_batch_size": TOTAL_BATCH_SIZE,
    }
    log("=" * 60)
    for k, v in env_info.items():
        log(f"  {k:20s}: {v}")
    log("=" * 60)
    return env_info


def setup_code_repo() -> Path:
    inputs = Path("/kaggle/input")
    extracted = [p for p in sorted(inputs.rglob("code_bundle")) if p.is_dir() and (p / "pyproject.toml").is_file()]
    if extracted:
        log(f"Found code_bundle directory at {extracted[0]}")
        shutil.copytree(extracted[0], REPO_COPY, dirs_exist_ok=True)
        return REPO_COPY

    bundles = sorted(inputs.rglob("code_bundle.zip"))
    if not bundles:
        raise FileNotFoundError("Code bundle was not found in Kaggle inputs")
    log(f"Extracting {bundles[0]}...")
    REPO_COPY.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundles[0], "r") as archive:
        archive.extractall(REPO_COPY)
    return REPO_COPY


def install_dependencies() -> None:
    log("Installing dependencies...")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "iterative-stratification==0.1.9",
            "PyYAML>=6.0.2",
            "Pillow>=10.0",
            "pycocotools>=2.0.7",
            "tensorboard>=2.15",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Peterande/D-FINE.git", str(DFINE_REPO)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(DFINE_REPO / "requirements.txt")],
        check=True,
    )


def patch_gradient_accumulation(source: str) -> str:
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
""": """            loss_to_backprop = loss / accumulate_steps
            scaler.scale(loss_to_backprop).backward()

            if should_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
""",
        """            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()
""": """            loss_to_backprop = loss / accumulate_steps
            loss_to_backprop.backward()

            if should_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()
""",
        """        if lr_warmup_scheduler:
            lr_warmup_scheduler.step()
""": """        if lr_warmup_scheduler and should_step:
            lr_warmup_scheduler.step()
""",
        """        if ema:
            ema.update(model)
""": """        if ema and should_step:
            ema.update(model)
""",
    }
    for old, new in replacements.items():
        if old in source:
            source = source.replace(old, new, 1)
    return source


def patch_dfine_engine() -> None:
    engine_file = DFINE_REPO / "src" / "solver" / "det_engine.py"
    content = engine_file.read_text(encoding="utf-8")
    patched = patch_gradient_accumulation(content)
    engine_file.write_text(patched, encoding="utf-8")
    log(f"Enabled gradient accumulation: micro={MICRO_BATCH_SIZE}, accumulate={ACCUMULATE_STEPS}")


def patch_dfine_with_frequency_module() -> None:
    """Inject FrequencyAwareFusionModule and DFINEModelWithFrequency into D-FINE."""
    log("Injecting FrequencyAwareFusionModule into D-FINE architecture...")
    dfine_py = DFINE_REPO / "src" / "zoo" / "dfine" / "dfine.py"
    content = dfine_py.read_text(encoding="utf-8")

    freq_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAwareFusionModule(nn.Module):
    def __init__(self, channels=384):
        super().__init__()
        self.channels = channels
        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        low = self.low_pass(x)
        high = x - low
        gate_input = torch.cat([x, high], dim=1)
        g = self.gate(gate_input)
        return x + self.alpha * (g * high)

@register()
class DFINEModelWithFrequency(DFINE):
    def __init__(self, backbone, encoder, decoder):
        super().__init__(backbone, encoder, decoder)
        self.freq_p3 = FrequencyAwareFusionModule(channels=384)

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        p3, p4, p5 = feats[0], feats[1], feats[2]
        refined_p3 = self.freq_p3(p3)
        encoded = self.encoder([refined_p3, p4, p5])
        return self.decoder(encoded, targets)
"""
    if "DFINEModelWithFrequency" not in content:
        dfine_py.write_text(content + "\n" + freq_code, encoding="utf-8")
        log("Successfully registered DFINEModelWithFrequency in D-FINE!")


def find_dataset_root() -> Path:
    candidates = [
        p for p in sorted(Path("/kaggle/input").rglob("data.yaml"))
        if "dataset_processed" not in p.parts and "dataset_roi" not in p.parts
    ]
    if not candidates:
        raise FileNotFoundError("Raw dataset root containing data.yaml was not found")
    log(f"Found dataset root at {candidates[0].parent}")
    return candidates[0].parent


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
        names = {i: name for i, name in enumerate(names)}

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


def find_checkpoint(filename: str) -> Path | None:
    candidates = sorted(Path("/kaggle/input").rglob(filename))
    if candidates:
        log(f"Found checkpoint {filename} at {candidates[0]}")
        return candidates[0]
    return None


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    env_info = inspect_environment()
    repo_root = setup_code_repo()
    install_dependencies()
    patch_dfine_engine()
    patch_dfine_with_frequency_module()

    sys.path.insert(0, str(repo_root / "src"))

    import yaml
    from scale_error.training.trainer_dfine import (
        _generate_dataset_config,
        _generate_experiment_config,
        _yolo_to_coco_json,
    )

    dataset_root = find_dataset_root()
    runtime_yaml = build_runtime_dataset(repo_root, dataset_root)
    cfg = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
    class_names = [cfg["names"][i] for i in sorted(cfg["names"].keys())]

    coco_dir = ARTIFACTS / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    train_json = coco_dir / "train.json"
    val_json = coco_dir / "val.json"

    _yolo_to_coco_json(Path(cfg["train"]), Path(cfg["train"]).parent / "labels", train_json, class_names)
    _yolo_to_coco_json(Path(cfg["val"]), Path(cfg["val"]).parent / "labels", val_json, class_names)

    dataset_cfg_path = DFINE_REPO / "configs" / "dataset" / "glass_defect_frequency.yml"
    _generate_dataset_config(Path(cfg["train"]), train_json, Path(cfg["val"]), val_json, len(class_names), dataset_cfg_path)

    # Generate Frequency Experiment Config
    exp_cfg_path = _generate_experiment_config(
        DFINE_REPO,
        EXP_DIR,
        epochs=SCREENING_EPOCHS,
        eval_size=960,
        batch_size=MICRO_BATCH_SIZE,
        dataset_config_name="glass_defect_frequency.yml",
        config_name="train_dfine_m_frequency.yml",
    )

    # Patch YAML config to use DFINEModelWithFrequency
    yml_text = exp_cfg_path.read_text(encoding="utf-8")
    yml_text = yml_text.replace("DFINE:\n  backbone: HGNetv2", "DFINE:\n  type: DFINEModelWithFrequency\n  backbone: HGNetv2")
    exp_cfg_path.write_text(yml_text, encoding="utf-8")
    shutil.copy2(exp_cfg_path, EXP_DIR / "resolved_config.yaml")
    log("Resolved config written to EXP_DIR / resolved_config.yaml")

    # Locate pretrained weights (resume47 baseline for identical control)
    pretrained_weights = find_checkpoint("dfine_m_coco.pth")
    if not pretrained_weights:
        pretrained_weights = find_checkpoint("resume47_best_stg1.pth")

    # Launch Training
    log(f"Launching Stage B Screening Run ({SCREENING_EPOCHS} epochs) with Frequency-Aware Refinement...")
    train_cmd = [
        sys.executable, str(DFINE_REPO / "train.py"),
        "-c", str(exp_cfg_path),
        "--use-amp",
        "--seed", "42",
    ]
    if pretrained_weights:
        train_cmd.extend(["-t", str(pretrained_weights)])

    log(f"Executing: {' '.join(train_cmd)}")
    with open(EXP_DIR / "training_stdout.log", "w", encoding="utf-8") as out_fp:
        proc = subprocess.Popen(train_cmd, cwd=str(DFINE_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            out_fp.write(line)
            out_fp.flush()
        proc.wait()

    if proc.returncode != 0:
        log(f"Training failed with return code {proc.returncode}")
        sys.exit(proc.returncode)

    # Save checkpoints to standard filenames
    src_best = EXP_DIR / "best_stg1.pth"
    src_last = EXP_DIR / "last.pth"
    if src_best.exists():
        shutil.copy2(src_best, EXP_DIR / "best.pth")
        shutil.copy2(src_best, WORKING / "frequency_screening_best.pth")
    if src_last.exists():
        shutil.copy2(src_last, WORKING / "frequency_screening_last.pth")

    # Parse and save training_log.csv
    log_txt = EXP_DIR / "log.txt"
    if log_txt.exists():
        shutil.copy2(log_txt, WORKING / "frequency_screening_log.txt")
        csv_rows = []
        for line in log_txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    bbox = d.get("test_coco_eval_bbox", [])
                    csv_rows.append({
                        "epoch": d.get("epoch"),
                        "train_loss": round(d.get("train_loss", 0.0), 4),
                        "val_mAP50_95": round(bbox[0] * 100, 2) if len(bbox) > 0 else 0.0,
                        "val_mAP50": round(bbox[1] * 100, 2) if len(bbox) > 1 else 0.0,
                        "val_mAP75": round(bbox[2] * 100, 2) if len(bbox) > 2 else 0.0,
                        "val_AP_small": round(bbox[3] * 100, 2) if len(bbox) > 3 else 0.0,
                        "val_AP_medium": round(bbox[4] * 100, 2) if len(bbox) > 4 else 0.0,
                        "val_AP_large": round(bbox[5] * 100, 2) if len(bbox) > 5 else 0.0,
                    })
                except Exception:
                    pass
        if csv_rows:
            csv_path = EXP_DIR / "training_log.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            shutil.copy2(csv_path, WORKING / "frequency_screening_training_log.csv")
            log(f"Saved {len(csv_rows)} validation epochs to {csv_path}")

    log("Screening run completed successfully!")


if __name__ == "__main__":
    main()
