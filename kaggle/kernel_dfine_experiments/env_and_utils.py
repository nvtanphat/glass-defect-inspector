"""Environment discovery and experiment management utilities for Kaggle.

Ensures strict compliance with:
- Dynamic GPU/CUDA/PyTorch/Disk detection (never hardcode GPU)
- Output directory isolation under /kaggle/working/glass-defect-experiments/
- Systematic saving of best.pth, last.pth, metrics.json, resolved_config.yaml, training_log.csv
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def inspect_environment(base_dir: Path | str = "/kaggle/working") -> dict[str, Any]:
    """Inspect and report the running hardware and software environment."""
    base_dir = Path(base_dir)
    env_info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "gpu_name": None,
        "total_vram_gb": 0.0,
        "free_disk_gb": 0.0,
        "suggested_batch_size": 4,
    }

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024**3), 2)
        env_info["gpu_name"] = gpu_name
        env_info["total_vram_gb"] = vram_gb

        # Dynamic batch size heuristic:
        # T4 (15-16 GB VRAM) -> batch 4 (micro-batch 4, accumulate 2 = effective 8)
        # P100 (16 GB VRAM) -> batch 4
        # A100 (40 GB VRAM) -> batch 8
        if vram_gb >= 35.0:
            env_info["suggested_batch_size"] = 8
        else:
            env_info["suggested_batch_size"] = 4

    try:
        total, used, free = shutil.disk_usage(base_dir if base_dir.exists() else ".")
        env_info["free_disk_gb"] = round(free / (1024**3), 2)
    except Exception:
        env_info["free_disk_gb"] = 0.0

    print("=" * 65)
    print(" KAGGLE RUNTIME ENVIRONMENT INSPECTION")
    print("=" * 65)
    print(f" Python  : {env_info['python_version']}")
    print(f" PyTorch : {env_info['pytorch_version']}")
    print(f" CUDA    : {env_info['cuda_version']} (Available: {env_info['cuda_available']})")
    print(f" GPU     : {env_info['gpu_name']} ({env_info['total_vram_gb']} GB VRAM)")
    print(f" Free Disk: {env_info['free_disk_gb']} GB")
    print(f" Suggested Micro-Batch Size: {env_info['suggested_batch_size']}")
    print("=" * 65)

    return env_info


def setup_experiment_dir(experiment_name: str, base_working_dir: Path | str = "/kaggle/working") -> Path:
    """Setup structured experiment folder inside /kaggle/working/glass-defect-experiments/

    Experiments:
        - baseline
        - p2
        - frequency
        - p2_frequency
        - balanced
        - reports
    """
    root = Path(base_working_dir) / "glass-defect-experiments"
    exp_dir = root / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    return exp_dir


class ExperimentLogger:
    """Logs training metrics to training_log.csv and saves resolved configs and metrics."""

    def __init__(self, exp_dir: Path, config: dict[str, Any]):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Save resolved config immediately
        cfg_path = self.exp_dir / "resolved_config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

        self.csv_path = self.exp_dir / "training_log.csv"
        self.csv_file = None
        self.csv_writer = None
        self.headers_written = False

    def log_epoch(self, epoch_metrics: dict[str, Any]):
        """Append epoch results to training_log.csv."""
        if not self.headers_written:
            is_new = not self.csv_path.exists()
            self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(epoch_metrics.keys()))
            if is_new:
                self.csv_writer.writeheader()
            self.headers_written = True

        self.csv_writer.writerow(epoch_metrics)
        self.csv_file.flush()

    def save_final_metrics(self, metrics: dict[str, Any]):
        """Save final metrics.json."""
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Logger] Saved final metrics to {metrics_path}")

    def close(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
