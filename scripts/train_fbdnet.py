"""Fine-tune FBDNet-M locally: E0 baseline and E1 scale-aware NWD."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DFINE_DIR = ROOT / "results" / "D-FINE"

DEFAULT_WEIGHTS = ROOT / "weights" / "dfine_m_fbdnet_best.pth"


def log(msg: str) -> None:
    print(f"\n[FBDNet-NWD] {msg}\n", flush=True)


def locate_checkpoint(custom_path: str | None = None) -> Path:
    """Find the best FBDNet-M checkpoint."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {custom_path}")

    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    raise FileNotFoundError("Provide --weights pointing to an existing FBDNet-M checkpoint")


def run_training(
    exp: str,
    weights_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = "cuda",
) -> Path:
    """Launch training using D-FINE train.py with --tuning flag."""
    if exp == "e0":
        config_name = "fbdnet_m_e0_baseline_20ep.yml"
        exp_name = "E0_baseline"
    elif exp == "e1":
        config_name = "fbdnet_m_e1_sanwd_20ep.yml"
        exp_name = "E1_sanwd"
    else:
        raise ValueError(f"Unknown experiment: {exp}")

    config_path = DFINE_DIR / "configs" / "dfine" / "custom" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    log(f"Starting {exp_name} fine-tuning ({epochs} epochs) with config: {config_path.name}")
    log(f"Tuning weights from: {weights_path}")
    log(f"Output directory: {output_dir}")

    train_script = DFINE_DIR / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"Missing D-FINE training runtime: {train_script}")

    # Check available GPUs
    try:
        import torch

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        num_gpus = 0

    cmd = [
        sys.executable,
        str(train_script),
        "--device",
        device,
        "-c",
        str(config_path),
        "-t",
        str(weights_path),  # --tuning loads model weights, fresh optimizer & scheduler
        "--output-dir",
        str(output_dir),
        "-u",
        f"epochs={epochs}",
        f"train_dataloader.total_batch_size={batch_size}",
        f"train_dataloader.num_workers={num_workers}",
    ]

    # If running with 2 GPUs locally, use torchrun for DDP
    if device == "cuda" and num_gpus > 1:
        log(f"Detected {num_gpus} GPUs, launching with torchrun DDP...")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            str(train_script),
            "-c",
            str(config_path),
            "-t",
            str(weights_path),
            "--output-dir",
            str(output_dir),
            "-u",
            f"epochs={epochs}",
            f"train_dataloader.total_batch_size={batch_size}",
            f"train_dataloader.num_workers={num_workers}",
        ]

    log(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    log(f"{exp_name} training completed successfully.")
    return output_dir / "log.txt"


def main():
    parser = argparse.ArgumentParser(description="Run FBDNet-M E0/E1 Experiments on local GPUs")
    parser.add_argument(
        "--exp",
        type=str,
        default="e1",
        choices=["e0", "e1", "both"],
        help="Experiment to run: e0 (baseline), e1 (+Scale-Aware NWD), or both",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained dfine_m_fbdnet_best.pth",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of fine-tuning epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Total batch size (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory",
    )
    parser.add_argument("--device", default="cuda", help="Training device: cuda or cpu")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    weights_path = locate_checkpoint(args.weights)

    base_out = Path(args.output_dir) if args.output_dir else (ROOT / "runs" / "fbdnet_nwd")

    e0_log = None
    e1_log = None

    if args.exp in ["e0", "both"]:
        out_e0 = base_out / "output_fbdnet_m_e0_baseline"
        out_e0.mkdir(parents=True, exist_ok=True)
        e0_log = run_training(
            "e0",
            weights_path=weights_path,
            output_dir=out_e0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )

    if args.exp in ["e1", "both"]:
        out_e1 = base_out / "output_fbdnet_m_e1_sanwd"
        out_e1.mkdir(parents=True, exist_ok=True)
        e1_log = run_training(
            "e1",
            weights_path=weights_path,
            output_dir=out_e1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )

    if args.exp == "both" and e0_log and e1_log and e0_log.exists() and e1_log.exists():
        log("Generating comparison report between E0 and E1...")
        compare_script = ROOT / "scripts" / "compare_fbdnet_e0_e1.py"
        report_file = base_out / "comparison_e0_vs_e1.md"
        subprocess.run(
            [
                sys.executable,
                str(compare_script),
                "--e0",
                str(e0_log),
                "--e1",
                str(e1_log),
                "--output",
                str(report_file),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
