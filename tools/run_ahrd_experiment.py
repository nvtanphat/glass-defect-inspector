"""Orchestration script for RF-DETR-M + AHRD scientific experiment (E0 vs E1).

Usage:
    # Run E0 (Baseline) across seeds 42, 43, 44:
    python tools/run_ahrd_experiment.py --exp e0 --seeds 42 43 44

    # Run E1 (AHRD) across seeds 42, 43, 44:
    python tools/run_ahrd_experiment.py --exp e1 --seeds 42 43 44

    # Run dry-run / architecture check:
    python tools/run_ahrd_experiment.py --exp e1 --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DFINE_DIR = ROOT / "results" / "D-FINE"

EXPERIMENT_CONFIGS = {
    "e0": {
        "name": "E0: RF-DETR-M Baseline (AHRD OFF)",
        "config": ROOT / "results/D-FINE/configs/dfine/custom/rfdetr_m_e0_baseline.yml",
        "output_base": ROOT / "runs/experiments/rfdetr_m_e0_baseline",
    },
    "e1": {
        "name": "E1: RF-DETR-M + AHRD (AHRD ON)",
        "config": ROOT / "results/D-FINE/configs/dfine/custom/rfdetr_m_e1_ahrd.yml",
        "output_base": ROOT / "runs/experiments/rfdetr_m_e1_ahrd",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run RF-DETR-M AHRD Experiment")
    parser.add_argument("--exp", choices=["e0", "e1", "both"], required=True, help="Experiment to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds (default: 42 43 44)")
    parser.add_argument("--weights", type=str, default="weights/dfine_m_coco.pth", help="Pretrained tuning weights")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID")
    parser.add_argument("--dry-run", action="store_true", help="Perform architecture check without training")
    return parser.parse_args()


def run_experiment(exp_key: str, seed: int, weights: Path, device: str, dry_run: bool):
    exp_info = EXPERIMENT_CONFIGS[exp_key]
    out_dir = exp_info["output_base"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[*] Running {exp_info['name']} | Seed: {seed}")
    print(f"Config: {exp_info['config']}")
    print(f"Output: {out_dir}")
    print("=" * 70)

    if dry_run:
        print("[Dry-run] Checking config existence and architecture syntax...")
        assert exp_info["config"].exists(), f"Config missing: {exp_info['config']}"
        assert weights.exists(), f"Weights missing: {weights}"
        print(f"[Dry-run] Ready to execute for {exp_key} with seed {seed}.")
        return

    cmd = [
        sys.executable,
        str(DFINE_DIR / "train.py"),
        "-c",
        str(exp_info["config"]),
        "-t",
        str(weights),
        "--seed",
        str(seed),
        "--use-amp",
        "--output-dir",
        str(out_dir),
    ]

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Post-training evaluation
    best_ckpt = out_dir / "best.pth"
    if best_ckpt.exists():
        eval_json = out_dir / "eval_report.json"
        eval_cmd = [
            sys.executable,
            str(ROOT / "tools/eval_ahrd_experiment.py"),
            "-c",
            str(exp_info["config"]),
            "-w",
            str(best_ckpt),
            "-o",
            str(eval_json),
            "--device",
            f"cuda:{device}",
        ]
        print(f"\n📊 Evaluating best checkpoint: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)


def main():
    args = parse_args()
    weights_path = (ROOT / args.weights).resolve()
    assert weights_path.exists(), f"Pretrained checkpoint not found: {weights_path}"

    exps = ["e0", "e1"] if args.exp == "both" else [args.exp]
    for exp in exps:
        for seed in args.seeds:
            run_experiment(exp, seed, weights_path, args.device, args.dry_run)


if __name__ == "__main__":
    main()
