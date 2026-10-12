"""Comparison script for Experiment E0 (Baseline) vs Experiment E1 (Scale-Aware NWD).

Compares:
- Overall mAP (mAP 50:95, AP50, AP75)
- Scale-stratified metrics (AP_small, AP_medium, AP_large)
- Per-class AP across the 5 glass defect categories:
    1. DiVat
    2. DiVatLoiLom
    3. LoiChi
    4. LoiNhua
    5. LoiTray
- Inference throughput and latency (FPS, ms/img)
- Absolute and relative percentage delta between E1 and E0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

CLASS_NAMES = [
    "DiVat",
    "DiVatLoiLom",
    "LoiChi",
    "LoiNhua",
    "LoiTray",
]

COCO_METRIC_NAMES = [
    "mAP50:95",
    "mAP50",
    "mAP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def parse_log_file(log_path: Path) -> Dict[str, Any]:
    """Parse D-FINE log.txt file to extract best or final evaluation metrics."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    best_record = None
    best_map = -1.0
    last_record = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
                if "test_coco_eval_bbox" in record:
                    last_record = record
                    bbox_metrics = record["test_coco_eval_bbox"]
                    if isinstance(bbox_metrics, list) and len(bbox_metrics) > 0:
                        map_val = bbox_metrics[0]
                        if map_val > best_map:
                            best_map = map_val
                            best_record = record
            except Exception:
                continue

    target_record = best_record or last_record
    if target_record is None:
        raise ValueError(f"No valid test_coco_eval_bbox records found in {log_path}")

    return target_record


def parse_metrics_json(json_path: Path) -> Dict[str, Any]:
    """Parse a JSON file containing evaluation metrics."""
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_data(path_str: str) -> Dict[str, Any]:
    """Load experiment metrics from either a JSON file or log.txt."""
    path = Path(path_str)
    if path.suffix == ".json":
        data = parse_metrics_json(path)
        # If it has coco_eval array or dict
        if "test_coco_eval_bbox" in data:
            return data
        return data
    else:
        return parse_log_file(path)


def format_delta(v1: Optional[float], v0: Optional[float], pct: bool = True) -> str:
    """Format difference between E1 and E0."""
    if v1 is None or v0 is None:
        return "N/A"
    diff = v1 - v0
    sign = "+" if diff >= 0 else ""
    if pct:
        rel_diff = ((v1 - v0) / v0 * 100.0) if abs(v0) > 1e-6 else 0.0
        return f"{sign}{diff:.2f}% ({sign}{rel_diff:.1f}%)"
    else:
        return f"{sign}{diff:.2f}"


def build_comparison_report(
    e0_data: Dict[str, Any],
    e1_data: Dict[str, Any],
    e0_label: str = "E0: Baseline (Loss gốc)",
    e1_label: str = "E1: + Scale-Aware NWD",
    fps_e0: Optional[float] = None,
    fps_e1: Optional[float] = None,
) -> str:
    """Generate Markdown comparison report."""
    # Extract standard 12 COCO metrics
    m0 = e0_data.get("test_coco_eval_bbox", [])
    m1 = e1_data.get("test_coco_eval_bbox", [])

    def get_val(arr: list, idx: int) -> Optional[float]:
        if isinstance(arr, list) and idx < len(arr):
            val = arr[idx]
            return val * 100.0 if val <= 1.0 else val
        return None

    # COCO bbox indices:
    # 0: mAP50:95, 1: mAP50, 2: mAP75, 3: AP_small, 4: AP_medium, 5: AP_large
    metrics_summary = [
        ("mAP (0.50:0.95)", get_val(m0, 0), get_val(m1, 0)),
        ("AP50", get_val(m0, 1), get_val(m1, 1)),
        ("AP75", get_val(m0, 2), get_val(m1, 2)),
        ("AP_small (<32² px)", get_val(m0, 3), get_val(m1, 3)),
        ("AP_medium (32²-96² px)", get_val(m0, 4), get_val(m1, 4)),
        ("AP_large (>96² px)", get_val(m0, 5), get_val(m1, 5)),
    ]

    report_lines = [
        "# So Sánh Thí Nghiệm FBDNet-M: E0 Baseline vs E1 + Scale-Aware NWD",
        "",
        f"- **E0 Checkpoint:** `dfine_m_fbdnet_best.pth` (epoch 89, mAP 42.42%) -> Fine-tune 20 ep (Loss gốc)",
        f"- **E1 Checkpoint:** `dfine_m_fbdnet_best.pth` -> Fine-tune 20 ep (+ Scale-Aware NWD, λ=0.2, C=16px)",
        "",
        "## 1. Tổng Hợp Chỉ Số COCO Detection",
        "",
        f"| Metric | {e0_label} | {e1_label} | Chênh lệch (Δ) |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for name, v0, v1 in metrics_summary:
        v0_str = f"{v0:.2f}%" if v0 is not None else "N/A"
        v1_str = f"{v1:.2f}%" if v1 is not None else "N/A"
        delta_str = format_delta(v1, v0)
        report_lines.append(f"| **{name}** | {v0_str} | {v1_str} | **{delta_str}** |")

    # Per-class AP if present
    classes_e0 = e0_data.get("per_class_ap", {})
    classes_e1 = e1_data.get("per_class_ap", {})
    if classes_e0 or classes_e1:
        report_lines.extend([
            "",
            "## 2. AP Từng Lớp Khuyết Tật (5 Classes)",
            "",
            f"| Class | {e0_label} | {e1_label} | Chênh lệch (Δ) |",
            "| :--- | :---: | :---: | :---: |",
        ])
        for cname in CLASS_NAMES:
            c0 = classes_e0.get(cname)
            c1 = classes_e1.get(cname)
            c0_pct = c0 * 100.0 if (c0 is not None and c0 <= 1.0) else c0
            c1_pct = c1 * 100.0 if (c1 is not None and c1 <= 1.0) else c1
            c0_str = f"{c0_pct:.2f}%" if c0_pct is not None else "N/A"
            c1_str = f"{c1_pct:.2f}%" if c1_pct is not None else "N/A"
            delta_str = format_delta(c1_pct, c0_pct)
            report_lines.append(f"| **{cname}** | {c0_str} | {c1_str} | **{delta_str}** |")

    # Throughput comparison
    if fps_e0 is not None or fps_e1 is not None:
        report_lines.extend([
            "",
            "## 3. Tốc Độ Suy Luận (Inference Throughput & Latency)",
            "",
            f"| Cấu hình | FPS (img/s) | Latency (ms/img) |",
            "| :--- | :---: | :---: |",
        ])
        if fps_e0 is not None:
            lat0 = 1000.0 / fps_e0 if fps_e0 > 0 else 0
            report_lines.append(f"| **{e0_label}** | {fps_e0:.1f} FPS | {lat0:.2f} ms |")
        if fps_e1 is not None:
            lat1 = 1000.0 / fps_e1 if fps_e1 > 0 else 0
            report_lines.append(f"| **{e1_label}** | {fps_e1:.1f} FPS | {lat1:.2f} ms |")

    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Compare FBDNet-M E0 Baseline vs E1 Scale-Aware NWD")
    parser.add_argument("--e0", type=str, required=True, help="Path to E0 log.txt or metrics.json")
    parser.add_argument("--e1", type=str, required=True, help="Path to E1 log.txt or metrics.json")
    parser.add_argument("--fps-e0", type=float, default=None, help="FPS for E0 model")
    parser.add_argument("--fps-e1", type=float, default=None, help="FPS for E1 model")
    parser.add_argument("--output", type=str, default=None, help="Output markdown report path")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    e0_data = load_experiment_data(args.e0)
    e1_data = load_experiment_data(args.e1)

    report = build_comparison_report(
        e0_data,
        e1_data,
        fps_e0=args.fps_e0,
        fps_e1=args.fps_e1,
    )
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n[Saved report to {out_path}]")


if __name__ == "__main__":
    main()
