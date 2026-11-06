"""Lớp bọc đánh giá kiểm định mô hình D-FINE-M / FBDNet-M (D-FINE evaluation wrapper).

Đánh giá checkpoint D-FINE theo định dạng chuẩn COCO, cung cấp giao diện nhất quán
với bộ đánh giá dựa trên Ultralytics.

Cách sử dụng dòng lệnh:
    python -m scale_error.evaluation.evaluator_dfine \
        --weights results/best.pth \
        --dfine-repo results/D-FINE \
        --data data/processed_roi/runtime_data.yaml \
        --output artifacts/evaluation/metrics_dfine.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def evaluate_dfine(
    weights: str | Path,
    dfine_repo: str | Path,
    data_yaml: str | Path,
    output_json: str | Path,
    eval_size: int = 960,
    device: str = "0",
) -> dict:
    """Đánh giá mô hình D-FINE trên tập dữ liệu kiểm thử test split.

    Tham số:
        weights: Đường dẫn file trọng số đã huấn luyện (best.pth).
        dfine_repo: Đường dẫn thư mục mã nguồn D-FINE.
        data_yaml: Đường dẫn file cấu hình runtime_data.yaml.
        output_json: Đường dẫn file JSON ghi kết quả đánh giá.
        eval_size: Kích thước cạnh ảnh đánh giá (mặc định: 960).
        device: Chỉ số GPU CUDA.
    """
    dfine_repo = Path(dfine_repo).resolve()
    weights = Path(weights).resolve()
    output_json = Path(output_json)

    # 1. Chuẩn bị dữ liệu kiểm thử dạng COCO JSON
    data_cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = data_cfg["names"]
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=int)]
    else:
        class_names = list(names)

    test_images = Path(data_cfg.get("test", data_cfg["val"]))
    test_labels_candidates = [
        test_images.parent.parent / "labels" / "test",
        test_images.parent.parent / "labels",
        test_images.with_name("labels"),
    ]
    test_labels = test_images  # dự phòng
    for candidate in test_labels_candidates:
        if candidate.exists():
            test_labels = candidate
            break

    # Chuyển đổi nhãn YOLO sang định dạng COCO JSON
    from scale_error.training.trainer_dfine import (
        _generate_dataset_config,
        _generate_experiment_config,
        _yolo_to_coco_json,
    )

    eval_dir = output_json.parent / "dfine_eval_data"
    eval_dir.mkdir(parents=True, exist_ok=True)
    test_json = eval_dir / "test.json"
    _yolo_to_coco_json(test_images, test_labels, test_json, class_names)

    # 2. Sinh cấu hình đánh giá trỏ chính xác vào tập test này
    dataset_config_name = "scale_error_eval_generated.yml"
    dataset_config = dfine_repo / "configs" / "dataset" / dataset_config_name
    _generate_dataset_config(
        test_images,
        test_json,
        test_images,
        test_json,
        len(class_names),
        dataset_config,
    )
    model_config = _generate_experiment_config(
        dfine_repo,
        output_json.parent / "dfine_eval_output",
        epochs=100,
        eval_size=eval_size,
        batch_size=8,
        dataset_config_name=dataset_config_name,
        config_name="dfine_hgnetv2_m_scale_error_eval_generated.yml",
    )

    # 3. Chạy lệnh đánh giá thông qua train.py --test-only của D-FINE
    print(f"[D-FINE] Bắt đầu đánh giá checkpoint: {weights}")
    cmd = [
        sys.executable,
        str(dfine_repo / "train.py"),
        "-c",
        str(model_config),
        "-r",
        str(weights),
        "--test-only",
        "--device",
        f"cuda:{device}",
    ]

    print(f"[D-FINE] Lệnh thực thi: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(dfine_repo))
    if result.returncode != 0:
        raise RuntimeError(
            f"Đánh giá D-FINE thất bại với mã lỗi {result.returncode}:\n{result.stderr[-4000:]}"
        )

    # 4. Phân tích kết quả đầu ra COCO evaluation
    metrics = _parse_coco_eval_output(result.stdout)
    if "mAP50_95" not in metrics:
        raise RuntimeError(
            "Đánh giá D-FINE hoàn tất nhưng không trích xuất được metrics COCO. "
            f"Đoạn cuối log:\n{result.stdout[-4000:]}"
        )
    metrics["model"] = "D-FINE-M"
    metrics["weights"] = str(weights)
    metrics["eval_size"] = eval_size

    # 5. Lưu kết quả ra file JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[D-FINE] Đã lưu kết quả metrics tại -> {output_json}")

    return metrics


def _parse_coco_eval_output(stdout: str) -> dict:
    """Phân tích các dòng log kết quả COCO evaluation từ stdout của D-FINE."""
    metrics = {}

    for line in stdout.splitlines():
        line = line.strip()
        if "Average Precision" in line and "IoU=0.50:0.95" in line and "area=   all" in line:
            try:
                metrics["mAP50_95"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass
        elif "Average Precision" in line and "IoU=0.50 " in line and "area=   all" in line:
            try:
                metrics["mAP50"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass
        elif "Average Precision" in line and "IoU=0.75" in line and "area=   all" in line:
            try:
                metrics["mAP75"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass
        elif "Average Precision" in line and "area= small" in line:
            try:
                metrics["mAP_small"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass
        elif "Average Precision" in line and "area=medium" in line:
            try:
                metrics["mAP_medium"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass
        elif "Average Precision" in line and "area= large" in line:
            try:
                metrics["mAP_large"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá mô hình D-FINE-M")
    parser.add_argument("--weights", required=True, help="Đường dẫn file best.pth")
    parser.add_argument("--dfine-repo", required=True, help="Đường dẫn mã nguồn D-FINE")
    parser.add_argument("--data", required=True, help="Đường dẫn file runtime_data.yaml")
    parser.add_argument("--output", default="artifacts/evaluation/metrics_dfine.json")
    parser.add_argument("--eval-size", type=int, default=960)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    evaluate_dfine(
        args.weights,
        args.dfine_repo,
        args.data,
        args.output,
        eval_size=args.eval_size,
        device=args.device,
    )
