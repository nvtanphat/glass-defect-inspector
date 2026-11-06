"""Giao diện dòng lệnh (CLI) trung tâm của hệ thống phát hiện khuyết tật bề mặt kính scale-error.

Cung cấp các lệnh:
- prepare: Tải dữ liệu, kiểm tra định dạng và phân chia tập train/val/test độc lập.
- train: Huấn luyện các mô hình Ultralytics (YOLO26m, YOLO11m, RT-DETR-L).
- train-dfine: Huấn luyện mô hình D-FINE-M / FBDNet-M trên framework D-FINE.
- evaluate: Đánh giá mô hình Ultralytics theo chuẩn COCO mAP.
- evaluate-dfine: Đánh giá mô hình D-FINE-M / FBDNet-M.
- predict: Suy luận phát hiện khuyết tật trên ảnh mới.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scale_error.data.roboflow_io import download_dataset
from scale_error.data.split import split_original_valid
from scale_error.data.validate import validate_dataset
from scale_error.evaluation.evaluator import evaluate
from scale_error.inference.predictor import predict
from scale_error.training.trainer import train


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng bộ phân tích tham số dòng lệnh cho toàn bộ pipeline của dự án."""
    p = argparse.ArgumentParser(
        prog="scale-error",
        description="Bộ công cụ dòng lệnh phát hiện khuyết tật chai lọ và bề mặt kính công nghiệp.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1. Lệnh chuẩn bị dữ liệu (prepare)
    prepare = sub.add_parser("prepare", help="Tải và phân chia tập dữ liệu huấn luyện/kiểm thử")
    prepare.add_argument("--output", required=True, help="Thư mục đích lưu dữ liệu tải về")
    prepare.add_argument("--workspace", default="datacpn", help="Workspace trên Roboflow")
    prepare.add_argument("--project", default="scale_error_dc-orqpx", help="Tên project Roboflow")
    prepare.add_argument("--version", type=int, default=26, help="Phiên bản dataset trên Roboflow")
    prepare.add_argument("--split-output", default="data/processed", help="Thư mục lưu dữ liệu sau phân chia")

    # 2. Lệnh huấn luyện mô hình Ultralytics (train)
    tr = sub.add_parser("train", help="Huấn luyện YOLO26m / YOLO11m / RT-DETR-L (Ultralytics)")
    tr.add_argument("--data", required=True, help="Đường dẫn file data.yaml")
    tr.add_argument("--config", required=True, help="Đường dẫn file cấu hình huấn luyện (.yaml)")
    tr.add_argument("--project-dir", default="artifacts/runs", help="Thư mục lưu kết quả huấn luyện")
    tr.add_argument(
        "--dataset-report",
        default=None,
        help="Đường dẫn file dataset_report.json để tính trọng số lớp mất cân bằng",
    )

    # 3. Lệnh huấn luyện mô hình D-FINE-M / FBDNet-M (train-dfine)
    tr_dfine = sub.add_parser("train-dfine", help="Huấn luyện D-FINE-M / FBDNet-M")
    tr_dfine.add_argument("--data", required=True, help="Đường dẫn file runtime_data.yaml")
    tr_dfine.add_argument("--dfine-repo", required=True, help="Đường dẫn mã nguồn D-FINE")
    tr_dfine.add_argument(
        "--weights",
        required=True,
        help="Đường dẫn trọng số tiền huấn luyện (ví dụ: dfine_m_coco.pth)",
    )
    tr_dfine.add_argument("--output-dir", default="artifacts/runs/dfine_m", help="Thư mục lưu checkpoint")
    tr_dfine.add_argument("--epochs", type=int, default=100, help="Số epoch huấn luyện")
    tr_dfine.add_argument("--eval-size", type=int, default=960, help="Kích thước ảnh đánh giá")
    tr_dfine.add_argument("--batch-size", type=int, default=4, help="Kích thước batch size")
    tr_dfine.add_argument("--resume", default=None, help="Đường dẫn checkpoint để tiếp tục huấn luyện")

    # 4. Lệnh đánh giá mô hình Ultralytics (evaluate)
    ev = sub.add_parser("evaluate", help="Đánh giá mô hình YOLO26m / YOLO11m / RT-DETR-L")
    ev.add_argument("--weights", required=True, help="Đường dẫn file trọng số (.pt)")
    ev.add_argument("--data", required=True, help="Đường dẫn file data.yaml")
    ev.add_argument("--output", default="artifacts/evaluation/metrics.json", help="File JSON xuất kết quả")
    ev.add_argument("--nms-free", action="store_true", help="Bỏ qua NMS đối với các mô hình NMS-free")

    # 5. Lệnh đánh giá mô hình D-FINE-M (evaluate-dfine)
    ev_dfine = sub.add_parser("evaluate-dfine", help="Đánh giá mô hình D-FINE-M / FBDNet-M")
    ev_dfine.add_argument("--weights", required=True, help="Đường dẫn file trọng số best.pth")
    ev_dfine.add_argument("--dfine-repo", required=True, help="Đường dẫn mã nguồn D-FINE")
    ev_dfine.add_argument("--data", required=True, help="Đường dẫn file runtime_data.yaml")
    ev_dfine.add_argument("--output", default="artifacts/evaluation/metrics_dfine.json", help="File xuất kết quả")
    ev_dfine.add_argument("--eval-size", type=int, default=960, help="Kích thước ảnh đánh giá")
    ev_dfine.add_argument("--device", default="0", help="Chỉ số thiết bị GPU (0, 1, ... hoặc cpu)")

    # 6. Lệnh dự đoán suy luận (predict)
    pr = sub.add_parser("predict", help="Suy luận phát hiện khuyết tật trên ảnh mới")
    pr.add_argument("--weights", required=True, help="Đường dẫn file trọng số mô hình")
    pr.add_argument("--source", required=True, help="Đường dẫn ảnh nguồn hoặc thư mục ảnh")
    pr.add_argument("--output", default="artifacts/predictions", help="Thư mục lưu ảnh đã vẽ bounding box")
    pr.add_argument("--nms-free", action="store_true", help="Bỏ qua NMS đối với mô hình NMS-free")

    return p


def main() -> None:
    """Điểm nhập thực thi chính của CLI."""
    args = build_parser().parse_args()

    # Nhánh xử lý tải và phân chia dữ liệu
    if args.cmd == "prepare":
        dataset_root = download_dataset(
            output_dir=args.output,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
        )
        validate_dataset(dataset_root, Path(args.split_output) / "dataset_report.json")
        _, runtime_yaml = split_original_valid(dataset_root, args.split_output)
        print(f"Hoàn thành chuẩn bị dữ liệu. File runtime: {runtime_yaml}")

    # Nhánh huấn luyện Ultralytics YOLO
    elif args.cmd == "train":
        train(args.data, args.config, args.project_dir, dataset_report=args.dataset_report)

    # Nhánh huấn luyện D-FINE-M / FBDNet-M
    elif args.cmd == "train-dfine":
        from scale_error.training.trainer_dfine import train_dfine

        train_dfine(
            args.data,
            args.dfine_repo,
            args.weights,
            args.output_dir,
            epochs=args.epochs,
            eval_size=args.eval_size,
            batch_size=args.batch_size,
            resume_checkpoint=args.resume,
        )

    # Nhánh đánh giá mô hình Ultralytics YOLO
    elif args.cmd == "evaluate":
        metrics = evaluate(args.weights, args.data, nms_free=args.nms_free)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Đã lưu kết quả đánh giá tại: {out}")

    # Nhánh đánh giá mô hình D-FINE-M
    elif args.cmd == "evaluate-dfine":
        from scale_error.evaluation.evaluator_dfine import evaluate_dfine

        metrics = evaluate_dfine(
            args.weights,
            args.dfine_repo,
            args.data,
            eval_size=args.eval_size,
            device=args.device,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Đã lưu kết quả đánh giá D-FINE tại: {out}")

    # Nhánh suy luận dự đoán ảnh mới
    elif args.cmd == "predict":
        predict(args.weights, args.source, args.output, nms_free=args.nms_free)
        print(f"Đã lưu kết quả suy luận tại: {args.output}")


if __name__ == "__main__":
    main()
