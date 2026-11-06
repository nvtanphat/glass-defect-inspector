"""Mô-đun điều phối huấn luyện cho các mô hình Ultralytics YOLO26m / YOLO11m / RT-DETR-L.

Hỗ trợ:
- Gắn hàm mất mát Varifocal Loss (VFL) tùy biến vào hook on_train_start của Ultralytics.
- Tự động áp dụng trọng số lớp (class weights) từ file dataset_report.json để trị mất cân bằng.
- Kiểm tra hợp lệ các chế độ tiền xử lý (raw vs roi_crop).
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR, YOLO
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

from scale_error.config import load_yaml
from scale_error.training.losses import VFLDetectionLoss, compute_class_weights


def _attach_vfl_criterion(class_weights):
    """Xây dựng callback on_train_start để thay thế hàm mất mát mặc định
    bằng Varifocal Loss + trọng số lớp.
    """

    def _callback(trainer) -> None:
        det_model = trainer.model
        if class_weights is not None:
            det_model.class_weights = class_weights.to(next(det_model.parameters()).device)

        existing_criterion = det_model.init_criterion()
        if isinstance(existing_criterion, E2ELoss):
            det_model.criterion = E2ELoss(det_model, loss_fn=VFLDetectionLoss)
        elif isinstance(existing_criterion, v8DetectionLoss):
            det_model.criterion = VFLDetectionLoss(det_model)
        else:
            raise TypeError(
                f"Hàm mất mát không được hỗ trợ {type(existing_criterion).__name__}; "
                "VFLDetectionLoss chỉ bọc được v8DetectionLoss hoặc E2ELoss."
            )

    return _callback


def train(
    data_yaml: str | Path,
    config_path: str | Path,
    project_dir: str | Path,
    run_name: str | None = None,
    dataset_report: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
):
    """Thực thi huấn luyện mô hình phát hiện khuyết tật.

    Tham số:
        data_yaml: Đường dẫn file cấu hình runtime_data.yaml.
        config_path: Đường dẫn file cấu hình huấn luyện (.yaml).
        project_dir: Thư mục lưu kết quả runs/.
        run_name: Tên thư mục con lưu checkpoint (tùy chọn).
        dataset_report: Đường dẫn file dataset_report.json để tính class weights.
        resume_checkpoint: Đường dẫn checkpoint để tiếp tục huấn luyện dang dở.
    """
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)
    model_name = str(cfg.pop("model"))
    preprocessing = str(cfg.pop("preprocessing", "raw"))
    roi_config = cfg.pop("roi_config", None)
    use_vfl = bool(cfg.pop("use_vfl", False))
    use_class_weights = bool(cfg.pop("use_class_weights", False))
    project_dir = Path(project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model_cls = RTDETR if model_name.startswith("rtdetr") else YOLO
    model_source = str(Path(resume_checkpoint).resolve()) if resume_checkpoint else model_name
    model = model_cls(model_source)

    if use_vfl and model_cls is not YOLO:
        raise ValueError("VFLDetectionLoss hiện chỉ hỗ trợ cho các mô hình YOLO.")
    if use_class_weights and not use_vfl:
        raise ValueError("Áp dụng trọng số lớp yêu cầu use_vfl=true.")
    if preprocessing not in {"raw", "roi_crop"}:
        raise ValueError(f"Chế độ tiền xử lý không được hỗ trợ: {preprocessing}")
    if preprocessing == "roi_crop":
        roi_config_path = Path(roi_config) if roi_config is not None else None
        if roi_config_path is not None and not roi_config_path.is_absolute():
            roi_config_path = config_path.parent.parent / roi_config_path
        if roi_config_path is None or not roi_config_path.exists():
            raise FileNotFoundError(
                "Chế độ preprocessing=roi_crop yêu cầu file cấu hình roi_config tồn tại."
            )
    elif roi_config is not None:
        raise ValueError("roi_config chỉ hợp lệ khi bật preprocessing=roi_crop.")

    if use_vfl:
        report_path = None
        if dataset_report is not None and Path(dataset_report).exists():
            report_path = Path(dataset_report)
        else:
            # Tự động tìm kiếm file dataset_report.json tại các vị trí tiêu chuẩn
            data_yaml_p = Path(data_yaml).resolve()
            candidates = [
                data_yaml_p.parent / "dataset_report.json",
                data_yaml_p.parent.parent / "dataset_report.json",
                Path("data/processed_roi/dataset_report.json"),
                Path("data/processed/dataset_report.json"),
                Path("artifacts/dataset_report.json"),
            ]
            for cand in candidates:
                if cand.exists():
                    report_path = cand
                    break

        if use_class_weights and report_path is None:
            raise FileNotFoundError(
                "use_class_weights=true nhưng không tìm thấy dataset_report.json; "
                "vui lòng truyền tham số --dataset-report một cách tường minh."
            )

        class_weights = compute_class_weights(report_path) if report_path else None
        model.add_callback("on_train_start", _attach_vfl_criterion(class_weights))

    train_args = {
        "data": str(Path(data_yaml).resolve()),
        "project": str(project_dir),
        "name": run_name or Path(model_name).stem.replace("-", "_"),
        "exist_ok": True,
        **cfg,
    }
    if resume_checkpoint:
        train_args["resume"] = str(Path(resume_checkpoint).resolve())

    return model.train(
        **train_args,
    )
