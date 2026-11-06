"""Quy trình tổ hợp mô hình (Ensemble Pipeline) cho Glass Defect Inspector.

Nạp đồng thời nhiều mô hình đa cấu trúc (ví dụ: FBDNet-M + YOLO26m-ROI-VFL + D-FINE-M),
thực thi suy luận đồng bộ với cơ chế cắt ROI và phân vùng SAHI Tiling,
sau đó hợp nhất các bounding box dự đoán bằng thuật toán Weighted Boxes Fusion (WBF).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from glass_inspector.paths import WORKSPACE_ROOT
from scale_error.ensemble.wbf import (
    CLASS_NAMES,
    DEFAULT_PER_CLASS_WEIGHTS,
    fuse_detection_lists,
)


class GlassDefectEnsemble:
    """Đường ống tổ hợp đa mô hình sử dụng thuật toán Weighted Boxes Fusion (WBF)."""

    def __init__(
        self,
        model_paths: list[str | Path] | None = None,
        model_types: list[str] | None = None,
        weights: list[float] | None = None,
        per_class_weights: dict[str, list[float]] | None = None,
        iou_threshold: float = 0.58,
        skip_box_threshold: float = 0.05,
        root_dir: Path = WORKSPACE_ROOT,
    ) -> None:
        """Khởi tạo pipeline tổ hợp.

        Tham số:
            model_paths: Danh sách đường dẫn tới các file trọng số mô hình thành viên.
            model_types: Danh sách loại mô hình tương ứng ('fbdnet', 'dfine', 'yolo').
            weights: Trọng số tổng thể của từng mô hình.
            per_class_weights: Ma trận trọng số thích ứng theo từng phân lớp khuyết tật.
            iou_threshold: Ngưỡng IoU để gom cụm các hộp trùng nhau trong WBF.
            skip_box_threshold: Ngưỡng điểm tin cậy tối thiểu để đưa hộp vào tổ hợp.
            root_dir: Thư mục gốc của không gian làm việc.
        """
        self.root_dir = Path(root_dir)
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.weights = weights
        self.per_class_weights = per_class_weights or DEFAULT_PER_CLASS_WEIGHTS

        # Cấu hình mặc định: Transformer hàng đầu (FBDNet-M) + CNN hàng đầu (YOLO26m-ROI-VFL)
        if model_paths is None:
            self.model_paths = [
                self.root_dir / "weights" / "dfine_m_fbdnet_best.pth",
                self.root_dir / "weights" / "yolo26m_roi_vfl_best.pt",
            ]
            self.model_types = ["fbdnet", "yolo"]
        else:
            self.model_paths = [Path(p) for p in model_paths]
            self.model_types = model_types or [
                ("fbdnet" if "fbdnet" in p.name.lower() or "dfine" in p.name.lower() else "yolo")
                for p in self.model_paths
            ]

        self.loaded_models: list[Any] = []
        self._is_initialized = False

    def load_models(self) -> None:
        """Nạp các mô hình thành viên vào bộ nhớ RAM/VRAM (chỉ gọi khi bắt đầu suy luận)."""
        if self._is_initialized:
            return

        from glass_inspector.inference_engine import load_dfine_model, load_yolo_model

        self.loaded_models = []
        for path, mtype in zip(self.model_paths, self.model_types, strict=False):
            if not path.is_file():
                raise FileNotFoundError(f"Không tìm thấy file trọng số checkpoint tại: {path}")

            if mtype in {"dfine", "fbdnet"}:
                model, postprocessor = load_dfine_model(str(path))
                self.loaded_models.append(
                    {"model": model, "postprocessor": postprocessor, "type": mtype}
                )
            else:
                yolo = load_yolo_model(str(path))
                self.loaded_models.append({"model": yolo, "type": "yolo"})

        self._is_initialized = True

    def predict_image(
        self,
        image: Image.Image | str | Path,
        conf_threshold: float = 0.15,
        use_per_class_weights: bool = True,
        enable_tiling: bool = True,
    ) -> dict[str, Any]:
        """Thực hiện suy luận tổ hợp trên một ảnh bề mặt kính đơn lẻ.

        Trả về:
            dict chứa:
                - detections: danh sách hộp hợp nhất (class_name, confidence, x1, y1, x2, y2)
                - num_detections: số lượng phát hiện
                - individual_predictions: kết quả phát hiện thô của từng mô hình
                - timing_ms: thời gian chi tiết từng mô hình và thời gian chạy thuật toán WBF
        """
        self.load_models()
        from glass_inspector.inference_engine import run_inspection

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        individual_dets: list[list[tuple[str, float, int, int, int, int]]] = []
        model_timings: list[float] = []

        t_start_total = time.perf_counter()

        # Chạy từng mô hình qua engine kiểm định (bao gồm ROI và SAHI Tiling)
        for path, mtype in zip(self.model_paths, self.model_types, strict=False):
            t_m_start = time.perf_counter()
            result = run_inspection(
                image_source=pil_img,
                weights_path=path,
                is_dfine=(mtype in {"dfine", "fbdnet"}),
                conf_threshold=conf_threshold,
                iou_threshold=self.iou_threshold,
                enable_roi=True,
                enable_tiling=enable_tiling,
                nms_mode="standard_nms",
            )
            model_timings.append((time.perf_counter() - t_m_start) * 1000)

            cur_dets = [
                (
                    d["class_name"],
                    float(d["confidence"]),
                    d["bbox"][0],
                    d["bbox"][1],
                    d["bbox"][2],
                    d["bbox"][3],
                )
                for d in result.get("detections", [])
            ]
            individual_dets.append(cur_dets)

        # Hợp nhất các hộp dự đoán bằng Weighted Boxes Fusion
        t_fuse_start = time.perf_counter()
        fused_dets = fuse_detection_lists(
            detection_lists=individual_dets,
            weights=self.weights,
            per_class_weights=self.per_class_weights if use_per_class_weights else None,
            class_names=CLASS_NAMES,
            iou_thr=self.iou_threshold,
            skip_box_thr=self.skip_box_threshold,
        )
        fusion_time_ms = (time.perf_counter() - t_fuse_start) * 1000
        total_time_ms = (time.perf_counter() - t_start_total) * 1000

        formatted_detections = [
            {
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
            }
            for cls_name, conf, x1, y1, x2, y2 in fused_dets
        ]

        return {
            "detections": formatted_detections,
            "num_detections": len(formatted_detections),
            "individual_detections": individual_dets,
            "timing_ms": {
                "total_pipeline_ms": round(total_time_ms, 2),
                "fusion_wbf_ms": round(fusion_time_ms, 2),
                "model_ms": [round(t, 2) for t in model_timings],
            },
        }

    def evaluate_offline_from_json(
        self,
        prediction_json_files: list[str | Path],
        ground_truth_json: str | Path,
        output_report_file: str | Path = "results/ensemble_evaluation_report.json",
        use_per_class_weights: bool = True,
    ) -> dict[str, Any]:
        """Hợp nhất các file JSON dự đoán offline từ nhiều mô hình và xuất báo cáo COCO mAP.

        Cho phép đánh giá tốc độ cao mà không cần nạp lại các mô hình GPU nặng.
        """
        out_report = Path(output_report_file)
        out_report.parent.mkdir(parents=True, exist_ok=True)

        print(f"Đang tải nhãn Ground Truth từ: {ground_truth_json}")
        with open(ground_truth_json, encoding="utf-8") as f:
            _ = json.load(f)

        pred_datasets = []
        for p_file in prediction_json_files:
            print(f"Đang đọc dự đoán từ: {p_file}")
            with open(p_file, encoding="utf-8") as f:
                pred_datasets.append(json.load(f))

        # Nhóm các dự đoán theo image_id
        all_img_ids = set()
        for p_data in pred_datasets:
            for item in p_data:
                all_img_ids.add(item["image_id"])

        fused_predictions: list[dict[str, Any]] = []

        for img_id in sorted(list(all_img_ids)):
            img_preds = []
            for p_data in pred_datasets:
                matching = [p for p in p_data if p["image_id"] == img_id]
                cur_det = []
                for m in matching:
                    cls_id = int(m["category_id"])
                    cls_name = (
                        CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
                    )
                    conf = float(m["score"])
                    bx, by, bw, bh = m["bbox"]
                    cur_det.append((cls_name, conf, int(bx), int(by), int(bx + bw), int(by + bh)))
                img_preds.append(cur_det)

            # Thực hiện WBF cho từng ảnh
            fused = fuse_detection_lists(
                detection_lists=img_preds,
                weights=self.weights,
                per_class_weights=self.per_class_weights if use_per_class_weights else None,
                class_names=CLASS_NAMES,
                iou_thr=self.iou_threshold,
                skip_box_thr=self.skip_box_threshold,
            )

            for cls_name, conf, x1, y1, x2, y2 in fused:
                cls_id = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0
                fused_predictions.append(
                    {
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": round(float(conf), 5),
                    }
                )

        fused_out_file = out_report.parent / "fused_ensemble_predictions.json"
        with open(fused_out_file, "w", encoding="utf-8") as f:
            json.dump(fused_predictions, f, indent=2)

        print(
            f"Đã lưu các dự đoán hợp nhất tại {fused_out_file} "
            f"(Tổng số: {len(fused_predictions)} bounding box)"
        )
        report_data = {
            "num_models": len(prediction_json_files),
            "total_fused_detections": len(fused_predictions),
            "iou_threshold": self.iou_threshold,
            "skip_box_threshold": self.skip_box_threshold,
            "fused_predictions_file": str(fused_out_file),
            "status": "READY_FOR_BENCHMARK_EVAL",
        }
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        return report_data
