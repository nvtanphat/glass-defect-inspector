"""Read-only FBDNet-M test evaluation and source-backed confusion/report artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from glass_inspector.inference_engine import CLASS_NAMES, load_dfine_model
from processing.apply_roi_crop import _crop_boxes_to_roi
from scale_error.data.validate import list_images, parse_yolo_label_file, read_names
from scale_error.evaluation.detection_confusion import confusion_statistics, detection_confusion


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_ground_truth(dataset_yaml, roi_path, limit=None):
    config = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8"))
    assert list(read_names(dataset_yaml).values()) == CLASS_NAMES, "Class mapping mismatch"
    images_root = Path(config["test"])
    labels_root = images_root.parent / "labels"
    paths = list_images(images_root)
    if len(paths) != 890:
        raise RuntimeError(f"BLOCKED: expected 890 test images, found {len(paths)}")
    groups = json.loads(Path(roi_path).read_text(encoding="utf-8"))["groups"]
    for group in groups.values():
        x1, y1, x2, y2 = group["roi_xyxyn"]
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError("Invalid local ROI configuration")
    if limit:
        paths = paths[:limit]
    images, annotations, manifest = [], [], []
    original_boxes = 0
    for image_id, path in enumerate(paths, 1):
        label = labels_root / path.relative_to(images_root).with_suffix(".txt")
        if not label.is_file():
            raise RuntimeError(f"BLOCKED: missing label {label.name}")
        boxes, errors = parse_yolo_label_file(label, 5)
        if errors:
            raise ValueError("; ".join(errors))
        with Image.open(path) as image:
            width, height = image.size
        roi = tuple(groups.get(f"{width}x{height}", {}).get("roi_xyxyn", [0, 0, 1, 1]))
        x1, y1, x2, y2 = [
            round(value * size)
            for value, size in zip(roi, [width, height, width, height], strict=True)
        ]
        roi_width, roi_height = x2 - x1, y2 - y1
        original_boxes += len(boxes)
        transformed = _crop_boxes_to_roi(boxes, roi, 0.1)
        if len(transformed) != len(boxes):
            raise RuntimeError(f"BLOCKED: ROI dropped boxes in {path.name}")
        for category, x, y, w, h in transformed:
            # The local ROI tool writes cropped YOLO labels with six decimal places.
            x, y, w, h = [round(value, 6) for value in (x, y, w, h)]
            bbox = [
                (x - w / 2) * roi_width,
                (y - h / 2) * roi_height,
                w * roi_width,
                h * roi_height,
            ]
            annotations.append(
                dict(
                    id=len(annotations) + 1,
                    image_id=image_id,
                    category_id=int(category),
                    bbox=bbox,
                    area=bbox[2] * bbox[3],
                    iscrowd=0,
                )
            )
        images.append(dict(id=image_id, file_name=path.name, width=roi_width, height=roi_height))
        manifest.append(
            dict(
                image_id=image_id,
                path=path.relative_to(ROOT).as_posix(),
                sha256=sha256(path),
                label_path=label.relative_to(ROOT).as_posix(),
                label_sha256=sha256(label),
                raw_size=[width, height],
                crop_xyxy=[x1, y1, x2, y2],
                boxes=len(boxes),
            )
        )
    if not limit and original_boxes != 1573:
        raise RuntimeError(f"BLOCKED: expected 1573 test boxes, found {original_boxes}")
    return dict(
        info={"description": "Fresh FBDNet ROI test annotations; original files untouched"},
        images=images,
        annotations=annotations,
        categories=[dict(id=index, name=name) for index, name in enumerate(CLASS_NAMES)],
    ), manifest


def decode_roi(record):
    path = ROOT / record["path"]
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unreadable image: {path.name}")
    x1, y1, x2, y2 = record["crop_xyxy"]
    if (x1, y1, x2, y2) == (0, 0, image.shape[1], image.shape[0]):
        with Image.open(path) as original:
            return original.convert("RGB")
    # Match the local ROI cv2.imwrite JPEG defaults without saving cropped copies to disk.
    success, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2])
    if not success:
        raise RuntimeError("ROI JPEG encoding failed")
    with Image.open(io.BytesIO(encoded.tobytes())) as cropped:
        return cropped.convert("RGB")


def coco_metrics(ground_truth, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = COCO()
    truth.dataset = ground_truth
    truth.createIndex()
    if predictions:
        detected = truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = dict(
            images=ground_truth["images"], categories=ground_truth["categories"], annotations=[]
        )
        detected.createIndex()
    evaluator = COCOeval(truth, detected, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = [
        "mAP50_95",
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
    overall = {key: float(value) for key, value in zip(keys, evaluator.stats, strict=True)}
    per_class = []
    precision = evaluator.eval["precision"]
    for index, name in enumerate(CLASS_NAMES):
        row = dict(class_name=name)
        for key, tensor in [
            ("mAP50_95", precision[:, :, index, 0, -1]),
            ("mAP50", precision[0, :, index, 0, -1]),
            ("mAP75", precision[5, :, index, 0, -1]),
        ]:
            valid = tensor[tensor >= 0]
            row[key] = float(valid.mean()) if valid.size else None
        per_class.append(row)
    return overall, per_class


def render_reports(output, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CLASS_NAMES + ["Background"]
    matrix = np.array(result["confusion_matrix"])
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    for filename, data, title, percentage in [
        ("confusion_counts.png", matrix, "FBDNet-M · Test · Số lượng", False),
        (
            "confusion_normalized.png",
            normalized * 100,
            "FBDNet-M · Test · Chuẩn hóa theo nhãn thực",
            True,
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9.5, 8), dpi=180)
        plot = ax.imshow(
            data, cmap="Blues", vmin=0, vmax=100 if percentage else max(matrix.max(), 1)
        )
        ax.set_xticks(range(6), labels, rotation=30, ha="right")
        ax.set_yticks(range(6), labels)
        ax.set_xlabel("Lớp dự đoán")
        ax.set_ylabel("Lớp thực tế")
        ax.set_title(
            title
            + f"\nconf ≥ {result['protocol']['confidence']} · IoU ≥ {result['protocol']['iou']}"
        )
        for i in range(6):
            for j in range(6):
                value = data[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.1f}%" if percentage else str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if value > data.max() * 0.55 else "#15334d",
                )
        fig.colorbar(plot, ax=ax, shrink=0.8, label="% theo hàng" if percentage else "Số box")
        fig.tight_layout()
        fig.savefig(output / filename, bbox_inches="tight")
        plt.close(fig)
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ground_truth / predicted"] + labels)
        writer.writerows([[name] + row.tolist() for name, row in zip(labels, matrix, strict=True)])
    overall = result["coco_metrics"]
    operating = result["operating_metrics"]
    p = result["protocol"]
    report = [
        "# Báo cáo đánh giá FBDNet-M",
        "",
        f"Đánh giá mới trên **{p['images']} ảnh test / {p['boxes_after_roi']} box**. Không huấn luyện lại; không thay số liệu lịch sử.",
        "",
        f"Checkpoint: `{p['checkpoint']}` · SHA256 `{p['checkpoint_sha256']}`.",
        "",
        "## Kết quả COCO",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---:|",
    ]
    report += [f"| {key} | {value * 100:.2f}% |" for key, value in overall.items()]
    report += [
        "",
        "## Kết quả theo lớp",
        "",
        "| Lớp | GT | AP50 | AP75 | AP50:95 | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_class"]:
        report.append(
            f"| {row['class_name']} | {row['support']} | {row['mAP50'] * 100:.2f}% | {row['mAP75'] * 100:.2f}% | {row['mAP50_95'] * 100:.2f}% | {row['precision'] * 100:.2f}% | {row['recall'] * 100:.2f}% | {row['f1'] * 100:.2f}% | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    report += [
        "",
        "## Ma trận nhầm lẫn",
        "",
        "![Số lượng](confusion_counts.png)",
        "",
        "![Chuẩn hóa](confusion_normalized.png)",
        "",
        f"Hàng là nhãn thực, cột là nhãn dự đoán. Ghép box một-một theo IoU giảm dần, không buộc cùng lớp. Ngưỡng confidence {p['confidence']}, IoU {p['iou']}. Cột Background là lỗi không được phát hiện; hàng Background là dự đoán không khớp nhãn. Ô Background/Background không biểu diễn true negative.",
        "",
        f"TP **{operating['tp']}** · FP **{operating['fp']}** · FN **{operating['fn']}**. Bỏ sót vào Background **{operating['missed_to_background']}**, báo ảo từ Background **{operating['false_alarms_from_background']}**, nhầm lớp **{operating['wrong_class']}**. FP/FN theo lớp bao gồm nhầm lớp.",
        "",
        f"Micro precision **{operating['precision'] * 100:.2f}%**, recall **{operating['recall'] * 100:.2f}%**, F1 **{operating['f1'] * 100:.2f}%**.",
        "",
        "## Quy trình và truy vết",
        "",
        "- Dùng code gốc local của dự án, checkpoint local và processing/roi_config.json.",
        "- RGB, crop ROI bằng OpenCV và JPEG mặc định quality 95; resize trực tiếp 960×960, chia pixel cho 255. Nhãn crop giữ 6 chữ số thập phân như công cụ ROI local. Không tiling, không NMS bổ sung.",
        f"- Box trước/sau ROI: **{p['boxes_before_roi']} / {p['boxes_after_roi']}**; không mất ảnh hoặc box.",
        "- COCO AP dùng toàn bộ score, maxDets 100, IoU 0.50–0.95. Precision/recall/F1 và ma trận dùng ngưỡng cố định nêu trên; đây là hai phép đánh giá khác nhau.",
        f"- Thiết bị: **{p['device']}**, {p['threads']} CPU threads, batch {p['batch_size']}. Forward trung bình **{result['performance']['forward_ms_per_image']:.2f} ms/ảnh**; không phải số đo FPS trên T4.",
        "- `predictions.jsonl` chứa dự đoán thật theo từng ảnh; `ground_truth.json` là bản nhãn đánh giá mới; `evaluation.json` chứa cấu hình, hash ảnh/nhãn và toàn bộ chỉ số.",
        "",
        "## Giới hạn",
        "",
        "- Kết quả này dùng bộ test nguồn 890 ảnh và ROI hiện hành, không dùng bộ COCO lịch sử chỉ có 1.483 box. Không xem là tái lập số liệu cũ nếu chưa đối chiếu protocol của lần đo cũ.",
        "- 42,42% trong log FBDNet cũ là validation trong quá trình train; không thay bằng kết quả test mới này.",
        "- Audit đã nêu rủi ro ảnh cùng nhóm capture/augmentation giữa các split. Chưa có cơ sở tuyên bố 0% leakage hoặc năng lực tổng quát trên dây chuyền khác.",
        "- Chỉ số được tính với nhãn hiện có. Báo cáo không suy diễn chất lượng PASS chỉ vì không có detection.",
        "",
    ]
    confusion_pairs = sorted(
        [
            (int(matrix[i, j]), CLASS_NAMES[i], CLASS_NAMES[j])
            for i in range(5)
            for j in range(5)
            if i != j and matrix[i, j]
        ],
        reverse=True,
    )
    worst_class = min(result["per_class"], key=lambda row: row["recall"])
    detail = [
        "## Phân tích sai số",
        "",
        f"Lớp có recall thấp nhất tại ngưỡng đã chọn: **{worst_class['class_name']}**, "
        f"**{worst_class['recall'] * 100:.2f}%** trên **{worst_class['support']}** nhãn. "
        "Đây là kết quả ở ngưỡng cố định, không phải AP trung bình trên nhiều ngưỡng.",
        "",
        "| Nhãn thực → dự đoán nhầm | Số box |",
        "|---|---:|",
    ]
    detail += [
        f"| {truth} → {predicted} | {count} |" for count, truth, predicted in confusion_pairs
    ]
    if not confusion_pairs:
        detail.append("| Không có nhầm lớp tại ngưỡng đã chọn | 0 |")
    detail += [
        "",
        "### Ảnh cần xem lại",
        "",
        "| Image ID | File ảnh nguồn | Bỏ sót vào background | Báo ảo | Nhầm lớp |",
        "|---|---|---:|---:|---:|",
    ]
    records = {row["image_id"]: row for row in p["input_manifest"]}
    image_errors = sorted(
        result["image_errors"],
        key=lambda row: (
            -(row["missed"] + row["false_alarms"] + row["wrong_class"]),
            row["image_id"],
        ),
    )
    detail += [
        f"| {row['image_id']} | {Path(records[row['image_id']]['path']).name} | "
        f"{row['missed']} | {row['false_alarms']} | {row['wrong_class']} |"
        for row in image_errors[:10]
    ]
    detail += [
        "",
        "Các lớp ít nhãn (đặc biệt LoiTray) có độ bất định lớn hơn. "
        "Không suy diễn chênh lệch nhỏ là cải thiện có ý nghĩa thống kê.",
        "",
    ]
    location = report.index("## Quy trình và truy vết")
    report[location:location] = detail
    report += [
        "## Chạy lại",
        "",
        "```powershell",
        "python -B scripts/evaluate_fbdnet.py --output reports/fbdnet_m_NEW --threads 4 --batch-size 4",
        "```",
        "",
        "Dùng thư mục output mới; công cụ từ chối ghi đè kết quả cũ.",
        "",
    ]
    markdown = "\n".join(report)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    write_pdf(output, result)


def write_pdf(output, result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image as PDFImage
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font = Path("C:/Windows/Fonts/arial.ttf")
    if not font.exists():
        font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    pdfmetrics.registerFont(TTFont("ReportFont", str(font)))
    style = ParagraphStyle("body", fontName="ReportFont", fontSize=10, leading=15)
    heading = ParagraphStyle("heading", parent=style, fontSize=17, leading=24, spaceAfter=12)
    document = SimpleDocTemplate(
        str(output / "report.pdf"),
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )
    p = result["protocol"]
    content = [
        Paragraph("FBDNet-M · Báo cáo đánh giá test", heading),
        Paragraph(
            f"{p['images']} ảnh · {p['boxes_after_roi']} nhãn · input 960×960 · {p['device']}",
            style,
        ),
        Spacer(1, 12),
    ]
    table = [["Chỉ số", "Kết quả"]] + [
        [key, f"{value * 100:.2f}%"] for key, value in result["coco_metrics"].items()
    ]
    for rows in [
        table,
        [["Lớp", "GT", "AP50:95", "Precision", "Recall", "F1"]]
        + [
            [
                row["class_name"],
                row["support"],
                f"{row['mAP50_95'] * 100:.2f}%",
                f"{row['precision'] * 100:.2f}%",
                f"{row['recall'] * 100:.2f}%",
                f"{row['f1'] * 100:.2f}%",
            ]
            for row in result["per_class"]
        ],
        [["Lớp", "AP50", "AP75", "TP", "FP", "FN"]]
        + [
            [
                row["class_name"],
                f"{row['mAP50'] * 100:.2f}%",
                f"{row['mAP75'] * 100:.2f}%",
                row["tp"],
                row["fp"],
                row["fn"],
            ]
            for row in result["per_class"]
        ],
    ]:
        element = Table(rows, repeatRows=1, hAlign="LEFT")
        element.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "ReportFont"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff5")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        content += [element, Spacer(1, 14)]
    content += [
        Paragraph(
            f"Micro precision {result['operating_metrics']['precision'] * 100:.2f}%, "
            f"recall {result['operating_metrics']['recall'] * 100:.2f}%, "
            f"F1 {result['operating_metrics']['f1'] * 100:.2f}%.",
            style,
        ),
        Paragraph(
            f"Ma trận: confidence ≥ {p['confidence']}, IoU ≥ {p['iou']}. Hàng = nhãn thực; cột = dự đoán. FP/FN bao gồm nhầm lớp. COCO AP không áp dụng ngưỡng confidence này.",
            style,
        ),
    ]
    for filename, title in [
        ("confusion_counts.png", "Ma trận nhầm lẫn · Số lượng"),
        ("confusion_normalized.png", "Ma trận nhầm lẫn · Chuẩn hóa theo hàng"),
    ]:
        content += [
            PageBreak(),
            Paragraph(title, heading),
            PDFImage(str(output / filename), width=520, height=438),
            Spacer(1, 12),
            Paragraph(
                "Cột Background: lỗi bỏ sót. Hàng Background: báo ảo. Không có true-negative count cho object detection.",
                style,
            ),
        ]
    content += [PageBreak(), Paragraph("Quy trình, truy vết và giới hạn", heading)]
    for text in [
        "Bộ đánh giá: code gốc trong repo, weights local và ROI local.",
        f"Checkpoint: {p['checkpoint']}",
        f"SHA256: {p['checkpoint_sha256']}",
        f"ROI SHA256: {p['roi_sha256']}",
        f"Box trước/sau ROI: {p['boxes_before_roi']} / {p['boxes_after_roi']}; không sửa dữ liệu gốc.",
        "Suy luận dùng kiến trúc FBDNet-M gốc (HGNetv2FBD + FBDHybridEncoder, AFTA/SFIF/DCFSB), nạp weights strict; không thay loss hoặc huấn luyện lại.",
        "42,42% trong log cũ là validation; báo cáo này tính test mới. Bộ COCO cũ 1.483 box không được dùng. Rủi ro capture/augmentation giữa các split còn cần xác minh; không tuyên bố 0% leakage.",
        "Nguồn tính toán chi tiết: evaluation.json, ground_truth.json, predictions.jsonl và confusion_matrix.csv trong cùng thư mục. Số đo tốc độ CPU không đại diện cho GPU T4.",
    ]:
        content += [Paragraph(text, style), Spacer(1, 10)]
    document.build(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/dfine_m_fbdnet_best.pth")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/runtime_data.yaml")
    parser.add_argument("--roi", type=Path, default=ROOT / "processing/roi_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/fbdnet_m")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="Explicit pilot subset; never a full-test report")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.threads <= 0 or args.batch_size <= 0:
        parser.error("Resource limits must be positive")
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        parser.error("Confidence and IoU must be in [0, 1]")
    if args.weights.name != "dfine_m_fbdnet_best.pth":
        parser.error("Use the verified FBDNet-M checkpoint, not a substituted model")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Output directory must be empty; historical results are never overwritten")
    torch.set_num_threads(args.threads)
    ground_truth, manifest = prepare_ground_truth(args.data, args.roi, args.limit)
    (output / "ground_truth.json").write_text(json.dumps(ground_truth), encoding="utf-8")
    checkpoint_sha = sha256(args.weights)
    protocol = dict(
        status="PROCESSING",
        pilot=bool(args.limit),
        checkpoint=args.weights.relative_to(ROOT).as_posix(),
        checkpoint_sha256=checkpoint_sha,
        roi_sha256=sha256(args.roi),
        dataset_yaml_sha256=sha256(args.data),
        images=len(manifest),
        boxes_before_roi=sum(row["boxes"] for row in manifest),
        boxes_after_roi=len(ground_truth["annotations"]),
        input_size=[960, 960],
        batch_size=args.batch_size,
        threads=args.threads,
        device="cpu",
        confidence=args.confidence,
        iou=args.iou,
        matching="Class-agnostic, descending IoU, one-to-one",
        input_manifest=manifest,
    )
    print(
        f"AUDIT PASS: {protocol['images']} images, {protocol['boxes_after_roi']} boxes, SHA256 {checkpoint_sha}",
        flush=True,
    )
    model, postprocessor = load_dfine_model(str(args.weights))
    protocol["runtime_selection"] = "Original local repository code, explicitly selected by user"
    assert type(model.backbone).__name__ == "HGNetv2FBD"
    assert type(model.encoder).__name__ == "FBDHybridEncoder"
    protocol["architecture"] = dict(
        backbone=type(model.backbone).__name__,
        encoder=type(model.encoder).__name__,
        decoder=type(model.decoder).__name__,
    )
    protocol["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    runtime_root = ROOT / "results/D-FINE/src"
    protocol["runtime_source_sha256"] = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(runtime_root.rglob("*.py"))
    }
    transform = transforms.Compose([transforms.Resize((960, 960)), transforms.ToTensor()])
    all_predictions, forward_seconds = [], []
    started = time.perf_counter()
    with (
        (output / "predictions.jsonl").open("x", encoding="utf-8") as stream,
        torch.inference_mode(),
    ):
        for offset in range(0, len(manifest), args.batch_size):
            batch = manifest[offset : offset + args.batch_size]
            tensors = [transform(decode_roi(row)) for row in batch]
            sizes = torch.tensor(
                [
                    [
                        row["crop_xyxy"][2] - row["crop_xyxy"][0],
                        row["crop_xyxy"][3] - row["crop_xyxy"][1],
                    ]
                    for row in batch
                ]
            )
            inputs = torch.stack(tensors)
            begin = time.perf_counter()
            labels, boxes, scores = postprocessor(model(inputs), sizes)
            forward_seconds.append(time.perf_counter() - begin)
            for index, row in enumerate(batch):
                predictions = []
                for label, box, score in zip(
                    labels[index].tolist(),
                    boxes[index].tolist(),
                    scores[index].tolist(),
                    strict=True,
                ):
                    x1, y1, x2, y2 = box
                    if not 0 <= label < 5 or not np.isfinite(box).all() or not np.isfinite(score):
                        raise ValueError("Nonfinite or invalid-class prediction")
                    predictions.append(
                        dict(
                            image_id=row["image_id"],
                            category_id=int(label),
                            bbox=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                            score=score,
                        )
                    )
                all_predictions.extend(predictions)
                stream.write(
                    json.dumps(dict(image_id=row["image_id"], predictions=predictions)) + "\n"
                )
            stream.flush()
            done = offset + len(batch)
            elapsed = time.perf_counter() - started
            print(
                f"INFERENCE {done}/{len(manifest)} elapsed={elapsed:.1f}s eta={elapsed / done * (len(manifest) - done):.1f}s",
                flush=True,
            )
    assert sha256(args.weights) == checkpoint_sha, "Checkpoint changed during evaluation"
    overall, per_class_ap = coco_metrics(ground_truth, all_predictions)
    matrix, errors = detection_confusion(
        ground_truth["images"],
        ground_truth["annotations"],
        all_predictions,
        confidence=args.confidence,
        iou=args.iou,
    )
    per_class_operating, totals = confusion_statistics(matrix, CLASS_NAMES)
    per_class = [
        ap | operating for ap, operating in zip(per_class_ap, per_class_operating, strict=True)
    ]
    assert [row["support"] for row in per_class] == [
        Counter(a["category_id"] for a in ground_truth["annotations"])[i] for i in range(5)
    ]
    protocol["status"] = "PILOT_COMPLETED" if args.limit else "COMPLETED"
    result = dict(
        generated_at=datetime.now(UTC).isoformat(),
        model="FBDNet-M",
        protocol=protocol,
        coco_metrics=overall,
        per_class=per_class,
        operating_metrics=totals,
        confusion_matrix=matrix.tolist(),
        image_errors=errors,
        performance=dict(
            forward_ms_per_image=sum(forward_seconds) / len(manifest) * 1000,
            elapsed_seconds=time.perf_counter() - started,
        ),
        predictions_sha256=sha256(output / "predictions.jsonl"),
        ground_truth_sha256=sha256(output / "ground_truth.json"),
    )
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.limit:
        render_reports(output, result)
    print(
        f"{protocol['status']}: {output}; COCO mAP50:95={overall['mAP50_95'] * 100:.4f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
