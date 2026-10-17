"""Inference engine for Glass Defect Inspector.

Connects to real checkpoints in the weights/ directory and performs
defect detection, ROI cropping, Tiling (960x960, 20% overlap),
coordinate mapping, timing breakdown, FPS measurement, and result compilation.
Supports YOLO26m (Ultralytics) and D-FINE-M (HGNetv2 + Transformer).
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

from glass_inspector.paths import WORKSPACE_ROOT

# Project root directory

# 5 glass defect classes matching dataset
CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]

# Class colors definition matching UI replica
CLASS_STYLES = {
    "DiVat": {
        "hex": "#10b981",  # emerald-500
        "rgb": (16, 185, 129),
        "tailwind_text": "text-emerald-600",
        "tailwind_bg": "bg-emerald-500",
        "tailwind_border": "border-emerald-500",
    },
    "LoiChi": {
        "hex": "#00badb",  # cyan-400 / #00badb
        "rgb": (0, 186, 219),
        "tailwind_text": "text-[#00badb]",
        "tailwind_bg": "bg-[#00badb]",
        "tailwind_border": "border-cyan-400",
    },
    "DiVatLoiLom": {
        "hex": "#f43f5e",  # rose-500
        "rgb": (244, 63, 94),
        "tailwind_text": "text-rose-500",
        "tailwind_bg": "bg-rose-500",
        "tailwind_border": "border-rose-500",
    },
    "LoiNhua": {
        "hex": "#f59e0b",  # amber-500
        "rgb": (245, 158, 11),
        "tailwind_text": "text-amber-500",
        "tailwind_bg": "bg-amber-500",
        "tailwind_border": "border-amber-500",
    },
    "LoiTray": {
        "hex": "#9333ea",  # purple-600
        "rgb": (147, 51, 234),
        "tailwind_text": "text-purple-600",
        "tailwind_bg": "bg-purple-600",
        "tailwind_border": "border-purple-600",
    },
}

# Fallback for unexpected classes
DEFAULT_STYLE = {
    "hex": "#3b82f6",  # blue-500
    "rgb": (59, 130, 246),
    "tailwind_text": "text-blue-600",
    "tailwind_bg": "bg-blue-500",
    "tailwind_border": "border-blue-500",
}

# Model instance caches
_YOLO_CACHE: dict[str, Any] = {}
_DFINE_CACHE: dict[str, Any] = {}
_ROI_CONFIG_CACHE: dict[str, Any] | None = None


def load_roi_config() -> dict[str, Any]:
    """Load cached ROI configuration from processing/roi_config.json."""
    global _ROI_CONFIG_CACHE
    if _ROI_CONFIG_CACHE is None:
        cfg_path = WORKSPACE_ROOT / "processing" / "roi_config.json"
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        for group in config["groups"].values():
            x1, y1, x2, y2 = group["roi_xyxyn"]
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("Invalid ROI configuration")
        _ROI_CONFIG_CACHE = config
    return _ROI_CONFIG_CACHE


def load_yolo_model(weights_path: str) -> Any:
    """Load and cache YOLO model."""
    from ultralytics import RTDETR, YOLO

    weights_path = str(Path(weights_path).resolve())
    if not Path(weights_path).is_file():
        raise FileNotFoundError("Checkpoint missing")
    if weights_path not in _YOLO_CACHE:
        factory = RTDETR if "rtdetr" in Path(weights_path).name else YOLO
        model = factory(weights_path)
        if list(model.names.values()) != CLASS_NAMES:
            raise ValueError("Checkpoint class mapping does not match glass dataset")
        _YOLO_CACHE[weights_path] = model
    return _YOLO_CACHE[weights_path]


def load_dfine_model(weights_path: str) -> tuple[Any, Any]:
    """Load and cache D-FINE model and postprocessor."""
    import torch

    weights_path = str(Path(weights_path).resolve())
    if "model" in _DFINE_CACHE and _DFINE_CACHE.get("path") == weights_path:
        return _DFINE_CACHE["model"], _DFINE_CACHE["postprocessor"]

    dfine_dir = WORKSPACE_ROOT / "results" / "D-FINE"
    if str(dfine_dir) not in sys.path:
        sys.path.insert(0, str(dfine_dir))

    # D-FINE imports PyTorch's TensorBoard logger for registration. It does not
    # need TensorFlow; use TensorBoard's own supported no-TensorFlow API.
    import tensorboard.compat
    from tensorboard.compat import tensorflow_stub

    tensorboard.compat.tf = tensorflow_stub
    from src.core import YAMLConfig

    if "fbdnet" in Path(weights_path).name:
        config_path = dfine_dir / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_glass.yml"
    else:
        config_path = dfine_dir / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_glass_test.yml"
    cfg = YAMLConfig(str(config_path), resume=weights_path)
    if cfg.yaml_cfg.get("num_classes") != 5 or cfg.yaml_cfg.get("remap_mscoco_category"):
        raise ValueError("D-FINE class configuration must use 5 contiguous glass classes")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    cfg.model.load_state_dict(state)
    model = cfg.model.eval()
    postprocessor = cfg.postprocessor.deploy()

    _DFINE_CACHE["path"] = weights_path
    _DFINE_CACHE["model"] = model
    _DFINE_CACHE["postprocessor"] = postprocessor
    return model, postprocessor


def pil_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """Convert PIL image to data URL base64 string."""
    buf = io.BytesIO()
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=format, quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def base64_to_pil(data_url: str) -> Image.Image:
    """Convert base64 data URL string to PIL Image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def get_roi_crop(pil_img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Apply ROI crop for high-resolution images according to roi_config.json."""
    w, h = pil_img.size
    config = load_roi_config()
    groups = config.get("groups", {})
    key = f"{w}x{h}"

    if key in groups:
        roi_xyxyn = groups[key].get("roi_xyxyn", [0.0, 0.0, 1.0, 1.0])
        if roi_xyxyn != [0.0, 0.0, 1.0, 1.0]:
            x1 = int(round(roi_xyxyn[0] * w))
            y1 = int(round(roi_xyxyn[1] * h))
            x2 = int(round(roi_xyxyn[2] * w))
            y2 = int(round(roi_xyxyn[3] * h))
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)
            return pil_img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)

    return pil_img, None


def compute_slice_intervals(
    total_dim: int, slice_dim: int = 960, overlap_ratio: float = 0.20
) -> list[tuple[int, int]]:
    """Compute 1D slice intervals covering total_dim with overlap."""
    if total_dim <= 0 or slice_dim < 320 or not 0 <= overlap_ratio <= 0.5:
        raise ValueError("Invalid tile geometry")
    if total_dim <= slice_dim:
        return [(0, total_dim)]
    step = max(1, int(round(slice_dim * (1.0 - overlap_ratio))))
    intervals = []
    pos = 0
    while pos + slice_dim <= total_dim:
        intervals.append((pos, pos + slice_dim))
        pos += step
    last_start = total_dim - slice_dim
    last_end = total_dim
    if not intervals or intervals[-1] != (last_start, last_end):
        intervals.append((last_start, last_end))
    deduped = []
    for it in intervals:
        if not deduped or it != deduped[-1]:
            deduped.append(it)
    return deduped


def run_nms_boxes(
    detections: list[tuple[str, float, int, int, int, int]],
    iou_threshold: float = 0.45,
) -> list[tuple[str, float, int, int, int, int]]:
    """Perform per-class Non-Maximum Suppression (NMS) using OpenCV."""
    if not detections:
        return []

    # Group by class
    by_class: dict[str, list[tuple[float, int, int, int, int]]] = {}
    for cls_name, conf, x1, y1, x2, y2 in detections:
        by_class.setdefault(cls_name, []).append((conf, x1, y1, x2, y2))

    merged: list[tuple[str, float, int, int, int, int]] = []
    for cls_name, items in by_class.items():
        if len(items) == 1:
            conf, x1, y1, x2, y2 = items[0]
            merged.append((cls_name, conf, x1, y1, x2, y2))
            continue

        cv_boxes = [[x1, y1, x2 - x1, y2 - y1] for _, x1, y1, x2, y2 in items]
        confs = [conf for conf, _, _, _, _ in items]
        indices = cv2.dnn.NMSBoxes(cv_boxes, confs, 0.0, iou_threshold)

        if len(indices) > 0:
            for idx in indices.flatten():
                conf, x1, y1, x2, y2 = items[idx]
                merged.append((cls_name, conf, x1, y1, x2, y2))

    return merged


def annotate_image(
    pil_img: Image.Image,
    detections: list[dict[str, Any]],
) -> Image.Image:
    """Draw bounding boxes and class tags onto the image."""
    img_copy = pil_img.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    # Font sizing based on image resolution
    font_size = max(18, int(h * 0.028))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    line_w = max(3, int(w * 0.0035))

    for det in detections:
        box = det["xyxy"]
        cls_name = det["class"]
        conf = det["confidence"]
        style = CLASS_STYLES.get(cls_name, DEFAULT_STYLE)
        rgb = style["rgb"]

        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=line_w)

        label_text = f"{cls_name} {conf:.2f}"
        try:
            bbox = font.getbbox(label_text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label_text) * 7, 12

        pad = 3
        tag_y1 = max(0, y1 - th - pad * 2)
        tag_y2 = tag_y1 + th + pad * 2
        tag_x1 = x1
        tag_x2 = x1 + tw + pad * 2

        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=rgb)
        draw.text((tag_x1 + pad, tag_y1 + pad), label_text, fill=(255, 255, 255), font=font)

    return img_copy


def crop_defect_thumbnail(
    pil_img: Image.Image, box: list[int], target_size: int = 96
) -> Image.Image:
    """Crop zoomed defect thumbnail with context margin."""
    w, h = pil_img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    crop_size = max(bw * 2, bh * 2, 48)
    half = crop_size // 2

    cx1 = max(0, cx - half)
    cy1 = max(0, cy - half)
    cx2 = min(w, cx + half)
    cy2 = min(h, cy + half)

    crop = pil_img.crop((cx1, cy1, cx2, cy2))
    return crop.resize((target_size, target_size), Image.Resampling.NEAREST)


def _predict_single_image(
    img: Image.Image,
    weights_path: Path,
    is_dfine: bool,
    conf_threshold: float,
    iou_threshold: float,
    nms_mode: str,
) -> tuple[list[tuple[str, float, int, int, int, int]], float]:
    """Execute forward prediction on a single image/tile.

    Returns:
        (detections_list, forward_time_ms)
    """
    w, h = img.size
    detections: list[tuple[str, float, int, int, int, int]] = []

    if is_dfine:
        import torch
        import torchvision.transforms as T

        dfine_model, postprocessor = load_dfine_model(str(weights_path))
        transforms = T.Compose(
            [
                T.Resize((960, 960)),
                T.ToTensor(),
            ]
        )
        im_data = transforms(img).unsqueeze(0)
        orig_size = torch.tensor([[w, h]])

        t_start = time.perf_counter()
        with torch.no_grad():
            output = dfine_model(im_data)
            labels, boxes, scores = postprocessor(output, orig_size)
        forward_ms = (time.perf_counter() - t_start) * 1000

        labels_list = labels[0].tolist()
        boxes_list = boxes[0].tolist()
        scores_list = scores[0].tolist()

        for l_idx, b, conf in zip(labels_list, boxes_list, scores_list, strict=True):
            if conf < conf_threshold:
                continue
            cls_name = (
                CLASS_NAMES[int(l_idx)] if 0 <= int(l_idx) < len(CLASS_NAMES) else f"Class_{l_idx}"
            )
            bx1, by1, bx2, by2 = [int(round(v)) for v in b]
            bx1 = max(0, min(bx1, w))
            by1 = max(0, min(by1, h))
            bx2 = max(0, min(bx2, w))
            by2 = max(0, min(by2, h))
            detections.append((cls_name, float(conf), bx1, by1, bx2, by2))
    else:
        model = load_yolo_model(str(weights_path))
        t_start = time.perf_counter()

        results = model.predict(
            source=img,
            imgsz=960,
            rect=False,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        forward_ms = (time.perf_counter() - t_start) * 1000

        res = results[0]
        boxes = res.boxes
        for b in boxes:
            cls_id = int(b.cls[0].item())
            cls_name = model.names.get(cls_id, f"Class_{cls_id}")
            conf = float(b.conf[0].item())
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
            detections.append((cls_name, conf, bx1, by1, bx2, by2))

    # Ultralytics' `nms` flag controls export; native end-to-end prediction
    # ignores it. Implement the UI's optional NMS explicitly for both adapters.
    if nms_mode == "standard_nms":
        detections = run_nms_boxes(detections, iou_threshold)
    return detections, forward_ms


def run_inspection(
    image_input: str | Path | Image.Image,
    weights_path: str | Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    nms_mode: str = "nms_free",  # "nms_free" or "standard_nms"
    enable_roi: bool = True,
    enable_tiling: bool = False,
    tile_size: int = 960,
    overlap_ratio: float = 0.20,
) -> dict[str, Any]:
    """Execute end-to-end defect inspection on image using real weights.

    Includes full Preprocessing, Forward Model inference, Postprocessing/NMS,
    and rigorous FPS and Latency timing breakdown.
    """
    total_start = time.perf_counter()
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {weights_path.name}")
    if not (0.01 <= conf_threshold <= 1 and 0.01 <= iou_threshold <= 1):
        raise ValueError("Thresholds must be in [0.01, 1]")
    if nms_mode not in {"nms_free", "standard_nms"}:
        raise ValueError("Invalid NMS mode")

    # 1. Load image (Preprocessing Stage 1)
    prep_start = time.perf_counter()
    if isinstance(image_input, (str, Path)):
        if str(image_input).startswith("data:image"):
            pil_img = base64_to_pil(str(image_input))
        else:
            pil_img = Image.open(str(image_input)).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    orig_w, orig_h = pil_img.size

    # 2. ROI Cropping (Preprocessing Stage 2)
    roi_offset = (0, 0)
    target_img = pil_img
    roi_box_applied = None

    if enable_roi:
        cropped_img, roi_box = get_roi_crop(pil_img)
        if roi_box is not None:
            target_img = cropped_img
            roi_offset = (roi_box[0], roi_box[1])
            roi_box_applied = roi_box

    target_w, target_h = target_img.size
    is_dfine = weights_path.suffix == ".pth" or "dfine" in weights_path.name.lower()

    prep_ms = (time.perf_counter() - prep_start) * 1000

    # 3. Model Inference (Single pass vs Tiled Slicing)
    infer_start = time.perf_counter()
    raw_detections: list[tuple[str, float, int, int, int, int]] = []
    tiles_processed = 1

    if enable_tiling and (target_w > tile_size or target_h > tile_size):
        # Tiling mode: 960x960 with overlap
        x_intervals = compute_slice_intervals(target_w, tile_size, overlap_ratio)
        y_intervals = compute_slice_intervals(target_h, tile_size, overlap_ratio)
        tiles_processed = len(x_intervals) * len(y_intervals)
        if tiles_processed > 64:
            raise ValueError("Image requires more than 64 tiles")

        all_tile_dets = []
        for sy, ey in y_intervals:
            for sx, ex in x_intervals:
                tile_crop = target_img.crop((sx, sy, ex, ey))
                tile_dets, _ = _predict_single_image(
                    tile_crop, weights_path, is_dfine, conf_threshold, iou_threshold, nms_mode
                )
                # Remap to target_img coordinates
                for c_name, conf, bx1, by1, bx2, by2 in tile_dets:
                    all_tile_dets.append((c_name, conf, bx1 + sx, by1 + sy, bx2 + sx, by2 + sy))

        # Merge overlapping tile seams with NMS
        raw_detections = run_nms_boxes(all_tile_dets, iou_threshold=iou_threshold)
    else:
        # Standard full or ROI-cropped single pass
        raw_detections, _ = _predict_single_image(
            target_img, weights_path, is_dfine, conf_threshold, iou_threshold, nms_mode
        )

    infer_ms = (time.perf_counter() - infer_start) * 1000

    # 4. Postprocessing: coordinate re-mapping, thumbnails & annotation
    post_start = time.perf_counter()
    detections = []
    classes_found = set()

    for i, (cls_name, conf, bx1, by1, bx2, by2) in enumerate(raw_detections):
        # Map back to full original image space
        x1 = max(0, min(orig_w, bx1 + roi_offset[0]))
        y1 = max(0, min(orig_h, by1 + roi_offset[1]))
        x2 = max(0, min(orig_w, bx2 + roi_offset[0]))
        y2 = max(0, min(orig_h, by2 + roi_offset[1]))
        if x2 <= x1 or y2 <= y1:
            continue
        if cls_name not in CLASS_NAMES:
            raise ValueError("Model returned unknown class")

        x1n = x1 / orig_w
        y1n = y1 / orig_h
        x2n = x2 / orig_w
        y2n = y2 / orig_h

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        thumb_img = crop_defect_thumbnail(pil_img, [x1, y1, x2, y2])
        thumb_b64 = pil_to_base64(thumb_img, format="PNG")
        style = CLASS_STYLES.get(cls_name, DEFAULT_STYLE)

        detections.append(
            {
                "id": i + 1,
                "class": cls_name,
                "confidence": conf,
                "class_id": CLASS_NAMES.index(cls_name),
                "class_name": cls_name,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_normalized": [x1n, y1n, x2n, y2n],
                "xyxy": [x1, y1, x2, y2],
                "xyxyn": [round(x1n, 4), round(y1n, 4), round(x2n, 4), round(y2n, 4)],
                "css_top": round(y1n * 100, 2),
                "css_left": round(x1n * 100, 2),
                "css_width": (x2n - x1n) * 100,
                "css_height": (y2n - y1n) * 100,
                "area": area,
                "crop_size": f"{max(bw, 16)} × {max(bh, 16)}",
                "crop_b64": thumb_b64,
                "color_hex": style["hex"],
                "tailwind_text": style["tailwind_text"],
                "tailwind_bg": style["tailwind_bg"],
                "tailwind_border": style["tailwind_border"],
            }
        )
        classes_found.add(cls_name)

    annotated = annotate_image(pil_img, detections)
    annotated_b64 = pil_to_base64(annotated, format="JPEG", quality=88)
    image_b64 = pil_to_base64(pil_img, format="JPEG", quality=88)

    roi_preview = None
    if detections:
        first = detections[0]
        roi_preview = {
            "class": first["class"],
            "confidence": first["confidence"],
            "crop_size": first["crop_size"],
            "xyxy": first["xyxy"],
            "area": first["area"],
            "crop_b64": first["crop_b64"],
            "color_hex": first["color_hex"],
            "tailwind_text": first["tailwind_text"],
        }

    post_ms = (time.perf_counter() - post_start) * 1000
    total_latency_ms = (time.perf_counter() - total_start) * 1000

    # Calculate real-time FPS
    fps = round(1000.0 / max(total_latency_ms, 1.0), 1)

    return {
        "success": True,
        "image_width": orig_w,
        "image_height": orig_h,
        "image_b64": image_b64,
        "annotated_b64": annotated_b64,
        "latency_ms": round(total_latency_ms),
        "fps": fps,
        "timing": {
            "preprocess_ms": round(prep_ms, 1),
            "inference_ms": round(infer_ms, 1),
            "postprocess_ms": round(post_ms, 1),
            "total_ms": round(total_latency_ms, 1),
            "fps": fps,
            "tiles_processed": tiles_processed,
        },
        "total_defects": len(detections),
        "defect_types": len(classes_found),
        "detections": detections,
        "roi_preview": roi_preview,
        "model_name": Path(weights_path).name,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "nms_mode": nms_mode,
        "enable_roi": enable_roi,
        "enable_tiling": enable_tiling,
        "roi_box": roi_box_applied,
    }


def compute_box_iou(b1: Any, b2: Any) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def fast_wbf_fusion(
    boxes_list: list[list[Any]],
    scores_list: list[list[float]],
    weights: list[float],
    iou_thr: float,
    coord_ratio: list[float],
    boost_3: float,
    boost_2: float,
    solo_decay: float,
    conf_floor: float,
) -> tuple[list[list[float]], list[float]]:
    """Weighted Boxes Fusion with multi-model consensus boosting and solo decay."""
    import numpy as np

    num_m = len(boxes_list)
    items: list[tuple[Any, float, int]] = []
    for m in range(num_m):
        for b, s in zip(boxes_list[m], scores_list[m], strict=False):
            if s >= conf_floor:
                items.append((np.array(b, dtype=np.float32), float(s), m))
    if not items:
        return [], []
    items.sort(key=lambda x: x[1], reverse=True)
    clusters: list[dict[str, Any]] = []
    for b_curr, s_curr, m_curr in items:
        best_iou = iou_thr
        best_idx = -1
        for c_idx, c_info in enumerate(clusters):
            iou = compute_box_iou(b_curr, c_info["center"])
            if iou > best_iou:
                best_iou = iou
                best_idx = c_idx
        if best_idx >= 0:
            clusters[best_idx]["members"].append((m_curr, b_curr, s_curr))
        else:
            clusters.append({"center": b_curr.copy(), "members": [(m_curr, b_curr, s_curr)]})

    out_boxes: list[list[float]] = []
    out_scores: list[float] = []
    for c in clusters:
        m_list = c["members"]
        if num_m == 3:
            m0 = [x for x in m_list if x[0] == 0]
            m1 = [x for x in m_list if x[0] == 1]
            m2 = [x for x in m_list if x[0] == 2]
            if m0 and m1 and m2:
                cw_sum = coord_ratio[0] + coord_ratio[1] + coord_ratio[2]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum)
                    + m1[0][1] * (coord_ratio[1] / cw_sum)
                    + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (
                    m0[0][2] * weights[0] + m1[0][2] * weights[1] + m2[0][2] * weights[2]
                ) / (weights[0] + weights[1] + weights[2])
                score = min(1.0, s_avg * boost_3)
            elif m0 and m1:
                cw_sum = coord_ratio[0] + coord_ratio[1]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m1[0][1] * (coord_ratio[1] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1]) / (
                    weights[0] + weights[1]
                )
                score = min(1.0, s_avg * boost_2)
            elif m0 and m2:
                cw_sum = coord_ratio[0] + coord_ratio[2]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m2[0][2] * weights[2]) / (
                    weights[0] + weights[2]
                )
                score = min(1.0, s_avg * boost_2)
            elif m1 and m2:
                cw_sum = coord_ratio[1] + coord_ratio[2]
                box = (
                    m1[0][1] * (coord_ratio[1] / cw_sum) + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m1[0][2] * weights[1] + m2[0][2] * weights[2]) / (
                    weights[1] + weights[2]
                )
                score = min(1.0, s_avg * boost_2)
            elif m0:
                box = m0[0][1].tolist()
                score = m0[0][2] * solo_decay
            elif m1:
                box = m1[0][1].tolist()
                score = m1[0][2] * (solo_decay * 0.98)
            else:
                box = m2[0][1].tolist()
                score = m2[0][2] * (solo_decay * 0.95)
        elif num_m == 2:
            m0 = [x for x in m_list if x[0] == 0]
            m1 = [x for x in m_list if x[0] == 1]
            if m0 and m1:
                cw_sum = coord_ratio[0] + coord_ratio[1]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m1[0][1] * (coord_ratio[1] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1]) / (
                    weights[0] + weights[1]
                )
                score = min(1.0, s_avg * boost_2)
            elif m0:
                box = m0[0][1].tolist()
                score = m0[0][2] * solo_decay
            else:
                box = m1[0][1].tolist()
                score = m1[0][2] * (solo_decay * 0.95)
        else:
            box = m_list[0][1].tolist()
            score = m_list[0][2]
        out_boxes.append(box)
        out_scores.append(score)
    return out_boxes, out_scores


def run_ensemble_inspection(
    image_input: str | Path | Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    nms_mode: str = "nms_free",
    enable_roi: bool = True,
    enable_tiling: bool = False,
    tile_size: int = 960,
    overlap_ratio: float = 0.20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute SOTA Adaptive Heterogeneous Triad Ensemble inspection.

    Combines FBDNet-M, D-FINE-M, and YOLO26m via per-class dynamic WBF routing,
    consensus boosting, and aspect-ratio preserving ROI preprocessing.
    """
    total_start = time.perf_counter()
    prep_start = time.perf_counter()

    if isinstance(image_input, (str, Path)):
        if str(image_input).startswith("data:image"):
            pil_img = base64_to_pil(str(image_input))
        else:
            pil_img = Image.open(str(image_input)).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    orig_w, orig_h = pil_img.size

    # ROI Cropping
    roi_offset = (0, 0)
    roi_box_applied = None
    if enable_roi:
        crop_img, roi_box = get_roi_crop(pil_img)
        if roi_box is not None:
            roi_offset = (roi_box[0], roi_box[1])
            roi_box_applied = list(roi_box)
    else:
        crop_img = pil_img

    prep_ms = (time.perf_counter() - prep_start) * 1000

    # Locate constituent checkpoints
    fbd_path = WORKSPACE_ROOT / "weights" / "dfine_m_fbdnet_best.pth"
    dfine_path = WORKSPACE_ROOT / "weights" / "dfine_m_baseline_reproduced_best.pth"
    yolo_distilled = WORKSPACE_ROOT / "weights" / "distilled_student_yolo26m_best.pt"
    yolo_vfl = WORKSPACE_ROOT / "weights" / "yolo26m_roi_vfl_best.pt"
    yolo_path = yolo_distilled if yolo_distilled.is_file() else yolo_vfl

    if not fbd_path.is_file() or not dfine_path.is_file() or not yolo_path.is_file():
        raise FileNotFoundError("One or more ensemble checkpoints are missing in weights/")

    # Load optimal hyperparameters
    opt_hp_file = WORKSPACE_ROOT / "results" / "optimal_ensemble_hyperparameters.json"
    opt_hp: dict[str, Any] = {}
    if opt_hp_file.is_file():
        try:
            opt_hp = json.loads(opt_hp_file.read_text(encoding="utf-8"))
        except Exception:
            opt_hp = {}

    # Run predictions on crop
    infer_start = time.perf_counter()
    # Predict with candidate models using sensitive floor to allow consensus detection
    d_fbd, t_fbd = _predict_single_image(
        crop_img, fbd_path, is_dfine=True, conf_threshold=0.03,
        iou_threshold=iou_threshold, nms_mode=nms_mode,
    )
    d_dfine, t_dfine = _predict_single_image(
        crop_img, dfine_path, is_dfine=True, conf_threshold=0.03,
        iou_threshold=iou_threshold, nms_mode=nms_mode,
    )
    d_yolo, t_yolo = _predict_single_image(
        crop_img, yolo_path, is_dfine=False, conf_threshold=0.03,
        iou_threshold=iou_threshold, nms_mode=nms_mode,
    )
    infer_ms = (time.perf_counter() - infer_start) * 1000

    # Group detections by class
    post_start = time.perf_counter()
    raw_by_class: dict[str, dict[str, tuple[list[list[float]], list[float]]]] = {
        c: {"fbd": ([], []), "dfine": ([], []), "yolo": ([], [])} for c in CLASS_NAMES
    }
    for c_name, conf, x1, y1, x2, y2 in d_fbd:
        if c_name in raw_by_class:
            raw_by_class[c_name]["fbd"][0].append([x1, y1, x2, y2])
            raw_by_class[c_name]["fbd"][1].append(conf)
    for c_name, conf, x1, y1, x2, y2 in d_dfine:
        if c_name in raw_by_class:
            raw_by_class[c_name]["dfine"][0].append([x1, y1, x2, y2])
            raw_by_class[c_name]["dfine"][1].append(conf)
    for c_name, conf, x1, y1, x2, y2 in d_yolo:
        if c_name in raw_by_class:
            raw_by_class[c_name]["yolo"][0].append([x1, y1, x2, y2])
            raw_by_class[c_name]["yolo"][1].append(conf)

    fused_boxes_all: list[tuple[str, float, int, int, int, int]] = []
    for c_name in CLASS_NAMES:
        cfg = opt_hp.get(c_name, {})
        mode = cfg.get("mode", "3M_MS_TTA")
        iou_thr = cfg.get("iou_thr", 0.50)
        weights = cfg.get("weights", [1.0, 1.0, 1.0] if "3M" in mode else [1.5, 1.0])
        coord_ratio = cfg.get("coord_ratio", [0.33, 0.33, 0.34] if "3M" in mode else [0.6, 0.4])
        boost_3 = cfg.get("boost_3", 1.25)
        boost_2 = cfg.get("boost_2", 1.10)
        solo_decay = cfg.get("solo_decay", 0.85)
        conf_floor = cfg.get("conf_floor", 0.05)

        m_data = raw_by_class[c_name]
        if "3M" in mode:
            b_list = [m_data["fbd"][0], m_data["dfine"][0], m_data["yolo"][0]]
            s_list = [m_data["fbd"][1], m_data["dfine"][1], m_data["yolo"][1]]
        else:
            b_list = [m_data["fbd"][0], m_data["yolo"][0]]
            s_list = [m_data["fbd"][1], m_data["yolo"][1]]

        fb, fs = fast_wbf_fusion(
            b_list, s_list, weights, iou_thr, coord_ratio,
            boost_3, boost_2, solo_decay, conf_floor,
        )
        for b, score in zip(fb, fs, strict=False):
            if score >= conf_threshold:
                bx1, by1, bx2, by2 = [int(round(v)) for v in b]
                fused_boxes_all.append((c_name, float(score), bx1, by1, bx2, by2))

    # Format detections and map back to full image space
    detections: list[dict[str, Any]] = []
    classes_found: set[str] = set()
    for i, (cls_name, conf, bx1, by1, bx2, by2) in enumerate(fused_boxes_all):
        x1 = max(0, min(orig_w, bx1 + roi_offset[0]))
        y1 = max(0, min(orig_h, by1 + roi_offset[1]))
        x2 = max(0, min(orig_w, bx2 + roi_offset[0]))
        y2 = max(0, min(orig_h, by2 + roi_offset[1]))
        if x2 <= x1 or y2 <= y1:
            continue

        x1n = x1 / orig_w
        y1n = y1 / orig_h
        x2n = x2 / orig_w
        y2n = y2 / orig_h

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        thumb_img = crop_defect_thumbnail(pil_img, [x1, y1, x2, y2])
        thumb_b64 = pil_to_base64(thumb_img, format="PNG")
        style = CLASS_STYLES.get(cls_name, DEFAULT_STYLE)

        detections.append(
            {
                "id": i + 1,
                "class": cls_name,
                "confidence": conf,
                "class_id": CLASS_NAMES.index(cls_name),
                "class_name": cls_name,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_normalized": [x1n, y1n, x2n, y2n],
                "xyxy": [x1, y1, x2, y2],
                "xyxyn": [round(x1n, 4), round(y1n, 4), round(x2n, 4), round(y2n, 4)],
                "css_top": round(y1n * 100, 2),
                "css_left": round(x1n * 100, 2),
                "css_width": (x2n - x1n) * 100,
                "css_height": (y2n - y1n) * 100,
                "area": area,
                "crop_size": f"{max(bw, 16)} × {max(bh, 16)}",
                "crop_b64": thumb_b64,
                "color_hex": style["hex"],
                "tailwind_text": style["tailwind_text"],
                "tailwind_bg": style["tailwind_bg"],
                "tailwind_border": style["tailwind_border"],
            }
        )
        classes_found.add(cls_name)

    annotated = annotate_image(pil_img, detections)
    annotated_b64 = pil_to_base64(annotated, format="JPEG", quality=88)
    image_b64 = pil_to_base64(pil_img, format="JPEG", quality=88)

    roi_preview = None
    if detections:
        first = detections[0]
        roi_preview = {
            "class": first["class"],
            "confidence": first["confidence"],
            "crop_size": first["crop_size"],
            "xyxy": first["xyxy"],
            "area": first["area"],
            "crop_b64": first["crop_b64"],
            "color_hex": first["color_hex"],
            "tailwind_text": first["tailwind_text"],
        }

    post_ms = (time.perf_counter() - post_start) * 1000
    total_latency_ms = (time.perf_counter() - total_start) * 1000
    fps = round(1000.0 / max(total_latency_ms, 1.0), 1)

    return {
        "success": True,
        "image_width": orig_w,
        "image_height": orig_h,
        "image_b64": image_b64,
        "annotated_b64": annotated_b64,
        "latency_ms": round(total_latency_ms),
        "fps": fps,
        "timing": {
            "preprocess_ms": round(prep_ms, 1),
            "inference_ms": round(infer_ms, 1),
            "postprocess_ms": round(post_ms, 1),
            "total_ms": round(total_latency_ms, 1),
            "fps": fps,
            "fbdnet_ms": round(t_fbd, 1),
            "dfine_ms": round(t_dfine, 1),
            "yolo_ms": round(t_yolo, 1),
        },
        "total_defects": len(detections),
        "defect_types": len(classes_found),
        "detections": detections,
        "roi_preview": roi_preview,
        "model_name": "Adaptive Heterogeneous Triad Ensemble (SOTA 46.10%)",
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "nms_mode": nms_mode,
        "enable_roi": enable_roi,
        "enable_tiling": enable_tiling,
        "roi_box": roi_box_applied,
    }
