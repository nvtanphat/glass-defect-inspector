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

    config_path = dfine_dir / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_glass.yml"
    cfg = YAMLConfig(str(config_path), resume=weights_path)
    if "fbdnet" in Path(weights_path).name:
        # Use checkpoint's original module implementation, never substitute D-FINE.
        cfg.yaml_cfg["DFINE"].update(backbone="HGNetv2FBD", encoder="FBDHybridEncoder")
        cfg.yaml_cfg["HGNetv2FBD"] = dict(cfg.yaml_cfg["HGNetv2"], use_afta=True)
        cfg.yaml_cfg["FBDHybridEncoder"] = dict(
            cfg.yaml_cfg["HybridEncoder"], use_sfif=True, use_dcfsb=True
        )
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
