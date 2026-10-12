"""On-demand detector Grad-CAM using the saved image and actual class score."""

from __future__ import annotations

import io

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from glass_inspector.inference_engine import (
    CLASS_NAMES,
    compute_slice_intervals,
    load_dfine_model,
    load_yolo_model,
)

VERSION = "detector-gradcam-v2"


def select_query(logits, boxes, target_box=None, category=None):
    """Match a saved detection spatially; without one explain the strongest raw score."""
    if category is None:
        index = int(logits.detach().sigmoid().argmax())
        return index // logits.shape[1], index % logits.shape[1]
    boxes = boxes.detach()
    target = boxes.new_tensor(target_box)
    intersection = (
        (torch.minimum(boxes[:, 2:], target[2:]) - torch.maximum(boxes[:, :2], target[:2]))
        .clamp(min=0)
        .prod(1)
    )
    area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
    target_area = (target[2:] - target[:2]).clamp(min=0).prod()
    overlap = intersection / (area + target_area - intersection).clamp(min=1e-8)
    candidates = torch.where(overlap >= 0.5)[0]
    query = (
        candidates[logits.detach()[candidates, category].argmax()]
        if len(candidates)
        else overlap.argmax()
    )
    return int(query), category


def normalize_cam(features, gradients, size):
    maps = []
    gradient_sum = 0.0
    for feature, gradient in zip(features, gradients, strict=True):
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all() or not torch.isfinite(feature).all():
            raise ValueError("Nonfinite explanation")
        gradient_sum += float(gradient.detach().abs().sum())
        weights = gradient.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * feature).sum(1, keepdim=True).relu()
        maps.append(
            torch.nn.functional.interpolate(cam, size=size, mode="bilinear", align_corners=False)[
                0, 0
            ]
        )
    if not maps:
        raise ValueError("Model did not expose connected spatial features")
    cam = torch.stack(maps).sum(0).detach().cpu().numpy()
    spread = float(cam.max() - cam.min())
    return ((cam - cam.min()) / spread if spread > 1e-12 else np.zeros_like(cam)), {
        "flat_cam": spread <= 1e-12,
        "nonzero_gradient": gradient_sum > 0,
    }


def _xyxy(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)


def explain(image, weights, adapter, result, config):
    """Return an overlay JPEG and metadata; never update weights or quality decisions."""
    image = image.convert("RGB")
    detection = max(result.get("detections", []), key=lambda d: d["confidence"], default=None)
    roi = result.get("roi_box") or [0, 0, *image.size]
    crop = image.crop(tuple(roi))
    origin = [roi[0], roi[1]]
    if config.get("enable_tiling"):
        candidates = []
        for sy, ey in compute_slice_intervals(
            crop.height, config["tile_size"], config["overlap_ratio"]
        ):
            for sx, ex in compute_slice_intervals(
                crop.width, config["tile_size"], config["overlap_ratio"]
            ):
                if detection:
                    x1, y1, x2, y2 = detection["bbox_xyxy"]
                    overlap = max(0, min(ex + origin[0], x2) - max(sx + origin[0], x1)) * max(
                        0, min(ey + origin[1], y2) - max(sy + origin[1], y1)
                    )
                else:
                    overlap = 0
                candidates.append((overlap, (sx, sy, ex, ey)))
        _, tile = max(candidates, key=lambda entry: entry[0])
        crop = crop.crop(tile)
        origin[0] += tile[0]
        origin[1] += tile[1]
    target = (
        None
        if not detection
        else [
            detection["bbox_xyxy"][0] - origin[0],
            detection["bbox_xyxy"][1] - origin[1],
            detection["bbox_xyxy"][2] - origin[0],
            detection["bbox_xyxy"][3] - origin[1],
        ]
    )
    category = detection["class_id"] if detection else None
    features = []
    backbone_features = []
    backbone_hook = None
    if adapter in {"dfine", "fbdnet"}:
        from torchvision.transforms import Compose, Resize, ToTensor

        model, _ = load_dfine_model(str(weights))
        tensor = Compose([Resize((960, 960)), ToTensor()])(crop).unsqueeze(0)
        hook = model.backbone.register_forward_hook(
            lambda module, inputs, output: features.extend(output)
        )
        layer = "backbone outputs"
        padding = None
        if target:
            target = [
                target[0] / crop.width,
                target[1] / crop.height,
                target[2] / crop.width,
                target[3] / crop.height,
            ]
    else:
        wrapper = load_yolo_model(str(weights))
        model = wrapper.model
        head = model.model[-1]
        # Inference uses rect=False. Reuse its transform to preserve exact letterbox/resize.
        pixels = wrapper.predictor.pre_transform([np.asarray(crop)[:, :, ::-1]])[0]
        tensor = torch.from_numpy(np.ascontiguousarray(pixels[:, :, ::-1].transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).float() / 255
        hook = head.register_forward_pre_hook(lambda module, inputs: features.extend(inputs[0]))
        layer = "detection head input features"
        backbone_index = len(model.yaml["backbone"]) - 1
        backbone_layer = model.model[backbone_index]
        backbone_hook = backbone_layer.register_forward_hook(
            lambda module, inputs, output: backbone_features.append(output)
        )
        padding = None
        if adapter == "yolo":
            ratio = min(pixels.shape[1] / crop.width, pixels.shape[0] / crop.height)
            resized = (round(crop.width * ratio), round(crop.height * ratio))
            left = round((pixels.shape[1] - resized[0]) / 2 - 0.1)
            top = round((pixels.shape[0] - resized[1]) / 2 - 0.1)
            padding = (left, top, *resized)
            if target:
                target = [
                    target[0] * ratio + left,
                    target[1] * ratio + top,
                    target[2] * ratio + left,
                    target[3] * ratio + top,
                ]
        elif target:
            target = [
                target[0] / crop.width,
                target[1] / crop.height,
                target[2] / crop.width,
                target[3] / crop.height,
            ]
    try:
        parameter = next(model.parameters())
        with torch.inference_mode(False), torch.enable_grad():
            tensor = tensor.to(device=parameter.device, dtype=parameter.dtype).requires_grad_(True)
            output = model(tensor)
            if adapter in {"dfine", "fbdnet"}:
                logits = output["pred_logits"][0]
                boxes = _xyxy(output["pred_boxes"][0])
            elif adapter == "rtdetr":
                raw = output[1]
                logits = raw[1][-1, 0]
                boxes = _xyxy(raw[0][-1, 0])
            else:
                raw = output[1]
                if "one2one" in raw:
                    raw = raw["one2one" if head.end2end else "one2many"]
                logits = raw["scores"][0].transpose(0, 1)
                decoded = head._inference(raw)[0, :4].transpose(0, 1)
                boxes = decoded if head.end2end or head.xyxy else _xyxy(decoded)
            query, category = select_query(logits, boxes, target, category)
            score = logits[query, category]
            gradients = torch.autograd.grad(score, features + backbone_features, allow_unused=True)
            cam, metadata = normalize_cam(features, gradients[: len(features)], tensor.shape[-2:])
            if (
                metadata["flat_cam"]
                and backbone_features
                and any(gradient is not None for gradient in gradients[len(features) :])
            ):
                fallback, fallback_metadata = normalize_cam(
                    backbone_features, gradients[len(features) :], tensor.shape[-2:]
                )
                if not fallback_metadata["flat_cam"]:
                    cam, metadata = fallback, fallback_metadata
                    layer = f"backbone layer {backbone_index} ({type(backbone_layer).__name__})"
            probability = float(score.detach().sigmoid())
    finally:
        hook.remove()
        if backbone_hook is not None:
            backbone_hook.remove()
    if padding:
        left, top, width, height = padding
        cam = cam[top : top + height, left : left + width]
    cam = cv2.resize(cam, crop.size)
    heat = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    alpha = (cam * 0.55)[..., None]
    overlay = Image.fromarray(np.uint8(np.asarray(crop) * (1 - alpha) + heat * alpha))
    full = image.copy()
    full.paste(overlay, tuple(origin))
    if detection:
        ImageDraw.Draw(full).rectangle(detection["bbox_xyxy"], outline="#ffffff", width=3)
    stream = io.BytesIO()
    full.save(stream, "JPEG", quality=90)
    metadata.update(
        version=VERSION,
        target_class=CLASS_NAMES[category],
        score=probability,
        query=query,
        target_layer=layer,
        target_kind="detection" if detection else "below_threshold_candidate",
        explained_region=[*origin, origin[0] + crop.width, origin[1] + crop.height],
    )
    return stream.getvalue(), metadata
