"""Bounded public contracts and conservative quality policy."""

from __future__ import annotations

import base64
import binascii
import io
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_BATCH = 4
MIMES = {"image/jpeg": "JPEG", "image/png": "PNG", "image/bmp": "BMP"}


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=1, max_length=512)


class DetectionResult(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    class_id: int = Field(ge=0, le=4)
    class_name: Literal["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]
    model_id: str
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def valid_geometry(self):
        for bbox in (self.bbox_xyxy, self.bbox_normalized):
            if not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3]):
                raise ValueError("Bounding box must have positive area")
        if any(v > 1 for v in self.bbox_normalized):
            raise ValueError("Normalized box must be inside original image")
        names = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
        if names[self.class_id] != self.class_name:
            raise ValueError("Class ID and name disagree")
        return self


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    success: bool
    inspection_id: str
    model_id: str
    model_version: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    conclusion: Literal["PASS", "REJECT", "REVIEW"]
    detections: list[DetectionResult]


class InspectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=80)
    conf_threshold: float = Field(default=0.25, ge=0.01, le=1, allow_inf_nan=False)
    iou_threshold: float = Field(default=0.45, ge=0.01, le=1, allow_inf_nan=False)
    nms_mode: Literal["nms_free", "standard_nms"] | None = None
    enable_roi: bool | None = None
    enable_tiling: bool = False
    tile_size: int = Field(default=960, ge=640, le=1280)
    overlap_ratio: float = Field(default=0.2, ge=0, le=0.3, allow_inf_nan=False)


class PredictRequest(InspectionConfig):
    image_data: str = Field(max_length=14_000_000)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conclusion: Literal["PASS", "REJECT", "REVIEW"]
    reviewer: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=3, max_length=2000)
    expected_version: int = Field(ge=0)

    @field_validator("reviewer", "reason")
    @classmethod
    def nonblank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Must not be blank")
        return value


def decode_image(raw: bytes, mime: str) -> Image.Image:
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("File exceeds 10 MiB")
    if mime not in MIMES:
        raise ValueError("Only JPEG, PNG and BMP are supported")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            w, h = image.size
            if w < 32 or h < 32 or w > 10000 or h > 10000 or w * h > MAX_PIXELS:
                raise ValueError("Image dimensions exceed limits (32..10000, <=20M pixels)")
            if image.format != MIMES[mime] or getattr(image, "n_frames", 1) != 1:
                raise ValueError("MIME does not match decoded image or image is animated")
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Corrupt or unsupported image") from exc


def decode_data_url(value: str):
    try:
        header, encoded = value.split(",", 1)
        if not header.startswith("data:") or not header.endswith(";base64"):
            raise ValueError("A base64 image data URL is required")
        mime = header[5:-7]
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid image data URL; server paths are not accepted") from exc
    return raw, decode_image(raw, mime)


def quality_policy(results):
    if any(r["detections"] for r in results):
        return "REJECT", "Detected glass defects above configured confidence threshold."
    return (
        "REVIEW",
        (
            "No defect detected; absence of detection does not prove quality. "
            "Technician review required."
        ),
    )
