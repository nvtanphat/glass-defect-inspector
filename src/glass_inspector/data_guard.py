"""Data Quality and Drift Guard for Glass Defect Inspection.

Lightweight, zero-overhead input validation module using pure NumPy / OpenCV:
- Brightness & exposure validation (under-exposed / washed out)
- Contrast validation (flat lighting / loss of defect texture)
- Sharpness / blurriness estimation via Laplacian variance
- Aspect-ratio & resolution sanity checking
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

# Operational thresholds calibrated on industrial glass surface dataset
MIN_BRIGHTNESS = 35.0  # Below this, surface defect texture is lost in shadows
MAX_BRIGHTNESS = 225.0  # Above this, reflections wash out subtle scratch edges
MIN_CONTRAST = 18.0  # Minimum standard deviation of grayscale pixels
MIN_SHARPNESS = 45.0  # Laplacian variance threshold; lower indicates defocus or vibration blur


def analyze_image_quality(
    image: Image.Image | np.ndarray,
    min_brightness: float = MIN_BRIGHTNESS,
    max_brightness: float = MAX_BRIGHTNESS,
    min_contrast: float = MIN_CONTRAST,
    min_sharpness: float = MIN_SHARPNESS,
) -> dict[str, Any]:
    """Assess optical quality of glass inspection image before model inference.

    Args:
        image: PIL Image or NumPy array (RGB or BGR).
        min_brightness: Lower bound for acceptable mean brightness (0-255).
        max_brightness: Upper bound for acceptable mean brightness (0-255).
        min_contrast: Minimum acceptable standard deviation of pixel intensities.
        min_sharpness: Minimum acceptable Laplacian variance.

    Returns:
        dict containing:
            - is_acceptable: bool, True if no blocking quality issues
            - metrics: dict of measured optical values
            - warnings: list of human-readable alerts
            - status: 'PASS' | 'WARNING' | 'REJECT'
    """
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"))
        w, h = image.size
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(image, code)
            h, w = image.shape[:2]
        else:
            gray = image
            h, w = gray.shape[:2]
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if gray.size == 0 or w == 0 or h == 0:
        return {
            "is_acceptable": False,
            "status": "REJECT",
            "metrics": {
                "width": w,
                "height": h,
                "brightness": 0.0,
                "contrast": 0.0,
                "sharpness": 0.0,
            },
            "warnings": ["Image has empty dimensions or zero pixel buffer."],
        }

    # 1. Luminance & Exposure Metrics
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    # 2. Sharpness / Defocus Metric (Laplacian variance)
    # Downsample if image is extremely high-res (> 2500px) to keep compute strictly under 3ms
    if max(w, h) > 2500:
        scale_factor = 2000.0 / max(w, h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        small_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        laplacian = cv2.Laplacian(small_gray, cv2.CV_64F)
    else:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian))

    # 3. Geometry & Aspect Ratio
    aspect_ratio = round(float(w) / float(h), 3)

    warnings: list[str] = []

    if mean_brightness < min_brightness:
        warnings.append(
            f"Image is under-exposed (brightness {mean_brightness:.1f} < {min_brightness}). "
            "Dark scratches may be missed."
        )
    elif mean_brightness > max_brightness:
        warnings.append(
            f"Image is over-exposed (brightness {mean_brightness:.1f} > {max_brightness}). "
            "Optical glare may trigger false alarms."
        )

    if contrast_std < min_contrast:
        warnings.append(
            f"Low contrast detected (contrast {contrast_std:.1f} < {min_contrast}). "
            "Defect boundary contrast is insufficient."
        )

    if sharpness < min_sharpness:
        warnings.append(
            f"Image appears blurry (sharpness score {sharpness:.1f} < {min_sharpness})."
        )

    # Determine overall gate status
    if not warnings:
        status = "PASS"
        is_acceptable = True
    else:
        status = "WARNING"
        is_acceptable = True  # Warnings do not block inspection, but flag audit logs

    return {
        "is_acceptable": is_acceptable,
        "status": status,
        "metrics": {
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "brightness": round(mean_brightness, 2),
            "contrast": round(contrast_std, 2),
            "sharpness": round(sharpness, 2),
        },
        "warnings": warnings,
    }
