"""Unit tests for Data Quality and Drift Guard."""

from __future__ import annotations

import numpy as np
from PIL import Image

from glass_inspector.data_guard import analyze_image_quality


def test_normal_image_passes_guard():
    # Create synthetic natural-like image with contrast and edges
    arr = np.random.randint(60, 180, (200, 200, 3), dtype=np.uint8)
    # Add clear edge patterns
    arr[50:150, 50:150] = 210
    img = Image.fromarray(arr)

    res = analyze_image_quality(img)
    assert res["is_acceptable"] is True
    assert res["status"] in {"PASS", "WARNING"}
    assert "brightness" in res["metrics"]
    assert "contrast" in res["metrics"]
    assert "sharpness" in res["metrics"]


def test_underexposed_image_triggers_warning():
    # Extremely dark image (mean brightness < 15)
    dark_arr = np.full((100, 100, 3), 10, dtype=np.uint8)
    img = Image.fromarray(dark_arr)

    res = analyze_image_quality(img)
    assert res["status"] == "WARNING"
    assert any("under-exposed" in w.lower() for w in res["warnings"])
    assert res["metrics"]["brightness"] < 35.0


def test_overexposed_image_triggers_warning():
    # Extremely bright washed-out image (mean brightness > 235)
    bright_arr = np.full((100, 100, 3), 245, dtype=np.uint8)
    img = Image.fromarray(bright_arr)

    res = analyze_image_quality(img)
    assert res["status"] == "WARNING"
    assert any("over-exposed" in w.lower() for w in res["warnings"])
    assert res["metrics"]["brightness"] > 225.0


def test_flat_contrast_image_triggers_warning():
    # Completely uniform gray image (contrast std == 0)
    flat_arr = np.full((100, 100), 120, dtype=np.uint8)

    res = analyze_image_quality(flat_arr)
    assert res["status"] == "WARNING"
    assert any("low contrast" in w.lower() for w in res["warnings"])
    assert res["metrics"]["contrast"] < 5.0


def test_empty_image_rejected():
    empty_arr = np.zeros((0, 0), dtype=np.uint8)
    res = analyze_image_quality(empty_arr)
    assert res["is_acceptable"] is False
    assert res["status"] == "REJECT"
