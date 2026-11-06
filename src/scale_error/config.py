"""Mô-đun tiện ích đọc và ghi file cấu hình YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc file cấu hình YAML và trả về từ điển Python.

    Tham số:
        path: Đường dẫn tới file YAML cần đọc.

    Trả về:
        dict[str, Any]: Dữ liệu cấu hình đã phân tích.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Dữ liệu gốc của file YAML phải là một mapping/dict: {path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Ghi từ điển dữ liệu cấu hình ra file định dạng YAML.

    Tham số:
        data: Từ điển cấu hình cần lưu.
        path: Đường dẫn file YAML cần tạo.

    Trả về:
        Path: Đường dẫn tới file YAML đã lưu.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return path
