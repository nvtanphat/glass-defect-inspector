"""Mô-đun giao tiếp Roboflow API: Tải bộ dữ liệu Scale_Error_DC tự động."""

from __future__ import annotations

import os
from pathlib import Path

from roboflow import Roboflow


def get_roboflow_api_key() -> str:
    """Lấy API key của Roboflow từ biến môi trường ROBOFLOW_API_KEY."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "Không tìm thấy biến môi trường ROBOFLOW_API_KEY. Vui lòng thiết lập trong biến môi trường hoặc file .env."
    )


def download_dataset(
    output_dir: str | Path,
    workspace: str = "datacpn",
    project: str = "scale_error_dc-orqpx",
    version: int = 26,
    model_format: str = "yolov8",
    overwrite: bool = False,
) -> Path:
    """Tải tập dữ liệu từ Roboflow Universe về thư mục cục bộ.

    Tham số:
        output_dir: Thư mục lưu dữ liệu tải về.
        workspace: Tên workspace trên Roboflow (mặc định: 'datacpn').
        project: Tên dự án trên Roboflow (mặc định: 'scale_error_dc-orqpx').
        version: Phiên bản bộ dữ liệu (mặc định: 26 ứng với v26).
        model_format: Định dạng xuất nhãn (mặc định: 'yolov8' tương thích YOLO/DETR).
        overwrite: Có ghi đè dữ liệu cũ nếu đã tồn tại hay không.

    Trả về:
        Path: Đường dẫn thư mục dữ liệu đã tải thành công.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=get_roboflow_api_key())
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(model_format=model_format, location=str(output_dir), overwrite=overwrite)
    )
    return Path(dataset.location).resolve()
