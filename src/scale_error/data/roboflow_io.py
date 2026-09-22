from __future__ import annotations

import os
from pathlib import Path

from roboflow import Roboflow


def get_roboflow_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "ROBOFLOW_API_KEY not found. Set it as an environment variable or in .env."
    )


def download_dataset(
    output_dir: str | Path,
    workspace: str = "datacpn",
    project: str = "scale_error_dc-orqpx",
    version: int = 26,
    model_format: str = "yolov8",
    overwrite: bool = False,
) -> Path:
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
