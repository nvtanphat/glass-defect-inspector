from __future__ import annotations

import os
from pathlib import Path

from roboflow import Roboflow


def get_roboflow_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
    except Exception as exc:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not found. Set it as an environment variable or Kaggle Secret."
        ) from exc
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY is empty.")
    return key


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
