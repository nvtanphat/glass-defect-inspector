"""Installed application assets and configurable external runtime resources."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CHECKOUT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_ROOT = CHECKOUT_ROOT if (CHECKOUT_ROOT / "pyproject.toml").is_file() else Path.cwd()
WORKSPACE_ROOT = Path(os.getenv("INSPECTOR_ROOT", str(DEFAULT_ROOT))).resolve()
WEB_DIR = PACKAGE_DIR / "web"
SAMPLES_DIR = PACKAGE_DIR / "sample_images"
