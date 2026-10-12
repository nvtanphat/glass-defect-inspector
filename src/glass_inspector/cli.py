"""Start the single-process HTTP application after applying migrations."""

import os

import uvicorn


def main():
    uvicorn.run(
        "glass_inspector.server:app",
        host=os.getenv("INSPECTOR_HOST", "127.0.0.1"),
        port=int(os.getenv("INSPECTOR_PORT", "8501")),
        workers=1,
    )
