"""Server-owned model identities; READY requires a real-image inference probe."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import time
from pathlib import Path

from glass_inspector.inference_engine import (
    CLASS_NAMES,
    WORKSPACE_ROOT,
    load_dfine_model,
    load_yolo_model,
    run_ensemble_inspection,
    run_inspection,
)
from glass_inspector.paths import PACKAGE_DIR, SAMPLES_DIR

MODEL_SPECS = [
    (
        "apex-triad-ensemble",
        "Adaptive Heterogeneous Triad Ensemble (SOTA 46.10%)",
        "optimal_ensemble_hyperparameters.json",
        "ensemble",
        True,
    ),
    ("yolo11m", "YOLO11m", "yolo11m_best.pt", "yolo", False),
    ("yolo26m-base", "YOLO26m Base", "yolo26m_960_best.pt", "yolo", False),
    ("yolo26m-roi-vfl", "YOLO26m + ROI + VFL", "yolo26m_roi_vfl_best.pt", "yolo", True),
    ("rtdetr-l", "RT-DETR-L", "rtdetr_l_best.pt", "rtdetr", False),
    ("dfine-m", "D-FINE-M", "dfine_m_baseline_reproduced_best.pth", "dfine", True),
    ("fbdnet-m", "FBDNet-M", "dfine_m_fbdnet_best.pth", "fbdnet", True),
    (
        "yolo26m-distilled",
        "YOLO26m Distilled Student (Triad KD)",
        "distilled_student_yolo26m_best.pt",
        "yolo",
        True,
    ),
]


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, root: Path = WORKSPACE_ROOT):
        self.root = root
        self.records = {}
        for model_id, name, filename, adapter, roi in MODEL_SPECS:
            self.records[model_id] = dict(
                id=model_id,
                name=name,
                filename=filename,
                adapter=adapter,
                enable_roi=roi,
                input_size=[960, 960],
                nms_modes=["standard_nms"]
                if model_id == "yolo11m"
                else ["nms_free", "standard_nms"],
                default_nms="standard_nms" if model_id == "yolo11m" else "nms_free",
                class_names=CLASS_NAMES,
                status="UNAVAILABLE",
                reason="Not verified in this process",
                preprocessing="RGB; ROI by exact resolution; "
                + (
                    "Triad Heterogeneous Ensemble (FBDNet + D-FINE + YOLO)"
                    if adapter == "ensemble"
                    else ("square letterbox 960 / 255" if adapter == "yolo" else "resize 960 / 255")
                ),
                is_default=model_id == "apex-triad-ensemble",
            )

    def probe(self, image: Path | None = None):
        image = image or SAMPLES_DIR / "sample_bottle_1.jpg"
        import torch

        torch.set_num_threads(2)
        for record in self.records.values():
            if record["adapter"] == "ensemble":
                path = self.root / "results" / record["filename"]
            else:
                path = self.root / "weights" / record["filename"]
            if not path.is_file():
                record.update(status="UNAVAILABLE", reason="Trained checkpoint missing")
                continue
            record.update(
                sha256=checksum(path),
                bytes=path.stat().st_size,
                size_mb=round(path.stat().st_size / 1024**2, 1),
            )
            if not image.is_file():
                record.update(reason="Real smoke-test image missing")
                continue
            started = time.perf_counter()
            try:
                record["runtime_sources"] = {
                    "inference_sha256": checksum(PACKAGE_DIR / "inference_engine.py"),
                    "roi_sha256": checksum(self.root / "processing/roi_config.json"),
                }
                record["runtime_packages"] = {
                    name: importlib.metadata.version(name)
                    for name in ["torch", "torchvision", "ultralytics", "Pillow", "numpy"]
                }
                if record["adapter"] == "yolo":
                    model = load_yolo_model(str(path))
                    architecture = model.model.yaml
                    family = "yolo11" if record["id"] == "yolo11m" else "yolo26"
                    if architecture.get("scale") != "m" or family not in architecture.get(
                        "yaml_file", ""
                    ):
                        raise ValueError(
                            "Checkpoint architecture is not the registered medium model"
                        )
                    args = model.ckpt.get("train_args", {})
                    if args.get("imgsz") != 960:
                        raise ValueError(
                            "Checkpoint input size does not match registered 960px profile"
                        )
                    record["architecture"] = {
                        "yaml_file": architecture.get("yaml_file"),
                        "scale": "m",
                    }
                    record["provenance"] = {
                        k: args.get(k) for k in ("model", "data", "epochs", "name")
                    }
                if record["adapter"] == "rtdetr":
                    model = load_yolo_model(str(path))
                    architecture = model.model.yaml
                    if "rtdetr-l" not in architecture.get("yaml_file", ""):
                        raise ValueError("Checkpoint is not the registered RT-DETR-L architecture")
                    if model.ckpt.get("train_args", {}).get("imgsz") != 960:
                        raise ValueError("RT-DETR checkpoint input profile is not 960px")
                    record["architecture"] = dict(
                        yaml_file=architecture.get("yaml_file"), variant="L"
                    )
                if record["adapter"] in {"dfine", "fbdnet"}:
                    required = [
                        "calflops",
                        "transformers",
                        "loguru",
                        "faster_coco_eval",
                        "tensorboard",
                    ]
                    missing = [name for name in required if importlib.util.find_spec(name) is None]
                    if missing:
                        raise ImportError(
                            "D-FINE/FBDNet dependencies missing: " + ", ".join(missing)
                        )
                    source = self.root / "results/D-FINE"
                    record["runtime_sources"]["architecture_sources"] = {
                        str(p.relative_to(source)): checksum(p)
                        for p in sorted((source / "src").rglob("*.py"))
                    }
                    record["provenance"] = dict(
                        report="results/metrics_fbdnet_m.json"
                        if record["adapter"] == "fbdnet"
                        else "results/metrics_dfine_m.json",
                        limitation="Original training identity is not embedded in this checkpoint; "
                        "local report/config linkage, strict state load and smoke tested. "
                        "Historical COCO artifact is a different ROI variant.",
                    )
                if record["adapter"] == "ensemble":
                    record["architecture"] = {
                        "type": "Heterogeneous Triad Ensemble",
                        "components": ["FBDNet-M", "D-FINE-M", "YOLO26m-Distilled"],
                        "fusion": "Adaptive Per-Class Weighted Boxes Fusion (WBF)",
                        "target_mAP": "46.10%",
                    }
                    record["provenance"] = {
                        "report": "results/ensemble_benchmark_report.json",
                        "hyperparameters": "results/optimal_ensemble_hyperparameters.json",
                        "sota_mAP50_95": 0.4610,
                    }
                    result = run_ensemble_inspection(image, enable_roi=record["enable_roi"])
                else:
                    result = run_inspection(
                        image, path, enable_roi=record["enable_roi"], nms_mode=record["default_nms"]
                    )
                if record["adapter"] in {"dfine", "fbdnet"}:
                    model, _ = load_dfine_model(str(path))
                    expected = (
                        ("HGNetv2FBD", "FBDHybridEncoder")
                        if record["adapter"] == "fbdnet"
                        else ("HGNetv2", "HybridEncoder")
                    )
                    actual = (type(model.backbone).__name__, type(model.encoder).__name__)
                    if actual != expected or type(model.decoder).__name__ != "DFINETransformer":
                        raise ValueError("Loaded architecture does not match registered model")
                    record["architecture"] = dict(
                        backbone=actual[0],
                        encoder=actual[1],
                        decoder=type(model.decoder).__name__,
                        strict_state_load=True,
                        parameters=sum(p.numel() for p in model.parameters()),
                    )
                # Strict loaders check names/state shapes; validate all decoded outputs too.
                for d in result["detections"]:
                    if d["class_name"] != CLASS_NAMES[d["class_id"]]:
                        raise ValueError("Invalid class mapping")
                record.update(
                    status="READY",
                    reason=None,
                    smoke_image=str(image.relative_to(self.root))
                    if image.is_relative_to(self.root)
                    else f"package:sample_images/{image.name}",
                    smoke_defects=result["total_defects"],
                    smoke_latency_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}".replace(str(self.root), "<project>")
                record.update(status="UNAVAILABLE", reason=reason[:1500])
            print(f"Model {record['id']}: {record['status']}", flush=True)
        return self.public()

    def public(self):
        return list(self.records.values())

    def resolve(self, model_id: str):
        record = self.records.get(model_id)
        if record is None:
            raise ValueError("Unknown model_id; server paths are not accepted")
        if record["status"] != "READY":
            raise LookupError(f"Model unavailable: {record['reason']}")
        if record["adapter"] == "ensemble":
            path = self.root / "results" / record["filename"]
        else:
            path = self.root / "weights" / record["filename"]
        # Recheck bytes before every inference; version must identify the weights actually used.
        if not path.is_file() or checksum(path) != record["sha256"]:
            record.update(status="UNAVAILABLE", reason="Checkpoint changed after verification")
            raise LookupError(record["reason"])
        if (
            checksum(self.root / "processing/roi_config.json")
            != record["runtime_sources"]["roi_sha256"]
        ):
            record.update(
                status="UNAVAILABLE", reason="ROI configuration changed; restart and reverify"
            )
            raise LookupError(record["reason"])
        return record, path
