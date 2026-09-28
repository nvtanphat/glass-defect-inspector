from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR, YOLO
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

from scale_error.config import load_yaml
from scale_error.training.losses import VFLDetectionLoss, compute_class_weights


def _attach_vfl_criterion(class_weights):
    """Build an on_train_start callback that swaps in Varifocal Loss + class weights.

    `model.train(...)` rebuilds `model.model` from scratch inside the trainer
    (ultralytics/engine/model.py Model.train -> Trainer.get_model), so setting
    a criterion before calling `.train()` is silently discarded. `on_train_start`
    fires after `Trainer._setup_train()` has built and device-placed the real
    model (ultralytics/engine/trainer.py), so the attached criterion survives.
    """

    def _callback(trainer) -> None:
        det_model = trainer.model
        if class_weights is not None:
            det_model.class_weights = class_weights.to(next(det_model.parameters()).device)

        existing_criterion = det_model.init_criterion()
        if isinstance(existing_criterion, E2ELoss):
            det_model.criterion = E2ELoss(det_model, loss_fn=VFLDetectionLoss)
        elif isinstance(existing_criterion, v8DetectionLoss):
            det_model.criterion = VFLDetectionLoss(det_model)
        else:
            raise TypeError(
                f"Unsupported criterion {type(existing_criterion).__name__}; "
                "VFLDetectionLoss only wraps v8DetectionLoss/E2ELoss."
            )

    return _callback


def train(
    data_yaml: str | Path,
    config_path: str | Path,
    project_dir: str | Path,
    run_name: str | None = None,
    dataset_report: str | Path | None = None,
):
    cfg = load_yaml(config_path)
    model_name = str(cfg.pop("model"))
    preprocessing = str(cfg.pop("preprocessing", "raw"))
    roi_config = cfg.pop("roi_config", None)
    use_vfl = bool(cfg.pop("use_vfl", False))
    use_class_weights = bool(cfg.pop("use_class_weights", False))
    project_dir = Path(project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model_cls = RTDETR if model_name.startswith("rtdetr") else YOLO
    model = model_cls(model_name)

    if use_vfl and model_cls is not YOLO:
        raise ValueError("VFLDetectionLoss is only supported for YOLO models")
    if use_class_weights and not use_vfl:
        raise ValueError("Class weights require use_vfl=true")
    if preprocessing not in {"raw", "roi_crop"}:
        raise ValueError(f"Unsupported preprocessing mode: {preprocessing}")
    if preprocessing == "roi_crop":
        if roi_config is None or not Path(roi_config).exists():
            raise FileNotFoundError(
                "preprocessing=roi_crop requires an existing roi_config path"
            )
    elif roi_config is not None:
        raise ValueError("roi_config is only valid with preprocessing=roi_crop")

    if use_vfl:
        report_path = None
        if dataset_report is not None and Path(dataset_report).exists():
            report_path = Path(dataset_report)
        else:
            # Auto-detect dataset_report.json from standard locations
            data_yaml_p = Path(data_yaml).resolve()
            candidates = [
                data_yaml_p.parent / "dataset_report.json",
                data_yaml_p.parent.parent / "dataset_report.json",
                Path("data/processed_roi/dataset_report.json"),
                Path("data/processed/dataset_report.json"),
                Path("artifacts/dataset_report.json"),
            ]
            for cand in candidates:
                if cand.exists():
                    report_path = cand
                    break

        if use_class_weights and report_path is None:
            raise FileNotFoundError(
                "use_class_weights=true but dataset_report.json was not found; "
                "pass --dataset-report explicitly"
            )

        class_weights = compute_class_weights(report_path) if report_path else None
        model.add_callback("on_train_start", _attach_vfl_criterion(class_weights))

    return model.train(
        data=str(Path(data_yaml).resolve()),
        project=str(project_dir),
        name=run_name or Path(model_name).stem.replace("-", "_"),
        exist_ok=True,
        **cfg,
    )
