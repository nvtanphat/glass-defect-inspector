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
    project_dir = Path(project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model_cls = RTDETR if model_name.startswith("rtdetr") else YOLO
    model = model_cls(model_name)

    if model_cls is YOLO and dataset_report is not None:
        class_weights = compute_class_weights(dataset_report)
        model.add_callback("on_train_start", _attach_vfl_criterion(class_weights))

    return model.train(
        data=str(Path(data_yaml).resolve()),
        project=str(project_dir),
        name=run_name or Path(model_name).stem.replace("-", "_"),
        exist_ok=True,
        **cfg,
    )
