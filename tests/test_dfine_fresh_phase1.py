import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "kaggle"
    / "kernel_dfine_fresh_phase1"
    / "run_dfine_fresh_phase1.py"
)
SPEC = importlib.util.spec_from_file_location("run_dfine_fresh_phase1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fresh_phase1_uses_reference_training_schedule() -> None:
    assert MODULE.PHASE1_EPOCHS == 40
    assert MODULE.FINAL_TARGET_EPOCHS == 100
    assert MODULE.STOP_AUGMENT_EPOCH == 88
    assert MODULE.TRAIN_TOTAL_BATCH_SIZE == 8
    assert MODULE.MICRO_BATCH_SIZE == 4
    assert MODULE.ACCUMULATE_STEPS == 2


def test_reference_augmentation_patch() -> None:
    source = """                        {"type": "RandomPhotometricDistort", "p": 0.3},
                        {"type": "RandomIoUCrop", "p": 0.5},
                            "RandomPhotometricDistort",
                            "RandomIoUCrop",
"""
    patched = MODULE.patch_reference_augmentation(source)
    assert '"RandomPhotometricDistort", "p": 0.5' in patched
    assert '"RandomZoomOut", "fill": 0' in patched
    assert '"RandomIoUCrop", "p": 0.8' in patched
    assert patched.count('"RandomZoomOut"') == 2


def test_gradient_accumulation_patch() -> None:
    source = """import math
import sys
    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()
"""
    patched = MODULE.patch_gradient_accumulation(source)
    assert "DFINE_ACCUMULATE_STEPS" in patched
    assert "loss / accumulate_steps" in patched
    assert "if should_step and ema is not None" in patched
    assert "epoch_step=optimizer_steps" in patched
