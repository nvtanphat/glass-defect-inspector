import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "kaggle"
    / "kernel_dfine_resume"
    / "run_dfine_resume.py"
)
SPEC = importlib.util.spec_from_file_location("run_dfine_resume", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
patch_solver_source = MODULE.patch_solver_source


def _upstream_fragment() -> str:
    return """            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / \"last.pth\"]
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        \"epoch\": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / \"best_stg1.pth\"))
                        print(f\"Refresh EMA at epoch {epoch} with decay {self.ema.decay}\")
"""


def test_patch_keeps_stage2_state_and_saves_last_checkpoint() -> None:
    patched = patch_solver_source(_upstream_fragment())

    assert "if self.output_dir:" in patched
    assert "Keep stage-2 state" in patched
    assert "self.load_resume_state" not in patched
    assert "self.ema.decay -= 0.0001" not in patched


def test_patch_rejects_an_unknown_upstream_solver() -> None:
    try:
        patch_solver_source("class DetSolver: pass")
    except RuntimeError as error:
        assert "checkpoint condition" in str(error)
    else:
        raise AssertionError("Unknown upstream source must not be patched silently")
