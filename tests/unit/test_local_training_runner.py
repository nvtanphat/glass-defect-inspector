"""Check local launch behavior without training or accessing external services."""

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/train_fbdnet.py"
SPEC = importlib.util.spec_from_file_location("local_fbdnet_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.mark.parametrize("device,distributed", [("cpu", False), ("cuda:1", False), ("cuda", True)])
def test_launch_respects_selected_device(monkeypatch, tmp_path, device, distributed):
    runtime = tmp_path / "D-FINE"
    config = runtime / "configs/dfine/custom/fbdnet_m_e1_sanwd_20ep.yml"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    (runtime / "train.py").write_text("", encoding="utf-8")
    weights = tmp_path / "best.pth"
    weights.touch()
    monkeypatch.setattr(RUNNER, "DFINE_DIR", runtime)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    commands = []
    monkeypatch.setattr(RUNNER.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd))
    RUNNER.run_training("e1", weights, tmp_path / "output", epochs=3, device=device)
    assert len(commands) == 1
    command = commands[0]
    assert ("torch.distributed.run" in command) is distributed
    assert command[command.index("-t") + 1] == str(weights)
    assert "epochs=3" in command
    if not distributed:
        assert command[command.index("--device") + 1] == device


def test_missing_explicit_checkpoint_does_not_fall_back(tmp_path, monkeypatch):
    available = tmp_path / "default.pth"
    available.touch()
    monkeypatch.setattr(RUNNER, "DEFAULT_WEIGHTS", available)
    with pytest.raises(FileNotFoundError):
        RUNNER.locate_checkpoint(str(tmp_path / "missing.pth"))
