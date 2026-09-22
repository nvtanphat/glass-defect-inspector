from pathlib import Path

from scale_error.config import load_yaml


def test_exactly_one_training_config():
    configs = list(Path("configs").glob("*.yaml"))
    assert len(configs) == 1
    assert configs[0].name == "train.yaml"


def test_accuracy_first_config():
    cfg = load_yaml("configs/train.yaml")
    assert cfg["model"] == "yolo26m.pt"
    assert cfg["imgsz"] == 960
    assert cfg["epochs"] == 120
    assert cfg["optimizer"] == "MuSGD"
    assert cfg["patience"] == 15
