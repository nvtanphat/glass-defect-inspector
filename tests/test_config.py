from pathlib import Path

from scale_error.config import load_yaml


def test_training_configs_exist():
    assert Path("configs/train_yolo26m.yaml").exists()
    assert Path("configs/train_rtdetr.yaml").exists()


def test_accuracy_first_config():
    cfg = load_yaml("configs/train_yolo26m.yaml")
    assert cfg["model"] == "yolo26m.pt"
    assert cfg["imgsz"] == 960
    assert cfg["epochs"] == 120
    assert cfg["optimizer"] == "MuSGD"
    assert cfg["patience"] == 15
