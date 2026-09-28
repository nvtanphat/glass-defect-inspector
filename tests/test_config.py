from pathlib import Path

from scale_error.config import load_yaml


def test_training_configs_exist():
    assert Path("configs/train_yolo26m.yaml").exists()
    assert Path("configs/train_rtdetr.yaml").exists()


def test_accuracy_first_config():
    cfg = load_yaml("configs/train_yolo26m.yaml")
    assert cfg["model"] == "yolo26m.pt"
    assert cfg["preprocessing"] == "roi_crop"
    assert Path(cfg["roi_config"]).exists()
    assert cfg["use_vfl"] is True
    assert cfg["use_class_weights"] is True
    assert cfg["imgsz"] == 960
    assert cfg["epochs"] == 100
    assert cfg["optimizer"] == "MuSGD"
    assert cfg["patience"] == 15


def test_baselines_do_not_enable_yolo26_loss_recipe():
    for path in ("configs/train_yolo11m.yaml", "configs/train_rtdetr.yaml"):
        cfg = load_yaml(path)
        assert cfg["preprocessing"] == "raw"
        assert cfg["use_vfl"] is False
        assert cfg["use_class_weights"] is False
