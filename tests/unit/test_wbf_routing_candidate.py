from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[2] / "kaggle/kernel_wbf_routing_candidate/run_wbf_routing.py"
)
SPEC = importlib.util.spec_from_file_location("wbf_routing_candidate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample_streams():
    # D-FINE is confidently wrong; selecting the FBD/YOLO route must remove it.
    def packed(box, score):
        return [([box], [score])] + [([], []) for _ in range(4)]

    streams = {
        "f_orig": packed([0, 0, 10, 10], 0.8),
        "d_orig": packed([100, 100, 110, 110], 0.99),
        "y_orig": packed([0, 0, 10, 10], 0.7),
    }
    streams.update(
        f_tta=streams["f_orig"],
        d_tta=streams["d_orig"],
        y_tta=streams["y_orig"],
        y_ms_tta=streams["y_orig"],
        y_ms_orig=streams["y_orig"],
    )
    return {1: streams}


def test_frozen_two_model_route_excludes_disabled_model():
    config = {
        "mode": "fbd_yolo_original",
        "weights": [1, 10, 1],
        "iou_thr": 0.5,
        "conf_floor": 0.01,
        "consensus_boost": 1.0,
    }
    predictions = MODULE.generate_class_predictions_fast(0, config, sample_streams())
    assert len(predictions) == 1
    assert predictions[0]["bbox"] == pytest.approx([0, 0, 10, 10])
    assert predictions[0]["score"] == pytest.approx(0.75)


def test_old_three_model_config_retains_baseline_behavior():
    config = {"weights": [1, 1, 1], "iou_thr": 0.5, "conf_floor": 0.01, "consensus_boost": 1.0}
    predictions = MODULE.generate_class_predictions_fast(0, config, sample_streams())
    assert len(predictions) == 2
    assert max(p["score"] for p in predictions) == pytest.approx(0.99)


def test_unknown_route_fails_closed():
    with pytest.raises(ValueError, match="Unknown frozen routing mode"):
        MODULE.generate_class_predictions_fast(0, {"mode": "typo"}, sample_streams())


def test_routing_tuner_rejects_test_before_search(monkeypatch):
    def forbidden(*args):
        pytest.fail("Baseline search must not run on Test")

    monkeypatch.setattr(MODULE, "_tune_baseline_wbf", forbidden)
    bundle = MODULE.GroundTruthBundle("test", Path("test_ground_truth.json"), {})
    with pytest.raises(ValueError, match="Data Leakage Detected"):
        MODULE.tune_wbf_hyperparameters({}, bundle)


def test_cache_roundtrip_and_wrong_split_rejection(tmp_path):
    path = tmp_path / "val_streams.json.gz"
    metadata = {"split": "val", "image_ids": [1], "checkpoint_sha256": {"fbd": "abc"}}
    MODULE.save_prediction_cache(path, metadata, sample_streams())
    loaded = MODULE.load_prediction_cache(path, metadata)
    config = {
        "mode": "triad_ms_tta",
        "weights": [1, 1, 1],
        "iou_thr": 0.5,
        "conf_floor": 0.01,
        "consensus_boost": 1.0,
    }
    assert MODULE.generate_class_predictions_fast(
        0, config, loaded
    ) == MODULE.generate_class_predictions_fast(0, config, sample_streams())
    with pytest.raises(ValueError, match="provenance differs"):
        MODULE.load_prediction_cache(path, dict(metadata, split="test"))
    with pytest.raises(ValueError, match="provenance differs"):
        MODULE.load_prediction_cache(path, dict(metadata, checkpoint_sha256={"fbd": "changed"}))


def test_stability_subsets_are_disjoint_and_retain_all_annotations():
    coco = {
        "images": [{"id": i, "file_name": f"val_{i}.jpg"} for i in range(10)],
        "annotations": [{"id": i, "image_id": i, "category_id": 0} for i in range(10)],
    }
    bundle = MODULE.GroundTruthBundle("val", Path("val_ground_truth.json"), coco)
    folds = MODULE.routing_stability_folds(bundle)
    ids = [{i["id"] for i in f.coco["images"]} for f in folds]
    assert not ids[0] & ids[1]
    assert ids[0] | ids[1] == set(range(10))
    assert sum(len(f.coco["annotations"]) for f in folds) == 10
    assert all(a["image_id"] in ids[n] for n, f in enumerate(folds) for a in f.coco["annotations"])


def test_candidate_defaults_to_validation_only():
    assert MODULE.parse_args([]).phase == "tune"


@pytest.mark.parametrize("bad_dfine", [True, False])
def test_routing_acceptance_and_stop_use_actual_coco_ap(tmp_path, monkeypatch, bad_dfine):
    pytest.importorskip("pycocotools")
    images = [
        {"id": i, "file_name": f"val_{i}.jpg", "width": 512, "height": 512} for i in range(12)
    ]
    annotations = [
        {
            "id": i * 5 + c + 1,
            "image_id": i,
            "category_id": c,
            "bbox": [0, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        }
        for i in range(12)
        for c in range(5)
    ]
    coco = {
        "info": {"split": "val"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(MODULE.CLASS_NAMES)],
    }
    path = tmp_path / "val_ground_truth.json"
    path.write_text(json.dumps(coco), encoding="utf-8")
    bundle = MODULE.GroundTruthBundle("val", path, coco)
    streams = {}
    for i in range(12):
        entry = sample_streams()[1]
        if not bad_dfine:
            entry["d_orig"] = entry["f_orig"]
            entry["d_tta"] = entry["f_tta"]
        streams[i] = {key: [value[0] for _ in range(5)] for key, value in entry.items()}
    seed = {
        "source_split": "val",
        "frozen": True,
        "per_class": {
            name: {"weights": [1, 1, 1], "iou_thr": 0.5, "conf_floor": 0.01, "consensus_boost": 1.0}
            for name in MODULE.CLASS_NAMES
        },
    }
    monkeypatch.setattr(MODULE, "_tune_baseline_wbf", lambda *args: seed)
    artifact = MODULE.tune_wbf_hyperparameters(streams, bundle)
    evidence = artifact["routing_ablation"]
    assert evidence["accepted_after_global_detection_cap"] is bad_dfine
    assert evidence["candidate_assembled_val"]["mAP50_95"] == pytest.approx(100)
    if bad_dfine:
        assert (
            evidence["candidate_assembled_val"]["mAP50_95"]
            > evidence["baseline_assembled_val"]["mAP50_95"]
        )
        assert {cfg["mode"] for cfg in artifact["per_class"].values()} == {"fbd_yolo_original"}
    else:
        assert evidence["baseline_assembled_val"]["mAP50_95"] == pytest.approx(100)
        assert all("mode" not in config for config in artifact["per_class"].values())
