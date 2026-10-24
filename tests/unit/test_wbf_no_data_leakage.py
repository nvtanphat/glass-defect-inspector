from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "kaggle"
    / "kernel_apex_hypersearch"
    / "run_apex_hypersearch.py"
)
SPEC = importlib.util.spec_from_file_location("wbf_val_test_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
WBF_SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WBF_SCRIPT
SPEC.loader.exec_module(WBF_SCRIPT)


def test_tuner_rejects_test_ground_truth_with_required_error() -> None:
    test_ground_truth = WBF_SCRIPT.GroundTruthBundle(
        split="test",
        source=Path("test_ground_truth.json"),
        coco={},
    )

    with pytest.raises(ValueError, match="^Data Leakage Detected!$"):
        WBF_SCRIPT.tune_wbf_hyperparameters({}, test_ground_truth)


def test_tuner_fails_closed_for_unprovenanced_ground_truth() -> None:
    with pytest.raises(ValueError, match="^Data Leakage Detected!$"):
        WBF_SCRIPT.tune_wbf_hyperparameters({}, {"images": [], "annotations": []})


def test_embedded_test_manifest_contains_only_test_filenames(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(WBF_SCRIPT, "TEST_IMAGE_MANIFEST_PATH", tmp_path / "not-attached.json")

    names = WBF_SCRIPT.load_test_image_names_only()

    assert len(names) == 890
    assert all(name.lower().endswith((".jpg", ".jpeg", ".png")) for name in names)
    assert (
        WBF_SCRIPT.test_image_manifest_sha256()
        == WBF_SCRIPT.hashlib.sha256(WBF_SCRIPT.test_image_manifest_bytes()).hexdigest()
    )


def test_parser_ignores_jupyter_kernel_connection_argument() -> None:
    args = WBF_SCRIPT.parse_args(["-f", "/tmp/kernel-connection.json", "--phase", "tune"])

    assert args.phase == "tune"


def test_script_initializes_without_file_global() -> None:
    module_name = "wbf_console_test"
    module = ModuleType(module_name)
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    sys.modules[module_name] = module

    try:
        exec(compile(source, str(SCRIPT_PATH), "exec"), module.__dict__)
    finally:
        del sys.modules[module_name]

    assert module.SCRIPT_DIR == Path.cwd()


def test_three_model_wbf_applies_consensus_boost_and_weighted_coordinates() -> None:
    boxes, scores = WBF_SCRIPT.fast_weighted_boxes_fusion_3m(
        boxes_list=[[[0, 0, 10, 10]], [[0, 0, 12, 10]], []],
        scores_list=[[0.8], [0.6], []],
        weights=[1.0, 1.0, 1.0],
        iou_thr=0.5,
        conf_floor=0.01,
        consensus_boost=1.2,
    )

    assert len(boxes) == len(scores) == 1
    assert boxes[0] == pytest.approx([0, 0, (10 * 0.8 + 12 * 0.6) / 1.4, 10])
    assert scores[0] == pytest.approx(0.84)


def test_yolo_val_labels_are_converted_to_split_tagged_coco(tmp_path, monkeypatch) -> None:
    from PIL import Image

    data_root = tmp_path / "dataset"
    image_root = data_root / "valid" / "images"
    label_root = data_root / "valid" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    (data_root / "data.yaml").write_text(
        "val: ../valid/images\nnames: ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']\n",
        encoding="utf-8",
    )
    Image.new("RGB", (1024, 1024)).save(image_root / "sample_val.jpg")
    Image.new("RGB", (1024, 1024)).save(image_root / "sample_test.jpg")
    (label_root / "sample_val.txt").write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (label_root / "sample_test.txt").write_text(
        "this Test label must not be read\n", encoding="utf-8"
    )
    manifest = tmp_path / "test_image_manifest.json"
    manifest.write_text(
        json.dumps({"split": "test", "image_file_names": ["sample_test.jpg"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(WBF_SCRIPT, "DATA_DIR", data_root)
    monkeypatch.setattr(WBF_SCRIPT, "TEST_IMAGE_MANIFEST_PATH", manifest)
    monkeypatch.setitem(WBF_SCRIPT.EXPECTED_IMAGE_COUNTS, "test", 1)

    output = WBF_SCRIPT.build_ground_truth_from_yolo_split("val", tmp_path / "output")
    coco = json.loads(output.read_text(encoding="utf-8"))

    assert output.name == "val_ground_truth.json"
    assert coco["info"]["split"] == "val"
    assert len(coco["images"]) == 1
    assert coco["images"][0]["file_name"] == "sample_val.jpg"
    assert coco["annotations"][0]["category_id"] == 2
    assert coco["annotations"][0]["bbox"] == pytest.approx([409.6, 307.2, 204.8, 409.6])


def test_combined_valid_folder_uses_test_filenames_only_and_never_reads_test_labels(
    tmp_path, monkeypatch
) -> None:
    from PIL import Image

    data_root = tmp_path / "dataset"
    image_root = data_root / "valid" / "images"
    label_root = data_root / "valid" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    (data_root / "data.yaml").write_text(
        "val: ../valid/images\nnames: ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']\n",
        encoding="utf-8",
    )
    Image.new("RGB", (1024, 1024)).save(image_root / "sample_val.jpg")
    Image.new("RGB", (1024, 1024)).save(image_root / "sample_test.jpg")
    (label_root / "sample_val.txt").write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (label_root / "sample_test.txt").write_text(
        "this Test label must not be read\n", encoding="utf-8"
    )
    manifest = tmp_path / "test_image_manifest.json"
    manifest.write_text(
        json.dumps({"split": "test", "image_file_names": ["sample_test.jpg"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(WBF_SCRIPT, "DATA_DIR", data_root)
    monkeypatch.setattr(WBF_SCRIPT, "TEST_IMAGE_MANIFEST_PATH", manifest)
    monkeypatch.setitem(WBF_SCRIPT.EXPECTED_IMAGE_COUNTS, "test", 1)

    output = WBF_SCRIPT.build_ground_truth_from_yolo_split("val", tmp_path / "output")
    coco = json.loads(output.read_text(encoding="utf-8"))

    assert [image["file_name"] for image in coco["images"]] == ["sample_val.jpg"]
    assert len(coco["annotations"]) == 1


def test_split_image_resolution_never_falls_back_to_another_split(tmp_path, monkeypatch) -> None:
    from PIL import Image

    data_root = tmp_path / "dataset"
    val_images = data_root / "valid" / "images"
    val_labels = data_root / "valid" / "labels"
    train_images = data_root / "train" / "images"
    val_images.mkdir(parents=True)
    val_labels.mkdir(parents=True)
    train_images.mkdir(parents=True)
    (data_root / "data.yaml").write_text(
        "val: ../valid/images\nnames: ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']\n",
        encoding="utf-8",
    )
    Image.new("RGB", (20, 20)).save(train_images / "leaked.jpg")
    manifest = tmp_path / "test_image_manifest.json"
    manifest.write_text(
        json.dumps({"split": "test", "image_file_names": ["test_only.jpg"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(WBF_SCRIPT, "DATA_DIR", data_root)
    monkeypatch.setattr(WBF_SCRIPT, "TEST_IMAGE_MANIFEST_PATH", manifest)
    monkeypatch.setitem(WBF_SCRIPT.EXPECTED_IMAGE_COUNTS, "test", 1)

    with pytest.raises(FileNotFoundError, match="Val image is absent from its declared split"):
        WBF_SCRIPT.resolve_split_image_paths([{"file_name": "leaked.jpg"}], "val")


def test_split_discovery_does_not_use_another_input_dataset(tmp_path, monkeypatch) -> None:
    expected_root = tmp_path / "scale-error-dataset"
    other_root = tmp_path / "unrelated-dataset"
    images = other_root / "valid" / "images"
    labels = other_root / "valid" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (other_root / "data.yaml").write_text(
        "val: valid/images\nnames: ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "test_image_manifest.json"
    manifest.write_text(
        json.dumps({"split": "test", "image_file_names": ["test_only.jpg"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(WBF_SCRIPT, "DATA_DIR", expected_root)
    monkeypatch.setattr(WBF_SCRIPT, "TEST_IMAGE_MANIFEST_PATH", manifest)
    monkeypatch.setitem(WBF_SCRIPT.EXPECTED_IMAGE_COUNTS, "test", 1)

    with pytest.raises(
        FileNotFoundError, match="Could not identify one labeled val images directory"
    ):
        WBF_SCRIPT._find_yolo_split_images("val")


def test_split_discovery_finds_kaggle_mount_name_with_suffix(tmp_path, monkeypatch) -> None:
    expected_root = tmp_path / "scale-error-dc-v26-yolov8"
    mounted_root = tmp_path / "scale-error-dc-v26-yolov8-2"
    images = mounted_root / "valid" / "images"
    labels = mounted_root / "valid" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (mounted_root / "data.yaml").write_text(
        "val: ../valid/images\nnames: ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(WBF_SCRIPT, "DATA_DIR", expected_root)
    monkeypatch.setattr(WBF_SCRIPT, "KAGGLE_INPUT_DIR", tmp_path, raising=False)

    assert WBF_SCRIPT._find_yolo_split_images("val") == images


def test_split_discovery_uses_exact_roboflow_dataset_identity(tmp_path, monkeypatch) -> None:
    input_root = tmp_path / "input"
    mounted_root = input_root / "roboflow-export"
    images = mounted_root / "valid" / "images"
    labels = mounted_root / "valid" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (mounted_root / "data.yaml").write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "names: ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']\n"
        "roboflow:\n"
        "  workspace: datacpn\n"
        "  project: scale_error_dc-orqpx\n"
        "  version: 26\n",
        encoding="utf-8",
    )
    unrelated = input_root / "other-roboflow-export"
    (unrelated / "valid" / "images").mkdir(parents=True)
    (unrelated / "valid" / "labels").mkdir(parents=True)
    (unrelated / "data.yaml").write_text(
        "val: ../valid/images\n"
        "names: ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']\n"
        "roboflow:\n"
        "  workspace: datacpn\n"
        "  project: other-project\n"
        "  version: 26\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(WBF_SCRIPT, "DATA_DIR", input_root / "scale-error-dc-v26-yolov8")
    monkeypatch.setattr(WBF_SCRIPT, "KAGGLE_INPUT_DIR", input_root, raising=False)

    assert WBF_SCRIPT._find_yolo_split_images("val") == images


def test_combined_test_image_root_is_limited_to_manifest_names(tmp_path, monkeypatch) -> None:
    from PIL import Image

    data_root = tmp_path / "dataset"
    images = data_root / "valid" / "images"
    labels = data_root / "valid" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (data_root / "data.yaml").write_text(
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "names: ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']\n",
        encoding="utf-8",
    )
    Image.new("RGB", (20, 20)).save(images / "test_only.jpg")
    Image.new("RGB", (20, 20)).save(images / "not_test.jpg")
    manifest = tmp_path / "test_image_manifest.json"
    manifest.write_text(
        json.dumps({"split": "test", "image_file_names": ["test_only.jpg"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(WBF_SCRIPT, "DATA_DIR", data_root)
    monkeypatch.setattr(WBF_SCRIPT, "TEST_IMAGE_MANIFEST_PATH", manifest)
    monkeypatch.setitem(WBF_SCRIPT.EXPECTED_IMAGE_COUNTS, "test", 1)

    resolved = WBF_SCRIPT.resolve_split_image_paths([{"file_name": "test_only.jpg"}], "test")

    assert resolved == {"test_only.jpg": images / "test_only.jpg"}
    with pytest.raises(
        ValueError, match="Test ground truth filenames must match the Test manifest"
    ):
        WBF_SCRIPT.resolve_split_image_paths([{"file_name": "not_test.jpg"}], "test")


def test_val_test_audit_rejects_exact_and_near_duplicate_images() -> None:
    val = [{"file_name": "val.jpg", "sha256": "same", "dhash": "0000000000000000"}]
    exact_test = [{"file_name": "test.jpg", "sha256": "same", "dhash": "ffffffffffffffff"}]
    with pytest.raises(ValueError, match="Exact Val/Test image duplicate"):
        WBF_SCRIPT.assert_validation_test_images_disjoint(val, exact_test)

    near_test = [{"file_name": "test.jpg", "sha256": "different", "dhash": "0000000000000001"}]
    with pytest.raises(ValueError, match="Near-duplicate Val/Test images"):
        WBF_SCRIPT.assert_validation_test_images_disjoint(val, near_test)


def test_checkpoint_state_extraction_handles_missing_ema() -> None:
    state = {"layer.weight": object()}

    assert WBF_SCRIPT.checkpoint_state_dict({"ema": None, "model": state}) is state
    assert WBF_SCRIPT.checkpoint_state_dict({"ema": {"module": state}, "model": {}}) is state
