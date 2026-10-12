"""Exercise evaluation loops with the real deployed postprocessor, without training."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]


def deployed_postprocessor():
    source = ROOT / "results/D-FINE/src/zoo/dfine/postprocessor.py"
    if not source.exists():
        pytest.skip("Original D-FINE postprocessor unavailable")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "torchvision": torchvision,
        "register": lambda: (lambda cls: cls),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["DFINEPostProcessor"](num_classes=5, num_top_queries=1).deploy()


class PostprocessorFixture:
    """Deterministic tensor inputs for testing output schema and size mapping."""

    def __init__(self):
        self.postprocessor = deployed_postprocessor()
        self.batch_sizes = []

    def __call__(self, tensors, sizes):
        batch_size = len(tensors)
        self.batch_sizes.append(batch_size)
        logits = torch.full((batch_size, 1, 5), -10.0)
        for index in range(batch_size):
            logits[index, 0, index % 5] = 10.0
        boxes = torch.tensor([0.5, 0.5, 0.4, 0.2]).repeat(batch_size, 1, 1)
        return self.postprocessor({"pred_logits": logits, "pred_boxes": boxes}, sizes)


def test_evaluation_handles_deployed_batches_and_rectangular_images():
    script = ROOT / "scripts/eval_ahrd_experiment.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    tree = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    loop = next(
        node
        for node in tree.body
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "i"
        )
    )
    image_path = ROOT / "src/glass_inspector/sample_images/sample_replica.jpg"
    with Image.open(image_path) as original:
        width, height = original.size
    assert width != height, "Regression requires a real rectangular source image"
    images = [
        {"id": index, "file_name": image_path.name, "width": 1, "height": 1} for index in range(5)
    ]
    model = PostprocessorFixture()
    namespace = {
        "torch": torch,
        "Image": Image,
        "Path": Path,
        "dev": "cpu",
        "args": SimpleNamespace(batch_size=4, device="cpu"),
        "img_dir": image_path.parent,
        "images": images,
        "images_info": images,
        "tf": transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]),
        "model": model,
        "full_model": model,
        "preds": [],
        "predictions": [],
    }
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(script), "exec"), namespace)
    predictions = namespace["predictions"]
    assert model.batch_sizes == [4, 1]
    assert len(predictions) == 5
    for index, prediction in enumerate(predictions):
        assert prediction["image_id"] == index
        assert prediction["category_id"] == index % 4
        assert prediction["score"] > 0.99
        assert prediction["bbox"] == pytest.approx(
            [width * 0.3, height * 0.4, width * 0.4, height * 0.2], rel=1e-5
        )
