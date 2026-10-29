import io

import pytest
from PIL import Image
from pydantic import ValidationError

from glass_inspector.inference_engine import run_nms_boxes
from glass_inspector.registry import ModelRegistry
from glass_inspector.schemas import (
    DetectionResult,
    InspectionConfig,
    ReviewRequest,
    decode_data_url,
    decode_image,
    quality_policy,
)


def test_path_injection_and_thresholds_rejected():
    with pytest.raises(ValidationError):
        InspectionConfig(model_id="model", weights_path="/etc/passwd")
    with pytest.raises(ValueError):
        decode_data_url("C:/server/private.jpg")
    for value in [float("nan"), float("inf"), -1, 2]:
        with pytest.raises(ValidationError):
            InspectionConfig(model_id="model", conf_threshold=value)
    with pytest.raises(ValueError):
        ModelRegistry().resolve("../weights/test.pt")


def test_decoder_limits_and_mime_validation():
    data = io.BytesIO()
    Image.new("RGB", (64, 64)).save(data, "PNG")
    assert decode_image(data.getvalue(), "image/png").size == (64, 64)
    with pytest.raises(ValueError):
        decode_image(data.getvalue(), "image/jpeg")
    with pytest.raises(ValueError):
        decode_image(b"broken", "image/png")
    with pytest.raises(ValueError):
        decode_image(b"x" * (10 * 1024 * 1024 + 1), "image/png")
    big = io.BytesIO()
    Image.new("RGB", (5000, 5000)).save(big, "PNG")
    with pytest.raises(ValueError, match="dimensions"):
        decode_image(big.getvalue(), "image/png")


def test_no_detection_requires_review():
    assert quality_policy([{"detections": []}])[0] == "REVIEW"
    assert quality_policy([{"detections": [{"class_id": 0}]}])[0] == "REJECT"
    with pytest.raises(ValidationError):
        ReviewRequest(conclusion="PASS", reviewer=" ", reason="   ", expected_version=0)


def test_detection_contract_rejects_invalid_geometry_and_mapping():
    valid = dict(
        class_id=0,
        class_name="DiVat",
        confidence=0.5,
        bbox_xyxy=[10, 20, 30, 40],
        bbox_normalized=[0.1, 0.2, 0.3, 0.4],
        model_id="test-schema",
        latency_ms=10,
    )
    DetectionResult(**valid)
    for changes in [
        {"class_name": "LoiChi"},
        {"bbox_normalized": [0.1, 0.2, 1.1, 0.4]},
        {"bbox_xyxy": [10, 20, 10, 40]},
        {"confidence": float("nan")},
    ]:
        with pytest.raises(ValidationError):
            DetectionResult(**dict(valid, **changes))


def test_standard_nms_suppresses_overlaps_within_class_only():
    boxes = [
        ("DiVat", 0.9, 10, 10, 30, 30),
        ("DiVat", 0.8, 11, 11, 31, 31),
        ("LoiChi", 0.7, 10, 10, 30, 30),
    ]
    result = run_nms_boxes(boxes, 0.45)
    assert len(result) == 2
    assert {b[0] for b in result} == {"DiVat", "LoiChi"}
    assert ("DiVat", 0.9, 10, 10, 30, 30) in result
    low = run_nms_boxes([("DiVat", 0.02, 10, 10, 30, 30), ("DiVat", 0.01, 11, 11, 31, 31)], 0.45)
    assert low == [("DiVat", 0.02, 10, 10, 30, 30)]


def test_chunked_upload_boundary_releases_capacity(monkeypatch):
    import asyncio

    import glass_inspector.server as server

    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 4)

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    boundary = server.RequestBoundary(downstream, token="")
    scope = dict(
        type="http", method="POST", path="/api/predict", headers=[], client=("127.0.0.1", 1)
    )

    async def exercise(messages):
        result = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            result.append(message)

        await boundary(scope, receive, send)
        return result[0]["status"]

    assert (
        asyncio.run(
            exercise(
                [
                    {"type": "http.request", "body": b"123", "more_body": True},
                    {"type": "http.request", "body": b"45", "more_body": False},
                ]
            )
        )
        == 413
    )
    assert (
        asyncio.run(exercise([{"type": "http.request", "body": b"1", "more_body": False}])) == 200
    )


@pytest.mark.parametrize("authorization", [b"Bearer invalid", b"Bearer \xff", b"Bearer \xc3\xa9"])
def test_malformed_bearer_token_is_unauthorized(authorization):
    import asyncio

    from glass_inspector.server import RequestBoundary

    async def downstream(scope, receive, send):
        pytest.fail("Unauthorized requests must not reach the application")

    async def receive():
        pytest.fail("Unauthorized request bodies must not be consumed")

    async def exercise():
        messages = []

        async def send(message):
            messages.append(message)

        scope = dict(
            type="http",
            method="POST",
            path="/api/predict",
            headers=[(b"authorization", authorization)],
            client=("127.0.0.1", 1),
        )
        await RequestBoundary(downstream, "internal-token")(scope, receive, send)
        assert messages[0]["status"] == 401

    asyncio.run(exercise())


def test_browser_cookie_cannot_be_forged_or_reused_after_expiry(monkeypatch):
    import glass_inspector.server as server

    monkeypatch.setattr(server.time, "time", lambda: 1000)
    cookie = server.issue_session("private-password")
    header = f"{server.SESSION_COOKIE}={cookie}".encode()
    assert server.valid_session(header, "private-password")
    assert not server.valid_session(header, "different-password")
    assert not server.valid_session(header + b"tampered", "private-password")
    assert not server.valid_session(b"broken=\xff", "private-password")
    monkeypatch.setattr(server.time, "time", lambda: 1000 + server.SESSION_SECONDS + 1)
    assert not server.valid_session(header, "private-password")


def test_ensemble_and_distilled_registry_manifest():
    from glass_inspector.inference_engine import compute_box_iou, fast_wbf_fusion

    registry = ModelRegistry()
    manifest = {m["id"]: m for m in registry.public()}

    assert "apex-triad-ensemble" in manifest
    assert "yolo26m-distilled" in manifest
    assert manifest["apex-triad-ensemble"]["adapter"] == "ensemble"
    assert manifest["apex-triad-ensemble"]["is_default"] is True
    assert manifest["yolo26m-distilled"]["adapter"] == "yolo"

    # Test fast_wbf_fusion
    b1 = [10.0, 10.0, 50.0, 50.0]
    b2 = [12.0, 12.0, 48.0, 48.0]
    iou = compute_box_iou(b1, b2)
    assert iou > 0.7

    boxes_list = [[b1], [b2]]
    scores_list = [[0.85], [0.90]]
    weights = [1.0, 1.0]
    fused_boxes, fused_scores = fast_wbf_fusion(
        boxes_list,
        scores_list,
        weights,
        iou_thr=0.5,
        coord_ratio=[0.5, 0.5],
        boost_3=1.2,
        boost_2=1.1,
        solo_decay=0.85,
        conf_floor=0.05,
    )
    assert len(fused_boxes) == 1
    assert fused_scores[0] > 0.85
