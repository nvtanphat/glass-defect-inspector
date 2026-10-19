"""Real PostgreSQL + real weights. No inference or database mocks."""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glass_inspector.registry import checksum
from glass_inspector.server import create_app

ROOT = Path(__file__).resolve().parents[2]
DATABASE = os.getenv("INSPECTOR_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE,
    reason="BLOCKED: set INSPECTOR_TEST_DATABASE_URL to a migrated PostgreSQL test database",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    application = create_app(
        DATABASE, tmp_path_factory.mktemp("inspections"), token="integration-token"
    )
    with TestClient(application, headers={"Authorization": "Bearer integration-token"}) as value:
        yield value


def config(**overrides):
    return dict(model_id="yolo26m-base", enable_roi=False, **overrides)


def upload(client, options=None, images=None):
    paths = images or [ROOT / "src/glass_inspector/sample_images/sample_bottle_1.jpg"]
    return client.post(
        "/api/inspections",
        data={"config": json.dumps(options or config())},
        files=[("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths],
    )


def require_model(client, model_id):
    ready = {m["id"] for m in client.get("/api/models").json()["models"]}
    if model_id not in ready:
        pytest.skip(f"BLOCKED: {model_id} unavailable; inspect /api/models")


def test_boundaries_and_ready(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/models", headers={"Authorization": ""}).status_code == 401
    assert (
        client.post(
            "/api/predict", json={}, headers={"Origin": "https://attacker.invalid"}
        ).status_code
        == 403
    )
    assert (
        client.post("/api/predict", json={"model_id": "x", "image_data": "/etc/passwd"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/predict", json={"model_id": "x", "image_data": "a", "weights_path": "/etc/passwd"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/inspections",
            data={"config": json.dumps(config())},
            files={"files": ("bad.png", b"not an image", "image/png")},
        ).status_code
        == 422
    )
    assert client.get("/api/inspections?page=0").status_code == 422
    assert client.get("/api/inspections/not-found").status_code == 404
    assert client.get("/api/dashboard").status_code == 200
    assert client.get("/api/inspections").status_code == 200
    assert (
        client.post(
            "/api/inspections",
            data={"config": json.dumps(config())},
            files={
                "files": (
                    "wrong.jpg",
                    (ROOT / "src/glass_inspector/sample_images/sample_bottle_1.jpg").read_bytes(),
                    "image/png",
                )
            },
        ).status_code
        == 422
    )


def test_browser_login_cookie_and_logout(client):
    anonymous = TestClient(client.app)
    assert anonymous.get("/api/models").status_code == 401
    assert anonymous.post("/api/session", json={"password": "incorrect"}).status_code == 401
    assert (
        anonymous.post(
            "/api/session",
            json={"password": "integration-token"},
            headers={"Origin": "https://attacker.invalid"},
        ).status_code
        == 403
    )
    login = anonymous.post("/api/session", json={"password": "integration-token"})
    assert login.status_code == 200
    assert "HttpOnly" in login.headers["set-cookie"]
    assert "SameSite=strict" in login.headers["set-cookie"]
    assert "integration-token" not in login.headers["set-cookie"]
    assert anonymous.get("/api/models").status_code == 200
    assert anonymous.get("/api/dashboard").status_code == 200
    assert anonymous.delete("/api/session").status_code == 200
    assert anonymous.get("/api/models").status_code == 401


@pytest.mark.parametrize("model_id", ["yolo26m-base", "yolo26m-roi-vfl", "dfine-m", "fbdnet-m"])
def test_actual_models(client, model_id):
    require_model(client, model_id)
    response = upload(client, dict(model_id=model_id))
    assert response.status_code == 201, response.text
    detail = response.json()
    assert detail["status"] == "COMPLETED"
    assert detail["model_version"] == checksum(
        ROOT / "weights" / client.app.state.registry.records[model_id]["filename"]
    )
    result = detail["images"][0]["result"]
    for d in result["detections"]:
        assert d["class_id"] in range(5) and d["model_id"] == model_id
        assert 0 <= d["confidence"] <= 1
        x1, y1, x2, y2 = d["bbox_xyxy"]
        assert 0 <= x1 < x2 <= result["image_width"] and 0 <= y1 < y2 <= result["image_height"]
        expected = [
            x1 / result["image_width"],
            y1 / result["image_height"],
            x2 / result["image_width"],
            y2 / result["image_height"],
        ]
        assert d["bbox_normalized"] == expected


def test_zero_detection_save_review_exports_restart(client):
    require_model(client, "yolo26m-base")
    before = client.get("/api/dashboard").json()["total_inspections"]
    response = upload(client, config(conf_threshold=1))
    assert response.status_code == 201, response.text
    detail = response.json()
    session_id = detail["id"]
    assert detail["images"][0]["result"]["detections"] == []
    assert detail["conclusion"] == "REVIEW"
    assert detail["config"]["automatic_assessment"]["conclusion"] == "REVIEW"
    for key in ["original_url", "annotated_url"]:
        assert client.get(detail["images"][0][key]).status_code == 200
    review = dict(
        conclusion="PASS",
        reviewer="Integration technician",
        reason="Visually verified real image",
        expected_version=0,
    )
    updated = client.post(f"/api/inspections/{session_id}/reviews", json=review)
    assert updated.status_code == 200
    assert updated.json()["reviews"][0]["previous_conclusion"] == "REVIEW"
    assert client.post(f"/api/inspections/{session_id}/reviews", json=review).status_code == 409
    for format in ["json", "csv", "pdf"]:
        response = client.get(f"/api/inspections/{session_id}/export?format={format}")
        assert response.status_code == 200 and len(response.content) > 100
        if format == "pdf":
            assert response.content.startswith(b"%PDF")
            from io import BytesIO

            from pypdf import PdfReader

            text = "\n".join(p.extract_text() for p in PdfReader(BytesIO(response.content)).pages)
            assert session_id in text and "Integration technician" in text
            assert "No defect detected" in text
        if format == "csv":
            assert session_id in response.text
    assert client.get("/api/dashboard").json()["total_inspections"] == before + 1
    filtered = client.get(f"/api/inspections?q={session_id}&conclusion=PASS").json()
    assert filtered["total"] == 1
    # New application lifecycle uses the same PostgreSQL records and storage.
    restarted = create_app(
        DATABASE, client.app.state.inspections.storage, token="integration-token"
    )
    with TestClient(restarted, headers={"Authorization": "Bearer integration-token"}) as other:
        persisted = other.get(f"/api/inspections/{session_id}").json()
        assert persisted["conclusion"] == "PASS" and len(persisted["reviews"]) == 1
        assert other.get(persisted["images"][0]["annotated_url"]).status_code == 200


def test_batch_and_capacity(client):
    require_model(client, "yolo26m-base")
    paths = [
        ROOT / "src/glass_inspector/sample_images/sample_bottle_1.jpg",
        ROOT / "src/glass_inspector/sample_images/sample_bottle_2.jpg",
    ]
    response = upload(client, images=paths)
    assert response.status_code == 201, response.text
    assert len(response.json()["images"]) == 2
    assert upload(client, images=paths * 3).status_code == 422
    with ThreadPoolExecutor(max_workers=3) as executor:
        responses = list(executor.map(lambda _: upload(client), range(3)))
    codes = [r.status_code for r in responses]
    assert 201 in codes and 429 in codes, codes
    assert all(code in {201, 429} for code in codes)


def test_real_large_and_unlabeled_image(client):
    require_model(client, "yolo26m-base")
    manifest = ROOT / "reports/inspection/inspector_data_manifest.jsonl"
    if not manifest.is_file():
        pytest.skip("BLOCKED: local dataset manifest missing")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    large = next(ROOT / r["path"] for r in rows if r["size"] == [5328, 3000])
    empty = next(ROOT / r["path"] for r in rows if not r["boxes"])
    if not large.is_file() or not empty.is_file():
        pytest.skip("BLOCKED: local dataset not accessible")
    response = upload(client, images=[large, empty])
    assert response.status_code == 201, response.text
    assert {im["result"]["image_width"] for im in response.json()["images"]} == {
        next(r["size"][0] for r in rows if ROOT / r["path"] == path) for path in [large, empty]
    }


def test_json_compatibility(client):
    require_model(client, "yolo26m-base")
    raw = (ROOT / "src/glass_inspector/sample_images/sample_bottle_1.jpg").read_bytes()
    response = client.post(
        "/api/predict",
        json=dict(config(), image_data="data:image/jpeg;base64," + base64.b64encode(raw).decode()),
    )
    assert response.status_code == 201, response.text
    assert response.json()["inspection_id"]


def test_real_standard_nms_configuration_is_persisted(client):
    require_model(client, "yolo26m-base")
    response = upload(client, config(nms_mode="standard_nms"))
    assert response.status_code == 201, response.text
    detail = response.json()
    assert detail["config"]["nms_mode"] == "standard_nms"
    assert detail["images"][0]["result"]["nms_mode"] == "standard_nms"


def test_inference_failure_is_persisted_and_not_approved(client, monkeypatch):
    require_model(client, "yolo26m-base")

    def fail(*args, **kwargs):
        raise RuntimeError("Deliberately injected inference failure for recovery test")

    monkeypatch.setattr("glass_inspector.services.run_inspection", fail)
    response = upload(client)
    assert response.status_code == 500
    session_id = response.json()["detail"]["inspection_id"]
    detail = client.get(f"/api/inspections/{session_id}").json()
    assert detail["status"] == "FAILED" and detail["conclusion"] == "REVIEW"
    assert detail["images"][0]["annotated_url"] is None
    assert (
        client.post(
            f"/api/inspections/{session_id}/reviews",
            json={
                "conclusion": "PASS",
                "reviewer": "Test technician",
                "reason": "Must be rejected",
                "expected_version": 0,
            },
        ).status_code
        == 409
    )


def test_interrupted_session_recovery(client):
    require_model(client, "yolo26m-base")
    from uuid import uuid4

    from glass_inspector.database import Inspection

    session_id = str(uuid4())
    with client.app.state.sessions.begin() as db:
        db.add(
            Inspection(
                id=session_id,
                status="PROCESSING",
                model_id="yolo26m-base",
                model_version=client.app.state.registry.records["yolo26m-base"]["sha256"],
                config=config(),
            )
        )
    client.app.state.inspections.initialize()
    detail = client.get(f"/api/inspections/{session_id}").json()
    assert detail["status"] == "FAILED" and detail["conclusion"] == "REVIEW"
    assert "restarted" in detail["error"]


def test_concurrent_reviews_keep_one_version_and_automatic_assessment(client):
    require_model(client, "yolo26m-base")
    created = upload(client, config(conf_threshold=1)).json()
    session_id = created["id"]

    def submit(conclusion):
        return client.post(
            f"/api/inspections/{session_id}/reviews",
            json=dict(
                conclusion=conclusion,
                reviewer="Concurrent review validation",
                reason="Testing real PostgreSQL row locking",
                expected_version=0,
            ),
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(submit, ["REJECT", "REVIEW"]))
    assert sorted(statuses) == [200, 409]
    detail = client.get(f"/api/inspections/{session_id}").json()
    assert detail["version"] == 1 and len(detail["reviews"]) == 1
    assert detail["config"]["automatic_assessment"]["conclusion"] == "REVIEW"


@pytest.mark.parametrize("model_id", ["yolo26m-base", "yolo26m-roi-vfl", "dfine-m", "fbdnet-m"])
def test_real_gradcam_is_persisted_and_does_not_change_assessment(client, model_id):
    require_model(client, model_id)
    created = upload(client, {"model_id": model_id}).json()
    image_id = created["images"][0]["id"]
    endpoint = f"/api/images/{image_id}/gradcam"
    response = client.post(endpoint)
    assert response.status_code == 200, response.text
    metadata = response.json()
    assert metadata["model_id"] == model_id
    assert metadata["method"] == "gradcam"
    assert metadata["model_version"] == created["model_version"]
    assert metadata["nonzero_gradient"]
    assert not metadata["flat_cam"], "Real sample should expose spatial Grad-CAM signal"
    overlay = client.get(metadata["image_url"])
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/jpeg"
    assert overlay.content.startswith(b"\xff\xd8")
    detail = client.get(metadata["detail_url"])
    assert detail.status_code == 200 and detail.content.startswith(b"\xff\xd8")
    assert client.post(endpoint).json() == metadata
    assert client.get(metadata["image_url"]).content == overlay.content
    saved = client.get(f"/api/inspections/{created['id']}").json()
    assert saved["images"][0]["result"]["gradcam"]["query"] == metadata["query"]
    assert saved["conclusion"] == created["conclusion"]
    assert (
        saved["images"][0]["result"]["detections"] == created["images"][0]["result"]["detections"]
    )


def test_selected_defects_have_separate_explanations_and_cache_files(client):
    require_model(client, "yolo26m-roi-vfl")
    created = upload(
        client, {"model_id": "yolo26m-roi-vfl", "conf_threshold": 0.1, "enable_roi": False}
    ).json()
    image = created["images"][0]
    detections = image["result"]["detections"]
    assert len(detections) >= 2
    snapshots = []
    for detection in detections[:2]:
        endpoint = (
            f"/api/images/{image['id']}/gradcam?detection_id={detection['id']}"
        )
        response = client.post(endpoint)
        assert response.status_code == 200, response.text
        metadata = response.json()
        assert metadata["detection_id"] == detection["id"]
        assert metadata["target_class"] == detection["class_name"]
        assert metadata["score"] == pytest.approx(detection["confidence"], abs=1e-4)
        full = client.get(metadata["image_url"]).content
        assert client.post(endpoint).json() == metadata
        snapshots.append((metadata, full))
    assert snapshots[0][0]["image_url"] != snapshots[1][0]["image_url"]
    assert snapshots[0][1] != snapshots[1][1]
    assert client.get(snapshots[0][0]["image_url"]).content == snapshots[0][1]
    invalid = client.post(f"/api/images/{image['id']}/gradcam?detection_id=10000")
    assert invalid.status_code == 422


def test_gradcam_rejects_stale_checkpoint_and_requires_authentication(client):
    from glass_inspector.database import Inspection

    require_model(client, "yolo26m-base")
    created = upload(client).json()
    image_id = created["images"][0]["id"]
    endpoint = f"/api/images/{image_id}/gradcam"
    assert client.post(endpoint, headers={"Authorization": ""}).status_code == 401
    with client.app.state.sessions.begin() as db:
        db.get(Inspection, created["id"]).model_version = "0" * 64
    assert client.post(endpoint).status_code == 409
    assert client.get(endpoint).status_code == 404
