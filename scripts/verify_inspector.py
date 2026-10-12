"""Exercise a running inspector with real local images and write a concise receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def wait_until_ready(session, base):
    deadline = time.monotonic() + 120
    while True:
        try:
            if session.get(base + "/api/ready", timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("BLOCKED: server did not become ready within 120 seconds")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", default="reports/inspection/inspector_validation.json")
    parser.add_argument(
        "--check-session", help="Verify persisted session after server/container restart"
    )
    args = parser.parse_args()
    session = requests.Session()
    token = os.getenv("INSPECTOR_E2E_TOKEN", "")
    if token:
        session.headers["Authorization"] = "Bearer " + token
    base = args.url.rstrip("/")
    wait_until_ready(session, base)

    def get(path):
        response = session.get(base + path, timeout=120)
        response.raise_for_status()
        return response

    if args.check_session:
        detail = get("/api/inspections/" + args.check_session).json()
        assert detail["status"] == "COMPLETED"
        assert detail["reviews"]
        assert get(detail["images"][0]["annotated_url"]).content.startswith(b"\xff\xd8")
        output = Path(args.output)
        receipt = json.loads(output.read_text())
        receipt["process_restart"] = dict(
            status="PASS",
            session_id=args.check_session,
            checked_at=datetime.now(UTC).isoformat(),
            conclusion=detail["conclusion"],
        )
        output.write_text(json.dumps(receipt, indent=2))
        print("Restart persistence PASS", args.check_session)
        return
    manifest = get("/api/models").json()
    receipt = dict(timestamp=datetime.now(UTC).isoformat(), url=base, models=manifest, checks=[])
    if token:
        assert (
            session.get(base + "/api/models", headers={"Authorization": ""}, timeout=30).status_code
            == 401
        )
        receipt["checks"].append(dict(test="token_authentication", status="PASS"))
    else:
        receipt["checks"].append(
            dict(
                test="token_authentication",
                status="BLOCKED",
                reason="INSPECTOR_E2E_TOKEN not set",
            )
        )
    assert (
        session.post(
            base + "/api/predict",
            json={"model_id": "x", "image_data": "/etc/passwd"},
            timeout=30,
        ).status_code
        == 422
    )
    assert (
        session.post(
            base + "/api/inspections",
            data={"config": json.dumps({"model_id": "yolo26m-base"})},
            files=[("files", ("corrupt.png", b"not an image", "image/png"))],
            timeout=30,
        ).status_code
        == 422
    )
    receipt["checks"].append(dict(test="path_and_corrupt_upload", status="PASS"))
    image = ROOT / "src/glass_inspector/sample_images/sample_bottle_1.jpg"
    raw = image.read_bytes()

    def upload(config, images=None):
        files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in (images or [image])]
        return session.post(
            base + "/api/inspections", data={"config": json.dumps(config)}, files=files, timeout=180
        )

    for model in manifest["models"]:
        response = upload(dict(model_id=model["id"]))
        response.raise_for_status()
        detail = response.json()
        assert detail["status"] == "COMPLETED"
        assert detail["images"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
        receipt["checks"].append(
            dict(
                test="real_model_inference",
                model_id=model["id"],
                session_id=detail["id"],
                status="PASS",
                detections=detail["images"][0]["result"]["total_defects"],
                latency_ms=detail["images"][0]["result"]["latency_ms"],
            )
        )
    ready = {m["id"] for m in manifest["models"]}
    for model in manifest["unavailable"]:
        unavailable = upload(dict(model_id=model["id"]))
        assert unavailable.status_code == 503, unavailable.text
    assert upload(dict(model_id="../weights/private.pt")).status_code == 422
    receipt["checks"].append(dict(test="registry_unavailable_and_server_paths", status="PASS"))
    assert "yolo26m-base" in ready, "BLOCKED: YOLO26m Base required for zero-detection regression"
    response = upload(dict(model_id="yolo26m-base", conf_threshold=1))
    response.raise_for_status()
    detail = response.json()
    inspection_id = detail["id"]
    assert detail["images"][0]["result"]["detections"] == [] and detail["conclusion"] == "REVIEW"
    review = session.post(
        base + f"/api/inspections/{inspection_id}/reviews",
        json=dict(
            conclusion="REVIEW",
            reviewer="Deployment validation",
            reason="Recorded validation workflow; no production quality approval",
            expected_version=0,
        ),
        timeout=30,
    )
    review.raise_for_status()
    exports = {}
    for format in ["json", "csv", "pdf"]:
        exported = get(f"/api/inspections/{inspection_id}/export?format={format}")
        assert len(exported.content) > 100
        exports[format] = dict(
            bytes=len(exported.content), sha256=hashlib.sha256(exported.content).hexdigest()
        )
    receipt["checks"].append(
        dict(
            test="zero_detection_save_review_exports",
            status="PASS",
            session_id=inspection_id,
            exports=exports,
        )
    )
    receipt["restart_session_id"] = inspection_id
    paths = [image, ROOT / "src/glass_inspector/sample_images/sample_bottle_2.jpg"]
    batch = upload(dict(model_id="yolo26m-base"), paths)
    batch.raise_for_status()
    assert len(batch.json()["images"]) == 2
    receipt["checks"].append(dict(test="batch", status="PASS", session_id=batch.json()["id"]))
    with ThreadPoolExecutor(max_workers=3) as pool:
        codes = list(
            pool.map(lambda _: upload(dict(model_id="yolo26m-base")).status_code, range(3))
        )
    assert 201 in codes and 429 in codes and all(c in {201, 429} for c in codes), codes
    receipt["checks"].append(dict(test="concurrent_capacity", status="PASS", http_statuses=codes))
    receipt["dashboard"] = get("/api/dashboard").json()
    data_manifest = ROOT / "reports/inspection/inspector_data_manifest.jsonl"
    if data_manifest.is_file():
        rows = [json.loads(line) for line in data_manifest.read_text().splitlines()]
        large = next(ROOT / r["path"] for r in rows if r["size"] == [5328, 3000])
        empty = next(ROOT / r["path"] for r in rows if not r["boxes"])
        if large.is_file() and empty.is_file():
            response = upload(dict(model_id="yolo26m-base"), [large, empty])
            response.raise_for_status()
            assert response.json()["status"] == "COMPLETED"
            receipt["checks"].append(
                dict(
                    test="real_large_and_empty_label_images",
                    status="PASS",
                    session_id=response.json()["id"],
                    caveat="An empty source label is not proof of a defect-free image.",
                )
            )
        else:
            receipt["checks"].append(
                dict(
                    test="real_large_and_empty_label_images",
                    status="BLOCKED",
                    reason="Dataset inaccessible",
                )
            )
    else:
        receipt["checks"].append(
            dict(
                test="real_large_and_empty_label_images",
                status="BLOCKED",
                reason="Dataset manifest missing",
            )
        )
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2))
    print("Validation PASS; restart session:", inspection_id)


if __name__ == "__main__":
    main()
