"""Browser -> HTTP -> real inference -> PostgreSQL -> browser workflow."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = os.getenv("INSPECTOR_E2E_URL")
pytestmark = pytest.mark.skipif(
    not BASE, reason="BLOCKED: start a real server and set INSPECTOR_E2E_URL"
)


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page.goto(BASE)
        token = os.getenv("INSPECTOR_E2E_TOKEN", "")
        if token:
            page.locator("#login-dialog").wait_for(state="visible")
            page.locator("#api-token").fill(token)
            page.locator("#connect-api").click()
        page.wait_for_function(
            "document.querySelector('#model-weights').options.length >= 1 && "
            "!document.querySelector('#btn-run').disabled"
        )
        yield page
        browser.close()


def run_image(page, filename="sample_bottle_1.jpg", confidence=0.25, model_id="yolo26m-base"):
    page.locator("[data-screen=workspace]").click()
    page.locator("#image-file-input").set_input_files(
        ROOT / "src/glass_inspector/sample_images" / filename
    )
    page.wait_for_function("appState.imageWidth > 0 && appState.currentImage != null")
    if not page.locator("#inspection-options").evaluate("(element) => element.open"):
        page.locator("#inspection-options summary").click()
    page.select_option("#model-weights", model_id)
    page.evaluate("(value) => { confSlider.value=value; }", confidence)
    with page.expect_response(
        lambda response: response.url.endswith("/api/inspections")
        and response.request.method == "POST",
        timeout=90000,
    ) as pending:
        page.locator("#btn-run").click()
    response = pending.value
    assert response.status == 201, response.text()
    page.wait_for_function("appState.result != null && !appState.isInspecting")
    detail = response.json()
    assert (
        detail["images"][0]["sha256"]
        == hashlib.sha256(
            (ROOT / "src/glass_inspector/sample_images" / filename).read_bytes()
        ).hexdigest()
    )
    return dict(
        detail["images"][0]["result"], inspection_id=detail["id"], conclusion=detail["conclusion"]
    )


def assert_boxes_match(page):
    page.wait_for_timeout(300)
    errors = page.evaluate("""() => {
      const r = viewportImg.getBoundingClientRect();
      return appState.detections.map((d,i) => {
        const b = document.querySelectorAll('.defect-box')[i].getBoundingClientRect();
        const n = d.bbox_normalized;
        return Math.max(Math.abs(b.left-(r.left+n[0]*r.width)),
          Math.abs(b.top-(r.top+n[1]*r.height)),
          Math.abs(b.width-(n[2]-n[0])*r.width),
          Math.abs(b.height-(n[3]-n[1])*r.height));
      });
    }""")
    assert errors and max(errors) < 1, errors


def test_gradcam_toggle_saved_history_and_image_reset(page):
    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    result = run_image(page)
    original = page.locator("#viewport-img").get_attribute("src")
    with page.expect_response(
        lambda response: response.url.split("?")[0].endswith("/gradcam")
        and response.request.method == "POST",
        timeout=90000,
    ) as pending:
        page.locator("#btn-gradcam").click()
    assert pending.value.status == 200, pending.value.text()
    assert pending.value.json()["method"] == "hirescam"
    assert pending.value.json()["detection_id"] == page.evaluate("appState.selectedDefectId")
    page.wait_for_function(
        "document.getElementById('btn-gradcam').getAttribute('aria-pressed') === 'true'"
    )
    assert page.locator("#viewport-img").get_attribute("src") != original
    page.screenshot(path=str(ROOT / "docs/assets/inspector-gradcam.png"), full_page=True)
    with page.expect_download() as downloaded:
        page.locator("#btn-save-image").click()
    assert "hirescam" in downloaded.value.suggested_filename
    page.locator("#btn-gradcam").click()
    assert page.locator("#viewport-img").get_attribute("src") == original
    page.locator("[data-screen=history]").click()
    page.locator(f'[data-inspection-id="{result["inspection_id"]}"]').click()
    page.get_by_role("button", name="Xem vùng chú ý", exact=True).click()
    page.get_by_alt_text("Bản đồ Grad-CAM trên ảnh kiểm định").wait_for(state="visible")
    assert page.get_by_role("link", name="Tải ảnh vùng chú ý").is_visible()
    page.get_by_label("Cách giải thích ảnh đã lưu").select_option("gradcam")
    with page.expect_response(
        lambda response: response.url.split("?")[0].endswith("/gradcam")
        and response.request.method == "POST",
        timeout=90000,
    ) as switched:
        page.get_by_role("button", name="Xem vùng chú ý", exact=True).click()
    assert switched.value.status == 200 and switched.value.json()["method"] == "gradcam"
    run_image(page, confidence=1)
    page.locator("#btn-gradcam").click()
    page.wait_for_function(
        "document.getElementById('btn-gradcam').getAttribute('aria-pressed') === 'true'"
    )
    assert "Chưa có lỗi vượt ngưỡng" in page.locator("#gradcam-caption").inner_text()
    page.locator("#image-file-input").set_input_files(
        ROOT / "src/glass_inspector/sample_images/sample_bottle_2.jpg"
    )
    page.wait_for_function("appState.result === null")
    assert page.locator("#btn-gradcam").is_disabled()
    assert not errors, errors


def test_selecting_another_defect_changes_the_explanation_target(page):
    result = run_image(page, confidence=0.1, model_id="yolo26m-roi-vfl")
    assert len(result["detections"]) >= 2
    second = result["detections"][1]
    page.locator(f"#row-det-{second['id']}").click()
    with page.expect_response(
        lambda response: response.url.split("?")[0].endswith("/gradcam")
        and response.request.method == "POST",
        timeout=90000,
    ) as pending:
        page.locator("#btn-gradcam").click()
    assert pending.value.status == 200, pending.value.text()
    assert pending.value.json()["detection_id"] == second["id"]
    assert pending.value.json()["target_class"] == second["class_name"]
    page.wait_for_function(
        "document.getElementById('btn-gradcam').getAttribute('aria-pressed') === 'true'"
    )
    assert page.locator("#roi-thumbnail").get_attribute("src").startswith("blob:")
    page.locator(f"#row-det-{result['detections'][0]['id']}").click()
    assert page.locator("#btn-gradcam").get_attribute("aria-pressed") == "false"
    assert page.evaluate("appState.gradcamImage === null")


def test_browser_boxes_resize_zoom_fullscreen_and_reset(page):
    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    result = run_image(page)
    assert result["detections"]
    assert_boxes_match(page)
    for size in [{"width": 1280, "height": 800}, {"width": 1920, "height": 1080}]:
        page.set_viewport_size(size)
        assert_boxes_match(page)
    for _ in range(3):
        page.locator("#btn-zoom").click()
        assert_boxes_match(page)
    page.locator("#btn-fullscreen").click()
    page.wait_for_function("document.fullscreenElement != null")
    assert_boxes_match(page)
    page.evaluate("document.exitFullscreen()")
    page.wait_for_function("document.fullscreenElement == null")
    page.screenshot(path=str(ROOT / "docs/assets/inspector-workspace.png"), full_page=True)
    page.locator("#image-file-input").set_input_files(
        ROOT / "src/glass_inspector/sample_images/sample_bottle_2.jpg"
    )
    page.wait_for_function("appState.result === null && appState.annotatedImage === null")
    assert page.locator(".defect-box").count() == 0
    assert page.evaluate("appState.zoomLevel") == 100
    page.evaluate(
        "window.postMessage({type:'streamlit:render',args:{current_result:{success:true}}}, '*')"
    )
    page.wait_for_timeout(100)
    assert page.evaluate("appState.result === null")
    assert not errors, errors


def test_browser_zero_detection_history_review_export_dashboard(page):
    result = run_image(page, confidence=1)
    assert result["detections"] == [] and result["conclusion"] == "REVIEW"
    assert "Cần người kiểm tra xác nhận" in page.locator("#detections-tbody").inner_text()
    assert "(OK" not in page.locator("#detections-tbody").inner_text()
    with page.expect_download() as downloaded:
        page.locator("#btn-export-json").click()
    assert downloaded.value.suggested_filename.endswith(".json")
    with page.expect_download() as downloaded:
        page.locator("#btn-save-image").click()
    assert downloaded.value.suggested_filename.endswith(".jpg")
    page.locator("[data-screen=history]").click()
    page.locator(f'[data-inspection-id="{result["inspection_id"]}"]').click()
    page.locator("#reviewer").fill("Browser technician")
    page.locator("#review-reason").fill("Verified saved image and result in browser E2E")
    page.select_option("#review-conclusion", "PASS")
    page.locator("#submit-review").click()
    page.wait_for_function("document.querySelector('#review-summary')?.textContent.includes('Đạt')")
    page.screenshot(path=str(ROOT / "docs/assets/inspector-review.png"), full_page=True)
    page.get_by_role("button", name="Xem vị trí lỗi trên ảnh").click()
    page.locator("#workspace-screen").wait_for(state="visible")
    page.wait_for_function("appState.result != null")
    assert page.evaluate("appState.result.total_defects") == 0
    # Navigating away must retain Workspace's saved image download.
    page.locator("[data-screen=dashboard]").click()
    page.wait_for_selector("#dashboard-total")
    assert "Tổng lượt kiểm tra:" in page.locator("#dashboard-total").inner_text()
    page.screenshot(path=str(ROOT / "docs/assets/inspector-dashboard.png"), full_page=True)
    page.reload()
    page.wait_for_function("!document.querySelector('#btn-run').disabled")
    assert not page.locator("#login-dialog").is_visible()
    page.locator("[data-screen=history]").click()
    page.locator(f'[data-inspection-id="{result["inspection_id"]}"]').click()
    page.wait_for_selector("#review-summary")
    assert "Đạt" in page.locator("#review-summary").inner_text()


def test_latest_navigation_wins_over_pending_image_fetch(page):
    held = []
    page.route("**/api/images/*/original", lambda route: held.append(route))
    page.get_by_role("button", name="Xem vị trí lỗi trên ảnh").click()
    page.wait_for_timeout(100)
    assert held
    page.locator("[data-screen=dashboard]").click()
    page.wait_for_selector("#dashboard-total")
    for route in held:
        route.continue_()
    page.wait_for_timeout(500)
    assert page.locator("#dashboard-total").is_visible()
    page.unroute("**/api/images/*/original")


def test_browser_batch(page):
    page.locator("[data-screen=workspace]").click()
    page.locator("#image-file-input").set_input_files(
        [
            ROOT / "src/glass_inspector/sample_images/sample_bottle_1.jpg",
            ROOT / "src/glass_inspector/sample_images/sample_bottle_2.jpg",
        ]
    )
    page.wait_for_function("appState.files.length === 2")
    if not page.locator("#inspection-options").evaluate("(element) => element.open"):
        page.locator("#inspection-options summary").click()
    page.select_option("#model-weights", "yolo26m-base")
    page.evaluate("confSlider.value=.25; confSlider.dispatchEvent(new Event('input'))")
    with page.expect_response(
        lambda response: "/api/inspections" in response.url and response.request.method == "POST",
        timeout=90000,
    ) as pending:
        page.locator("#btn-run").click()
    assert pending.value.status == 201, pending.value.text()
    page.wait_for_selector("#review-summary")
    assert len(pending.value.json()["images"]) == 2
    page.locator("[data-screen=workspace]").click()
    assert page.evaluate("appState.result === null && appState.annotatedImage === null")
    assert page.locator(".defect-box").count() == 0


def test_failed_retry_clears_saved_result(page):
    run_image(page)
    assert page.evaluate("appState.annotatedImage !== null")
    page.route(
        "**/api/inspections",
        lambda route: route.fulfill(
            status=503, content_type="application/json", body='{"detail":"Model unavailable"}'
        ),
    )
    try:
        page.locator("#btn-run").click()
        page.wait_for_function("!appState.isInspecting && appState.result === null")
        assert page.evaluate("appState.annotatedImage === null && appState.inspectionId === null")
        assert page.locator(".defect-box").count() == 0
    finally:
        page.unroute("**/api/inspections")


def test_saved_workspace_can_download_and_inspect_again(page):
    result = run_image(page)
    page.locator("[data-screen=history]").click()
    page.locator(f'[data-inspection-id="{result["inspection_id"]}"]').click()
    page.get_by_role("button", name="Xem vị trí lỗi trên ảnh").click()
    page.locator("#workspace-screen").wait_for(state="visible")
    page.locator("[data-screen=dashboard]").click()
    page.wait_for_selector("#dashboard-total")
    page.locator("[data-screen=workspace]").click()
    with page.expect_download() as download:
        page.locator("#btn-save-image").click()
    assert Path(download.value.path()).stat().st_size > 100
    with page.expect_response(
        lambda response: response.url.endswith("/api/inspections")
        and response.request.method == "POST",
        timeout=90000,
    ) as pending:
        page.locator("#btn-run").click()
    assert pending.value.status == 201, pending.value.text()
    page.wait_for_function("appState.result != null && !appState.isInspecting")
    assert page.evaluate("appState.inspectionId") != result["inspection_id"]


def test_login_reload_expired_session_and_logout(page):
    token = os.getenv("INSPECTOR_E2E_TOKEN")
    if not token:
        pytest.skip("BLOCKED: authentication test needs INSPECTOR_E2E_TOKEN")
    page.context.clear_cookies()
    page.reload()
    page.locator("#login-dialog").wait_for(state="visible")
    page.locator("#api-token").fill("incorrect-password")
    page.locator("#connect-api").click()
    page.wait_for_function(
        "document.querySelector('#login-error').textContent.includes('Mật khẩu chưa đúng')"
    )
    assert "401" not in page.locator("body").inner_text()
    page.locator("#api-token").fill(token)
    page.locator("#connect-api").click()
    page.wait_for_function("!document.querySelector('#btn-run').disabled")
    assert page.locator("#api-token").input_value() == ""
    assert "inspector_session" not in page.evaluate("document.cookie")
    page.reload()
    page.wait_for_function("!document.querySelector('#btn-run').disabled")
    assert not page.locator("#login-dialog").is_visible()
    text = page.locator("body").inner_text()
    assert not any(
        term in text for term in ["SHA256", "NMS", "ROI", "checkpoint", "Bearer", "XYXY"]
    )
    page.locator("#account-button").click()
    page.locator("#login-dialog").wait_for(state="visible")
    assert page.evaluate("fetch('/api/models').then(response => response.status)") == 401
