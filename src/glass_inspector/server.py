"""FastAPI modular monolith. Run one worker; migrate PostgreSQL before startup."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import socket
import threading
import time
from contextlib import asynccontextmanager
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from glass_inspector.database import connect
from glass_inspector.inference_engine import WORKSPACE_ROOT, pil_to_base64
from glass_inspector.paths import SAMPLES_DIR, WEB_DIR
from glass_inspector.registry import ModelRegistry
from glass_inspector.schemas import (
    MAX_BATCH,
    MAX_FILE_BYTES,
    InspectionConfig,
    LoginRequest,
    PredictionResult,
    PredictRequest,
    ReviewRequest,
    decode_data_url,
    decode_image,
)
from glass_inspector.services import (
    InferenceService,
    InspectionReviewService,
    InspectionService,
    ReportingService,
)

PROJECT_ROOT = WORKSPACE_ROOT
HTML_FILE = WEB_DIR / "index.html"
MAX_REQUEST_BYTES = 42 * 1024 * 1024
log = logging.getLogger("inspector")
SESSION_COOKIE = "inspector_session"
SESSION_SECONDS = 8 * 60 * 60


def issue_session(token):
    payload = f"{int(time.time()) + SESSION_SECONDS}.{secrets.token_urlsafe(24)}"
    signature = hmac.new(token.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def valid_session(cookie_header, token):
    if not token:
        return False
    try:
        cookies = SimpleCookie()
        cookies.load(cookie_header.decode("ascii"))
        value = cookies[SESSION_COOKIE].value
        expires, nonce, signature = value.split(".")
        if not nonce or not time.time() < int(expires) <= time.time() + SESSION_SECONDS:
            return False
        expected = hmac.new(
            token.encode(), f"{expires}.{nonce}".encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature.encode(), expected.encode())
    except (CookieError, KeyError, ValueError, UnicodeError):
        return False


class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        fields = record.msg if isinstance(record.msg, dict) else {"event": record.getMessage()}
        return json.dumps(
            dict(
                **fields,
                level=record.levelname,
                inspection_id=getattr(record, "inspection_id", None),
                exception=self.formatException(record.exc_info) if record.exc_info else None,
            )
        )


def get_sample_images_dict():
    mapping = {
        "replica": "sample_replica.jpg",
        "bottle1": "sample_bottle_1.jpg",
        "bottle2": "sample_bottle_2.jpg",
        "bottle3": "sample_bottle_3.jpg",
    }
    samples = {}
    for key, name in mapping.items():
        path = SAMPLES_DIR / name
        if path.is_file():
            with Image.open(path) as image:
                samples[key] = pil_to_base64(image.convert("RGB"))
    return samples


class RequestBoundary:
    """Authenticate before reading a bounded body, including chunked requests."""

    def __init__(self, app, token):
        self.app, self.token = app, token
        self.upload_gate = threading.BoundedSemaphore(2)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        path = scope["path"]
        protected = path.startswith("/api/") and path not in {"/api/health", "/api/session"}
        if protected:
            if self.token:
                provided = headers.get(b"authorization", b"")
                bearer_valid = hmac.compare_digest(
                    provided, ("Bearer " + self.token).encode("utf-8")
                )
                if not bearer_valid and not valid_session(headers.get(b"cookie", b""), self.token):
                    return await JSONResponse({"detail": "Authentication required"}, 401)(
                        scope, receive, send
                    )
            elif scope.get("client", ("unknown",))[0] not in {"127.0.0.1", "::1", "testclient"}:
                return await JSONResponse(
                    {"detail": "Remote access requires INSPECTOR_API_TOKEN"}, 403
                )(scope, receive, send)
        origin = headers.get(b"origin")
        host = headers.get(b"host", b"")
        if origin and origin not in {b"http://" + host, b"https://" + host}:
            return await JSONResponse({"detail": "Cross-origin requests forbidden"}, 403)(
                scope, receive, send
            )
        if scope["method"] in {"POST", "PUT", "PATCH"}:
            acquired = self.upload_gate.acquire(blocking=False)
            if not acquired:
                return await JSONResponse({"detail": "Upload capacity reached"}, 429)(
                    scope, receive, send
                )
            body = bytearray()
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(receive(), timeout=20)
                    except TimeoutError:
                        return await JSONResponse({"detail": "Upload timed out"}, 408)(
                            scope, receive, send
                        )
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                        return await JSONResponse({"detail": "Request exceeds 42 MiB"}, 413)(
                            scope, receive, send
                        )
                    body.extend(chunk)
                    if not message.get("more_body"):
                        break
                sent = False

                async def bounded_receive():
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                await self.app(scope, bounded_receive, send)
            finally:
                self.upload_gate.release()
        else:
            await self.app(scope, receive, send)


def create_app(database_url=None, storage=None, token=None):
    database_url = database_url or os.getenv(
        "DATABASE_URL", "postgresql+psycopg://inspector:inspector@127.0.0.1:5432/inspector"
    )
    storage = Path(
        storage or os.getenv("INSPECTOR_STORAGE", str(PROJECT_ROOT / "var/inspections"))
    ).resolve()
    token = token if token is not None else os.getenv("INSPECTOR_API_TOKEN", "")
    registry = ModelRegistry()
    engine, sessions = connect(database_url)
    inference = InferenceService(registry)
    inspections = InspectionService(sessions, storage, inference)
    reviews = InspectionReviewService(sessions)
    reporting = ReportingService(sessions, inspections)

    @asynccontextmanager
    async def lifespan(application):
        handler = logging.StreamHandler()
        handler.setFormatter(JSONLogFormatter())
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        with engine.connect() as db:
            revision = db.execute(text("SELECT version_num FROM alembic_version")).scalar()
            if revision != "0001":
                raise RuntimeError("Apply alembic upgrade head before starting the server")
        await run_in_threadpool(registry.probe)
        inspections.initialize()
        yield
        engine.dispose()
        log.removeHandler(handler)

    application = FastAPI(title="Glass Defect Inspector", version="2.0.0", lifespan=lifespan)
    application.add_middleware(RequestBoundary, token=token)
    application.state.registry = registry
    application.state.inspections = inspections
    application.state.inference = inference
    application.state.sessions = sessions

    @application.middleware("http")
    async def access_log(request, call_next):
        request_id = str(uuid4())
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request_failed")
            response = JSONResponse({"detail": "Internal error", "request_id": request_id}, 500)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        log.info(
            dict(
                event="request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )
        )
        return response

    @application.get("/", include_in_schema=False)
    def ui():
        return FileResponse(HTML_FILE, media_type="text/html")

    @application.get("/ui/workflow.js", include_in_schema=False)
    def workflow_js():
        return FileResponse(HTML_FILE.parent / "workflow.js", media_type="application/javascript")

    @application.get("/ui/user-text.js", include_in_schema=False)
    def user_text_js():
        return FileResponse(HTML_FILE.parent / "user-text.js", media_type="application/javascript")

    @application.get("/ui/styles.css", include_in_schema=False)
    def stylesheet():
        return FileResponse(HTML_FILE.parent / "styles.css", media_type="text/css")

    @application.get("/api/health")
    def health():
        return {"status": "ok"}

    @application.post("/api/session")
    def login(body: LoginRequest, request: Request):
        if not token:
            raise HTTPException(503, "Access password has not been configured")
        if not hmac.compare_digest(body.password.encode(), token.encode()):
            raise HTTPException(401, "Incorrect access password")
        response = JSONResponse({"authenticated": True})
        response.set_cookie(
            SESSION_COOKIE,
            issue_session(token),
            max_age=SESSION_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
        )
        return response

    @application.delete("/api/session")
    def logout():
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(SESSION_COOKIE, httponly=True, samesite="strict")
        return response

    @application.get("/api/ready")
    def ready():
        try:
            with engine.connect() as db:
                db.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(503, "Database unavailable") from exc
        if not any(m["status"] == "READY" for m in registry.public()):
            raise HTTPException(503, "No verified model available")
        return {"status": "ready"}

    @application.get("/api/models")
    def models():
        manifest = registry.public()
        return dict(
            models=[m for m in manifest if m["status"] == "READY"],
            unavailable=[m for m in manifest if m["status"] != "READY"],
        )

    @application.get("/api/samples")
    def samples():
        return {"samples": get_sample_images_dict()}

    @application.post("/api/predict", status_code=201, response_model=PredictionResult)
    def predict(body: PredictRequest):
        try:
            raw, image = decode_data_url(body.image_data)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        config = InspectionConfig(**body.model_dump(exclude={"image_data"}))
        inspection_id, results = inspections.create([("upload", raw, image)], config)
        return dict(
            results[0],
            inspection_id=inspection_id,
            image_id=inspections.detail(inspection_id)["images"][0]["id"],
            conclusion=inspections.detail(inspection_id)["conclusion"],
        )

    @application.post("/api/inspections", status_code=201)
    async def inspect(files: list[UploadFile], config: str = Form(..., max_length=4096)):
        try:
            if not 1 <= len(files) <= MAX_BATCH:
                raise ValueError("Batch size must be 1..4 images")
            parsed = InspectionConfig.model_validate_json(config)
            uploads = []
            for file in files:
                raw = await file.read(MAX_FILE_BYTES + 1)
                image = await run_in_threadpool(decode_image, raw, file.content_type)
                uploads.append((file.filename or "upload", raw, image))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            for file in files:
                await file.close()
        inspection_id, _ = await run_in_threadpool(inspections.create, uploads, parsed)
        return inspections.detail(inspection_id)

    @application.get("/api/inspections")
    def history(
        q: str = Query("", max_length=80),
        status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] | None = None,
        conclusion: Literal["PASS", "REJECT", "REVIEW"] | None = None,
        model_id: str | None = Query(None, max_length=80),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
    ):
        return inspections.history(q, status, conclusion, model_id, page, page_size)

    @application.get("/api/inspections/{inspection_id}")
    def detail(inspection_id: str):
        return inspections.detail(inspection_id)

    @application.post("/api/inspections/{inspection_id}/reviews")
    def review(inspection_id: str, body: ReviewRequest):
        reviews.review(inspection_id, body)
        return inspections.detail(inspection_id)

    @application.get("/api/inspections/{inspection_id}/export")
    def export(inspection_id: str, format: Literal["json", "csv", "pdf"] = "json"):
        data, mime = reporting.export(inspection_id, format)
        safe_id = inspections.detail(inspection_id)["id"]
        return Response(
            data,
            media_type=mime,
            headers={
                "Content-Disposition": f'attachment; filename="inspection-{safe_id}.{format}"'
            },
        )

    @application.post("/api/images/{image_id}/gradcam")
    def gradcam(
        image_id: str,
        detection_id: int | None = Query(None, ge=1, le=10000),
        method: str | None = None,
    ):
        return inspections.gradcam(image_id, detection_id=detection_id)

    @application.get("/api/images/{image_id}/{kind}")
    def image(
        image_id: str,
        kind: Literal["original", "annotated", "gradcam", "cam-detail"],
        detection_id: int | None = Query(None, ge=1, le=10000),
        method: str | None = None,
    ):
        path = inspections.image_path(image_id, kind, method, detection_id)
        if kind == "original":
            with Image.open(path) as im:
                stream = io.BytesIO()
                ImageOps.exif_transpose(im).convert("RGB").save(stream, "PNG")
            return Response(stream.getvalue(), media_type="image/png")
        return FileResponse(path, media_type="image/jpeg")

    @application.get("/api/dashboard")
    def dashboard():
        return reporting.dashboard()

    return application


app = create_app()


def is_port_in_use(port, host="127.0.0.1"):
    with socket.socket() as sock:
        return sock.connect_ex((host, port)) == 0


def find_available_port(start_port=8501, max_tries=20):
    for port in range(start_port, start_port + max_tries):
        if not is_port_in_use(port):
            return port
    raise RuntimeError("No available port")


def start_server(port=8501, host="127.0.0.1"):
    import uvicorn

    actual_port = find_available_port(port)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=actual_port, workers=1))
    threading.Thread(target=server.run, daemon=True).start()
    return server, actual_port


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
