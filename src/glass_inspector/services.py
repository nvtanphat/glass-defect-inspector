"""Transactional inspection, review and reporting services."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import logging
import os
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from PIL import Image, ImageOps
from sqlalchemy import func, select, update

from glass_inspector.database import (
    Detection,
    Inspection,
    InspectionImage,
    InspectionReview,
    ModelRecord,
)
from glass_inspector.gradcam import VERSION as GRADCAM_VERSION
from glass_inspector.gradcam import explain
from glass_inspector.inference_engine import (
    CLASS_NAMES,
    compute_slice_intervals,
    load_roi_config,
    run_inspection,
)
from glass_inspector.schemas import DetectionResult, quality_policy

log = logging.getLogger("inspector")


class InferenceService:
    def __init__(self, registry):
        self.registry = registry
        # One bounded worker per process protects model caches and CPU/GPU memory.
        self.gate = threading.BoundedSemaphore(1)

    def predict(self, image, record, weights, config):
        options = config.model_dump(exclude={"model_id"})
        result = run_inspection(image, weights, **options)
        result.update(model_id=record["id"], model_version=record["sha256"])
        for detection in result["detections"]:
            detection.update(model_id=record["id"], latency_ms=result["latency_ms"])
            DetectionResult.model_validate(detection)
        return result


class InspectionService:
    def __init__(self, sessions, storage: Path, inference):
        self.sessions, self.storage, self.inference = sessions, storage, inference
        storage.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        with self.sessions.begin() as db:
            for record in self.inference.registry.public():
                db.merge(ModelRecord(id=record["id"], status=record["status"], manifest=record))
            # Single server process: unfinished jobs cannot survive a process restart.
            db.execute(
                update(Inspection)
                .where(Inspection.status.in_(["PENDING", "PROCESSING"]))
                .values(
                    status="FAILED",
                    error="Server restarted before completion",
                    explanation="Interrupted processing; review required",
                    conclusion="REVIEW",
                )
            )

    def create(self, uploads, config):
        if not self.inference.gate.acquire(blocking=False):
            raise HTTPException(
                429, "Inference capacity reached; retry later", headers={"Retry-After": "5"}
            )
        inspection_id = str(uuid4())
        try:
            try:
                record, weights = self.inference.registry.resolve(config.model_id)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            except LookupError as exc:
                raise HTTPException(503, str(exc)) from exc
            if config.enable_roi is None:
                config = config.model_copy(update={"enable_roi": record["enable_roi"]})
            if config.nms_mode is None:
                config = config.model_copy(update={"nms_mode": record["default_nms"]})
            if config.nms_mode not in record["nms_modes"]:
                raise HTTPException(422, "Selected model requires its native NMS postprocessor")
            if config.enable_tiling:
                for _, _, image in uploads:
                    width, height = image.size
                    group = (
                        load_roi_config()["groups"].get(f"{width}x{height}")
                        if config.enable_roi
                        else None
                    )
                    if group:
                        x1, y1, x2, y2 = group["roi_xyxyn"]
                        width, height = (
                            round(x2 * width) - round(x1 * width),
                            round(y2 * height) - round(y1 * height),
                        )
                    count = len(
                        compute_slice_intervals(width, config.tile_size, config.overlap_ratio)
                    ) * len(compute_slice_intervals(height, config.tile_size, config.overlap_ratio))
                    if count > 64:
                        raise HTTPException(422, "Tiling exceeds 64 tiles per image")
            folder = self.storage / inspection_id
            folder.mkdir()
            images = []
            for _filename, raw, image in uploads:
                image_id = str(uuid4())
                original = folder / f"{image_id}.original"
                self.atomic_write(original, raw)
                images.append((image_id, original, image))
            with self.sessions.begin() as db:
                snapshot = config.model_dump()
                snapshot["model_manifest"] = record.copy()
                db.add(
                    Inspection(
                        id=inspection_id,
                        status="PENDING",
                        model_id=record["id"],
                        model_version=record["sha256"],
                        config=snapshot,
                    )
                )
                db.flush()
                for (image_id, original, _), (filename, raw, _) in zip(
                    images, uploads, strict=True
                ):
                    db.add(
                        InspectionImage(
                            id=image_id,
                            inspection_id=inspection_id,
                            filename=Path(filename.replace("\\", "/")).name[:255],
                            sha256=hashlib.sha256(raw).hexdigest(),
                            original_path=str(original),
                        )
                    )
            with self.sessions.begin() as db:
                db.get(Inspection, inspection_id).status = "PROCESSING"
            results = []
            for image_id, _original, image in images:
                result = self.inference.predict(image, record, weights, config)
                annotated = folder / f"{image_id}.annotated.jpg"
                self.atomic_write(
                    annotated, base64.b64decode(result["annotated_b64"].split(",", 1)[1])
                )
                stored = {
                    k: v for k, v in result.items() if k not in {"image_b64", "annotated_b64"}
                }
                with self.sessions.begin() as db:
                    row = db.get(InspectionImage, image_id)
                    row.annotated_path, row.result = str(annotated), stored
                    row.latency_ms = result["latency_ms"]
                    for d in result["detections"]:
                        db.add(
                            Detection(
                                image_id=image_id,
                                **{
                                    k: d[k]
                                    for k in (
                                        "class_id",
                                        "class_name",
                                        "confidence",
                                        "bbox_xyxy",
                                        "bbox_normalized",
                                        "model_id",
                                        "latency_ms",
                                    )
                                },
                            )
                        )
                results.append(result)
            conclusion, explanation = quality_policy(results)
            with self.sessions.begin() as db:
                row = db.get(Inspection, inspection_id)
                row.status, row.conclusion, row.explanation = "COMPLETED", conclusion, explanation
                row.config = dict(
                    row.config,
                    automatic_assessment=dict(
                        conclusion=conclusion,
                        explanation=explanation,
                        policy_version="glass-quality-v1",
                    ),
                )
            log.info("inspection_completed", extra={"inspection_id": inspection_id})
            return inspection_id, results
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("inspection_failed", extra={"inspection_id": inspection_id})
            with self.sessions.begin() as db:
                row = db.get(Inspection, inspection_id)
                if row:
                    row.status, row.error = "FAILED", type(exc).__name__
                    row.explanation = "Processing failed; no quality approval issued"
            raise HTTPException(
                500, {"message": "Inspection failed", "inspection_id": inspection_id}
            ) from exc
        finally:
            self.inference.gate.release()

    @staticmethod
    def atomic_write(path, raw):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def detail(self, inspection_id):
        with self.sessions() as db:
            row = db.get(Inspection, inspection_id)
            if row is None:
                raise HTTPException(404, "Inspection not found")
            data = self.summary(row)
            data.update(config=row.config, error=row.error, explanation=row.explanation)
            data["images"] = []
            for image in db.scalars(
                select(InspectionImage)
                .where(InspectionImage.inspection_id == inspection_id)
                .order_by(InspectionImage.id)
            ):
                data["images"].append(
                    dict(
                        id=image.id,
                        filename=image.filename,
                        sha256=image.sha256,
                        latency_ms=image.latency_ms,
                        result=image.result,
                        original_url=f"/api/images/{image.id}/original",
                        annotated_url=f"/api/images/{image.id}/annotated"
                        if image.annotated_path
                        else None,
                    )
                )
            data["reviews"] = [
                dict(
                    id=r.id,
                    reviewer=r.reviewer,
                    reason=r.reason,
                    previous_conclusion=r.previous_conclusion,
                    conclusion=r.conclusion,
                    version=r.version,
                    created_at=r.created_at.isoformat(),
                )
                for r in db.scalars(
                    select(InspectionReview)
                    .where(InspectionReview.inspection_id == inspection_id)
                    .order_by(InspectionReview.version)
                )
            ]
            return data

    @staticmethod
    def summary(row):
        return dict(
            id=row.id,
            created_at=row.created_at.isoformat(),
            status=row.status,
            conclusion=row.conclusion,
            model_id=row.model_id,
            model_version=row.model_version,
            version=row.version,
        )

    def history(self, q, status, conclusion, model_id, page, page_size):
        clauses = []
        if q:
            clauses.append(Inspection.id.contains(q, autoescape=True))
        for name, value in [("status", status), ("conclusion", conclusion), ("model_id", model_id)]:
            if value:
                clauses.append(getattr(Inspection, name) == value)
        with self.sessions() as db:
            total = db.scalar(select(func.count()).select_from(Inspection).where(*clauses))
            rows = db.scalars(
                select(Inspection)
                .where(*clauses)
                .order_by(Inspection.created_at.desc(), Inspection.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            return dict(
                items=[self.summary(r) for r in rows], total=total, page=page, page_size=page_size
            )

    def image_path(self, image_id, kind):
        with self.sessions() as db:
            row = db.get(InspectionImage, image_id)
            if row and kind == "gradcam":
                path = (
                    Path(row.original_path).parent / f"{image_id}.gradcam.jpg"
                    if (row.result or {}).get("gradcam")
                    else None
                )
            else:
                path = (
                    (row.original_path if kind == "original" else row.annotated_path)
                    if row
                    else None
                )
            if not path or not Path(path).is_file():
                raise HTTPException(404, "Image not found")
            return Path(path)

    def gradcam(self, image_id):
        if not self.inference.gate.acquire(blocking=False):
            raise HTTPException(
                429, "Inference capacity reached; retry later", headers={"Retry-After": "5"}
            )
        try:
            with self.sessions() as db:
                image = db.get(InspectionImage, image_id)
                if image is None:
                    raise HTTPException(404, "Image not found")
                inspection = db.get(Inspection, image.inspection_id)
                if inspection.status != "COMPLETED" or not image.result:
                    raise HTTPException(409, "Complete the inspection before requesting Grad-CAM")
                path = Path(image.original_path)
                saved_result = dict(image.result)
                config = dict(inspection.config)
                model_id, model_version = inspection.model_id, inspection.model_version
            output = path.parent / f"{image_id}.gradcam.jpg"
            cached = saved_result.get("gradcam")
            if cached and cached.get("version") == GRADCAM_VERSION and output.is_file():
                return dict(cached, image_url=f"/api/images/{image_id}/gradcam")
            try:
                record, weights = self.inference.registry.resolve(model_id)
            except (ValueError, LookupError) as exc:
                raise HTTPException(503, "The saved inspection model is unavailable") from exc
            if record["sha256"] != model_version:
                raise HTTPException(409, "Checkpoint changed since this inspection")
            if not path.is_file():
                raise HTTPException(404, "Original image not found")
            with Image.open(path) as original:
                source = ImageOps.exif_transpose(original).convert("RGB")
            raw, metadata = explain(source, weights, record["adapter"], saved_result, config)
            metadata.update(model_id=model_id, model_version=model_version)
            self.atomic_write(output, raw)
            with self.sessions.begin() as db:
                image = db.get(InspectionImage, image_id)
                image.result = dict(image.result, gradcam=metadata)
            return dict(metadata, image_url=f"/api/images/{image_id}/gradcam")
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("gradcam_failed", extra={"image_id": image_id})
            raise HTTPException(503, "Grad-CAM could not be generated for this image") from exc
        finally:
            self.inference.gate.release()


class InspectionReviewService:
    def __init__(self, sessions):
        self.sessions = sessions

    def review(self, inspection_id, request):
        with self.sessions.begin() as db:
            row = db.scalar(
                select(Inspection).where(Inspection.id == inspection_id).with_for_update()
            )
            if not row:
                raise HTTPException(404, "Inspection not found")
            if row.status != "COMPLETED":
                raise HTTPException(409, "Only completed inspections can be reviewed")
            if row.version != request.expected_version:
                raise HTTPException(409, "Review changed; reload before submitting")
            db.add(
                InspectionReview(
                    inspection_id=inspection_id,
                    reviewer=request.reviewer,
                    reason=request.reason,
                    previous_conclusion=row.conclusion,
                    conclusion=request.conclusion,
                    version=row.version + 1,
                )
            )
            row.conclusion, row.explanation = request.conclusion, request.reason
            row.version += 1


class ReportingService:
    def __init__(self, sessions, inspections):
        self.sessions, self.inspections = sessions, inspections

    def dashboard(self):
        with self.sessions() as db:
            total = db.scalar(select(func.count()).select_from(Inspection))
            completed = Inspection.status == "COMPLETED"
            conclusions = dict(
                db.execute(
                    select(Inspection.conclusion, func.count())
                    .where(completed)
                    .group_by(Inspection.conclusion)
                ).all()
            )
            classes = dict(
                db.execute(
                    select(Detection.class_name, func.count())
                    .join(InspectionImage, Detection.image_id == InspectionImage.id)
                    .join(Inspection, InspectionImage.inspection_id == Inspection.id)
                    .where(completed)
                    .group_by(Detection.class_name)
                ).all()
            )
            trend = [
                dict(date=str(day), conclusion=c, count=n)
                for day, c, n in db.execute(
                    select(func.date(Inspection.created_at), Inspection.conclusion, func.count())
                    .where(completed)
                    .group_by(func.date(Inspection.created_at), Inspection.conclusion)
                    .order_by(func.date(Inspection.created_at))
                ).all()
            ]
            latency, image_count = db.execute(
                select(func.avg(InspectionImage.latency_ms), func.count())
                .join(Inspection, InspectionImage.inspection_id == Inspection.id)
                .where(completed)
            ).one()
            return dict(
                total_inspections=total,
                statuses=dict(
                    db.execute(
                        select(Inspection.status, func.count()).group_by(Inspection.status)
                    ).all()
                ),
                conclusions={k: conclusions.get(k, 0) for k in ["PASS", "REJECT", "REVIEW"]},
                defect_classes={k: classes.get(k, 0) for k in CLASS_NAMES},
                trend=trend,
                completed_images=image_count,
                mean_latency_ms=latency,
                grain="Conclusions per completed session; defects and latency per completed image",
            )

    def export(self, inspection_id, format):
        detail = self.inspections.detail(inspection_id)
        if format == "json":
            import json

            return json.dumps(detail, ensure_ascii=False, indent=2).encode(), "application/json"
        if format == "csv":
            import json

            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "session_id",
                    "status",
                    "conclusion",
                    "model_id",
                    "model_sha256",
                    "image_id",
                    "image_sha256",
                    "class",
                    "confidence",
                    "bbox_xyxy",
                    "created_at",
                    "explanation",
                    "configuration",
                    "reviews",
                ]
            )
            for image in detail["images"]:
                for d in (image["result"] or {}).get("detections", []) or [None]:
                    writer.writerow(
                        [
                            inspection_id,
                            detail["status"],
                            detail["conclusion"],
                            detail["model_id"],
                            detail["model_version"],
                            image["id"],
                            image["sha256"],
                            d["class_name"] if d else "",
                            d["confidence"] if d else "",
                            d["bbox_xyxy"] if d else "",
                            detail["created_at"],
                            "'" + detail["explanation"]
                            if detail["explanation"].startswith(("=", "+", "-", "@"))
                            else detail["explanation"],
                            json.dumps(detail["config"], ensure_ascii=False),
                            json.dumps(detail["reviews"], ensure_ascii=False),
                        ]
                    )
            return stream.getvalue().encode("utf-8-sig"), "text/csv"
        from xml.sax.saxutils import escape

        import matplotlib
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.platypus import Paragraph

        if "InspectorUnicode" not in pdfmetrics.getRegisteredFontNames():
            font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
            pdfmetrics.registerFont(TTFont("InspectorUnicode", str(font_path)))

        stream = io.BytesIO()
        canvas = Canvas(stream)
        canvas.setTitle("Glass Defect Inspection " + inspection_id)
        for image in detail["images"]:
            canvas.drawString(35, 805, "Glass Defect Inspector")
            lines = [
                inspection_id,
                f"{detail['status']} / {detail['conclusion']}",
                f"Model: {detail['model_id']}",
                "Weights SHA256: " + detail["model_version"],
                "Image: " + image["id"],
                "Image SHA256: " + image["sha256"],
            ]
            for i, line in enumerate(lines):
                canvas.setFont("InspectorUnicode", 8)
                canvas.drawString(35, 780 - i * 16, line)
            if image["annotated_url"]:
                canvas.drawImage(
                    ImageReader(str(self.inspections.image_path(image["id"], "annotated"))),
                    35,
                    350,
                    width=520,
                    height=300,
                    preserveAspectRatio=True,
                )
            detections = (image["result"] or {}).get("detections", [])
            canvas.drawString(
                35,
                325,
                f"Detections: {len(detections)}; technician reviews: {len(detail['reviews'])}",
            )
            canvas.showPage()
            for start in range(0, len(detections), 36):
                canvas.setFont("InspectorUnicode", 9)
                canvas.drawString(35, 805, "Detections: " + image["id"])
                for i, d in enumerate(detections[start : start + 36]):
                    canvas.drawString(
                        35,
                        775 - i * 20,
                        f"{d['class_name']} conf={d['confidence']:.4f} xyxy={d['bbox_xyxy']}",
                    )
                canvas.showPage()
        # Export the decision explanation, exact configuration and append-only
        # review trail with Unicode and wrapping, including zero-detection sessions.
        style = getSampleStyleSheet()["BodyText"]
        style.fontName = "InspectorUnicode"
        import json

        texts = [
            "Inspection configuration and review trail",
            detail["explanation"],
            json.dumps(detail["config"], ensure_ascii=False, indent=2),
        ]
        texts.extend(
            f"v{r['version']} / {r['created_at']} / {r['reviewer']}: "
            f"{r['previous_conclusion']} -> {r['conclusion']} / {r['reason']}"
            for r in detail["reviews"]
        )
        y = 805
        for text in texts:
            # Limit paragraph size; long manifests flow over multiple pages.
            for line in text.splitlines():
                paragraph = Paragraph(escape(line), style)
                _, height = paragraph.wrap(520, 750)
                if y - height < 40:
                    canvas.showPage()
                    y = 805
                paragraph.drawOn(canvas, 35, y - height)
                y -= height + 5
        canvas.save()
        return stream.getvalue(), "application/pdf"
