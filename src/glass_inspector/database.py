"""PostgreSQL persistence. Schema changes are applied through Alembic only."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def now():
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ModelRecord(Base):
    __tablename__ = "model_registry"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    status: Mapped[str] = mapped_column(String(20))
    manifest: Mapped[dict] = mapped_column(JSON)


class Inspection(Base):
    __tablename__ = "inspections"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    conclusion: Mapped[str] = mapped_column(String(20), default="REVIEW", index=True)
    explanation: Mapped[str] = mapped_column(Text, default="Awaiting processing")
    model_id: Mapped[str] = mapped_column(ForeignKey("model_registry.id"))
    model_version: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=0)


class InspectionImage(Base):
    __tablename__ = "inspection_images"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    original_path: Mapped[str] = mapped_column(Text)
    annotated_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("inspection_images.id"), index=True)
    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float)
    bbox_xyxy: Mapped[list] = mapped_column(JSON)
    bbox_normalized: Mapped[list] = mapped_column(JSON)
    model_id: Mapped[str] = mapped_column(String(80))
    latency_ms: Mapped[float] = mapped_column(Float)


class InspectionReview(Base):
    __tablename__ = "inspection_reviews"
    __table_args__ = (UniqueConstraint("inspection_id", "version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    reviewer: Mapped[str] = mapped_column(String(120))
    reason: Mapped[str] = mapped_column(Text)
    previous_conclusion: Mapped[str] = mapped_column(String(20))
    conclusion: Mapped[str] = mapped_column(String(20))
    version: Mapped[int] = mapped_column(Integer)


def connect(url):
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": 5, "options": "-c timezone=UTC"},
    )
    return engine, sessionmaker(engine, expire_on_commit=False)
