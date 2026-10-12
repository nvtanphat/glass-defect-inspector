"""Initial inspection schema (immutable migration definition)."""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None


def upgrade():
    op.create_table(
        "model_registry",
        sa.Column("id", sa.String(80), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("manifest", sa.JSON, nullable=False),
    )
    op.create_table(
        "inspections",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("conclusion", sa.String(20), nullable=False),
        sa.Column("explanation", sa.Text, nullable=False),
        sa.Column("model_id", sa.String(80), sa.ForeignKey("model_registry.id"), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("config", sa.JSON, nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("version", sa.Integer, nullable=False),
        sa.CheckConstraint("status IN ('PENDING','PROCESSING','COMPLETED','FAILED')"),
        sa.CheckConstraint("conclusion IN ('PASS','REJECT','REVIEW')"),
    )
    op.create_table(
        "inspection_images",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("inspection_id", sa.String(36), sa.ForeignKey("inspections.id"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("original_path", sa.Text, nullable=False),
        sa.Column("annotated_path", sa.Text),
        sa.Column("result", sa.JSON),
        sa.Column("latency_ms", sa.Float),
    )
    op.create_table(
        "detections",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("image_id", sa.String(36), sa.ForeignKey("inspection_images.id"), nullable=False),
        sa.Column("class_id", sa.Integer, nullable=False),
        sa.Column("class_name", sa.String(40), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("bbox_xyxy", sa.JSON, nullable=False),
        sa.Column("bbox_normalized", sa.JSON, nullable=False),
        sa.Column("model_id", sa.String(80), nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.CheckConstraint("class_id BETWEEN 0 AND 4"),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1"),
    )
    op.create_table(
        "inspection_reviews",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("inspection_id", sa.String(36), sa.ForeignKey("inspections.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewer", sa.String(120), nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("previous_conclusion", sa.String(20), nullable=False),
        sa.Column("conclusion", sa.String(20), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.UniqueConstraint("inspection_id", "version"),
        sa.CheckConstraint("conclusion IN ('PASS','REJECT','REVIEW')"),
    )
    for table, column in [
        ("inspections", "created_at"),
        ("inspections", "status"),
        ("inspections", "conclusion"),
        ("inspection_images", "inspection_id"),
        ("detections", "image_id"),
        ("inspection_reviews", "inspection_id"),
    ]:
        op.create_index(f"ix_{table}_{column}", table, [column])


def downgrade():
    for table in [
        "inspection_reviews",
        "detections",
        "inspection_images",
        "inspections",
        "model_registry",
    ]:
        op.drop_table(table)
