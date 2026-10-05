"""Store ready reconciler and GPU-worker service instances.

Revision ID: 0005_service_heartbeats
Revises: 0004_execution_claims
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_service_heartbeats"
down_revision: str | Sequence[str] | None = "0004_execution_claims"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create current ready-service heartbeats without altering job state."""
    op.create_table(
        "service_heartbeat",
        sa.Column("service_role", sa.String(length=32), nullable=False),
        sa.Column("instance_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("last_ready_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(
            "service_role",
            "instance_id",
            name="service_heartbeat_pkey",
        ),
        sa.CheckConstraint(
            "service_role IN ('reconciler', 'gpu_worker')",
            name="service_heartbeat_role_check",
        ),
        sa.CheckConstraint(
            "(get_byte(uuid_send(instance_id), 6) & 240) = 64 "
            "AND (get_byte(uuid_send(instance_id), 8) & 192) = 128",
            name="service_heartbeat_instance_id_uuid4_check",
        ),
    )
    op.create_index(
        "service_heartbeat_role_last_ready_at_idx",
        "service_heartbeat",
        ["service_role", sa.text("last_ready_at DESC")],
    )


def downgrade() -> None:
    """Remove ready-service heartbeats without altering job state."""
    op.drop_index(
        "service_heartbeat_role_last_ready_at_idx",
        table_name="service_heartbeat",
    )
    op.drop_table("service_heartbeat")
