"""Store retry-safe private API idempotency keys.

Revision ID: 0006_private_api_idempotency
Revises: 0005_service_heartbeats
Create Date: 2026-10-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_private_api_idempotency"
down_revision: str | Sequence[str] | None = "0005_service_heartbeats"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the durable idempotency-key association table."""
    op.create_table(
        "transcription_job_idempotency",
        sa.Column(
            "job_id",
            sa.BigInteger(),
            sa.ForeignKey("transcription_job.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "idempotency_key",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("job_id", name="transcription_job_idempotency_pkey"),
        sa.UniqueConstraint(
            "idempotency_key",
            name="transcription_job_idempotency_key_key",
        ),
        sa.CheckConstraint(
            "(get_byte(uuid_send(idempotency_key), 6) & 240) = 64 "
            "AND (get_byte(uuid_send(idempotency_key), 8) & 192) = 128",
            name="transcription_job_idempotency_key_uuid4_check",
        ),
    )


def downgrade() -> None:
    """Remove idempotency-key storage while retaining the job lifecycle tables."""
    op.drop_table("transcription_job_idempotency")
