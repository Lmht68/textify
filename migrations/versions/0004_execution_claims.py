"""Add durable worker execution claims to Execution Attempts.

Revision ID: 0004_execution_claims
Revises: 0003_dispatch_recovery
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_execution_claims"
down_revision: str | Sequence[str] | None = "0003_dispatch_recovery"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add reclaimable worker ownership to processing Execution Attempts."""
    op.add_column(
        "transcription_job_execution_attempt",
        sa.Column("worker_owner", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "transcription_job_execution_attempt",
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "transcription_job_execution_attempt",
        sa.Column(
            "claim_lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        "transcription_job_attempt_claim_state_check",
        "transcription_job_execution_attempt",
        "((worker_owner IS NULL AND last_heartbeat_at IS NULL "
        "AND claim_lease_expires_at IS NULL) OR "
        "(worker_owner IS NOT NULL AND last_heartbeat_at IS NOT NULL "
        "AND claim_lease_expires_at IS NOT NULL))",
    )
    op.create_check_constraint(
        "transcription_job_attempt_worker_owner_uuid4_check",
        "transcription_job_execution_attempt",
        "worker_owner IS NULL OR "
        "((get_byte(uuid_send(worker_owner), 6) & 240) = 64 "
        "AND (get_byte(uuid_send(worker_owner), 8) & 192) = 128)",
    )
    op.create_check_constraint(
        "transcription_job_attempt_claim_lease_order_check",
        "transcription_job_execution_attempt",
        "last_heartbeat_at <= claim_lease_expires_at",
    )
    op.create_index(
        "transcription_job_attempt_claim_lease_expires_at_job_id_idx",
        "transcription_job_execution_attempt",
        ["claim_lease_expires_at", "job_id"],
        postgresql_where=sa.text("claim_lease_expires_at IS NOT NULL"),
    )
    op.execute(
        "UPDATE transcription_job_execution_attempt AS attempt "
        "SET worker_owner = gen_random_uuid(), "
        "last_heartbeat_at = job.started_at, "
        "claim_lease_expires_at = job.started_at "
        "FROM transcription_job AS job "
        "WHERE job.id = attempt.job_id AND job.status = 'processing'"
    )


def downgrade() -> None:
    """Remove worker execution claim state from Execution Attempts."""
    op.drop_index(
        "transcription_job_attempt_claim_lease_expires_at_job_id_idx",
        table_name="transcription_job_execution_attempt",
    )
    op.drop_constraint(
        "transcription_job_attempt_claim_lease_order_check",
        "transcription_job_execution_attempt",
        type_="check",
    )
    op.drop_constraint(
        "transcription_job_attempt_worker_owner_uuid4_check",
        "transcription_job_execution_attempt",
        type_="check",
    )
    op.drop_constraint(
        "transcription_job_attempt_claim_state_check",
        "transcription_job_execution_attempt",
        type_="check",
    )
    op.drop_column(
        "transcription_job_execution_attempt",
        "claim_lease_expires_at",
    )
    op.drop_column("transcription_job_execution_attempt", "last_heartbeat_at")
    op.drop_column("transcription_job_execution_attempt", "worker_owner")
