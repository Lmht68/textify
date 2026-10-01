"""Add durable publication recovery state to Execution Attempts.

Revision ID: 0003_dispatch_recovery
Revises: 0002_execution_attempts
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_dispatch_recovery"
down_revision: str | Sequence[str] | None = "0002_execution_attempts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add due scheduling, publication lease, and publication history state."""
    op.add_column(
        "transcription_job_execution_attempt",
        sa.Column("next_dispatch_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "transcription_job_execution_attempt",
        sa.Column(
            "publication_lease_owner",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "transcription_job_execution_attempt",
        sa.Column(
            "publication_lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "transcription_job_execution_attempt",
        sa.Column("last_dispatched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "transcription_job_execution_attempt",
        sa.Column(
            "dispatch_count",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.execute(
        "UPDATE transcription_job_execution_attempt AS attempt "
        "SET next_dispatch_at = job.submitted_at "
        "FROM transcription_job AS job "
        "WHERE job.id = attempt.job_id"
    )
    op.execute("SET CONSTRAINTS ALL IMMEDIATE")
    op.alter_column(
        "transcription_job_execution_attempt",
        "next_dispatch_at",
        nullable=False,
    )
    op.create_check_constraint(
        "transcription_job_attempt_publication_lease_pair_check",
        "transcription_job_execution_attempt",
        "(publication_lease_owner IS NULL) = (publication_lease_expires_at IS NULL)",
    )
    op.create_check_constraint(
        "transcription_job_attempt_lease_owner_uuid4_check",
        "transcription_job_execution_attempt",
        "publication_lease_owner IS NULL OR "
        "((get_byte(uuid_send(publication_lease_owner), 6) & 240) = 64 "
        "AND (get_byte(uuid_send(publication_lease_owner), 8) & 192) = 128)",
    )
    op.create_check_constraint(
        "transcription_job_attempt_dispatch_history_check",
        "transcription_job_execution_attempt",
        "dispatch_count >= 0 AND "
        "((dispatch_count = 0 AND last_dispatched_at IS NULL) OR "
        "(dispatch_count > 0 AND last_dispatched_at IS NOT NULL))",
    )
    op.create_index(
        "transcription_job_attempt_next_dispatch_at_job_id_idx",
        "transcription_job_execution_attempt",
        ["next_dispatch_at", "job_id"],
    )


def downgrade() -> None:
    """Remove publication recovery state while retaining Execution Attempts."""
    op.drop_index(
        "transcription_job_attempt_next_dispatch_at_job_id_idx",
        table_name="transcription_job_execution_attempt",
    )
    op.drop_constraint(
        "transcription_job_attempt_dispatch_history_check",
        "transcription_job_execution_attempt",
        type_="check",
    )
    op.drop_constraint(
        "transcription_job_attempt_lease_owner_uuid4_check",
        "transcription_job_execution_attempt",
        type_="check",
    )
    op.drop_constraint(
        "transcription_job_attempt_publication_lease_pair_check",
        "transcription_job_execution_attempt",
        type_="check",
    )
    op.drop_column("transcription_job_execution_attempt", "dispatch_count")
    op.drop_column("transcription_job_execution_attempt", "last_dispatched_at")
    op.drop_column(
        "transcription_job_execution_attempt",
        "publication_lease_expires_at",
    )
    op.drop_column(
        "transcription_job_execution_attempt",
        "publication_lease_owner",
    )
    op.drop_column("transcription_job_execution_attempt", "next_dispatch_at")
