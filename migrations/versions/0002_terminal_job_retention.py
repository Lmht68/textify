"""Add terminal Transcription Job retention index.

Revision ID: 0002_terminal_job_retention
Revises: 0001_transcription_jobs
Create Date: 2026-08-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_terminal_job_retention"
down_revision: str | Sequence[str] | None = "0001_transcription_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the index used to delete expired terminal jobs."""
    op.create_index(
        "transcription_job_status_finished_at_idx",
        "transcription_job",
        ["status", "finished_at"],
    )


def downgrade() -> None:
    """Drop the index used to delete expired terminal jobs."""
    op.drop_index(
        "transcription_job_status_finished_at_idx",
        table_name="transcription_job",
    )
