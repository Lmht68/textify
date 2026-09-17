"""Create durable Transcription Job storage.

Revision ID: 0001_transcription_jobs
Revises:
Create Date: 2026-08-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_transcription_jobs"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create normalized durable Transcription Job tables."""
    op.create_table(
        "transcription_job",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("submitted_url", sa.String(length=2048), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("outcome", sa.String(length=10)),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("queue_deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column(
            "cancellation_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("error_code", sa.String(length=64)),
        sa.Column("error_message", sa.Text()),
        sa.UniqueConstraint("public_id", name="transcription_job_public_id_key"),
        sa.CheckConstraint(
            "length(public_id) = 36 "
            "AND public_id GLOB '????????-????-4???-[89ab]???-????????????' "
            "AND length(replace(public_id, '-', '')) = 32 "
            "AND replace(public_id, '-', '') NOT GLOB '*[^0-9a-f]*'",
            name="transcription_job_public_id_uuid4_check",
        ),
        sa.CheckConstraint(
            "length(submitted_url) BETWEEN 1 AND 2048",
            name="transcription_job_submitted_url_length_check",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'processing', 'finished')",
            name="transcription_job_status_check",
        ),
        sa.CheckConstraint(
            "outcome IS NULL OR outcome IN ('succeeded', 'failed', 'cancelled')",
            name="transcription_job_outcome_check",
        ),
        sa.CheckConstraint(
            "queue_deadline_at > submitted_at "
            "AND (started_at IS NULL OR started_at >= submitted_at) "
            "AND (finished_at IS NULL OR finished_at >= COALESCE(started_at, submitted_at))",
            name="transcription_job_timestamp_order_check",
        ),
        sa.CheckConstraint(
            "(error_code IS NULL AND error_message IS NULL) "
            "OR (error_code IS NOT NULL AND error_message IS NOT NULL "
            "AND length(error_message) > 0)",
            name="transcription_job_error_pair_check",
        ),
        sa.CheckConstraint(
            "(status = 'queued' AND outcome IS NULL AND started_at IS NULL "
            "AND finished_at IS NULL AND cancellation_requested = 0 "
            "AND error_code IS NULL AND error_message IS NULL) "
            "OR (status = 'processing' AND started_at IS NOT NULL AND outcome IS NULL "
            "AND finished_at IS NULL AND error_code IS NULL AND error_message IS NULL) "
            "OR (status = 'finished' AND outcome IS NOT NULL AND finished_at IS NOT NULL "
            "AND ((outcome = 'succeeded' AND started_at IS NOT NULL "
            "AND cancellation_requested = 0 AND error_code IS NULL "
            "AND error_message IS NULL) "
            "OR (outcome = 'failed' AND cancellation_requested = 0 "
            "AND error_code IS NOT NULL AND error_message IS NOT NULL) "
            "OR (outcome = 'cancelled' AND error_code IS NULL "
            "AND error_message IS NULL)))",
            name="transcription_job_lifecycle_check",
        ),
    )
    op.create_index(
        "transcription_job_status_submitted_at_id_idx",
        "transcription_job",
        ["status", "submitted_at", "id"],
    )
    op.create_index(
        "transcription_job_status_queue_deadline_at_idx",
        "transcription_job",
        ["status", "queue_deadline_at"],
    )
    op.create_table(
        "transcription_job_exclusion",
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("transcription_job.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("field_path", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint(
            "job_id", "field_path", name="transcription_job_exclusion_pkey"
        ),
        sa.CheckConstraint(
            "field_path IN ("
            "'source.platform', 'source.video_id', 'source.url', 'source.title', "
            "'source.description', 'source.channel', 'source.duration_seconds', "
            "'transcript.method', 'transcript.language', 'transcript.text', "
            "'transcript.segments'"
            ")",
            name="transcription_job_exclusion_field_path_check",
        ),
    )
    op.create_table(
        "transcription_job_result",
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("transcription_job.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("source_platform", sa.String(length=10)),
        sa.Column("source_video_id", sa.String(length=255)),
        sa.Column("source_url", sa.String(length=2048)),
        sa.Column("source_title", sa.Text()),
        sa.Column("source_description", sa.Text()),
        sa.Column("source_channel", sa.Text()),
        sa.Column("source_duration_seconds", sa.Integer()),
        sa.Column("transcript_method", sa.String(length=20)),
        sa.Column("transcript_language", sa.String(length=35)),
        sa.Column("transcript_text", sa.Text()),
        sa.CheckConstraint(
            "source_platform IS NULL OR source_platform IN "
            "('youtube', 'instagram', 'facebook', 'tiktok', 'x')",
            name="transcription_job_result_platform_check",
        ),
        sa.CheckConstraint(
            "transcript_method IS NULL OR transcript_method IN "
            "('youtube_captions', 'faster_whisper')",
            name="transcription_job_result_method_check",
        ),
        sa.CheckConstraint(
            "source_duration_seconds IS NULL OR source_duration_seconds > 0",
            name="transcription_job_result_duration_check",
        ),
        sa.CheckConstraint(
            "source_video_id IS NULL OR length(source_video_id) BETWEEN 1 AND 255",
            name="transcription_job_result_video_id_length_check",
        ),
        sa.CheckConstraint(
            "source_url IS NULL OR length(source_url) BETWEEN 1 AND 2048",
            name="transcription_job_result_source_url_length_check",
        ),
        sa.CheckConstraint(
            "transcript_language IS NULL OR length(transcript_language) BETWEEN 1 AND 35",
            name="transcription_job_result_language_length_check",
        ),
    )
    op.create_table(
        "transcription_job_segment",
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("transcription_job_result.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint(
            "job_id", "ordinal", name="transcription_job_segment_pkey"
        ),
        sa.CheckConstraint(
            "ordinal >= 0 AND start_seconds >= 0 AND end_seconds >= start_seconds "
            "AND length(text) > 0",
            name="transcription_job_segment_values_check",
        ),
    )


def downgrade() -> None:
    """Drop normalized durable Transcription Job tables in dependency order."""
    op.drop_table("transcription_job_segment")
    op.drop_table("transcription_job_result")
    op.drop_table("transcription_job_exclusion")
    op.drop_index(
        "transcription_job_status_queue_deadline_at_idx",
        table_name="transcription_job",
    )
    op.drop_index(
        "transcription_job_status_submitted_at_id_idx",
        table_name="transcription_job",
    )
    op.drop_table("transcription_job")
