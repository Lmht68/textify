"""Create durable PostgreSQL Transcription Job storage.

Revision ID: 0001_postgresql_jobs
Revises:
Create Date: 2026-08-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_postgresql_jobs"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create normalized durable PostgreSQL Transcription Job tables."""
    op.create_table(
        "transcription_job",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=False),
            primary_key=True,
        ),
        sa.Column("public_id", postgresql.UUID(as_uuid=True), nullable=False),
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
            server_default=sa.text("false"),
        ),
        sa.Column("error_code", sa.String(length=64)),
        sa.Column("error_message", sa.Text()),
        sa.UniqueConstraint("public_id", name="transcription_job_public_id_key"),
        sa.CheckConstraint(
            "(get_byte(uuid_send(public_id), 6) & 240) = 64 "
            "AND (get_byte(uuid_send(public_id), 8) & 192) = 128",
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
            "AND (finished_at IS NULL OR finished_at >= "
            "COALESCE(started_at, submitted_at))",
            name="transcription_job_timestamp_order_check",
        ),
        sa.CheckConstraint(
            "(error_code IS NULL AND error_message IS NULL) "
            "OR (error_code IN ("
            "'invalid_url', 'unsupported_platform', 'invalid_request', "
            "'unsupported_content', 'video_too_long', 'invalid_media_duration', "
            "'unsupported_media', 'no_usable_transcript', "
            "'metadata_retrieval_failed', 'audio_download_failed', "
            "'transcription_failed', 'transcription_capacity_exceeded', "
            "'job_not_found', 'job_already_finished', 'job_store_unavailable', "
            "'queue_timeout', 'worker_interrupted', 'metadata_timeout', "
            "'audio_download_timeout', 'transcription_timeout', 'internal_error'"
            ") AND error_message IS NOT NULL AND length(error_message) > 0)",
            name="transcription_job_error_pair_check",
        ),
        sa.CheckConstraint(
            "(status = 'queued' AND outcome IS NULL AND started_at IS NULL "
            "AND finished_at IS NULL AND cancellation_requested = FALSE "
            "AND error_code IS NULL AND error_message IS NULL) "
            "OR (status = 'processing' AND started_at IS NOT NULL "
            "AND outcome IS NULL AND finished_at IS NULL "
            "AND error_code IS NULL AND error_message IS NULL) "
            "OR (status = 'finished' AND outcome IS NOT NULL "
            "AND finished_at IS NOT NULL AND ("
            "(outcome = 'succeeded' AND started_at IS NOT NULL "
            "AND cancellation_requested = FALSE AND error_code IS NULL "
            "AND error_message IS NULL) "
            "OR (outcome = 'failed' AND cancellation_requested = FALSE "
            "AND error_code IS NOT NULL AND error_message IS NOT NULL "
            "AND (started_at IS NOT NULL OR error_code = 'queue_timeout')) "
            "OR (outcome = 'cancelled' AND error_code IS NULL "
            "AND error_message IS NULL AND ("
            "(started_at IS NULL AND cancellation_requested = FALSE) "
            "OR (started_at IS NOT NULL AND cancellation_requested = TRUE)"
            "))))",
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
    op.create_index(
        "transcription_job_status_finished_at_idx",
        "transcription_job",
        ["status", "finished_at"],
    )
    op.create_table(
        "transcription_job_exclusion",
        sa.Column(
            "job_id",
            sa.BigInteger(),
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
            sa.BigInteger(),
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
            "transcript_language IS NULL OR "
            "length(transcript_language) BETWEEN 1 AND 35",
            name="transcription_job_result_language_length_check",
        ),
    )
    op.create_table(
        "transcription_job_segment",
        sa.Column(
            "job_id",
            sa.BigInteger(),
            sa.ForeignKey("transcription_job_result.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(precision=53), nullable=False),
        sa.Column("end_seconds", sa.Float(precision=53), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint(
            "job_id", "ordinal", name="transcription_job_segment_pkey"
        ),
        sa.CheckConstraint(
            "ordinal >= 0 "
            "AND start_seconds >= 0 "
            "AND end_seconds >= start_seconds "
            "AND start_seconds NOT IN ("
            "'NaN'::double precision, 'Infinity'::double precision, "
            "'-Infinity'::double precision"
            ") "
            "AND end_seconds NOT IN ("
            "'NaN'::double precision, 'Infinity'::double precision, "
            "'-Infinity'::double precision"
            ") "
            "AND length(text) > 0",
            name="transcription_job_segment_values_check",
        ),
    )
    op.execute(
        """
        CREATE FUNCTION enforce_transcription_job_storage_invariants()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            affected_job_id bigint;
            job_record transcription_job%ROWTYPE;
            result_record transcription_job_result%ROWTYPE;
            exclusion_paths text[];
            result_count bigint;
            segment_count bigint;
            minimum_segment_ordinal integer;
            maximum_segment_ordinal integer;
        BEGIN
            IF TG_TABLE_NAME = 'transcription_job' THEN
                affected_job_id := NEW.id;
            ELSIF TG_OP = 'DELETE' THEN
                affected_job_id := OLD.job_id;
            ELSE
                affected_job_id := NEW.job_id;
            END IF;

            SELECT *
            INTO job_record
            FROM transcription_job
            WHERE id = affected_job_id;

            IF NOT FOUND THEN
                RETURN NULL;
            END IF;

            SELECT count(*)
            INTO result_count
            FROM transcription_job_result
            WHERE job_id = affected_job_id;

            IF job_record.status <> 'finished'
            OR job_record.outcome <> 'succeeded' THEN
                IF result_count <> 0 THEN
                    RAISE EXCEPTION 'Transcription job storage invariant violated.'
                    USING ERRCODE = '23514';
                END IF;
                RETURN NULL;
            END IF;

            IF result_count <> 1 THEN
                RAISE EXCEPTION 'Transcription job storage invariant violated.'
                USING ERRCODE = '23514';
            END IF;

            SELECT *
            INTO result_record
            FROM transcription_job_result
            WHERE job_id = affected_job_id;

            SELECT array_agg(field_path ORDER BY field_path)
            INTO exclusion_paths
            FROM transcription_job_exclusion
            WHERE job_id = affected_job_id;
            exclusion_paths := COALESCE(exclusion_paths, ARRAY[]::text[]);

            IF (result_record.source_platform IS NULL)
            <> ('source.platform' = ANY(exclusion_paths))
            OR (result_record.source_video_id IS NULL)
            <> ('source.video_id' = ANY(exclusion_paths))
            OR (result_record.source_url IS NULL)
            <> ('source.url' = ANY(exclusion_paths))
            OR (result_record.source_title IS NULL)
            <> ('source.title' = ANY(exclusion_paths))
            OR (result_record.source_description IS NULL)
            <> ('source.description' = ANY(exclusion_paths))
            OR (result_record.source_channel IS NULL)
            <> ('source.channel' = ANY(exclusion_paths))
            OR (result_record.source_duration_seconds IS NULL)
            <> ('source.duration_seconds' = ANY(exclusion_paths))
            OR (result_record.transcript_method IS NULL)
            <> ('transcript.method' = ANY(exclusion_paths))
            OR (result_record.transcript_language IS NULL)
            <> ('transcript.language' = ANY(exclusion_paths))
            OR (result_record.transcript_text IS NULL)
            <> ('transcript.text' = ANY(exclusion_paths)) THEN
                RAISE EXCEPTION 'Transcription job storage invariant violated.'
                USING ERRCODE = '23514';
            END IF;

            SELECT count(*), min(ordinal), max(ordinal)
            INTO segment_count, minimum_segment_ordinal, maximum_segment_ordinal
            FROM transcription_job_segment
            WHERE job_id = affected_job_id;

            IF 'transcript.segments' = ANY(exclusion_paths) THEN
                IF segment_count <> 0 THEN
                    RAISE EXCEPTION 'Transcription job storage invariant violated.'
                    USING ERRCODE = '23514';
                END IF;
            ELSIF segment_count > 0
            AND (
                minimum_segment_ordinal <> 0
                OR maximum_segment_ordinal <> segment_count - 1
            ) THEN
                RAISE EXCEPTION 'Transcription job storage invariant violated.'
                USING ERRCODE = '23514';
            END IF;

            RETURN NULL;
        END;
        $$;
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER transcription_job_storage_job_check
        AFTER INSERT OR UPDATE ON transcription_job
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW
        EXECUTE FUNCTION enforce_transcription_job_storage_invariants();
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER transcription_job_storage_exclusion_check
        AFTER INSERT OR UPDATE OR DELETE ON transcription_job_exclusion
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW
        EXECUTE FUNCTION enforce_transcription_job_storage_invariants();
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER transcription_job_storage_result_check
        AFTER INSERT OR UPDATE OR DELETE ON transcription_job_result
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW
        EXECUTE FUNCTION enforce_transcription_job_storage_invariants();
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER transcription_job_storage_segment_check
        AFTER INSERT OR UPDATE OR DELETE ON transcription_job_segment
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW
        EXECUTE FUNCTION enforce_transcription_job_storage_invariants();
        """
    )


def downgrade() -> None:
    """Drop normalized durable PostgreSQL Transcription Job tables."""
    op.execute(
        "DROP TRIGGER transcription_job_storage_segment_check "
        "ON transcription_job_segment"
    )
    op.execute(
        "DROP TRIGGER transcription_job_storage_result_check "
        "ON transcription_job_result"
    )
    op.execute(
        "DROP TRIGGER transcription_job_storage_exclusion_check "
        "ON transcription_job_exclusion"
    )
    op.execute("DROP TRIGGER transcription_job_storage_job_check ON transcription_job")
    op.execute("DROP FUNCTION enforce_transcription_job_storage_invariants()")
    op.drop_table("transcription_job_segment")
    op.drop_table("transcription_job_result")
    op.drop_table("transcription_job_exclusion")
    op.drop_index(
        "transcription_job_status_finished_at_idx",
        table_name="transcription_job",
    )
    op.drop_index(
        "transcription_job_status_queue_deadline_at_idx",
        table_name="transcription_job",
    )
    op.drop_index(
        "transcription_job_status_submitted_at_id_idx",
        table_name="transcription_job",
    )
    op.drop_table("transcription_job")
