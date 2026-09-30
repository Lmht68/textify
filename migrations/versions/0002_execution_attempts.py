"""Add one private Execution Attempt to every Transcription Job.

Revision ID: 0002_execution_attempts
Revises: 0001_postgresql_jobs
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_execution_attempts"
down_revision: str | Sequence[str] | None = "0001_postgresql_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create and backfill the required private Execution Attempt relation."""
    op.create_table(
        "transcription_job_execution_attempt",
        sa.Column(
            "job_id",
            sa.BigInteger(),
            sa.ForeignKey("transcription_job.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "execution_attempt_token",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "execution_attempt_token",
            name="transcription_job_execution_attempt_token_key",
        ),
        sa.CheckConstraint(
            "(get_byte(uuid_send(execution_attempt_token), 6) & 240) = 64 "
            "AND (get_byte(uuid_send(execution_attempt_token), 8) & 192) = 128",
            name="transcription_job_execution_attempt_token_uuid4_check",
        ),
    )
    op.execute(
        "INSERT INTO transcription_job_execution_attempt "
        "(job_id, execution_attempt_token) "
        "SELECT id, gen_random_uuid() FROM transcription_job"
    )
    _replace_storage_invariant(require_execution_attempt=True)
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER transcription_job_storage_execution_attempt_check
        AFTER INSERT OR UPDATE OR DELETE ON transcription_job_execution_attempt
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW
        EXECUTE FUNCTION enforce_transcription_job_storage_invariants();
        """
    )


def downgrade() -> None:
    """Restore issue-02 storage invariants and remove Execution Attempts."""
    op.execute(
        "DROP TRIGGER transcription_job_storage_execution_attempt_check "
        "ON transcription_job_execution_attempt"
    )
    _replace_storage_invariant(require_execution_attempt=False)
    op.drop_table("transcription_job_execution_attempt")


def _replace_storage_invariant(*, require_execution_attempt: bool) -> None:
    """Install the deferred storage invariant for the active schema revision.

    Args:
        require_execution_attempt: Whether every persisted job must own one attempt.
    """
    execution_attempt_declaration = (
        "            execution_attempt_count bigint;\n"
        if require_execution_attempt
        else ""
    )
    execution_attempt_validation = (
        """
            SELECT count(*)
            INTO execution_attempt_count
            FROM transcription_job_execution_attempt
            WHERE job_id = affected_job_id;

            IF execution_attempt_count <> 1 THEN
                RAISE EXCEPTION 'Transcription job storage invariant violated.'
                USING ERRCODE = '23514';
            END IF;
        """
        if require_execution_attempt
        else ""
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION enforce_transcription_job_storage_invariants()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            affected_job_id bigint;
            job_record transcription_job%ROWTYPE;
            result_record transcription_job_result%ROWTYPE;
            exclusion_paths text[];
            result_count bigint;
{execution_attempt_declaration}            segment_count bigint;
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

{execution_attempt_validation}            SELECT count(*)
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
