"""SQLite persistence for durable Transcription Jobs."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

if TYPE_CHECKING:
    from textify.transcription.jobs import (
        NewQueuedTranscriptionJob,
        QueuedTranscriptionJob,
    )

metadata = MetaData()

transcription_job = Table(
    "transcription_job",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("public_id", String(36), nullable=False),
    Column("submitted_url", String(2048), nullable=False),
    Column("status", String(10), nullable=False),
    Column("outcome", String(10)),
    Column("submitted_at", DateTime(timezone=True), nullable=False),
    Column("queue_deadline_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Column("cancellation_requested", Boolean, nullable=False, server_default=text("0")),
    Column("error_code", String(64)),
    Column("error_message", Text),
    UniqueConstraint("public_id", name="transcription_job_public_id_key"),
    CheckConstraint(
        "length(public_id) = 36 "
        "AND public_id GLOB '????????-????-4???-[89ab]???-????????????' "
        "AND length(replace(public_id, '-', '')) = 32 "
        "AND replace(public_id, '-', '') NOT GLOB '*[^0-9a-f]*'",
        name="transcription_job_public_id_uuid4_check",
    ),
    CheckConstraint(
        "length(submitted_url) BETWEEN 1 AND 2048",
        name="transcription_job_submitted_url_length_check",
    ),
    CheckConstraint(
        "status IN ('queued', 'processing', 'finished')",
        name="transcription_job_status_check",
    ),
    CheckConstraint(
        "outcome IS NULL OR outcome IN ('succeeded', 'failed', 'cancelled')",
        name="transcription_job_outcome_check",
    ),
    CheckConstraint(
        "queue_deadline_at > submitted_at "
        "AND (started_at IS NULL OR started_at >= submitted_at) "
        "AND (finished_at IS NULL OR finished_at >= COALESCE(started_at, submitted_at))",
        name="transcription_job_timestamp_order_check",
    ),
    CheckConstraint(
        "(error_code IS NULL AND error_message IS NULL) "
        "OR (error_code IS NOT NULL AND error_message IS NOT NULL "
        "AND length(error_message) > 0)",
        name="transcription_job_error_pair_check",
    ),
    CheckConstraint(
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

Index(
    "transcription_job_status_submitted_at_id_idx",
    transcription_job.c.status,
    transcription_job.c.submitted_at,
    transcription_job.c.id,
)
Index(
    "transcription_job_status_queue_deadline_at_idx",
    transcription_job.c.status,
    transcription_job.c.queue_deadline_at,
)

transcription_job_exclusion = Table(
    "transcription_job_exclusion",
    metadata,
    Column(
        "job_id",
        Integer,
        ForeignKey("transcription_job.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("field_path", String(64), nullable=False),
    PrimaryKeyConstraint(
        "job_id", "field_path", name="transcription_job_exclusion_pkey"
    ),
    CheckConstraint(
        "field_path IN ("
        "'source.platform', 'source.video_id', 'source.url', 'source.title', "
        "'source.description', 'source.channel', 'source.duration_seconds', "
        "'transcript.method', 'transcript.language', 'transcript.text', "
        "'transcript.segments'"
        ")",
        name="transcription_job_exclusion_field_path_check",
    ),
)

transcription_job_result = Table(
    "transcription_job_result",
    metadata,
    Column(
        "job_id",
        Integer,
        ForeignKey("transcription_job.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("source_platform", String(10)),
    Column("source_video_id", String(255)),
    Column("source_url", String(2048)),
    Column("source_title", Text),
    Column("source_description", Text),
    Column("source_channel", Text),
    Column("source_duration_seconds", Integer),
    Column("transcript_method", String(20)),
    Column("transcript_language", String(35)),
    Column("transcript_text", Text),
    CheckConstraint(
        "source_platform IS NULL OR source_platform IN "
        "('youtube', 'instagram', 'facebook', 'tiktok', 'x')",
        name="transcription_job_result_platform_check",
    ),
    CheckConstraint(
        "transcript_method IS NULL OR transcript_method IN "
        "('youtube_captions', 'faster_whisper')",
        name="transcription_job_result_method_check",
    ),
    CheckConstraint(
        "source_duration_seconds IS NULL OR source_duration_seconds > 0",
        name="transcription_job_result_duration_check",
    ),
    CheckConstraint(
        "source_video_id IS NULL OR length(source_video_id) BETWEEN 1 AND 255",
        name="transcription_job_result_video_id_length_check",
    ),
    CheckConstraint(
        "source_url IS NULL OR length(source_url) BETWEEN 1 AND 2048",
        name="transcription_job_result_source_url_length_check",
    ),
    CheckConstraint(
        "transcript_language IS NULL OR length(transcript_language) BETWEEN 1 AND 35",
        name="transcription_job_result_language_length_check",
    ),
)

transcription_job_segment = Table(
    "transcription_job_segment",
    metadata,
    Column(
        "job_id",
        Integer,
        ForeignKey("transcription_job_result.job_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("start_seconds", Float, nullable=False),
    Column("end_seconds", Float, nullable=False),
    Column("text", Text, nullable=False),
    PrimaryKeyConstraint("job_id", "ordinal", name="transcription_job_segment_pkey"),
    CheckConstraint(
        "ordinal >= 0 AND start_seconds >= 0 AND end_seconds >= start_seconds "
        "AND length(text) > 0",
        name="transcription_job_segment_values_check",
    ),
)


class TranscriptionJobCapacityError(RuntimeError):
    """Indicate durable job admission has reached its configured maximum."""


class TranscriptionJobStoreUnavailableError(RuntimeError):
    """Indicate SQLite could not safely persist or read a Transcription Job."""


class SqliteTranscriptionJobRepository:
    """Persist safe queued-job snapshots through one SQLite write boundary."""

    def __init__(
        self,
        engine: AsyncEngine,
        maximum_outstanding_jobs: int,
        queue_timeout: timedelta,
        clock: Callable[[], datetime],
    ) -> None:
        """Initialize SQLite storage policy.

        Args:
            engine: Verified application-owned asynchronous SQLite engine.
            maximum_outstanding_jobs: Maximum queued and processing jobs allowed.
            queue_timeout: Absolute queued lifetime assigned during admission.
            clock: UTC clock read only after the admission lock is held.

        Raises:
            ValueError: If the configured capacity or queue timeout is not positive.
        """
        if maximum_outstanding_jobs <= 0:
            raise ValueError("maximum_outstanding_jobs must be positive.")
        if queue_timeout <= timedelta():
            raise ValueError("queue_timeout must be positive.")
        self._engine = engine
        self._maximum_outstanding_jobs = maximum_outstanding_jobs
        self._queue_timeout = queue_timeout
        self._clock = clock

    async def create_queued(
        self,
        job: NewQueuedTranscriptionJob,
    ) -> QueuedTranscriptionJob:
        """Atomically admit and persist one queued Transcription Job.

        Args:
            job: Validated private source URL and sorted immutable exclusions.

        Returns:
            Safe committed queued-job snapshot without private submission fields.

        Raises:
            TranscriptionJobCapacityError: If outstanding jobs already meet capacity.
            TranscriptionJobStoreUnavailableError: If SQLite cannot commit the job.
        """
        try:
            async with self._engine.connect() as connection:
                try:
                    await connection.execute(text("BEGIN IMMEDIATE"))
                    outstanding_jobs = await self._count_outstanding_jobs(connection)
                    if outstanding_jobs >= self._maximum_outstanding_jobs:
                        await connection.rollback()
                        raise TranscriptionJobCapacityError()

                    submitted_at = _as_utc(self._clock())
                    queue_deadline_at = submitted_at + self._queue_timeout
                    public_id = uuid4()
                    insert_result = await connection.execute(
                        insert(transcription_job).values(
                            public_id=str(public_id),
                            submitted_url=job.submitted_url,
                            status="queued",
                            submitted_at=submitted_at,
                            queue_deadline_at=queue_deadline_at,
                            cancellation_requested=False,
                        )
                    )
                    internal_id = insert_result.lastrowid
                    if not isinstance(internal_id, int):
                        raise TranscriptionJobStoreUnavailableError()
                    if job.exclusions:
                        await connection.execute(
                            insert(transcription_job_exclusion),
                            [
                                {"job_id": internal_id, "field_path": field_path}
                                for field_path in job.exclusions
                            ],
                        )
                    await connection.commit()
                except TranscriptionJobCapacityError:
                    raise
                except TranscriptionJobStoreUnavailableError:
                    await connection.rollback()
                    raise
                except SQLAlchemyError as exc:
                    try:
                        await connection.rollback()
                    except SQLAlchemyError as rollback_exc:
                        raise TranscriptionJobStoreUnavailableError() from rollback_exc
                    raise TranscriptionJobStoreUnavailableError() from exc
        except TranscriptionJobCapacityError:
            raise
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

        return _queued_job(
            internal_id=internal_id,
            public_id=public_id,
            submitted_at=submitted_at,
            queue_deadline_at=queue_deadline_at,
        )

    async def get_queued(self, public_id: UUID) -> QueuedTranscriptionJob | None:
        """Read one queued job by its bearer capability.

        Args:
            public_id: Canonical UUIDv4 capability supplied by the coordinator.

        Returns:
            Safe queued-job snapshot, or ``None`` when no such job exists.

        Raises:
            TranscriptionJobStoreUnavailableError: If SQLite cannot read safely.
        """
        statement = select(
            transcription_job.c.id,
            transcription_job.c.public_id,
            transcription_job.c.status,
            transcription_job.c.submitted_at,
            transcription_job.c.queue_deadline_at,
        ).where(transcription_job.c.public_id == str(public_id))
        try:
            async with self._engine.connect() as connection:
                row = (await connection.execute(statement)).mappings().one_or_none()
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

        if row is None:
            return None
        return _queued_job_from_row(row)

    async def _count_outstanding_jobs(self, connection: AsyncConnection) -> int:
        """Count queued and processing jobs while the writer lock is held."""
        count = await connection.scalar(
            select(func.count())
            .select_from(transcription_job)
            .where(transcription_job.c.status.in_(("queued", "processing")))
        )
        if not isinstance(count, int):
            raise TranscriptionJobStoreUnavailableError()
        return count


def _queued_job_from_row(row: RowMapping) -> QueuedTranscriptionJob:
    """Translate a private SQL row into a safe queued-job snapshot."""
    internal_id = row["id"]
    public_id = row["public_id"]
    status = row["status"]
    submitted_at = row["submitted_at"]
    queue_deadline_at = row["queue_deadline_at"]
    if (
        not isinstance(internal_id, int)
        or not isinstance(public_id, str)
        or status != "queued"
        or not isinstance(submitted_at, datetime)
        or not isinstance(queue_deadline_at, datetime)
    ):
        raise TranscriptionJobStoreUnavailableError()
    try:
        parsed_public_id = UUID(public_id)
    except ValueError as exc:
        raise TranscriptionJobStoreUnavailableError() from exc
    if parsed_public_id.version != 4 or str(parsed_public_id) != public_id:
        raise TranscriptionJobStoreUnavailableError()
    return _queued_job(
        internal_id=internal_id,
        public_id=parsed_public_id,
        submitted_at=_as_utc(submitted_at),
        queue_deadline_at=_as_utc(queue_deadline_at),
    )


def _queued_job(
    *,
    internal_id: int,
    public_id: UUID,
    submitted_at: datetime,
    queue_deadline_at: datetime,
) -> QueuedTranscriptionJob:
    """Construct a queued-job domain snapshot without loading private fields."""
    from textify.transcription.jobs import JobStatus, QueuedTranscriptionJob

    return QueuedTranscriptionJob(
        internal_id=internal_id,
        public_id=public_id,
        status=JobStatus.QUEUED,
        submitted_at=submitted_at,
        queue_deadline_at=queue_deadline_at,
    )


def _as_utc(value: datetime) -> datetime:
    """Normalize SQLAlchemy SQLite datetime values to aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
