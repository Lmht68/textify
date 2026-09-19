"""SQLite persistence for durable Transcription Jobs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

from pydantic import TypeAdapter, ValidationError
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
    update,
)
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from textify.transcription.exceptions import QueueTimeoutError
from textify.transcription.schemas import ErrorDetail, TranscriptionResponse
from textify.transcription.types import ResponseFieldPath

if TYPE_CHECKING:
    from textify.transcription.jobs import (
        ClaimedTranscriptionJob,
        FailedTranscriptionJob,
        NewQueuedTranscriptionJob,
        ProcessingTranscriptionJob,
        QueuedTranscriptionJob,
        SucceededTranscriptionJob,
        TranscriptionJob,
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

_RESPONSE_ADAPTER: TypeAdapter[TranscriptionResponse] = TypeAdapter(
    TranscriptionResponse
)
_RESPONSE_FIELD_PATHS = frozenset(
    {
        "source.platform",
        "source.video_id",
        "source.url",
        "source.title",
        "source.description",
        "source.channel",
        "source.duration_seconds",
        "transcript.method",
        "transcript.language",
        "transcript.text",
        "transcript.segments",
    }
)
_SOURCE_RESULT_COLUMNS = (
    ("source.platform", "platform", "source_platform"),
    ("source.video_id", "video_id", "source_video_id"),
    ("source.url", "url", "source_url"),
    ("source.title", "title", "source_title"),
    ("source.description", "description", "source_description"),
    ("source.channel", "channel", "source_channel"),
    ("source.duration_seconds", "duration_seconds", "source_duration_seconds"),
)
_TRANSCRIPT_RESULT_COLUMNS = (
    ("transcript.method", "method", "transcript_method"),
    ("transcript.language", "language", "transcript_language"),
    ("transcript.text", "text", "transcript_text"),
)


class TranscriptionJobCapacityError(RuntimeError):
    """Indicate durable job admission has reached its configured maximum."""


class TranscriptionJobStoreUnavailableError(RuntimeError):
    """Indicate SQLite could not safely persist or read a Transcription Job."""


class SqliteTranscriptionJobRepository:
    """Persist Transcription Job lifecycle snapshots through one SQLite boundary."""

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
            clock: UTC clock read only after a writer lock is held.

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

    async def claim_next(self) -> ClaimedTranscriptionJob | None:
        """Claim the next eligible queued job without executing provider work.

        Returns:
            Private committed execution inputs, or ``None`` when none is eligible.

        Raises:
            TranscriptionJobStoreUnavailableError: If SQLite cannot claim safely.
        """
        try:
            async with self._engine.connect() as connection:
                try:
                    await connection.execute(text("BEGIN IMMEDIATE"))
                    now = _as_utc(self._clock())
                    await connection.execute(
                        update(transcription_job)
                        .where(
                            transcription_job.c.status == "queued",
                            transcription_job.c.queue_deadline_at <= now,
                        )
                        .values(
                            status="finished",
                            outcome="failed",
                            finished_at=now,
                            error_code=QueueTimeoutError.code,
                            error_message=QueueTimeoutError.message,
                        )
                    )
                    candidate = (
                        (
                            await connection.execute(
                                select(
                                    transcription_job.c.id,
                                    transcription_job.c.submitted_url,
                                )
                                .where(
                                    transcription_job.c.status == "queued",
                                    transcription_job.c.queue_deadline_at > now,
                                )
                                .order_by(
                                    transcription_job.c.submitted_at,
                                    transcription_job.c.id,
                                )
                                .limit(1)
                            )
                        )
                        .mappings()
                        .one_or_none()
                    )
                    if candidate is None:
                        await connection.commit()
                        return None

                    internal_id, submitted_url = _claimed_input_from_row(candidate)
                    exclusions = await self._load_exclusions(connection, internal_id)
                    transition = await connection.execute(
                        update(transcription_job)
                        .where(
                            transcription_job.c.id == internal_id,
                            transcription_job.c.status == "queued",
                        )
                        .values(status="processing", started_at=now)
                    )
                    if transition.rowcount != 1:
                        raise TranscriptionJobStoreUnavailableError()
                    await connection.commit()
                except TranscriptionJobStoreUnavailableError:
                    await connection.rollback()
                    raise
                except SQLAlchemyError as exc:
                    try:
                        await connection.rollback()
                    except SQLAlchemyError as rollback_exc:
                        raise TranscriptionJobStoreUnavailableError() from rollback_exc
                    raise TranscriptionJobStoreUnavailableError() from exc
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

        return _claimed_job(internal_id, submitted_url, exclusions)

    async def publish_success(
        self,
        internal_id: int,
        projected_result: TranscriptionResponse,
    ) -> bool:
        """Atomically persist one projected result and succeeded terminal transition.

        Args:
            internal_id: Private identifier held only by the claiming consumer.
            projected_result: Fully projected and validated public result.

        Returns:
            ``True`` when this consumer committed the successful terminal outcome,
            otherwise ``False`` when another terminal transition owns the job.

        Raises:
            TranscriptionJobStoreUnavailableError: If SQLite cannot publish safely.
        """
        validated_result = _validate_response(projected_result)
        result_values = _result_values(internal_id, validated_result)
        segment_values = _segment_values(internal_id, validated_result)
        try:
            async with self._engine.connect() as connection:
                try:
                    await connection.execute(text("BEGIN IMMEDIATE"))
                    await connection.execute(
                        insert(transcription_job_result).values(result_values)
                    )
                    if segment_values:
                        await connection.execute(
                            insert(transcription_job_segment), segment_values
                        )
                    finished_at = _as_utc(self._clock())
                    transition = await connection.execute(
                        update(transcription_job)
                        .where(
                            transcription_job.c.id == internal_id,
                            transcription_job.c.status == "processing",
                            transcription_job.c.cancellation_requested.is_(False),
                        )
                        .values(
                            status="finished",
                            outcome="succeeded",
                            finished_at=finished_at,
                        )
                    )
                    if transition.rowcount != 1:
                        await connection.rollback()
                        return False
                    await connection.commit()
                except TranscriptionJobStoreUnavailableError:
                    await connection.rollback()
                    raise
                except SQLAlchemyError as exc:
                    try:
                        await connection.rollback()
                    except SQLAlchemyError as rollback_exc:
                        raise TranscriptionJobStoreUnavailableError() from rollback_exc
                    raise TranscriptionJobStoreUnavailableError() from exc
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc
        return True

    async def publish_failure(
        self,
        internal_id: int,
        error_code: str,
        error_message: str,
    ) -> bool:
        """Atomically persist one safe failed terminal transition.

        Args:
            internal_id: Private identifier held only by the claiming consumer.
            error_code: Validated stable public error code.
            error_message: Validated safe nonempty public error message.

        Returns:
            ``True`` when this consumer committed the failed terminal outcome,
            otherwise ``False`` when another terminal transition owns the job.

        Raises:
            TranscriptionJobStoreUnavailableError: If validation or SQLite
                publication cannot complete safely.
        """
        error = _validate_error_detail(error_code, error_message)
        try:
            async with self._engine.connect() as connection:
                try:
                    await connection.execute(text("BEGIN IMMEDIATE"))
                    finished_at = _as_utc(self._clock())
                    transition = await connection.execute(
                        update(transcription_job)
                        .where(
                            transcription_job.c.id == internal_id,
                            transcription_job.c.status == "processing",
                            transcription_job.c.cancellation_requested.is_(False),
                        )
                        .values(
                            status="finished",
                            outcome="failed",
                            finished_at=finished_at,
                            error_code=error.code,
                            error_message=error.message,
                        )
                    )
                    if transition.rowcount != 1:
                        await connection.rollback()
                        return False
                    await connection.commit()
                except TranscriptionJobStoreUnavailableError:
                    await connection.rollback()
                    raise
                except SQLAlchemyError as exc:
                    try:
                        await connection.rollback()
                    except SQLAlchemyError as rollback_exc:
                        raise TranscriptionJobStoreUnavailableError() from rollback_exc
                    raise TranscriptionJobStoreUnavailableError() from exc
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc
        return True

    async def get_job(self, public_id: UUID) -> TranscriptionJob | None:
        """Read one safe Transcription Job snapshot by bearer capability.

        Args:
            public_id: Canonical UUIDv4 capability supplied by the coordinator.

        Returns:
            Safe queued, processing, successful, or failed snapshot, or ``None`` when
            unknown.
        Raises:
            TranscriptionJobStoreUnavailableError: If SQLite cannot read safely.
        """
        try:
            async with self._engine.connect() as connection:
                job_row = (
                    (
                        await connection.execute(
                            select(transcription_job).where(
                                transcription_job.c.public_id == str(public_id)
                            )
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if job_row is None:
                    return None

                status = job_row["status"]
                if status == "queued":
                    return _queued_job_from_row(job_row)
                if status == "processing":
                    return _processing_job_from_row(job_row)
                if status != "finished":
                    raise TranscriptionJobStoreUnavailableError()
                if job_row["outcome"] == "failed":
                    return _failed_job_from_row(job_row)
                if job_row["outcome"] != "succeeded":
                    raise TranscriptionJobStoreUnavailableError()
                internal_id = _internal_id_from_row(job_row)
                exclusions = await self._load_exclusions(connection, internal_id)
                result_row = (
                    (
                        await connection.execute(
                            select(transcription_job_result).where(
                                transcription_job_result.c.job_id == internal_id
                            )
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if result_row is None:
                    raise TranscriptionJobStoreUnavailableError()
                segment_rows = (
                    (
                        await connection.execute(
                            select(transcription_job_segment)
                            .where(transcription_job_segment.c.job_id == internal_id)
                            .order_by(transcription_job_segment.c.ordinal)
                        )
                    )
                    .mappings()
                    .all()
                )
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

        return _succeeded_job_from_rows(job_row, result_row, segment_rows, exclusions)

    async def _load_exclusions(
        self,
        connection: AsyncConnection,
        internal_id: int,
    ) -> tuple[ResponseFieldPath, ...]:
        """Load and validate a job's immutable response projection."""
        rows = (
            (
                await connection.execute(
                    select(transcription_job_exclusion.c.field_path)
                    .where(transcription_job_exclusion.c.job_id == internal_id)
                    .order_by(transcription_job_exclusion.c.field_path)
                )
            )
            .mappings()
            .all()
        )
        return _exclusions_from_rows(rows)

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


def _claimed_input_from_row(row: RowMapping) -> tuple[int, str]:
    """Validate private claim inputs selected under the writer lock."""
    internal_id = _internal_id_from_row(row)
    submitted_url = row["submitted_url"]
    if not isinstance(submitted_url, str) or not submitted_url:
        raise TranscriptionJobStoreUnavailableError()
    return internal_id, submitted_url


def _claimed_job(
    internal_id: int,
    submitted_url: str,
    exclusions: tuple[ResponseFieldPath, ...],
) -> ClaimedTranscriptionJob:
    """Construct a private committed claim snapshot."""
    from textify.transcription.jobs import ClaimedTranscriptionJob

    return ClaimedTranscriptionJob(
        internal_id=internal_id,
        submitted_url=submitted_url,
        exclusions=exclusions,
    )


def _queued_job_from_row(row: RowMapping) -> QueuedTranscriptionJob:
    """Translate a private SQL row into a safe queued-job snapshot."""
    if row["status"] != "queued":
        raise TranscriptionJobStoreUnavailableError()
    queue_deadline_at = _datetime_from_row(row, "queue_deadline_at")
    return _queued_job(
        internal_id=_internal_id_from_row(row),
        public_id=_public_id_from_row(row),
        submitted_at=_datetime_from_row(row, "submitted_at"),
        queue_deadline_at=queue_deadline_at,
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


def _processing_job_from_row(row: RowMapping) -> ProcessingTranscriptionJob:
    """Translate a private SQL row into a safe processing-job snapshot."""
    if row["status"] != "processing":
        raise TranscriptionJobStoreUnavailableError()
    cancellation_requested = row["cancellation_requested"]
    if not isinstance(cancellation_requested, bool):
        raise TranscriptionJobStoreUnavailableError()
    from textify.transcription.jobs import JobStatus, ProcessingTranscriptionJob

    return ProcessingTranscriptionJob(
        public_id=_public_id_from_row(row),
        status=JobStatus.PROCESSING,
        submitted_at=_datetime_from_row(row, "submitted_at"),
        started_at=_datetime_from_row(row, "started_at"),
        cancellation_requested=cancellation_requested,
    )


def _succeeded_job_from_rows(
    job_row: RowMapping,
    result_row: RowMapping,
    segment_rows: Sequence[RowMapping],
    exclusions: tuple[ResponseFieldPath, ...],
) -> SucceededTranscriptionJob:
    """Translate a terminal row and normalized result rows into one snapshot."""
    if (
        job_row["status"] != "finished"
        or job_row["outcome"] != "succeeded"
        or job_row["cancellation_requested"] is not False
    ):
        raise TranscriptionJobStoreUnavailableError()
    from textify.transcription.jobs import (
        JobOutcome,
        JobStatus,
        SucceededTranscriptionJob,
    )

    return SucceededTranscriptionJob(
        public_id=_public_id_from_row(job_row),
        status=JobStatus.FINISHED,
        outcome=JobOutcome.SUCCEEDED,
        submitted_at=_datetime_from_row(job_row, "submitted_at"),
        started_at=_datetime_from_row(job_row, "started_at"),
        finished_at=_datetime_from_row(job_row, "finished_at"),
        result=_response_from_result_rows(result_row, segment_rows, exclusions),
    )


def _failed_job_from_row(row: RowMapping) -> FailedTranscriptionJob:
    """Translate a failed terminal row without loading result data."""
    if (
        row["status"] != "finished"
        or row["outcome"] != "failed"
        or row["cancellation_requested"] is not False
    ):
        raise TranscriptionJobStoreUnavailableError()
    started_at_value = row["started_at"]
    started_at = (
        None if started_at_value is None else _datetime_from_row(row, "started_at")
    )
    error = _error_detail_from_row(row)
    from textify.transcription.jobs import (
        FailedTranscriptionJob,
        JobOutcome,
        JobStatus,
    )

    return FailedTranscriptionJob(
        public_id=_public_id_from_row(row),
        status=JobStatus.FINISHED,
        outcome=JobOutcome.FAILED,
        submitted_at=_datetime_from_row(row, "submitted_at"),
        started_at=started_at,
        finished_at=_datetime_from_row(row, "finished_at"),
        error_code=error.code,
        error_message=error.message,
    )


def _response_from_result_rows(
    result_row: RowMapping,
    segment_rows: Sequence[RowMapping],
    exclusions: tuple[ResponseFieldPath, ...],
) -> TranscriptionResponse:
    """Reconstruct and validate a projected response from normalized rows."""
    excluded_fields = frozenset(exclusions)
    source = _response_section(
        result_row,
        excluded_fields,
        _SOURCE_RESULT_COLUMNS,
    )
    transcript = _response_section(
        result_row,
        excluded_fields,
        _TRANSCRIPT_RESULT_COLUMNS,
    )
    if "transcript.segments" in excluded_fields:
        if segment_rows:
            raise TranscriptionJobStoreUnavailableError()
    else:
        transcript["segments"] = _segment_response_values(segment_rows)
    return _validate_response({"source": source, "transcript": transcript})


def _response_section(
    result_row: RowMapping,
    excluded_fields: frozenset[ResponseFieldPath],
    columns: Sequence[tuple[str, str, str]],
) -> dict[str, object]:
    """Load one response object and enforce its null-projection invariants."""
    section: dict[str, object] = {}
    for field_path, response_key, result_column in columns:
        value = result_row[result_column]
        if field_path in excluded_fields:
            if value is not None:
                raise TranscriptionJobStoreUnavailableError()
            continue
        if value is None:
            raise TranscriptionJobStoreUnavailableError()
        section[response_key] = value
    return section


def _segment_response_values(
    segment_rows: Sequence[RowMapping],
) -> tuple[dict[str, object], ...]:
    """Return contiguous ordered segment response values from normalized rows."""
    segments: list[dict[str, object]] = []
    for ordinal, row in enumerate(segment_rows):
        if row["ordinal"] != ordinal:
            raise TranscriptionJobStoreUnavailableError()
        segments.append(
            {
                "start": row["start_seconds"],
                "end": row["end_seconds"],
                "text": row["text"],
            }
        )
    return tuple(segments)


def _result_values(
    internal_id: int,
    result: TranscriptionResponse,
) -> dict[str, object]:
    """Translate a validated projection into nullable normalized result scalars."""
    source = result["source"]
    transcript = result["transcript"]
    return {
        "job_id": internal_id,
        "source_platform": source.get("platform"),
        "source_video_id": source.get("video_id"),
        "source_url": source.get("url"),
        "source_title": source.get("title"),
        "source_description": source.get("description"),
        "source_channel": source.get("channel"),
        "source_duration_seconds": source.get("duration_seconds"),
        "transcript_method": transcript.get("method"),
        "transcript_language": transcript.get("language"),
        "transcript_text": transcript.get("text"),
    }


def _segment_values(
    internal_id: int,
    result: TranscriptionResponse,
) -> list[dict[str, object]]:
    """Translate retained response segments into ordered normalized rows."""
    segments = result["transcript"].get("segments")
    if segments is None:
        return []
    return [
        {
            "job_id": internal_id,
            "ordinal": ordinal,
            "start_seconds": segment["start"],
            "end_seconds": segment["end"],
            "text": segment["text"],
        }
        for ordinal, segment in enumerate(segments)
    ]


def _exclusions_from_rows(
    rows: Sequence[RowMapping],
) -> tuple[ResponseFieldPath, ...]:
    """Validate stored immutable exclusions against the application vocabulary."""
    exclusions: list[ResponseFieldPath] = []
    for row in rows:
        field_path = row["field_path"]
        if not isinstance(field_path, str) or field_path not in _RESPONSE_FIELD_PATHS:
            raise TranscriptionJobStoreUnavailableError()
        exclusions.append(cast(ResponseFieldPath, field_path))
    if len(set(exclusions)) != len(exclusions):
        raise TranscriptionJobStoreUnavailableError()
    return tuple(exclusions)


def _validate_response(value: object) -> TranscriptionResponse:
    """Validate one projected response before write and after reconstruction."""
    try:
        return _RESPONSE_ADAPTER.validate_python(value)
    except (TypeError, ValidationError, ValueError) as exc:
        raise TranscriptionJobStoreUnavailableError() from exc


def _error_detail_from_row(row: RowMapping) -> ErrorDetail:
    """Validate a persisted safe public error pair."""
    return _validate_error_detail(row["error_code"], row["error_message"])


def _validate_error_detail(error_code: object, error_message: object) -> ErrorDetail:
    """Validate one safe public error pair before writing or after reading."""
    try:
        return ErrorDetail.model_validate(
            {"code": error_code, "message": error_message}
        )
    except (TypeError, ValidationError, ValueError) as exc:
        raise TranscriptionJobStoreUnavailableError() from exc


def _internal_id_from_row(row: RowMapping) -> int:
    """Return a validated private database identifier from one job row."""
    internal_id = row["id"]
    if isinstance(internal_id, bool) or not isinstance(internal_id, int):
        raise TranscriptionJobStoreUnavailableError()
    return internal_id


def _public_id_from_row(row: RowMapping) -> UUID:
    """Return one canonical UUIDv4 bearer capability from a job row."""
    public_id = row["public_id"]
    if not isinstance(public_id, str):
        raise TranscriptionJobStoreUnavailableError()
    try:
        parsed_public_id = UUID(public_id)
    except ValueError as exc:
        raise TranscriptionJobStoreUnavailableError() from exc
    if parsed_public_id.version != 4 or str(parsed_public_id) != public_id:
        raise TranscriptionJobStoreUnavailableError()
    return parsed_public_id


def _datetime_from_row(row: RowMapping, column: str) -> datetime:
    """Read one required SQL timestamp as an aware UTC value."""
    value = row[column]
    if not isinstance(value, datetime):
        raise TranscriptionJobStoreUnavailableError()
    return _as_utc(value)


def _as_utc(value: datetime) -> datetime:
    """Normalize SQLAlchemy SQLite datetime values to aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
