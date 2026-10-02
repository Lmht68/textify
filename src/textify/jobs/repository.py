"""PostgreSQL persistence for durable Transcription Jobs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    delete,
    exists,
    func,
    insert,
    or_,
    select,
    text,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from textify.errors import ErrorDetail
from textify.jobs.contracts import (
    TranscriptionJobAlreadyFinishedError,
    TranscriptionJobCapacityError,
    TranscriptionJobStoreUnavailableError,
)
from textify.jobs.exceptions import QueueTimeoutError, WorkerInterruptedError
from textify.jobs.types import (
    CancelledTranscriptionJob,
    ClaimedTranscriptionJob,
    ClaimHeartbeat,
    ExecutionClaim,
    FailedTranscriptionJob,
    JobDispatch,
    JobOutcome,
    JobStatus,
    NewQueuedTranscriptionJob,
    ProcessingTranscriptionJob,
    QueuedTranscriptionJob,
    SucceededTranscriptionJob,
    TranscriptionJob,
)
from textify.transcription.schemas import TranscriptionResponse
from textify.transcription.types import ResponseFieldPath

_LIFECYCLE_ADVISORY_LOCK_KEY = -255_152_064_224_511_608

metadata = MetaData()

transcription_job = Table(
    "transcription_job",
    metadata,
    Column("id", BigInteger, Identity(always=False), primary_key=True),
    Column("public_id", PostgreSQLUUID(as_uuid=True), nullable=False),
    Column("submitted_url", String(2048), nullable=False),
    Column("status", String(10), nullable=False),
    Column("outcome", String(10)),
    Column("submitted_at", DateTime(timezone=True), nullable=False),
    Column("queue_deadline_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Column(
        "cancellation_requested", Boolean, nullable=False, server_default=text("false")
    ),
    Column("error_code", String(64)),
    Column("error_message", Text),
    UniqueConstraint("public_id", name="transcription_job_public_id_key"),
    CheckConstraint(
        "(get_byte(uuid_send(public_id), 6) & 240) = 64 "
        "AND (get_byte(uuid_send(public_id), 8) & 192) = 128",
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
        "AND (finished_at IS NULL OR finished_at >= "
        "COALESCE(started_at, submitted_at))",
        name="transcription_job_timestamp_order_check",
    ),
    CheckConstraint(
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
    CheckConstraint(
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
Index(
    "transcription_job_status_finished_at_idx",
    transcription_job.c.status,
    transcription_job.c.finished_at,
)

transcription_job_execution_attempt = Table(
    "transcription_job_execution_attempt",
    metadata,
    Column(
        "job_id",
        BigInteger,
        ForeignKey("transcription_job.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "execution_attempt_token",
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    ),
    Column("next_dispatch_at", DateTime(timezone=True), nullable=False),
    Column("publication_lease_owner", PostgreSQLUUID(as_uuid=True)),
    Column("publication_lease_expires_at", DateTime(timezone=True)),
    Column("last_dispatched_at", DateTime(timezone=True)),
    Column(
        "dispatch_count",
        BigInteger,
        nullable=False,
        server_default=text("0"),
    ),
    Column("worker_owner", PostgreSQLUUID(as_uuid=True)),
    Column("last_heartbeat_at", DateTime(timezone=True)),
    Column("claim_lease_expires_at", DateTime(timezone=True)),
    UniqueConstraint(
        "execution_attempt_token",
        name="transcription_job_execution_attempt_token_key",
    ),
    CheckConstraint(
        "(get_byte(uuid_send(execution_attempt_token), 6) & 240) = 64 "
        "AND (get_byte(uuid_send(execution_attempt_token), 8) & 192) = 128",
        name="transcription_job_execution_attempt_token_uuid4_check",
    ),
    CheckConstraint(
        "(publication_lease_owner IS NULL) = (publication_lease_expires_at IS NULL)",
        name="transcription_job_attempt_publication_lease_pair_check",
    ),
    CheckConstraint(
        "publication_lease_owner IS NULL OR "
        "((get_byte(uuid_send(publication_lease_owner), 6) & 240) = 64 "
        "AND (get_byte(uuid_send(publication_lease_owner), 8) & 192) = 128)",
        name="transcription_job_attempt_lease_owner_uuid4_check",
    ),
    CheckConstraint(
        "dispatch_count >= 0 AND "
        "((dispatch_count = 0 AND last_dispatched_at IS NULL) OR "
        "(dispatch_count > 0 AND last_dispatched_at IS NOT NULL))",
        name="transcription_job_attempt_dispatch_history_check",
    ),
    CheckConstraint(
        "((worker_owner IS NULL AND last_heartbeat_at IS NULL "
        "AND claim_lease_expires_at IS NULL) OR "
        "(worker_owner IS NOT NULL AND last_heartbeat_at IS NOT NULL "
        "AND claim_lease_expires_at IS NOT NULL))",
        name="transcription_job_attempt_claim_state_check",
    ),
    CheckConstraint(
        "worker_owner IS NULL OR "
        "((get_byte(uuid_send(worker_owner), 6) & 240) = 64 "
        "AND (get_byte(uuid_send(worker_owner), 8) & 192) = 128)",
        name="transcription_job_attempt_worker_owner_uuid4_check",
    ),
    CheckConstraint(
        "last_heartbeat_at <= claim_lease_expires_at",
        name="transcription_job_attempt_claim_lease_order_check",
    ),
)

Index(
    "transcription_job_attempt_next_dispatch_at_job_id_idx",
    transcription_job_execution_attempt.c.next_dispatch_at,
    transcription_job_execution_attempt.c.job_id,
)

Index(
    "transcription_job_attempt_claim_lease_expires_at_job_id_idx",
    transcription_job_execution_attempt.c.claim_lease_expires_at,
    transcription_job_execution_attempt.c.job_id,
    postgresql_where=transcription_job_execution_attempt.c.claim_lease_expires_at.is_not(
        None
    ),
)

transcription_job_exclusion = Table(
    "transcription_job_exclusion",
    metadata,
    Column(
        "job_id",
        BigInteger,
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
        BigInteger,
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
        BigInteger,
        ForeignKey("transcription_job_result.job_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("start_seconds", Float(precision=53), nullable=False),
    Column("end_seconds", Float(precision=53), nullable=False),
    Column("text", Text, nullable=False),
    PrimaryKeyConstraint("job_id", "ordinal", name="transcription_job_segment_pkey"),
    CheckConstraint(
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


class PostgresTranscriptionJobRepository:
    """Persist Transcription Job lifecycle snapshots through PostgreSQL."""

    def __init__(
        self,
        engine: AsyncEngine,
        maximum_outstanding_jobs: int,
        queue_timeout: timedelta,
        terminal_retention: timedelta,
        clock: Callable[[], datetime],
    ) -> None:
        """Initialize PostgreSQL storage policy.

        Args:
            engine: Verified application-owned asynchronous PostgreSQL engine.
            maximum_outstanding_jobs: Maximum queued and processing jobs allowed.
            queue_timeout: Absolute queued lifetime assigned during admission.
            terminal_retention: Post-completion capability lifetime before deletion.
            clock: UTC clock read only after the lifecycle lock is held.

        Raises:
            ValueError: If capacity, queue timeout, or terminal retention is nonpositive.
        """
        if maximum_outstanding_jobs <= 0:
            raise ValueError("maximum_outstanding_jobs must be positive.")
        if queue_timeout <= timedelta():
            raise ValueError("queue_timeout must be positive.")
        if terminal_retention <= timedelta():
            raise ValueError("terminal_retention must be positive.")
        self._engine = engine
        self._maximum_outstanding_jobs = maximum_outstanding_jobs
        self._queue_timeout = queue_timeout
        self._terminal_retention = terminal_retention
        self._clock = clock

    @asynccontextmanager
    async def _serialized_lifecycle_transaction(
        self,
    ) -> AsyncIterator[AsyncConnection]:
        """Yield one transaction serialized across PostgreSQL application sessions.

        Yields:
            Open transaction holding the Transcription Job lifecycle advisory lock.
        """
        async with self._engine.begin() as connection:
            await connection.execute(
                text("SELECT pg_advisory_xact_lock(:lock_key)"),
                {"lock_key": _LIFECYCLE_ADVISORY_LOCK_KEY},
            )
            yield connection

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
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot commit the job.
        """
        try:
            async with self._serialized_lifecycle_transaction() as connection:
                outstanding_jobs = await self._count_outstanding_jobs(connection)
                if outstanding_jobs >= self._maximum_outstanding_jobs:
                    raise TranscriptionJobCapacityError()

                submitted_at = _as_utc(self._clock())
                queue_deadline_at = submitted_at + self._queue_timeout
                public_id = uuid4()
                execution_attempt_token = uuid4()
                internal_id = await connection.scalar(
                    insert(transcription_job)
                    .values(
                        public_id=public_id,
                        submitted_url=job.submitted_url,
                        status="queued",
                        submitted_at=submitted_at,
                        queue_deadline_at=queue_deadline_at,
                        cancellation_requested=False,
                    )
                    .returning(transcription_job.c.id)
                )
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
                await connection.execute(
                    insert(transcription_job_execution_attempt).values(
                        job_id=internal_id,
                        execution_attempt_token=execution_attempt_token,
                        next_dispatch_at=submitted_at,
                        dispatch_count=0,
                    )
                )
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

    async def expire_queued_jobs(self, limit: int) -> int:
        """Fail a bounded batch of queued jobs that reached their deadline.

        Args:
            limit: Maximum queued jobs to transition in one transaction.

        Returns:
            Number of jobs transitioned to the queue-timeout outcome.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot expire jobs.
            ValueError: If ``limit`` is not positive.
        """
        _require_positive_limit(limit)
        try:
            async with self._engine.begin() as connection:
                now = _as_utc(self._clock())
                expired_rows = (
                    (
                        await connection.execute(
                            select(transcription_job.c.id)
                            .where(
                                transcription_job.c.status == "queued",
                                transcription_job.c.cancellation_requested.is_(False),
                                transcription_job.c.queue_deadline_at <= now,
                            )
                            .order_by(
                                transcription_job.c.queue_deadline_at,
                                transcription_job.c.id,
                            )
                            .limit(limit)
                            .with_for_update(skip_locked=True)
                        )
                    )
                    .mappings()
                    .all()
                )
                expired_job_ids = tuple(
                    _internal_id_from_row(row) for row in expired_rows
                )
                if not expired_job_ids:
                    return 0
                transition = await connection.execute(
                    update(transcription_job)
                    .where(
                        transcription_job.c.id.in_(expired_job_ids),
                        transcription_job.c.status == "queued",
                        transcription_job.c.cancellation_requested.is_(False),
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
                if (
                    transition.rowcount is None
                    or transition.rowcount < 0
                    or transition.rowcount > len(expired_job_ids)
                ):
                    raise TranscriptionJobStoreUnavailableError()
                return transition.rowcount
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

    async def lease_due_dispatches(
        self,
        lease_owner: UUID,
        lease_duration: timedelta,
        limit: int,
    ) -> tuple[JobDispatch, ...]:
        """Lease a bounded FIFO batch of due queued Job Dispatches.

        Args:
            lease_owner: Private UUIDv4 identity for this reconciler instance.
            lease_duration: Positive lease duration for unrecorded publication.
            limit: Maximum Execution Attempts to lease.

        Returns:
            Leased private dispatches after their transaction commits.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot lease safely.
            ValueError: If lease arguments are invalid.
        """
        _require_uuid4(lease_owner, "lease_owner")
        _require_positive_duration(lease_duration, "lease_duration")
        _require_positive_limit(limit)
        try:
            async with self._engine.begin() as connection:
                now = _as_utc(self._clock())
                lease_expires_at = now + lease_duration
                due_rows = (
                    (
                        await connection.execute(
                            select(
                                transcription_job.c.id,
                                transcription_job_execution_attempt.c.execution_attempt_token,
                            )
                            .join(
                                transcription_job_execution_attempt,
                                transcription_job.c.id
                                == transcription_job_execution_attempt.c.job_id,
                            )
                            .where(
                                transcription_job.c.status == "queued",
                                transcription_job.c.cancellation_requested.is_(False),
                                transcription_job.c.queue_deadline_at > now,
                                transcription_job_execution_attempt.c.next_dispatch_at
                                <= now,
                                or_(
                                    transcription_job_execution_attempt.c.publication_lease_owner.is_(
                                        None
                                    ),
                                    transcription_job_execution_attempt.c.publication_lease_expires_at
                                    <= now,
                                ),
                            )
                            .order_by(
                                transcription_job.c.submitted_at,
                                transcription_job.c.id,
                            )
                            .limit(limit)
                            .with_for_update(
                                of=transcription_job_execution_attempt,
                                skip_locked=True,
                            )
                        )
                    )
                    .mappings()
                    .all()
                )
                candidates = tuple(_dispatch_from_row(row) for row in due_rows)
                if not candidates:
                    return ()

                queued_parent_exists = exists(
                    select(transcription_job.c.id).where(
                        transcription_job.c.id
                        == transcription_job_execution_attempt.c.job_id,
                        transcription_job.c.status == "queued",
                        transcription_job.c.cancellation_requested.is_(False),
                        transcription_job.c.queue_deadline_at > now,
                    )
                )
                result = await connection.execute(
                    update(transcription_job_execution_attempt)
                    .where(
                        transcription_job_execution_attempt.c.job_id.in_(
                            tuple(dispatch.internal_job_id for dispatch in candidates)
                        ),
                        queued_parent_exists,
                    )
                    .values(
                        publication_lease_owner=lease_owner,
                        publication_lease_expires_at=lease_expires_at,
                    )
                    .returning(
                        transcription_job_execution_attempt.c.job_id.label("id"),
                        transcription_job_execution_attempt.c.execution_attempt_token,
                    )
                )
                leased_dispatches = tuple(
                    _dispatch_from_row(row) for row in result.mappings().all()
                )
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

        leased_dispatch_keys = {
            (dispatch.internal_job_id, dispatch.execution_attempt_token)
            for dispatch in leased_dispatches
        }
        if len(leased_dispatch_keys) != len(leased_dispatches):
            raise TranscriptionJobStoreUnavailableError()
        return tuple(
            dispatch
            for dispatch in candidates
            if (dispatch.internal_job_id, dispatch.execution_attempt_token)
            in leased_dispatch_keys
        )

    async def record_dispatch_published(
        self,
        dispatch: JobDispatch,
        lease_owner: UUID,
        redispatch_after: timedelta,
    ) -> bool:
        """Record a broker publication and release the matching lease.

        Args:
            dispatch: Private attempt identity published to the broker.
            lease_owner: UUIDv4 reconciler identity that owns the lease.
            redispatch_after: Positive delay before another publication is due.

        Returns:
            ``True`` when the matching lease was recorded, otherwise ``False``.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot record safely.
            ValueError: If the lease owner or redispatch interval is invalid.
        """
        _require_uuid4(lease_owner, "lease_owner")
        _require_positive_duration(redispatch_after, "redispatch_after")
        try:
            async with self._engine.begin() as connection:
                dispatched_at = _as_utc(self._clock())
                result = await connection.execute(
                    update(transcription_job_execution_attempt)
                    .where(
                        transcription_job_execution_attempt.c.job_id
                        == dispatch.internal_job_id,
                        transcription_job_execution_attempt.c.execution_attempt_token
                        == dispatch.execution_attempt_token,
                        transcription_job_execution_attempt.c.publication_lease_owner
                        == lease_owner,
                    )
                    .values(
                        next_dispatch_at=dispatched_at + redispatch_after,
                        publication_lease_owner=None,
                        publication_lease_expires_at=None,
                        last_dispatched_at=dispatched_at,
                        dispatch_count=(
                            transcription_job_execution_attempt.c.dispatch_count + 1
                        ),
                    )
                    .returning(transcription_job_execution_attempt.c.job_id.label("id"))
                )
                row = result.mappings().one_or_none()
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

        if row is None:
            return False
        return _internal_id_from_row(row) == dispatch.internal_job_id

    async def release_dispatch_leases(
        self,
        dispatches: tuple[JobDispatch, ...],
        lease_owner: UUID,
    ) -> None:
        """Release this reconciler's bounded batch of publication leases.

        Args:
            dispatches: Private attempts leased by this reconciler.
            lease_owner: UUIDv4 identity required to release each lease.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot release safely.
            ValueError: If the lease owner is invalid.
        """
        _require_uuid4(lease_owner, "lease_owner")
        if not dispatches:
            return
        dispatch_keys = tuple(
            (dispatch.internal_job_id, dispatch.execution_attempt_token)
            for dispatch in dispatches
        )
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    update(transcription_job_execution_attempt)
                    .where(
                        transcription_job_execution_attempt.c.publication_lease_owner
                        == lease_owner,
                        tuple_(
                            transcription_job_execution_attempt.c.job_id,
                            transcription_job_execution_attempt.c.execution_attempt_token,
                        ).in_(dispatch_keys),
                    )
                    .values(
                        publication_lease_owner=None,
                        publication_lease_expires_at=None,
                    )
                )
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

    async def claim_dispatch(
        self,
        dispatch: JobDispatch,
        worker_owner: UUID,
        lease_duration: timedelta,
    ) -> ClaimedTranscriptionJob | None:
        """Claim one delivered Job Dispatch before loading private execution inputs.

        Args:
            dispatch: Private identifier and token supplied by the broker.
            worker_owner: UUIDv4 identity for the worker process claiming the attempt.
            lease_duration: Positive duration before an unrenewed claim expires.

        Returns:
            Private execution inputs for one committed claim, or ``None`` when stale.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot claim safely.
            ValueError: If the worker owner or lease duration is invalid.
        """
        _require_uuid4(dispatch.execution_attempt_token, "execution_attempt_token")
        _require_uuid4(worker_owner, "worker_owner")
        _require_positive_duration(lease_duration, "lease_duration")
        claim = ExecutionClaim(dispatch=dispatch, worker_owner=worker_owner)
        attempt_matches_dispatch = exists(
            select(transcription_job_execution_attempt.c.job_id).where(
                transcription_job_execution_attempt.c.job_id
                == dispatch.internal_job_id,
                transcription_job_execution_attempt.c.execution_attempt_token
                == dispatch.execution_attempt_token,
            )
        )
        try:
            async with self._serialized_lifecycle_transaction() as connection:
                now = _as_utc(self._clock())
                claim_lease_expires_at = now + lease_duration
                timeout_transition = await connection.execute(
                    update(transcription_job)
                    .where(
                        transcription_job.c.id == dispatch.internal_job_id,
                        transcription_job.c.status == "queued",
                        transcription_job.c.cancellation_requested.is_(False),
                        transcription_job.c.queue_deadline_at <= now,
                        attempt_matches_dispatch,
                    )
                    .values(
                        status="finished",
                        outcome="failed",
                        finished_at=now,
                        error_code=QueueTimeoutError.code,
                        error_message=QueueTimeoutError.message,
                    )
                )
                if timeout_transition.rowcount not in {0, 1}:
                    raise TranscriptionJobStoreUnavailableError()
                if timeout_transition.rowcount == 1:
                    return None

                oldest_eligible_job_id = await connection.scalar(
                    select(transcription_job.c.id)
                    .where(
                        transcription_job.c.status == "queued",
                        transcription_job.c.cancellation_requested.is_(False),
                        transcription_job.c.queue_deadline_at > now,
                    )
                    .order_by(
                        transcription_job.c.submitted_at,
                        transcription_job.c.id,
                    )
                    .limit(1)
                )
                if oldest_eligible_job_id is None:
                    return None
                if isinstance(oldest_eligible_job_id, bool) or not isinstance(
                    oldest_eligible_job_id, int
                ):
                    raise TranscriptionJobStoreUnavailableError()
                if oldest_eligible_job_id != dispatch.internal_job_id:
                    return None

                claim_transition = await connection.execute(
                    update(transcription_job)
                    .where(
                        transcription_job.c.id == dispatch.internal_job_id,
                        transcription_job.c.status == "queued",
                        transcription_job.c.cancellation_requested.is_(False),
                        transcription_job.c.queue_deadline_at > now,
                        attempt_matches_dispatch,
                    )
                    .values(status="processing", started_at=now)
                )
                if claim_transition.rowcount == 0:
                    return None
                if claim_transition.rowcount != 1:
                    raise TranscriptionJobStoreUnavailableError()

                ownership_transition = await connection.execute(
                    update(transcription_job_execution_attempt)
                    .where(
                        transcription_job_execution_attempt.c.job_id
                        == dispatch.internal_job_id,
                        transcription_job_execution_attempt.c.execution_attempt_token
                        == dispatch.execution_attempt_token,
                        transcription_job_execution_attempt.c.worker_owner.is_(None),
                    )
                    .values(
                        worker_owner=worker_owner,
                        last_heartbeat_at=now,
                        claim_lease_expires_at=claim_lease_expires_at,
                    )
                )
                if ownership_transition.rowcount != 1:
                    raise TranscriptionJobStoreUnavailableError()

                claimed_row = (
                    (
                        await connection.execute(
                            select(
                                transcription_job.c.id,
                                transcription_job.c.submitted_url,
                            )
                            .join(
                                transcription_job_execution_attempt,
                                transcription_job.c.id
                                == transcription_job_execution_attempt.c.job_id,
                            )
                            .where(
                                transcription_job.c.id == dispatch.internal_job_id,
                                transcription_job_execution_attempt.c.execution_attempt_token
                                == dispatch.execution_attempt_token,
                                transcription_job_execution_attempt.c.worker_owner
                                == worker_owner,
                                transcription_job.c.status == "processing",
                            )
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if claimed_row is None:
                    raise TranscriptionJobStoreUnavailableError()
                internal_id, submitted_url = _claimed_input_from_row(claimed_row)
                if internal_id != dispatch.internal_job_id:
                    raise TranscriptionJobStoreUnavailableError()
                exclusions = await self._load_exclusions(connection, internal_id)
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc
        return _claimed_job(claim, submitted_url, exclusions)

    async def heartbeat_claims(
        self,
        claims: tuple[ExecutionClaim, ...],
        lease_duration: timedelta,
    ) -> tuple[ClaimHeartbeat, ...]:
        """Renew matching claims and return their authoritative Cancellation state.

        Args:
            claims: Bounded active worker claims to renew.
            lease_duration: Positive duration before an unrenewed claim expires.

        Returns:
            Renewed claims only, each with durable Cancellation state.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot renew safely.
            ValueError: If a claim owner, token, or lease duration is invalid.
        """
        _require_positive_duration(lease_duration, "lease_duration")
        if not claims:
            return ()
        for claim in claims:
            _require_uuid4(
                claim.dispatch.execution_attempt_token,
                "execution_attempt_token",
            )
            _require_uuid4(claim.worker_owner, "worker_owner")

        claim_keys = tuple(
            (
                claim.dispatch.internal_job_id,
                claim.dispatch.execution_attempt_token,
                claim.worker_owner,
            )
            for claim in claims
        )
        claims_by_key = {
            (
                claim.dispatch.internal_job_id,
                claim.dispatch.execution_attempt_token,
                claim.worker_owner,
            ): claim
            for claim in claims
        }
        try:
            async with self._engine.begin() as connection:
                now = _as_utc(self._clock())
                result = await connection.execute(
                    update(transcription_job_execution_attempt)
                    .where(
                        transcription_job.c.id
                        == transcription_job_execution_attempt.c.job_id,
                        transcription_job.c.status == "processing",
                        transcription_job_execution_attempt.c.claim_lease_expires_at
                        > now,
                        tuple_(
                            transcription_job_execution_attempt.c.job_id,
                            transcription_job_execution_attempt.c.execution_attempt_token,
                            transcription_job_execution_attempt.c.worker_owner,
                        ).in_(claim_keys),
                    )
                    .values(
                        last_heartbeat_at=now,
                        claim_lease_expires_at=now + lease_duration,
                    )
                    .returning(
                        transcription_job_execution_attempt.c.job_id.label("id"),
                        transcription_job_execution_attempt.c.execution_attempt_token,
                        transcription_job_execution_attempt.c.worker_owner,
                        transcription_job.c.cancellation_requested,
                    )
                )
                heartbeat_rows = result.mappings().all()
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

        renewed_cancellations: dict[ExecutionClaim, bool] = {}
        for row in heartbeat_rows:
            dispatch = _dispatch_from_row(row)
            worker_owner = row["worker_owner"]
            cancellation_requested = row["cancellation_requested"]
            if (
                not isinstance(worker_owner, UUID)
                or worker_owner.version != 4
                or not isinstance(cancellation_requested, bool)
            ):
                raise TranscriptionJobStoreUnavailableError()
            claim_key = (
                dispatch.internal_job_id,
                dispatch.execution_attempt_token,
                worker_owner,
            )
            matched_claim = claims_by_key.get(claim_key)
            if matched_claim is None:
                raise TranscriptionJobStoreUnavailableError()
            renewed_cancellations[matched_claim] = cancellation_requested
        return tuple(
            ClaimHeartbeat(
                claim=claim,
                cancellation_requested=renewed_cancellations[claim],
            )
            for claim in claims
            if claim in renewed_cancellations
        )

    async def recover_expired_claims(self, limit: int) -> int:
        """Finish a bounded ordered batch of expired processing claims.

        Args:
            limit: Maximum expired Execution Attempts to terminalize.

        Returns:
            Number of claims finished as cancelled or worker-interrupted.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot recover safely.
            ValueError: If ``limit`` is not positive.
        """
        _require_positive_limit(limit)
        try:
            async with self._serialized_lifecycle_transaction() as connection:
                now = _as_utc(self._clock())
                expired_rows = (
                    (
                        await connection.execute(
                            select(
                                transcription_job.c.id,
                                transcription_job.c.cancellation_requested,
                            )
                            .join(
                                transcription_job_execution_attempt,
                                transcription_job.c.id
                                == transcription_job_execution_attempt.c.job_id,
                            )
                            .where(
                                transcription_job.c.status == "processing",
                                transcription_job_execution_attempt.c.claim_lease_expires_at
                                <= now,
                            )
                            .order_by(
                                transcription_job_execution_attempt.c.claim_lease_expires_at,
                                transcription_job.c.id,
                            )
                            .limit(limit)
                            .with_for_update(
                                of=transcription_job_execution_attempt,
                                skip_locked=True,
                            )
                        )
                    )
                    .mappings()
                    .all()
                )
                if not expired_rows:
                    return 0

                cancelled_job_ids: list[int] = []
                interrupted_job_ids: list[int] = []
                for row in expired_rows:
                    job_id = _internal_id_from_row(row)
                    cancellation_requested = row["cancellation_requested"]
                    if not isinstance(cancellation_requested, bool):
                        raise TranscriptionJobStoreUnavailableError()
                    if cancellation_requested:
                        cancelled_job_ids.append(job_id)
                    else:
                        interrupted_job_ids.append(job_id)

                transitioned_count = 0
                if cancelled_job_ids:
                    cancelled_transition = await connection.execute(
                        update(transcription_job)
                        .where(
                            transcription_job.c.id.in_(cancelled_job_ids),
                            transcription_job.c.status == "processing",
                            transcription_job.c.cancellation_requested.is_(True),
                        )
                        .values(
                            status="finished",
                            outcome="cancelled",
                            finished_at=now,
                        )
                    )
                    if cancelled_transition.rowcount != len(cancelled_job_ids):
                        raise TranscriptionJobStoreUnavailableError()
                    transitioned_count += cancelled_transition.rowcount
                if interrupted_job_ids:
                    interrupted_transition = await connection.execute(
                        update(transcription_job)
                        .where(
                            transcription_job.c.id.in_(interrupted_job_ids),
                            transcription_job.c.status == "processing",
                            transcription_job.c.cancellation_requested.is_(False),
                        )
                        .values(
                            status="finished",
                            outcome="failed",
                            finished_at=now,
                            error_code=WorkerInterruptedError.code,
                            error_message=WorkerInterruptedError.message,
                        )
                    )
                    if interrupted_transition.rowcount != len(interrupted_job_ids):
                        raise TranscriptionJobStoreUnavailableError()
                    transitioned_count += interrupted_transition.rowcount

                expired_job_ids = tuple(cancelled_job_ids + interrupted_job_ids)
                cleared_claims = await connection.execute(
                    update(transcription_job_execution_attempt)
                    .where(
                        transcription_job_execution_attempt.c.job_id.in_(
                            expired_job_ids
                        ),
                        transcription_job_execution_attempt.c.claim_lease_expires_at
                        <= now,
                    )
                    .values(
                        worker_owner=None,
                        last_heartbeat_at=None,
                        claim_lease_expires_at=None,
                    )
                )
                if cleared_claims.rowcount != transitioned_count:
                    raise TranscriptionJobStoreUnavailableError()
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc
        return transitioned_count

    async def delete_expired_terminal_jobs(self) -> None:
        """Delete terminal jobs whose finished time has reached retention cutoff.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot delete safely.
        """
        try:
            async with self._serialized_lifecycle_transaction() as connection:
                now = _as_utc(self._clock())
                cutoff = now - self._terminal_retention
                await connection.execute(
                    delete(transcription_job).where(
                        transcription_job.c.status == "finished",
                        transcription_job.c.finished_at <= cutoff,
                    )
                )
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

    async def request_cancellation(
        self,
        public_id: UUID,
    ) -> ProcessingTranscriptionJob | CancelledTranscriptionJob | None:
        """Durably accept one cancellation through its public capability.

        Args:
            public_id: Canonical UUIDv4 capability supplied by the coordinator.

        Returns:
            Processing or cancelled snapshot, or ``None`` when the capability is
            unknown.

        Raises:
            TranscriptionJobAlreadyFinishedError: If the job has a non-cancelled
                terminal outcome.
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot safely record
                the cancellation.
        """
        try:
            async with self._serialized_lifecycle_transaction() as connection:
                job_row = (
                    (
                        await connection.execute(
                            select(transcription_job).where(
                                transcription_job.c.public_id == public_id
                            )
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if job_row is None:
                    return None
                now = _as_utc(self._clock())
                cutoff = now - self._terminal_retention
                if _terminal_job_is_expired(job_row, cutoff):
                    return None

                status = job_row["status"]
                if status == "queued":
                    internal_id = _internal_id_from_row(job_row)
                    transition = await connection.execute(
                        update(transcription_job)
                        .where(
                            transcription_job.c.id == internal_id,
                            transcription_job.c.status == "queued",
                        )
                        .values(
                            status="finished",
                            outcome="cancelled",
                            started_at=None,
                            finished_at=now,
                            cancellation_requested=False,
                        )
                    )
                    if transition.rowcount != 1:
                        raise TranscriptionJobStoreUnavailableError()
                    cancelled_row = (
                        (
                            await connection.execute(
                                select(transcription_job).where(
                                    transcription_job.c.id == internal_id
                                )
                            )
                        )
                        .mappings()
                        .one_or_none()
                    )
                    if cancelled_row is None:
                        raise TranscriptionJobStoreUnavailableError()
                    return _cancelled_job_from_row(cancelled_row)

                if status == "processing":
                    internal_id = _internal_id_from_row(job_row)
                    await connection.execute(
                        update(transcription_job)
                        .where(
                            transcription_job.c.id == internal_id,
                            transcription_job.c.status == "processing",
                            transcription_job.c.cancellation_requested.is_(False),
                        )
                        .values(cancellation_requested=True)
                    )
                    processing_row = (
                        (
                            await connection.execute(
                                select(transcription_job).where(
                                    transcription_job.c.id == internal_id
                                )
                            )
                        )
                        .mappings()
                        .one_or_none()
                    )
                    if processing_row is None:
                        raise TranscriptionJobStoreUnavailableError()
                    return _processing_job_from_row(processing_row)

                if status != "finished":
                    raise TranscriptionJobStoreUnavailableError()
                if job_row["outcome"] == "cancelled":
                    return _cancelled_job_from_row(job_row)
                if job_row["outcome"] in {"succeeded", "failed"}:
                    raise TranscriptionJobAlreadyFinishedError()
                raise TranscriptionJobStoreUnavailableError()
        except TranscriptionJobAlreadyFinishedError:
            raise
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

    async def publish_cancelled(self, claim: ExecutionClaim) -> bool:
        """Publish Cancellation after active provider work has cleaned up.

        Args:
            claim: Exact worker-owned claim that completed provider cleanup.

        Returns:
            ``True`` when Cancellation committed the terminal outcome, otherwise
            ``False`` when the claim is stale or another transition owns the job.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot publish safely.
            ValueError: If the claim token or worker owner is invalid.
        """
        _require_uuid4(
            claim.dispatch.execution_attempt_token, "execution_attempt_token"
        )
        _require_uuid4(claim.worker_owner, "worker_owner")
        try:
            async with self._serialized_lifecycle_transaction() as connection:
                finished_at = _as_utc(self._clock())
                active_claim = exists(
                    select(transcription_job_execution_attempt.c.job_id).where(
                        transcription_job_execution_attempt.c.job_id
                        == claim.dispatch.internal_job_id,
                        transcription_job_execution_attempt.c.execution_attempt_token
                        == claim.dispatch.execution_attempt_token,
                        transcription_job_execution_attempt.c.worker_owner
                        == claim.worker_owner,
                        transcription_job_execution_attempt.c.claim_lease_expires_at
                        > finished_at,
                    )
                )
                transition = await connection.execute(
                    update(transcription_job)
                    .where(
                        transcription_job.c.id == claim.dispatch.internal_job_id,
                        transcription_job.c.status == "processing",
                        transcription_job.c.cancellation_requested.is_(True),
                        active_claim,
                    )
                    .values(
                        status="finished",
                        outcome="cancelled",
                        finished_at=finished_at,
                    )
                )
                if transition.rowcount != 1:
                    return False
                await self._clear_execution_claim(connection, claim)
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc
        return True

    async def publish_success(
        self,
        claim: ExecutionClaim,
        projected_result: TranscriptionResponse,
    ) -> bool:
        """Atomically persist one projected result and succeeded terminal transition.

        Args:
            claim: Exact worker-owned claim that produced the result.
            projected_result: Fully projected and validated public result.

        Returns:
            ``True`` when this consumer committed the successful terminal outcome,
            otherwise ``False`` when the claim is stale or Cancellation was accepted.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot publish safely.
            ValueError: If the claim token or worker owner is invalid.
        """
        _require_uuid4(
            claim.dispatch.execution_attempt_token, "execution_attempt_token"
        )
        _require_uuid4(claim.worker_owner, "worker_owner")
        validated_result = _validate_response(projected_result)
        result_values = _result_values(
            claim.dispatch.internal_job_id,
            validated_result,
        )
        segment_values = _segment_values(
            claim.dispatch.internal_job_id,
            validated_result,
        )
        try:
            async with self._serialized_lifecycle_transaction() as connection:
                finished_at = _as_utc(self._clock())
                active_claim = exists(
                    select(transcription_job_execution_attempt.c.job_id).where(
                        transcription_job_execution_attempt.c.job_id
                        == claim.dispatch.internal_job_id,
                        transcription_job_execution_attempt.c.execution_attempt_token
                        == claim.dispatch.execution_attempt_token,
                        transcription_job_execution_attempt.c.worker_owner
                        == claim.worker_owner,
                        transcription_job_execution_attempt.c.claim_lease_expires_at
                        > finished_at,
                    )
                )
                transition = await connection.execute(
                    update(transcription_job)
                    .where(
                        transcription_job.c.id == claim.dispatch.internal_job_id,
                        transcription_job.c.status == "processing",
                        transcription_job.c.cancellation_requested.is_(False),
                        active_claim,
                    )
                    .values(
                        status="finished",
                        outcome="succeeded",
                        finished_at=finished_at,
                    )
                )
                if transition.rowcount != 1:
                    return False
                await self._clear_execution_claim(connection, claim)
                await connection.execute(
                    insert(transcription_job_result).values(result_values)
                )
                if segment_values:
                    await connection.execute(
                        insert(transcription_job_segment),
                        segment_values,
                    )
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc
        return True

    async def publish_failure(
        self,
        claim: ExecutionClaim,
        error_code: str,
        error_message: str,
    ) -> bool:
        """Atomically persist one safe failed terminal transition.

        Args:
            claim: Exact worker-owned claim that encountered the failure.
            error_code: Validated stable public error code.
            error_message: Validated safe nonempty public error message.

        Returns:
            ``True`` when this consumer committed the failed terminal outcome,
            otherwise ``False`` when the claim is stale or Cancellation was accepted.

        Raises:
            TranscriptionJobStoreUnavailableError: If validation or PostgreSQL
                publication cannot complete safely.
            ValueError: If the claim token or worker owner is invalid.
        """
        _require_uuid4(
            claim.dispatch.execution_attempt_token, "execution_attempt_token"
        )
        _require_uuid4(claim.worker_owner, "worker_owner")
        error = _validate_error_detail(error_code, error_message)
        try:
            async with self._serialized_lifecycle_transaction() as connection:
                finished_at = _as_utc(self._clock())
                active_claim = exists(
                    select(transcription_job_execution_attempt.c.job_id).where(
                        transcription_job_execution_attempt.c.job_id
                        == claim.dispatch.internal_job_id,
                        transcription_job_execution_attempt.c.execution_attempt_token
                        == claim.dispatch.execution_attempt_token,
                        transcription_job_execution_attempt.c.worker_owner
                        == claim.worker_owner,
                        transcription_job_execution_attempt.c.claim_lease_expires_at
                        > finished_at,
                    )
                )
                transition = await connection.execute(
                    update(transcription_job)
                    .where(
                        transcription_job.c.id == claim.dispatch.internal_job_id,
                        transcription_job.c.status == "processing",
                        transcription_job.c.cancellation_requested.is_(False),
                        active_claim,
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
                    return False
                await self._clear_execution_claim(connection, claim)
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc
        return True

    async def _clear_execution_claim(
        self,
        connection: AsyncConnection,
        claim: ExecutionClaim,
    ) -> None:
        """Clear a terminalized claim within its winning lifecycle transaction."""
        cleared = await connection.execute(
            update(transcription_job_execution_attempt)
            .where(
                transcription_job_execution_attempt.c.job_id
                == claim.dispatch.internal_job_id,
                transcription_job_execution_attempt.c.execution_attempt_token
                == claim.dispatch.execution_attempt_token,
                transcription_job_execution_attempt.c.worker_owner
                == claim.worker_owner,
            )
            .values(
                worker_owner=None,
                last_heartbeat_at=None,
                claim_lease_expires_at=None,
            )
            .returning(transcription_job_execution_attempt.c.job_id.label("id"))
        )
        cleared_row = cleared.mappings().one_or_none()
        if (
            cleared_row is None
            or _internal_id_from_row(cleared_row) != claim.dispatch.internal_job_id
        ):
            raise TranscriptionJobStoreUnavailableError()

    async def get_job(self, public_id: UUID) -> TranscriptionJob | None:
        """Read one safe Transcription Job snapshot by bearer capability.

        Args:
            public_id: Canonical UUIDv4 capability supplied by the coordinator.

        Returns:
            Safe queued, processing, successful, failed, or cancelled snapshot, or
            ``None`` when unknown.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot read safely.
        """
        try:
            async with self._engine.connect() as connection, connection.begin():
                await connection.execute(
                    text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                )
                job_row = (
                    (
                        await connection.execute(
                            select(transcription_job).where(
                                transcription_job.c.public_id == public_id
                            )
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                snapshot: TranscriptionJob | None = None
                if job_row is not None:
                    cutoff = _as_utc(self._clock()) - self._terminal_retention
                    if not _terminal_job_is_expired(job_row, cutoff):
                        status = job_row["status"]
                        if status == "queued":
                            snapshot = _queued_job_from_row(job_row)
                        elif status == "processing":
                            snapshot = _processing_job_from_row(job_row)
                        elif status == "finished":
                            outcome = job_row["outcome"]
                            if outcome == "failed":
                                snapshot = _failed_job_from_row(job_row)
                            elif outcome == "cancelled":
                                snapshot = _cancelled_job_from_row(job_row)
                            elif outcome == "succeeded":
                                internal_id = _internal_id_from_row(job_row)
                                exclusions = await self._load_exclusions(
                                    connection,
                                    internal_id,
                                )
                                result_row = (
                                    (
                                        await connection.execute(
                                            select(transcription_job_result).where(
                                                transcription_job_result.c.job_id
                                                == internal_id
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
                                            .where(
                                                transcription_job_segment.c.job_id
                                                == internal_id
                                            )
                                            .order_by(
                                                transcription_job_segment.c.ordinal
                                            )
                                        )
                                    )
                                    .mappings()
                                    .all()
                                )
                                snapshot = _succeeded_job_from_rows(
                                    job_row,
                                    result_row,
                                    segment_rows,
                                    exclusions,
                                )
                            else:
                                raise TranscriptionJobStoreUnavailableError()
                        else:
                            raise TranscriptionJobStoreUnavailableError()
        except TranscriptionJobStoreUnavailableError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise TranscriptionJobStoreUnavailableError() from exc

        return snapshot

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


def _dispatch_from_row(row: RowMapping) -> JobDispatch:
    """Translate one private dispatch-only SQL row into a Job Dispatch."""
    execution_attempt_token = row["execution_attempt_token"]
    if (
        not isinstance(execution_attempt_token, UUID)
        or execution_attempt_token.version != 4
    ):
        raise TranscriptionJobStoreUnavailableError()
    return JobDispatch(
        internal_job_id=_internal_id_from_row(row),
        execution_attempt_token=execution_attempt_token,
    )


def _claimed_input_from_row(row: RowMapping) -> tuple[int, str]:
    """Validate private claim inputs selected after a successful state transition."""
    internal_id = _internal_id_from_row(row)
    submitted_url = row["submitted_url"]
    if not isinstance(submitted_url, str) or not submitted_url:
        raise TranscriptionJobStoreUnavailableError()
    return internal_id, submitted_url


def _claimed_job(
    claim: ExecutionClaim,
    submitted_url: str,
    exclusions: tuple[ResponseFieldPath, ...],
) -> ClaimedTranscriptionJob:
    """Construct one private committed claim snapshot."""
    return ClaimedTranscriptionJob(
        claim=claim,
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


def _cancelled_job_from_row(row: RowMapping) -> CancelledTranscriptionJob:
    """Translate a cancelled terminal row and enforce its legal origins."""
    if (
        row["status"] != "finished"
        or row["outcome"] != "cancelled"
        or row["error_code"] is not None
        or row["error_message"] is not None
    ):
        raise TranscriptionJobStoreUnavailableError()

    cancellation_requested = row["cancellation_requested"]
    started_at_value = row["started_at"]
    if started_at_value is None:
        if cancellation_requested is not False:
            raise TranscriptionJobStoreUnavailableError()
        started_at = None
    else:
        if cancellation_requested is not True:
            raise TranscriptionJobStoreUnavailableError()
        started_at = _datetime_from_row(row, "started_at")

    return CancelledTranscriptionJob(
        public_id=_public_id_from_row(row),
        status=JobStatus.FINISHED,
        outcome=JobOutcome.CANCELLED,
        submitted_at=_datetime_from_row(row, "submitted_at"),
        started_at=started_at,
        finished_at=_datetime_from_row(row, "finished_at"),
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


def _terminal_job_is_expired(row: RowMapping, cutoff: datetime) -> bool:
    """Return whether a terminal row has reached its retention deadline."""
    if row["status"] != "finished":
        return False
    return _datetime_from_row(row, "finished_at") <= cutoff


def _internal_id_from_row(row: RowMapping) -> int:
    """Return a validated private database identifier from one job row."""
    internal_id = row["id"]
    if isinstance(internal_id, bool) or not isinstance(internal_id, int):
        raise TranscriptionJobStoreUnavailableError()
    return internal_id


def _public_id_from_row(row: RowMapping) -> UUID:
    """Return one native UUIDv4 bearer capability from a PostgreSQL row."""
    public_id = row["public_id"]
    if not isinstance(public_id, UUID) or public_id.version != 4:
        raise TranscriptionJobStoreUnavailableError()
    return public_id


def _require_uuid4(value: UUID, name: str) -> None:
    """Reject non-UUIDv4 values at a repository method boundary."""
    if not isinstance(value, UUID) or value.version != 4:
        raise ValueError(f"{name} must be a UUIDv4.")


def _require_positive_duration(value: timedelta, name: str) -> None:
    """Reject invalid repository scheduling durations."""
    if not isinstance(value, timedelta) or value <= timedelta():
        raise ValueError(f"{name} must be a positive timedelta.")


def _require_positive_limit(limit: int) -> None:
    """Reject invalid repository batch limits."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer.")


def _datetime_from_row(row: RowMapping, column: str) -> datetime:
    """Read one required SQL timestamp as an aware UTC value."""
    value = row[column]
    if not isinstance(value, datetime):
        raise TranscriptionJobStoreUnavailableError()
    return _as_utc(value)


def _as_utc(value: datetime) -> datetime:
    """Normalize PostgreSQL timestamps to aware UTC values."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
