"""Application coordination for durable Transcription Jobs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from textify.identifiers import parse_canonical_uuid4
from textify.jobs.contracts import (
    TranscriptionJobAlreadyFinishedError,
    TranscriptionJobApiRepository,
    TranscriptionJobCapacityError,
    TranscriptionJobIdempotencyConflictError,
    TranscriptionJobStoreUnavailableError,
)
from textify.jobs.exceptions import (
    IdempotencyConflictError,
    JobAlreadyFinishedError,
    JobNotFoundError,
    JobStoreUnavailableError,
    TranscriptionCapacityExceededError,
)
from textify.jobs.types import (
    CancelledTranscriptionJob,
    NewQueuedTranscriptionJob,
    QueuedTranscriptionJob,
    TranscriptionJob,
    TranscriptionJobCancellationResult,
    TranscriptionJobRequestIdentity,
)
from textify.logging import LogEvent, bind_request_log_fields, emit_event
from textify.transcription import inspection
from textify.transcription.types import ResponseFieldPath

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class TranscriptionJobService:
    """Validate and operate durable Transcription Jobs outside HTTP concerns."""

    def __init__(
        self,
        repository: TranscriptionJobApiRepository,
        on_store_unavailable: Callable[[], None],
    ) -> None:
        """Initialize Transcription Job HTTP lifecycle operations.

        Args:
            repository: Durable persistence boundary for HTTP lifecycle operations.
            on_store_unavailable: Readiness callback for the first store failure.
        """
        self._repository = repository
        self._on_store_unavailable = on_store_unavailable
        self._lock = asyncio.Lock()
        self._store_available = True

    def _require_store_available(self) -> None:
        """Raise after a durable-store failure has latched service unavailability."""
        if not self._store_available:
            raise JobStoreUnavailableError()

    async def _mark_store_unavailable(self) -> None:
        """Latch the first durable-store failure and make readiness unavailable."""
        async with self._lock:
            if self._store_available:
                self._store_available = False
                self._on_store_unavailable()

    async def submit(
        self,
        submitted_url: str,
        exclusions: frozenset[ResponseFieldPath],
        idempotency_key: UUID,
    ) -> QueuedTranscriptionJob:
        """Validate and durably queue one Supported Platform transcription.

        Args:
            submitted_url: Source URL supplied by the client.
            exclusions: Validated immutable response projection exclusions.
            idempotency_key: Canonical UUIDv4 for this intentional submission.

        Returns:
            Safe committed queued-job acceptance snapshot.

        Raises:
            InvalidUrlError: If the submitted URL is malformed.
            UnsupportedPlatformError: If the submitted URL platform is unsupported.
            TranscriptionCapacityExceededError: If durable admission is full.
            IdempotencyConflictError: If the key identifies another request.
            JobStoreUnavailableError: If the durable job store cannot commit safely.
        """
        submitted = inspection.classify_submitted_url(submitted_url)
        bind_request_log_fields(platform=submitted.platform.value)
        sorted_exclusions = tuple(sorted(exclusions))
        new_job = NewQueuedTranscriptionJob(
            request_identity=TranscriptionJobRequestIdentity(
                submitted_url=submitted_url,
                exclusions=sorted_exclusions,
            ),
            idempotency_key=idempotency_key,
        )
        try:
            async with self._lock:
                self._require_store_available()
                queued_job = await self._repository.create_queued(new_job)
        except TranscriptionJobCapacityError as exc:
            raise TranscriptionCapacityExceededError() from exc
        except TranscriptionJobIdempotencyConflictError as exc:
            raise IdempotencyConflictError() from exc
        except TranscriptionJobStoreUnavailableError as exc:
            await self._mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        emit_event(
            logger,
            LogEvent.TRANSCRIPTION_JOB_ADMITTED,
            internal_job_id=queued_job.internal_id,
            job_status=queued_job.status.value,
        )
        return queued_job

    async def get_status(self, capability: str) -> TranscriptionJob:
        """Retrieve one Transcription Job through a canonical UUIDv4 capability.

        Args:
            capability: Opaque client-supplied bearer capability path value.

        Returns:
            Safe current job snapshot.

        Raises:
            JobNotFoundError: If the capability is malformed, noncanonical, or unknown.
            JobStoreUnavailableError: If the durable job store cannot read safely.
        """
        public_id = _parse_capability(capability)
        try:
            async with self._lock:
                self._require_store_available()
                job = await self._repository.get_job(public_id)
        except TranscriptionJobStoreUnavailableError as exc:
            await self._mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        if job is None:
            raise JobNotFoundError()
        return job

    async def cancel(
        self,
        capability: str,
    ) -> TranscriptionJobCancellationResult:
        """Accept cancellation through one canonical Transcription Job capability.

        Args:
            capability: Opaque client-supplied bearer capability path value.

        Returns:
            Committed Cancellation result whose ``first_accepted`` field identifies
            whether this transaction made the first Cancellation transition.

        Raises:
            JobNotFoundError: If the capability is malformed, noncanonical, or unknown.
            JobAlreadyFinishedError: If the job has a non-cancelled terminal outcome.
            JobStoreUnavailableError: If durable cancellation cannot commit safely.
        """
        public_id = _parse_capability(capability)
        try:
            async with self._lock:
                self._require_store_available()
                cancellation = await self._repository.request_cancellation(public_id)
        except TranscriptionJobAlreadyFinishedError as exc:
            raise JobAlreadyFinishedError() from exc
        except TranscriptionJobStoreUnavailableError as exc:
            await self._mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        if cancellation is None:
            raise JobNotFoundError()
        if cancellation.first_accepted:
            job = cancellation.job
            if isinstance(job, CancelledTranscriptionJob):
                emit_event(
                    logger,
                    LogEvent.TRANSCRIPTION_JOB_CANCELLATION_ACCEPTED,
                    internal_job_id=cancellation.internal_job_id,
                    job_status=job.status.value,
                    job_outcome=job.outcome.value,
                )
                emit_event(
                    logger,
                    LogEvent.TRANSCRIPTION_JOB_FINISHED,
                    internal_job_id=cancellation.internal_job_id,
                    job_status=job.status.value,
                    job_outcome=job.outcome.value,
                )
            else:
                emit_event(
                    logger,
                    LogEvent.TRANSCRIPTION_JOB_CANCELLATION_ACCEPTED,
                    internal_job_id=cancellation.internal_job_id,
                    job_status=job.status.value,
                )
        return cancellation


def _parse_capability(capability: str) -> UUID:
    """Parse only canonical lowercase UUIDv4 bearer capabilities."""
    try:
        return parse_canonical_uuid4(capability)
    except ValueError as exc:
        raise JobNotFoundError() from exc
