"""Application coordination for durable Transcription Jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from textify.jobs.contracts import (
    TranscriptionJobAlreadyFinishedError,
    TranscriptionJobApiRepository,
    TranscriptionJobCapacityError,
    TranscriptionJobExecutionSignals,
    TranscriptionJobStoreUnavailableError,
)
from textify.jobs.exceptions import (
    JobAlreadyFinishedError,
    JobNotFoundError,
    JobStoreUnavailableError,
    TranscriptionCapacityExceededError,
)
from textify.jobs.runtime import TranscriptionJobLifecycleState
from textify.jobs.types import (
    CancelledTranscriptionJob,
    NewQueuedTranscriptionJob,
    ProcessingCancellation,
    ProcessingTranscriptionJob,
    QueuedTranscriptionJob,
    TranscriptionJob,
)
from textify.logging import bind_request_log_fields
from textify.transcription import inspection
from textify.transcription.types import ResponseFieldPath


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class TranscriptionJobService:
    """Validate and operate durable Transcription Jobs outside HTTP concerns."""

    def __init__(
        self,
        repository: TranscriptionJobApiRepository,
        execution_signals: TranscriptionJobExecutionSignals,
        lifecycle_state: TranscriptionJobLifecycleState,
    ) -> None:
        """Initialize Transcription Job HTTP lifecycle operations.

        Args:
            repository: Durable persistence boundary for HTTP lifecycle operations.
            execution_signals: Narrow execution notifications and store-failure latch.
            lifecycle_state: Shared admission and durable-store health state.
        """
        self._repository = repository
        self._execution_signals = execution_signals
        self._lifecycle_state = lifecycle_state

    def _require_store_available(self) -> None:
        """Raise when the caller holds the lifecycle lock after a store failure."""
        if not self._lifecycle_state.store_available:
            raise JobStoreUnavailableError()

    async def submit(
        self,
        submitted_url: str,
        exclusions: frozenset[ResponseFieldPath],
    ) -> QueuedTranscriptionJob:
        """Validate and durably queue one Supported Platform transcription.

        Args:
            submitted_url: Source URL supplied by the client.
            exclusions: Validated immutable response projection exclusions.

        Returns:
            Safe committed queued-job snapshot.

        Raises:
            InvalidUrlError: If the submitted URL is malformed.
            UnsupportedPlatformError: If the submitted URL platform is unsupported.
            TranscriptionCapacityExceededError: If durable admission is full.
            JobStoreUnavailableError: If the durable job store cannot commit safely.
        """
        submitted = inspection.classify_submitted_url(submitted_url)
        bind_request_log_fields(platform=submitted.platform.value)
        new_job = NewQueuedTranscriptionJob(
            submitted_url=submitted_url,
            exclusions=tuple(sorted(exclusions)),
        )
        try:
            async with self._lifecycle_state.lock:
                self._require_store_available()
                if not self._lifecycle_state.admission_open:
                    raise TranscriptionCapacityExceededError()
                queued_job = await self._repository.create_queued(new_job)
        except TranscriptionJobCapacityError as exc:
            raise TranscriptionCapacityExceededError() from exc
        except TranscriptionJobStoreUnavailableError as exc:
            await self._execution_signals.mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        self._execution_signals.notify_work_available()
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
            async with self._lifecycle_state.lock:
                self._require_store_available()
                job = await self._repository.get_job(public_id)
        except TranscriptionJobStoreUnavailableError as exc:
            await self._execution_signals.mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        if job is None:
            raise JobNotFoundError()
        return job

    async def cancel(
        self,
        capability: str,
    ) -> ProcessingTranscriptionJob | CancelledTranscriptionJob:
        """Accept cancellation through one canonical Transcription Job capability.

        Args:
            capability: Opaque client-supplied bearer capability path value.

        Returns:
            Current processing snapshot after durable acceptance, or the stable
            cancelled terminal snapshot.

        Raises:
            JobNotFoundError: If the capability is malformed, noncanonical, or unknown.
            JobAlreadyFinishedError: If the job has a non-cancelled terminal outcome.
            JobStoreUnavailableError: If durable cancellation cannot commit safely.
        """
        public_id = _parse_capability(capability)
        job: ProcessingTranscriptionJob | CancelledTranscriptionJob
        try:
            async with self._lifecycle_state.lock:
                self._require_store_available()
                cancellation = await self._repository.request_cancellation(public_id)
                if cancellation is None:
                    raise JobNotFoundError()
                if isinstance(cancellation, ProcessingCancellation):
                    self._execution_signals.request_execution_cancellation(
                        cancellation.internal_id
                    )
                    job = cancellation.job
                else:
                    job = cancellation
        except TranscriptionJobAlreadyFinishedError as exc:
            raise JobAlreadyFinishedError() from exc
        except TranscriptionJobStoreUnavailableError as exc:
            await self._execution_signals.mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        self._execution_signals.notify_work_available()
        return job


def _parse_capability(capability: str) -> UUID:
    """Parse only canonical lowercase UUIDv4 bearer capabilities."""
    try:
        public_id = UUID(capability)
    except (AttributeError, TypeError, ValueError) as exc:
        raise JobNotFoundError() from exc
    if public_id.version != 4 or str(public_id) != capability:
        raise JobNotFoundError()
    return public_id
