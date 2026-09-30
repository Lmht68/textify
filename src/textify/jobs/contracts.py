"""Storage-neutral seams for durable Transcription Job lifecycle ownership."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from textify.jobs.types import (
    CancelledTranscriptionJob,
    ClaimedTranscriptionJob,
    JobDispatch,
    NewQueuedTranscriptionJob,
    ProcessingTranscriptionJob,
    QueuedTranscriptionJob,
    TranscriptionJob,
)
from textify.transcription.schemas import TranscriptionResponse


class TranscriptionJobCapacityError(RuntimeError):
    """Indicate durable job admission has reached its configured maximum."""


class TranscriptionJobStoreUnavailableError(RuntimeError):
    """Indicate durable job storage cannot safely complete an operation."""


class TranscriptionJobAlreadyFinishedError(RuntimeError):
    """Indicate cancellation targeted a completed non-cancelled job."""


class TranscriptionJobApiRepository(Protocol):
    """Provide durable operations required by the Transcription Job HTTP lifecycle."""

    async def create_queued(
        self,
        job: NewQueuedTranscriptionJob,
    ) -> QueuedTranscriptionJob:
        """Persist one validated queued Transcription Job.

        Args:
            job: Private source URL and sorted immutable response exclusions.

        Returns:
            Safe committed queued-job snapshot.

        Raises:
            TranscriptionJobCapacityError: If durable admission is full.
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def get_job(self, public_id: UUID) -> TranscriptionJob | None:
        """Read one Transcription Job through its bearer capability.

        Args:
            public_id: Canonical UUIDv4 public capability.

        Returns:
            Safe current snapshot, or ``None`` when no visible job exists.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def request_cancellation(
        self,
        public_id: UUID,
    ) -> ProcessingTranscriptionJob | CancelledTranscriptionJob | None:
        """Durably request cancellation of one Transcription Job.

        Args:
            public_id: Canonical UUIDv4 public capability.

        Returns:
            Processing or cancelled snapshot, or ``None`` when no visible job exists.

        Raises:
            TranscriptionJobAlreadyFinishedError: If the job has another terminal
                outcome.
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...


class TranscriptionJobDispatchRepository(Protocol):
    """Provide committed dispatch and retention operations to the reconciler."""

    async def list_queued_dispatches(self) -> tuple[JobDispatch, ...]:
        """List eligible queued Execution Attempts without private job inputs.

        Returns:
            Dispatches ordered by immutable submission time and private job identifier.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def delete_expired_terminal_jobs(self) -> None:
        """Delete expired terminal Transcription Jobs.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...


class TranscriptionJobExecutionRepository(Protocol):
    """Provide durable operations required by one claimed Execution Attempt."""

    async def claim_dispatch(
        self, dispatch: JobDispatch
    ) -> ClaimedTranscriptionJob | None:
        """Conditionally claim one delivered Job Dispatch before loading inputs.

        Args:
            dispatch: Private identifier and token received from the broker.

        Returns:
            Private claimed execution inputs, or ``None`` when delivery is stale.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def publish_cancelled(self, dispatch: JobDispatch) -> bool:
        """Conditionally publish cancellation after execution cleanup completes.

        Args:
            dispatch: Private identifier and token from the committed claim.

        Returns:
            ``True`` when this operation committed cancellation.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def publish_success(
        self,
        dispatch: JobDispatch,
        projected_result: TranscriptionResponse,
    ) -> bool:
        """Conditionally publish one successful Transcription Job result.

        Args:
            dispatch: Private identifier and token from the committed claim.
            projected_result: Validated result after immutable field projection.

        Returns:
            ``True`` when this operation committed success.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def publish_failure(
        self,
        dispatch: JobDispatch,
        error_code: str,
        error_message: str,
    ) -> bool:
        """Conditionally publish one safe failed Transcription Job outcome.

        Args:
            dispatch: Private identifier and token from the committed claim.
            error_code: Stable safe public error code.
            error_message: Safe public error message.

        Returns:
            ``True`` when this operation committed failure.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...
