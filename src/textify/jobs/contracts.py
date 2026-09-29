"""Storage-neutral seams for durable Transcription Job lifecycle ownership."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from textify.jobs.types import (
    CancellationResult,
    ClaimedTranscriptionJob,
    NewQueuedTranscriptionJob,
    QueuedTranscriptionJob,
    RecoveredTranscriptionJobs,
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
    ) -> CancellationResult | None:
        """Durably request cancellation of one Transcription Job.

        Args:
            public_id: Canonical UUIDv4 public capability.

        Returns:
            Processing cancellation details, a cancelled snapshot, or ``None`` when
            no visible job exists.

        Raises:
            TranscriptionJobAlreadyFinishedError: If the job has another terminal
                outcome.
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...


class TranscriptionJobExecutionRepository(Protocol):
    """Provide durable operations required by Transcription Job execution."""

    async def claim_next(self) -> ClaimedTranscriptionJob | None:
        """Claim the next eligible queued Transcription Job.

        Returns:
            Private claimed execution inputs, or ``None`` when no job is eligible.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def recover_interrupted_jobs(self) -> RecoveredTranscriptionJobs:
        """Reconcile Transcription Jobs left active by an interrupted process.

        Returns:
            Private identifiers grouped by committed recovery transition.

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

    async def publish_cancelled(self, internal_id: int) -> bool:
        """Conditionally publish cancellation after execution cleanup completes.

        Args:
            internal_id: Private identifier from the committed claim.

        Returns:
            ``True`` when this operation committed cancellation.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def publish_success(
        self,
        internal_id: int,
        projected_result: TranscriptionResponse,
    ) -> bool:
        """Conditionally publish one successful Transcription Job result.

        Args:
            internal_id: Private identifier from the committed claim.
            projected_result: Validated result after immutable field projection.

        Returns:
            ``True`` when this operation committed success.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...

    async def publish_failure(
        self,
        internal_id: int,
        error_code: str,
        error_message: str,
    ) -> bool:
        """Conditionally publish one safe failed Transcription Job outcome.

        Args:
            internal_id: Private identifier from the committed claim.
            error_code: Stable safe public error code.
            error_message: Safe public error message.

        Returns:
            ``True`` when this operation committed failure.

        Raises:
            TranscriptionJobStoreUnavailableError: If durable storage is unavailable.
        """
        ...


class TranscriptionJobExecutionSignals(Protocol):
    """Signal in-process execution without exposing execution ownership to HTTP."""

    def notify_work_available(self) -> None:
        """Wake execution consumers after a durable job state change."""
        ...

    def request_execution_cancellation(self, internal_id: int) -> None:
        """Request cancellation of one runner-owned active execution.

        Args:
            internal_id: Private identifier returned by a processing cancellation.
        """
        ...

    async def mark_store_unavailable(self) -> None:
        """Latch durable storage failure and stop in-process execution."""
        ...
