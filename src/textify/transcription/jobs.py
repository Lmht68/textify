"""Application coordination for durable queued Transcription Jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from textify.logging import bind_request_log_fields
from textify.transcription import inspection
from textify.transcription.exceptions import (
    JobNotFoundError,
    JobStoreUnavailableError,
    TranscriptionCapacityExceededError,
)
from textify.transcription.job_repository import (
    SqliteTranscriptionJobRepository,
    TranscriptionJobCapacityError,
    TranscriptionJobStoreUnavailableError,
)
from textify.transcription.types import ResponseFieldPath


class JobStatus(StrEnum):
    """Represent a Transcription Job's coarse lifecycle position."""

    QUEUED = "queued"
    PROCESSING = "processing"
    FINISHED = "finished"


class JobOutcome(StrEnum):
    """Represent the terminal disposition of a finished Transcription Job."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class NewQueuedTranscriptionJob:
    """Hold private data required to persist one newly admitted job."""

    submitted_url: str
    exclusions: tuple[ResponseFieldPath, ...]


@dataclass(frozen=True, slots=True)
class QueuedTranscriptionJob:
    """Hold the safe persisted projection of an accepted queued job."""

    internal_id: int
    public_id: UUID
    status: JobStatus
    submitted_at: datetime
    queue_deadline_at: datetime


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class TranscriptionJobCoordinator:
    """Validate, admit, and inspect queued Transcription Jobs."""

    def __init__(self, repository: SqliteTranscriptionJobRepository) -> None:
        """Initialize the durable queued-job application boundary.

        Args:
            repository: SQLite adapter that atomically admits and retrieves jobs.
        """
        self._repository = repository

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
            return await self._repository.create_queued(new_job)
        except TranscriptionJobCapacityError as exc:
            raise TranscriptionCapacityExceededError() from exc
        except TranscriptionJobStoreUnavailableError as exc:
            raise JobStoreUnavailableError() from exc

    async def get_status(self, capability: str) -> QueuedTranscriptionJob:
        """Retrieve one queued job through a canonical UUIDv4 capability.

        Args:
            capability: Opaque client-supplied bearer capability path value.

        Returns:
            Safe queued-job snapshot.

        Raises:
            JobNotFoundError: If the capability is malformed, noncanonical, or unknown.
            JobStoreUnavailableError: If the durable job store cannot read safely.
        """
        public_id = _parse_capability(capability)
        try:
            queued_job = await self._repository.get_queued(public_id)
        except TranscriptionJobStoreUnavailableError as exc:
            raise JobStoreUnavailableError() from exc
        if queued_job is None:
            raise JobNotFoundError()
        return queued_job


def _parse_capability(capability: str) -> UUID:
    """Parse only canonical lowercase UUIDv4 bearer capabilities."""
    try:
        public_id = UUID(capability)
    except (AttributeError, TypeError, ValueError) as exc:
        raise JobNotFoundError() from exc
    if public_id.version != 4 or str(public_id) != capability:
        raise JobNotFoundError()
    return public_id
