"""Lifecycle types for durable Transcription Jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from textify.errors import PublicErrorCode
from textify.transcription.schemas import TranscriptionResponse
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
class JobDispatch:
    """Identify one private Execution Attempt delivered for processing."""

    internal_job_id: int
    execution_attempt_token: UUID = field(repr=False)

    def to_payload(self) -> dict[str, int | str]:
        """Serialize the exact JSON object permitted on the delivery broker.

        Returns:
            The private internal job identifier and canonical UUIDv4 attempt token.
        """
        return {
            "internal_job_id": self.internal_job_id,
            "execution_attempt_token": str(self.execution_attempt_token),
        }

    @classmethod
    def from_payload(cls, payload: object) -> JobDispatch | None:
        """Parse one exact, canonical broker dispatch payload.

        Args:
            payload: Deserialized JSON object received from the delivery broker.

        Returns:
            A validated private dispatch, or ``None`` when the payload is invalid.
        """
        if not isinstance(payload, dict) or set(payload) != {
            "internal_job_id",
            "execution_attempt_token",
        }:
            return None

        internal_job_id = payload["internal_job_id"]
        serialized_token = payload["execution_attempt_token"]
        if (
            isinstance(internal_job_id, bool)
            or not isinstance(internal_job_id, int)
            or internal_job_id <= 0
            or not isinstance(serialized_token, str)
        ):
            return None

        try:
            execution_attempt_token = UUID(serialized_token)
        except ValueError:
            return None
        if (
            str(execution_attempt_token) != serialized_token
            or execution_attempt_token.version != 4
        ):
            return None
        return cls(
            internal_job_id=internal_job_id,
            execution_attempt_token=execution_attempt_token,
        )


@dataclass(frozen=True, slots=True)
class NewQueuedTranscriptionJob:
    """Hold private data required to persist one newly admitted job."""

    submitted_url: str
    exclusions: tuple[ResponseFieldPath, ...]


@dataclass(frozen=True, slots=True)
class ClaimedTranscriptionJob:
    """Hold private execution inputs claimed for one Job Dispatch."""

    dispatch: JobDispatch
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


@dataclass(frozen=True, slots=True)
class ProcessingTranscriptionJob:
    """Hold the safe persisted projection of a processing job."""

    public_id: UUID
    status: JobStatus
    submitted_at: datetime
    started_at: datetime
    cancellation_requested: bool


@dataclass(frozen=True, slots=True)
class CancelledTranscriptionJob:
    """Hold the safe persisted projection of a cancelled finished job."""

    public_id: UUID
    status: JobStatus
    outcome: JobOutcome
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime


@dataclass(frozen=True, slots=True)
class SucceededTranscriptionJob:
    """Hold the safe persisted projection of a successful finished job."""

    public_id: UUID
    status: JobStatus
    outcome: JobOutcome
    submitted_at: datetime
    started_at: datetime
    finished_at: datetime
    result: TranscriptionResponse


@dataclass(frozen=True, slots=True)
class FailedTranscriptionJob:
    """Hold the safe persisted projection of a failed finished job."""

    public_id: UUID
    status: JobStatus
    outcome: JobOutcome
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime
    error_code: PublicErrorCode
    error_message: str


type TranscriptionJob = (
    QueuedTranscriptionJob
    | ProcessingTranscriptionJob
    | SucceededTranscriptionJob
    | FailedTranscriptionJob
    | CancelledTranscriptionJob
)
