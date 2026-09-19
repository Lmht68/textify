"""Lifecycle types for durable Transcription Jobs."""

from dataclasses import dataclass
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
class NewQueuedTranscriptionJob:
    """Hold private data required to persist one newly admitted job."""

    submitted_url: str
    exclusions: tuple[ResponseFieldPath, ...]


@dataclass(frozen=True, slots=True)
class ClaimedTranscriptionJob:
    """Hold private execution inputs durably claimed by one consumer."""

    internal_id: int
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
)
