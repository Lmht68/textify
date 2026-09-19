"""HTTP schemas for Transcription Job submission and status inspection."""

from datetime import datetime
from typing import Literal

from pydantic import UUID4, BaseModel, ConfigDict

from textify.transcription.schemas import ErrorDetail, TranscriptionResponse


class ActiveTranscriptionJobLinks(BaseModel):
    """Expose stable active-job links for one Transcription Job capability."""

    model_config = ConfigDict(extra="forbid", strict=True)

    self: str
    cancel: str


class QueuedTranscriptionJobResponse(BaseModel):
    """Serialize the safe public representation of an accepted queued job."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: UUID4
    status: Literal["queued"]
    submitted_at: datetime
    links: ActiveTranscriptionJobLinks


class ProcessingTranscriptionJobResponse(BaseModel):
    """Serialize the safe public representation of a processing job."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: UUID4
    status: Literal["processing"]
    submitted_at: datetime
    started_at: datetime
    cancellation_requested: bool
    links: ActiveTranscriptionJobLinks


class FinishedTranscriptionJobLinks(BaseModel):
    """Expose the stable terminal-job link for one capability."""

    model_config = ConfigDict(extra="forbid", strict=True)

    self: str


class SucceededTranscriptionJobResponse(BaseModel):
    """Serialize the complete public result of a successful finished job."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: UUID4
    status: Literal["finished"]
    outcome: Literal["succeeded"]
    submitted_at: datetime
    started_at: datetime
    finished_at: datetime
    result: TranscriptionResponse
    links: FinishedTranscriptionJobLinks


class FailedTranscriptionJobResponse(BaseModel):
    """Serialize the safe reason for a failed finished job."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: UUID4
    status: Literal["finished"]
    outcome: Literal["failed"]
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime
    error: ErrorDetail
    links: FinishedTranscriptionJobLinks


type TranscriptionJobResponse = (
    QueuedTranscriptionJobResponse
    | ProcessingTranscriptionJobResponse
    | SucceededTranscriptionJobResponse
    | FailedTranscriptionJobResponse
)
