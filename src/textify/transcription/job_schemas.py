"""HTTP schemas for queued Transcription Job submission and inspection."""

from datetime import datetime
from typing import Literal

from pydantic import UUID4, BaseModel, ConfigDict


class TranscriptionJobLinks(BaseModel):
    """Expose stable relative links for one Transcription Job capability."""

    model_config = ConfigDict(extra="forbid", strict=True)

    self: str
    cancel: str


class QueuedTranscriptionJobResponse(BaseModel):
    """Serialize the safe public representation of an accepted queued job."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: UUID4
    status: Literal["queued"]
    submitted_at: datetime
    links: TranscriptionJobLinks
