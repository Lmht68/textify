"""HTTP schemas for the Textify transcript API."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from textify.transcription.types import (
    Platform,
    Segment,
    Source,
    Transcript,
    TranscriptionResult,
    TranscriptMethod,
)


class TranscriptRequest(BaseModel):
    """Validate one submitted source URL."""

    model_config = ConfigDict(extra="forbid", strict=True)

    url: str = Field(min_length=1, max_length=2048)


class SourceResponse(BaseModel):
    """Serialize canonical source identity and metadata."""

    platform: Platform
    video_id: str
    url: str
    title: str
    description: str
    channel: str
    duration_seconds: int

    @classmethod
    def from_source(cls, source: Source) -> Self:
        """Build a response from one domain Source.

        Args:
            source: Canonical source domain value.

        Returns:
            Response representation of the Source.
        """
        return cls(
            platform=source.platform,
            video_id=source.video_id,
            url=source.url,
            title=source.title,
            description=source.description,
            channel=source.channel,
            duration_seconds=source.duration_seconds,
        )


class SegmentResponse(BaseModel):
    """Serialize one normalized timed transcript segment."""

    start: float
    end: float
    text: str

    @classmethod
    def from_segment(cls, segment: Segment) -> Self:
        """Build a response from one domain Segment.

        Args:
            segment: Normalized timed segment domain value.

        Returns:
            Response representation of the Segment.
        """
        return cls(start=segment.start, end=segment.end, text=segment.text)


class TranscriptResponse(BaseModel):
    """Serialize normalized transcript content and provenance."""

    method: TranscriptMethod
    language: str
    text: str
    segments: tuple[SegmentResponse, ...]

    @classmethod
    def from_transcript(cls, transcript: Transcript) -> Self:
        """Build a response from one domain Transcript.

        Args:
            transcript: Normalized transcript domain value.

        Returns:
            Response representation of the Transcript.
        """
        return cls(
            method=transcript.method,
            language=transcript.language,
            segments=tuple(
                SegmentResponse.from_segment(segment) for segment in transcript.segments
            ),
            text=transcript.text,
        )


class TranscriptionResponse(BaseModel):
    """Serialize complete source metadata and its transcript."""

    source: SourceResponse
    transcript: TranscriptResponse

    @classmethod
    def from_result(cls, result: TranscriptionResult) -> Self:
        """Build a response from one domain TranscriptionResult.

        Args:
            result: Complete source and transcript domain result.

        Returns:
            Response representation of the transcription result.
        """
        return cls(
            source=SourceResponse.from_source(result.source),
            transcript=TranscriptResponse.from_transcript(result.transcript),
        )
