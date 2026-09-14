"""HTTP schemas for the Textify transcript API."""

from typing import Literal, Self

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

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [{"url": "https://www.youtube.com/watch?v=AbCdEf12345"}]
        },
    )

    url: str = Field(
        min_length=1,
        max_length=2048,
        description="HTTPS URL for one supported public video.",
    )


class SourceResponse(BaseModel):
    """Serialize canonical source identity and metadata."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "platform": "youtube",
                    "video_id": "AbCdEf12345",
                    "url": "https://www.youtube.com/watch?v=AbCdEf12345",
                    "title": "Synthetic example video",
                    "description": "",
                    "channel": "Synthetic channel",
                    "duration_seconds": 42,
                }
            ]
        }
    )

    platform: Platform = Field(description="Normalized supported video platform.")
    video_id: str = Field(description="Canonical provider video identifier.")
    url: str = Field(description="Canonical HTTPS source URL.")
    title: str = Field(
        description="Provider title, or an empty string when unavailable."
    )
    description: str = Field(
        description="Provider description, or an empty string when unavailable."
    )
    channel: str = Field(
        description="Provider channel or uploader, or an empty string when unavailable."
    )
    duration_seconds: int = Field(
        gt=0,
        description="Validated positive video duration in seconds.",
    )

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

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Synthetic transcript segment.",
                }
            ]
        }
    )

    start: float = Field(
        ge=0,
        description="Finite segment start offset in seconds.",
    )
    end: float = Field(
        ge=0,
        description="Finite segment end offset in seconds.",
    )
    text: str = Field(description="Normalized nonempty segment text.")

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

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "method": "youtube_captions",
                    "language": "en",
                    "text": "Synthetic transcript segment.",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "Synthetic transcript segment.",
                        }
                    ],
                }
            ]
        }
    )

    method: TranscriptMethod = Field(description="Transcript acquisition method.")
    language: str = Field(description="Normalized BCP 47 language tag.")
    text: str = Field(
        description="Space-joined text from the ordered normalized segments."
    )
    segments: tuple[SegmentResponse, ...] = Field(
        description="Ordered normalized timed transcript segments."
    )

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

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source": {
                        "platform": "youtube",
                        "video_id": "AbCdEf12345",
                        "url": "https://www.youtube.com/watch?v=AbCdEf12345",
                        "title": "Synthetic example video",
                        "description": "",
                        "channel": "Synthetic channel",
                        "duration_seconds": 42,
                    },
                    "transcript": {
                        "method": "youtube_captions",
                        "language": "en",
                        "text": "Synthetic transcript segment.",
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 1.0,
                                "text": "Synthetic transcript segment.",
                            }
                        ],
                    },
                }
            ]
        }
    )

    source: SourceResponse = Field(description="Canonical source metadata.")
    transcript: TranscriptResponse = Field(description="Normalized transcript result.")

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


PublicErrorCode = Literal[
    "invalid_url",
    "unsupported_platform",
    "invalid_request",
    "unsupported_content",
    "video_too_long",
    "invalid_media_duration",
    "unsupported_media",
    "no_usable_transcript",
    "metadata_retrieval_failed",
    "audio_download_failed",
    "transcription_failed",
    "transcription_capacity_exceeded",
    "metadata_timeout",
    "audio_download_timeout",
    "transcription_timeout",
    "internal_error",
]


class ErrorDetail(BaseModel):
    """Describe one stable public API failure."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "code": "invalid_url",
                    "message": "The submitted URL is invalid.",
                }
            ]
        }
    )

    code: PublicErrorCode = Field(description="Stable machine-readable error code.")
    message: str = Field(
        min_length=1,
        description="Safe human-readable message that is not byte-stable.",
    )


class ErrorResponse(BaseModel):
    """Serialize the safe public API error envelope."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "invalid_url",
                        "message": "The submitted URL is invalid.",
                    }
                }
            ]
        }
    )

    error: ErrorDetail = Field(description="Safe stable error detail.")
