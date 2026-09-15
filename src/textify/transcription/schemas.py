"""HTTP schemas for the Textify transcript API."""

from typing import Annotated, Literal, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from textify.transcription.types import (
    Platform,
    Source,
    TimedTranscript,
    Transcript,
    TranscriptionResult,
    TranscriptMethod,
)

ResponseFieldPath = Literal[
    "source.platform",
    "source.video_id",
    "source.url",
    "source.title",
    "source.description",
    "source.channel",
    "source.duration_seconds",
    "transcript.method",
    "transcript.language",
    "transcript.text",
    "transcript.segments",
]


class TranscriptRequest(BaseModel):
    """Validate one submitted source URL and response field exclusions."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [
                {"url": "https://www.youtube.com/watch?v=AbCdEf12345"},
                {
                    "url": "https://www.youtube.com/watch?v=AbCdEf12345",
                    "exclude": ["transcript.segments"],
                },
            ]
        },
    )

    url: str = Field(
        min_length=1,
        max_length=2048,
        description="HTTPS URL for one supported public video.",
    )
    exclude: list[ResponseFieldPath] = Field(
        default_factory=list,
        description=(
            "Exact case-sensitive source or transcript leaf paths to omit "
            "from the response."
        ),
    )

    @model_validator(mode="after")
    def validate_excluded_fields(self) -> Self:
        """Reject duplicate exclusions and empty response roots."""
        excluded_fields = frozenset(self.exclude)
        if len(excluded_fields) != len(self.exclude):
            raise ValueError("exclude must not contain duplicate paths.")
        if {
            "source.platform",
            "source.video_id",
            "source.url",
            "source.title",
            "source.description",
            "source.channel",
            "source.duration_seconds",
        }.issubset(excluded_fields):
            raise ValueError("exclude must retain at least one source field.")
        if {
            "transcript.method",
            "transcript.language",
            "transcript.text",
            "transcript.segments",
        }.issubset(excluded_fields):
            raise ValueError("exclude must retain at least one transcript field.")
        return self


class SourceResponse(TypedDict, total=False):
    """Serialize canonical source identity and projected metadata."""

    platform: Annotated[
        Platform, Field(description="Normalized supported video platform.")
    ]
    video_id: Annotated[str, Field(description="Canonical provider video identifier.")]
    url: Annotated[str, Field(description="Canonical HTTPS source URL.")]
    title: Annotated[
        str, Field(description="Provider title, or an empty string when unavailable.")
    ]
    description: Annotated[
        str,
        Field(description="Provider description, or an empty string when unavailable."),
    ]
    channel: Annotated[
        str,
        Field(
            description="Provider channel or uploader, or an empty string when "
            "unavailable."
        ),
    ]
    duration_seconds: Annotated[
        int,
        Field(
            gt=0,
            description="Validated positive video duration in seconds.",
        ),
    ]


type.__setattr__(
    SourceResponse,
    "__pydantic_config__",
    ConfigDict(
        json_schema_extra={
            "minProperties": 1,
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
            ],
        }
    ),
)


class SegmentResponse(TypedDict):
    """Serialize one normalized timed transcript segment."""

    start: Annotated[
        float,
        Field(
            ge=0,
            description="Finite segment start offset in seconds.",
        ),
    ]
    end: Annotated[
        float,
        Field(
            ge=0,
            description="Finite segment end offset in seconds.",
        ),
    ]
    text: Annotated[
        str,
        Field(
            min_length=1,
            description="Normalized nonempty segment text.",
        ),
    ]


type.__setattr__(
    SegmentResponse,
    "__pydantic_config__",
    ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Synthetic transcript segment.",
                }
            ]
        }
    ),
)


class TranscriptResponse(TypedDict, total=False):
    """Serialize normalized transcript content and projected provenance."""

    method: Annotated[
        TranscriptMethod,
        Field(description="Transcript acquisition method."),
    ]
    language: Annotated[
        str,
        Field(description="Normalized BCP 47 language tag."),
    ]
    text: Annotated[
        str,
        Field(description="Space-joined text from the ordered normalized segments."),
    ]
    segments: Annotated[
        tuple[SegmentResponse, ...],
        Field(description="Ordered normalized timed transcript segments."),
    ]


type.__setattr__(
    TranscriptResponse,
    "__pydantic_config__",
    ConfigDict(
        json_schema_extra={
            "minProperties": 1,
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
            ],
        }
    ),
)


class TranscriptionResponse(TypedDict):
    """Serialize complete projected source metadata and transcript."""

    source: Annotated[
        SourceResponse,
        Field(description="Canonical projected source metadata."),
    ]
    transcript: Annotated[
        TranscriptResponse,
        Field(description="Normalized projected transcript result."),
    ]


type.__setattr__(
    TranscriptionResponse,
    "__pydantic_config__",
    ConfigDict(
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
                },
                {
                    "source": {
                        "platform": "youtube",
                        "video_id": "AbCdEf12345",
                        "url": "https://www.youtube.com/watch?v=AbCdEf12345",
                        "title": "Synthetic example video",
                        "channel": "Synthetic channel",
                        "duration_seconds": 42,
                    },
                    "transcript": {
                        "method": "youtube_captions",
                        "language": "en",
                        "text": "Synthetic transcript segment.",
                    },
                },
            ]
        }
    ),
)


def build_transcription_response(
    result: TranscriptionResult,
    excluded_fields: frozenset[ResponseFieldPath],
) -> TranscriptionResponse:
    """Build one validated projected API response.

    Args:
        result: Complete source and transcript domain result.
        excluded_fields: Exact validated response leaf paths to omit.

    Returns:
        Response representation with nonempty source and transcript roots.

    Raises:
        TypeError: If retained segments are unavailable on the transcript.
    """
    return {
        "source": _build_source_response(result.source, excluded_fields),
        "transcript": _build_transcript_response(
            result.transcript,
            excluded_fields,
        ),
    }


def _build_source_response(
    source: Source,
    excluded_fields: frozenset[ResponseFieldPath],
) -> SourceResponse:
    """Build projected source metadata in its public field order."""
    response: SourceResponse = {}
    if "source.platform" not in excluded_fields:
        response["platform"] = source.platform
    if "source.video_id" not in excluded_fields:
        response["video_id"] = source.video_id
    if "source.url" not in excluded_fields:
        response["url"] = source.url
    if "source.title" not in excluded_fields:
        response["title"] = source.title
    if "source.description" not in excluded_fields:
        response["description"] = source.description
    if "source.channel" not in excluded_fields:
        response["channel"] = source.channel
    if "source.duration_seconds" not in excluded_fields:
        response["duration_seconds"] = source.duration_seconds
    return response


def _build_transcript_response(
    transcript: Transcript,
    excluded_fields: frozenset[ResponseFieldPath],
) -> TranscriptResponse:
    """Build projected transcript data in its public field order."""
    response: TranscriptResponse = {}
    if "transcript.method" not in excluded_fields:
        response["method"] = transcript.method
    if "transcript.language" not in excluded_fields:
        response["language"] = transcript.language
    if "transcript.text" not in excluded_fields:
        response["text"] = transcript.text
    if "transcript.segments" not in excluded_fields:
        if not isinstance(transcript, TimedTranscript):
            raise TypeError("retained segments require a TimedTranscript.")
        response["segments"] = tuple(
            SegmentResponse(
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
            for segment in transcript.segments
        )
    return response


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
