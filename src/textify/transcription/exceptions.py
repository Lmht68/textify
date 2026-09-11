"""Safe domain errors and HTTP response handlers for transcription."""

import logging
from typing import ClassVar, cast

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Base class for safe, parameterless transcription errors.

    Attributes:
        code: Stable machine-readable API error code.
        status_code: HTTP status associated with the error code.
        message: Safe nonempty message for API consumers.
    """

    code: ClassVar[str]
    status_code: ClassVar[int]
    message: ClassVar[str]

    def __init__(self) -> None:
        """Initialize the error with its fixed safe message."""
        super().__init__(self.message)


class InvalidUrlError(TranscriptionError):
    """Indicate a malformed or unsupported URL shape."""

    code = "invalid_url"
    status_code = 400
    message = "The submitted URL is invalid."


class UnsupportedPlatformError(TranscriptionError):
    """Indicate a syntactically valid URL for an unsupported platform."""

    code = "unsupported_platform"
    status_code = 400
    message = "The submitted platform is not supported."


class UnsupportedContentError(TranscriptionError):
    """Indicate inaccessible, private, blocked, or login-gated content."""

    code = "unsupported_content"
    status_code = 422
    message = "The source content cannot be transcribed."


class VideoTooLongError(TranscriptionError):
    """Indicate media exceeding the configured duration limit."""

    code = "video_too_long"
    status_code = 422
    message = "The source video exceeds the duration limit."


class InvalidMediaDurationError(TranscriptionError):
    """Indicate missing, nonpositive, or nonfinite source duration."""

    code = "invalid_media_duration"
    status_code = 422
    message = "The source media has an invalid duration."


class UnsupportedMediaError(TranscriptionError):
    """Indicate a source that does not provide usable video media."""

    code = "unsupported_media"
    status_code = 422
    message = "The source media is not supported."


class NoUsableTranscriptError(TranscriptionError):
    """Indicate eligible content with no usable transcript."""

    code = "no_usable_transcript"
    status_code = 422
    message = "No usable transcript is available."


class MetadataRetrievalFailedError(TranscriptionError):
    """Indicate ordinary metadata provider failure."""

    code = "metadata_retrieval_failed"
    status_code = 502
    message = "Source metadata could not be retrieved."


class AudioDownloadFailedError(TranscriptionError):
    """Indicate ordinary audio download provider failure."""

    code = "audio_download_failed"
    status_code = 502
    message = "Source audio could not be downloaded."


class TranscriptionFailedError(TranscriptionError):
    """Indicate ordinary native transcription failure."""

    code = "transcription_failed"
    status_code = 502
    message = "The source could not be transcribed."


class TranscriptionCapacityExceededError(TranscriptionError):
    """Indicate exhausted admission or native inference queue capacity."""

    status_code = 503
    code = "transcription_capacity_exceeded"
    message = "Transcription capacity is currently unavailable."


class MetadataTimeoutError(TranscriptionError):
    """Indicate metadata retrieval timing out."""

    code = "metadata_timeout"
    status_code = 504
    message = "Source metadata retrieval timed out."


class AudioDownloadTimeoutError(TranscriptionError):
    """Indicate audio downloading timing out."""

    code = "audio_download_timeout"
    status_code = 504
    message = "Source audio download timed out."


class TranscriptionTimeoutError(TranscriptionError):
    """Indicate native transcription timing out."""

    code = "transcription_timeout"
    status_code = 504
    message = "Source transcription timed out."


async def transcription_error_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Convert a domain error to its stable safe HTTP response.

    Args:
        _request: Request that triggered the domain error.
        exc: Exception registered as a ``TranscriptionError`` handler.

    Returns:
        Error response with the domain status and code.
    """
    error = cast(TranscriptionError, exc)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
    )


async def unhandled_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
    """Return a generic safe response for unexpected failures.

    Args:
        _request: Request that triggered the unexpected error.
        _exc: Unexpected exception raised while handling the request.

    Returns:
        Generic internal-error response.
    """
    logger.error("unhandled error")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred.",
            }
        },
    )
