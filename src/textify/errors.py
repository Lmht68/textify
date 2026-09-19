"""Stable public API errors and safe HTTP error handlers."""

from typing import ClassVar, Literal, cast

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from textify.logging import bind_request_log_fields

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
    "job_not_found",
    "job_store_unavailable",
    "queue_timeout",
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


class TextifyError(Exception):
    """Base class for safe, parameterless Textify errors.

    Attributes:
        code: Stable machine-readable API error code.
        status_code: HTTP status associated with the error code.
        message: Safe nonempty message for API consumers.
    """

    code: ClassVar[PublicErrorCode]
    status_code: ClassVar[int]
    message: ClassVar[str]

    def __init__(self) -> None:
        """Initialize the error with its fixed safe message."""
        super().__init__(self.message)


class InvalidRequestError(TextifyError):
    """Indicate a malformed HTTP request payload."""

    code = "invalid_request"
    status_code = 422
    message = "The request is invalid."


class InternalError(TextifyError):
    """Indicate an unexpected internal Textify failure."""

    code = "internal_error"
    status_code = 500
    message = "An unexpected error occurred."


async def textify_error_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Convert a domain error to its stable safe HTTP response.

    Args:
        _request: Request that triggered the domain error.
        exc: Exception registered as a ``TextifyError`` handler.

    Returns:
        Error response with the domain status and code.
    """
    error = cast(TextifyError, exc)
    bind_request_log_fields(code=error.code)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
    )


async def request_validation_error_handler(
    _request: Request,
    _exc: Exception,
) -> JSONResponse:
    """Return a safe response for malformed HTTP request payloads.

    Args:
        _request: Request that failed FastAPI validation.
        _exc: Validation details intentionally excluded from the response.

    Returns:
        Stable invalid-request response without rejected input values.
    """
    return await textify_error_handler(_request, InvalidRequestError())


async def unhandled_error_handler(
    _request: Request,
    _exc: Exception,
) -> JSONResponse:
    """Return a generic safe response for unexpected failures.

    Args:
        _request: Request that triggered the unexpected error.
        _exc: Unexpected exception raised while handling the request.

    Returns:
        Generic internal-error response.
    """
    return await textify_error_handler(_request, InternalError())
