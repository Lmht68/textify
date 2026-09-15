"""FastAPI endpoint for synchronous Textify transcription."""

import asyncio
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, status

from textify.transcription.schemas import (
    ErrorResponse,
    TranscriptionResponse,
    TranscriptRequest,
    build_transcription_response,
)
from textify.transcription.service import TranscriptService

router = APIRouter(prefix="/api")


async def get_transcript_service(request: Request) -> TranscriptService:
    """Return the lifespan-owned transcript service.

    Args:
        request: Current HTTP request.

    Returns:
        Process-lifetime TranscriptService instance.

    Raises:
        RuntimeError: If the application has not completed startup.
    """
    service = getattr(request.app.state, "transcript_service", None)
    if service is None:
        raise RuntimeError("Transcript service is unavailable before startup.")
    return cast(TranscriptService, service)


type TranscriptServiceDependency = Annotated[
    TranscriptService,
    Depends(get_transcript_service),
]


async def _wait_for_disconnect(request: Request) -> None:
    """Wait until the ASGI server reports that the client disconnected.

    Args:
        request: HTTP request whose receive channel is monitored.
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _cancel_and_await[T](task: asyncio.Task[T]) -> None:
    """Cancel one child task and consume its terminal outcome.

    Args:
        task: Child task owned by the route lifecycle.
    """
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@router.post(
    "/transcripts",
    status_code=status.HTTP_200_OK,
    response_model=TranscriptionResponse,
    summary="Create a transcript",
    description=(
        "Synchronously retrieve a normalized transcript for one public Facebook, "
        "Instagram, TikTok, X, or YouTube video."
    ),
    response_description="Canonical source metadata and its normalized transcript.",
    tags=["transcripts"],
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": (
                "Invalid input URL. Possible codes: `invalid_url`, "
                "`unsupported_platform`."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "invalid_url",
                            "message": "The submitted URL is invalid.",
                        }
                    }
                }
            },
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": (
                "Invalid request or ineligible source. Possible codes: "
                "`invalid_request`, `unsupported_content`, `video_too_long`, "
                "`invalid_media_duration`, `unsupported_media`, "
                "`no_usable_transcript`."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "invalid_request",
                            "message": "The request is invalid.",
                        }
                    }
                }
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Unexpected server failure. Possible code: `internal_error`.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "internal_error",
                            "message": "An unexpected error occurred.",
                        }
                    }
                }
            },
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": (
                "Provider or inference failure. Possible codes: "
                "`metadata_retrieval_failed`, `audio_download_failed`, "
                "`transcription_failed`."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "metadata_retrieval_failed",
                            "message": "Source metadata could not be retrieved.",
                        }
                    }
                }
            },
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": (
                "Transcription admission or inference capacity is unavailable. "
                "Possible code: `transcription_capacity_exceeded`."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "transcription_capacity_exceeded",
                            "message": "Transcription capacity is currently unavailable.",
                        }
                    }
                }
            },
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "model": ErrorResponse,
            "description": (
                "Provider or inference timeout. Possible codes: "
                "`metadata_timeout`, `audio_download_timeout`, "
                "`transcription_timeout`."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "metadata_timeout",
                            "message": "Source metadata retrieval timed out.",
                        }
                    }
                }
            },
        },
    },
)
async def create_transcript(
    payload: TranscriptRequest,
    http_request: Request,
    transcript_service: TranscriptServiceDependency,
) -> TranscriptionResponse:
    """Create one synchronous Supported Platform transcript.

    Args:
        payload: Strictly validated submitted URL and response exclusions.
        http_request: HTTP request whose disconnect signal cancels the lifecycle.
        transcript_service: Lifespan-owned application service.

    Returns:
        Canonical projected source metadata and normalized transcript.
    """
    excluded_fields = frozenset(payload.exclude)
    service_task = asyncio.create_task(
        transcript_service.transcribe(
            payload.url,
            include_segments="transcript.segments" not in excluded_fields,
        ),
        name="transcript-service",
    )
    disconnect_task = asyncio.create_task(
        _wait_for_disconnect(http_request),
        name="transcript-disconnect-watcher",
    )
    try:
        completed, _ = await asyncio.wait(
            {service_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if service_task in completed:
            await _cancel_and_await(disconnect_task)
            result = service_task.result()
            return build_transcription_response(result, excluded_fields)

        await _cancel_and_await(service_task)
        raise asyncio.CancelledError
    finally:
        await _cancel_and_await(service_task)
        await _cancel_and_await(disconnect_task)
