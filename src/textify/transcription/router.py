"""FastAPI endpoint for synchronous Textify transcription."""

import asyncio
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, status

from textify.transcription.schemas import TranscriptionResponse, TranscriptRequest
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
    summary="Create a transcript",
    description=(
        "Retrieve a normalized timed transcript for one public TikTok or YouTube video."
    ),
    response_description="Canonical source metadata and its transcript.",
    tags=["transcripts"],
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "The URL is invalid or its platform is disabled."
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The source cannot produce an eligible transcript."
        },
        status.HTTP_502_BAD_GATEWAY: {"description": "An external provider failed."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": (
                "Transcription admission or inference queue capacity was exceeded."
            )
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "An external provider timed out."
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
        payload: Strictly validated submitted source URL.
        http_request: HTTP request whose disconnect signal cancels the lifecycle.
        transcript_service: Lifespan-owned application service.

    Returns:
        Canonical source metadata and a normalized timed transcript.
    """
    service_task = asyncio.create_task(
        transcript_service.transcribe(payload.url),
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
            return TranscriptionResponse.from_result(result)

        await _cancel_and_await(service_task)
        raise asyncio.CancelledError
    finally:
        await _cancel_and_await(service_task)
        await _cancel_and_await(disconnect_task)
