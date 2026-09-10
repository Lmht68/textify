"""FastAPI endpoint for synchronous Textify transcription."""

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


@router.post(
    "/transcripts",
    status_code=status.HTTP_200_OK,
    summary="Create a transcript",
    description="Retrieve a normalized timed transcript for one public TikTok video.",
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
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "An external provider timed out."
        },
    },
)
async def create_transcript(
    request: TranscriptRequest,
    transcript_service: TranscriptServiceDependency,
) -> TranscriptionResponse:
    """Create one synchronous TikTok transcript.

    Args:
        request: Strictly validated submitted source URL.
        transcript_service: Lifespan-owned application service.

    Returns:
        Canonical source metadata and a normalized timed transcript.
    """
    result = await transcript_service.transcribe(request.url)
    return TranscriptionResponse.from_result(result)
