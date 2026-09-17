"""FastAPI routes for durable queued Transcription Jobs."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, Response, status

from textify.transcription.job_schemas import (
    QueuedTranscriptionJobResponse,
    TranscriptionJobLinks,
)
from textify.transcription.jobs import (
    QueuedTranscriptionJob,
    TranscriptionJobCoordinator,
)
from textify.transcription.schemas import ErrorResponse, TranscriptRequest

router = APIRouter(prefix="/api/transcription-jobs")


async def get_transcription_job_coordinator(
    request: Request,
) -> TranscriptionJobCoordinator:
    """Return the lifespan-owned durable job coordinator.

    Args:
        request: Current HTTP request with application state.

    Returns:
        Application-owned queued-job coordinator.

    Raises:
        RuntimeError: If the application lifespan has not initialized the coordinator.
    """
    coordinator = getattr(request.app.state, "transcription_job_coordinator", None)
    if coordinator is None:
        raise RuntimeError(
            "Transcription Job coordinator is unavailable before startup."
        )
    return cast(TranscriptionJobCoordinator, coordinator)


type TranscriptionJobCoordinatorDependency = Annotated[
    TranscriptionJobCoordinator,
    Depends(get_transcription_job_coordinator),
]


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["transcription-jobs"],
    responses={
        status.HTTP_202_ACCEPTED: {
            "description": "Queued Transcription Job accepted after durable commit.",
            "headers": {
                "Location": {
                    "description": "Relative capability URL for this job.",
                    "schema": {"type": "string"},
                },
                "Retry-After": {
                    "description": "Suggested seconds before status inspection.",
                    "schema": {"type": "integer", "example": 2},
                },
                "Cache-Control": {
                    "description": "Bearer capability responses must not be cached.",
                    "schema": {"type": "string", "example": "no-store"},
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Possible codes: `invalid_url`, `unsupported_platform`.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Possible code: `invalid_request`.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": (
                "Possible codes: `transcription_capacity_exceeded`, "
                "`job_store_unavailable`."
            ),
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Possible code: `internal_error`.",
        },
    },
)
async def create_transcription_job(
    request: TranscriptRequest,
    response: Response,
    coordinator: TranscriptionJobCoordinatorDependency,
) -> QueuedTranscriptionJobResponse:
    """Accept one valid Supported Platform URL as a durable queued job.

    Args:
        request: Strict submission payload with immutable response exclusions.
        response: Outgoing response used to publish acceptance headers.
        coordinator: Lifespan-owned durable job coordinator.

    Returns:
        Committed queued-job capability representation.
    """
    queued_job = await coordinator.submit(request.url, frozenset(request.exclude))
    queued_response = _queued_response(queued_job)
    response.headers["Location"] = queued_response.links.self
    response.headers["Retry-After"] = "2"
    response.headers["Cache-Control"] = "no-store"
    return queued_response


@router.get(
    "/{job_id}",
    tags=["transcription-jobs"],
    responses={
        status.HTTP_200_OK: {
            "description": "Current safe representation of a queued job.",
            "headers": {
                "Retry-After": {
                    "description": "Suggested seconds before the next inspection.",
                    "schema": {"type": "integer", "example": 2},
                },
                "Cache-Control": {
                    "description": "Bearer capability responses must not be cached.",
                    "schema": {"type": "string", "example": "no-store"},
                },
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Possible code: `job_not_found`.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Possible code: `job_store_unavailable`.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Possible code: `internal_error`.",
        },
    },
)
async def get_transcription_job(
    job_id: str,
    response: Response,
    coordinator: TranscriptionJobCoordinatorDependency,
) -> QueuedTranscriptionJobResponse:
    """Inspect one queued Transcription Job through its bearer capability.

    Args:
        job_id: Opaque path capability parsed by the coordinator.
        response: Outgoing response used to publish safe caching headers.
        coordinator: Lifespan-owned durable job coordinator.

    Returns:
        Current queued-job representation for the capability.
    """
    queued_job = await coordinator.get_status(job_id)
    response.headers["Retry-After"] = "2"
    response.headers["Cache-Control"] = "no-store"
    return _queued_response(queued_job)


def _queued_response(
    queued_job: QueuedTranscriptionJob,
) -> QueuedTranscriptionJobResponse:
    """Build a public queued response from a safe committed domain snapshot."""
    public_id = str(queued_job.public_id)
    self_link = f"/api/transcription-jobs/{public_id}"
    return QueuedTranscriptionJobResponse(
        id=queued_job.public_id,
        status="queued",
        submitted_at=queued_job.submitted_at,
        links=TranscriptionJobLinks(
            self=self_link,
            cancel=f"{self_link}/cancellation",
        ),
    )
