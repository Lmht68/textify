"""FastAPI routes for durable Transcription Jobs."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, Response, status

from textify.transcription.job_schemas import (
    ActiveTranscriptionJobLinks,
    FinishedTranscriptionJobLinks,
    ProcessingTranscriptionJobResponse,
    QueuedTranscriptionJobResponse,
    SucceededTranscriptionJobResponse,
    TranscriptionJobResponse,
)
from textify.transcription.jobs import (
    ProcessingTranscriptionJob,
    QueuedTranscriptionJob,
    SucceededTranscriptionJob,
    TranscriptionJob,
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
        Application-owned durable job coordinator.

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
    response_model=TranscriptionJobResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Current safe representation of a Transcription Job.",
            "headers": {
                "Retry-After": {
                    "description": "Suggested seconds before active-job inspection.",
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
) -> TranscriptionJobResponse:
    """Inspect one Transcription Job through its bearer capability.

    Args:
        job_id: Opaque path capability parsed by the coordinator.
        response: Outgoing response used to publish safe caching headers.
        coordinator: Lifespan-owned durable job coordinator.

    Returns:
        Current typed job representation for the capability.
    """
    job = await coordinator.get_status(job_id)
    job_response = _job_response(job)
    if isinstance(job, (QueuedTranscriptionJob, ProcessingTranscriptionJob)):
        response.headers["Retry-After"] = "2"
    response.headers["Cache-Control"] = "no-store"
    return job_response


def _job_response(job: TranscriptionJob) -> TranscriptionJobResponse:
    """Build a typed public representation from one safe domain snapshot."""
    if isinstance(job, QueuedTranscriptionJob):
        return _queued_response(job)
    if isinstance(job, ProcessingTranscriptionJob):
        return _processing_response(job)
    if isinstance(job, SucceededTranscriptionJob):
        return _succeeded_response(job)
    raise RuntimeError("Unknown Transcription Job snapshot.")


def _queued_response(
    queued_job: QueuedTranscriptionJob,
) -> QueuedTranscriptionJobResponse:
    """Build a public queued response from a safe committed domain snapshot."""
    self_link = _self_link(queued_job.public_id)
    return QueuedTranscriptionJobResponse(
        id=queued_job.public_id,
        status="queued",
        submitted_at=queued_job.submitted_at,
        links=ActiveTranscriptionJobLinks(
            self=self_link,
            cancel=f"{self_link}/cancellation",
        ),
    )


def _processing_response(
    processing_job: ProcessingTranscriptionJob,
) -> ProcessingTranscriptionJobResponse:
    """Build a public processing response from a safe domain snapshot."""
    self_link = _self_link(processing_job.public_id)
    return ProcessingTranscriptionJobResponse(
        id=processing_job.public_id,
        status="processing",
        submitted_at=processing_job.submitted_at,
        started_at=processing_job.started_at,
        cancellation_requested=processing_job.cancellation_requested,
        links=ActiveTranscriptionJobLinks(
            self=self_link,
            cancel=f"{self_link}/cancellation",
        ),
    )


def _succeeded_response(
    succeeded_job: SucceededTranscriptionJob,
) -> SucceededTranscriptionJobResponse:
    """Build a complete public successful response from a safe domain snapshot."""
    self_link = _self_link(succeeded_job.public_id)
    return SucceededTranscriptionJobResponse(
        id=succeeded_job.public_id,
        status="finished",
        outcome="succeeded",
        submitted_at=succeeded_job.submitted_at,
        started_at=succeeded_job.started_at,
        finished_at=succeeded_job.finished_at,
        result=succeeded_job.result,
        links=FinishedTranscriptionJobLinks(self=self_link),
    )


def _self_link(public_id: object) -> str:
    """Build one relative bearer-capability status URL."""
    return f"/api/transcription-jobs/{public_id}"
