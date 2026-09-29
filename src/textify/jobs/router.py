"""FastAPI routes for durable Transcription Jobs."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Path, Request, Response, status

from textify.errors import ErrorDetail, ErrorResponse, InvalidRequestError
from textify.jobs.schemas import (
    ActiveTranscriptionJobLinks,
    CancelledTranscriptionJobResponse,
    FailedTranscriptionJobResponse,
    FinishedTranscriptionJobLinks,
    ProcessingTranscriptionJobResponse,
    QueuedTranscriptionJobResponse,
    SucceededTranscriptionJobResponse,
    TranscriptionJobResponse,
)
from textify.jobs.service import TranscriptionJobService
from textify.jobs.types import (
    CancelledTranscriptionJob,
    FailedTranscriptionJob,
    ProcessingTranscriptionJob,
    QueuedTranscriptionJob,
    SucceededTranscriptionJob,
    TranscriptionJob,
)
from textify.transcription.schemas import TranscriptRequest

router = APIRouter(prefix="/api/transcription-jobs")


async def get_transcription_job_service(
    request: Request,
) -> TranscriptionJobService:
    """Return the lifespan-owned durable job service.

    Args:
        request: Current HTTP request with application state.

    Returns:
        Application-owned durable job service.

    Raises:
        RuntimeError: If the application lifespan has not initialized the service.
    """
    service = getattr(request.app.state, "transcription_job_service", None)
    if service is None:
        raise RuntimeError("Transcription Job service is unavailable before startup.")
    return cast(TranscriptionJobService, service)


type TranscriptionJobServiceDependency = Annotated[
    TranscriptionJobService,
    Depends(get_transcription_job_service),
]

JobCapability = Annotated[
    str,
    Path(
        description=(
            "Opaque lowercase UUIDv4 bearer capability. Possession authorizes "
            "same-origin status inspection and cancellation. Malformed and "
            "noncanonical values return `404 job_not_found`."
        )
    ),
]

_REQUEST_ID_HEADER = {
    "description": "Fresh server-generated identifier for this HTTP request.",
    "schema": {"type": "string", "format": "uuid"},
}
_CACHE_CONTROL_HEADER = {
    "description": "Bearer capability responses are never cacheable.",
    "schema": {"type": "string", "example": "no-store"},
}
_RETRY_AFTER_HEADER = {
    "description": "Present only while work is active; seconds before status inspection.",
    "schema": {"type": "integer", "example": 2},
}
_JOB_RESPONSE_HEADERS = {
    "X-Request-ID": _REQUEST_ID_HEADER,
    "Cache-Control": _CACHE_CONTROL_HEADER,
}
_ACTIVE_JOB_RESPONSE_HEADERS = {
    **_JOB_RESPONSE_HEADERS,
    "Retry-After": _RETRY_AFTER_HEADER,
}


async def _require_empty_request_body(request: Request) -> None:
    """Reject nonempty cancellation bodies before durable lifecycle mutation.

    Args:
        request: Current cancellation action request.

    Raises:
        InvalidRequestError: If any request-body byte is present.
    """
    async for chunk in request.stream():
        if chunk:
            raise InvalidRequestError()


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["transcription-jobs"],
    summary="Submit a durable Transcription Job",
    description=(
        "Commit eligible work before returning a same-origin relative `Location` "
        "and JSON links. The links are bearer capabilities: possession authorizes "
        "polling and cancellation, so clients must not disclose them."
    ),
    responses={
        status.HTTP_202_ACCEPTED: {
            "description": "Queued Transcription Job accepted after durable commit.",
            "headers": {
                **_ACTIVE_JOB_RESPONSE_HEADERS,
                "Location": {
                    "description": (
                        "Same-origin relative status URL for the new bearer capability."
                    ),
                    "schema": {"type": "string"},
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Possible codes: `invalid_url`, `unsupported_platform`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Possible code: `invalid_request`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": (
                "Possible codes: `transcription_capacity_exceeded`, "
                "`job_store_unavailable`."
            ),
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Possible code: `internal_error`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
    },
)
async def create_transcription_job(
    request: TranscriptRequest,
    response: Response,
    service: TranscriptionJobServiceDependency,
) -> QueuedTranscriptionJobResponse:
    """Accept one valid Supported Platform URL as a durable queued job.

    Args:
        request: Strict submission payload with immutable response exclusions.
        response: Outgoing response used to publish acceptance headers.
        service: Lifespan-owned durable job service.

    Returns:
        Committed queued-job capability representation.
    """
    queued_job = await service.submit(request.url, frozenset(request.exclude))
    queued_response = _queued_response(queued_job)
    response.headers["Location"] = queued_response.links.self
    response.headers["Retry-After"] = "2"
    return queued_response


@router.get(
    "/{job_id}",
    tags=["transcription-jobs"],
    response_model=TranscriptionJobResponse,
    summary="Inspect a Transcription Job",
    description=(
        "Poll the same-origin relative bearer capability returned by submission. "
        "Responses are `queued`, `processing`, or `finished`; finished outcomes are "
        "`succeeded`, `failed`, or `cancelled`. Provider, timeout, `queue_timeout`, "
        "`worker_interrupted`, and `internal_error` failures are terminal failed-job "
        "reasons in HTTP `200` responses, not separate HTTP failures."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Current safe representation of a Transcription Job.",
            "headers": _ACTIVE_JOB_RESPONSE_HEADERS,
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Possible code: `job_not_found`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Possible code: `invalid_request`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Possible code: `job_store_unavailable`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Possible code: `internal_error`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
    },
)
async def get_transcription_job(
    job_id: JobCapability,
    response: Response,
    service: TranscriptionJobServiceDependency,
) -> TranscriptionJobResponse:
    """Inspect one Transcription Job through its bearer capability.

    Args:
        job_id: Opaque path capability parsed by the service.
        response: Outgoing response used to publish safe caching headers.
        service: Lifespan-owned durable job service.

    Returns:
        Current typed job representation for the capability.
    """
    job = await service.get_status(job_id)
    job_response = _job_response(job)
    if isinstance(job, (QueuedTranscriptionJob, ProcessingTranscriptionJob)):
        response.headers["Retry-After"] = "2"
    return job_response


@router.put(
    "/{job_id}/cancellation",
    tags=["transcription-jobs"],
    response_model=None,
    summary="Cancel a Transcription Job",
    description=(
        "Submit no request body to accept cancellation through the same-origin "
        "bearer capability. A queued job becomes cancelled with `200`; a processing "
        "job returns `202` while owned resources finish cleanup."
    ),
    responses={
        status.HTTP_200_OK: {
            "model": CancelledTranscriptionJobResponse,
            "description": "Cancelled Transcription Job after queued or completed cancellation.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_202_ACCEPTED: {
            "model": ProcessingTranscriptionJobResponse,
            "description": "Processing cancellation durably accepted.",
            "headers": _ACTIVE_JOB_RESPONSE_HEADERS,
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Possible code: `job_not_found`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "Possible code: `job_already_finished`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Possible code: `invalid_request`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Possible code: `job_store_unavailable`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Possible code: `internal_error`.",
            "headers": _JOB_RESPONSE_HEADERS,
        },
    },
)
async def cancel_transcription_job(
    job_id: JobCapability,
    request: Request,
    response: Response,
    service: TranscriptionJobServiceDependency,
) -> ProcessingTranscriptionJobResponse | CancelledTranscriptionJobResponse:
    """Accept cancellation for one queued or processing Transcription Job.

    Args:
        job_id: Opaque path capability parsed by the service.
        request: Raw request used to reject every nonempty body.
        response: Outgoing response used to publish processing retry guidance.
        service: Lifespan-owned durable job service.

    Returns:
        Current processing or cancelled terminal representation for the capability.
    """
    await _require_empty_request_body(request)
    job = await service.cancel(job_id)
    if isinstance(job, ProcessingTranscriptionJob):
        response.status_code = status.HTTP_202_ACCEPTED
        response.headers["Retry-After"] = "2"
        return _processing_response(job)
    return _cancelled_response(job)


def _job_response(job: TranscriptionJob) -> TranscriptionJobResponse:
    """Build a typed public representation from one safe domain snapshot."""
    if isinstance(job, QueuedTranscriptionJob):
        return _queued_response(job)
    if isinstance(job, ProcessingTranscriptionJob):
        return _processing_response(job)
    if isinstance(job, SucceededTranscriptionJob):
        return _succeeded_response(job)
    if isinstance(job, FailedTranscriptionJob):
        return _failed_response(job)
    if isinstance(job, CancelledTranscriptionJob):
        return _cancelled_response(job)
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


def _failed_response(
    failed_job: FailedTranscriptionJob,
) -> FailedTranscriptionJobResponse:
    """Build a safe public representation of a failed finished job."""
    self_link = _self_link(failed_job.public_id)
    return FailedTranscriptionJobResponse(
        id=failed_job.public_id,
        status="finished",
        outcome="failed",
        submitted_at=failed_job.submitted_at,
        started_at=failed_job.started_at,
        finished_at=failed_job.finished_at,
        error=ErrorDetail(
            code=failed_job.error_code,
            message=failed_job.error_message,
        ),
        links=FinishedTranscriptionJobLinks(self=self_link),
    )


def _cancelled_response(
    cancelled_job: CancelledTranscriptionJob,
) -> CancelledTranscriptionJobResponse:
    """Build a safe public representation of a cancelled finished job."""
    self_link = _self_link(cancelled_job.public_id)
    return CancelledTranscriptionJobResponse(
        id=cancelled_job.public_id,
        status="finished",
        outcome="cancelled",
        submitted_at=cancelled_job.submitted_at,
        started_at=cancelled_job.started_at,
        finished_at=cancelled_job.finished_at,
        links=FinishedTranscriptionJobLinks(self=self_link),
    )


def _self_link(public_id: object) -> str:
    """Build one relative bearer-capability status URL."""
    return f"/api/transcription-jobs/{public_id}"
