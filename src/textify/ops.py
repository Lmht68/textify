"""Operational HTTP endpoints for the application."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Represent a successful application health check."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "ok"}]})

    status: Literal["ok"] = Field(
        default="ok",
        description="Application readiness state after successful startup.",
    )


router = APIRouter()

_REQUEST_ID_HEADER = {
    "description": "Fresh server-generated identifier for this HTTP request.",
    "schema": {"type": "string", "format": "uuid"},
}


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    response_model=HealthResponse,
    summary="Check application readiness",
    description=(
        "Return success only while lifespan startup completed and durable "
        "Transcription Job storage remains available."
    ),
    response_description="The application is ready to serve Transcription Jobs.",
    tags=["health"],
    responses={
        status.HTTP_200_OK: {"headers": {"X-Request-ID": _REQUEST_ID_HEADER}},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": (
                "Startup is incomplete or a runtime durable-store failure made "
                "the application unready."
            ),
            "headers": {"X-Request-ID": _REQUEST_ID_HEADER},
        },
    },
)
async def health(request: Request) -> HealthResponse:
    """Return readiness after lifespan-created dependencies are available.

    Args:
        request: Current HTTP request.

    Returns:
        Stable ready payload.

    Raises:
        HTTPException: If startup has not completed.
    """
    if not bool(getattr(request.app.state, "ready", False)):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return HealthResponse()
