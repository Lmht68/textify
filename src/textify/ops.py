"""Operational HTTP endpoints for the application."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Represent a successful application health check."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "ok"}]})

    status: Literal["ok"] = Field(
        default="ok",
        description="Application topology readiness after successful startup.",
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
        "Return success only while PostgreSQL, the Redis broker, and fresh "
        "reconciler and model-ready GPU-worker service heartbeats are ready."
    ),
    response_description="The application processing topology is ready.",
    tags=["health"],
    responses={
        status.HTTP_200_OK: {"headers": {"X-Request-ID": _REQUEST_ID_HEADER}},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": (
                "Startup is incomplete or a required processing dependency is unready."
            ),
            "headers": {"X-Request-ID": _REQUEST_ID_HEADER},
        },
    },
)
async def health(request: Request) -> HealthResponse:
    """Return full processing-topology readiness without dependency details.

    Args:
        request: Current HTTP request.

    Returns:
        Stable ready payload.

    Raises:
        HTTPException: If startup is incomplete or any required dependency is unready.
    """
    readiness = getattr(request.app.state, "readiness", None)
    if (
        not bool(getattr(request.app.state, "ready", False))
        or readiness is None
        or not await readiness.is_ready()
    ):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return HealthResponse()
