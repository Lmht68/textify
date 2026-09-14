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


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    response_model=HealthResponse,
    summary="Check application readiness",
    description="Return success only after lifespan startup has completed.",
    response_description="The application is ready to serve transcript requests.",
    tags=["health"],
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
