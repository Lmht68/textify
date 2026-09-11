"""Runnable FastAPI composition root for Textify."""

import shutil
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from textify.config import AppConfig
from textify.logging import configure_logging
from textify.ops import router as operations_router
from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import (
    TranscriptionError,
    transcription_error_handler,
    unhandled_error_handler,
)
from textify.transcription.router import router as transcription_router
from textify.transcription.service import (
    TranscriptionAdaptersFactory,
    TranscriptService,
    build_transcription_adapters,
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
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "The request is invalid.",
            }
        },
    )


def _available_temporary_media_bytes(root: Path) -> int:
    """Return currently available space in the temporary-media filesystem."""
    try:
        return shutil.disk_usage(root).free
    except OSError as exc:
        raise RuntimeError(
            "Unable to determine temporary media root capacity."
        ) from exc


def _validate_temporary_media_capacity(
    settings: TranscriptionConfig,
    *,
    available_bytes: Callable[[Path], int],
) -> None:
    """Reject startup when temporary media cannot hold the configured quota."""
    required_bytes = (
        settings.max_pending_transcriptions + settings.transcription_concurrency + 1
    ) * settings.max_media_bytes
    try:
        available_media_bytes = available_bytes(settings.temporary_media_root)
    except OSError as exc:
        raise RuntimeError(
            "Unable to determine temporary media root capacity."
        ) from exc
    if available_media_bytes < required_bytes:
        raise RuntimeError(
            "Temporary media root requires "
            f"{required_bytes} available bytes; "
            f"{available_media_bytes} are available."
        )


def create_app(
    app_config: AppConfig | None = None,
    transcription_config: TranscriptionConfig | None = None,
    adapters_factory: TranscriptionAdaptersFactory = build_transcription_adapters,
    available_temporary_media_bytes: Callable[
        [Path], int
    ] = _available_temporary_media_bytes,
) -> FastAPI:
    """Create the configured Textify FastAPI application.

    Args:
        app_config: Optional application configuration for composition or tests.
        transcription_config: Optional transcription configuration for composition or tests.
        adapters_factory: Factory that creates the complete provider adapter bundle.
        available_temporary_media_bytes: Reader for current writable media capacity.

    Returns:
        Unstarted FastAPI application with startup-owned model lifecycle.
    """
    resolved_app_config = app_config if app_config is not None else AppConfig()
    resolved_transcription_config = (
        transcription_config
        if transcription_config is not None
        else TranscriptionConfig()
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        """Create process-lifetime state before accepting traffic."""
        configure_logging(resolved_app_config.log_level)
        resolved_transcription_config.temporary_media_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        _validate_temporary_media_capacity(
            resolved_transcription_config,
            available_bytes=available_temporary_media_bytes,
        )
        adapters = adapters_factory(resolved_transcription_config)
        service = TranscriptService(
            adapters,
            resolved_transcription_config,
        )
        application.state.transcript_service = service
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False
            await service.shutdown()

    application = FastAPI(
        title="Textify",
        description="Synchronous transcripts for supported public source videos.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.ready = False
    application.add_exception_handler(TranscriptionError, transcription_error_handler)
    application.add_exception_handler(
        RequestValidationError, request_validation_error_handler
    )
    application.add_exception_handler(Exception, unhandled_error_handler)
    application.include_router(operations_router)
    application.include_router(transcription_router)
    return application


app = create_app()


def main() -> None:
    """Start one Textify Uvicorn worker using environment configuration."""
    config = AppConfig()
    uvicorn.run(
        "textify.main:app",
        host=config.host,
        port=config.port,
        workers=1,
    )
