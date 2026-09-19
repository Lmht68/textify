"""Runnable FastAPI composition root for Textify."""

import asyncio
import logging
import shutil
import tempfile
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from textify.config import AppConfig
from textify.errors import (
    TextifyError,
    request_validation_error_handler,
    textify_error_handler,
    unhandled_error_handler,
)
from textify.jobs.config import JobConfig
from textify.jobs.database import (
    create_application_engine,
    verify_application_database,
)
from textify.jobs.http import TranscriptionJobHeadersMiddleware
from textify.jobs.repository import SqliteTranscriptionJobRepository
from textify.jobs.router import router as transcription_job_router
from textify.jobs.service import TranscriptionJobCoordinator, utc_now
from textify.logging import configure_logging, request_log_context
from textify.ops import router as operations_router
from textify.transcription.config import TranscriptionConfig
from textify.transcription.service import (
    TranscriptionAdaptersFactory,
    TranscriptionExecutor,
    build_transcription_adapters,
)

logger = logging.getLogger(__name__)


class _RequestObservabilityMiddleware:
    """Correlate each HTTP response with safe terminal request logging."""

    def __init__(self, app: ASGIApp) -> None:
        """Store the next application in the ASGI chain.

        Args:
            app: Downstream ASGI application.
        """
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Add a trusted correlation ID and terminal request record.

        Args:
            scope: Current ASGI connection scope.
            receive: ASGI message receiver.
            send: ASGI message sender.
        """
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = str(uuid4())
        request_started_at = time.monotonic()
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            """Add the generated correlation ID to an HTTP response start."""
            nonlocal response_started
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
                response_started = True
            await send(message)

        with request_log_context(request_id):
            try:
                await self._app(scope, receive, send_with_request_id)
            except asyncio.CancelledError:
                if not response_started:
                    logger.info(
                        "request interrupted",
                        extra={
                            "stage": "request",
                            "elapsed_seconds": max(
                                0.0,
                                time.monotonic() - request_started_at,
                            ),
                        },
                    )
                raise
            except Exception as exc:
                if response_started:
                    raise
                response = await unhandled_error_handler(
                    Request(scope, receive=receive),
                    exc,
                )
                await response(scope, receive, send_with_request_id)

            if response_started:
                logger.info(
                    "request completed",
                    extra={
                        "stage": "request",
                        "elapsed_seconds": max(
                            0.0,
                            time.monotonic() - request_started_at,
                        ),
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
    transcription_config: TranscriptionConfig,
    job_worker_count: int,
    *,
    available_bytes: Callable[[Path], int],
) -> None:
    """Reject startup when temporary media cannot hold the configured quota."""
    required_bytes = (
        job_worker_count + transcription_config.transcription_concurrency + 1
    ) * transcription_config.max_media_bytes
    try:
        available_media_bytes = available_bytes(
            transcription_config.temporary_media_root
        )
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


def _ensure_writable_temporary_media_root(root: Path) -> None:
    """Create and verify the temporary media root is writable.

    Args:
        root: Directory reserved for request-scoped media.

    Raises:
        RuntimeError: If the directory cannot be created or used for a temporary file.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root):
            pass
    except OSError as exc:
        raise RuntimeError(
            "Temporary media root must be a writable directory."
        ) from exc


def create_app(
    app_config: AppConfig | None = None,
    transcription_config: TranscriptionConfig | None = None,
    adapters_factory: TranscriptionAdaptersFactory = build_transcription_adapters,
    available_temporary_media_bytes: Callable[
        [Path], int
    ] = _available_temporary_media_bytes,
    *,
    job_config: JobConfig | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> FastAPI:
    """Create the configured Textify FastAPI application.

    Args:
        app_config: Optional application configuration for composition or tests.
        transcription_config: Optional transcription configuration for composition or tests.
        adapters_factory: Factory that creates the complete provider adapter bundle.
        available_temporary_media_bytes: Reader for current writable media capacity.
        job_config: Optional durable-job configuration for composition or tests.
        clock: UTC clock supplied to durable job storage.

    Returns:
        Unstarted FastAPI application with startup-owned model lifecycle.
    """
    resolved_app_config = app_config if app_config is not None else AppConfig()
    resolved_transcription_config = (
        transcription_config
        if transcription_config is not None
        else TranscriptionConfig()
    )
    resolved_job_config = job_config if job_config is not None else JobConfig()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        """Create process-lifetime state before accepting traffic."""
        configure_logging(resolved_app_config.log_level)
        _ensure_writable_temporary_media_root(
            resolved_transcription_config.temporary_media_root
        )
        _validate_temporary_media_capacity(
            resolved_transcription_config,
            resolved_job_config.job_worker_count,
            available_bytes=available_temporary_media_bytes,
        )
        database_path = resolved_job_config.database_path
        if database_path is None:
            raise RuntimeError("TEXTIFY_DATABASE_PATH must be configured.")
        engine = create_application_engine(database_path)
        try:
            await verify_application_database(engine)
            repository = SqliteTranscriptionJobRepository(
                engine=engine,
                maximum_outstanding_jobs=resolved_job_config.max_outstanding_jobs,
                queue_timeout=timedelta(
                    seconds=resolved_job_config.job_queue_timeout_seconds
                ),
                clock=clock,
            )
            adapters = adapters_factory(resolved_transcription_config)
            executor = TranscriptionExecutor(adapters, resolved_transcription_config)
            coordinator = TranscriptionJobCoordinator(
                repository,
                executor,
                resolved_job_config.job_worker_count,
            )
            application.state.transcription_job_coordinator = coordinator
            await coordinator.start()
            application.state.ready = True
            try:
                yield
            finally:
                application.state.ready = False
                await coordinator.shutdown()
        finally:
            await engine.dispose()

    application = FastAPI(
        title="Textify",
        description=(
            "Durable Transcription Job acceptance and status inspection for eligible "
            "public YouTube, Facebook, Instagram, TikTok, and X videos."
        ),
        version="0.1.0",
        openapi_tags=[
            {
                "name": "health",
                "description": (
                    "Readiness after the lifespan has loaded process-owned "
                    "transcription dependencies."
                ),
            },
            {
                "name": "transcription-jobs",
                "description": (
                    "Durable queued Transcription Job acceptance and "
                    "bearer-capability status inspection."
                ),
            },
        ],
        lifespan=lifespan,
    )
    application.state.ready = False
    application.add_middleware(_RequestObservabilityMiddleware)
    application.add_middleware(TranscriptionJobHeadersMiddleware)
    application.add_exception_handler(TextifyError, textify_error_handler)
    application.add_exception_handler(
        RequestValidationError, request_validation_error_handler
    )
    application.include_router(operations_router)
    application.include_router(transcription_job_router)
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
        access_log=False,
    )
