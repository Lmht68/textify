"""Runnable FastAPI composition root for Textify."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
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
from textify.jobs.config import (
    CacheConfig,
    JobConfig,
    JobDispatchConfig,
    validate_redis_role_isolation,
)
from textify.jobs.database import (
    create_application_engine,
    verify_application_database,
)
from textify.jobs.http import TranscriptionJobHeadersMiddleware
from textify.jobs.readiness import (
    ApplicationReadiness,
    PostgresServiceHeartbeatRepository,
    RedisBrokerReadiness,
)
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.router import router as transcription_job_router
from textify.jobs.service import TranscriptionJobService, utc_now
from textify.logging import configure_logging, request_log_context
from textify.ops import router as operations_router

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


def create_app(
    app_config: AppConfig | None = None,
    *,
    job_config: JobConfig | None = None,
    dispatch_config: JobDispatchConfig | None = None,
    cache_config: CacheConfig | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> FastAPI:
    """Create the configured PostgreSQL-backed Textify FastAPI application.

    Args:
        app_config: Optional application configuration for composition or tests.
        job_config: Optional durable-job configuration for composition or tests.
        dispatch_config: Optional broker configuration used only for readiness.
        cache_config: Optional validation-only future cache configuration.
        clock: UTC clock supplied to durable job storage.

    Returns:
        Unstarted FastAPI application with durable admission and full topology
        readiness checks.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        """Create durable admission and aggregate readiness state before traffic."""
        application.state.ready = False
        application.state.readiness = None
        resolved_app_config = app_config if app_config is not None else AppConfig()
        resolved_job_config = (
            job_config if job_config is not None else JobConfig()  # type: ignore[call-arg]
        )
        resolved_dispatch_config = (
            dispatch_config if dispatch_config is not None else JobDispatchConfig()  # type: ignore[call-arg]
        )
        resolved_cache_config = (
            cache_config if cache_config is not None else CacheConfig()
        )
        validate_redis_role_isolation(resolved_dispatch_config, resolved_cache_config)

        configure_logging(resolved_app_config.log_level)
        engine = create_application_engine(str(resolved_job_config.database_url))
        broker_readiness: RedisBrokerReadiness | None = None
        try:
            await verify_application_database(engine)
            repository = PostgresTranscriptionJobRepository(
                engine=engine,
                maximum_outstanding_jobs=resolved_job_config.max_outstanding_jobs,
                queue_timeout=timedelta(
                    seconds=resolved_job_config.job_queue_timeout_seconds
                ),
                terminal_retention=timedelta(
                    seconds=resolved_job_config.job_retention_seconds
                ),
                clock=clock,
            )

            def mark_application_unready() -> None:
                """Make readiness fail after a durable-store failure."""
                application.state.ready = False

            application.state.transcription_job_service = TranscriptionJobService(
                repository,
                mark_application_unready,
            )
            heartbeat_repository = PostgresServiceHeartbeatRepository(engine)
            broker_readiness = RedisBrokerReadiness(resolved_dispatch_config)
            application.state.readiness = ApplicationReadiness(
                engine,
                heartbeat_repository,
                broker_readiness,
                timedelta(
                    seconds=resolved_dispatch_config.service_heartbeat_ttl_seconds
                ),
            )
            application.state.ready = True
            try:
                yield
            finally:
                application.state.ready = False
        finally:
            application.state.readiness = None
            if broker_readiness is not None:
                await broker_readiness.aclose()
            await engine.dispose()

    application = FastAPI(
        title="Textify",
        description=(
            "Durable Transcription Job acceptance, bearer-capability status inspection, "
            "and cancellation for eligible public YouTube, Facebook, Instagram, TikTok, "
            "and X videos."
        ),
        version="0.1.0",
        openapi_tags=[
            {
                "name": "health",
                "description": (
                    "Readiness across PostgreSQL, the Redis broker, and fresh "
                    "reconciler and model-ready GPU-worker service heartbeats."
                ),
            },
            {
                "name": "transcription-jobs",
                "description": (
                    "Durable queued Transcription Job acceptance, bearer-capability "
                    "status inspection, and cancellation."
                ),
            },
        ],
        lifespan=lifespan,
    )
    application.state.ready = False
    application.state.readiness = None
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
