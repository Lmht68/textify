"""HTTP contract tests for PostgreSQL-backed durable Transcription Jobs."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient, Response

from textify.config import AppConfig, Environment
from textify.jobs.config import CacheConfig, JobConfig, JobDispatchConfig
from textify.main import create_app

_TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"


def _app_config() -> AppConfig:
    """Build isolated FastAPI configuration.

    Returns:
        Application configuration independent of environment files.
    """
    return AppConfig(
        environment=Environment.LOCAL,
        log_level="INFO",
        host="127.0.0.1",
        port=8182,
    )


def _job_config(
    database_url: str,
    *,
    maximum_outstanding_jobs: int = 8,
) -> JobConfig:
    """Build isolated PostgreSQL lifecycle configuration.

    Args:
        database_url: Disposable migrated PostgreSQL database URL.
        maximum_outstanding_jobs: Maximum queued or processing job count.

    Returns:
        Durable job configuration independent of environment files.
    """
    return JobConfig(
        database_url=database_url,  # type: ignore[arg-type]
        max_outstanding_jobs=maximum_outstanding_jobs,
        job_queue_timeout_seconds=20,
        job_retention_seconds=86_400,
        _env_file=None,  # type: ignore[call-arg]
    )


def _dispatch_config() -> JobDispatchConfig:
    """Build unreachable Redis settings for isolated API lifecycle tests.

    Returns:
        Broker configuration that keeps health unready without blocking admission.
    """
    return JobDispatchConfig(
        broker_url="redis://127.0.0.1:1/15",  # type: ignore[arg-type]
        _env_file=None,  # type: ignore[call-arg]
    )


def _cache_config() -> CacheConfig:
    """Build absent cache settings for isolated API lifecycle tests.

    Returns:
        Validation-only cache configuration with no endpoint.
    """
    return CacheConfig(_env_file=None)  # type: ignore[call-arg]


@asynccontextmanager
async def _client_for(
    database_url: str,
    *,
    maximum_outstanding_jobs: int = 8,
) -> AsyncGenerator[AsyncClient, None]:
    """Yield a started PostgreSQL-only application client.

    Args:
        database_url: Disposable migrated PostgreSQL database URL.
        maximum_outstanding_jobs: Maximum queued or processing job count.

    Yields:
        Client connected to the started FastAPI application.
    """
    application = create_app(
        _app_config(),
        job_config=_job_config(
            database_url,
            maximum_outstanding_jobs=maximum_outstanding_jobs,
        ),
        dispatch_config=_dispatch_config(),
        cache_config=_cache_config(),
    )
    async with (
        application.router.lifespan_context(application),
        AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as client,
    ):
        yield client


def _error_code(response: Response) -> str:
    """Read one stable public error code.

    Args:
        response: Error response returned by the HTTP contract.

    Returns:
        Stable machine-readable public error code.
    """
    payload = response.json()
    assert isinstance(payload, dict)
    error = payload["error"]
    assert isinstance(error, dict)
    code = error["code"]
    assert isinstance(code, str)
    return code


@pytest.mark.asyncio
async def test_api_admits_queued_job_with_bearer_capability_headers(
    migrated_postgresql_database_url: str,
) -> None:
    """Commit queued work before returning its relative bearer capability."""
    async with _client_for(migrated_postgresql_database_url) as client:
        response = await client.post(
            "/api/transcription-jobs",
            json={"url": _TIKTOK_URL},
        )
        body = response.json()
        status = await client.get(response.headers["Location"])

    assert response.status_code == 202
    assert response.headers["Location"] == body["links"]["self"]
    assert response.headers["Retry-After"] == "2"
    assert response.headers["Cache-Control"] == "no-store"
    assert body["status"] == "queued"
    assert body["links"]["cancel"] == f"{body['links']['self']}/cancellation"
    assert status.status_code == 200
    assert status.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_api_cancels_queued_job_without_worker_dependencies(
    migrated_postgresql_database_url: str,
) -> None:
    """Persist queued cancellation directly through the API's PostgreSQL boundary."""
    async with _client_for(migrated_postgresql_database_url) as client:
        submission = await client.post(
            "/api/transcription-jobs",
            json={"url": _TIKTOK_URL},
        )
        cancellation = await client.put(
            f"{submission.headers['Location']}/cancellation",
        )
        status = await client.get(submission.headers["Location"])

    assert cancellation.status_code == 200
    assert cancellation.json()["status"] == "finished"
    assert cancellation.json()["outcome"] == "cancelled"
    assert status.json()["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_api_rejects_invalid_capabilities_and_nonempty_cancellation(
    migrated_postgresql_database_url: str,
) -> None:
    """Preserve capability opacity and cancellation's empty-body contract."""
    async with _client_for(migrated_postgresql_database_url) as client:
        malformed = await client.get("/api/transcription-jobs/not-a-capability")
        unknown = await client.get(f"/api/transcription-jobs/{uuid4()}")
        submission = await client.post(
            "/api/transcription-jobs",
            json={"url": _TIKTOK_URL},
        )
        nonempty_cancellation = await client.put(
            f"{submission.headers['Location']}/cancellation",
            content=b"not-empty",
        )

    assert malformed.status_code == 404
    assert _error_code(malformed) == "job_not_found"
    assert unknown.status_code == 404
    assert _error_code(unknown) == "job_not_found"
    assert nonempty_cancellation.status_code == 422
    assert _error_code(nonempty_cancellation) == "invalid_request"


@pytest.mark.asyncio
async def test_api_rejects_invalid_requests_and_durable_capacity(
    migrated_postgresql_database_url: str,
) -> None:
    """Keep request validation and PostgreSQL admission capacity stable."""
    async with _client_for(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
    ) as client:
        invalid_request = await client.post(
            "/api/transcription-jobs",
            json={"url": _TIKTOK_URL, "unknown": "value"},
        )
        first_submission = await client.post(
            "/api/transcription-jobs",
            json={"url": _TIKTOK_URL},
        )
        capacity_rejected = await client.post(
            "/api/transcription-jobs",
            json={"url": _TIKTOK_URL},
        )

    assert invalid_request.status_code == 422
    assert _error_code(invalid_request) == "invalid_request"
    assert first_submission.status_code == 202
    assert capacity_rejected.status_code == 503
    assert _error_code(capacity_rejected) == "transcription_capacity_exceeded"
