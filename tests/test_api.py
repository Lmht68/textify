"""ASGI proofs for PostgreSQL-only FastAPI lifecycle ownership."""

from pathlib import Path
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient, Response
from pydantic import ValidationError
from sqlalchemy import text

from textify.config import AppConfig, Environment
from textify.jobs.config import CacheConfig, JobConfig, JobDispatchConfig
from textify.jobs.database import create_application_engine
from textify.main import create_app

TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"


def _app_config() -> AppConfig:
    """Create isolated application configuration for ASGI tests.

    Returns:
        Configuration independent of local environment files.
    """
    return AppConfig(
        environment=Environment.LOCAL,
        log_level="INFO",
        host="127.0.0.1",
        port=8182,
    )


def _job_config(database_url: str, *, maximum_outstanding_jobs: int = 8) -> JobConfig:
    """Create isolated PostgreSQL job configuration for ASGI tests.

    Args:
        database_url: Per-test PostgreSQL database upgraded through Alembic.
        maximum_outstanding_jobs: Maximum concurrent queued or processing jobs.

    Returns:
        Configuration independent of local environment files.
    """
    return JobConfig(
        database_url=database_url,  # type: ignore[arg-type]
        max_outstanding_jobs=maximum_outstanding_jobs,
        job_queue_timeout_seconds=20,
        job_retention_seconds=86_400,
        _env_file=None,  # type: ignore[call-arg]
    )


def _dispatch_config(broker_url: str) -> JobDispatchConfig:
    """Create isolated Redis broker settings for ASGI tests.

    Args:
        broker_url: Valid Redis URL, whether or not its server is reachable.

    Returns:
        Dispatch configuration independent of local environment files.
    """
    return JobDispatchConfig(
        broker_url=broker_url,  # type: ignore[arg-type]
        _env_file=None,  # type: ignore[call-arg]
    )


def _cache_config() -> CacheConfig:
    """Create an omitted-cache configuration for ASGI tests.

    Returns:
        Configuration with no cache client or endpoint.
    """
    return CacheConfig(_env_file=None)  # type: ignore[call-arg]


def _assert_generated_request_id(response: Response) -> str:
    """Assert and return the response's server-generated canonical UUIDv4.

    Args:
        response: HTTP response carrying the correlation header.

    Returns:
        Canonical server-generated UUIDv4 request identifier.
    """
    request_id = response.headers["X-Request-ID"]
    parsed_request_id = UUID(request_id)
    assert parsed_request_id.version == 4
    assert str(parsed_request_id) == request_id
    return request_id


def test_app_config_defaults_to_deployment_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the documented deployment port when no environment override exists."""
    monkeypatch.delenv("TEXTIFY_PORT", raising=False)

    settings = AppConfig(_env_file=None)  # type: ignore[call-arg]

    assert settings.port == 8182


@pytest.mark.asyncio
async def test_api_admits_jobs_during_broker_readiness_degradation(
    migrated_postgresql_database_url: str,
) -> None:
    """Keep direct durable admission available while full health reports 503."""
    application = create_app(
        _app_config(),
        job_config=_job_config(migrated_postgresql_database_url),
        dispatch_config=_dispatch_config("redis://127.0.0.1:1/15"),
        cache_config=_cache_config(),
    )

    async with application.router.lifespan_context(application):
        assert application.state.ready is True
        assert not hasattr(application.state, "transcription_job_runner")
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health_response = await client.get("/health")
            submission_response = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL},
            )
            status_response = await client.get(submission_response.headers["Location"])

    assert health_response.status_code == 503
    assert health_response.json() == {"detail": "Service Unavailable"}
    assert _assert_generated_request_id(health_response)
    assert submission_response.status_code == 202
    assert (
        submission_response.headers["Location"]
        == submission_response.json()["links"]["self"]
    )
    assert submission_response.json()["status"] == "queued"
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_api_rejects_unconfigured_database_at_lifespan_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Defer required PostgreSQL validation until the application lifespan begins."""
    monkeypatch.delenv("TEXTIFY_DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    application = create_app(
        _app_config(),
        dispatch_config=_dispatch_config("redis://127.0.0.1:1/15"),
        cache_config=_cache_config(),
    )

    with pytest.raises(ValidationError):
        async with application.router.lifespan_context(application):
            pass

    assert application.state.ready is False


@pytest.mark.asyncio
async def test_api_rejects_unavailable_postgresql() -> None:
    """Map an unreachable PostgreSQL server to the stable startup failure."""
    application = create_app(
        _app_config(),
        job_config=_job_config(
            "postgresql+asyncpg://textify:change-me@127.0.0.1:1/textify"
        ),
        dispatch_config=_dispatch_config("redis://127.0.0.1:1/15"),
        cache_config=_cache_config(),
    )

    with pytest.raises(
        RuntimeError,
        match=r"^Transcription job database is unavailable\.$",
    ):
        async with application.router.lifespan_context(application):
            pass

    assert application.state.ready is False


@pytest.mark.asyncio
async def test_api_rejects_postgresql_without_current_migration(
    postgresql_database_url: str,
) -> None:
    """Map PostgreSQL without the expected Alembic revision to schema failure."""
    storage_engine = create_application_engine(postgresql_database_url)
    try:
        async with storage_engine.begin() as connection:
            await connection.execute(
                text("CREATE TABLE alembic_version (version_num varchar(32) NOT NULL)")
            )
            await connection.execute(
                text("INSERT INTO alembic_version (version_num) VALUES ('0000_stale')")
            )
    finally:
        await storage_engine.dispose()

    application = create_app(
        _app_config(),
        job_config=_job_config(postgresql_database_url),
        dispatch_config=_dispatch_config("redis://127.0.0.1:1/15"),
        cache_config=_cache_config(),
    )

    with pytest.raises(
        RuntimeError,
        match=r"^Transcription job database schema is not current\.$",
    ):
        async with application.router.lifespan_context(application):
            pass

    assert application.state.ready is False


@pytest.mark.asyncio
async def test_api_adds_fresh_request_ids_to_job_route_outcomes(
    migrated_postgresql_database_url: str,
) -> None:
    """Health, job, validation, and unmatched routes get fresh request IDs."""
    inbound_request_id = "inbound-request-id-sentinel"
    application = create_app(
        _app_config(),
        job_config=_job_config(migrated_postgresql_database_url),
        dispatch_config=_dispatch_config("redis://127.0.0.1:1/15"),
        cache_config=_cache_config(),
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health_response = await client.get(
                "/health",
                headers={"X-Request-ID": inbound_request_id},
            )
            submission_response = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL},
                headers={"X-Request-ID": inbound_request_id},
            )
            validation_response = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL, "unknown": "value"},
                headers={"X-Request-ID": inbound_request_id},
            )
            unmatched_response = await client.get(
                "/unmatched",
                headers={"X-Request-ID": inbound_request_id},
            )

    responses = (
        health_response,
        submission_response,
        validation_response,
        unmatched_response,
    )
    request_ids = {_assert_generated_request_id(response) for response in responses}
    assert health_response.status_code == 503
    assert submission_response.status_code == 202
    assert validation_response.status_code == 422
    assert unmatched_response.status_code == 404
    assert len(request_ids) == len(responses)
    assert inbound_request_id not in request_ids
