"""HTTP contract tests for PostgreSQL-backed durable Transcription Jobs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient, Response

from textify.config import AppConfig, Environment
from textify.jobs.celery_app import create_celery_app
from textify.jobs.config import CacheConfig, JobConfig, JobDispatchConfig
from textify.jobs.database import create_application_engine
from textify.jobs.dispatch import CeleryJobDispatchPublisher
from textify.jobs.reconciler import TranscriptionJobReconciler
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.main import create_app
from textify.transcription.schemas import TranscriptionResponse
from textify.transcription.types import Platform, TranscriptMethod

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
    job_retention_seconds: int = 86_400,
) -> JobConfig:
    """Build isolated PostgreSQL lifecycle configuration.

    Args:
        database_url: Disposable migrated PostgreSQL database URL.
        maximum_outstanding_jobs: Maximum queued or processing job count.
        job_retention_seconds: Terminal capability lifetime in seconds.
    Returns:
        Durable job configuration independent of environment files.
    """
    return JobConfig(
        database_url=database_url,  # type: ignore[arg-type]
        max_outstanding_jobs=maximum_outstanding_jobs,
        job_queue_timeout_seconds=20,
        job_retention_seconds=job_retention_seconds,
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


@pytest.mark.asyncio
async def test_api_retention_hides_expired_terminal_job_and_preserves_active_jobs(
    migrated_postgresql_database_url: str,
) -> None:
    """Hide only terminal capabilities that reach their retention cutoff."""
    initial_clock = datetime(2026, 8, 24, tzinfo=UTC)
    retention = timedelta(seconds=10)
    clock = [initial_clock]
    application = create_app(
        _app_config(),
        job_config=_job_config(
            migrated_postgresql_database_url,
            job_retention_seconds=int(retention.total_seconds()),
        ),
        dispatch_config=_dispatch_config(),
        cache_config=_cache_config(),
        clock=lambda: clock[0],
    )
    engine = create_application_engine(migrated_postgresql_database_url)
    repository = PostgresTranscriptionJobRepository(
        engine,
        maximum_outstanding_jobs=8,
        queue_timeout=timedelta(seconds=20),
        terminal_retention=retention,
        clock=lambda: clock[0],
    )
    reconciler = TranscriptionJobReconciler(
        repository,
        CeleryJobDispatchPublisher(create_celery_app(_dispatch_config())),
    )
    projected_result: TranscriptionResponse = {
        "source": {
            "platform": Platform.YOUTUBE,
            "video_id": "AbCdEf12345",
            "url": "https://www.youtube.com/watch?v=AbCdEf12345",
            "title": "Retention result",
            "description": "",
            "channel": "Textify",
            "duration_seconds": 1,
        },
        "transcript": {
            "method": TranscriptMethod.YOUTUBE_CAPTIONS,
            "language": "en",
            "text": "Retention result.",
            "segments": ({"start": 0.0, "end": 1.0, "text": "Retention result."},),
        },
    }
    try:
        async with (
            application.router.lifespan_context(application),
            AsyncClient(
                transport=ASGITransport(app=application),
                base_url="http://test",
            ) as client,
        ):
            terminal_submission = await client.post(
                "/api/transcription-jobs",
                json={"url": _TIKTOK_URL},
            )
            terminal_location = terminal_submission.headers["Location"]
            assert terminal_submission.status_code == 202

            terminal_lease_owner = uuid4()
            terminal_dispatches = await repository.lease_due_dispatches(
                terminal_lease_owner,
                timedelta(seconds=1),
                limit=1,
            )
            assert len(terminal_dispatches) == 1
            terminal_dispatch = terminal_dispatches[0]
            assert await repository.record_dispatch_published(
                terminal_dispatch,
                terminal_lease_owner,
                timedelta(seconds=1),
            )
            terminal_claim = await repository.claim_dispatch(
                terminal_dispatch,
                uuid4(),
                timedelta(seconds=30),
            )
            assert terminal_claim is not None
            assert await repository.publish_success(
                terminal_claim.claim,
                projected_result,
            )

            complete_terminal = await client.get(terminal_location)
            assert complete_terminal.status_code == 200
            assert complete_terminal.json()["status"] == "finished"
            assert complete_terminal.json()["outcome"] == "succeeded"
            assert complete_terminal.json()["result"] == {
                "source": {
                    "platform": "youtube",
                    "video_id": "AbCdEf12345",
                    "url": "https://www.youtube.com/watch?v=AbCdEf12345",
                    "title": "Retention result",
                    "description": "",
                    "channel": "Textify",
                    "duration_seconds": 1,
                },
                "transcript": {
                    "method": "youtube_captions",
                    "language": "en",
                    "text": "Retention result.",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "Retention result.",
                        }
                    ],
                },
            }

            clock[0] = initial_clock + retention - timedelta(microseconds=1)
            processing_submission = await client.post(
                "/api/transcription-jobs",
                json={"url": f"{_TIKTOK_URL}?job=processing"},
            )
            processing_location = processing_submission.headers["Location"]
            assert processing_submission.status_code == 202

            processing_lease_owner = uuid4()
            processing_dispatches = await repository.lease_due_dispatches(
                processing_lease_owner,
                timedelta(seconds=1),
                limit=1,
            )
            assert len(processing_dispatches) == 1
            processing_dispatch = processing_dispatches[0]
            assert await repository.record_dispatch_published(
                processing_dispatch,
                processing_lease_owner,
                timedelta(seconds=1),
            )
            processing_claim = await repository.claim_dispatch(
                processing_dispatch,
                uuid4(),
                timedelta(seconds=30),
            )
            assert processing_claim is not None

            queued_submission = await client.post(
                "/api/transcription-jobs",
                json={"url": f"{_TIKTOK_URL}?job=queued"},
            )
            queued_location = queued_submission.headers["Location"]
            assert queued_submission.status_code == 202

            await reconciler.cleanup_once()
            before_cutoff = await asyncio.gather(
                client.get(terminal_location),
                client.get(processing_location),
                client.get(queued_location),
            )
            assert [response.status_code for response in before_cutoff] == [
                200,
                200,
                200,
            ]

            clock[0] = initial_clock + retention
            await reconciler.cleanup_once()
            expired_terminal = await client.get(terminal_location)
            processing = await client.get(processing_location)
            queued = await client.get(queued_location)
    finally:
        await engine.dispose()

    assert expired_terminal.status_code == 404
    assert _error_code(expired_terminal) == "job_not_found"
    assert processing.status_code == 200
    assert processing.json()["status"] == "processing"
    assert queued.status_code == 200
    assert queued.json()["status"] == "queued"
