"""ASGI proofs for PostgreSQL-only FastAPI lifecycle ownership."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient, HTTPError, Response
from pydantic import ValidationError
from sqlalchemy import text
from starlette.responses import Response as StarletteResponse
from starlette.types import Message, Receive, Scope, Send

from textify.config import AppConfig, Environment
from textify.jobs.config import CacheConfig, JobConfig, JobDispatchConfig
from textify.jobs.database import create_application_engine
from textify.logging import LogServiceRole, configure_structured_logging
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


@pytest.mark.asyncio
async def test_api_logs_completed_unmatched_requests_with_safe_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit a correlated completion event without logging an unmatched raw path."""
    raw_path_sentinel = "raw-path-sentinel"
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    application = create_app(_app_config())

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/{raw_path_sentinel}?query=raw-query-sentinel")

    request_id = _assert_generated_request_id(response)
    assert response.status_code == 404
    rendered_output = capsys.readouterr().err
    rendered_events = [json.loads(line) for line in rendered_output.splitlines()]
    assert len(rendered_events) == 1
    rendered_event = rendered_events[0]
    assert {
        field_name: rendered_event[field_name]
        for field_name in (
            "event",
            "request_id",
            "http_method",
            "http_status_code",
            "stage",
        )
    } == {
        "event": "request_completed",
        "request_id": request_id,
        "http_method": "GET",
        "http_status_code": 404,
        "stage": "request",
    }
    assert rendered_event["elapsed_seconds"] >= 0
    assert raw_path_sentinel not in rendered_output
    assert "raw-query-sentinel" not in rendered_output


@pytest.mark.asyncio
async def test_api_logs_matched_route_templates_without_capability_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit a matched template without retaining raw path or query values."""
    capability_sentinel = "capability-sentinel"
    query_sentinel = "query-sentinel"
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    application = create_app(_app_config())

    @application.get("/observed/{capability}")
    async def observe_route(capability: str) -> dict[str, bool]:
        return {"accepted": bool(capability)}

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/observed/{capability_sentinel}?query={query_sentinel}"
        )

    request_id = _assert_generated_request_id(response)
    assert response.status_code == 200
    rendered_output = capsys.readouterr().err
    rendered_events = [json.loads(line) for line in rendered_output.splitlines()]
    assert len(rendered_events) == 1
    rendered_event = rendered_events[0]
    assert {
        field_name: rendered_event[field_name]
        for field_name in (
            "event",
            "request_id",
            "http_method",
            "http_status_code",
            "http_route",
            "stage",
        )
    } == {
        "event": "request_completed",
        "request_id": request_id,
        "http_method": "GET",
        "http_status_code": 200,
        "http_route": "/observed/{capability}",
        "stage": "request",
    }
    assert rendered_event["elapsed_seconds"] >= 0
    assert capability_sentinel not in rendered_output
    assert query_sentinel not in rendered_output


@pytest.mark.asyncio
async def test_api_logs_pre_response_cancellation_without_exception_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit an interruption event when a route cancels before response start."""
    capability_sentinel = "pre-response-cancel-sentinel"
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    application = create_app(_app_config())

    @application.get("/cancellation-interrupted/{capability}")
    async def cancel_before_response(capability: str) -> None:
        assert capability == capability_sentinel
        raise asyncio.CancelledError

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(asyncio.CancelledError):
            await client.get(
                f"/cancellation-interrupted/{capability_sentinel}"
                "?query=raw-query-sentinel"
            )

    rendered_output = capsys.readouterr().err
    rendered_events = [json.loads(line) for line in rendered_output.splitlines()]
    assert len(rendered_events) == 1
    rendered_event = rendered_events[0]
    assert {
        field_name: rendered_event[field_name]
        for field_name in ("event", "http_method", "http_route", "stage")
    } == {
        "event": "request_interrupted",
        "http_method": "GET",
        "http_route": "/cancellation-interrupted/{capability}",
        "stage": "request",
    }
    assert rendered_event["elapsed_seconds"] >= 0
    assert "http_status_code" not in rendered_event
    assert capability_sentinel not in rendered_output
    assert "raw-query-sentinel" not in rendered_output
    assert "Traceback" not in rendered_output


@pytest.mark.asyncio
async def test_api_completes_pre_response_failures_with_a_safe_500(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return the safe error response and preserve completion terminal state."""
    capability_sentinel = "pre-response-failure-capability"
    exception_sentinel = "pre-response-failure-exception"
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    application = create_app(_app_config())

    @application.get("/safe-errors/{capability}")
    async def fail_before_response(capability: str) -> None:
        assert capability == capability_sentinel
        raise RuntimeError(exception_sentinel)

    transport = ASGITransport(app=application, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/safe-errors/{capability_sentinel}")

    request_id = _assert_generated_request_id(response)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    rendered_output = capsys.readouterr().err
    rendered_events = [json.loads(line) for line in rendered_output.splitlines()]
    assert len(rendered_events) == 1
    rendered_event = rendered_events[0]
    assert {
        field_name: rendered_event[field_name]
        for field_name in (
            "event",
            "request_id",
            "http_method",
            "http_status_code",
            "http_route",
            "stage",
        )
    } == {
        "event": "request_completed",
        "request_id": request_id,
        "http_method": "GET",
        "http_status_code": 500,
        "http_route": "/safe-errors/{capability}",
        "stage": "request",
    }
    assert "code" not in rendered_event
    assert rendered_event["elapsed_seconds"] >= 0
    assert capability_sentinel not in rendered_output
    assert exception_sentinel not in rendered_output
    assert "Traceback" not in rendered_output


@pytest.mark.asyncio
async def test_api_logs_interruption_after_response_start(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit an interrupted terminal event with the observed response status."""
    capability_sentinel = "post-response-cancel-capability"
    exception_sentinel = "post-response-cancel-exception"
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    application = create_app(_app_config())

    class CancelledAfterStartResponse(StarletteResponse):
        """Send a response start before simulating a disconnected client."""

        async def __call__(
            self,
            scope: Scope,
            receive: Receive,
            send: Send,
        ) -> None:
            response_start: Message = {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
            await send(response_start)
            raise asyncio.CancelledError(exception_sentinel)

    @application.get("/interrupted-after-start/{capability}")
    async def cancel_after_response_start(
        capability: str,
    ) -> StarletteResponse:
        assert capability == capability_sentinel
        return CancelledAfterStartResponse()

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(asyncio.CancelledError):
            await client.get(f"/interrupted-after-start/{capability_sentinel}")

    rendered_output = capsys.readouterr().err
    rendered_events = [json.loads(line) for line in rendered_output.splitlines()]
    assert len(rendered_events) == 1
    rendered_event = rendered_events[0]
    assert {
        field_name: rendered_event[field_name]
        for field_name in (
            "event",
            "http_method",
            "http_status_code",
            "http_route",
            "stage",
        )
    } == {
        "event": "request_interrupted",
        "http_method": "GET",
        "http_status_code": 204,
        "http_route": "/interrupted-after-start/{capability}",
        "stage": "request",
    }
    assert rendered_event["elapsed_seconds"] >= 0
    assert capability_sentinel not in rendered_output
    assert exception_sentinel not in rendered_output
    assert "Traceback" not in rendered_output


@pytest.mark.asyncio
async def test_api_logs_failure_after_response_start(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit a safe failed terminal event after the response has started."""
    capability_sentinel = "post-response-failure-capability"
    exception_sentinel = "post-response-failure-exception"
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    application = create_app(_app_config())

    class FailingAfterStartResponse(StarletteResponse):
        """Send a response start before simulating an unexpected failure."""

        async def __call__(
            self,
            scope: Scope,
            receive: Receive,
            send: Send,
        ) -> None:
            response_start: Message = {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
            await send(response_start)
            raise RuntimeError(exception_sentinel)

    @application.get("/failed-after-start/{capability}")
    async def fail_after_response_start(capability: str) -> StarletteResponse:
        assert capability == capability_sentinel
        return FailingAfterStartResponse()

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(RuntimeError, match=exception_sentinel):
            await client.get(f"/failed-after-start/{capability_sentinel}")

    rendered_output = capsys.readouterr().err
    rendered_events = [json.loads(line) for line in rendered_output.splitlines()]
    assert len(rendered_events) == 1
    rendered_event = rendered_events[0]
    assert {
        field_name: rendered_event[field_name]
        for field_name in (
            "event",
            "http_method",
            "http_status_code",
            "http_route",
            "stage",
            "code",
        )
    } == {
        "event": "request_failed",
        "http_method": "GET",
        "http_status_code": 204,
        "http_route": "/failed-after-start/{capability}",
        "stage": "request",
        "code": "internal_error",
    }
    assert rendered_event["elapsed_seconds"] >= 0
    assert capability_sentinel not in rendered_output
    assert exception_sentinel not in rendered_output
    assert "Traceback" not in rendered_output


def test_app_config_defaults_to_deployment_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the documented deployment port when no environment override exists."""
    monkeypatch.delenv("TEXTIFY_PORT", raising=False)

    settings = AppConfig(_env_file=None)  # type: ignore[call-arg]

    assert settings.port == 8182


def test_api_main_bootstraps_json_logging_before_configured_debug_level() -> None:
    """Retain one API identity while applying the validated configured log level."""
    child_program = textwrap.dedent(
        """
        import importlib
        import logging

        from textify.logging import LogEvent, emit_event

        main_module = importlib.import_module("textify.main")

        def fake_uvicorn_run(application, **settings):
            assert application is main_module.app
            assert settings["workers"] == 1
            assert settings["access_log"] is False
            assert settings["log_config"] is None
            emit_event(
                logging.getLogger("textify.test_api"),
                LogEvent.SOURCE_INSPECTION_FAILED,
                stage="metadata",
                code="configured-debug-event",
            )

        main_module.uvicorn.run = fake_uvicorn_run
        main_module.main()
        """
    )
    environment = {
        **os.environ,
        "TEXTIFY_HOST": "127.0.0.1",
        "TEXTIFY_LOG_LEVEL": "DEBUG",
        "TEXTIFY_PORT": "8182",
    }

    completed = subprocess.run(
        [sys.executable, "-c", child_program],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0
    rendered_events = [json.loads(line) for line in completed.stderr.splitlines()]
    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "source_inspection_failed",
        "service_stopping",
    ]
    assert rendered_events[1]["level"] == "DEBUG"
    assert {event["service_role"] for event in rendered_events} == {"api"}
    assert len({event["instance_id"] for event in rendered_events}) == 1


def test_api_main_promotes_a_swallowed_lifespan_failure_marker() -> None:
    """Treat a returned Uvicorn process with a failure marker as unsuccessful."""
    child_program = textwrap.dedent(
        """
        import importlib

        main_module = importlib.import_module("textify.main")

        def fake_uvicorn_run(application, **settings):
            application.state.observability_failure_stage = "runtime"

        main_module.uvicorn.run = fake_uvicorn_run
        main_module.main()
        """
    )
    environment = {
        **os.environ,
        "TEXTIFY_HOST": "127.0.0.1",
        "TEXTIFY_LOG_LEVEL": "INFO",
        "TEXTIFY_PORT": "8182",
    }

    completed = subprocess.run(
        [sys.executable, "-c", child_program],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode != 0
    rendered_events = [json.loads(line) for line in completed.stderr.splitlines()]
    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "service_failed",
    ]
    assert rendered_events[1]["stage"] == "runtime"
    assert rendered_events[1]["code"] == "internal_error"
    assert "service_stopping" not in completed.stderr
    assert "Traceback" not in completed.stderr


def test_api_main_reports_invalid_configuration_without_exception_output() -> None:
    """Exit with one safe startup failure event for invalid environment settings."""
    configuration_sentinel = "invalid-port-sentinel"
    environment = {
        **os.environ,
        "TEXTIFY_LOG_LEVEL": "INFO",
        "TEXTIFY_PORT": configuration_sentinel,
    }

    completed = subprocess.run(
        [sys.executable, "-c", "from textify.main import main; main()"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode != 0
    rendered_events = [json.loads(line) for line in completed.stderr.splitlines()]
    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "service_failed",
    ]
    assert rendered_events[1]["stage"] == "startup"
    assert rendered_events[1]["code"] == "internal_error"
    assert "service_stopping" not in completed.stderr
    assert configuration_sentinel not in completed.stderr
    assert "Traceback" not in completed.stderr


def test_api_main_reports_unavailable_database_without_uvicorn_output() -> None:
    """Exit with one safe startup failure when Uvicorn cannot start the lifespan."""
    environment = {
        **os.environ,
        "TEXTIFY_BROKER_URL": "redis://127.0.0.1:1/15",
        "TEXTIFY_DATABASE_URL": (
            "postgresql+asyncpg://textify:unavailable@127.0.0.1:1/textify"
        ),
        "TEXTIFY_HOST": "127.0.0.1",
        "TEXTIFY_LOG_LEVEL": "INFO",
        "TEXTIFY_PORT": "8182",
    }

    completed = subprocess.run(
        [sys.executable, "-c", "from textify.main import main; main()"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0
    rendered_events = [json.loads(line) for line in completed.stderr.splitlines()]
    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "service_failed",
    ]
    assert rendered_events[1]["stage"] == "startup"
    assert rendered_events[1]["code"] == "internal_error"
    assert "service_stopping" not in completed.stderr
    assert "postgresql+asyncpg://" not in completed.stderr
    assert "redis://" not in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.asyncio
async def test_api_main_runs_uvicorn_with_only_correlated_json_stderr(
    migrated_postgresql_database_url: str,
) -> None:
    """Serve one request and stop cleanly without Uvicorn-owned output."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved_socket:
        reserved_socket.bind(("127.0.0.1", 0))
        port = int(reserved_socket.getsockname()[1])
    raw_path_sentinel = "api-smoke-raw-path"
    query_sentinel = "api-smoke-query"
    environment = {
        **os.environ,
        "TEXTIFY_BROKER_URL": "redis://127.0.0.1:1/15",
        "TEXTIFY_DATABASE_URL": migrated_postgresql_database_url,
        "TEXTIFY_HOST": "127.0.0.1",
        "TEXTIFY_LOG_LEVEL": "INFO",
        "TEXTIFY_PORT": str(port),
    }
    environment.pop("TEXTIFY_CACHE_URL", None)
    process: subprocess.Popen[str] = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", "from textify.main import main; main()"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )

    rendered_output = ""
    try:
        response: Response | None = None
        async with AsyncClient(timeout=1.0) as client:
            for _ in range(100):
                if process.poll() is not None:
                    break
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{port}/{raw_path_sentinel}"
                        f"?query={query_sentinel}"
                    )
                except HTTPError:
                    await asyncio.sleep(0.05)
                else:
                    break
        if response is None:
            raise AssertionError("API subprocess did not accept a loopback request.")
        request_id = _assert_generated_request_id(response)
        assert response.status_code == 404

        process.send_signal(signal.SIGTERM)
        try:
            _, rendered_output = await asyncio.to_thread(
                process.communicate,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.communicate, timeout=5)
            raise AssertionError(
                "API subprocess did not stop after the termination signal."
            ) from None
    finally:
        if process.poll() is None:
            process.kill()
            await asyncio.to_thread(process.communicate, timeout=5)

    rendered_events: list[dict[str, object]] = []
    for line in rendered_output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            raise AssertionError("API stderr included a non-JSON line.") from None
        assert isinstance(event, dict)
        rendered_events.append(event)

    assert process.returncode in (0, -signal.SIGTERM)
    events = [event["event"] for event in rendered_events]
    assert events.count("service_starting") == 1
    assert events.count("service_ready") == 1
    assert events.count("request_completed") == 1
    assert events.count("service_stopping") == 1
    assert (
        events.index("service_starting")
        < events.index("service_ready")
        < events.index("request_completed")
        < events.index("service_stopping")
    )
    assert "service_failed" not in events
    assert {event["service_role"] for event in rendered_events} == {"api"}
    assert {event["service"] for event in rendered_events} == {"textify"}
    assert len({event["instance_id"] for event in rendered_events}) == 1
    for event in rendered_events:
        assert type(event["timestamp"]) is str
        assert type(event["level"]) is str
        assert type(event["logger"]) is str
        assert type(event["message"]) is str
        assert type(event["event"]) is str
        assert type(event["service"]) is str
        assert type(event["service_role"]) is str
        assert type(event["instance_id"]) is str
        for field_name in (
            "duration_seconds",
            "http_status_code",
            "internal_job_id",
        ):
            if field_name in event:
                assert type(event[field_name]) is int
        if "elapsed_seconds" in event:
            assert type(event["elapsed_seconds"]) in {int, float}
        for field_name in (
            "code",
            "http_method",
            "http_route",
            "job_outcome",
            "job_status",
            "method",
            "platform",
            "request_id",
            "source_id",
            "stage",
        ):
            if field_name in event:
                assert type(event[field_name]) is str

    request_events = [
        event for event in rendered_events if event["event"] == "request_completed"
    ]
    assert request_events[0]["request_id"] == request_id
    assert "http_route" not in request_events[0]
    assert raw_path_sentinel not in rendered_output
    assert query_sentinel not in rendered_output
    assert "postgresql+asyncpg://" not in rendered_output
    assert "redis://" not in rendered_output
    assert "Traceback" not in rendered_output


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
    assert application.state.observability_stage == "startup"
    assert application.state.observability_failure_stage == "startup"


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
