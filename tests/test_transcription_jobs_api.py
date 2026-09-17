"""ASGI contracts for accepting and inspecting queued Transcription Jobs."""

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, TypedDict, cast
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from textify.config import AppConfig, Environment
from textify.main import create_app
from textify.transcription import acquisition, inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.service import TranscriptionAdapters


class QueuedJobLinks(TypedDict):
    """Represent the nested public links of a queued Transcription Job."""

    self: str
    cancel: str


class QueuedJobPayload(TypedDict):
    """Represent one queued Transcription Job response body."""

    id: str
    status: str
    submitted_at: str
    links: QueuedJobLinks


YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class ProviderCallGuard:
    """Fail a test if queued admission invokes any provider boundary."""

    def __init__(self, calls: list[str]) -> None:
        """Initialize the shared provider-call record.

        Args:
            calls: Names of provider methods unexpectedly invoked by the application.
        """
        self._calls = calls

    def __getattr__(self, method_name: str) -> NoReturn:
        """Reject any provider method lookup during queued-only operations.

        Args:
            method_name: Provider operation the queued flow must not access.

        Raises:
            AssertionError: Always, because consumers are not started in issue 02.
        """
        self._calls.append(method_name)
        raise AssertionError(
            f"Queued jobs must not call provider method {method_name}."
        )


class ProviderGuardFactory:
    """Construct executor dependencies that make provider access observable."""

    def __init__(self) -> None:
        """Initialize creation and provider-use observations."""
        self.calls = 0
        self.provider_calls: list[str] = []

    def __call__(self, _config: TranscriptionConfig) -> TranscriptionAdapters:
        """Build guards that are safe until a forbidden provider operation occurs.

        Args:
            _config: Transcription configuration unused by queued-only admission.

        Returns:
            Adapter bundle whose provider methods fail if invoked.
        """
        self.calls += 1
        guard = ProviderCallGuard(self.provider_calls)
        return TranscriptionAdapters(
            metadata_extractor=cast(inspection.MetadataExtractor, guard),
            caption_provider=cast(acquisition.CaptionProvider, guard),
            audio_downloader=cast(acquisition.AudioDownloader, guard),
            whisper_transcriber=cast(acquisition.WhisperTranscriber, guard),
        )


def _app_config(database_path: Path | None) -> AppConfig:
    """Create isolated application configuration for queued-job ASGI tests."""
    return AppConfig(
        environment=Environment.LOCAL,
        log_level="INFO",
        host="127.0.0.1",
        port=8182,
        database_path=database_path,
    )


def _transcription_config(temporary_media_root: Path) -> TranscriptionConfig:
    """Create minimal deterministic transcription settings for queued-job tests."""
    return TranscriptionConfig(
        temporary_media_root=temporary_media_root,
        max_media_bytes=1024,
        max_outstanding_jobs=8,
        job_queue_timeout_seconds=20,
    )


def _application(
    database_path: Path | None,
    temporary_media_root: Path,
    factory: ProviderGuardFactory,
) -> FastAPI:
    """Build one app with a real database and inert provider boundaries."""
    return create_app(
        _app_config(database_path),
        _transcription_config(temporary_media_root),
        factory,
        available_temporary_media_bytes=lambda _root: 1 << 60,
    )


def _assert_safe_error(
    response: Response,
    expected_status: int,
    expected_code: str,
) -> None:
    """Assert the stable public error envelope and bearer-cache policy."""
    payload = response.json()
    assert response.status_code == expected_status
    assert response.headers["Cache-Control"] == "no-store"
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message"}
    assert payload["error"]["code"] == expected_code
    assert isinstance(payload["error"]["message"], str)
    assert payload["error"]["message"]
    _assert_generated_request_id(response)


def _assert_generated_request_id(response: Response) -> str:
    """Assert and return one canonical server-generated UUIDv4 request ID."""
    request_id = response.headers["X-Request-ID"]
    parsed_request_id = UUID(request_id)
    assert parsed_request_id.version == 4
    assert str(parsed_request_id) == request_id
    return request_id


def _assert_queued_response(
    response: Response,
    status_code: int,
) -> QueuedJobPayload:
    """Assert the public queued representation and common bearer headers."""
    payload = cast(QueuedJobPayload, response.json())
    assert response.status_code == status_code
    assert set(payload) == {"id", "status", "submitted_at", "links"}
    public_id = UUID(payload["id"])
    assert public_id.version == 4
    assert str(public_id) == payload["id"]
    assert payload["status"] == "queued"
    submitted_at = datetime.fromisoformat(payload["submitted_at"])
    assert submitted_at.tzinfo is not None
    assert submitted_at.astimezone(UTC).utcoffset() == UTC.utcoffset(submitted_at)
    assert payload["links"] == {
        "self": f"/api/transcription-jobs/{public_id}",
        "cancel": f"/api/transcription-jobs/{public_id}/cancellation",
    }
    assert response.headers["Retry-After"] == "2"
    assert response.headers["Cache-Control"] == "no-store"
    _assert_generated_request_id(response)
    return payload


async def test_api_accepts_and_reopens_queued_job_without_provider_work(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Commit a queued job, persist normalized exclusions, and reopen its status.

    Args:
        migrated_database_path: Database upgraded through the actual Alembic revision.
        tmp_path: Per-test media-root owner.
    """
    first_factory = ProviderGuardFactory()
    first_application = _application(
        migrated_database_path,
        tmp_path / "first-media",
        first_factory,
    )
    async with first_application.router.lifespan_context(first_application):
        transport = ASGITransport(app=first_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": YOUTUBE_URL, "exclude": ["transcript.segments"]},
            )

    accepted_payload = _assert_queued_response(accepted, 202)
    location = accepted.headers["Location"]
    assert location == accepted_payload["links"]["self"]
    assert first_factory.calls == 1
    assert first_factory.provider_calls == []

    with sqlite3.connect(migrated_database_path) as connection:
        exclusions = connection.execute(
            "SELECT field_path FROM transcription_job_exclusion ORDER BY field_path"
        ).fetchall()
    assert exclusions == [("transcript.segments",)]

    reopened_factory = ProviderGuardFactory()
    reopened_application = _application(
        migrated_database_path,
        tmp_path / "reopened-media",
        reopened_factory,
    )
    async with reopened_application.router.lifespan_context(reopened_application):
        transport = ASGITransport(app=reopened_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            inspected = await client.get(location)

    inspected_payload = _assert_queued_response(inspected, 200)
    assert inspected_payload == accepted_payload
    assert reopened_factory.calls == 1
    assert reopened_factory.provider_calls == []


async def test_api_queues_exactly_eight_after_immediate_validation_failures(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Preserve capacity after invalid input and atomically bound nine submissions.

    Args:
        migrated_database_path: Database upgraded through the actual Alembic revision.
        tmp_path: Per-test media-root owner.
    """
    factory = ProviderGuardFactory()
    application = _application(migrated_database_path, tmp_path / "media", factory)
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            invalid_response = await client.post(
                "/api/transcription-jobs",
                json={"url": YOUTUBE_URL, "unknown": "value"},
            )
            unsupported_response = await client.post(
                "/api/transcription-jobs",
                json={"url": "https://example.com/video"},
            )
            _assert_safe_error(invalid_response, 422, "invalid_request")
            _assert_safe_error(unsupported_response, 400, "unsupported_platform")

            responses = await asyncio.gather(
                *[
                    client.post(
                        "/api/transcription-jobs",
                        json={"url": YOUTUBE_URL},
                    )
                    for _ in range(9)
                ]
            )
            accepted = [
                response for response in responses if response.status_code == 202
            ]
            rejected = [
                response for response in responses if response.status_code == 503
            ]
            assert len(accepted) == 8
            assert len(rejected) == 1
            _assert_safe_error(
                rejected[0],
                503,
                "transcription_capacity_exceeded",
            )
            capabilities = {
                _assert_queued_response(response, 202)["id"] for response in accepted
            }
            assert len(capabilities) == 8
            inspected = await asyncio.gather(
                *[
                    client.get(f"/api/transcription-jobs/{capability}")
                    for capability in capabilities
                ]
            )

    assert all(response.status_code == 200 for response in inspected)
    assert factory.provider_calls == []


async def test_api_hides_unknown_and_noncanonical_job_capabilities(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Return the same safe not-found envelope for every invalid capability form.

    Args:
        migrated_database_path: Database upgraded through the actual Alembic revision.
        tmp_path: Per-test media-root owner.
    """
    factory = ProviderGuardFactory()
    application = _application(migrated_database_path, tmp_path / "media", factory)
    unknown_capability = "00000000-0000-4000-8000-000000000000"
    uppercase_capability = "00000000-0000-4000-8000-000000000000".upper()
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for capability in (unknown_capability, uppercase_capability, "not-a-uuid"):
                response = await client.get(f"/api/transcription-jobs/{capability}")
                _assert_safe_error(response, 404, "job_not_found")

    assert factory.provider_calls == []


async def test_api_returns_store_error_without_phantom_admission(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Map a deferred commit failure safely and roll back its provisional job.

    Args:
        migrated_database_path: Database upgraded through the actual Alembic revision.
        tmp_path: Per-test media-root owner.
    """
    _install_deferred_commit_failure(migrated_database_path)
    factory = ProviderGuardFactory()
    application = _application(migrated_database_path, tmp_path / "media", factory)
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            failed = await client.post(
                "/api/transcription-jobs",
                json={"url": YOUTUBE_URL},
            )
            _assert_safe_error(failed, 503, "job_store_unavailable")
            assert "Location" not in failed.headers
            _remove_deferred_commit_failure(migrated_database_path)
            accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": YOUTUBE_URL},
            )

    _assert_queued_response(accepted, 202)
    assert factory.provider_calls == []


async def test_api_logs_only_safe_queued_job_context(
    migrated_database_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep bearer capabilities, private URLs, paths, and storage locations out of logs.

    Args:
        migrated_database_path: Database upgraded through the actual Alembic revision.
        tmp_path: Per-test media-root owner.
        capsys: Capture fixture used to inspect structured application logging.
    """
    factory = ProviderGuardFactory()
    application = _application(migrated_database_path, tmp_path / "media", factory)
    async with application.router.lifespan_context(application):
        capsys.readouterr()
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": YOUTUBE_URL, "exclude": ["transcript.segments"]},
            )
            accepted_payload = _assert_queued_response(accepted, 202)
            location = accepted.headers["Location"]
            inspected = await client.get(location)
            _assert_queued_response(inspected, 200)
            unknown = await client.get(
                "/api/transcription-jobs/00000000-0000-4000-8000-000000000000"
            )
            _assert_safe_error(unknown, 404, "job_not_found")
            _install_deferred_commit_failure(migrated_database_path)
            store_failure = await client.post(
                "/api/transcription-jobs",
                json={"url": YOUTUBE_URL},
            )
            _assert_safe_error(store_failure, 503, "job_store_unavailable")
            _remove_deferred_commit_failure(migrated_database_path)

    formatted_logs = capsys.readouterr().err
    request_id = _assert_generated_request_id(accepted)
    assert f"request_id={request_id}" in formatted_logs
    assert "platform=youtube" in formatted_logs
    assert "code=job_not_found" in formatted_logs
    assert "code=job_store_unavailable" in formatted_logs
    for sensitive_value in (
        accepted_payload["id"],
        YOUTUBE_URL,
        "transcript.segments",
        location,
        accepted_payload["links"]["cancel"],
        str(migrated_database_path),
    ):
        assert sensitive_value not in formatted_logs
    assert factory.provider_calls == []


@pytest.mark.parametrize("failure_case", ("missing", "empty", "stale", "read_only"))
async def test_api_fails_startup_before_adapter_construction_for_invalid_storage(
    failure_case: str,
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Reject missing, unmigrated, stale, and non-writable durable storage.

    Args:
        failure_case: Invalid storage condition to construct for this test.
        migrated_database_path: Fresh migrated database for the read-only case.
        tmp_path: Per-test filesystem root.
    """
    database_path: Path | None
    if failure_case == "missing":
        database_path = tmp_path / "missing.sqlite3"
    elif failure_case == "empty":
        database_path = tmp_path / "empty.sqlite3"
        database_path.touch()
    elif failure_case == "stale":
        database_path = tmp_path / "stale.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE alembic_version (version_num VARCHAR(32))")
            connection.execute("INSERT INTO alembic_version VALUES ('0000_stale')")
    else:
        database_path = migrated_database_path
        database_path.chmod(0o444)

    factory = ProviderGuardFactory()
    application = _application(database_path, tmp_path / "media", factory)
    try:
        with pytest.raises(RuntimeError):
            async with application.router.lifespan_context(application):
                pass
    finally:
        if failure_case == "read_only":
            database_path.chmod(0o644)

    assert factory.calls == 0
    assert application.state.ready is False
    if failure_case == "missing":
        assert not database_path.exists()
    elif failure_case == "empty":
        with sqlite3.connect(database_path) as connection:
            assert _application_table_names(connection) == set()


def _install_deferred_commit_failure(database_path: Path) -> None:
    """Install a test-only deferred foreign-key failure triggered at commit time."""
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE transcription_job_commit_failure (
                job_id INTEGER NOT NULL,
                FOREIGN KEY (job_id) REFERENCES transcription_job(id)
                    DEFERRABLE INITIALLY DEFERRED
            );
            CREATE TRIGGER transcription_job_commit_failure_trigger
            AFTER INSERT ON transcription_job
            BEGIN
                INSERT INTO transcription_job_commit_failure (job_id)
                VALUES (NEW.id + 1);
            END;
            """
        )


def _remove_deferred_commit_failure(database_path: Path) -> None:
    """Remove the test-only deferred foreign-key failure after rollback proof."""
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            DROP TRIGGER transcription_job_commit_failure_trigger;
            DROP TABLE transcription_job_commit_failure;
            """
        )


def _application_table_names(connection: sqlite3.Connection) -> set[str]:
    """Return application-owned durable-job tables from one SQLite connection."""
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name LIKE 'transcription_job%'"
    ).fetchall()
    return {str(row[0]) for row in rows}
