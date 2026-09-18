"""ASGI tests for application startup and job-route observability."""

import threading
from pathlib import Path
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient, Response

from textify.config import AppConfig, Environment
from textify.main import _validate_temporary_media_capacity, create_app
from textify.transcription import acquisition, inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.service import TranscriptionAdapters
from textify.transcription.types import (
    Segment,
    TimedTranscript,
    Transcript,
    TranscriptMethod,
)

TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"


class StartupCaptionProvider:
    """Expose no optional captions during startup and observability tests."""

    def list_tracks(self, video_id: str) -> tuple[acquisition.CaptionTrack, ...]:
        """Return no caption tracks.

        Args:
            video_id: YouTube identifier ignored by this deterministic provider.

        Returns:
            An empty caption-track collection.
        """
        del video_id
        return ()


class StartupMetadataExtractor:
    """Return valid TikTok metadata if a background worker starts execution."""

    def extract(
        self,
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> inspection.ExtractedMetadata:
        """Return metadata accepted by the production normalization boundary.

        Args:
            provider_url: Validated provider URL.
            deadline: Monotonic deadline unused by this deterministic provider.
            cancellation_event: Worker cancellation signal, which must remain unset.

        Returns:
            Valid TikTok metadata.
        """
        del deadline
        assert not cancellation_event.is_set()
        return inspection.ExtractedMetadata(
            {
                "id": "1234567890123456789",
                "extractor_key": "TikTok",
                "webpage_url": provider_url,
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12,
                "formats": ({"vcodec": "h264"},),
            }
        )


class StartupAudioDownloader:
    """Create a minimal audio input if a background worker starts execution."""

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Create and return deterministic temporary media.

        Args:
            source_url: Validated provider URL ignored by this deterministic provider.
            destination: Existing private directory for media output.
            deadline: Monotonic deadline unused by this deterministic provider.
            cancellation_event: Worker cancellation signal, which must remain unset.

        Returns:
            Completed audio path inside destination.
        """
        del source_url, deadline
        assert not cancellation_event.is_set()
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"audio")
        return audio_path


class StartupTranscriber:
    """Return a valid transcript if a background worker starts execution."""

    def transcribe(
        self, audio_path: Path, *, include_segments: bool = True
    ) -> Transcript:
        """Return a normalized text-only or timed transcript.

        Args:
            audio_path: Completed temporary audio supplied by the downloader.
            include_segments: Whether the worker needs timed segments.

        Returns:
            A valid normalized transcript.
        """
        assert audio_path.is_file()
        if not include_segments:
            return Transcript(TranscriptMethod.FASTER_WHISPER, "en", "One")
        return TimedTranscript(
            TranscriptMethod.FASTER_WHISPER,
            "en",
            "One",
            (Segment(0.0, 1.0, "One"),),
        )


class StartupAdaptersFactory:
    """Build a deterministic provider bundle and count startup construction."""

    def __init__(self) -> None:
        """Initialize the startup-construction counter."""
        self.calls = 0

    def __call__(self, _settings: TranscriptionConfig) -> TranscriptionAdapters:
        """Build a production-shaped deterministic adapter bundle.

        Args:
            _settings: Validated application configuration unused by test adapters.

        Returns:
            Complete provider adapter bundle.
        """
        self.calls += 1
        return TranscriptionAdapters(
            metadata_extractor=StartupMetadataExtractor(),
            caption_provider=StartupCaptionProvider(),
            audio_downloader=StartupAudioDownloader(),
            whisper_transcriber=StartupTranscriber(),
        )


def _app_config(database_path: Path) -> AppConfig:
    """Create isolated application configuration for ASGI tests.

    Args:
        database_path: Per-test SQLite database upgraded through Alembic.

    Returns:
        Configuration independent of local environment files.
    """
    return AppConfig(
        environment=Environment.LOCAL,
        log_level="INFO",
        host="127.0.0.1",
        port=8182,
        database_path=database_path,
    )


def _transcription_config(temporary_media_root: Path) -> TranscriptionConfig:
    """Create isolated transcription configuration for ASGI tests.

    Args:
        temporary_media_root: Temporary root owned by one test.

    Returns:
        Configuration independent of local environment files.
    """
    return TranscriptionConfig(
        max_duration_seconds=1800,
        temporary_media_root=temporary_media_root,
        beam_size=1,
        vad_filter=True,
        temperature=0.0,
        condition_on_previous_text=True,
        transcription_concurrency=1,
        job_worker_count=1,
        max_outstanding_jobs=8,
        job_queue_timeout_seconds=20,
        max_media_bytes=1024,
        metadata_timeout_seconds=30.0,
        audio_download_timeout_seconds=300.0,
        transcription_timeout_seconds=1800.0,
        initial_prompt="test prompt",
        hf_token=None,
    )


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


def test_temporary_media_capacity_accepts_exact_quota(tmp_path: Path) -> None:
    """The configured quota accepts exactly the required free bytes."""
    settings = _transcription_config(tmp_path).model_copy(
        update={
            "job_worker_count": 2,
            "transcription_concurrency": 3,
            "max_media_bytes": 100,
        }
    )

    _validate_temporary_media_capacity(
        settings,
        available_bytes=lambda _root: 600,
    )


def test_temporary_media_capacity_rejects_one_byte_short(tmp_path: Path) -> None:
    """The configured quota rejects free space below the exact requirement."""
    settings = _transcription_config(tmp_path).model_copy(
        update={
            "job_worker_count": 2,
            "transcription_concurrency": 3,
            "max_media_bytes": 100,
        }
    )

    with pytest.raises(
        RuntimeError,
        match=("Temporary media root requires 600 available bytes; 599 are available."),
    ):
        _validate_temporary_media_capacity(
            settings,
            available_bytes=lambda _root: 599,
        )


def test_temporary_media_capacity_translates_disk_usage_failure(tmp_path: Path) -> None:
    """An unavailable capacity reader exposes the stable startup failure."""
    settings = _transcription_config(tmp_path)

    def unavailable_capacity_reader(_root: Path) -> int:
        """Raise the filesystem failure that must stay internal.

        Args:
            _root: Temporary-media root ignored by this failure double.

        Raises:
            OSError: Always, to emulate a failed disk capacity query.
        """
        raise OSError("disk query failed")

    with pytest.raises(
        RuntimeError,
        match="Unable to determine temporary media root capacity.",
    ) as error:
        _validate_temporary_media_capacity(
            settings,
            available_bytes=unavailable_capacity_reader,
        )

    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_api_starts_at_exact_temporary_media_capacity(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Startup succeeds when free space equals the configured media quota."""
    settings = _transcription_config(tmp_path).model_copy(
        update={
            "job_worker_count": 2,
            "transcription_concurrency": 3,
            "max_media_bytes": 100,
        }
    )
    application = create_app(
        _app_config(migrated_database_path),
        settings,
        StartupAdaptersFactory(),
        available_temporary_media_bytes=lambda _root: 600,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health_response = await client.get("/health")

    assert health_response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_api_fails_startup_before_model_when_temporary_media_capacity_is_low(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Low capacity aborts startup after root creation but before adapters load."""
    media_root = tmp_path / "media"
    settings = _transcription_config(media_root).model_copy(
        update={
            "job_worker_count": 2,
            "transcription_concurrency": 3,
            "max_media_bytes": 100,
        }
    )
    factory = StartupAdaptersFactory()
    application = create_app(
        _app_config(migrated_database_path),
        settings,
        factory,
        available_temporary_media_bytes=lambda _root: 599,
    )

    with pytest.raises(
        RuntimeError,
        match=("Temporary media root requires 600 available bytes; 599 are available."),
    ):
        async with application.router.lifespan_context(application):
            pass

    assert media_root.is_dir()
    assert factory.calls == 0
    assert application.state.ready is False


@pytest.mark.asyncio
async def test_api_adds_fresh_request_ids_to_job_route_outcomes(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Health, submission, validation, and unmatched routes get fresh UUIDv4 IDs."""
    inbound_request_id = "inbound-request-id-sentinel"
    application = create_app(
        _app_config(migrated_database_path),
        _transcription_config(tmp_path),
        StartupAdaptersFactory(),
        available_temporary_media_bytes=lambda _root: 1 << 60,
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

    assert health_response.status_code == 200
    assert submission_response.status_code == 202
    assert validation_response.status_code == 422
    assert validation_response.json()["error"]["code"] == "invalid_request"
    assert unmatched_response.status_code == 404
    request_ids = {
        _assert_generated_request_id(response)
        for response in (
            health_response,
            submission_response,
            validation_response,
            unmatched_response,
        )
    }
    assert len(request_ids) == 4
    assert inbound_request_id not in request_ids
