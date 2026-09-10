"""ASGI tests for the public Textify transcript API."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from textify.config import AppConfig, Environment
from textify.main import create_app
from textify.transcription.acquisition import CaptionTrack
from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import (
    AudioDownloadFailedError,
    AudioDownloadTimeoutError,
    InvalidMediaDurationError,
    InvalidUrlError,
    MetadataRetrievalFailedError,
    MetadataTimeoutError,
    NoUsableTranscriptError,
    TranscriptionError,
    TranscriptionFailedError,
    TranscriptionTimeoutError,
    UnsupportedContentError,
    UnsupportedMediaError,
    UnsupportedPlatformError,
    VideoTooLongError,
)
from textify.transcription.inspection import ExtractedMetadata
from textify.transcription.service import TranscriptionAdapters
from textify.transcription.types import (
    Segment,
    Transcript,
    TranscriptionResult,
    TranscriptMethod,
)

DIRECT_TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"
SHORT_TIKTOK_URL = "https://vm.tiktok.com/abcdefgh"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class ApiCaptionProvider:
    """Supply no dormant captions for TikTok-only API tests."""

    def list_tracks(self, video_id: str) -> Sequence[CaptionTrack]:
        """Return no caption tracks.

        Args:
            video_id: Video identifier ignored by the current baseline.

        Returns:
            Empty caption track sequence.
        """
        del video_id
        return ()


class ApiMetadataExtractor:
    """Return deterministic external-shaped TikTok metadata."""

    def __init__(self) -> None:
        """Initialize call recording."""
        self.calls: list[str] = []

    def extract(self, canonical_url: str) -> ExtractedMetadata:
        """Return valid TikTok metadata for one provider URL.

        Args:
            canonical_url: Validated TikTok provider URL.

        Returns:
            External-shaped metadata accepted by the provider boundary.
        """
        self.calls.append(canonical_url)
        return ExtractedMetadata(
            {
                "id": "1234567890123456789",
                "extractor_key": "TikTok",
                "webpage_url": DIRECT_TIKTOK_URL,
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12,
                "formats": ({"vcodec": "h264"},),
            }
        )


class ApiAudioDownloader:
    """Create request-owned audio while recording the submitted URL."""

    def __init__(self) -> None:
        """Initialize call and directory recording."""
        self.calls: list[str] = []
        self.request_directories: list[Path] = []

    def download(self, source_url: str, destination: Path) -> Path:
        """Create a deterministic request-owned audio file.

        Args:
            source_url: Original submitted TikTok URL.
            destination: Request-scoped temporary directory.

        Returns:
            Created audio path inside ``destination``.
        """
        self.calls.append(source_url)
        self.request_directories.append(destination)
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"media")
        return audio_path


class ApiTranscriber:
    """Return a deterministic normalized Transcript."""

    def transcribe(self, audio_path: Path) -> Transcript:
        """Return one normalized transcript for test audio.

        Args:
            audio_path: Request-owned audio path.

        Returns:
            Deterministic timed Transcript.
        """
        assert audio_path.is_file()
        segment = Segment(0.0, 1.0, "One")
        return Transcript(TranscriptMethod.FASTER_WHISPER, "en", (segment,), "One")


class CountingAdaptersFactory:
    """Build deterministic adapters and record process-lifetime construction."""

    def __init__(self) -> None:
        """Initialize construction and provider call recording."""
        self.calls = 0
        self.metadata_extractors: list[ApiMetadataExtractor] = []
        self.audio_downloaders: list[ApiAudioDownloader] = []

    def __call__(self, config: TranscriptionConfig) -> TranscriptionAdapters:
        """Build one deterministic adapter bundle.

        Args:
            config: Lifespan configuration used only for interface compatibility.

        Returns:
            Complete deterministic provider bundle.
        """
        del config
        self.calls += 1
        metadata_extractor = ApiMetadataExtractor()
        audio_downloader = ApiAudioDownloader()
        self.metadata_extractors.append(metadata_extractor)
        self.audio_downloaders.append(audio_downloader)
        return TranscriptionAdapters(
            metadata_extractor=metadata_extractor,
            caption_provider=ApiCaptionProvider(),
            audio_downloader=audio_downloader,
            whisper_transcriber=ApiTranscriber(),
        )


class StartupFailureFactory:
    """Fail deterministically while constructing lifespan-owned adapters."""

    def __call__(self, config: TranscriptionConfig) -> TranscriptionAdapters:
        """Raise a safe startup failure before readiness.

        Args:
            config: Lifespan configuration ignored by this failing factory.

        Raises:
            RuntimeError: Always, to emulate unavailable native startup.
        """
        del config
        raise RuntimeError("native startup failed")


class ErrorService:
    """Raise one configured domain error from the route boundary."""

    def __init__(self, error: TranscriptionError) -> None:
        """Initialize the domain error to expose through HTTP.

        Args:
            error: Safe domain error to raise.
        """
        self._error = error

    async def transcribe(self, submitted_url: str) -> TranscriptionResult:
        """Raise the configured safe error.

        Args:
            submitted_url: Submitted URL intentionally ignored by the fake.

        Raises:
            TranscriptionError: Configured safe domain error.
        """
        del submitted_url
        raise self._error


def app_config() -> AppConfig:
    """Create isolated application configuration for ASGI tests.

    Returns:
        Configuration independent of local environment files.
    """
    return AppConfig(
        environment=Environment.LOCAL,
        log_level="INFO",
        host="127.0.0.1",
        port=8000,
    )


def transcription_config(temporary_media_root: Path) -> TranscriptionConfig:
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
        initial_prompt="test prompt",
        hf_token=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("submitted_url", (DIRECT_TIKTOK_URL, SHORT_TIKTOK_URL))
async def test_api_transcribes_current_tiktok_forms_with_one_lifespan_model(
    tmp_path: Path,
    submitted_url: str,
) -> None:
    """Readiness and direct or short TikTok requests work without provider I/O."""
    factory = CountingAdaptersFactory()
    application = create_app(
        app_config(),
        transcription_config(tmp_path),
        factory,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health_response = await client.get("/health")
            transcript_response = await client.post(
                "/api/transcripts",
                json={"url": submitted_url},
            )

    payload = transcript_response.json()
    assert health_response.json() == {"status": "ok"}
    assert transcript_response.status_code == 200
    assert payload["source"]["platform"] == "tiktok"
    assert payload["transcript"] == {
        "method": "faster_whisper",
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "One"}],
        "text": "One",
    }
    assert factory.calls == 1
    assert factory.audio_downloaders[0].calls == [submitted_url]
    assert all(
        not directory.exists()
        for directory in factory.audio_downloaders[0].request_directories
    )


@pytest.mark.asyncio
async def test_api_returns_safe_request_validation_error(tmp_path: Path) -> None:
    """Malformed payloads return invalid_request without echoing their value."""
    application = create_app(
        app_config(),
        transcription_config(tmp_path),
        CountingAdaptersFactory(),
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/transcripts",
                json={"url": YOUTUBE_URL, "unknown": "value"},
            )

    payload = response.json()
    assert response.status_code == 422
    assert payload["error"]["code"] == "invalid_request"
    assert YOUTUBE_URL not in str(payload)
    assert "value" not in str(payload)


@pytest.mark.asyncio
async def test_api_maps_all_baseline_domain_errors(tmp_path: Path) -> None:
    """Each stable domain status/code pair is observable through HTTP."""
    error_cases = (
        (InvalidUrlError(), 400, "invalid_url"),
        (UnsupportedPlatformError(), 400, "unsupported_platform"),
        (UnsupportedContentError(), 422, "unsupported_content"),
        (VideoTooLongError(), 422, "video_too_long"),
        (InvalidMediaDurationError(), 422, "invalid_media_duration"),
        (UnsupportedMediaError(), 422, "unsupported_media"),
        (NoUsableTranscriptError(), 422, "no_usable_transcript"),
        (MetadataRetrievalFailedError(), 502, "metadata_retrieval_failed"),
        (AudioDownloadFailedError(), 502, "audio_download_failed"),
        (TranscriptionFailedError(), 502, "transcription_failed"),
        (MetadataTimeoutError(), 504, "metadata_timeout"),
        (AudioDownloadTimeoutError(), 504, "audio_download_timeout"),
        (TranscriptionTimeoutError(), 504, "transcription_timeout"),
    )
    application = create_app(
        app_config(),
        transcription_config(tmp_path),
        CountingAdaptersFactory(),
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for error, expected_status, expected_code in error_cases:
                application.state.transcript_service = ErrorService(error)
                response = await client.post(
                    "/api/transcripts",
                    json={"url": DIRECT_TIKTOK_URL},
                )
                assert response.status_code == expected_status
                assert response.json()["error"]["code"] == expected_code


@pytest.mark.asyncio
async def test_api_fails_startup_before_readiness_when_adapter_factory_fails(
    tmp_path: Path,
) -> None:
    """Native construction failures abort lifespan before the app is ready."""
    application = create_app(
        app_config(),
        transcription_config(tmp_path),
        StartupFailureFactory(),
    )

    with pytest.raises(RuntimeError, match="native startup failed"):
        async with application.router.lifespan_context(application):
            pass

    assert application.state.ready is False
