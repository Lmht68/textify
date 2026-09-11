"""ASGI tests for the public Textify transcript API."""

import asyncio
import json
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from starlette.types import Message, Scope

from textify.config import AppConfig, Environment
from textify.main import _validate_temporary_media_capacity, create_app
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
    TranscriptionCapacityExceededError,
    TranscriptionError,
    TranscriptionFailedError,
    TranscriptionTimeoutError,
    UnsupportedContentError,
    UnsupportedMediaError,
    UnsupportedPlatformError,
    VideoTooLongError,
)
from textify.transcription.inspection import ExtractedMetadata, PreparedAudio
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


def _sufficient_temporary_media_bytes(_root: Path) -> int:
    """Return a capacity value independent of the test host filesystem."""
    return 1 << 60


def _controlled_tiktok_metadata() -> dict[str, object]:
    """Return valid raw TikTok metadata for controlled lifecycle providers."""
    return {
        "id": "1234567890123456789",
        "extractor_key": "TikTok",
        "webpage_url": DIRECT_TIKTOK_URL,
        "title": "Title",
        "description": "Description",
        "channel": "Creator",
        "duration": 12,
        "formats": ({"vcodec": "h264"},),
    }


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

    def extract(
        self,
        canonical_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> ExtractedMetadata:
        """Return valid TikTok metadata for one provider URL.

        Args:
            canonical_url: Validated TikTok provider URL.
            deadline: Monotonic absolute metadata deadline.
            cancellation_event: Cooperative request-cancellation signal.

        Returns:
            External-shaped metadata accepted by the provider boundary.
        """
        del deadline, cancellation_event
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

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Create a deterministic request-owned audio file.

        Args:
            source_url: Original submitted TikTok URL.
            destination: Request-scoped temporary directory.
            deadline: Monotonic absolute audio-download deadline.
            cancellation_event: Cooperative request-cancellation signal.

        Returns:
            Created audio path inside ``destination``.
        """
        del deadline, cancellation_event
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


@dataclass
class ControlledAdapterState:
    """Coordinate deterministic provider stages across concurrent ASGI requests."""

    metadata_waits_for_cancellation: bool = False
    metadata_prepares_audio_after_cancellation: bool = False
    metadata_prepares_audio: bool = False
    download_waits_for_cancellation: bool = False
    native_waits_for_release: bool = False
    metadata_delay_seconds: float = 0.0
    download_delay_seconds: float = 0.0
    download_size_bytes: int = 5
    native_failure: Exception | None = None
    native_result_texts: list[str] = field(default_factory=list)
    metadata_calls: list[str] = field(default_factory=list)
    metadata_deadlines: list[float] = field(default_factory=list)
    download_calls: list[str] = field(default_factory=list)
    download_deadlines: list[float] = field(default_factory=list)
    native_calls: list[Path] = field(default_factory=list)
    request_directories: list[Path] = field(default_factory=list)
    prepared_directories: list[Path] = field(default_factory=list)
    call_order: list[str] = field(default_factory=list)
    metadata_entered: threading.Event = field(default_factory=threading.Event)
    download_entered: threading.Event = field(default_factory=threading.Event)
    download_second_entered: threading.Event = field(default_factory=threading.Event)
    native_entered: threading.Event = field(default_factory=threading.Event)
    native_completed: threading.Event = field(default_factory=threading.Event)
    native_release: threading.Event = field(default_factory=threading.Event)


class ControlledMetadataExtractor:
    """Provide observable metadata behavior for lifecycle ASGI tests."""

    def __init__(
        self,
        state: ControlledAdapterState,
        temporary_media_root: Path,
    ) -> None:
        """Initialize shared state and temporary media ownership."""
        self._state = state
        self._temporary_media_root = temporary_media_root

    def extract(
        self,
        canonical_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> ExtractedMetadata:
        """Return metadata after the configured synchronous test behavior."""
        self._state.metadata_calls.append(canonical_url)
        self._state.metadata_deadlines.append(deadline)
        self._state.call_order.append("metadata")
        self._state.metadata_entered.set()
        if self._state.metadata_prepares_audio:
            return self._prepare_late_audio()
        if self._state.metadata_delay_seconds:
            threading.Event().wait(self._state.metadata_delay_seconds)
        if self._state.metadata_waits_for_cancellation:
            while not cancellation_event.wait(0.002):
                self._state.metadata_deadlines.append(deadline)
            if self._state.metadata_prepares_audio_after_cancellation:
                return self._prepare_late_audio()
        return ExtractedMetadata(_controlled_tiktok_metadata())

    def _prepare_late_audio(self) -> ExtractedMetadata:
        """Create prepared media whose cleanup remains with the service."""
        self._temporary_media_root.mkdir(parents=True, exist_ok=True)
        request_directory = Path(
            tempfile.mkdtemp(
                prefix="textify-controlled-inspection-",
                dir=str(self._temporary_media_root),
            )
        )
        audio_path = request_directory / "audio.webm"
        audio_path.write_bytes(b"media")
        self._state.prepared_directories.append(request_directory)
        return ExtractedMetadata(
            _controlled_tiktok_metadata(),
            PreparedAudio(audio_path, request_directory),
        )


class ControlledAudioDownloader:
    """Provide observable normal-download behavior for lifecycle ASGI tests."""

    def __init__(self, state: ControlledAdapterState) -> None:
        """Initialize shared request-stage state."""
        self._state = state

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Create media and apply the configured synchronous test behavior."""
        self._state.download_calls.append(source_url)
        self._state.download_deadlines.append(deadline)
        self._state.request_directories.append(destination)
        self._state.call_order.append("download")
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"x" * self._state.download_size_bytes)
        self._state.download_entered.set()
        if len(self._state.download_calls) == 2:
            self._state.download_second_entered.set()
        if self._state.download_delay_seconds:
            threading.Event().wait(self._state.download_delay_seconds)
        if self._state.download_waits_for_cancellation:
            while not cancellation_event.wait(0.002):
                self._state.download_deadlines.append(deadline)
        return audio_path


class ControlledTranscriber:
    """Provide observable native behavior for lifecycle ASGI tests."""

    def __init__(self, state: ControlledAdapterState) -> None:
        """Initialize shared native-stage state."""
        self._state = state

    def transcribe(self, audio_path: Path) -> Transcript:
        """Return a transcript after any configured non-cancellable wait."""
        assert audio_path.is_file()
        call_index = len(self._state.native_calls)
        self._state.native_calls.append(audio_path)
        self._state.call_order.append("native")
        self._state.native_entered.set()
        try:
            if self._state.native_waits_for_release:
                self._state.native_release.wait()
            if self._state.native_failure is not None:
                raise self._state.native_failure
            text = (
                self._state.native_result_texts[call_index]
                if call_index < len(self._state.native_result_texts)
                else "One"
            )
            segment = Segment(0.0, 1.0, text)
            return Transcript(TranscriptMethod.FASTER_WHISPER, "en", (segment,), text)
        finally:
            self._state.native_completed.set()


class ControlledAdaptersFactory:
    """Build one observable provider bundle for lifecycle ASGI tests."""

    def __init__(self, state: ControlledAdapterState) -> None:
        """Initialize the state observed by tests."""
        self._state = state
        self.calls = 0

    def __call__(self, config: TranscriptionConfig) -> TranscriptionAdapters:
        """Build the deterministic controlled provider bundle."""
        self.calls += 1
        return TranscriptionAdapters(
            metadata_extractor=ControlledMetadataExtractor(
                self._state,
                config.temporary_media_root,
            ),
            caption_provider=ApiCaptionProvider(),
            audio_downloader=ControlledAudioDownloader(self._state),
            whisper_transcriber=ControlledTranscriber(self._state),
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
        max_pending_transcriptions=2,
        max_media_bytes=1024,
        metadata_timeout_seconds=30.0,
        audio_download_timeout_seconds=300.0,
        transcription_queue_timeout_seconds=300.0,
        transcription_timeout_seconds=1800.0,
        initial_prompt="test prompt",
        hf_token=None,
    )


async def _wait_for_thread_event(event: threading.Event) -> None:
    """Await one synchronous provider-stage signal without blocking the loop."""
    assert await asyncio.to_thread(event.wait, 1.0)


async def _wait_for_condition(condition: Callable[[], bool]) -> None:
    """Await an observable provider-state condition with a bounded guard."""
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.005)
    pytest.fail("controlled provider condition was not reached")


async def _await_request(task: asyncio.Task[Response]) -> Response:
    """Await a request task without cancelling it when the test guard expires."""
    return await asyncio.wait_for(asyncio.shield(task), 1.0)


async def _assert_request_pending(task: asyncio.Task[Response]) -> None:
    """Prove a request remains pending without cancelling its lifecycle."""
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), 0.05)
    assert not task.done()


async def _start_disconnect_request(
    application: FastAPI,
    stage_entry: threading.Event,
) -> tuple[asyncio.Task[None], list[Message]]:
    """Start one synthetic ASGI request that disconnects after a controlled stage."""
    request_body = json.dumps({"url": DIRECT_TIKTOK_URL}).encode()
    body_delivered = False
    sent_messages: list[Message] = []

    async def receive() -> Message:
        nonlocal body_delivered
        if not body_delivered:
            body_delivered = True
            return {
                "type": "http.request",
                "body": request_body,
                "more_body": False,
            }
        await _wait_for_thread_event(stage_entry)
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent_messages.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "server": ("test", 80),
        "scheme": "http",
        "method": "POST",
        "root_path": "",
        "path": "/api/transcripts",
        "raw_path": b"/api/transcripts",
        "query_string": b"",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(request_body)).encode()),
        ],
        "client": ("test", 50000),
    }
    app_task = asyncio.create_task(application(scope, receive, send))
    return app_task, sent_messages


def test_temporary_media_capacity_accepts_exact_quota(tmp_path: Path) -> None:
    """The configured quota accepts exactly the required free bytes."""
    settings = transcription_config(tmp_path).model_copy(
        update={
            "transcription_concurrency": 2,
            "max_pending_transcriptions": 3,
            "max_media_bytes": 100,
        }
    )

    _validate_temporary_media_capacity(
        settings,
        available_bytes=lambda _root: 600,
    )


def test_temporary_media_capacity_rejects_one_byte_short(tmp_path: Path) -> None:
    """The configured quota rejects free space below the exact requirement."""
    settings = transcription_config(tmp_path).model_copy(
        update={
            "transcription_concurrency": 2,
            "max_pending_transcriptions": 3,
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


def test_temporary_media_capacity_translates_disk_usage_failure(
    tmp_path: Path,
) -> None:
    """An unavailable capacity reader exposes the stable startup failure."""
    settings = transcription_config(tmp_path)

    def unavailable_capacity_reader(_root: Path) -> int:
        """Raise the filesystem failure that must stay internal."""
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
async def test_api_starts_at_exact_temporary_media_capacity(tmp_path: Path) -> None:
    """Startup succeeds when free space equals the configured media quota."""
    settings = transcription_config(tmp_path).model_copy(
        update={
            "transcription_concurrency": 2,
            "max_pending_transcriptions": 3,
            "max_media_bytes": 100,
        }
    )
    application = create_app(
        app_config(),
        settings,
        CountingAdaptersFactory(),
        available_temporary_media_bytes=lambda _root: 600,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health_response = await client.get("/health")

    assert health_response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_api_fails_startup_before_model_when_temporary_media_capacity_is_low(
    tmp_path: Path,
) -> None:
    """Low capacity aborts startup after root creation but before adapters load."""
    media_root = tmp_path / "media"
    settings = transcription_config(media_root).model_copy(
        update={
            "transcription_concurrency": 2,
            "max_pending_transcriptions": 3,
            "max_media_bytes": 100,
        }
    )
    factory = CountingAdaptersFactory()
    application = create_app(
        app_config(),
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
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
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
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
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
        (TranscriptionCapacityExceededError(), 503, "transcription_capacity_exceeded"),
    )
    application = create_app(
        app_config(),
        transcription_config(tmp_path),
        CountingAdaptersFactory(),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
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
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    with pytest.raises(RuntimeError, match="native startup failed"):
        async with application.router.lifespan_context(application):
            pass

    assert application.state.ready is False


@pytest.mark.asyncio
async def test_api_rejects_oversized_completed_audio_before_native(
    tmp_path: Path,
) -> None:
    """A completed audio file above the cap never reaches native inference."""
    state = ControlledAdapterState(download_size_bytes=1025)
    application = create_app(
        app_config(),
        transcription_config(tmp_path),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/transcripts",
                json={"url": DIRECT_TIKTOK_URL},
            )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_media"
    assert state.native_calls == []
    assert all(not directory.exists() for directory in state.request_directories)


@pytest.mark.asyncio
async def test_api_rejects_work_beyond_active_and_pending_capacity_before_download(
    tmp_path: Path,
) -> None:
    """Reject a third request before it can invoke metadata or create media."""
    state = ControlledAdapterState(native_waits_for_release=True)
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={
                "transcription_concurrency": 1,
                "max_pending_transcriptions": 1,
                "transcription_queue_timeout_seconds": 1.0,
            }
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first_request = asyncio.create_task(
                client.post("/api/transcripts", json={"url": DIRECT_TIKTOK_URL})
            )
            await _wait_for_thread_event(state.native_entered)

            second_request = asyncio.create_task(
                client.post("/api/transcripts", json={"url": DIRECT_TIKTOK_URL})
            )
            await _wait_for_thread_event(state.download_second_entered)

            overload_response = await client.post(
                "/api/transcripts",
                json={"url": DIRECT_TIKTOK_URL},
            )
            assert overload_response.status_code == 503
            assert (
                overload_response.json()["error"]["code"]
                == "transcription_capacity_exceeded"
            )
            assert len(state.metadata_calls) == 2
            assert len(state.request_directories) == 2

            state.native_release.set()
            first_response = await _await_request(first_request)
            second_response = await _await_request(second_request)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert all(not directory.exists() for directory in state.request_directories)


@pytest.mark.asyncio
async def test_api_expires_the_inference_queue_and_cleans_waiting_media(
    tmp_path: Path,
) -> None:
    """Expire queue waiting without allowing a second native call to start."""
    state = ControlledAdapterState(native_waits_for_release=True)
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={
                "transcription_concurrency": 1,
                "transcription_queue_timeout_seconds": 0.05,
            }
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first_request = asyncio.create_task(
                client.post("/api/transcripts", json={"url": DIRECT_TIKTOK_URL})
            )
            await _wait_for_thread_event(state.native_entered)

            second_request = asyncio.create_task(
                client.post("/api/transcripts", json={"url": DIRECT_TIKTOK_URL})
            )
            await _wait_for_thread_event(state.download_second_entered)
            second_response = await _await_request(second_request)

            assert second_response.status_code == 503
            assert (
                second_response.json()["error"]["code"]
                == "transcription_capacity_exceeded"
            )
            assert len(state.native_calls) == 1
            assert not state.request_directories[1].exists()

            state.native_release.set()
            first_response = await _await_request(first_request)

    assert first_response.status_code == 200
    assert not state.request_directories[0].exists()


@pytest.mark.asyncio
async def test_api_expires_metadata_deadline_and_cleans_late_prepared_media(
    tmp_path: Path,
) -> None:
    """Bound metadata retries and reclaim duration-fallback media after timeout."""
    state = ControlledAdapterState(
        metadata_waits_for_cancellation=True,
        metadata_prepares_audio_after_cancellation=True,
    )
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={"metadata_timeout_seconds": 0.05}
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            request = asyncio.create_task(
                client.post("/api/transcripts", json={"url": DIRECT_TIKTOK_URL})
            )
            await _wait_for_thread_event(state.metadata_entered)
            response = await _await_request(request)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "metadata_timeout"
    assert len(state.metadata_deadlines) > 1
    assert len(set(state.metadata_deadlines)) == 1
    assert state.call_order == ["metadata"]
    assert state.download_calls == []
    assert state.native_calls == []
    assert all(not directory.exists() for directory in state.prepared_directories)


@pytest.mark.asyncio
async def test_api_expires_download_deadline_and_cleans_request_media(
    tmp_path: Path,
) -> None:
    """Bound normal-download retries and remove their request-owned media."""
    state = ControlledAdapterState(download_waits_for_cancellation=True)
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={"audio_download_timeout_seconds": 0.05}
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            request = asyncio.create_task(
                client.post("/api/transcripts", json={"url": DIRECT_TIKTOK_URL})
            )
            await _wait_for_thread_event(state.download_entered)
            response = await _await_request(request)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "audio_download_timeout"
    assert len(state.download_deadlines) > 1
    assert len(set(state.download_deadlines)) == 1
    assert state.call_order == ["metadata", "download"]
    assert state.native_calls == []
    assert all(not directory.exists() for directory in state.request_directories)


@pytest.mark.asyncio
async def test_api_returns_native_timeout_before_abandoned_work_completes(
    tmp_path: Path,
) -> None:
    """A native timeout returns while retained work still owns permit and media."""
    state = ControlledAdapterState(
        native_waits_for_release=True,
        native_result_texts=["abandoned", "fresh"],
    )
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={
                "transcription_timeout_seconds": 0.05,
                "transcription_concurrency": 1,
                "max_pending_transcriptions": 1,
                "transcription_queue_timeout_seconds": 1.0,
            }
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            timed_response = await client.post(
                "/api/transcripts",
                json={"url": DIRECT_TIKTOK_URL},
            )
            assert timed_response.status_code == 504
            assert timed_response.json()["error"]["code"] == "transcription_timeout"
            assert not state.native_completed.is_set()
            assert state.request_directories[0].exists()

            queued_request = asyncio.create_task(
                client.post("/api/transcripts", json={"url": DIRECT_TIKTOK_URL})
            )
            await _wait_for_thread_event(state.download_second_entered)
            await _assert_request_pending(queued_request)

            overload_response = await client.post(
                "/api/transcripts",
                json={"url": DIRECT_TIKTOK_URL},
            )
            assert overload_response.status_code == 503
            assert len(state.metadata_calls) == 2
            assert len(state.native_calls) == 1

            state.native_release.set()
            queued_response = await _await_request(queued_request)
            await _wait_for_condition(
                lambda: all(
                    not directory.exists() for directory in state.request_directories
                )
            )
            reusable_response = await client.post(
                "/api/transcripts",
                json={"url": DIRECT_TIKTOK_URL},
            )

    assert queued_response.status_code == 200
    assert queued_response.json()["transcript"]["text"] == "fresh"
    assert "abandoned" not in str(queued_response.json())
    assert reusable_response.status_code == 200
    assert all(not directory.exists() for directory in state.request_directories)


@pytest.mark.asyncio
async def test_api_consumes_abandoned_native_failure(tmp_path: Path) -> None:
    """A failed abandoned worker is consumed without an event-loop task warning."""
    state = ControlledAdapterState(
        native_waits_for_release=True,
        native_failure=Exception("native failure"),
    )
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={"transcription_timeout_seconds": 0.05}
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()
    handler_contexts: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: handler_contexts.append(context))
    try:
        async with application.router.lifespan_context(application):
            transport = ASGITransport(app=application)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                timeout_response = await client.post(
                    "/api/transcripts",
                    json={"url": DIRECT_TIKTOK_URL},
                )
                assert timeout_response.status_code == 504

                state.native_release.set()
                await _wait_for_thread_event(state.native_completed)
                await _wait_for_condition(
                    lambda: all(
                        not directory.exists()
                        for directory in state.request_directories
                    )
                )
                state.native_failure = None
                later_response = await client.post(
                    "/api/transcripts",
                    json={"url": DIRECT_TIKTOK_URL},
                )
    finally:
        loop.set_exception_handler(previous_exception_handler)

    assert later_response.status_code == 200
    await asyncio.sleep(0)
    assert handler_contexts == []


@pytest.mark.asyncio
async def test_api_shutdown_drains_abandoned_native_work(tmp_path: Path) -> None:
    """Lifespan exit waits for retained native work and its media cleanup."""
    state = ControlledAdapterState(native_waits_for_release=True)
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={"transcription_timeout_seconds": 0.05}
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )
    lifespan = application.router.lifespan_context(application)
    await lifespan.__aenter__()
    shutdown_task: asyncio.Task[bool | None] | None = None
    try:
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            timeout_response = await client.post(
                "/api/transcripts",
                json={"url": DIRECT_TIKTOK_URL},
            )
        assert timeout_response.status_code == 504

        shutdown_task = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        assert application.state.ready is False

        state.native_release.set()
        await asyncio.wait_for(asyncio.shield(shutdown_task), 1.0)
        await _wait_for_condition(
            lambda: all(
                not directory.exists() for directory in state.request_directories
            )
        )
    finally:
        if shutdown_task is None:
            state.native_release.set()
            await lifespan.__aexit__(None, None, None)
        elif not shutdown_task.done():
            state.native_release.set()
            await asyncio.wait_for(asyncio.shield(shutdown_task), 1.0)

    assert all(not directory.exists() for directory in state.request_directories)


@pytest.mark.asyncio
async def test_api_applies_each_stage_deadline_independently(tmp_path: Path) -> None:
    """Allow adjacent stages that each finish below their own deadline."""
    state = ControlledAdapterState(
        metadata_delay_seconds=0.04,
        download_delay_seconds=0.04,
    )
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={
                "metadata_timeout_seconds": 0.06,
                "audio_download_timeout_seconds": 0.06,
            }
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            started_at = time.monotonic()
            response = await client.post(
                "/api/transcripts",
                json={"url": DIRECT_TIKTOK_URL},
            )
            elapsed = time.monotonic() - started_at

    assert response.status_code == 200
    assert elapsed > 0.06
    assert state.call_order == ["metadata", "download", "native"]
    assert all(not directory.exists() for directory in state.request_directories)


@pytest.mark.asyncio
async def test_api_disconnect_returns_before_native_work_completes(
    tmp_path: Path,
) -> None:
    """A native disconnect returns while its prepared media remains retained."""
    state = ControlledAdapterState(
        metadata_prepares_audio=True,
        native_waits_for_release=True,
    )
    application = create_app(
        app_config(),
        transcription_config(tmp_path).model_copy(
            update={
                "transcription_concurrency": 1,
                "max_pending_transcriptions": 1,
                "transcription_queue_timeout_seconds": 1.0,
            }
        ),
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        disconnect_task, sent_messages = await _start_disconnect_request(
            application,
            state.native_entered,
        )
        await _wait_for_thread_event(state.native_entered)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(disconnect_task), 1.0)

        assert sent_messages == []
        assert state.prepared_directories[0].exists()

        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            queued_request = asyncio.create_task(
                client.post("/api/transcripts", json={"url": DIRECT_TIKTOK_URL})
            )
            await _wait_for_condition(lambda: len(state.prepared_directories) == 2)
            await _assert_request_pending(queued_request)
            assert len(state.native_calls) == 1

            overload_response = await client.post(
                "/api/transcripts",
                json={"url": DIRECT_TIKTOK_URL},
            )
            assert overload_response.status_code == 503
            assert len(state.metadata_calls) == 2

            state.native_release.set()
            queued_response = await _await_request(queued_request)
            await _wait_for_condition(
                lambda: all(
                    not directory.exists() for directory in state.prepared_directories
                )
            )

    assert queued_response.status_code == 200
    assert all(not directory.exists() for directory in state.prepared_directories)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("metadata", "download", "queue"))
async def test_api_disconnect_cancels_interruptible_pre_native_work(
    tmp_path: Path,
    stage: str,
) -> None:
    """Cancel metadata, download, or queue work without exposing a response."""
    if stage == "metadata":
        state = ControlledAdapterState(
            metadata_waits_for_cancellation=True,
            metadata_prepares_audio_after_cancellation=True,
        )
        config = transcription_config(tmp_path)
        stage_entry = state.metadata_entered
    elif stage == "download":
        state = ControlledAdapterState(download_waits_for_cancellation=True)
        config = transcription_config(tmp_path)
        stage_entry = state.download_entered
    else:
        state = ControlledAdapterState(native_waits_for_release=True)
        config = transcription_config(tmp_path).model_copy(
            update={"transcription_concurrency": 1}
        )
        stage_entry = state.download_second_entered

    application = create_app(
        app_config(),
        config,
        ControlledAdaptersFactory(state),
        available_temporary_media_bytes=_sufficient_temporary_media_bytes,
    )

    async with application.router.lifespan_context(application):
        if stage == "queue":
            transport = ASGITransport(app=application)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                active_request = asyncio.create_task(
                    client.post(
                        "/api/transcripts",
                        json={"url": DIRECT_TIKTOK_URL},
                    )
                )
                await _wait_for_thread_event(state.native_entered)
                disconnect_task, sent_messages = await _start_disconnect_request(
                    application,
                    stage_entry,
                )
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(disconnect_task), 1.0)

                assert sent_messages == []
                assert len(state.native_calls) == 1
                assert not state.request_directories[1].exists()

                state.native_release.set()
                active_response = await _await_request(active_request)
                assert active_response.status_code == 200
        else:
            disconnect_task, sent_messages = await _start_disconnect_request(
                application,
                stage_entry,
            )
            await _wait_for_thread_event(stage_entry)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(disconnect_task), 1.0)

            assert sent_messages == []
            assert state.native_calls == []
            if stage == "metadata":
                assert state.download_calls == []
                assert all(
                    not directory.exists() for directory in state.prepared_directories
                )
            else:
                assert all(
                    not directory.exists() for directory in state.request_directories
                )

    if stage == "queue":
        assert all(not directory.exists() for directory in state.request_directories)
