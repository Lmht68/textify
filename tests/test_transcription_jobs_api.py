"""ASGI proofs for successful durable Transcription Jobs."""

import asyncio
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict, cast
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from starlette.types import Message

from textify.config import AppConfig, Environment
from textify.jobs.config import JobConfig
from textify.jobs.service import utc_now
from textify.main import create_app
from textify.transcription import inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import (
    AudioDownloadFailedError,
    AudioDownloadTimeoutError,
    InvalidMediaDurationError,
    MetadataRetrievalFailedError,
    MetadataTimeoutError,
    NoUsableTranscriptError,
    TranscriptionError,
    TranscriptionFailedError,
    TranscriptionTimeoutError,
    UnsupportedContentError,
    UnsupportedMediaError,
    VideoTooLongError,
)
from textify.transcription.service import TranscriptionAdapters
from textify.transcription.types import (
    Platform,
    Segment,
    TimedTranscript,
    Transcript,
    TranscriptMethod,
)

TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
INSTAGRAM_URL = "https://www.instagram.com/reel/C0social_1"
FACEBOOK_URL = "https://www.facebook.com/watch?v=123456789012345"
X_URL = "https://x.com/creator/status/1234567890123456789"


class ActiveJobLinks(TypedDict):
    """Represent links while a Transcription Job remains active."""

    self: str
    cancel: str


class QueuedJobPayload(TypedDict):
    """Represent the public queued Transcription Job response."""

    id: str
    status: str
    submitted_at: str
    links: ActiveJobLinks


@dataclass
class ConcurrencyProbe:
    """Coordinate and observe concurrent controlled provider calls."""

    gated_metadata_urls: frozenset[str] = frozenset()
    gate_native_calls: bool = False
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
        repr=False,
    )
    _metadata_gate: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(0),
        init=False,
        repr=False,
    )
    _native_gate: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(0),
        init=False,
        repr=False,
    )
    _metadata_urls: list[str] = field(default_factory=list, init=False)
    _native_source_urls: list[str] = field(default_factory=list, init=False)
    _audio_source_urls: dict[Path, str] = field(default_factory=dict, init=False)
    _active_metadata_calls: int = field(default=0, init=False)
    _peak_metadata_calls: int = field(default=0, init=False)
    _active_native_calls: int = field(default=0, init=False)
    _peak_native_calls: int = field(default=0, init=False)

    @property
    def metadata_urls(self) -> tuple[str, ...]:
        """Return metadata provider URLs in entry order."""
        with self._condition:
            return tuple(self._metadata_urls)

    @property
    def native_source_urls(self) -> tuple[str, ...]:
        """Return native source URLs in entry order."""
        with self._condition:
            return tuple(self._native_source_urls)

    @property
    def active_metadata_calls(self) -> int:
        """Return the number of metadata calls currently executing."""
        with self._condition:
            return self._active_metadata_calls

    @property
    def peak_metadata_calls(self) -> int:
        """Return the peak number of concurrent metadata calls."""
        with self._condition:
            return self._peak_metadata_calls

    @property
    def active_native_calls(self) -> int:
        """Return the number of native calls currently executing."""
        with self._condition:
            return self._active_native_calls

    @property
    def peak_native_calls(self) -> int:
        """Return the peak number of concurrent native calls."""
        with self._condition:
            return self._peak_native_calls

    def enter_metadata(self, provider_url: str) -> None:
        """Record a metadata call and block configured provider URLs.

        Args:
            provider_url: Provider URL about to enter metadata extraction.
        """
        with self._condition:
            self._metadata_urls.append(provider_url)
            self._active_metadata_calls += 1
            self._peak_metadata_calls = max(
                self._peak_metadata_calls,
                self._active_metadata_calls,
            )
            self._condition.notify_all()
            should_block = provider_url in self.gated_metadata_urls
        if should_block:
            self._metadata_gate.acquire()

    def leave_metadata(self) -> None:
        """Record completion of one metadata call."""
        with self._condition:
            self._active_metadata_calls -= 1
            self._condition.notify_all()

    def register_audio_source(self, audio_path: Path, source_url: str) -> None:
        """Map a downloaded audio file to the provider source URL.

        Args:
            audio_path: Completed temporary media file.
            source_url: Provider URL supplied to the downloader.
        """
        with self._condition:
            self._audio_source_urls[audio_path] = source_url
            self._condition.notify_all()

    def enter_native(self, audio_path: Path) -> None:
        """Record a native call and block it when configured.

        Args:
            audio_path: Temporary media file supplied to native inference.

        Raises:
            AssertionError: If native inference receives an untracked audio file.
        """
        with self._condition:
            try:
                source_url = self._audio_source_urls[audio_path]
            except KeyError as exc:
                raise AssertionError(
                    "Native inference received untracked audio."
                ) from exc
            self._native_source_urls.append(source_url)
            self._active_native_calls += 1
            self._peak_native_calls = max(
                self._peak_native_calls,
                self._active_native_calls,
            )
            self._condition.notify_all()
            should_block = self.gate_native_calls
        if should_block:
            self._native_gate.acquire()

    def leave_native(self) -> None:
        """Record completion of one native inference call."""
        with self._condition:
            self._active_native_calls -= 1
            self._condition.notify_all()

    def release_metadata(self, count: int = 1) -> None:
        """Release configured metadata calls.

        Args:
            count: Number of metadata calls to release.

        Raises:
            ValueError: If ``count`` is not positive.
        """
        self._release(self._metadata_gate, count)

    def release_native(self, count: int = 1) -> None:
        """Release configured native calls.

        Args:
            count: Number of native calls to release.

        Raises:
            ValueError: If ``count`` is not positive.
        """
        self._release(self._native_gate, count)

    async def wait_for_metadata_calls(self, expected_count: int) -> None:
        """Wait asynchronously until metadata calls reach a target count.

        Args:
            expected_count: Minimum number of entered metadata calls.
        """
        await asyncio.to_thread(
            self._wait_for_call_count,
            expected_count,
            self._metadata_urls,
            "metadata",
        )

    async def wait_for_native_calls(self, expected_count: int) -> None:
        """Wait asynchronously until native calls reach a target count.

        Args:
            expected_count: Minimum number of entered native calls.
        """
        await asyncio.to_thread(
            self._wait_for_call_count,
            expected_count,
            self._native_source_urls,
            "native",
        )

    def _release(self, gate: threading.Semaphore, count: int) -> None:
        if count <= 0:
            raise ValueError("release count must be positive.")
        for _ in range(count):
            gate.release()

    def _wait_for_call_count(
        self,
        expected_count: int,
        calls: list[str],
        call_type: str,
    ) -> None:
        with self._condition:
            completed = self._condition.wait_for(
                lambda: len(calls) >= expected_count,
                timeout=2.0,
            )
            if not completed:
                raise AssertionError(
                    f"Timed out waiting for {expected_count} {call_type} calls."
                )


@dataclass
class ControlledAdapterState:
    """Coordinate deterministic provider work for Transcription Job ASGI tests."""

    block_metadata: bool = False
    caption_segments: tuple[tuple[float, float, str], ...] = ()
    caption_language: str = "en-US"
    probe: ConcurrencyProbe | None = None
    metadata_error: Exception | None = None
    download_error: Exception | None = None
    native_error: Exception | None = None
    metadata_delay_seconds: float = 0.0
    download_delay_seconds: float = 0.0
    native_delay_seconds: float = 0.0
    metadata_calls: list[str] = field(default_factory=list)
    download_calls: list[str] = field(default_factory=list)
    native_include_segments: list[bool] = field(default_factory=list)
    caption_calls: list[str] = field(default_factory=list)
    metadata_deadlines: list[float] = field(default_factory=list)
    download_deadlines: list[float] = field(default_factory=list)
    metadata_entry_times: list[float] = field(default_factory=list)
    download_entry_times: list[float] = field(default_factory=list)
    native_entry_times: list[float] = field(default_factory=list)
    request_directories: list[Path] = field(default_factory=list)
    audio_paths: list[Path] = field(default_factory=list)
    audio_paths_by_source: dict[str, Path] = field(default_factory=dict)
    metadata_entered: threading.Event = field(default_factory=threading.Event)
    metadata_release: threading.Event = field(default_factory=threading.Event)
    native_completed: threading.Event = field(default_factory=threading.Event)


class ControlledCaptionTrack:
    """Provide one deterministic original-language caption track."""

    def __init__(
        self,
        state: ControlledAdapterState,
    ) -> None:
        """Initialize the track with the shared observable state.

        Args:
            state: Provider state used to supply and observe captions.
        """
        self._state = state

    @property
    def language_code(self) -> str:
        """Return the configured original language code."""
        return self._state.caption_language

    @property
    def is_generated(self) -> bool:
        """Return whether this deterministic track was generated."""
        return False

    def fetch_segments(self) -> tuple[tuple[float, float, str], ...]:
        """Record and return normalized test caption segments."""
        self._state.caption_calls.append("fetch")
        return self._state.caption_segments


class ControlledCaptionProvider:
    """Expose optional deterministic captions without network access."""

    def __init__(self, state: ControlledAdapterState) -> None:
        """Initialize the provider with shared state.

        Args:
            state: Provider state used to expose optional caption tracks.
        """
        self._state = state

    def list_tracks(self, video_id: str) -> tuple[ControlledCaptionTrack, ...]:
        """Return the configured caption track only for caption-enabled tests.

        Args:
            video_id: Validated YouTube identifier supplied by the executor.

        Returns:
            The configured original-language track, if any.
        """
        self._state.caption_calls.append(f"list:{video_id}")
        if not self._state.caption_segments:
            return ()
        return (ControlledCaptionTrack(self._state),)


class ControlledMetadataExtractor:
    """Return valid external-shaped metadata and optionally block processing."""

    def __init__(self, state: ControlledAdapterState) -> None:
        """Initialize metadata behavior from shared state.

        Args:
            state: Provider state used to observe and block extraction.
        """
        self._state = state

    def extract(
        self,
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> inspection.ExtractedMetadata:
        """Return provider metadata for a validated submitted URL.

        Args:
            provider_url: URL classified by the durable worker.
            deadline: Monotonic operation deadline.
            cancellation_event: Worker-owned cooperative cancellation signal.

        Returns:
            Valid metadata accepted by the production normalization boundary.
        """
        probe = self._state.probe
        if probe is not None:
            probe.enter_metadata(provider_url)
        try:
            self._state.metadata_calls.append(provider_url)
            self._state.metadata_deadlines.append(deadline)
            self._state.metadata_entry_times.append(time.monotonic())
            self._state.metadata_entered.set()
            if self._state.block_metadata:
                while not self._state.metadata_release.wait(0.01):
                    assert not cancellation_event.is_set()
            if self._state.metadata_delay_seconds:
                time.sleep(self._state.metadata_delay_seconds)
            if self._state.metadata_error is not None:
                raise self._state.metadata_error
            return inspection.ExtractedMetadata(_metadata_for(provider_url))
        finally:
            if probe is not None:
                probe.leave_metadata()


class ControlledAudioDownloader:
    """Create small deterministic native-audio inputs."""

    def __init__(self, state: ControlledAdapterState) -> None:
        """Initialize download observation state.

        Args:
            state: Provider state used to record native fallback downloads.
        """
        self._state = state

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Create a bounded audio file for the production acquisition path.

        Args:
            source_url: Provider URL selected by URL inspection.
            destination: Existing private native-work directory.
            deadline: Monotonic operation deadline.
            cancellation_event: Worker-owned cooperative cancellation signal.

        Returns:
            A completed small local media file.
        """
        assert not cancellation_event.is_set()
        self._state.download_calls.append(source_url)
        self._state.download_deadlines.append(deadline)
        self._state.download_entry_times.append(time.monotonic())
        self._state.request_directories.append(destination)
        if self._state.download_error is not None:
            raise self._state.download_error
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"audio")
        self._state.audio_paths.append(audio_path)
        self._state.audio_paths_by_source[source_url] = audio_path
        if self._state.probe is not None:
            self._state.probe.register_audio_source(audio_path, source_url)
        if self._state.download_delay_seconds:
            time.sleep(self._state.download_delay_seconds)
        return audio_path


class ControlledTranscriber:
    """Return deterministic normalized native transcripts."""

    def __init__(self, state: ControlledAdapterState) -> None:
        """Initialize native execution observation state.

        Args:
            state: Provider state used to record segment requests.
        """
        self._state = state

    def transcribe(
        self, audio_path: Path, *, include_segments: bool = True
    ) -> Transcript:
        """Return a timed or text-only transcript according to the requested projection.

        Args:
            audio_path: Completed temporary audio created by the controlled downloader.
            include_segments: Whether normalized segments must be retained.

        Returns:
            A valid normalized transcript.
        """
        assert audio_path.is_file()
        self._state.native_entry_times.append(time.monotonic())
        probe = self._state.probe
        if probe is not None:
            probe.enter_native(audio_path)
        try:
            self._state.native_include_segments.append(include_segments)
            self._state.native_completed.set()
            if self._state.native_delay_seconds:
                time.sleep(self._state.native_delay_seconds)
            if self._state.native_error is not None:
                raise self._state.native_error
            if not include_segments:
                return Transcript(TranscriptMethod.FASTER_WHISPER, "en", "One")
            return TimedTranscript(
                TranscriptMethod.FASTER_WHISPER,
                "en",
                "One",
                (Segment(0.0, 1.0, "One"),),
            )
        finally:
            if probe is not None:
                probe.leave_native()


class ControlledAdaptersFactory:
    """Build production-shape adapters around deterministic provider state."""

    def __init__(self, state: ControlledAdapterState) -> None:
        """Initialize the factory.

        Args:
            state: Shared provider behavior and observations.
        """
        self.state = state
        self.calls = 0

    def __call__(self, _settings: TranscriptionConfig) -> TranscriptionAdapters:
        """Construct one production adapter bundle.

        Args:
            _settings: Validated application configuration unused by test adapters.

        Returns:
            Complete provider boundary bundle.
        """
        self.calls += 1
        return TranscriptionAdapters(
            metadata_extractor=ControlledMetadataExtractor(self.state),
            caption_provider=ControlledCaptionProvider(self.state),
            audio_downloader=ControlledAudioDownloader(self.state),
            whisper_transcriber=ControlledTranscriber(self.state),
        )


def _metadata_for(provider_url: str) -> dict[str, object]:
    """Return valid provider metadata for every supported direct test platform."""
    if "youtube" in provider_url or "youtu.be" in provider_url:
        return {
            "id": "dQw4w9WgXcQ",
            "extractor_key": "Youtube",
            "title": "Title",
            "description": "Description",
            "channel": "Creator",
            "duration": 12,
            "language": "en-US",
        }
    if "instagram" in provider_url:
        platform = Platform.INSTAGRAM
        video_id = "C0social_1"
        canonical_url = INSTAGRAM_URL
    elif "facebook" in provider_url or "fb.watch" in provider_url:
        platform = Platform.FACEBOOK
        video_id = "123456789012345"
        canonical_url = FACEBOOK_URL
    elif (
        "x.com" in provider_url
        or "twitter.com" in provider_url
        or "t.co" in provider_url
    ):
        platform = Platform.X
        video_id = "1234567890123456789"
        canonical_url = X_URL
    else:
        platform = Platform.TIKTOK
        video_id = "1234567890123456789"
        canonical_url = TIKTOK_URL
    extractor_key = {
        Platform.INSTAGRAM: "Instagram",
        Platform.FACEBOOK: "Facebook",
        Platform.TIKTOK: "TikTok",
        Platform.X: "Twitter",
    }[platform]
    formats: tuple[dict[str, str], ...]
    if platform is Platform.X:
        formats = (
            {"url": "https://video.twimg.com/example.mp4", "format_id": "http-832"},
        )
    else:
        formats = ({"vcodec": "h264"},)
    return {
        "id": video_id,
        "extractor_key": extractor_key,
        "webpage_url": canonical_url,
        "title": "Title",
        "description": "Description",
        "channel": "Creator",
        "duration": 12,
        "formats": formats,
    }


def _app_config() -> AppConfig:
    """Create isolated application configuration for durable-job tests."""
    return AppConfig(
        environment=Environment.LOCAL,
        log_level="INFO",
        host="127.0.0.1",
        port=8182,
    )


def _job_config(
    database_path: Path | None,
    *,
    job_worker_count: int = 1,
    job_queue_timeout_seconds: int = 20,
) -> JobConfig:
    """Create isolated durable-job configuration for durable-job tests."""
    return JobConfig(
        database_path=database_path,
        job_worker_count=job_worker_count,
        max_outstanding_jobs=8,
        job_queue_timeout_seconds=job_queue_timeout_seconds,
    )


def _transcription_config(
    temporary_media_root: Path,
    *,
    transcription_concurrency: int = 1,
    metadata_timeout_seconds: float = 30.0,
    audio_download_timeout_seconds: float = 300.0,
    transcription_timeout_seconds: float = 1800.0,
) -> TranscriptionConfig:
    """Create native-capacity settings for durable-job tests."""
    return TranscriptionConfig(
        temporary_media_root=temporary_media_root,
        max_media_bytes=1024,
        transcription_concurrency=transcription_concurrency,
        metadata_timeout_seconds=metadata_timeout_seconds,
        audio_download_timeout_seconds=audio_download_timeout_seconds,
        transcription_timeout_seconds=transcription_timeout_seconds,
    )


def _application(
    database_path: Path | None,
    temporary_media_root: Path,
    factory: ControlledAdaptersFactory,
    *,
    settings: TranscriptionConfig | None = None,
    job_config: JobConfig | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> FastAPI:
    """Build one application with real durable storage and controlled providers."""
    return create_app(
        _app_config(),
        settings
        if settings is not None
        else _transcription_config(temporary_media_root),
        factory,
        available_temporary_media_bytes=lambda _root: 1 << 60,
        job_config=(
            job_config if job_config is not None else _job_config(database_path)
        ),
        clock=clock,
    )


async def _wait_for_thread_event(event: threading.Event) -> None:
    """Wait for a synchronous provider event without blocking the event loop."""
    assert await asyncio.to_thread(event.wait, 2.0)


async def _poll_until_finished(
    client: AsyncClient,
    location: str,
) -> Response:
    """Poll one durable capability until it exposes a terminal snapshot."""
    latest_response: Response | None = None
    for _ in range(200):
        response = await client.get(location)
        latest_response = response
        if response.json().get("status") == "finished":
            return response
        await asyncio.sleep(0.01)
    raise AssertionError(f"Transcription Job did not finish: {latest_response!r}")


async def _wait_until(
    predicate: Callable[[], bool],
    failure_message: str,
) -> None:
    """Wait for an observable asynchronous condition without blocking the event loop."""
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(failure_message)


def _assert_generated_request_id(response: Response) -> str:
    """Assert and return one canonical server-generated request identifier."""
    request_id = response.headers["X-Request-ID"]
    parsed_request_id = UUID(request_id)
    assert parsed_request_id.version == 4
    assert str(parsed_request_id) == request_id
    return request_id


def _assert_submission(response: Response) -> QueuedJobPayload:
    """Assert the complete durable submission contract."""
    payload = cast(QueuedJobPayload, response.json())
    assert response.status_code == 202
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
    assert response.headers["Location"] == payload["links"]["self"]
    assert response.headers["Retry-After"] == "2"
    assert response.headers["Cache-Control"] == "no-store"
    _assert_generated_request_id(response)
    return payload


def _assert_processing(response: Response, location: str) -> dict[str, object]:
    """Assert the public active representation after a worker claim."""
    payload = response.json()
    assert response.status_code == 200
    assert set(payload) == {
        "id",
        "status",
        "submitted_at",
        "started_at",
        "cancellation_requested",
        "links",
    }
    assert payload["status"] == "processing"
    assert payload["cancellation_requested"] is False
    assert payload["links"] == {"self": location, "cancel": f"{location}/cancellation"}
    started_at = datetime.fromisoformat(payload["started_at"])
    assert started_at.tzinfo is not None
    assert response.headers["Retry-After"] == "2"
    assert response.headers["Cache-Control"] == "no-store"
    _assert_generated_request_id(response)
    return cast(dict[str, object], payload)


def _assert_succeeded(response: Response, location: str) -> dict[str, object]:
    """Assert the complete successful terminal representation and headers."""
    payload = response.json()
    assert response.status_code == 200
    assert set(payload) == {
        "id",
        "status",
        "outcome",
        "submitted_at",
        "started_at",
        "finished_at",
        "result",
        "links",
    }
    assert payload["status"] == "finished"
    assert payload["outcome"] == "succeeded"
    assert payload["links"] == {"self": location}
    finished_at = datetime.fromisoformat(payload["finished_at"])
    assert finished_at.tzinfo is not None
    assert "Retry-After" not in response.headers
    assert response.headers["Cache-Control"] == "no-store"
    _assert_generated_request_id(response)
    return cast(dict[str, object], payload)


def _assert_failed(
    response: Response,
    location: str,
    expected_code: str,
    *,
    started: bool,
) -> dict[str, object]:
    """Assert the complete safe terminal representation of a failed job."""
    payload = response.json()
    assert response.status_code == 200
    assert set(payload) == {
        "id",
        "status",
        "outcome",
        "submitted_at",
        "started_at",
        "finished_at",
        "error",
        "links",
    }
    assert payload["status"] == "finished"
    assert payload["outcome"] == "failed"
    submitted_at = datetime.fromisoformat(payload["submitted_at"])
    assert submitted_at.tzinfo is not None
    assert submitted_at.astimezone(UTC).utcoffset() == UTC.utcoffset(submitted_at)
    if started:
        started_at = datetime.fromisoformat(payload["started_at"])
        assert started_at.tzinfo is not None
        assert started_at.astimezone(UTC).utcoffset() == UTC.utcoffset(started_at)
    else:
        assert payload["started_at"] is None
    finished_at = datetime.fromisoformat(payload["finished_at"])
    assert finished_at.tzinfo is not None
    assert finished_at.astimezone(UTC).utcoffset() == UTC.utcoffset(finished_at)
    assert payload["error"]["code"] == expected_code
    assert isinstance(payload["error"]["message"], str)
    assert payload["error"]["message"]
    assert payload["links"] == {"self": location}
    assert "Retry-After" not in response.headers
    assert response.headers["Cache-Control"] == "no-store"
    _assert_generated_request_id(response)
    return cast(dict[str, object], payload)


def _assert_safe_error(
    response: Response, expected_status: int, expected_code: str
) -> None:
    """Assert the stable safe error envelope for bearer-capability routes."""
    payload = response.json()
    assert response.status_code == expected_status
    assert response.headers["Cache-Control"] == "no-store"
    assert payload["error"]["code"] == expected_code
    assert isinstance(payload["error"]["message"], str)
    assert payload["error"]["message"]
    _assert_generated_request_id(response)


async def test_api_processes_persists_and_reopens_successful_job(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Publish one blocked job atomically, then read it after application restart."""
    first_state = ControlledAdapterState(block_metadata=True)
    first_factory = ControlledAdaptersFactory(first_state)
    first_application = _application(
        migrated_database_path,
        tmp_path / "first-media",
        first_factory,
    )
    async with first_application.router.lifespan_context(first_application):
        transport = ASGITransport(app=first_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs", json={"url": TIKTOK_URL}
            )
            _assert_submission(accepted)
            location = accepted.headers["Location"]
            await _wait_for_thread_event(first_state.metadata_entered)
            processing = await client.get(location)
            _assert_processing(processing, location)
            assert _assert_generated_request_id(
                accepted
            ) != _assert_generated_request_id(processing)
            first_state.metadata_release.set()
            finished = await _poll_until_finished(client, location)

    finished_payload = _assert_succeeded(finished, location)
    expected_result = finished_payload["result"]
    assert isinstance(expected_result, dict)
    assert expected_result["source"]["platform"] == "tiktok"
    assert expected_result["transcript"]["text"] == "One"
    assert first_state.native_include_segments == [True]
    with sqlite3.connect(migrated_database_path) as connection:
        lifecycle = connection.execute(
            "SELECT status, outcome FROM transcription_job"
        ).fetchall()
        result_rows = connection.execute(
            "SELECT * FROM transcription_job_result"
        ).fetchall()
        segments = connection.execute(
            "SELECT ordinal, text FROM transcription_job_segment ORDER BY ordinal"
        ).fetchall()
    assert lifecycle == [("finished", "succeeded")]
    assert len(result_rows) == 1
    assert segments == [(0, "One")]

    reopened_state = ControlledAdapterState()
    reopened_factory = ControlledAdaptersFactory(reopened_state)
    reopened_application = _application(
        migrated_database_path,
        tmp_path / "reopened-media",
        reopened_factory,
    )
    async with reopened_application.router.lifespan_context(reopened_application):
        transport = ASGITransport(app=reopened_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            reopened = await client.get(location)

    reopened_payload = _assert_succeeded(reopened, location)
    assert reopened_payload["result"] == expected_result
    assert reopened_state.metadata_calls == []
    assert reopened_state.download_calls == []


@pytest.mark.parametrize(
    "excluded_path",
    (
        "source.platform",
        "source.video_id",
        "source.url",
        "source.title",
        "source.description",
        "source.channel",
        "source.duration_seconds",
        "transcript.method",
        "transcript.language",
        "transcript.text",
        "transcript.segments",
    ),
)
async def test_api_persists_every_requested_projection_exclusion(
    migrated_database_path: Path,
    tmp_path: Path,
    excluded_path: str,
) -> None:
    """Persist each response-field exclusion without dropping retained typed values."""
    state = ControlledAdapterState()
    application = _application(
        migrated_database_path, tmp_path / "media", ControlledAdaptersFactory(state)
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL, "exclude": [excluded_path]},
            )
            location = _assert_submission(accepted)["links"]["self"]
            finished = await _poll_until_finished(client, location)

    result = _assert_succeeded(finished, location)["result"]
    assert isinstance(result, dict)
    root, leaf = excluded_path.split(".")
    assert leaf not in result[root]
    assert result["source"]
    assert result["transcript"]
    if excluded_path != "source.platform":
        assert result["source"]["platform"] == "tiktok"
    if excluded_path != "source.duration_seconds":
        assert isinstance(result["source"]["duration_seconds"], int)
    if excluded_path != "transcript.method":
        assert result["transcript"]["method"] == "faster_whisper"
    if excluded_path != "transcript.segments":
        assert result["transcript"]["segments"] == [
            {"start": 0.0, "end": 1.0, "text": "One"}
        ]
    assert state.native_include_segments == [excluded_path != "transcript.segments"]
    with sqlite3.connect(migrated_database_path) as connection:
        segment_count = connection.execute(
            "SELECT COUNT(*) FROM transcription_job_segment"
        ).fetchone()[0]
    assert segment_count == (0 if excluded_path == "transcript.segments" else 1)


@pytest.mark.parametrize(
    ("submitted_url", "expected_platform", "uses_caption"),
    (
        (TIKTOK_URL, "tiktok", False),
        (INSTAGRAM_URL, "instagram", False),
        (FACEBOOK_URL, "facebook", False),
        (X_URL, "x", False),
        (YOUTUBE_URL, "youtube", True),
    ),
)
async def test_api_executes_supported_platforms_through_submission_and_polling(
    migrated_database_path: Path,
    tmp_path: Path,
    submitted_url: str,
    expected_platform: str,
    uses_caption: bool,
) -> None:
    """Keep provider orchestration behind the asynchronous Transcription Job path."""
    state = ControlledAdapterState(
        caption_segments=((0.0, 1.0, "Caption"),) if uses_caption else ()
    )
    application = _application(
        migrated_database_path, tmp_path / "media", ControlledAdaptersFactory(state)
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs", json={"url": submitted_url}
            )
            location = _assert_submission(accepted)["links"]["self"]
            finished = await _poll_until_finished(client, location)

    result = _assert_succeeded(finished, location)["result"]
    assert isinstance(result, dict)
    assert result["source"]["platform"] == expected_platform
    expected_method = "youtube_captions" if uses_caption else "faster_whisper"
    assert result["transcript"]["method"] == expected_method
    assert bool(state.download_calls) is not uses_caption
    assert state.native_include_segments == ([] if uses_caption else [True])


async def test_api_keeps_minimum_and_empty_projection_contracts(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Retain valid minimum projections and preserve an explicit empty projection list."""
    state = ControlledAdapterState()
    application = _application(
        migrated_database_path, tmp_path / "media", ControlledAdaptersFactory(state)
    )
    minimum_exclusions = [
        "source.video_id",
        "source.url",
        "source.title",
        "source.description",
        "source.channel",
        "source.duration_seconds",
        "transcript.method",
        "transcript.language",
        "transcript.segments",
    ]
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            minimum_accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL, "exclude": minimum_exclusions},
            )
            empty_accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL, "exclude": []},
            )
            minimum_finished = await _poll_until_finished(
                client,
                _assert_submission(minimum_accepted)["links"]["self"],
            )
            empty_finished = await _poll_until_finished(
                client,
                _assert_submission(empty_accepted)["links"]["self"],
            )

    minimum_result = _assert_succeeded(
        minimum_finished,
        minimum_finished.request.url.path,
    )["result"]
    empty_result = _assert_succeeded(empty_finished, empty_finished.request.url.path)[
        "result"
    ]
    assert minimum_result == {
        "source": {"platform": "tiktok"},
        "transcript": {"text": "One"},
    }
    assert isinstance(empty_result, dict)
    assert set(empty_result["source"]) == {
        "platform",
        "video_id",
        "url",
        "title",
        "description",
        "channel",
        "duration_seconds",
    }
    assert state.native_include_segments == [False, True]


async def test_api_uses_native_fallback_when_youtube_captions_are_unrelated(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Keep caption selection and native fallback behavior under worker ownership."""
    state = ControlledAdapterState(
        caption_segments=((0.0, 1.0, "bonjour"),),
        caption_language="fr",
    )
    application = _application(
        migrated_database_path, tmp_path / "media", ControlledAdaptersFactory(state)
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs", json={"url": YOUTUBE_URL}
            )
            location = _assert_submission(accepted)["links"]["self"]
            finished = await _poll_until_finished(client, location)

    result = _assert_succeeded(finished, location)["result"]
    assert isinstance(result, dict)
    assert result["transcript"]["method"] == "faster_whisper"
    assert state.caption_calls == ["list:dQw4w9WgXcQ"]
    assert state.native_include_segments == [True]


async def test_api_rejects_invalid_unknown_and_failed_admission_safely(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Keep immediate validation and durable-store errors outside provider execution."""
    state = ControlledAdapterState()
    application = _application(
        migrated_database_path, tmp_path / "media", ControlledAdaptersFactory(state)
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            invalid = await client.post(
                "/api/transcription-jobs", json={"url": TIKTOK_URL, "unknown": "value"}
            )
            unsupported = await client.post(
                "/api/transcription-jobs", json={"url": "https://example.com/video"}
            )
            unknown = await client.get(
                "/api/transcription-jobs/00000000-0000-4000-8000-000000000000"
            )

    _assert_safe_error(invalid, 422, "invalid_request")
    _assert_safe_error(unsupported, 400, "unsupported_platform")
    _assert_safe_error(unknown, 404, "job_not_found")
    assert state.metadata_calls == []


async def test_api_bounds_outstanding_jobs_while_the_consumer_is_blocked(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Keep eight durable jobs outstanding and reject the ninth admission."""
    state = ControlledAdapterState(block_metadata=True)
    application = _application(
        migrated_database_path, tmp_path / "media", ControlledAdaptersFactory(state)
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post(
                "/api/transcription-jobs", json={"url": TIKTOK_URL}
            )
            _assert_submission(first)
            await _wait_for_thread_event(state.metadata_entered)
            responses = await asyncio.gather(
                *[
                    client.post("/api/transcription-jobs", json={"url": TIKTOK_URL})
                    for _ in range(8)
                ]
            )
            accepted = [
                response for response in responses if response.status_code == 202
            ]
            rejected = [
                response for response in responses if response.status_code == 503
            ]
            assert len(accepted) == 7
            assert len(rejected) == 1
            _assert_safe_error(rejected[0], 503, "transcription_capacity_exceeded")
            with sqlite3.connect(migrated_database_path) as connection:
                outstanding_count = connection.execute(
                    "SELECT COUNT(*) FROM transcription_job "
                    "WHERE status IN ('queued', 'processing')"
                ).fetchone()[0]
            assert outstanding_count == 8
            state.metadata_release.set()


async def test_api_rolls_back_result_when_terminal_transition_is_aborted(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Never expose success when result insertion and terminal transition cannot commit."""
    with sqlite3.connect(migrated_database_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER transcription_job_finish_abort
            BEFORE UPDATE OF status ON transcription_job
            WHEN OLD.status = 'processing' AND NEW.status = 'finished'
            BEGIN
                SELECT RAISE(ABORT, 'blocked finish');
            END;
            """
        )
    state = ControlledAdapterState(block_metadata=True)
    application = _application(
        migrated_database_path, tmp_path / "media", ControlledAdaptersFactory(state)
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs", json={"url": TIKTOK_URL}
            )
            location = _assert_submission(accepted)["links"]["self"]
            await _wait_for_thread_event(state.metadata_entered)
            state.metadata_release.set()
            await _wait_for_thread_event(state.native_completed)
            await asyncio.sleep(0.05)
            processing = await client.get(location)
            _assert_processing(processing, location)

    with sqlite3.connect(migrated_database_path) as connection:
        result_count = connection.execute(
            "SELECT COUNT(*) FROM transcription_job_result"
        ).fetchone()[0]
        segment_count = connection.execute(
            "SELECT COUNT(*) FROM transcription_job_segment"
        ).fetchone()[0]
    assert result_count == 0
    assert segment_count == 0


async def test_api_completes_submission_after_response_delivery_disconnect(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Keep an accepted job independent from the client that submitted it."""
    state = ControlledAdapterState(block_metadata=True)
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
    )
    delivery_started = asyncio.Event()
    location_holder: list[str] = []
    request_body = (
        b'{"url":"https://www.tiktok.com/@creator/video/1234567890123456789"}'
    )
    delivered_body = False

    async def receive() -> dict[str, object]:
        """Provide the request body once, then report that the client disconnected."""
        nonlocal delivered_body
        if not delivered_body:
            delivered_body = True
            return {
                "type": "http.request",
                "body": request_body,
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        """Capture the committed Location and then block response delivery."""
        if message["type"] == "http.response.start":
            headers = cast(list[tuple[bytes, bytes]], message["headers"])
            location_holder.append(dict(headers)[b"location"].decode())
            delivery_started.set()
            await asyncio.Event().wait()

    async with application.router.lifespan_context(application):
        disconnected_request = asyncio.create_task(
            application(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.3"},
                    "http_version": "1.1",
                    "server": ("test", 80),
                    "scheme": "http",
                    "method": "POST",
                    "root_path": "",
                    "path": "/api/transcription-jobs",
                    "raw_path": b"/api/transcription-jobs",
                    "query_string": b"",
                    "headers": [
                        (b"host", b"test"),
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(request_body)).encode()),
                    ],
                    "client": ("test", 50000),
                },
                receive,
                send,
            )
        )
        await delivery_started.wait()
        disconnected_request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await disconnected_request
        if not location_holder:
            raise AssertionError(
                "Committed submission did not expose a Location header."
            )
        committed_location = location_holder[0]
        await _wait_for_thread_event(state.metadata_entered)
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            processing = await client.get(committed_location)
            _assert_processing(processing, committed_location)
            state.metadata_release.set()
            finished = await _poll_until_finished(client, committed_location)

    _assert_succeeded(finished, committed_location)


@pytest.mark.parametrize("failure_case", ("missing", "empty", "stale", "read_only"))
async def test_api_fails_startup_before_adapter_construction_for_invalid_storage(
    failure_case: str,
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Reject invalid durable storage before creating provider adapters."""
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

    factory = ControlledAdaptersFactory(ControlledAdapterState())
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


async def test_api_claims_each_job_once_in_fifo_order_with_four_consumers(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Claim queued caption jobs once and in FIFO order across four consumers."""
    provider_urls = tuple(
        f"{YOUTUBE_URL}&queue_position={position}" for position in range(8)
    )
    probe = ConcurrencyProbe(gated_metadata_urls=frozenset(provider_urls))
    state = ControlledAdapterState(
        caption_segments=((0.0, 1.0, "Caption"),),
        probe=probe,
    )
    factory = ControlledAdaptersFactory(state)
    settings = _transcription_config(
        tmp_path / "media",
        transcription_concurrency=2,
    )
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        factory,
        settings=settings,
        job_config=_job_config(migrated_database_path, job_worker_count=4),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                first_submissions = [
                    _assert_submission(
                        await client.post(
                            "/api/transcription-jobs",
                            json={"url": provider_url},
                        )
                    )
                    for provider_url in provider_urls[:4]
                ]
                await probe.wait_for_metadata_calls(4)
                second_submissions = [
                    _assert_submission(
                        await client.post(
                            "/api/transcription-jobs",
                            json={"url": provider_url},
                        )
                    )
                    for provider_url in provider_urls[4:]
                ]
                first_locations = [
                    submission["links"]["self"] for submission in first_submissions
                ]
                second_locations = [
                    submission["links"]["self"] for submission in second_submissions
                ]
                processing_responses = await asyncio.gather(
                    *(client.get(location) for location in first_locations)
                )
                for response, location in zip(
                    processing_responses, first_locations, strict=True
                ):
                    _assert_processing(response, location)
                queued_responses = await asyncio.gather(
                    *(client.get(location) for location in second_locations)
                )
                assert [response.json()["status"] for response in queued_responses] == [
                    "queued"
                ] * 4
                assert frozenset(probe.metadata_urls) == frozenset(provider_urls[:4])
                assert probe.peak_metadata_calls == 4

                for expected_count, provider_url in enumerate(
                    provider_urls[4:],
                    start=5,
                ):
                    probe.release_metadata()
                    await probe.wait_for_metadata_calls(expected_count)
                    assert probe.metadata_urls[-1] == provider_url

                probe.release_metadata(4)
                finished_responses = await asyncio.gather(
                    *(
                        _poll_until_finished(client, location)
                        for location in (*first_locations, *second_locations)
                    )
                )
            finally:
                probe.release_metadata(8)

    for response, location in zip(
        finished_responses,
        (*first_locations, *second_locations),
        strict=True,
    ):
        _assert_succeeded(response, location)
    assert len(probe.metadata_urls) == len(provider_urls)
    assert frozenset(probe.metadata_urls) == frozenset(provider_urls)
    assert probe.native_source_urls == ()
    assert state.download_calls == []
    assert factory.calls == 1


async def test_api_limits_native_inference_to_two_while_captions_bypass_permits(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Keep native inference at two calls while caption jobs bypass its permits."""
    native_urls = tuple(
        f"{TIKTOK_URL}?native_position={position}" for position in range(3)
    )
    caption_url = f"{YOUTUBE_URL}&caption_position=1"
    probe = ConcurrencyProbe(gate_native_calls=True)
    state = ControlledAdapterState(
        caption_segments=((0.0, 1.0, "Caption"),),
        probe=probe,
    )
    factory = ControlledAdaptersFactory(state)
    settings = _transcription_config(
        tmp_path / "media",
        transcription_concurrency=2,
    )
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        factory,
        settings=settings,
        job_config=_job_config(migrated_database_path, job_worker_count=4),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                native_submissions = [
                    _assert_submission(
                        await client.post(
                            "/api/transcription-jobs",
                            json={"url": provider_url},
                        )
                    )
                    for provider_url in native_urls
                ]
                await probe.wait_for_native_calls(2)
                caption_submission = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": caption_url},
                    )
                )
                caption_location = caption_submission["links"]["self"]
                caption_finished = await _poll_until_finished(client, caption_location)
                assert probe.active_native_calls == 2
                assert len(probe.native_source_urls) == 2
                assert probe.peak_native_calls == 2
                assert caption_url not in probe.native_source_urls
                assert caption_url not in state.download_calls

                probe.release_native(3)
                native_locations = [
                    submission["links"]["self"] for submission in native_submissions
                ]
                native_finished = await asyncio.gather(
                    *(
                        _poll_until_finished(client, location)
                        for location in native_locations
                    )
                )
            finally:
                probe.release_native(3)

    caption_result = _assert_succeeded(caption_finished, caption_location)["result"]
    assert isinstance(caption_result, dict)
    assert caption_result["transcript"]["method"] == "youtube_captions"
    for response, location in zip(native_finished, native_locations, strict=True):
        _assert_succeeded(response, location)
    assert probe.peak_native_calls == 2
    assert factory.calls == 1


async def test_api_retains_timed_out_native_permits_until_underlying_calls_finish(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Retain timed-out native permits until their underlying threads complete."""
    native_urls = tuple(
        f"{TIKTOK_URL}?timeout_position={position}" for position in range(4)
    )
    caption_urls = tuple(
        f"{YOUTUBE_URL}&timeout_caption_position={position}" for position in range(2)
    )
    probe = ConcurrencyProbe(
        gated_metadata_urls=frozenset(caption_urls),
        gate_native_calls=True,
    )
    state = ControlledAdapterState(
        caption_segments=((0.0, 1.0, "Caption"),),
        probe=probe,
    )
    factory = ControlledAdaptersFactory(state)
    settings = _transcription_config(
        tmp_path / "media",
        transcription_concurrency=2,
        transcription_timeout_seconds=0.05,
    )
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        factory,
        settings=settings,
        job_config=_job_config(migrated_database_path, job_worker_count=4),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                native_submissions = {
                    provider_url: _assert_submission(
                        await client.post(
                            "/api/transcription-jobs",
                            json={"url": provider_url},
                        )
                    )
                    for provider_url in native_urls
                }
                await probe.wait_for_native_calls(2)
                timed_native_urls = probe.native_source_urls
                timed_locations = [
                    native_submissions[provider_url]["links"]["self"]
                    for provider_url in timed_native_urls
                ]
                timed_media_paths = [
                    state.audio_paths_by_source[provider_url]
                    for provider_url in timed_native_urls
                ]
                timed_failed = await asyncio.gather(
                    *(
                        _poll_until_finished(client, location)
                        for location in timed_locations
                    )
                )
                for response, location in zip(
                    timed_failed, timed_locations, strict=True
                ):
                    _assert_failed(
                        response,
                        location,
                        "transcription_timeout",
                        started=True,
                    )
                assert all(audio_path.is_file() for audio_path in timed_media_paths)
                assert probe.active_native_calls == 2
                assert len(probe.native_source_urls) == 2
                caption_submissions = [
                    _assert_submission(
                        await client.post(
                            "/api/transcription-jobs",
                            json={"url": provider_url},
                        )
                    )
                    for provider_url in caption_urls
                ]
                await probe.wait_for_metadata_calls(6)
                assert probe.active_metadata_calls == 2
                assert frozenset(probe.metadata_urls[-2:]) == frozenset(caption_urls)
                assert probe.active_native_calls == 2
                assert len(probe.native_source_urls) == 2

                probe.release_metadata(2)
                caption_locations = [
                    submission["links"]["self"] for submission in caption_submissions
                ]
                caption_finished = await asyncio.gather(
                    *(
                        _poll_until_finished(client, location)
                        for location in caption_locations
                    )
                )
                assert len(probe.native_source_urls) == 2
                assert all(
                    caption_url not in probe.native_source_urls
                    and caption_url not in state.download_calls
                    for caption_url in caption_urls
                )

                probe.release_native(4)
                await probe.wait_for_native_calls(4)
                await _wait_until(
                    lambda: all(
                        not audio_path.exists() for audio_path in timed_media_paths
                    ),
                    "Timed-out native media was not released by its finalizer.",
                )
                for location in timed_locations:
                    _assert_failed(
                        await client.get(location),
                        location,
                        "transcription_timeout",
                        started=True,
                    )
                assert probe.peak_native_calls == 2
            finally:
                probe.release_metadata(2)
                probe.release_native(4)

    for response, location in zip(
        caption_finished,
        caption_locations,
        strict=True,
    ):
        _assert_succeeded(response, location)
    assert probe.peak_native_calls == 2
    assert factory.calls == 1


@pytest.mark.parametrize(
    ("stage", "error_type"),
    (
        ("metadata", UnsupportedContentError),
        ("metadata", VideoTooLongError),
        ("metadata", InvalidMediaDurationError),
        ("metadata", UnsupportedMediaError),
        ("metadata", MetadataRetrievalFailedError),
        ("metadata", MetadataTimeoutError),
        ("download", AudioDownloadFailedError),
        ("download", AudioDownloadTimeoutError),
        ("native", NoUsableTranscriptError),
        ("native", TranscriptionFailedError),
        ("native", TranscriptionTimeoutError),
    ),
)
async def test_api_persists_each_safe_execution_failure(
    stage: str,
    error_type: type[TranscriptionError],
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Expose every safe worker failure as one durable failed job."""
    state = ControlledAdapterState()
    error = error_type()
    if stage == "metadata":
        state.metadata_error = error
    elif stage == "download":
        state.download_error = error
    else:
        state.native_error = error
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL},
            )
            location = _assert_submission(accepted)["links"]["self"]
            failed = await _poll_until_finished(client, location)

    _assert_failed(failed, location, error.code, started=True)
    assert state.metadata_calls == [TIKTOK_URL]
    if stage == "metadata":
        assert state.download_calls == []
        assert state.native_include_segments == []
    elif stage == "download":
        assert state.download_calls == [TIKTOK_URL]
        assert state.native_include_segments == []
    else:
        assert state.download_calls == [TIKTOK_URL]
        assert state.native_include_segments == [True]


async def test_api_expires_queue_waits_at_the_deadline_without_provider_work(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Claim before a deadline and expire equal-deadline work without execution."""
    initial = datetime(2026, 8, 24, tzinfo=UTC)
    now = [initial]
    first_url = f"{TIKTOK_URL}?queue_deadline=first"
    before_deadline_url = f"{TIKTOK_URL}?queue_deadline=before"
    expired_url = f"{TIKTOK_URL}?queue_deadline=expired"
    probe = ConcurrencyProbe(
        gated_metadata_urls=frozenset({first_url, before_deadline_url})
    )
    state = ControlledAdapterState(probe=probe)
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
        clock=lambda: now[0],
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                first_location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs", json={"url": first_url}
                    )
                )["links"]["self"]
                await probe.wait_for_metadata_calls(1)
                before_deadline_location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": before_deadline_url},
                    )
                )["links"]["self"]
                expired_location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": expired_url},
                    )
                )["links"]["self"]

                now[0] = initial + timedelta(seconds=20, microseconds=-1)
                probe.release_metadata()
                await probe.wait_for_metadata_calls(2)
                assert probe.metadata_urls == (first_url, before_deadline_url)

                now[0] = initial + timedelta(seconds=20)
                probe.release_metadata()
                expired = await _poll_until_finished(client, expired_location)
                _assert_failed(
                    expired,
                    expired_location,
                    "queue_timeout",
                    started=False,
                )
                _assert_succeeded(
                    await _poll_until_finished(client, first_location),
                    first_location,
                )
                _assert_succeeded(
                    await _poll_until_finished(client, before_deadline_location),
                    before_deadline_location,
                )
            finally:
                probe.release_metadata(2)

    assert expired_url not in state.metadata_calls
    assert expired_url not in state.download_calls
    assert expired_url not in probe.native_source_urls


async def test_api_gives_each_processing_stage_its_full_deadline_after_queueing(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Keep queue delay outside metadata, download, and native stage budgets."""
    stage_timeout = 0.3
    first_url = f"{TIKTOK_URL}?deadline_budget=first"
    queued_url = f"{TIKTOK_URL}?deadline_budget=queued"
    state = ControlledAdapterState(
        block_metadata=True,
        metadata_delay_seconds=0.07,
        download_delay_seconds=0.12,
        native_delay_seconds=0.12,
    )
    settings = _transcription_config(
        tmp_path / "media",
        metadata_timeout_seconds=stage_timeout,
        audio_download_timeout_seconds=stage_timeout,
        transcription_timeout_seconds=stage_timeout,
    )
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
        settings=settings,
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                first_location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs", json={"url": first_url}
                    )
                )["links"]["self"]
                await _wait_for_thread_event(state.metadata_entered)
                queued_location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs", json={"url": queued_url}
                    )
                )["links"]["self"]
                queued_at = time.monotonic()
                await asyncio.sleep(0.06)
                state.metadata_release.set()
                queued_finished = await _poll_until_finished(client, queued_location)
                queued_elapsed = time.monotonic() - queued_at
                first_finished = await _poll_until_finished(client, first_location)
            finally:
                state.metadata_release.set()

    _assert_succeeded(first_finished, first_location)
    _assert_succeeded(queued_finished, queued_location)
    assert queued_elapsed > stage_timeout
    assert len(state.metadata_deadlines) == 2
    assert len(state.download_deadlines) == 2
    assert len(state.native_entry_times) == 2
    assert all(
        deadline - entered_at >= stage_timeout * 0.9
        for deadline, entered_at in zip(
            state.metadata_deadlines, state.metadata_entry_times, strict=True
        )
    )
    assert all(
        deadline - entered_at >= stage_timeout * 0.9
        for deadline, entered_at in zip(
            state.download_deadlines, state.download_entry_times, strict=True
        )
    )


class _UnexpectedProviderError(Exception):
    """Represent a provider error deliberately outside executor translation."""


async def test_api_publishes_unexpected_worker_failures_without_provider_details(
    caplog: pytest.LogCaptureFixture,
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Publish only internal_error when an unknown provider exception escapes."""
    sensitive_sentinel = "UNIQUE_PROVIDER_SECRET_SENTINEL"
    state = ControlledAdapterState(
        native_error=_UnexpectedProviderError(sensitive_sentinel)
    )
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
    )
    caplog.set_level(logging.ERROR)
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL},
            )
            location = _assert_submission(accepted)["links"]["self"]
            failed = await _poll_until_finished(client, location)

    _assert_failed(failed, location, "internal_error", started=True)
    assert sensitive_sentinel not in failed.text
    assert _UnexpectedProviderError.__name__ not in failed.text
    assert all(
        sensitive_sentinel not in record.getMessage()
        and _UnexpectedProviderError.__name__ not in record.getMessage()
        for record in caplog.records
    )


async def test_api_publishes_failure_only_after_request_media_cleanup(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Finish a native failure only after its request directory is removed."""
    state = ControlledAdapterState(native_error=TranscriptionFailedError())
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": TIKTOK_URL},
            )
            location = _assert_submission(accepted)["links"]["self"]
            failed = await _poll_until_finished(client, location)
            _assert_failed(failed, location, "transcription_failed", started=True)
            assert len(state.request_directories) == 1
            assert len(state.audio_paths) == 1
            assert not state.request_directories[0].exists()
            assert not state.audio_paths[0].exists()
