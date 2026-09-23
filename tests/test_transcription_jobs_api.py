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
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from starlette.types import Message
from yt_dlp.utils import DownloadCancelled

from textify.config import AppConfig, Environment
from textify.jobs.config import JobConfig
from textify.jobs.repository import TranscriptionJobStoreUnavailableError
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
    block_download_until_cancellation: bool = False
    caption_segments: tuple[tuple[float, float, str], ...] = ()
    caption_language: str = "en-US"
    probe: ConcurrencyProbe | None = None
    metadata_error: Exception | None = None
    download_error: Exception | None = None
    native_error: Exception | None = None
    metadata_delay_seconds: float = 0.0
    download_delay_seconds: float = 0.0
    native_delay_seconds: float = 0.0
    metadata_cancellation_prepared_directory: Path | None = None
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
    metadata_cancellation_observed: threading.Event = field(
        default_factory=threading.Event
    )
    download_cancellation_entered: threading.Event = field(
        default_factory=threading.Event
    )
    download_cancellation_observed: threading.Event = field(
        default_factory=threading.Event
    )
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
                    if cancellation_event.is_set():
                        self._state.metadata_cancellation_observed.set()
                        break
            if cancellation_event.is_set():
                self._state.metadata_cancellation_observed.set()
                prepared_directory = (
                    self._state.metadata_cancellation_prepared_directory
                )
                if prepared_directory is not None:
                    prepared_directory.mkdir(parents=True, exist_ok=True)
                    prepared_path = prepared_directory / "audio.webm"
                    prepared_path.write_bytes(b"prepared")
                    return inspection.ExtractedMetadata(
                        _metadata_for(provider_url),
                        inspection.PreparedAudio(prepared_path, prepared_directory),
                    )
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
        self._state.download_calls.append(source_url)
        self._state.download_deadlines.append(deadline)
        self._state.download_entry_times.append(time.monotonic())
        self._state.request_directories.append(destination)
        if self._state.download_error is not None:
            raise self._state.download_error
        audio_path = destination / "audio.webm"
        if self._state.block_download_until_cancellation:
            audio_path.write_bytes(b"partial")
            self._state.audio_paths.append(audio_path)
            self._state.download_cancellation_entered.set()
            if not cancellation_event.wait(2.0):
                raise AssertionError("Downloader did not receive cancellation.")
            self._state.download_cancellation_observed.set()
            raise DownloadCancelled("controlled download cancellation")
        assert not cancellation_event.is_set()
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
    job_retention_seconds: int = 86_400,
) -> JobConfig:
    """Create isolated durable-job configuration for durable-job tests."""
    return JobConfig(
        database_path=database_path,
        job_worker_count=job_worker_count,
        max_outstanding_jobs=8,
        job_queue_timeout_seconds=job_queue_timeout_seconds,
        job_retention_seconds=job_retention_seconds,
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
    retention_cleanup_interval: timedelta = timedelta(hours=1),
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
        retention_cleanup_interval=retention_cleanup_interval,
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


def _assert_processing(
    response: Response,
    location: str,
    *,
    expected_status: int = 200,
    cancellation_requested: bool = False,
) -> dict[str, object]:
    """Assert the public active representation after a worker claim."""
    payload = response.json()
    assert response.status_code == expected_status
    assert set(payload) == {
        "id",
        "status",
        "submitted_at",
        "started_at",
        "cancellation_requested",
        "links",
    }
    assert payload["status"] == "processing"
    assert payload["cancellation_requested"] is cancellation_requested
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


def _assert_cancelled(
    response: Response,
    location: str,
    *,
    started: bool,
) -> dict[str, object]:
    """Assert the complete safe terminal representation of a cancelled job."""
    payload = response.json()
    assert response.status_code == 200
    assert set(payload) == {
        "id",
        "status",
        "outcome",
        "submitted_at",
        "started_at",
        "finished_at",
        "links",
    }
    assert payload["status"] == "finished"
    assert payload["outcome"] == "cancelled"
    assert payload["links"] == {"self": location}
    if started:
        started_at = datetime.fromisoformat(payload["started_at"])
        assert started_at.tzinfo is not None
    else:
        assert payload["started_at"] is None
    finished_at = datetime.fromisoformat(payload["finished_at"])
    assert finished_at.tzinfo is not None
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


async def test_api_cancels_queued_jobs_and_rejects_nonempty_bodies(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Cancel queued work without invoking providers or accepting ignored bodies."""
    first_url = f"{TIKTOK_URL}?queued_cancellation=first"
    queued_url = f"{TIKTOK_URL}?queued_cancellation=second"
    state = ControlledAdapterState(block_metadata=True)
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                first_location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": first_url},
                    )
                )["links"]["self"]
                await _wait_for_thread_event(state.metadata_entered)
                queued_location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": queued_url},
                    )
                )["links"]["self"]

                for body in (b"{}", b"null", b" ", b"\x00"):
                    rejected = await client.put(
                        f"{queued_location}/cancellation",
                        content=body,
                    )
                    _assert_safe_error(rejected, 422, "invalid_request")
                    assert (await client.get(queued_location)).json()[
                        "status"
                    ] == "queued"

                cancelled = await client.put(f"{queued_location}/cancellation")
                cancelled_payload = _assert_cancelled(
                    cancelled,
                    queued_location,
                    started=False,
                )
                repeated = await client.put(f"{queued_location}/cancellation")
                repeated_payload = _assert_cancelled(
                    repeated,
                    queued_location,
                    started=False,
                )
                assert (
                    repeated_payload["finished_at"] == cancelled_payload["finished_at"]
                )
                assert (
                    repeated.headers["X-Request-ID"]
                    != cancelled.headers["X-Request-ID"]
                )
                state.metadata_release.set()
                _assert_succeeded(
                    await _poll_until_finished(client, first_location),
                    first_location,
                )
            finally:
                state.metadata_release.set()

    assert state.metadata_calls == [first_url]
    assert state.download_calls == [first_url]


async def test_api_cancels_metadata_after_late_prepared_media_cleanup(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Publish cancellation only after cooperative metadata media is removed."""
    prepared_directory = tmp_path / "late-metadata"
    state = ControlledAdapterState(
        block_metadata=True,
        metadata_cancellation_prepared_directory=prepared_directory,
    )
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                accepted = await client.post(
                    "/api/transcription-jobs",
                    json={"url": TIKTOK_URL},
                )
                location = _assert_submission(accepted)["links"]["self"]
                await _wait_for_thread_event(state.metadata_entered)

                cancellation = await client.put(f"{location}/cancellation")
                _assert_processing(
                    cancellation,
                    location,
                    expected_status=202,
                    cancellation_requested=True,
                )
                with sqlite3.connect(migrated_database_path) as connection:
                    lifecycle = connection.execute(
                        "SELECT status, cancellation_requested FROM transcription_job"
                    ).fetchone()
                assert lifecycle == ("processing", 1)
                await _wait_for_thread_event(state.metadata_cancellation_observed)
                await _wait_until(
                    lambda: not prepared_directory.exists(),
                    "Late metadata media was not removed before cancellation.",
                )
                cancelled = await _poll_until_finished(client, location)
            finally:
                state.metadata_release.set()

    _assert_cancelled(cancelled, location, started=True)
    assert state.download_calls == []
    assert state.native_include_segments == []


async def test_api_cancels_download_without_starting_native_work(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Resolve cooperative download cancellation after partial-media cleanup."""
    state = ControlledAdapterState(block_download_until_cancellation=True)
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
            await _wait_for_thread_event(state.download_cancellation_entered)

            cancellation = await client.put(f"{location}/cancellation")
            _assert_processing(
                cancellation,
                location,
                expected_status=202,
                cancellation_requested=True,
            )
            await _wait_for_thread_event(state.download_cancellation_observed)
            cancelled = await _poll_until_finished(client, location)

    _assert_cancelled(cancelled, location, started=True)
    assert state.native_include_segments == []
    assert all(not directory.exists() for directory in state.request_directories)


async def test_api_retains_cancelled_native_work_until_cleanup_releases_its_permit(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Keep cancelled native work processing until retained inference can clean up."""
    first_url = f"{TIKTOK_URL}?native_cancellation=first"
    second_url = f"{TIKTOK_URL}?native_cancellation=second"
    probe = ConcurrencyProbe(gate_native_calls=True)
    state = ControlledAdapterState(probe=probe)
    settings = _transcription_config(
        tmp_path / "media",
        transcription_concurrency=1,
        transcription_timeout_seconds=0.5,
    )
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
        settings=settings,
        job_config=_job_config(migrated_database_path, job_worker_count=2),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                first_submission = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": first_url},
                    )
                )
                first_location = first_submission["links"]["self"]
                await probe.wait_for_native_calls(1)
                first_audio_path = state.audio_paths_by_source[first_url]

                cancellation = await client.put(f"{first_location}/cancellation")
                _assert_processing(
                    cancellation,
                    first_location,
                    expected_status=202,
                    cancellation_requested=True,
                )
                repeated_processing = await client.put(f"{first_location}/cancellation")
                _assert_processing(
                    repeated_processing,
                    first_location,
                    expected_status=202,
                    cancellation_requested=True,
                )
                assert (
                    repeated_processing.headers["X-Request-ID"]
                    != cancellation.headers["X-Request-ID"]
                )
                await asyncio.sleep(0.6)
                _assert_processing(
                    await client.get(first_location),
                    first_location,
                    cancellation_requested=True,
                )
                assert first_audio_path.is_file()
                assert probe.active_native_calls == 1

                second_submission = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": second_url},
                    )
                )
                second_location = second_submission["links"]["self"]
                await _wait_until(
                    lambda: second_url in state.download_calls,
                    "Second native job did not reach audio acquisition.",
                )
                assert probe.native_source_urls == (first_url,)

                probe.release_native()
                first_cancelled = await _poll_until_finished(client, first_location)
                first_payload = _assert_cancelled(
                    first_cancelled,
                    first_location,
                    started=True,
                )
                assert not first_audio_path.exists()
                with sqlite3.connect(migrated_database_path) as connection:
                    result_count = connection.execute(
                        "SELECT COUNT(*) FROM transcription_job_result "
                        "WHERE job_id = ("
                        "SELECT id FROM transcription_job WHERE public_id = ?"
                        ")",
                        (first_submission["id"],),
                    ).fetchone()[0]
                    segment_count = connection.execute(
                        "SELECT COUNT(*) FROM transcription_job_segment "
                        "WHERE job_id = ("
                        "SELECT id FROM transcription_job WHERE public_id = ?"
                        ")",
                        (first_submission["id"],),
                    ).fetchone()[0]
                assert result_count == 0
                assert segment_count == 0

                repeated_terminal = await client.put(f"{first_location}/cancellation")
                repeated_payload = _assert_cancelled(
                    repeated_terminal,
                    first_location,
                    started=True,
                )
                assert repeated_payload["finished_at"] == first_payload["finished_at"]
                await probe.wait_for_native_calls(2)
                assert list(probe.native_source_urls) == [first_url, second_url]
                probe.release_native()
                _assert_succeeded(
                    await _poll_until_finished(client, second_location),
                    second_location,
                )
            finally:
                probe.release_native(2)


async def test_api_rejects_completed_unknown_and_deleted_cancellation_capabilities(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Preserve completed outcomes and hide malformed, unknown, and deleted jobs."""
    succeeded_state = ControlledAdapterState()
    succeeded_application = _application(
        migrated_database_path,
        tmp_path / "succeeded-media",
        ControlledAdaptersFactory(succeeded_state),
    )
    async with succeeded_application.router.lifespan_context(succeeded_application):
        transport = ASGITransport(app=succeeded_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            succeeded_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": TIKTOK_URL},
                )
            )
            succeeded_location = succeeded_submission["links"]["self"]
            succeeded = await _poll_until_finished(client, succeeded_location)
            succeeded_payload = _assert_succeeded(succeeded, succeeded_location)
            conflict = await client.put(f"{succeeded_location}/cancellation")
            _assert_safe_error(conflict, 409, "job_already_finished")
            assert (await client.get(succeeded_location)).json() == succeeded_payload

    failed_state = ControlledAdapterState(metadata_error=MetadataRetrievalFailedError())
    failed_application = _application(
        migrated_database_path,
        tmp_path / "failed-media",
        ControlledAdaptersFactory(failed_state),
    )
    async with failed_application.router.lifespan_context(failed_application):
        transport = ASGITransport(app=failed_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            failed_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": f"{TIKTOK_URL}?failed_cancellation=true"},
                )
            )
            failed_location = failed_submission["links"]["self"]
            failed = await _poll_until_finished(client, failed_location)
            failed_payload = _assert_failed(
                failed,
                failed_location,
                "metadata_retrieval_failed",
                started=True,
            )
            conflict = await client.put(f"{failed_location}/cancellation")
            _assert_safe_error(conflict, 409, "job_already_finished")
            assert (await client.get(failed_location)).json() == failed_payload

            malformed = await client.put(
                "/api/transcription-jobs/not-a-uuid/cancellation"
            )
            unknown = await client.put(
                "/api/transcription-jobs/00000000-0000-4000-8000-000000000000/"
                "cancellation"
            )
            with sqlite3.connect(migrated_database_path) as connection:
                connection.execute(
                    "DELETE FROM transcription_job WHERE public_id = ?",
                    (failed_submission["id"],),
                )
            deleted = await client.put(f"{failed_location}/cancellation")

    for response in (malformed, unknown, deleted):
        _assert_safe_error(response, 404, "job_not_found")
    assert malformed.json() == unknown.json() == deleted.json()
    assert (
        len(
            {
                malformed.headers["X-Request-ID"],
                unknown.headers["X-Request-ID"],
                deleted.headers["X-Request-ID"],
            }
        )
        == 3
    )
    assert succeeded_state.metadata_calls == [TIKTOK_URL]
    assert failed_state.metadata_calls == [f"{TIKTOK_URL}?failed_cancellation=true"]


async def test_api_recovers_every_durable_state_without_retries(
    capsys: pytest.CaptureFixture[str],
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Recover durable state without resubmitting or retrying interrupted jobs."""
    initial = datetime(2026, 8, 24, tzinfo=UTC)
    first_clock = [initial]
    transcript_sentinel = "transcript-sentinel"
    submitted_url_sentinel = "submitted-url-sentinel"
    provider_detail_sentinel = "provider-detail-sentinel"
    successful_url = f"{YOUTUBE_URL}&{submitted_url_sentinel}=successful"
    failed_url = f"{TIKTOK_URL}?{provider_detail_sentinel}=failed"
    cancelled_url = f"{TIKTOK_URL}?{submitted_url_sentinel}=cancelled"
    active_url = f"{TIKTOK_URL}?{submitted_url_sentinel}=active"
    expired_url = f"{TIKTOK_URL}?{submitted_url_sentinel}=expired"
    first_preserved_url = f"{TIKTOK_URL}?{submitted_url_sentinel}=first"
    second_preserved_url = f"{TIKTOK_URL}?{submitted_url_sentinel}=second"
    interrupted_url = f"{TIKTOK_URL}?{submitted_url_sentinel}=interrupted"
    cancelling_url = f"{TIKTOK_URL}?{submitted_url_sentinel}=cancelling"
    first_state = ControlledAdapterState(
        caption_segments=((0.0, 1.0, transcript_sentinel),)
    )
    first_application = _application(
        migrated_database_path,
        tmp_path / "first-media",
        ControlledAdaptersFactory(first_state),
        clock=lambda: first_clock[0],
    )

    async with first_application.router.lifespan_context(first_application):
        transport = ASGITransport(app=first_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            successful_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": successful_url},
                )
            )
            successful_location = successful_submission["links"]["self"]
            successful_payload = _assert_succeeded(
                await _poll_until_finished(client, successful_location),
                successful_location,
            )

            first_state.caption_segments = ()
            first_state.metadata_error = _UnexpectedProviderError(
                provider_detail_sentinel
            )
            failed_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": failed_url},
                )
            )
            failed_location = failed_submission["links"]["self"]
            failed_payload = _assert_failed(
                await _poll_until_finished(client, failed_location),
                failed_location,
                "internal_error",
                started=True,
            )
            first_state.metadata_error = None

            first_state.block_metadata = True
            first_state.metadata_entered.clear()
            first_state.metadata_release.clear()
            cancelled_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": cancelled_url},
                )
            )
            cancelled_location = cancelled_submission["links"]["self"]
            await _wait_for_thread_event(first_state.metadata_entered)
            _assert_processing(
                await client.put(f"{cancelled_location}/cancellation"),
                cancelled_location,
                expected_status=202,
                cancellation_requested=True,
            )
            cancelled_payload = _assert_cancelled(
                await _poll_until_finished(client, cancelled_location),
                cancelled_location,
                started=True,
            )

            first_state.metadata_entered.clear()
            first_state.metadata_release.clear()
            active_location = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": active_url},
                )
            )["links"]["self"]
            await _wait_for_thread_event(first_state.metadata_entered)
            expired_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": expired_url},
                )
            )
            first_preserved_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": first_preserved_url},
                )
            )
            second_preserved_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": second_preserved_url},
                )
            )
            interrupted_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": interrupted_url},
                )
            )
            cancelling_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": cancelling_url},
                )
            )

            shutdown_task = asyncio.create_task(
                first_application.state.transcription_job_coordinator.shutdown()
            )
            await asyncio.sleep(0)
            first_state.metadata_release.set()
            await shutdown_task
            _assert_succeeded(
                await client.get(active_location),
                active_location,
            )

    recovery_clock = initial + timedelta(seconds=5)
    first_deadline = recovery_clock + timedelta(seconds=10)
    second_deadline = recovery_clock + timedelta(seconds=20)
    recovery_database_time = recovery_clock.replace(tzinfo=None).isoformat(sep=" ")
    first_deadline_database_time = first_deadline.replace(tzinfo=None).isoformat(
        sep=" "
    )
    second_deadline_database_time = second_deadline.replace(tzinfo=None).isoformat(
        sep=" "
    )
    queued_submissions = (
        expired_submission,
        first_preserved_submission,
        second_preserved_submission,
        interrupted_submission,
        cancelling_submission,
    )
    with sqlite3.connect(migrated_database_path) as connection:
        internal_ids = {
            public_id: internal_id
            for internal_id, public_id in connection.execute(
                "SELECT id, public_id FROM transcription_job"
            )
        }
        connection.execute(
            "UPDATE transcription_job SET queue_deadline_at = ? WHERE public_id = ?",
            (recovery_database_time, expired_submission["id"]),
        )
        connection.execute(
            "UPDATE transcription_job SET queue_deadline_at = ? WHERE public_id = ?",
            (first_deadline_database_time, first_preserved_submission["id"]),
        )
        connection.execute(
            "UPDATE transcription_job SET queue_deadline_at = ? WHERE public_id = ?",
            (second_deadline_database_time, second_preserved_submission["id"]),
        )
        connection.execute(
            "UPDATE transcription_job "
            "SET status = 'processing', started_at = ? "
            "WHERE public_id = ?",
            (recovery_database_time, interrupted_submission["id"]),
        )
        connection.execute(
            "UPDATE transcription_job "
            "SET status = 'processing', started_at = ?, cancellation_requested = 1 "
            "WHERE public_id = ?",
            (recovery_database_time, cancelling_submission["id"]),
        )

    capsys.readouterr()
    second_clock = [recovery_clock]
    second_probe = ConcurrencyProbe(
        gated_metadata_urls=frozenset({first_preserved_url})
    )
    second_state = ControlledAdapterState(probe=second_probe)
    second_application = _application(
        migrated_database_path,
        tmp_path / "second-media",
        ControlledAdaptersFactory(second_state),
        job_config=_job_config(migrated_database_path, job_worker_count=1),
        clock=lambda: second_clock[0],
    )
    try:
        async with second_application.router.lifespan_context(second_application):
            transport = ASGITransport(app=second_application)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                await second_probe.wait_for_metadata_calls(1)
                first_preserved_location = first_preserved_submission["links"]["self"]
                first_processing = _assert_processing(
                    await client.get(first_preserved_location),
                    first_preserved_location,
                )
                assert first_processing["id"] == first_preserved_submission["id"]
                assert (
                    first_processing["submitted_at"]
                    == first_preserved_submission["submitted_at"]
                )
                second_preserved_location = second_preserved_submission["links"]["self"]
                assert (
                    await client.get(second_preserved_location)
                ).json() == second_preserved_submission

                expired_location = expired_submission["links"]["self"]
                expired_payload = _assert_failed(
                    await client.get(expired_location),
                    expired_location,
                    "queue_timeout",
                    started=False,
                )
                assert (
                    datetime.fromisoformat(cast(str, expired_payload["finished_at"]))
                    == recovery_clock
                )
                interrupted_location = interrupted_submission["links"]["self"]
                interrupted_payload = _assert_failed(
                    await client.get(interrupted_location),
                    interrupted_location,
                    "worker_interrupted",
                    started=True,
                )
                assert (
                    datetime.fromisoformat(cast(str, interrupted_payload["started_at"]))
                    == recovery_clock
                )
                assert (
                    datetime.fromisoformat(
                        cast(str, interrupted_payload["finished_at"])
                    )
                    == recovery_clock
                )
                cancelling_location = cancelling_submission["links"]["self"]
                cancelling_payload = _assert_cancelled(
                    await client.get(cancelling_location),
                    cancelling_location,
                    started=True,
                )
                assert (
                    datetime.fromisoformat(cast(str, cancelling_payload["started_at"]))
                    == recovery_clock
                )
                assert (
                    datetime.fromisoformat(cast(str, cancelling_payload["finished_at"]))
                    == recovery_clock
                )

                assert (
                    await client.get(successful_location)
                ).json() == successful_payload
                assert (await client.get(failed_location)).json() == failed_payload
                assert (
                    await client.get(cancelled_location)
                ).json() == cancelled_payload

                second_clock[0] = second_deadline
                second_probe.release_metadata()
                _assert_succeeded(
                    await _poll_until_finished(client, first_preserved_location),
                    first_preserved_location,
                )
                _assert_failed(
                    await _poll_until_finished(client, second_preserved_location),
                    second_preserved_location,
                    "queue_timeout",
                    started=False,
                )
    finally:
        second_probe.release_metadata()

    recovery_output = capsys.readouterr().err
    assert second_state.metadata_calls == [first_preserved_url]
    assert second_state.caption_calls == []
    assert second_state.download_calls == [first_preserved_url]
    assert second_state.native_include_segments == [True]
    assert second_probe.native_source_urls == (first_preserved_url,)
    assert recovery_output.count("internal_job_id=") == 3
    assert (
        f"internal_job_id={internal_ids[expired_submission['id']]}" in recovery_output
    )
    assert (
        f"internal_job_id={internal_ids[interrupted_submission['id']]}"
        in recovery_output
    )
    assert (
        f"internal_job_id={internal_ids[cancelling_submission['id']]}"
        in recovery_output
    )
    assert "code=queue_timeout" in recovery_output
    assert "code=worker_interrupted" in recovery_output
    for submission in queued_submissions + (
        successful_submission,
        failed_submission,
        cancelled_submission,
    ):
        assert submission["id"] not in recovery_output
        assert submission["links"]["self"] not in recovery_output
        assert submission["links"]["cancel"] not in recovery_output
    assert successful_url not in recovery_output
    assert transcript_sentinel not in recovery_output
    assert provider_detail_sentinel not in recovery_output


async def test_api_rolls_back_failed_startup_recovery(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Abort startup when any durable recovery transition cannot commit."""
    submitted_at = datetime(2026, 8, 24, tzinfo=UTC)
    recovery_clock = submitted_at + timedelta(seconds=5)
    submitted_database_time = submitted_at.replace(tzinfo=None).isoformat(sep=" ")
    recovery_database_time = recovery_clock.replace(tzinfo=None).isoformat(sep=" ")
    future_database_time = (
        (submitted_at + timedelta(seconds=20)).replace(tzinfo=None).isoformat(sep=" ")
    )
    expired_id = str(uuid4())
    interrupted_id = str(uuid4())
    cancelling_id = str(uuid4())
    rows = (
        (
            expired_id,
            f"{TIKTOK_URL}?recovery_rollback=expired",
            "queued",
            submitted_database_time,
            recovery_database_time,
            None,
            0,
        ),
        (
            interrupted_id,
            f"{TIKTOK_URL}?recovery_rollback=interrupted",
            "processing",
            submitted_database_time,
            future_database_time,
            recovery_database_time,
            0,
        ),
        (
            cancelling_id,
            f"{TIKTOK_URL}?recovery_rollback=cancelling",
            "processing",
            submitted_database_time,
            future_database_time,
            recovery_database_time,
            1,
        ),
    )
    with sqlite3.connect(migrated_database_path) as connection:
        connection.executemany(
            "INSERT INTO transcription_job ("
            "public_id, submitted_url, status, submitted_at, queue_deadline_at, "
            "started_at, cancellation_requested"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        before_recovery = connection.execute(
            "SELECT * FROM transcription_job ORDER BY id"
        ).fetchall()
        connection.execute(
            "CREATE TRIGGER abort_processing_recovery "
            "BEFORE UPDATE OF status ON transcription_job "
            "WHEN OLD.status = 'processing' "
            "BEGIN SELECT RAISE(ABORT, 'recovery update aborted'); END"
        )

    failed_state = ControlledAdapterState()
    failed_factory = ControlledAdaptersFactory(failed_state)
    failed_application = _application(
        migrated_database_path,
        tmp_path / "failed-recovery-media",
        failed_factory,
        clock=lambda: recovery_clock,
    )
    with pytest.raises(TranscriptionJobStoreUnavailableError):
        async with failed_application.router.lifespan_context(failed_application):
            pass
    assert failed_application.state.ready is False
    assert failed_factory.calls == 0
    with sqlite3.connect(migrated_database_path) as connection:
        after_failed_recovery = connection.execute(
            "SELECT * FROM transcription_job ORDER BY id"
        ).fetchall()
        connection.execute("DROP TRIGGER abort_processing_recovery")
    assert after_failed_recovery == before_recovery

    recovered_state = ControlledAdapterState()
    recovered_factory = ControlledAdaptersFactory(recovered_state)
    recovered_application = _application(
        migrated_database_path,
        tmp_path / "recovered-media",
        recovered_factory,
        clock=lambda: recovery_clock,
    )
    async with recovered_application.router.lifespan_context(recovered_application):
        transport = ASGITransport(app=recovered_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            _assert_failed(
                await client.get(f"/api/transcription-jobs/{expired_id}"),
                f"/api/transcription-jobs/{expired_id}",
                "queue_timeout",
                started=False,
            )
            _assert_failed(
                await client.get(f"/api/transcription-jobs/{interrupted_id}"),
                f"/api/transcription-jobs/{interrupted_id}",
                "worker_interrupted",
                started=True,
            )
            _assert_cancelled(
                await client.get(f"/api/transcription-jobs/{cancelling_id}"),
                f"/api/transcription-jobs/{cancelling_id}",
                started=True,
            )
    assert recovered_factory.calls == 1
    assert recovered_state.metadata_calls == []
    assert recovered_state.download_calls == []
    assert recovered_state.native_include_segments == []


async def test_api_shutdown_closes_admission_and_preserves_queued_deadline(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Finish active work without claiming queued work or extending its deadline."""
    initial = datetime(2026, 8, 24, tzinfo=UTC)
    first_clock = [initial]
    active_url = f"{TIKTOK_URL}?shutdown_deadline=active"
    queued_url = f"{TIKTOK_URL}?shutdown_deadline=queued"
    rejected_url = f"{TIKTOK_URL}?shutdown_deadline=rejected"
    first_state = ControlledAdapterState(block_metadata=True)
    first_application = _application(
        migrated_database_path,
        tmp_path / "first-media",
        ControlledAdaptersFactory(first_state),
        clock=lambda: first_clock[0],
    )

    async with first_application.router.lifespan_context(first_application):
        transport = ASGITransport(app=first_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                active_location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": active_url},
                    )
                )["links"]["self"]
                await _wait_for_thread_event(first_state.metadata_entered)
                queued_submission = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": queued_url},
                    )
                )
                shutdown_task = asyncio.create_task(
                    first_application.state.transcription_job_coordinator.shutdown()
                )
                await asyncio.sleep(0)
                rejected = await client.post(
                    "/api/transcription-jobs",
                    json={"url": rejected_url},
                )
                _assert_safe_error(
                    rejected,
                    503,
                    "transcription_capacity_exceeded",
                )
                first_state.metadata_release.set()
                await shutdown_task
                _assert_succeeded(
                    await client.get(active_location),
                    active_location,
                )
                assert (
                    await client.get(queued_submission["links"]["self"])
                ).json() == queued_submission
            finally:
                first_state.metadata_release.set()

    second_state = ControlledAdapterState()
    second_application = _application(
        migrated_database_path,
        tmp_path / "second-media",
        ControlledAdaptersFactory(second_state),
        clock=lambda: initial + timedelta(seconds=20),
    )
    async with second_application.router.lifespan_context(second_application):
        transport = ASGITransport(app=second_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            queued_location = queued_submission["links"]["self"]
            _assert_failed(
                await client.get(queued_location),
                queued_location,
                "queue_timeout",
                started=False,
            )

    assert first_state.metadata_calls == [active_url]
    assert first_state.download_calls == [active_url]
    assert first_state.native_include_segments == [True]
    assert rejected_url not in first_state.metadata_calls
    assert second_state.metadata_calls == []
    assert second_state.download_calls == []
    assert second_state.native_include_segments == []


async def test_api_shutdown_wakes_idle_consumers(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Wake every idle consumer before the lifespan invokes idempotent shutdown."""
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(ControlledAdapterState()),
        job_config=_job_config(migrated_database_path, job_worker_count=4),
    )
    async with application.router.lifespan_context(application):
        await asyncio.sleep(0)
        async with asyncio.timeout(1):
            await application.state.transcription_job_coordinator.shutdown()


async def test_api_shutdown_waits_for_retained_native_finalizers(
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Wait for retained native ownership before returning from shutdown."""
    native_url = f"{TIKTOK_URL}?shutdown_native=retained"
    probe = ConcurrencyProbe(gate_native_calls=True)
    state = ControlledAdapterState(probe=probe)
    settings = _transcription_config(
        tmp_path / "media",
        transcription_concurrency=1,
        transcription_timeout_seconds=0.05,
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
                location = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": native_url},
                    )
                )["links"]["self"]
                await probe.wait_for_native_calls(1)
                audio_path = state.audio_paths_by_source[native_url]
                _assert_failed(
                    await _poll_until_finished(client, location),
                    location,
                    "transcription_timeout",
                    started=True,
                )

                shutdown_task = asyncio.create_task(
                    application.state.transcription_job_coordinator.shutdown()
                )
                await asyncio.sleep(0)
                assert shutdown_task.done() is False
                assert probe.active_native_calls == 1
                assert audio_path.is_file()

                probe.release_native()
                await shutdown_task
                assert probe.active_native_calls == 0
                assert audio_path.exists() is False
            finally:
                probe.release_native()


def _internal_job_id_for_public_id(
    connection: sqlite3.Connection,
    public_id: str,
) -> int:
    """Return one persisted internal identifier for a public capability."""
    row = connection.execute(
        "SELECT id FROM transcription_job WHERE public_id = ?",
        (public_id,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _job_storage_counts(
    connection: sqlite3.Connection,
    internal_job_id: int,
) -> tuple[int, int, int, int]:
    """Count one job and its cascade-owned rows by durable internal identifier."""
    parent_count = connection.execute(
        "SELECT COUNT(*) FROM transcription_job WHERE id = ?",
        (internal_job_id,),
    ).fetchone()[0]
    exclusion_count = connection.execute(
        "SELECT COUNT(*) FROM transcription_job_exclusion WHERE job_id = ?",
        (internal_job_id,),
    ).fetchone()[0]
    result_count = connection.execute(
        "SELECT COUNT(*) FROM transcription_job_result WHERE job_id = ?",
        (internal_job_id,),
    ).fetchone()[0]
    segment_count = connection.execute(
        "SELECT COUNT(*) FROM transcription_job_segment WHERE job_id = ?",
        (internal_job_id,),
    ).fetchone()[0]
    return (
        int(parent_count),
        int(exclusion_count),
        int(result_count),
        int(segment_count),
    )


@pytest.mark.parametrize("outcome", ("succeeded", "failed", "cancelled"))
async def test_api_hides_terminal_jobs_at_the_retention_cutoff(
    outcome: str,
    capsys: pytest.CaptureFixture[str],
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Retain each terminal outcome through, but not at, its finish-based cutoff."""
    initial = datetime(2026, 8, 24, tzinfo=UTC)
    finished_at = initial + timedelta(minutes=10)
    now = [initial]
    state = ControlledAdapterState(block_metadata=True)
    if outcome == "failed":
        state.metadata_error = MetadataRetrievalFailedError()
    submitted_url = f"{TIKTOK_URL}?retention-cutoff={outcome}"
    application = _application(
        migrated_database_path,
        tmp_path / "media",
        ControlledAdaptersFactory(state),
        clock=lambda: now[0],
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": submitted_url},
                )
            )
            location = submission["links"]["self"]
            await _wait_for_thread_event(state.metadata_entered)
            now[0] = finished_at

            if outcome == "cancelled":
                _assert_processing(
                    await client.put(f"{location}/cancellation"),
                    location,
                    expected_status=202,
                    cancellation_requested=True,
                )
            else:
                state.metadata_release.set()

            terminal_response = await _poll_until_finished(client, location)
            if outcome == "succeeded":
                terminal_payload = _assert_succeeded(terminal_response, location)
            elif outcome == "failed":
                terminal_payload = _assert_failed(
                    terminal_response,
                    location,
                    MetadataRetrievalFailedError.code,
                    started=True,
                )
            else:
                terminal_payload = _assert_cancelled(
                    terminal_response,
                    location,
                    started=True,
                )
            assert (
                datetime.fromisoformat(cast(str, terminal_payload["finished_at"]))
                == finished_at
            )

            now[0] = finished_at + timedelta(seconds=86_400, microseconds=-1)
            before_cutoff = await client.get(location)
            assert before_cutoff.headers["Cache-Control"] == "no-store"
            assert before_cutoff.json() == terminal_payload

            now[0] = finished_at + timedelta(seconds=86_400)
            unknown_id = uuid4()
            expired_status = await client.get(location)
            unknown_status = await client.get(f"/api/transcription-jobs/{unknown_id}")
            expired_cancellation = await client.put(f"{location}/cancellation")
            unknown_cancellation = await client.put(
                f"/api/transcription-jobs/{unknown_id}/cancellation"
            )

            for response in (
                expired_status,
                unknown_status,
                expired_cancellation,
                unknown_cancellation,
            ):
                _assert_safe_error(response, 404, "job_not_found")
            assert expired_status.json() == unknown_status.json()
            assert expired_cancellation.json() == unknown_cancellation.json()
            with sqlite3.connect(migrated_database_path) as connection:
                assert (
                    connection.execute(
                        "SELECT COUNT(*) FROM transcription_job WHERE public_id = ?",
                        (submission["id"],),
                    ).fetchone()[0]
                    == 1
                )

    output = capsys.readouterr().err
    for sentinel in (
        submission["id"],
        location,
        f"{location}/cancellation",
        submitted_url,
    ):
        assert sentinel not in output


async def test_api_deletes_expired_terminal_job_cascades_during_startup(
    capsys: pytest.CaptureFixture[str],
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Delete expired successful jobs and every owned row before restart traffic."""
    initial = datetime(2026, 8, 24, tzinfo=UTC)
    finished_at = initial + timedelta(minutes=10)
    now = [initial]
    transcript_sentinel = "terminal-retention-transcript-sentinel"
    submitted_url = f"{YOUTUBE_URL}&terminal-retention-url-sentinel=restart"
    media_root = tmp_path / "terminal-retention-local-media-sentinel"
    first_state = ControlledAdapterState(
        block_metadata=True,
        caption_segments=((0.0, 1.0, transcript_sentinel),),
    )
    first_application = _application(
        migrated_database_path,
        media_root,
        ControlledAdaptersFactory(first_state),
        clock=lambda: now[0],
    )

    async with first_application.router.lifespan_context(first_application):
        transport = ASGITransport(app=first_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": submitted_url, "exclude": ["source.description"]},
                )
            )
            location = submission["links"]["self"]
            await _wait_for_thread_event(first_state.metadata_entered)
            now[0] = finished_at
            first_state.metadata_release.set()
            terminal_payload = _assert_succeeded(
                await _poll_until_finished(client, location),
                location,
            )
            assert (
                datetime.fromisoformat(cast(str, terminal_payload["finished_at"]))
                == finished_at
            )

    with sqlite3.connect(migrated_database_path) as connection:
        internal_job_id = _internal_job_id_for_public_id(connection, submission["id"])
        assert _job_storage_counts(connection, internal_job_id) == (1, 1, 1, 1)

    capsys.readouterr()
    now[0] = finished_at + timedelta(seconds=86_400)
    reopened_state = ControlledAdapterState()
    reopened_factory = ControlledAdaptersFactory(reopened_state)
    reopened_application = _application(
        migrated_database_path,
        tmp_path / "reopened-media",
        reopened_factory,
        clock=lambda: now[0],
    )
    async with reopened_application.router.lifespan_context(reopened_application):
        assert reopened_application.state.ready is True
        with sqlite3.connect(migrated_database_path) as connection:
            assert _job_storage_counts(connection, internal_job_id) == (0, 0, 0, 0)
        transport = ASGITransport(app=reopened_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            _assert_safe_error(await client.get(location), 404, "job_not_found")

    assert reopened_factory.calls == 1
    assert reopened_state.metadata_calls == []
    assert reopened_state.caption_calls == []
    assert reopened_state.download_calls == []
    assert reopened_state.native_include_segments == []
    output = capsys.readouterr().err
    for sentinel in (
        submission["id"],
        location,
        f"{location}/cancellation",
        submitted_url,
        str(media_root),
        "source.description",
        transcript_sentinel,
    ):
        assert sentinel not in output


async def test_api_rolls_back_failed_startup_terminal_retention_cleanup(
    capsys: pytest.CaptureFixture[str],
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Leave every retained row intact when startup terminal cleanup cannot commit."""
    initial = datetime(2026, 8, 24, tzinfo=UTC)
    finished_at = initial + timedelta(minutes=10)
    now = [initial]
    first_state = ControlledAdapterState(block_metadata=True)
    first_application = _application(
        migrated_database_path,
        tmp_path / "first-media",
        ControlledAdaptersFactory(first_state),
        clock=lambda: now[0],
    )

    async with first_application.router.lifespan_context(first_application):
        transport = ASGITransport(app=first_application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={
                        "url": f"{TIKTOK_URL}?retention-startup-delete=blocked",
                        "exclude": ["source.description"],
                    },
                )
            )
            location = submission["links"]["self"]
            await _wait_for_thread_event(first_state.metadata_entered)
            now[0] = finished_at
            first_state.metadata_release.set()
            _assert_succeeded(
                await _poll_until_finished(client, location),
                location,
            )

    with sqlite3.connect(migrated_database_path) as connection:
        internal_job_id = _internal_job_id_for_public_id(connection, submission["id"])
        before_cleanup = _job_storage_counts(connection, internal_job_id)
        connection.execute(
            """
            CREATE TRIGGER abort_terminal_retention_cleanup
            BEFORE DELETE ON transcription_job
            WHEN OLD.status = 'finished'
            BEGIN SELECT RAISE(ABORT, 'terminal retention cleanup blocked'); END
            """
        )

    capsys.readouterr()
    now[0] = finished_at + timedelta(seconds=86_400)
    failed_factory = ControlledAdaptersFactory(ControlledAdapterState())
    failed_application = _application(
        migrated_database_path,
        tmp_path / "failed-cleanup-media",
        failed_factory,
        clock=lambda: now[0],
    )
    with pytest.raises(TranscriptionJobStoreUnavailableError):
        async with failed_application.router.lifespan_context(failed_application):
            pass

    assert failed_application.state.ready is False
    assert failed_factory.calls == 0
    with sqlite3.connect(migrated_database_path) as connection:
        assert _job_storage_counts(connection, internal_job_id) == before_cleanup
        connection.execute("DROP TRIGGER abort_terminal_retention_cleanup")

    output = capsys.readouterr().err
    assert "transcription job retention cleanup failed" in output
    assert "code=job_store_unavailable" in output
    assert submission["id"] not in output
    assert location not in output


async def test_api_periodically_deletes_expired_terminal_jobs_and_checkpoints_wal(
    capsys: pytest.CaptureFixture[str],
    migrated_database_path: Path,
    tmp_path: Path,
) -> None:
    """Delete terminal cascades without hiding queued or processing jobs."""
    initial = datetime(2026, 8, 24, tzinfo=UTC)
    now = [initial]
    transcript_sentinel = "terminal-retention-periodic-transcript-sentinel"
    terminal_url = f"{YOUTUBE_URL}&terminal-retention-url-sentinel=periodic"
    processing_url = f"{TIKTOK_URL}?terminal-retention-processing-sentinel"
    queued_url = f"{TIKTOK_URL}?terminal-retention-queued-sentinel"
    media_root = tmp_path / "terminal-retention-local-media-sentinel"
    state = ControlledAdapterState(
        caption_segments=((0.0, 1.0, transcript_sentinel),),
    )
    application = _application(
        migrated_database_path,
        media_root,
        ControlledAdaptersFactory(state),
        clock=lambda: now[0],
        retention_cleanup_interval=timedelta(milliseconds=250),
    )

    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            terminal_submission = _assert_submission(
                await client.post(
                    "/api/transcription-jobs",
                    json={"url": terminal_url, "exclude": ["source.description"]},
                )
            )
            terminal_location = terminal_submission["links"]["self"]
            terminal_payload = _assert_succeeded(
                await _poll_until_finished(client, terminal_location),
                terminal_location,
            )
            terminal_finished_at = datetime.fromisoformat(
                cast(str, terminal_payload["finished_at"])
            )
            with sqlite3.connect(migrated_database_path) as connection:
                terminal_internal_id = _internal_job_id_for_public_id(
                    connection,
                    terminal_submission["id"],
                )
                assert _job_storage_counts(connection, terminal_internal_id) == (
                    1,
                    1,
                    1,
                    1,
                )

            state.block_metadata = True
            state.metadata_entered.clear()
            state.metadata_release.clear()
            try:
                processing_submission = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": processing_url},
                    )
                )
                processing_location = processing_submission["links"]["self"]
                await _wait_for_thread_event(state.metadata_entered)
                queued_submission = _assert_submission(
                    await client.post(
                        "/api/transcription-jobs",
                        json={"url": queued_url},
                    )
                )
                queued_location = queued_submission["links"]["self"]

                retention_lock = sqlite3.connect(migrated_database_path)
                try:
                    retention_lock.execute("BEGIN IMMEDIATE")
                    now[0] = terminal_finished_at + timedelta(seconds=86_400)
                    _assert_safe_error(
                        await client.get(terminal_location),
                        404,
                        "job_not_found",
                    )
                    with sqlite3.connect(migrated_database_path) as connection:
                        assert _job_storage_counts(
                            connection,
                            terminal_internal_id,
                        ) == (1, 1, 1, 1)
                finally:
                    retention_lock.rollback()
                    retention_lock.close()

                await _wait_until(
                    lambda: _terminal_job_rows_are_deleted(
                        migrated_database_path,
                        terminal_internal_id,
                    ),
                    "Periodic terminal retention cleanup did not delete the job.",
                )

                _assert_processing(
                    await client.get(processing_location),
                    processing_location,
                )
                assert (await client.get(queued_location)).json() == queued_submission
                with sqlite3.connect(migrated_database_path) as connection:
                    assert _job_storage_counts(connection, terminal_internal_id) == (
                        0,
                        0,
                        0,
                        0,
                    )
                await _wait_until(
                    lambda: _wal_is_checkpointed(migrated_database_path),
                    "Periodic terminal retention cleanup did not checkpoint the WAL.",
                )
            finally:
                state.metadata_release.set()

    output = capsys.readouterr().err
    for sentinel in (
        terminal_submission["id"],
        terminal_location,
        f"{terminal_location}/cancellation",
        terminal_url,
        str(media_root),
        "source.description",
        transcript_sentinel,
    ):
        assert sentinel not in output


def _terminal_job_rows_are_deleted(
    database_path: Path,
    internal_job_id: int,
) -> bool:
    """Return whether a terminal job and all cascade-owned rows are absent."""
    with sqlite3.connect(database_path) as connection:
        return _job_storage_counts(connection, internal_job_id) == (0, 0, 0, 0)


def _wal_is_checkpointed(database_path: Path) -> bool:
    """Return whether every current WAL page has already been checkpointed."""
    with sqlite3.connect(database_path) as connection:
        checkpoint = connection.execute("PRAGMA wal_checkpoint(NOOP)").fetchone()
    return (
        checkpoint is not None and checkpoint[0] == 0 and checkpoint[1] == checkpoint[2]
    )
