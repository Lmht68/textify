"""Tests for provider adapters and bounded yt-dlp retries."""

import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import pytest
from pytest import MonkeyPatch
from yt_dlp.utils import DownloadCancelled

from textify.transcription import acquisition
from textify.transcription.acquisition import (
    FasterWhisperTranscriber,
    YouTubeCaptionProvider,
    YtDlpAudioDownloader,
    acquire_transcript,
)
from textify.transcription.config import TranscriptionConfig
from textify.transcription.types import RawSegment, TranscriptMethod
from textify.transcription.util import (
    MediaByteLimitExceeded,
    build_yt_dlp_progress_hook,
    extract_info_with_retries,
)


@dataclass(frozen=True)
class FakeWhisperSegment:
    """Represent one external-shaped Faster-Whisper segment."""

    start: object
    end: object
    text: object


@dataclass(frozen=True)
class FakeWhisperInfo:
    """Represent external Faster-Whisper language metadata."""

    language: object


class FakeWhisperModel:
    """Return deterministic external-shaped Faster-Whisper output."""

    def __init__(
        self,
        segments: tuple[FakeWhisperSegment, ...],
        language: object,
    ) -> None:
        """Initialize deterministic raw model output.

        Args:
            segments: Lazy-model segment values to return.
            language: Language metadata to return.
        """
        self._segments = segments
        self._language = language

    def transcribe(
        self,
        audio: str,
        *,
        beam_size: int,
        vad_filter: bool,
        temperature: float,
        condition_on_previous_text: bool,
        initial_prompt: str,
    ) -> tuple[Iterable[FakeWhisperSegment], FakeWhisperInfo]:
        """Return the configured external model result.

        Args:
            audio: Local audio path.
            beam_size: Requested beam size.
            vad_filter: Requested VAD setting.
            temperature: Requested decoding temperature.
            condition_on_previous_text: Requested context setting.
            initial_prompt: Requested transcription prompt.

        Returns:
            Deterministic raw segment iterable and language metadata.
        """
        del (
            audio,
            beam_size,
            vad_filter,
            temperature,
            condition_on_previous_text,
            initial_prompt,
        )
        return iter(self._segments), FakeWhisperInfo(self._language)


class FakeCaptionTrack:
    """Supply one observable original caption track."""

    def __init__(
        self,
        name: str,
        language_code: str,
        is_generated: bool,
        segments: tuple[RawSegment, ...],
        failure: Exception | None = None,
    ) -> None:
        """Initialize one deterministic caption track.

        Args:
            name: Test-visible candidate identity.
            language_code: Provider-reported original language tag.
            is_generated: Whether the provider generated the track.
            segments: Timed segments to return when fetched.
            failure: Optional provider failure to raise when fetched.
        """
        self.name = name
        self._language_code = language_code
        self._is_generated = is_generated
        self._segments = segments
        self._failure = failure
        self.fetch_calls = 0

    @property
    def language_code(self) -> str:
        """Return the configured provider language tag."""
        return self._language_code

    @property
    def is_generated(self) -> bool:
        """Return the configured generated-track flag."""
        return self._is_generated

    def fetch_segments(self) -> Sequence[RawSegment]:
        """Record and return configured caption segments.

        Returns:
            Timed raw caption segments.

        Raises:
            Exception: The configured provider failure.
        """
        self.fetch_calls += 1
        if self._failure is not None:
            raise self._failure
        return self._segments


class FakeCaptionProvider:
    """List configured caption tracks while recording requests."""

    def __init__(
        self,
        tracks: tuple[FakeCaptionTrack, ...] = (),
        failure: Exception | None = None,
    ) -> None:
        """Initialize deterministic caption listing behavior.

        Args:
            tracks: Caption tracks returned in provider order.
            failure: Optional provider failure to raise during listing.
        """
        self._tracks = tracks
        self._failure = failure
        self.list_calls: list[str] = []

    def list_tracks(self, video_id: str) -> Sequence[FakeCaptionTrack]:
        """Record a video lookup and return configured tracks.

        Args:
            video_id: Stable external video identity.

        Returns:
            Configured caption tracks.

        Raises:
            Exception: The configured provider failure.
        """
        self.list_calls.append(video_id)
        if self._failure is not None:
            raise self._failure
        return self._tracks


@dataclass(frozen=True)
class FakeCaptionSnippet:
    """Represent one external caption snippet."""

    start: object
    duration: object
    text: object


class FakeLibraryCaptionTrack:
    """Expose a library-shaped caption track and observe forbidden translation."""

    def __init__(
        self,
        language_code: str,
        is_generated: bool,
        snippets: tuple[FakeCaptionSnippet, ...],
    ) -> None:
        """Initialize library-shaped timed captions.

        Args:
            language_code: Provider-reported track language tag.
            is_generated: Whether the provider generated the track.
            snippets: Provider caption snippets.
        """
        self.language_code = language_code
        self.is_generated = is_generated
        self._snippets = snippets
        self.fetch_arguments: list[bool] = []
        self.translate_calls = 0

    def fetch(
        self, preserve_formatting: bool = False
    ) -> tuple[FakeCaptionSnippet, ...]:
        """Record formatting and return configured snippets.

        Args:
            preserve_formatting: Whether source formatting should be retained.

        Returns:
            Configured timed snippets.
        """
        self.fetch_arguments.append(preserve_formatting)
        return self._snippets

    def translate(self, language_code: str) -> "FakeLibraryCaptionTrack":
        """Record a forbidden translation request.

        Args:
            language_code: Requested translated language.

        Returns:
            This track.
        """
        del language_code
        self.translate_calls += 1
        return self


class FakeYouTubeTranscriptApi:
    """Supply library-shaped tracks while recording each list client."""

    tracks: ClassVar[tuple[FakeLibraryCaptionTrack, ...]] = ()
    list_calls: ClassVar[list[str]] = []
    client_count: ClassVar[int] = 0

    def __init__(self) -> None:
        """Record one synchronous caption-library client."""
        type(self).client_count += 1

    def list(self, video_id: str) -> tuple[FakeLibraryCaptionTrack, ...]:
        """Record and return configured library tracks.

        Args:
            video_id: Stable external video identity.

        Returns:
            Configured library tracks.
        """
        type(self).list_calls.append(video_id)
        return type(self).tracks


class FakeYoutubeDL:
    """Supply configurable yt-dlp results while recording constructor options."""

    captured_options: ClassVar[list[dict[str, object]]] = []
    prepared_path: ClassVar[Path] = Path()
    progress_status: ClassVar[dict[str, object] | None] = None
    raw_metadata: ClassVar[Mapping[str, object]] = {}

    def __init__(self, options: dict[str, object]) -> None:
        """Capture one yt-dlp constructor option mapping."""
        self._options = options
        self.captured_options.append(options)

    def __enter__(self) -> Self:
        """Return the configured fake context."""
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Accept every context-manager exit."""
        del exception_type, exception, traceback

    def extract_info(
        self,
        source_url: str,
        *,
        download: bool,
    ) -> Mapping[str, object]:
        """Return configured metadata after optional transfer progress."""
        del source_url
        if download and self.progress_status is not None:
            progress_hooks = self._options["progress_hooks"]
            assert isinstance(progress_hooks, list)
            for progress_hook in progress_hooks:
                assert callable(progress_hook)
                progress_hook(self.progress_status)
        return self.raw_metadata

    def prepare_filename(self, metadata: Mapping[str, object]) -> Path:
        """Return the configured completed-media path."""
        del metadata
        return self.prepared_path


def isolated_transcription_config() -> TranscriptionConfig:
    """Return a test configuration independent of local environment files."""
    return TranscriptionConfig(
        whisper_model="large-v3-turbo",
        whisper_revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        whisper_device="cuda",
        whisper_device_index=0,
        whisper_compute_type="float16",
        max_duration_seconds=1800,
        temporary_media_root=Path("/tmp/textify-test"),
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


@pytest.mark.parametrize(
    ("unusable_track_names", "expected_track_name", "expected_fetches"),
    (
        (frozenset(), "exact_manual", ("exact_manual",)),
        (
            frozenset({"exact_manual"}),
            "exact_generated",
            ("exact_manual", "exact_generated"),
        ),
        (
            frozenset({"exact_manual", "exact_generated"}),
            "base_manual",
            ("exact_manual", "exact_generated", "base_manual"),
        ),
        (
            frozenset({"exact_manual", "exact_generated", "base_manual"}),
            "base_generated",
            ("exact_manual", "exact_generated", "base_manual", "base_generated"),
        ),
    ),
)
def test_acquire_transcript_prefers_original_declared_language_tracks(
    unusable_track_names: frozenset[str],
    expected_track_name: str,
    expected_fetches: tuple[str, ...],
) -> None:
    """Caption acquisition ranks exact then base original-language tracks."""
    track_specs = (
        ("exact_manual", "EN_us", False),
        ("exact_generated", "en-US", True),
        ("base_manual", "en", False),
        ("base_generated", "en", True),
        ("regional_sibling", "en-GB", False),
        ("unrelated", "fr", False),
        ("invalid", "not a language", False),
    )
    tracks = tuple(
        FakeCaptionTrack(
            name,
            language_code,
            is_generated,
            () if name in unusable_track_names else ((0.0, 1.0, name),),
        )
        for name, language_code, is_generated in track_specs
    )
    provider = FakeCaptionProvider(tracks)

    transcript = acquire_transcript(provider, "dQw4w9WgXcQ", "en-US")

    assert transcript is not None
    assert transcript.text == expected_track_name
    assert provider.list_calls == ["dQw4w9WgXcQ"]
    assert (
        tuple(track.name for track in tracks if track.fetch_calls) == expected_fetches
    )


def test_acquire_transcript_preserves_provider_order_for_equal_rank_tracks() -> None:
    """Equal-rank tracks retain listing order until one is usable."""
    first_track = FakeCaptionTrack("first", "en-US", True, ())
    second_track = FakeCaptionTrack("second", "en-US", True, ((0.0, 1.0, "two"),))
    provider = FakeCaptionProvider((first_track, second_track))

    transcript = acquire_transcript(provider, "dQw4w9WgXcQ", "en-US")

    assert transcript is not None
    assert transcript.text == "two"
    assert (first_track.fetch_calls, second_track.fetch_calls) == (1, 1)


@pytest.mark.parametrize(
    "declared_language",
    (None, "und", "not a language tag", "x-private"),
)
def test_acquire_transcript_skips_ambiguous_caption_declarations(
    declared_language: str | None,
) -> None:
    """Missing or ambiguous declarations bypass caption-provider listing."""
    provider = FakeCaptionProvider()

    transcript = acquire_transcript(
        provider,
        "dQw4w9WgXcQ",
        declared_language,
    )

    assert transcript is None
    assert provider.list_calls == []


@pytest.mark.parametrize(
    "failure_type",
    (acquisition._CaptionProviderFailure, acquisition._CaptionProviderTimeout),
)
def test_acquire_transcript_preserves_listing_provider_failures(
    failure_type: type[Exception],
) -> None:
    """Caption listing failures retain their private boundary types."""
    provider = FakeCaptionProvider(failure=failure_type())

    with pytest.raises(failure_type):
        acquire_transcript(provider, "dQw4w9WgXcQ", "en-US")


def test_acquire_transcript_tries_next_track_after_ordinary_fetch_failure() -> None:
    """An ordinary original-track failure falls through to the next candidate."""
    failed_track = FakeCaptionTrack(
        "failed",
        "en-US",
        False,
        (),
        acquisition._CaptionProviderFailure(),
    )
    usable_track = FakeCaptionTrack("usable", "en-US", True, ((0.0, 1.0, "text"),))
    provider = FakeCaptionProvider((failed_track, usable_track))

    transcript = acquire_transcript(provider, "dQw4w9WgXcQ", "en-US")

    assert transcript is not None
    assert transcript.text == "text"
    assert (failed_track.fetch_calls, usable_track.fetch_calls) == (1, 1)


def test_acquire_transcript_stops_after_caption_fetch_timeout() -> None:
    """A candidate timeout abandons optional caption traversal immediately."""
    timed_out_track = FakeCaptionTrack(
        "timed_out",
        "en-US",
        False,
        (),
        TimeoutError(),
    )
    later_track = FakeCaptionTrack("later", "en-US", True, ((0.0, 1.0, "text"),))
    provider = FakeCaptionProvider((timed_out_track, later_track))

    with pytest.raises(acquisition._CaptionProviderTimeout):
        acquire_transcript(provider, "dQw4w9WgXcQ", "en-US")

    assert (timed_out_track.fetch_calls, later_track.fetch_calls) == (1, 0)


def test_youtube_caption_adapter_fetches_original_timed_snippets(
    monkeypatch: MonkeyPatch,
) -> None:
    """The library adapter normalizes only original unformatted caption snippets."""
    track = FakeLibraryCaptionTrack(
        "EN_us",
        False,
        (
            FakeCaptionSnippet(2.0, 1.0, " second "),
            FakeCaptionSnippet(0.0, 1.0, " first "),
            FakeCaptionSnippet(2.0, 0.0, " overlap "),
            FakeCaptionSnippet(3.0, 1.0, "\t"),
        ),
    )
    FakeYouTubeTranscriptApi.tracks = (track,)
    FakeYouTubeTranscriptApi.list_calls.clear()
    FakeYouTubeTranscriptApi.client_count = 0
    monkeypatch.setattr(acquisition, "YouTubeTranscriptApi", FakeYouTubeTranscriptApi)

    transcript = acquire_transcript(
        YouTubeCaptionProvider(),
        "dQw4w9WgXcQ",
        "en-US",
    )

    assert transcript is not None
    assert transcript.method is TranscriptMethod.YOUTUBE_CAPTIONS
    assert transcript.language == "en-US"
    assert transcript.text == "first second overlap"
    assert [
        (segment.start, segment.end, segment.text) for segment in transcript.segments
    ] == [
        (0.0, 1.0, "first"),
        (2.0, 3.0, "second"),
        (2.0, 2.0, "overlap"),
    ]
    assert FakeYouTubeTranscriptApi.client_count == 1
    assert FakeYouTubeTranscriptApi.list_calls == ["dQw4w9WgXcQ"]
    assert track.fetch_arguments == [False]
    assert track.translate_calls == 0


def test_load_whisper_transcriber_uses_model_environment_settings(
    monkeypatch: MonkeyPatch,
) -> None:
    """Configured model settings are forwarded to Faster-Whisper."""
    construction_arguments: list[tuple[str, dict[str, str | int]]] = []

    def record_model(
        model_name: str,
        **model_options: str | int,
    ) -> object:
        construction_arguments.append((model_name, model_options))
        return object()

    monkeypatch.setattr(
        "textify.transcription.acquisition.ctranslate2.get_cuda_device_count",
        lambda: 1,
    )
    monkeypatch.setattr(acquisition, "WhisperModel", record_model)
    settings = isolated_transcription_config().model_copy(
        update={
            "whisper_model": "small.en",
            "whisper_revision": "release-2026-09",
            "whisper_device": "cpu",
            "whisper_device_index": 2,
            "whisper_compute_type": "int8",
        }
    )

    adapter = acquisition.load_whisper_transcriber(settings)

    assert isinstance(adapter, FasterWhisperTranscriber)
    assert construction_arguments == [
        (
            "small.en",
            {
                "compute_type": "int8",
                "device": "cpu",
                "device_index": 2,
                "revision": "release-2026-09",
            },
        )
    ]


def test_faster_whisper_adapter_normalizes_timed_external_segments(
    tmp_path: Path,
) -> None:
    """Faster-Whisper output becomes a normalized timed transcript."""
    audio_path = tmp_path / "audio.webm"
    audio_path.touch()
    adapter = FasterWhisperTranscriber(
        FakeWhisperModel(
            (
                FakeWhisperSegment(2.0, 3.0, "  second "),
                FakeWhisperSegment(0.0, 1.0, "first"),
            ),
            "en_us",
        ),
        isolated_transcription_config(),
    )

    transcript = adapter.transcribe(audio_path)

    assert transcript.method is TranscriptMethod.FASTER_WHISPER
    assert transcript.language == "en-US"
    assert transcript.text == "first second"
    assert [(segment.start, segment.end) for segment in transcript.segments] == [
        (0.0, 1.0),
        (2.0, 3.0),
    ]


def test_faster_whisper_adapter_accepts_silent_media(tmp_path: Path) -> None:
    """Empty native model output is a successful silent transcript."""
    audio_path = tmp_path / "audio.webm"
    audio_path.touch()
    adapter = FasterWhisperTranscriber(
        FakeWhisperModel((), "en"),
        isolated_transcription_config(),
    )

    transcript = adapter.transcribe(audio_path)

    assert transcript.language == "und"
    assert transcript.segments == ()
    assert transcript.text == ""


def test_yt_dlp_retries_exactly_ten_attempts() -> None:
    """Retry behavior retains the documented no-delay ten-attempt limit."""
    attempts = 0

    def extract(source_url: str, *, download: bool) -> str:
        """Fail nine times, then return the source URL.

        Args:
            source_url: URL passed by the retry helper.
            download: Download flag passed by the retry helper.

        Returns:
            Source URL after the tenth call.
        """
        nonlocal attempts
        assert download is False
        attempts += 1
        if attempts < 10:
            raise RuntimeError("transient provider error")
        return source_url

    result = extract_info_with_retries(
        extract,
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        download=False,
        deadline=time.monotonic() + 1.0,
        cancellation_event=threading.Event(),
    )

    assert result.endswith("1234567890123456789")
    assert attempts == 10


def test_yt_dlp_expired_deadline_prevents_the_first_attempt() -> None:
    """An elapsed operation deadline rejects work before invoking yt-dlp."""
    attempts = 0

    def extract(source_url: str, *, download: bool) -> str:
        """Record an invocation that the expired deadline must prevent."""
        nonlocal attempts
        del download
        attempts += 1
        return source_url

    with pytest.raises(TimeoutError, match="deadline expired"):
        extract_info_with_retries(
            extract,
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            download=False,
            deadline=time.monotonic() - 1.0,
            cancellation_event=threading.Event(),
        )

    assert attempts == 0


def test_yt_dlp_cancellation_between_retries_prevents_the_next_attempt() -> None:
    """A cancellation signal set by one failure stops the next retry."""
    attempts = 0
    cancellation_event = threading.Event()

    def extract(source_url: str, *, download: bool) -> str:
        """Fail once after setting the cooperative cancellation signal."""
        nonlocal attempts
        del source_url, download
        attempts += 1
        cancellation_event.set()
        raise RuntimeError("retryable failure")

    with pytest.raises(DownloadCancelled, match="operation cancelled"):
        extract_info_with_retries(
            extract,
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            download=False,
            deadline=time.monotonic() + 1.0,
            cancellation_event=cancellation_event,
        )

    assert attempts == 1


def test_yt_dlp_progress_hook_enforces_deadline_and_cancellation() -> None:
    """Transfer callbacks retain deadline, cancellation, and inclusive byte limits."""
    expired_hook = build_yt_dlp_progress_hook(
        deadline=time.monotonic() - 1.0,
        cancellation_event=threading.Event(),
        max_media_bytes=100,
    )
    with pytest.raises(TimeoutError, match="deadline expired"):
        expired_hook({"status": "downloading"})

    cancellation_event = threading.Event()
    cancelled_hook = build_yt_dlp_progress_hook(
        deadline=time.monotonic() + 1.0,
        cancellation_event=cancellation_event,
        max_media_bytes=100,
    )
    cancellation_event.set()
    with pytest.raises(DownloadCancelled, match="operation cancelled"):
        cancelled_hook({"status": "downloading"})

    limited_hook = build_yt_dlp_progress_hook(
        deadline=time.monotonic() + 1.0,
        cancellation_event=threading.Event(),
        max_media_bytes=100,
    )
    limited_hook({"status": "downloading", "downloaded_bytes": 100})
    limited_hook({"status": "finished", "total_bytes": 100})
    with pytest.raises(MediaByteLimitExceeded):
        limited_hook({"status": "downloading", "downloaded_bytes": 101})
    with pytest.raises(MediaByteLimitExceeded):
        limited_hook({"status": "finished", "total_bytes": 101})


def test_audio_downloader_enforces_progress_byte_limit(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native audio download exposes streamed over-limit media to the policy layer."""
    destination = tmp_path / "media"
    destination.mkdir()
    FakeYoutubeDL.captured_options.clear()
    FakeYoutubeDL.raw_metadata = {}
    FakeYoutubeDL.prepared_path = destination / "audio.webm"
    FakeYoutubeDL.progress_status = {
        "status": "downloading",
        "downloaded_bytes": 101,
    }
    monkeypatch.setattr(
        "textify.transcription.acquisition.yt_dlp.YoutubeDL",
        FakeYoutubeDL,
    )

    downloader = YtDlpAudioDownloader(max_media_bytes=100)

    with pytest.raises(MediaByteLimitExceeded):
        downloader.download(
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            destination,
            deadline=time.monotonic() + 1.0,
            cancellation_event=threading.Event(),
        )

    assert FakeYoutubeDL.captured_options[0]["allowed_extractors"] == [
        r"^(?:facebook|facebook:reel|generic|instagram|tiktok|vm\.tiktok|youtube)$"
    ]
