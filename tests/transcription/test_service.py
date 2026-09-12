"""Tests for the application-facing TranscriptService lifecycle."""

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from textify.transcription import acquisition
from textify.transcription.acquisition import CaptionTrack
from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import (
    AudioDownloadTimeoutError,
    MetadataRetrievalFailedError,
    MetadataTimeoutError,
    TranscriptionFailedError,
    UnsupportedMediaError,
    UnsupportedPlatformError,
    VideoTooLongError,
)
from textify.transcription.inspection import ExtractedMetadata, PreparedAudio
from textify.transcription.service import TranscriptionAdapters, TranscriptService
from textify.transcription.types import (
    RawSegment,
    Segment,
    Transcript,
    TranscriptMethod,
)

DIRECT_TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
X_URL = "https://x.com/creator/status/1234567890123456789"


class EmptyCaptionProvider:
    """Supply no active caption tracks."""

    def list_tracks(self, video_id: str) -> Sequence[CaptionTrack]:
        """Return no caption tracks.

        Args:
            video_id: Video identifier ignored by the baseline.

        Returns:
            Empty caption track sequence.
        """
        del video_id
        return ()


class RecordingCaptionTrack:
    """Return deterministic caption segments or one provider failure."""

    def __init__(
        self,
        language_code: str,
        is_generated: bool,
        segments: tuple[RawSegment, ...] = (),
        failure: Exception | None = None,
    ) -> None:
        """Initialize deterministic caption track behavior.

        Args:
            language_code: Provider-reported original language tag.
            is_generated: Whether the provider generated the track.
            segments: Timed caption segments to return.
            failure: Optional fetch failure.
        """
        self._language_code = language_code
        self._is_generated = is_generated
        self._segments = segments
        self._failure = failure
        self.fetch_calls = 0

    @property
    def language_code(self) -> str:
        """Return the configured original language tag."""
        return self._language_code

    @property
    def is_generated(self) -> bool:
        """Return the configured generated-track flag."""
        return self._is_generated

    def fetch_segments(self) -> Sequence[RawSegment]:
        """Record and return configured timed captions.

        Returns:
            Configured raw caption segments.

        Raises:
            Exception: The configured caption fetch failure.
        """
        self.fetch_calls += 1
        if self._failure is not None:
            raise self._failure
        return self._segments


class RecordingCaptionProvider:
    """Return configured caption tracks while recording video lookups."""

    def __init__(
        self,
        tracks: Sequence[CaptionTrack] = (),
        failure: Exception | None = None,
    ) -> None:
        """Initialize deterministic caption provider behavior.

        Args:
            tracks: Tracks returned in provider order.
            failure: Optional listing failure.
        """
        self._tracks = tracks
        self._failure = failure
        self.calls: list[str] = []

    def list_tracks(self, video_id: str) -> Sequence[CaptionTrack]:
        """Record and return configured caption tracks.

        Args:
            video_id: Stable YouTube video identity.

        Returns:
            Configured caption tracks.

        Raises:
            Exception: The configured listing failure.
        """
        self.calls.append(video_id)
        if self._failure is not None:
            raise self._failure
        return self._tracks


class RecordingMetadataExtractor:
    """Return deterministic external-shaped metadata or one failure."""

    def __init__(
        self,
        metadata: Mapping[str, object],
        prepared_audio: PreparedAudio | None = None,
        failure: Exception | None = None,
    ) -> None:
        """Initialize deterministic metadata behavior.

        Args:
            metadata: External-shaped yt-dlp metadata mapping.
            prepared_audio: Optional audio produced during duration probing.
            failure: Optional provider failure to raise.
        """
        self.calls: list[str] = []
        self._metadata = metadata
        self._prepared_audio = prepared_audio
        self._failure = failure

    def extract(
        self,
        canonical_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> ExtractedMetadata:
        """Record and return the configured provider result.

        Args:
            canonical_url: Validated provider URL.
            deadline: Monotonic absolute metadata deadline.
            cancellation_event: Cooperative request-cancellation signal.

        Returns:
            Deterministic metadata result.

        Raises:
            Exception: Configured provider failure.
        """
        del deadline, cancellation_event
        self.calls.append(canonical_url)
        if self._failure is not None:
            raise self._failure
        return ExtractedMetadata(self._metadata, self._prepared_audio)


class RecordingAudioDownloader:
    """Create request-owned audio or raise one download failure."""

    def __init__(
        self,
        failure: Exception | None = None,
        size_bytes: int = 0,
    ) -> None:
        """Initialize deterministic audio download behavior.

        Args:
            failure: Optional provider failure to raise.
            size_bytes: Exact byte count for every completed test audio file.
        """
        self.calls: list[str] = []
        self.request_directories: list[Path] = []
        self._failure = failure
        self._size_bytes = size_bytes

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Create a request-owned fake audio file.

        Args:
            source_url: Original submitted TikTok URL.
            destination: Request-scoped destination directory.
            deadline: Monotonic absolute audio-download deadline.
            cancellation_event: Cooperative request-cancellation signal.

        Returns:
            Created request-owned audio file.

        Raises:
            Exception: Configured provider failure.
        """
        del deadline, cancellation_event
        self.calls.append(source_url)
        self.request_directories.append(destination)
        if self._failure is not None:
            raise self._failure
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"x" * self._size_bytes)
        return audio_path


class FixedTranscriber:
    """Return one Transcript or raise one inference failure."""

    def __init__(self, failure: Exception | None = None) -> None:
        """Initialize deterministic inference behavior.

        Args:
            failure: Optional provider failure to raise.
        """
        self.calls: list[Path] = []
        self._failure = failure

    def transcribe(self, audio_path: Path) -> Transcript:
        """Return a normalized transcript for the supplied audio.

        Args:
            audio_path: Request-owned local audio path.

        Returns:
            Deterministic timed Transcript.

        Raises:
            Exception: Configured inference failure.
        """
        self.calls.append(audio_path)
        if self._failure is not None:
            raise self._failure
        segment = Segment(0.0, 1.0, "Transcript")
        return Transcript(
            TranscriptMethod.FASTER_WHISPER,
            "en",
            (segment,),
            "Transcript",
        )


def tiktok_metadata(duration: object = 1800) -> Mapping[str, object]:
    """Return an external-shaped valid TikTok metadata mapping.

    Args:
        duration: Provider duration value.

    Returns:
        Metadata accepted by the TikTok normalization boundary.
    """
    return {
        "id": "1234567890123456789",
        "extractor_key": "TikTok",
        "webpage_url": DIRECT_TIKTOK_URL,
        "title": "A title",
        "description": "A description",
        "channel": "Creator",
        "duration": duration,
        "formats": ({"vcodec": "h264"},),
    }


def youtube_metadata(language: object = "en-US") -> Mapping[str, object]:
    """Return external-shaped valid YouTube metadata.

    Args:
        language: Optional provider-declared source language.

    Returns:
        Metadata accepted by the YouTube normalization boundary.
    """
    metadata: dict[str, object] = {
        "id": "dQw4w9WgXcQ",
        "title": "A title",
        "description": "A description",
        "channel": "Creator",
        "duration": 12,
    }
    if language is not None:
        metadata["language"] = language
    return metadata


def build_service(
    temporary_media_root: Path,
    metadata_extractor: RecordingMetadataExtractor,
    audio_downloader: RecordingAudioDownloader,
    transcriber: FixedTranscriber,
    caption_provider: acquisition.CaptionProvider | None = None,
) -> TranscriptService:
    """Build a TranscriptService from deterministic provider boundaries.

    Args:
        temporary_media_root: Root for request-scoped test media.
        metadata_extractor: Deterministic metadata provider.
        audio_downloader: Deterministic audio provider.
        transcriber: Deterministic native inference provider.
        caption_provider: Optional deterministic YouTube captions provider.

    Returns:
        Fully wired service under one inference permit.
    """
    config = TranscriptionConfig(
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
    adapters = TranscriptionAdapters(
        metadata_extractor=metadata_extractor,
        caption_provider=(
            caption_provider if caption_provider is not None else EmptyCaptionProvider()
        ),
        audio_downloader=audio_downloader,
        whisper_transcriber=transcriber,
    )
    return TranscriptService(adapters, config)


@pytest.mark.asyncio
async def test_transcribe_allows_the_inclusive_duration_and_cleans_media(
    tmp_path: Path,
) -> None:
    """A 1800-second source succeeds and leaves no request media behind."""
    extractor = RecordingMetadataExtractor(tiktok_metadata(1800))
    downloader = RecordingAudioDownloader()
    service = build_service(tmp_path, extractor, downloader, FixedTranscriber())

    result = await service.transcribe(DIRECT_TIKTOK_URL)

    assert result.source.duration_seconds == 1800
    assert result.transcript.text == "Transcript"
    assert downloader.calls == [DIRECT_TIKTOK_URL]
    assert all(not directory.exists() for directory in downloader.request_directories)


@pytest.mark.asyncio
async def test_transcribe_accepts_completed_audio_at_byte_limit(
    tmp_path: Path,
) -> None:
    """An audio file exactly at the configured limit reaches native inference."""
    extractor = RecordingMetadataExtractor(tiktok_metadata())
    downloader = RecordingAudioDownloader(size_bytes=1024)
    transcriber = FixedTranscriber()
    service = build_service(tmp_path, extractor, downloader, transcriber)

    result = await service.transcribe(DIRECT_TIKTOK_URL)

    assert result.transcript.text == "Transcript"
    assert len(transcriber.calls) == 1
    assert all(not directory.exists() for directory in downloader.request_directories)


@pytest.mark.asyncio
async def test_transcribe_rejects_oversized_injected_metadata(tmp_path: Path) -> None:
    """Metadata adapters cannot bypass the configured selected-media byte limit."""
    metadata = dict(tiktok_metadata())
    metadata["filesize"] = 1025
    extractor = RecordingMetadataExtractor(metadata)
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_service(tmp_path, extractor, downloader, transcriber)

    with pytest.raises(UnsupportedMediaError):
        await service.transcribe(DIRECT_TIKTOK_URL)

    assert downloader.calls == []
    assert transcriber.calls == []


@pytest.mark.asyncio
async def test_transcribe_rejects_oversized_prepared_audio(tmp_path: Path) -> None:
    """Prepared inspection audio cannot bypass the exact completed-file limit."""
    prepared_directory = tmp_path / "inspection"
    prepared_directory.mkdir()
    prepared_path = prepared_directory / "audio.webm"
    prepared_path.write_bytes(b"x" * 1025)
    extractor = RecordingMetadataExtractor(
        tiktok_metadata(),
        PreparedAudio(prepared_path, prepared_directory),
    )
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_service(tmp_path, extractor, downloader, transcriber)

    with pytest.raises(UnsupportedMediaError):
        await service.transcribe(DIRECT_TIKTOK_URL)

    assert downloader.calls == []
    assert transcriber.calls == []
    assert not prepared_directory.exists()


@pytest.mark.asyncio
async def test_transcribe_rejects_duration_above_the_inclusive_ceiling(
    tmp_path: Path,
) -> None:
    """A rounded duration above the configured ceiling never downloads audio."""
    extractor = RecordingMetadataExtractor(tiktok_metadata(1800.1))
    downloader = RecordingAudioDownloader()
    service = build_service(tmp_path, extractor, downloader, FixedTranscriber())

    with pytest.raises(VideoTooLongError):
        await service.transcribe(DIRECT_TIKTOK_URL)

    assert downloader.calls == []


@pytest.mark.asyncio
async def test_transcribe_cleans_inspection_audio_after_inference_failure(
    tmp_path: Path,
) -> None:
    """Prepared duration-probe audio is removed after ordinary inference failure."""
    prepared_directory = tmp_path / "inspection"
    prepared_directory.mkdir()
    prepared_path = prepared_directory / "audio.webm"
    prepared_path.touch()
    prepared_audio = PreparedAudio(prepared_path, prepared_directory)
    extractor = RecordingMetadataExtractor(tiktok_metadata(), prepared_audio)
    downloader = RecordingAudioDownloader()
    service = build_service(
        tmp_path,
        extractor,
        downloader,
        FixedTranscriber(RuntimeError("native provider failed")),
    )

    with pytest.raises(TranscriptionFailedError):
        await service.transcribe(DIRECT_TIKTOK_URL)

    assert not prepared_directory.exists()
    assert downloader.calls == []


@pytest.mark.asyncio
async def test_transcribe_rejects_still_disabled_platform_before_provider_calls(
    tmp_path: Path,
) -> None:
    """A recognizable disabled URL does not reach any provider boundary."""
    extractor = RecordingMetadataExtractor(tiktok_metadata())
    caption_provider = RecordingCaptionProvider()
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_service(
        tmp_path,
        extractor,
        downloader,
        transcriber,
        caption_provider,
    )

    with pytest.raises(UnsupportedPlatformError):
        await service.transcribe(X_URL)

    assert extractor.calls == []
    assert caption_provider.calls == []
    assert downloader.calls == []
    assert transcriber.calls == []


@pytest.mark.asyncio
async def test_transcribe_translates_metadata_and_download_timeouts(
    tmp_path: Path,
) -> None:
    """Metadata and audio stages retain distinct safe domain failure types."""
    metadata_timeout_service = build_service(
        tmp_path,
        RecordingMetadataExtractor(tiktok_metadata(), failure=TimeoutError()),
        RecordingAudioDownloader(),
        FixedTranscriber(),
    )
    download_timeout_service = build_service(
        tmp_path,
        RecordingMetadataExtractor(tiktok_metadata()),
        RecordingAudioDownloader(TimeoutError()),
        FixedTranscriber(),
    )

    with pytest.raises(MetadataTimeoutError):
        await metadata_timeout_service.transcribe(DIRECT_TIKTOK_URL)
    with pytest.raises(AudioDownloadTimeoutError):
        await download_timeout_service.transcribe(DIRECT_TIKTOK_URL)


@pytest.mark.asyncio
async def test_transcribe_translates_ordinary_metadata_failures(tmp_path: Path) -> None:
    """Ordinary metadata exceptions become a safe metadata stage error."""
    service = build_service(
        tmp_path,
        RecordingMetadataExtractor(tiktok_metadata(), failure=RuntimeError("failed")),
        RecordingAudioDownloader(),
        FixedTranscriber(),
    )

    with pytest.raises(MetadataRetrievalFailedError):
        await service.transcribe(DIRECT_TIKTOK_URL)


@pytest.mark.asyncio
async def test_transcribe_returns_youtube_captions_without_native_work(
    tmp_path: Path,
) -> None:
    """A usable original caption returns before audio download or inference."""
    caption_track = RecordingCaptionTrack(
        "en-US",
        False,
        ((0.0, 1.0, "caption"),),
    )
    caption_provider = RecordingCaptionProvider((caption_track,))
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_service(
        tmp_path,
        RecordingMetadataExtractor(youtube_metadata()),
        downloader,
        transcriber,
        caption_provider,
    )

    result = await service.transcribe(YOUTUBE_URL)

    assert result.source.video_id == "dQw4w9WgXcQ"
    assert result.source.url == YOUTUBE_URL
    assert result.transcript.method is TranscriptMethod.YOUTUBE_CAPTIONS
    assert result.transcript.text == "caption"
    assert caption_provider.calls == ["dQw4w9WgXcQ"]
    assert downloader.calls == []
    assert transcriber.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata_language", "tracks", "failure", "expected_caption_calls"),
    (
        ("en-US", (), None, 1),
        (None, (), None, 0),
        ("en-US", (), acquisition._CaptionProviderFailure(), 1),
        (
            "en-US",
            (RecordingCaptionTrack("en-US", False, ()),),
            None,
            1,
        ),
    ),
    ids=("missing", "ambiguous", "inaccessible", "unusable"),
)
async def test_transcribe_falls_back_after_unusable_youtube_captions(
    tmp_path: Path,
    metadata_language: object | None,
    tracks: Sequence[CaptionTrack],
    failure: Exception | None,
    expected_caption_calls: int,
) -> None:
    """Every optional-caption miss reaches the unchanged native path."""
    caption_provider = RecordingCaptionProvider(tracks, failure)
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_service(
        tmp_path,
        RecordingMetadataExtractor(youtube_metadata(metadata_language)),
        downloader,
        transcriber,
        caption_provider,
    )

    result = await service.transcribe(YOUTUBE_URL)

    assert result.transcript.method is TranscriptMethod.FASTER_WHISPER
    assert len(caption_provider.calls) == expected_caption_calls
    assert downloader.calls == [YOUTUBE_URL]
    assert len(transcriber.calls) == 1


@pytest.mark.asyncio
async def test_transcribe_cleans_prepared_audio_after_youtube_caption_success(
    tmp_path: Path,
) -> None:
    """Caption success cleans metadata-stage audio before releasing admission."""
    prepared_directory = tmp_path / "inspection"
    prepared_directory.mkdir()
    prepared_path = prepared_directory / "audio.webm"
    prepared_path.touch()
    caption_provider = RecordingCaptionProvider(
        (RecordingCaptionTrack("en-US", False, ((0.0, 1.0, "caption"),)),)
    )
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_service(
        tmp_path,
        RecordingMetadataExtractor(
            youtube_metadata(),
            PreparedAudio(prepared_path, prepared_directory),
        ),
        downloader,
        transcriber,
        caption_provider,
    )

    result = await service.transcribe(YOUTUBE_URL)

    assert result.transcript.method is TranscriptMethod.YOUTUBE_CAPTIONS
    assert not prepared_directory.exists()
    assert downloader.calls == []
    assert transcriber.calls == []


@pytest.mark.asyncio
async def test_transcribe_reuses_prepared_audio_after_youtube_caption_miss(
    tmp_path: Path,
) -> None:
    """A caption miss transfers inspection audio into the existing native path."""
    prepared_directory = tmp_path / "inspection"
    prepared_directory.mkdir()
    prepared_path = prepared_directory / "audio.webm"
    prepared_path.touch()
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_service(
        tmp_path,
        RecordingMetadataExtractor(
            youtube_metadata(),
            PreparedAudio(prepared_path, prepared_directory),
        ),
        downloader,
        transcriber,
        RecordingCaptionProvider(),
    )

    result = await service.transcribe(YOUTUBE_URL)

    assert result.transcript.method is TranscriptMethod.FASTER_WHISPER
    assert downloader.calls == []
    assert transcriber.calls == [prepared_path]
    assert not prepared_directory.exists()


@pytest.mark.asyncio
async def test_transcribe_preserves_native_failure_after_caption_failure(
    tmp_path: Path,
) -> None:
    """The terminal native safe error remains public after caption failure."""
    caption_provider = RecordingCaptionProvider(
        failure=acquisition._CaptionProviderFailure()
    )
    downloader = RecordingAudioDownloader()
    service = build_service(
        tmp_path,
        RecordingMetadataExtractor(youtube_metadata()),
        downloader,
        FixedTranscriber(RuntimeError("native provider failed")),
        caption_provider,
    )

    with pytest.raises(TranscriptionFailedError):
        await service.transcribe(YOUTUBE_URL)

    assert caption_provider.calls == ["dQw4w9WgXcQ"]
    assert downloader.calls == [YOUTUBE_URL]
    assert all(not directory.exists() for directory in downloader.request_directories)
