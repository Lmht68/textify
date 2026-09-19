"""Tests for the application-facing TranscriptionExecutor lifecycle."""

import asyncio
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from textify.transcription import acquisition, inspection
from textify.transcription.acquisition import CaptionTrack
from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import (
    AudioDownloadTimeoutError,
    MetadataRetrievalFailedError,
    MetadataTimeoutError,
    TranscriptionFailedError,
    UnsupportedMediaError,
    VideoTooLongError,
)
from textify.transcription.inspection import ExtractedMetadata, PreparedAudio
from textify.transcription.service import (
    TranscriptionAdapters,
    TranscriptionExecutor,
)
from textify.transcription.types import (
    RawSegment,
    Segment,
    TimedTranscript,
    Transcript,
    TranscriptMethod,
)

DIRECT_TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
UNKNOWN_PLATFORM_URL = "https://example.com/video/123"


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
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> ExtractedMetadata:
        """Record and return the configured provider result.

        Args:
            provider_url: Validated provider URL.
            deadline: Monotonic absolute metadata deadline.
            cancellation_event: Cooperative request-cancellation signal.

        Returns:
            Deterministic metadata result.

        Raises:
            Exception: Configured provider failure.
        """
        del deadline, cancellation_event
        self.calls.append(provider_url)
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
            source_url: Validated provider URL.
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
        self.include_segments_calls: list[bool] = []
        self._failure = failure

    def transcribe(
        self,
        audio_path: Path,
        *,
        include_segments: bool = True,
    ) -> Transcript:
        """Return a normalized transcript for the supplied audio.

        Args:
            audio_path: Request-owned local audio path.
            include_segments: Whether to retain normalized timed segments.

        Returns:
            Deterministic text-only or timed Transcript.

        Raises:
            Exception: Configured inference failure.
        """
        self.calls.append(audio_path)
        self.include_segments_calls.append(include_segments)
        if self._failure is not None:
            raise self._failure
        if not include_segments:
            return Transcript(
                method=TranscriptMethod.FASTER_WHISPER,
                language="en",
                text="Transcript",
            )
        segment = Segment(0.0, 1.0, "Transcript")
        return TimedTranscript(
            method=TranscriptMethod.FASTER_WHISPER,
            language="en",
            text="Transcript",
            segments=(segment,),
        )


class BlockingMetadataExtractor:
    """Block metadata extraction until the caller cancellation event is set."""

    def __init__(self, temporary_media_root: Path, *, block: bool) -> None:
        """Initialize one controllable metadata boundary.

        Args:
            temporary_media_root: Root where late inspection media is created.
            block: Whether extraction waits for cooperative cancellation.
        """
        self._temporary_media_root = temporary_media_root
        self._block = block
        self.calls: list[str] = []
        self.entered = threading.Event()
        self.prepared_directories: list[Path] = []

    def extract(
        self,
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> ExtractedMetadata:
        """Return valid metadata after an optional cancellation handshake.

        Args:
            provider_url: Validated provider URL passed to metadata inspection.
            deadline: Monotonic metadata deadline ignored by this test boundary.
            cancellation_event: Cooperative cancellation signal supplied by caller.

        Returns:
            Valid TikTok metadata and late inspection media when blocked.
        """
        del deadline
        self.calls.append(provider_url)
        if not self._block:
            return ExtractedMetadata(tiktok_metadata())

        self.entered.set()
        cancellation_event.wait()
        prepared_directory = self._temporary_media_root / "late-inspection"
        prepared_directory.mkdir(parents=True)
        prepared_path = prepared_directory / "audio.webm"
        prepared_path.touch()
        self.prepared_directories.append(prepared_directory)
        return ExtractedMetadata(
            tiktok_metadata(),
            PreparedAudio(prepared_path, prepared_directory),
        )


class BlockingAudioDownloader:
    """Block a partial audio download until caller cancellation is set."""

    def __init__(self, *, block: bool) -> None:
        """Initialize one controllable audio-download boundary.

        Args:
            block: Whether audio completion waits for cooperative cancellation.
        """
        self._block = block
        self.calls: list[str] = []
        self.entered = threading.Event()
        self.request_directories: list[Path] = []

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Create partial audio and optionally wait for cooperative cancellation.

        Args:
            source_url: Validated provider URL passed to audio acquisition.
            destination: Request-owned output directory.
            deadline: Monotonic audio-download deadline ignored by this boundary.
            cancellation_event: Cooperative cancellation signal supplied by caller.

        Returns:
            Completed local audio path after the optional cancellation handshake.
        """
        del deadline
        self.calls.append(source_url)
        self.request_directories.append(destination)
        audio_path = destination / "audio.webm"
        audio_path.touch()
        if self._block:
            self.entered.set()
            cancellation_event.wait()
        return audio_path


def build_transcription_config(temporary_media_root: Path) -> TranscriptionConfig:
    """Build deterministic transcription configuration for executor tests.

    Args:
        temporary_media_root: Root for request-scoped test media.

    Returns:
        Configuration with one native inference permit and stable limits.
    """
    return TranscriptionConfig(
        max_duration_seconds=1800,
        temporary_media_root=temporary_media_root,
        beam_size=1,
        vad_filter=True,
        temperature=0.0,
        condition_on_previous_text=True,
        transcription_concurrency=1,
        max_media_bytes=1024,
        metadata_timeout_seconds=30.0,
        audio_download_timeout_seconds=300.0,
        transcription_timeout_seconds=1800.0,
        initial_prompt="test prompt",
        hf_token=None,
    )


def build_execution(
    config: TranscriptionConfig,
    metadata_extractor: inspection.MetadataExtractor,
    audio_downloader: acquisition.AudioDownloader,
    transcriber: acquisition.WhisperTranscriber,
    caption_provider: acquisition.CaptionProvider | None = None,
) -> TranscriptionExecutor:
    """Build request-independent execution from deterministic provider boundaries.

    Args:
        config: Shared validated execution configuration.
        metadata_extractor: Deterministic metadata provider.
        audio_downloader: Deterministic audio provider.
        transcriber: Deterministic native inference provider.
        caption_provider: Optional deterministic YouTube captions provider.

    Returns:
        Fully wired execution interface under one inference permit.
    """
    adapters = TranscriptionAdapters(
        metadata_extractor=metadata_extractor,
        caption_provider=(
            caption_provider if caption_provider is not None else EmptyCaptionProvider()
        ),
        audio_downloader=audio_downloader,
        whisper_transcriber=transcriber,
    )
    return TranscriptionExecutor(adapters, config)


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
        "extractor_key": "Youtube",
        "title": "A title",
        "description": "A description",
        "channel": "Creator",
        "duration": 12,
    }
    if language is not None:
        metadata["language"] = language
    return metadata


@pytest.mark.asyncio
async def test_transcribe_allows_the_inclusive_duration_and_cleans_media(
    tmp_path: Path,
) -> None:
    """A 1800-second source succeeds and leaves no request media behind."""
    extractor = RecordingMetadataExtractor(tiktok_metadata(1800))
    downloader = RecordingAudioDownloader()
    service = build_execution(
        build_transcription_config(tmp_path), extractor, downloader, FixedTranscriber()
    )

    result = await service.execute(
        inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
        cancellation_event=threading.Event(),
    )

    assert result.source.duration_seconds == 1800
    assert result.transcript.text == "Transcript"
    assert downloader.calls == [DIRECT_TIKTOK_URL]
    assert all(not directory.exists() for directory in downloader.request_directories)


@pytest.mark.asyncio
async def test_transcribe_forwards_text_only_selection_to_native_inference(
    tmp_path: Path,
) -> None:
    """Text-only native requests retain provider calls and media cleanup."""
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(tiktok_metadata()),
        downloader,
        transcriber,
    )

    result = await service.execute(
        inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
        cancellation_event=threading.Event(),
        include_segments=False,
    )

    assert result.transcript.method is TranscriptMethod.FASTER_WHISPER
    assert result.transcript.text == "Transcript"
    assert not hasattr(result.transcript, "segments")
    assert downloader.calls == [DIRECT_TIKTOK_URL]
    assert transcriber.include_segments_calls == [False]
    assert all(not directory.exists() for directory in downloader.request_directories)


@pytest.mark.asyncio
async def test_transcribe_preserves_validated_provider_parameters(
    tmp_path: Path,
) -> None:
    """Metadata and native download retain provider-issued TikTok parameters."""
    provider_url = (
        f"{DIRECT_TIKTOK_URL}?is_from_webapp=1&sender_device=pc"
        "&web_id=7615327134046848533"
    )
    submitted_url = f"{provider_url}#fragment"
    extractor = RecordingMetadataExtractor(tiktok_metadata())
    downloader = RecordingAudioDownloader()
    service = build_execution(
        build_transcription_config(tmp_path), extractor, downloader, FixedTranscriber()
    )

    await service.execute(
        inspection.classify_submitted_url(submitted_url),
        cancellation_event=threading.Event(),
    )

    assert extractor.calls == [provider_url]
    assert downloader.calls == [provider_url]


@pytest.mark.asyncio
async def test_transcribe_accepts_completed_audio_at_byte_limit(
    tmp_path: Path,
) -> None:
    """An audio file exactly at the configured limit reaches native inference."""
    extractor = RecordingMetadataExtractor(tiktok_metadata())
    downloader = RecordingAudioDownloader(size_bytes=1024)
    transcriber = FixedTranscriber()
    service = build_execution(
        build_transcription_config(tmp_path), extractor, downloader, transcriber
    )

    result = await service.execute(
        inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
        cancellation_event=threading.Event(),
    )

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
    service = build_execution(
        build_transcription_config(tmp_path), extractor, downloader, transcriber
    )

    with pytest.raises(UnsupportedMediaError):
        await service.execute(
            inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
            cancellation_event=threading.Event(),
        )

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
    service = build_execution(
        build_transcription_config(tmp_path), extractor, downloader, transcriber
    )

    with pytest.raises(UnsupportedMediaError):
        await service.execute(
            inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
            cancellation_event=threading.Event(),
        )

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
    service = build_execution(
        build_transcription_config(tmp_path), extractor, downloader, FixedTranscriber()
    )

    with pytest.raises(VideoTooLongError):
        await service.execute(
            inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
            cancellation_event=threading.Event(),
        )

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
    service = build_execution(
        build_transcription_config(tmp_path),
        extractor,
        downloader,
        FixedTranscriber(RuntimeError("native provider failed")),
    )

    with pytest.raises(TranscriptionFailedError):
        await service.execute(
            inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
            cancellation_event=threading.Event(),
        )

    assert not prepared_directory.exists()
    assert downloader.calls == []


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_transcribe_translates_metadata_and_download_timeouts(
    tmp_path: Path,
) -> None:
    """Metadata and audio stages retain distinct safe domain failure types."""
    metadata_timeout_service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(tiktok_metadata(), failure=TimeoutError()),
        RecordingAudioDownloader(),
        FixedTranscriber(),
    )
    download_timeout_service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(tiktok_metadata()),
        RecordingAudioDownloader(TimeoutError()),
        FixedTranscriber(),
    )

    with pytest.raises(MetadataTimeoutError):
        await metadata_timeout_service.execute(
            inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
            cancellation_event=threading.Event(),
        )
    with pytest.raises(AudioDownloadTimeoutError):
        await download_timeout_service.execute(
            inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
            cancellation_event=threading.Event(),
        )


@pytest.mark.asyncio
async def test_transcribe_translates_ordinary_metadata_failures(tmp_path: Path) -> None:
    """Ordinary metadata exceptions become a safe metadata stage error."""
    service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(tiktok_metadata(), failure=RuntimeError("failed")),
        RecordingAudioDownloader(),
        FixedTranscriber(),
    )

    with pytest.raises(MetadataRetrievalFailedError):
        await service.execute(
            inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
            cancellation_event=threading.Event(),
        )


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
    service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(youtube_metadata()),
        downloader,
        transcriber,
        caption_provider,
    )

    result = await service.execute(
        inspection.classify_submitted_url(YOUTUBE_URL),
        cancellation_event=threading.Event(),
    )

    assert result.source.video_id == "dQw4w9WgXcQ"
    assert result.source.url == YOUTUBE_URL
    assert result.transcript.method is TranscriptMethod.YOUTUBE_CAPTIONS
    assert result.transcript.text == "caption"
    assert caption_provider.calls == ["dQw4w9WgXcQ"]
    assert downloader.calls == []
    assert transcriber.calls == []


@pytest.mark.asyncio
async def test_transcribe_forwards_text_only_selection_to_youtube_captions(
    tmp_path: Path,
) -> None:
    """Text-only caption requests retain provider calls and prepared-media cleanup."""
    prepared_directory = tmp_path / "inspection"
    prepared_directory.mkdir()
    prepared_path = prepared_directory / "audio.webm"
    prepared_path.touch()
    caption_track = RecordingCaptionTrack(
        "en-US",
        False,
        ((0.0, 1.0, "caption"),),
    )
    caption_provider = RecordingCaptionProvider((caption_track,))
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(
            youtube_metadata(),
            PreparedAudio(prepared_path, prepared_directory),
        ),
        downloader,
        transcriber,
        caption_provider,
    )

    result = await service.execute(
        inspection.classify_submitted_url(YOUTUBE_URL),
        cancellation_event=threading.Event(),
        include_segments=False,
    )

    assert result.transcript.method is TranscriptMethod.YOUTUBE_CAPTIONS
    assert result.transcript.text == "caption"
    assert not hasattr(result.transcript, "segments")
    assert caption_provider.calls == ["dQw4w9WgXcQ"]
    assert caption_track.fetch_calls == 1
    assert downloader.calls == []
    assert transcriber.calls == []
    assert not prepared_directory.exists()


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
    service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(youtube_metadata(metadata_language)),
        downloader,
        transcriber,
        caption_provider,
    )

    result = await service.execute(
        inspection.classify_submitted_url(YOUTUBE_URL),
        cancellation_event=threading.Event(),
    )

    assert result.transcript.method is TranscriptMethod.FASTER_WHISPER
    assert len(caption_provider.calls) == expected_caption_calls
    assert downloader.calls == [YOUTUBE_URL]
    assert len(transcriber.calls) == 1


@pytest.mark.asyncio
async def test_transcribe_cleans_prepared_audio_after_youtube_caption_success(
    tmp_path: Path,
) -> None:
    """Caption success cleans metadata-stage audio before caller work ends."""
    prepared_directory = tmp_path / "inspection"
    prepared_directory.mkdir()
    prepared_path = prepared_directory / "audio.webm"
    prepared_path.touch()
    caption_provider = RecordingCaptionProvider(
        (RecordingCaptionTrack("en-US", False, ((0.0, 1.0, "caption"),)),)
    )
    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(
            youtube_metadata(),
            PreparedAudio(prepared_path, prepared_directory),
        ),
        downloader,
        transcriber,
        caption_provider,
    )

    result = await service.execute(
        inspection.classify_submitted_url(YOUTUBE_URL),
        cancellation_event=threading.Event(),
    )

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
    service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(
            youtube_metadata(),
            PreparedAudio(prepared_path, prepared_directory),
        ),
        downloader,
        transcriber,
        RecordingCaptionProvider(),
    )

    result = await service.execute(
        inspection.classify_submitted_url(YOUTUBE_URL),
        cancellation_event=threading.Event(),
    )

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
    service = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(youtube_metadata()),
        downloader,
        FixedTranscriber(RuntimeError("native provider failed")),
        caption_provider,
    )

    with pytest.raises(TranscriptionFailedError):
        await service.execute(
            inspection.classify_submitted_url(YOUTUBE_URL),
            cancellation_event=threading.Event(),
        )

    assert caption_provider.calls == ["dQw4w9WgXcQ"]
    assert downloader.calls == [YOUTUBE_URL]
    assert all(not directory.exists() for directory in downloader.request_directories)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caption_provider", "expected_method", "expected_text"),
    (
        (
            RecordingCaptionProvider(
                (
                    RecordingCaptionTrack(
                        "en-US",
                        False,
                        ((0.0, 1.0, "caption"),),
                    ),
                )
            ),
            TranscriptMethod.YOUTUBE_CAPTIONS,
            "caption",
        ),
        (
            RecordingCaptionProvider(),
            TranscriptMethod.FASTER_WHISPER,
            "Transcript",
        ),
    ),
    ids=("caption_bypass", "native_fallback"),
)
async def test_transcription_execution_runs_caption_bypass_and_native_fallback_without_http_request(
    tmp_path: Path,
    caption_provider: RecordingCaptionProvider,
    expected_method: TranscriptMethod,
    expected_text: str,
) -> None:
    """Direct execution bypasses HTTP while retaining caption and native branches."""
    prepared_directory: Path | None = None
    prepared_audio: PreparedAudio | None = None
    if expected_method is TranscriptMethod.YOUTUBE_CAPTIONS:
        prepared_directory = tmp_path / "caption-inspection"
        prepared_directory.mkdir()
        prepared_path = prepared_directory / "audio.webm"
        prepared_path.touch()
        prepared_audio = PreparedAudio(prepared_path, prepared_directory)

    downloader = RecordingAudioDownloader()
    transcriber = FixedTranscriber()
    executor = build_execution(
        build_transcription_config(tmp_path),
        RecordingMetadataExtractor(youtube_metadata(), prepared_audio),
        downloader,
        transcriber,
        caption_provider,
    )

    result = await executor.execute(
        inspection.classify_submitted_url(YOUTUBE_URL),
        cancellation_event=threading.Event(),
    )

    assert result.source.video_id == "dQw4w9WgXcQ"
    assert result.source.url == YOUTUBE_URL
    assert result.transcript.method is expected_method
    assert result.transcript.text == expected_text
    assert caption_provider.calls == ["dQw4w9WgXcQ"]
    if expected_method is TranscriptMethod.YOUTUBE_CAPTIONS:
        assert downloader.calls == []
        assert transcriber.calls == []
        assert prepared_directory is not None
        assert not prepared_directory.exists()
    else:
        assert downloader.calls == [YOUTUBE_URL]
        assert len(transcriber.calls) == 1
        assert all(
            not directory.exists() for directory in downloader.request_directories
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("metadata", "audio_download"))
async def test_transcription_execution_uses_caller_cancellation_for_pre_native_work(
    tmp_path: Path,
    stage: str,
) -> None:
    """Direct execution exposes one caller-owned pre-native cancellation signal."""
    metadata_extractor = BlockingMetadataExtractor(
        tmp_path,
        block=stage == "metadata",
    )
    audio_downloader = BlockingAudioDownloader(
        block=stage == "audio_download",
    )
    caption_provider = RecordingCaptionProvider()
    transcriber = FixedTranscriber()
    executor = build_execution(
        build_transcription_config(tmp_path),
        metadata_extractor,
        audio_downloader,
        transcriber,
        caption_provider,
    )
    cancellation_event = threading.Event()
    task = asyncio.create_task(
        executor.execute(
            inspection.classify_submitted_url(DIRECT_TIKTOK_URL),
            cancellation_event=cancellation_event,
        )
    )
    stage_entered = (
        metadata_extractor.entered if stage == "metadata" else audio_downloader.entered
    )

    assert await asyncio.to_thread(stage_entered.wait, 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancellation_event.is_set()
    assert caption_provider.calls == []
    assert transcriber.calls == []
    if stage == "metadata":
        assert audio_downloader.calls == []
        assert len(metadata_extractor.calls) == 1
        assert all(
            not directory.exists()
            for directory in metadata_extractor.prepared_directories
        )
    else:
        assert metadata_extractor.calls == [DIRECT_TIKTOK_URL]
        assert audio_downloader.calls == [DIRECT_TIKTOK_URL]
        assert all(
            not directory.exists() for directory in audio_downloader.request_directories
        )
