"""Tests for the application-facing TranscriptService lifecycle."""

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

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
from textify.transcription.types import Segment, Transcript, TranscriptMethod

DIRECT_TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class EmptyCaptionProvider:
    """Supply no dormant caption tracks in TikTok-only tests."""

    def list_tracks(self, video_id: str) -> Sequence[CaptionTrack]:
        """Return no caption tracks.

        Args:
            video_id: Video identifier ignored by the baseline.

        Returns:
            Empty caption track sequence.
        """
        del video_id
        return ()


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


def build_service(
    temporary_media_root: Path,
    metadata_extractor: RecordingMetadataExtractor,
    audio_downloader: RecordingAudioDownloader,
    transcriber: FixedTranscriber,
    max_duration_seconds: int = 1800,
) -> TranscriptService:
    """Build a TranscriptService from deterministic provider boundaries.

    Args:
        temporary_media_root: Root for request-scoped test media.
        metadata_extractor: Deterministic metadata provider.
        audio_downloader: Deterministic audio provider.
        transcriber: Deterministic native inference provider.
        max_duration_seconds: Inclusive source duration ceiling.

    Returns:
        Fully wired service under one inference permit.
    """
    config = TranscriptionConfig(
        max_duration_seconds=max_duration_seconds,
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
        caption_provider=EmptyCaptionProvider(),
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
async def test_transcribe_rejects_disabled_platform_before_provider_calls(
    tmp_path: Path,
) -> None:
    """A recognizable non-TikTok URL does not reach any provider boundary."""
    extractor = RecordingMetadataExtractor(tiktok_metadata())
    downloader = RecordingAudioDownloader()
    service = build_service(tmp_path, extractor, downloader, FixedTranscriber())

    with pytest.raises(UnsupportedPlatformError):
        await service.transcribe(YOUTUBE_URL)

    assert extractor.calls == []
    assert downloader.calls == []


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
