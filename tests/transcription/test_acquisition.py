"""Tests for provider adapters and bounded yt-dlp retries."""

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from yt_dlp.utils import DownloadCancelled

from textify.transcription import acquisition
from textify.transcription.acquisition import FasterWhisperTranscriber
from textify.transcription.config import TranscriptionConfig
from textify.transcription.types import TranscriptMethod
from textify.transcription.util import (
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
        metadata_timeout_seconds=30.0,
        audio_download_timeout_seconds=300.0,
        transcription_queue_timeout_seconds=300.0,
        transcription_timeout_seconds=1800.0,
        initial_prompt="test prompt",
        hf_token=None,
    )


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
    """Transfer callbacks reject expired and explicitly cancelled operations."""
    expired_hook = build_yt_dlp_progress_hook(
        deadline=time.monotonic() - 1.0,
        cancellation_event=threading.Event(),
    )
    with pytest.raises(TimeoutError, match="deadline expired"):
        expired_hook({"status": "downloading"})

    cancellation_event = threading.Event()
    cancelled_hook = build_yt_dlp_progress_hook(
        deadline=time.monotonic() + 1.0,
        cancellation_event=cancellation_event,
    )
    cancellation_event.set()
    with pytest.raises(DownloadCancelled, match="operation cancelled"):
        cancelled_hook({"status": "downloading"})
