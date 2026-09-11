"""Tests for submitted URL classification and source duration validation."""

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Self

import pytest
from pytest import MonkeyPatch

from textify.transcription.exceptions import (
    InvalidMediaDurationError,
    InvalidUrlError,
    UnsupportedPlatformError,
)
from textify.transcription.inspection import (
    YtDlpMetadataExtractor,
    _normalize_duration,
    classify_submitted_url,
)
from textify.transcription.types import Platform
from textify.transcription.util import MediaByteLimitExceeded


class FakeYoutubeDL:
    """Supply configurable yt-dlp metadata while recording constructor options."""

    captured_options: ClassVar[list[dict[str, object]]] = []
    metadata_by_download: ClassVar[dict[bool, Mapping[str, object]]] = {}
    prepared_path: ClassVar[Path] = Path()
    progress_status: ClassVar[dict[str, object] | None] = None

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
        return self.metadata_by_download[download]

    def prepare_filename(self, metadata: Mapping[str, object]) -> Path:
        """Return the configured completed-media path."""
        del metadata
        return self.prepared_path


def _configure_fake_youtube_dl(
    monkeypatch: MonkeyPatch,
    metadata_by_download: dict[bool, Mapping[str, object]],
    *,
    prepared_path: Path,
    progress_status: dict[str, object] | None = None,
) -> None:
    """Configure the module-local yt-dlp fake for one inspection test."""
    FakeYoutubeDL.captured_options.clear()
    FakeYoutubeDL.metadata_by_download = metadata_by_download
    FakeYoutubeDL.prepared_path = prepared_path
    FakeYoutubeDL.progress_status = progress_status
    monkeypatch.setattr(
        "textify.transcription.inspection.yt_dlp.YoutubeDL",
        FakeYoutubeDL,
    )


@pytest.mark.parametrize(
    ("url", "expected_url"),
    (
        (
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            "https://www.tiktok.com/@creator/video/1234567890123456789",
        ),
        (
            "https://www.tiktok.com/embed/1234567890123456789",
            "https://www.tiktok.com/embed/1234567890123456789",
        ),
        (
            "https://www.tiktok.com/t/abcdefgh",
            "https://www.tiktok.com/t/abcdefgh",
        ),
        ("https://vm.tiktok.com/abcdefgh", "https://vm.tiktok.com/abcdefgh"),
        ("https://vt.tiktok.com/abcdefgh", "https://vt.tiktok.com/abcdefgh"),
    ),
)
def test_classify_submitted_url_accepts_current_tiktok_forms(
    url: str,
    expected_url: str,
) -> None:
    """Every enabled TikTok URL form remains a TikTok provider request."""
    submitted = classify_submitted_url(url)

    assert submitted.platform is Platform.TIKTOK
    assert submitted.provider_url == expected_url


@pytest.mark.parametrize(
    ("url", "error_type"),
    (
        ("https://www.tiktok.com/@creator", InvalidUrlError),
        ("https://www.tiktok.com.evil.example/@creator/video/123", InvalidUrlError),
        ("https://example.com/video/123", UnsupportedPlatformError),
        ("not a url", InvalidUrlError),
    ),
)
def test_classify_submitted_url_rejects_invalid_and_unknown_urls(
    url: str,
    error_type: type[Exception],
) -> None:
    """Malformed URL shapes differ from syntactically valid unknown platforms."""
    with pytest.raises(error_type):
        classify_submitted_url(url)


@pytest.mark.parametrize("duration", (True, 0, -1, float("inf")))
def test_normalize_duration_rejects_invalid_media_duration(duration: object) -> None:
    """Metadata duration must be finite, positive, and nonboolean."""
    with pytest.raises(InvalidMediaDurationError):
        _normalize_duration({"duration": duration})


def test_normalize_duration_rounds_positive_fraction_upward() -> None:
    """Positive provider duration is rounded upward for admission decisions."""
    assert _normalize_duration({"duration": 1799.1}) == 1800


def test_metadata_extractor_accepts_selected_media_at_byte_limit(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exact selected media size takes priority and accepts the inclusive limit."""
    metadata = {
        "duration": 12,
        "filesize": 100,
        "filesize_approx": 101,
    }
    _configure_fake_youtube_dl(
        monkeypatch,
        {False: metadata},
        prepared_path=tmp_path / "audio.webm",
    )
    extractor = YtDlpMetadataExtractor(1800, 100, tmp_path)

    extracted = extractor.extract(
        "https://www.tiktok.com/@creator/video/1234567890123456789",
        deadline=time.monotonic() + 1.0,
        cancellation_event=threading.Event(),
    )

    assert extracted.metadata == metadata
    assert FakeYoutubeDL.captured_options[0]["format"] == "bestaudio/best"


@pytest.mark.parametrize(
    "metadata",
    (
        {"duration": 12, "filesize": 101},
        {"duration": 12, "filesize": None, "filesize_approx": 101},
    ),
)
def test_metadata_extractor_rejects_oversized_selected_media(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    metadata: Mapping[str, object],
) -> None:
    """Exact and approximate selected sizes above the limit are rejected."""
    _configure_fake_youtube_dl(
        monkeypatch,
        {False: metadata},
        prepared_path=tmp_path / "audio.webm",
    )
    extractor = YtDlpMetadataExtractor(1800, 100, tmp_path)

    with pytest.raises(MediaByteLimitExceeded):
        extractor.extract(
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            deadline=time.monotonic() + 1.0,
            cancellation_event=threading.Event(),
        )


def test_duration_fallback_enforces_progress_byte_limit_and_cleans(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fallback download rejects streamed oversized media and removes its directory."""
    _configure_fake_youtube_dl(
        monkeypatch,
        {
            False: {"duration": None},
            True: {"duration": 12},
        },
        prepared_path=tmp_path / "audio.webm",
        progress_status={"status": "downloading", "downloaded_bytes": 101},
    )
    extractor = YtDlpMetadataExtractor(1800, 100, tmp_path)

    with pytest.raises(MediaByteLimitExceeded):
        extractor.extract(
            "https://www.tiktok.com/@creator/video/1234567890123456789",
            deadline=time.monotonic() + 1.0,
            cancellation_event=threading.Event(),
        )

    assert not any(tmp_path.iterdir())
