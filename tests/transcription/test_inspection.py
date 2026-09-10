"""Tests for submitted URL classification and source duration validation."""

import pytest

from textify.transcription.exceptions import (
    InvalidMediaDurationError,
    InvalidUrlError,
    UnsupportedPlatformError,
)
from textify.transcription.inspection import _normalize_duration, classify_submitted_url
from textify.transcription.types import Platform


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
