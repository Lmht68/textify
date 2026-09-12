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
    MetadataRetrievalFailedError,
    UnsupportedMediaError,
    UnsupportedPlatformError,
)
from textify.transcription.inspection import (
    SubmittedSource,
    YtDlpMetadataExtractor,
    _normalize_duration,
    classify_submitted_url,
    normalize_processed_metadata,
)
from textify.transcription.types import Platform
from textify.transcription.util import MediaByteLimitExceeded


class FakeYoutubeDL:
    """Supply configurable yt-dlp metadata while recording constructor options."""

    captured_options: ClassVar[list[dict[str, object]]] = []
    calls: ClassVar[list[tuple[str, bool]]] = []
    added_extractors: ClassVar[list[object]] = []
    metadata_by_download: ClassVar[dict[bool, Mapping[str, object]]] = {}
    prepared_path: ClassVar[Path] = Path()
    progress_status: ClassVar[dict[str, object] | None] = None

    def __init__(self, options: dict[str, object]) -> None:
        """Capture one yt-dlp constructor option mapping."""
        self._options = options
        self.captured_options.append(options)

    def add_info_extractor(self, extractor: object) -> None:
        """Record the Textify-only Facebook extractor registration."""
        self.added_extractors.append(extractor)

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
        self.calls.append((source_url, download))
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
    """Configure the shared yt-dlp factory fake for one inspection test."""
    FakeYoutubeDL.captured_options.clear()
    FakeYoutubeDL.calls.clear()
    FakeYoutubeDL.added_extractors.clear()
    FakeYoutubeDL.metadata_by_download = metadata_by_download
    FakeYoutubeDL.prepared_path = prepared_path
    FakeYoutubeDL.progress_status = progress_status
    monkeypatch.setattr(
        "textify.transcription.util.yt_dlp.YoutubeDL",
        FakeYoutubeDL,
    )


_YOUTUBE_VIDEO_ID = "dQw4w9WgXcQ"
_FACEBOOK_VIDEO_ID = "123456789012345"
_TIKTOK_VIDEO_ID = "1234567890123456789"
_X_STATUS_ID = "1234567890123456789"

_EXPECTED_YTDLP_POLICY: dict[str, object] = {
    "allowed_extractors": [
        r"^(?:facebook|facebook:reel|instagram|tiktok|twitter|twitter:shortener|vm\.tiktok|youtube)$"
    ],
    "format": "bestaudio/best",
    "ignoreconfig": True,
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "playlist_items": "1",
    "extract_flat": False,
    "ignoreerrors": False,
    "cookiefile": None,
    "cookiesfrombrowser": None,
    "usenetrc": False,
    "netrc_location": None,
    "netrc_cmd": None,
    "remote_components": [],
    "external_downloader": {"default": "native"},
    "external_downloader_args": {},
    "force_generic_extractor": False,
    "enable_file_urls": False,
    "default_search": None,
    "prefer_insecure": False,
}


def _assert_shared_ytdlp_policy(options: Mapping[str, object]) -> None:
    """Assert the complete non-overridable yt-dlp policy on one operation."""
    assert {
        option_name: options[option_name] for option_name in _EXPECTED_YTDLP_POLICY
    } == _EXPECTED_YTDLP_POLICY


_ACCEPTED_SUBMITTED_URL_CASES = (
    tuple(
        (
            Platform.TIKTOK,
            f"https://www.tiktok.com/{path}",
            f"https://www.tiktok.com/{path}",
        )
        for path in (
            f"@creator/video/{_TIKTOK_VIDEO_ID}",
            f"embed/{_TIKTOK_VIDEO_ID}",
            "t/short_1",
        )
    )
    + tuple(
        (
            Platform.TIKTOK,
            f"https://{host}/short_1",
            f"https://{host}/short_1",
        )
        for host in ("vm.tiktok.com", "vt.tiktok.com")
    )
    + tuple(
        (
            Platform.YOUTUBE,
            f"https://{host}/{path}",
            f"https://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}",
        )
        for host in (
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
        )
        for path in (
            f"watch?v={_YOUTUBE_VIDEO_ID}",
            f"shorts/{_YOUTUBE_VIDEO_ID}",
            f"embed/{_YOUTUBE_VIDEO_ID}",
            f"live/{_YOUTUBE_VIDEO_ID}",
        )
    )
    + (
        (
            Platform.YOUTUBE,
            f"https://youtu.be/{_YOUTUBE_VIDEO_ID}",
            f"https://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}",
        ),
    )
    + tuple(
        (
            Platform.INSTAGRAM,
            f"https://{host}/{path}",
            f"https://{host}/{path}",
        )
        for host in ("instagram.com", "www.instagram.com")
        for path in (
            "p/C0social_1",
            "tv/C0social_1",
            "reel/C0social_1",
            "reels/C0social_1",
        )
    )
    + tuple(
        (
            Platform.FACEBOOK,
            f"https://{host}/{path}",
            f"https://{host}/{path}",
        )
        for host in ("facebook.com", "www.facebook.com", "m.facebook.com")
        for path in (
            f"watch?v={_FACEBOOK_VIDEO_ID}",
            f"video.php?v={_FACEBOOK_VIDEO_ID}",
            f"video/video.php?v={_FACEBOOK_VIDEO_ID}",
            f"reel/{_FACEBOOK_VIDEO_ID}",
            "share/v/short_1",
            "share/owner/kind/short_1",
            f"creator/videos/{_FACEBOOK_VIDEO_ID}",
            f"creator/series_1/videos/{_FACEBOOK_VIDEO_ID}",
            f"creator/posts/{_FACEBOOK_VIDEO_ID}",
        )
    )
    + ((Platform.FACEBOOK, "https://fb.watch/short_1", "https://fb.watch/short_1"),)
    + tuple(
        (
            Platform.X,
            f"https://{host}/{path}",
            f"https://{host}/{path}",
        )
        for host in (
            "x.com",
            "www.x.com",
            "m.x.com",
            "mobile.x.com",
            "twitter.com",
            "www.twitter.com",
            "m.twitter.com",
            "mobile.twitter.com",
        )
        for path in (
            f"creator/status/{_X_STATUS_ID}",
            f"creator/status/{_X_STATUS_ID}/video/1",
            f"i/web/status/{_X_STATUS_ID}",
            f"i/web/status/{_X_STATUS_ID}/video/1",
            f"statuses/{_X_STATUS_ID}",
            f"statuses/{_X_STATUS_ID}/video/1",
        )
    )
    + ((Platform.X, "https://t.co/short_1", "https://t.co/short_1"),)
    + (
        (
            Platform.TIKTOK,
            f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}?utm_source=test#fragment",
            f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}?utm_source=test",
        ),
        (
            Platform.TIKTOK,
            f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}?V=1",
            f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}?V=1",
        ),
        (
            Platform.YOUTUBE,
            f"https://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}&utm_source=test#fragment",
            f"https://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}&utm_source=test",
        ),
        (
            Platform.INSTAGRAM,
            "https://www.instagram.com/reel/C0social_1/?utm_source=test#fragment",
            "https://www.instagram.com/reel/C0social_1?utm_source=test",
        ),
        (
            Platform.FACEBOOK,
            "https://www.facebook.com/share/v/1HQUctaBZS/",
            "https://www.facebook.com/share/v/1HQUctaBZS",
        ),
        (
            Platform.FACEBOOK,
            f"https://www.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}&utm_source=test#fragment",
            f"https://www.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}&utm_source=test",
        ),
        (
            Platform.FACEBOOK,
            f"https://www.facebook.com/reel/{_FACEBOOK_VIDEO_ID}?v=1",
            f"https://www.facebook.com/reel/{_FACEBOOK_VIDEO_ID}?v=1",
        ),
        (
            Platform.X,
            f"https://twitter.com/creator/status/{_X_STATUS_ID}?utm_source=test#fragment",
            f"https://twitter.com/creator/status/{_X_STATUS_ID}?utm_source=test",
        ),
    )
)


@pytest.mark.parametrize(
    ("platform", "url", "expected_provider_url"),
    _ACCEPTED_SUBMITTED_URL_CASES,
)
def test_classify_submitted_url_accepts_exact_matrix(
    platform: Platform,
    url: str,
    expected_provider_url: str,
) -> None:
    """Every matrix URL produces a validated provider request."""
    submitted = classify_submitted_url(url)

    assert submitted.platform is platform
    assert submitted.provider_url == expected_provider_url
    assert submitted.youtube_video_id == (
        _YOUTUBE_VIDEO_ID if platform is Platform.YOUTUBE else None
    )


@pytest.mark.parametrize(
    ("url", "error_type"),
    (
        (
            f"http://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}",
            InvalidUrlError,
        ),
        (
            f"ftp://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}",
            InvalidUrlError,
        ),
        ("https://[::1", InvalidUrlError),
        (
            f"https://user:password@www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}",
            InvalidUrlError,
        ),
        (
            f"https://www.youtube.com:443/watch?v={_YOUTUBE_VIDEO_ID}",
            InvalidUrlError,
        ),
        (f"https://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}%", InvalidUrlError),
        (
            f"https://www.youtube.com/%2Fwatch?v={_YOUTUBE_VIDEO_ID}",
            InvalidUrlError,
        ),
        (
            f"https://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}\n",
            InvalidUrlError,
        ),
        (
            f"https://www.tiktok.com//@creator/video/{_TIKTOK_VIDEO_ID}",
            InvalidUrlError,
        ),
        ("https://www.instagram.com/reel/C0social_1//", InvalidUrlError),
        (f"https://www.facebook.com/reel/{_FACEBOOK_VIDEO_ID}/", InvalidUrlError),
        ("https://www.facebook.com/share/v/1HQUctaBZS//", InvalidUrlError),
        (
            f"https://twitter.com/creator/status/{_X_STATUS_ID}/",
            InvalidUrlError,
        ),
        (
            f"https://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}&v={_YOUTUBE_VIDEO_ID}",
            InvalidUrlError,
        ),
        (
            f"https://www.youtube.com/shorts/{_YOUTUBE_VIDEO_ID}?v={_YOUTUBE_VIDEO_ID}",
            InvalidUrlError,
        ),
        (
            f"https://www.facebook.com/watch?V={_FACEBOOK_VIDEO_ID}",
            InvalidUrlError,
        ),
        (
            f"https://www.tiktok.com.evil.example/@creator/video/{_TIKTOK_VIDEO_ID}",
            InvalidUrlError,
        ),
        (
            f"https://www.youtube.com.evil.example/watch?v={_YOUTUBE_VIDEO_ID}",
            InvalidUrlError,
        ),
        (
            f"https://www.youtube-nocookie.com/embed/{_YOUTUBE_VIDEO_ID}",
            UnsupportedPlatformError,
        ),
        ("https://example.com/video/123", UnsupportedPlatformError),
        ("https://www.tiktok.com/@creator", InvalidUrlError),
        ("https://www.youtube.com/channel/textify", InvalidUrlError),
        ("https://www.instagram.com/textify", InvalidUrlError),
        ("https://www.facebook.com/textify", InvalidUrlError),
        ("https://x.com/textify", InvalidUrlError),
    ),
)
def test_classify_submitted_url_rejects_out_of_policy_urls(
    url: str,
    error_type: type[Exception],
) -> None:
    """Malformed, deceptive, and non-video URLs never produce provider access."""
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


def test_yt_dlp_metadata_operations_enforce_shared_policy(
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
    assert FakeYoutubeDL.calls == [
        ("https://www.tiktok.com/@creator/video/1234567890123456789", False)
    ]
    assert len(FakeYoutubeDL.captured_options) == 1
    assert set(FakeYoutubeDL.captured_options[0]) == set(_EXPECTED_YTDLP_POLICY)
    _assert_shared_ytdlp_policy(FakeYoutubeDL.captured_options[0])


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

    assert FakeYoutubeDL.calls == [
        ("https://www.tiktok.com/@creator/video/1234567890123456789", False),
        ("https://www.tiktok.com/@creator/video/1234567890123456789", True),
    ]
    assert len(FakeYoutubeDL.captured_options) == 2
    metadata_options, duration_options = FakeYoutubeDL.captured_options
    assert set(metadata_options) == set(_EXPECTED_YTDLP_POLICY)
    _assert_shared_ytdlp_policy(metadata_options)
    assert set(duration_options) == {
        *_EXPECTED_YTDLP_POLICY,
        "outtmpl",
        "match_filter",
        "progress_hooks",
    }
    _assert_shared_ytdlp_policy(duration_options)
    outtmpl = duration_options["outtmpl"]
    assert isinstance(outtmpl, str)
    assert Path(outtmpl).name == "audio.%(ext)s"
    assert callable(duration_options["match_filter"])
    progress_hooks = duration_options["progress_hooks"]
    assert isinstance(progress_hooks, list)
    assert len(progress_hooks) == 1
    assert callable(progress_hooks[0])
    assert len(FakeYoutubeDL.added_extractors) == 2
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("language", "expected_language"),
    (
        ("en_us", "en-US"),
        (None, None),
        ("not a language tag", None),
        ("und", None),
        ("x-private", None),
    ),
)
def test_normalize_processed_metadata_carries_safe_declared_language(
    language: object,
    expected_language: str | None,
) -> None:
    """YouTube metadata retains only usable canonical language declarations."""
    metadata = {
        "id": "dQw4w9WgXcQ",
        "extractor_key": "Youtube",
        "title": "Title",
        "duration": 12,
        "language": language,
    }
    submitted = SubmittedSource(
        Platform.YOUTUBE,
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    )

    normalized = normalize_processed_metadata(metadata, submitted)

    assert normalized.declared_language == expected_language


@pytest.mark.parametrize(
    ("submitted", "metadata", "expected_video_id", "expected_url"),
    (
        (
            SubmittedSource(
                Platform.TIKTOK,
                f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}",
            ),
            {
                "id": _TIKTOK_VIDEO_ID,
                "extractor_key": "TikTok",
                "webpage_url": (
                    f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}"
                    "?utm_source=test#fragment"
                ),
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12.1,
                "formats": ({"vcodec": "h264"},),
            },
            _TIKTOK_VIDEO_ID,
            f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}",
        ),
        (
            SubmittedSource(
                Platform.INSTAGRAM,
                "https://www.instagram.com/reel/C0submitted_1",
            ),
            {
                "id": "C0social_1",
                "extractor_key": "Instagram",
                "webpage_url": (
                    "https://www.instagram.com/reel/C0social_1?utm_source=test#fragment"
                ),
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12.1,
                "formats": ({"vcodec": "h264"},),
            },
            "C0social_1",
            "https://www.instagram.com/reel/C0social_1",
        ),
        (
            SubmittedSource(
                Platform.FACEBOOK,
                "https://www.facebook.com/watch?v=123456789012345",
            ),
            {
                "id": "123456789012345",
                "extractor_key": "Facebook",
                "webpage_url": (
                    f"https://m.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}"
                    "&utm_source=test#fragment"
                ),
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12.1,
                "formats": ({"vcodec": "h264"},),
            },
            "123456789012345",
            f"https://m.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}",
        ),
        (
            SubmittedSource(
                Platform.FACEBOOK,
                "https://www.facebook.com/reel/123456789012345",
            ),
            {
                "id": "123456789012345",
                "extractor_key": "FacebookReel",
                "webpage_url": "https://www.facebook.com/reel/123456789012345",
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12.1,
                "formats": ({"vcodec": "h264"},),
            },
            "123456789012345",
            "https://www.facebook.com/reel/123456789012345",
        ),
        (
            SubmittedSource(
                Platform.X,
                "https://x.com/creator/status/1234567890123456789",
            ),
            {
                "id": "1234567890123456789",
                "extractor_key": "Twitter",
                "webpage_url": (
                    f"https://x.com/creator/status/{_X_STATUS_ID}"
                    "?utm_source=test#fragment"
                ),
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12.1,
                "formats": (
                    {
                        "url": "https://video.twimg.com/example.mp4",
                        "format_id": "http-832",
                    },
                ),
            },
            "1234567890123456789",
            "https://x.com/creator/status/1234567890123456789",
        ),
        (
            SubmittedSource(
                Platform.X,
                "https://twitter.com/creator/status/1234567890123456789",
            ),
            {
                "id": "1234567890123456789",
                "extractor_key": "Twitter",
                "webpage_url": (
                    f"https://mobile.twitter.com/creator/status/{_X_STATUS_ID}"
                    "?utm_source=test#fragment"
                ),
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12.1,
                "formats": (
                    {
                        "url": "https://video.twimg.com/example.mp4",
                        "format_id": "http-832",
                    },
                ),
            },
            "1234567890123456789",
            "https://x.com/creator/status/1234567890123456789",
        ),
    ),
)
def test_normalize_processed_metadata_canonicalizes_provider_url(
    submitted: SubmittedSource,
    metadata: Mapping[str, object],
    expected_video_id: str,
    expected_url: str,
) -> None:
    """Social metadata uses provider-authoritative identity and fields."""
    normalized = normalize_processed_metadata(metadata, submitted)

    assert normalized.video_id == expected_video_id
    assert normalized.canonical_url == expected_url
    assert normalized.title == "Title"
    assert normalized.description == "Description"
    assert normalized.channel == "Creator"
    assert normalized.duration_seconds == 13
    assert normalized.declared_language is None


@pytest.mark.parametrize(
    ("platform", "submitted_url", "webpage_url", "extractor_key", "error_type"),
    (
        (
            Platform.FACEBOOK,
            f"https://www.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}",
            "not a URL",
            "Facebook",
            MetadataRetrievalFailedError,
        ),
        (
            Platform.FACEBOOK,
            f"https://www.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}",
            f"https://www.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}&v={_FACEBOOK_VIDEO_ID}",
            "Facebook",
            InvalidUrlError,
        ),
        (
            Platform.FACEBOOK,
            f"https://www.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}",
            "https://www.instagram.com/reel/C0social_1",
            "Facebook",
            InvalidUrlError,
        ),
        (
            Platform.FACEBOOK,
            f"https://www.facebook.com/watch?v={_FACEBOOK_VIDEO_ID}",
            "https://fb.watch/short_1",
            "Facebook",
            MetadataRetrievalFailedError,
        ),
        (
            Platform.TIKTOK,
            f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}",
            "https://vm.tiktok.com/short_1",
            "TikTok",
            MetadataRetrievalFailedError,
        ),
        (
            Platform.TIKTOK,
            f"https://www.tiktok.com/@creator/video/{_TIKTOK_VIDEO_ID}",
            "https://vt.tiktok.com/short_1",
            "TikTok",
            MetadataRetrievalFailedError,
        ),
        (
            Platform.X,
            f"https://x.com/creator/status/{_X_STATUS_ID}",
            "https://t.co/short_1",
            "Twitter",
            MetadataRetrievalFailedError,
        ),
    ),
)
def test_normalize_processed_metadata_rejects_out_of_policy_provider_url(
    platform: Platform,
    submitted_url: str,
    webpage_url: str,
    extractor_key: str,
    error_type: type[Exception],
) -> None:
    """Provider URLs retain stable malformed, cross-platform, and short-link errors."""
    metadata = {
        "id": "provider-authoritative-id",
        "extractor_key": extractor_key,
        "webpage_url": webpage_url,
        "formats": ({"vcodec": "h264"},),
    }

    with pytest.raises(error_type):
        normalize_processed_metadata(
            metadata,
            SubmittedSource(
                platform,
                submitted_url,
                _YOUTUBE_VIDEO_ID if platform is Platform.YOUTUBE else None,
            ),
        )


def test_normalize_processed_metadata_rejects_missing_codec_http_facebook_format() -> (
    None
):
    """Only X accepts yt-dlp's missing-codec direct HTTP video formats."""
    metadata = {
        "id": "123456789012345",
        "extractor_key": "Facebook",
        "webpage_url": "https://www.facebook.com/watch?v=123456789012345",
        "formats": (
            {
                "url": "https://video.twimg.com/example.mp4",
                "format_id": "http-832",
            },
        ),
    }

    with pytest.raises(UnsupportedMediaError):
        normalize_processed_metadata(
            metadata,
            SubmittedSource(
                Platform.FACEBOOK,
                "https://www.facebook.com/watch?v=123456789012345",
            ),
        )
