"""Validate Source URLs and normalize yt-dlp metadata."""

import logging
import re
import shutil
import socket
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import ceil, isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Final, Protocol, cast
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit

import av
import yt_dlp
from requests.exceptions import Timeout
from yt_dlp.utils import DownloadCancelled, DownloadError

from textify.transcription.exceptions import (
    InvalidMediaDurationError,
    InvalidUrlError,
    MetadataRetrievalFailedError,
    UnsupportedMediaError,
    UnsupportedPlatformError,
)
from textify.transcription.types import Platform
from textify.transcription.util import (
    build_yt_dlp_progress_hook,
    extract_info_with_retries,
    raise_if_cancelled_or_expired,
)

logger = logging.getLogger(__name__)

_MISSING = object()
_REDACTED_LOG_VALUE: Final[str] = "[REDACTED]"
_SENSITIVE_QUERY_PARTS: Final[frozenset[str]] = frozenset(
    {"api", "apikey", "auth", "authorization", "key", "password", "secret", "token"}
)

_YOUTUBE_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)
_INSTAGRAM_HOSTS: Final[frozenset[str]] = frozenset(
    {"instagram.com", "www.instagram.com"}
)
_FACEBOOK_HOSTS: Final[frozenset[str]] = frozenset(
    {"facebook.com", "www.facebook.com", "m.facebook.com"}
)
_FACEBOOK_SHORT_HOSTS: Final[frozenset[str]] = frozenset({"fb.watch"})
_TIKTOK_HOSTS: Final[frozenset[str]] = frozenset({"www.tiktok.com"})
_TIKTOK_SHORT_HOSTS: Final[frozenset[str]] = frozenset(
    {"vm.tiktok.com", "vt.tiktok.com"}
)
_X_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "x.com",
        "www.x.com",
        "m.x.com",
        "mobile.x.com",
        "twitter.com",
        "www.twitter.com",
        "m.twitter.com",
        "mobile.twitter.com",
    }
)
_X_SHORT_HOSTS: Final[frozenset[str]] = frozenset({"t.co"})
_SUPPORTED_HOSTS: Final[frozenset[str]] = frozenset(
    {
        *_YOUTUBE_HOSTS,
        *_INSTAGRAM_HOSTS,
        *_FACEBOOK_HOSTS,
        *_FACEBOOK_SHORT_HOSTS,
        *_TIKTOK_HOSTS,
        *_TIKTOK_SHORT_HOSTS,
        *_X_HOSTS,
        *_X_SHORT_HOSTS,
    }
)
_YOUTUBE_PATH_FORMS: Final[frozenset[str]] = frozenset({"shorts", "embed", "live"})
_VIDEO_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]{11}")
_SAFE_SEGMENT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]+")
_FACEBOOK_OWNER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._-]+")
_FACEBOOK_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:[0-9]+|pfbid[A-Za-z0-9_-]+)"
)
_TIKTOK_USER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._-]+")
_TIKTOK_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9]+")
_X_USER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]+")
_X_STATUS_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9]+")

_UNAVAILABLE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "private video",
        "this video is private",
        "video is private",
        "private post",
        "this post is private",
        "account is private",
        "private account",
        "this account is private",
        "video unavailable",
        "video is unavailable",
        "this video is unavailable",
        "video not found",
        "post not found",
        "video is not available",
        "this video is not available",
        "video has been removed",
        "this video has been removed",
        "video was removed",
        "video has been deleted",
        "this video has been deleted",
        "post is unavailable",
        "content is unavailable",
        "content unavailable",
        "content is not available",
        "content isn't available",
        "requires you to log in",
        "account has been terminated",
        "not available in your country",
        "not available in your region",
        "blocked in your country",
        "geo-restricted",
        "geo restricted",
        "geo-blocked",
        "geoblocked",
        "country restriction",
        "age-restricted",
        "age restricted",
        "confirm your age",
        "login required",
        "sign in required",
        "log in to confirm",
        "sign in to confirm",
        "requires login",
        "requires you to sign in",
    }
)
_COLLECTION_TYPES: Final[frozenset[str]] = frozenset(
    {"playlist", "multi_video", "url", "url_transparent"}
)
_METADATA_YTDLP_OPTIONS: Final[dict[str, object]] = {
    "allowed_extractors": [r"^(?:tiktok|vm\.tiktok)$"],
    "ignoreconfig": True,
    "no_warnings": True,
    "noplaylist": True,
    "quiet": True,
}

_AUDIO_OUTPUT_TEMPLATE: Final[str] = "audio.%(ext)s"
_INSPECTION_TEMP_PREFIX: Final[str] = "textify-inspection-"
_AUDIO_FALLBACK_YTDLP_OPTIONS: Final[dict[str, object]] = {
    **_METADATA_YTDLP_OPTIONS,
    "format": "bestaudio/best",
}


def _log_inspection_error(event: str, reason: str) -> None:
    """Log a safe, structured reason for an inspection failure."""
    logger.debug(
        event,
        extra={
            "stage": "transcription",
            "reason": reason,
        },
    )


def _cleanup_request_directory(request_directory: Path) -> None:
    """Remove private media without masking the operation's primary result."""
    if not request_directory.exists():
        return
    try:
        shutil.rmtree(request_directory)
    except OSError:
        logger.error(
            "temporary media cleanup failed",
            extra={
                "stage": "transcription",
                "reason": "temporary_media_cleanup_failed",
            },
        )


def _metadata_provider_error(reason: str) -> MetadataRetrievalFailedError:
    """Log a safe reason before returning the stable provider error."""
    _log_inspection_error("metadata provider error", reason)
    return MetadataRetrievalFailedError()


def _invalid_media_duration_error(reason: str) -> InvalidMediaDurationError:
    """Log a safe reason before returning the stable duration error."""
    _log_inspection_error("invalid media duration", reason)
    return InvalidMediaDurationError()


def _unsupported_media_error(reason: str) -> UnsupportedMediaError:
    """Log a safe reason before returning the stable media error."""
    _log_inspection_error("unsupported media", reason)
    return UnsupportedMediaError()


def _invalid_source_error(reason: str) -> InvalidUrlError:
    """Log a safe reason before returning the stable invalid URL error."""
    _log_inspection_error("invalid URL error", reason)
    return InvalidUrlError()


def _unsupported_platform_error(reason: str) -> UnsupportedPlatformError:
    """Log a safe reason before returning the stable platform error."""
    _log_inspection_error("unsupported platform error", reason)
    return UnsupportedPlatformError()


class _MetadataDurationLimitExceeded(Exception):
    """Indicate that inspection found a source exceeding the duration limit."""


class _DurationLimitDownloadCancelled(DownloadCancelled):  # type: ignore[misc]
    """Stop yt-dlp after the inspection duration filter rejects a source."""


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    """Contain an inspection-stage audio file and its private directory."""

    path: Path
    request_directory: Path

    def cleanup(self) -> None:
        """Delete the private request directory that owns the audio file."""
        _cleanup_request_directory(self.request_directory)


@dataclass(frozen=True, slots=True)
class ExtractedMetadata:
    """Contain yt-dlp metadata and audio prepared during metadata inspection."""

    metadata: Mapping[str, object]
    prepared_audio: PreparedAudio | None = None


class MetadataExtractor(Protocol):
    """Retrieve raw metadata and optional prepared audio for one provider URL."""

    def extract(
        self,
        canonical_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> ExtractedMetadata:
        """Return provider metadata and optional inspection-stage audio.

        Args:
            canonical_url: Minimal provider URL produced by URL inspection.
            deadline: Monotonic absolute deadline for metadata retrieval.
            cancellation_event: Signal set when request work must stop.

        Returns:
            ExtractedMetadata: Raw provider metadata and prepared audio when needed.
        """
        ...


class YtDlpMetadataExtractor:
    """Retrieve Source metadata and recover missing durations through audio download."""

    def __init__(self, max_duration_seconds: int, temp_media_dir: Path) -> None:
        """Initialize fallback media storage and the permitted video duration.

        Args:
            max_duration_seconds: Maximum duration accepted by the extraction pipeline.
            temp_media_dir: Root directory for private inspection media directories.
        """
        self._max_duration_seconds = max_duration_seconds
        self._temp_media_dir = temp_media_dir

    def extract(
        self,
        canonical_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> ExtractedMetadata:
        """Extract metadata and download audio when the provider omits duration.

        Args:
            canonical_url: Minimal provider URL produced by URL inspection.
            deadline: Monotonic absolute deadline for metadata retrieval.
            cancellation_event: Signal set when request work must stop.

        Returns:
            ExtractedMetadata: Raw yt-dlp metadata and optional prepared audio.

        Raises:
            MetadataRetrievalFailedError: If yt-dlp returns a non-mapping value.
            _MetadataDurationLimitExceeded: If fallback inspection exceeds the limit.
        """
        with yt_dlp.YoutubeDL(_METADATA_YTDLP_OPTIONS) as youtube_dl:
            raw_metadata = extract_info_with_retries(
                youtube_dl.extract_info,
                canonical_url,
                download=False,
                deadline=deadline,
                cancellation_event=cancellation_event,
            )

        if not isinstance(raw_metadata, Mapping):
            raise _metadata_provider_error("non_mapping_result")
        metadata = cast(Mapping[str, object], raw_metadata)
        raise_if_cancelled_or_expired(
            deadline=deadline,
            cancellation_event=cancellation_event,
        )
        if metadata.get("duration") is not None:
            return ExtractedMetadata(metadata)
        return self._download_audio_for_duration(
            canonical_url,
            deadline=deadline,
            cancellation_event=cancellation_event,
        )

    def _download_audio_for_duration(
        self,
        canonical_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> ExtractedMetadata:
        raise_if_cancelled_or_expired(
            deadline=deadline,
            cancellation_event=cancellation_event,
        )
        self._temp_media_dir.mkdir(parents=True, exist_ok=True)
        request_directory = Path(
            tempfile.mkdtemp(
                prefix=_INSPECTION_TEMP_PREFIX,
                dir=str(self._temp_media_dir),
            )
        ).resolve()

        def _duration_filter(
            info: dict[str, Any],
            *,
            incomplete: bool = False,
        ) -> str | None:
            duration = info.get("duration")
            if duration is not None and duration > self._max_duration_seconds:
                raise _DurationLimitDownloadCancelled(
                    f"Video duration ({duration}s) exceeds the {self._max_duration_seconds}s limit"
                )
            return None

        completed = False
        try:
            options = {
                **_AUDIO_FALLBACK_YTDLP_OPTIONS,
                "outtmpl": str(request_directory / _AUDIO_OUTPUT_TEMPLATE),
                "match_filter": _duration_filter,
                "progress_hooks": [
                    build_yt_dlp_progress_hook(
                        deadline=deadline,
                        cancellation_event=cancellation_event,
                    )
                ],
            }
            with yt_dlp.YoutubeDL(options) as youtube_dl:
                raw_metadata = extract_info_with_retries(
                    youtube_dl.extract_info,
                    canonical_url,
                    download=True,
                    deadline=deadline,
                    cancellation_event=cancellation_event,
                )
                if not isinstance(raw_metadata, Mapping) or "entries" in raw_metadata:
                    raise _metadata_provider_error("invalid_duration_download_result")
                prepared_path = youtube_dl.prepare_filename(raw_metadata)

            if not isinstance(prepared_path, (str, Path)):
                raise _metadata_provider_error("invalid_duration_download_path")
            audio_path = Path(prepared_path).resolve()
            if audio_path.parent != request_directory or not audio_path.is_file():
                raise _metadata_provider_error("duration_audio_file_missing")
            raise_if_cancelled_or_expired(
                deadline=deadline,
                cancellation_event=cancellation_event,
            )
            audio_duration = _probe_audio_duration(audio_path)
            raise_if_cancelled_or_expired(
                deadline=deadline,
                cancellation_event=cancellation_event,
            )
            if audio_duration > self._max_duration_seconds:
                raise _MetadataDurationLimitExceeded
            metadata = dict(cast(Mapping[str, object], raw_metadata))
            metadata["duration"] = audio_duration
            completed = True
            return ExtractedMetadata(
                metadata=metadata,
                prepared_audio=PreparedAudio(audio_path, request_directory),
            )
        except _DurationLimitDownloadCancelled as exc:
            raise _MetadataDurationLimitExceeded from exc
        finally:
            if not completed:
                _cleanup_request_directory(request_directory)


def _probe_audio_duration(audio_path: Path) -> float:
    """Return the finite positive duration of one downloaded audio file."""
    try:
        with av.open(str(audio_path)) as container:
            duration = container.duration
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise _metadata_provider_error("audio_duration_probe_failed") from exc
    if duration is None:
        raise _invalid_media_duration_error("audio_duration_unavailable")
    duration_seconds = duration / av.time_base
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise _invalid_media_duration_error("invalid_audio_duration")
    return duration_seconds


@dataclass(frozen=True, slots=True)
class SubmittedSource:
    """Carry validated provider input facts across Source inspection."""

    platform: Platform
    provider_url: str
    youtube_video_id: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizedMetadata:
    """Carry validated provider metadata before Source construction."""

    video_id: str
    canonical_url: str
    title: str
    description: str
    channel: str
    duration_seconds: int


def classify_submitted_url(submitted_url: str) -> SubmittedSource:
    """Validate a submitted URL and derive its minimal provider request.

    Args:
        submitted_url: URL supplied by the API caller.

    Returns:
        SubmittedSource: Platform and provider URL facts safe for provider access.

    Raises:
        InvalidUrlError: If the URL is malformed or has an unsupported shape.
        UnsupportedPlatformError: If the URL uses another host.
    """
    parsed_url, host, query_pairs = _parse_url(submitted_url)
    if host not in _SUPPORTED_HOSTS:
        if _looks_like_supported_host_trick(host):
            raise _invalid_source_error("deceptive_supported_host")
        raise _unsupported_platform_error("unsupported_host")
    path_segments = _path_segments(parsed_url.path)
    if host in _YOUTUBE_HOSTS:
        video_id = _youtube_video_id(host, path_segments, query_pairs)
        return SubmittedSource(
            platform=Platform.YOUTUBE,
            provider_url=f"https://www.youtube.com/watch?v={video_id}",
            youtube_video_id=video_id,
        )

    if host in _INSTAGRAM_HOSTS:
        _reject_video_query(query_pairs)
        _instagram_shortcode(path_segments)
        return SubmittedSource(Platform.INSTAGRAM, _minimal_url(parsed_url))

    if host in _FACEBOOK_HOSTS or host in _FACEBOOK_SHORT_HOSTS:
        _facebook_identity(host, path_segments, query_pairs)
        return SubmittedSource(Platform.FACEBOOK, _minimal_url(parsed_url))

    if host in _TIKTOK_HOSTS or host in _TIKTOK_SHORT_HOSTS:
        _reject_video_query(query_pairs)
        _tiktok_identity(host, path_segments)
        return SubmittedSource(Platform.TIKTOK, _minimal_url(parsed_url))

    if host in _X_HOSTS or host in _X_SHORT_HOSTS:
        _reject_video_query(query_pairs)
        _x_identity(host, path_segments)
        return SubmittedSource(Platform.X, _minimal_url(parsed_url))

    if _looks_like_supported_host_trick(host):
        raise _invalid_source_error("deceptive_supported_host")
    raise _unsupported_platform_error("unsupported_host")


def normalize_processed_metadata(
    metadata: Mapping[str, object],
    submitted: SubmittedSource,
) -> NormalizedMetadata:
    """Validate processed metadata and normalize it into Source fields.

    Args:
        metadata: One processed yt-dlp metadata mapping.
        submitted: Validated platform and URL facts for the request.

    Returns:
        NormalizedMetadata: Provider-authoritative identity and normalized fields.

    Raises:
        InvalidUrlError: If the provider result is a valid but unsupported shape.
        MetadataRetrievalFailedError: If the provider result is malformed.
    """
    if not isinstance(metadata, Mapping):
        raise _metadata_provider_error("non_mapping_metadata")
    _reject_collections_and_live(metadata, submitted.platform)

    if submitted.platform is Platform.YOUTUBE:
        video_id = submitted.youtube_video_id
        if video_id is None:
            raise _metadata_provider_error("missing_submitted_video_id")
        provider_id = metadata.get("id", _MISSING)
        if provider_id is not _MISSING and provider_id != video_id:
            raise _metadata_provider_error("provider_video_id_mismatch")
        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        expected_extractor_keys: frozenset[str] | None = None
    else:
        expected_extractor_keys = _EXPECTED_EXTRACTOR_KEYS[submitted.platform]
        extractor_key = metadata.get("extractor_key", _MISSING)
        if extractor_key is _MISSING:
            raise _metadata_provider_error("missing_extractor_key")
        if not isinstance(extractor_key, str) or not extractor_key.strip():
            raise _metadata_provider_error("invalid_extractor_key")
        if extractor_key not in expected_extractor_keys:
            raise _invalid_source_error("unexpected_extractor")
        video_id = _social_video_id(metadata)
        canonical_url = _canonical_social_url(metadata, submitted.platform)
        _validate_social_formats(metadata)

    title = _normalize_title(metadata, submitted.platform)
    description = _optional_text(metadata, "description") or ""
    channel_value = _optional_text(metadata, "channel")
    uploader_value = _optional_text(metadata, "uploader")
    channel = (
        channel_value
        if channel_value is not None and channel_value.strip()
        else uploader_value
        if uploader_value is not None and uploader_value.strip()
        else ""
    )
    duration_seconds = _normalize_duration(metadata)

    return NormalizedMetadata(
        video_id=video_id,
        canonical_url=canonical_url,
        title=title,
        description=description,
        channel=channel,
        duration_seconds=duration_seconds,
    )


_EXPECTED_EXTRACTOR_KEYS: Final[dict[Platform, frozenset[str]]] = {
    Platform.INSTAGRAM: frozenset({"Instagram"}),
    Platform.FACEBOOK: frozenset({"Facebook", "FacebookReel"}),
    Platform.TIKTOK: frozenset({"TikTok"}),
    Platform.X: frozenset({"Twitter"}),
}


def _parse_url(
    submitted_url: str,
) -> tuple[SplitResult, str, list[tuple[str, str]]]:
    if not submitted_url or _contains_control_character(submitted_url):
        raise _invalid_source_error("empty_or_control_character_url")
    if _contains_malformed_percent_encoding(submitted_url):
        raise _invalid_source_error("malformed_percent_encoding")

    try:
        parsed_url = urlsplit(submitted_url)
        hostname = parsed_url.hostname
        username = parsed_url.username
        password = parsed_url.password
        port = parsed_url.port
    except ValueError as exc:
        raise _invalid_source_error("malformed_url") from exc

    if parsed_url.scheme.casefold() != "https" or hostname is None:
        raise _invalid_source_error("non_https_or_missing_host")
    if username is not None or password is not None:
        raise _invalid_source_error("url_credentials")
    if port is not None or _authority_has_port(parsed_url.netloc):
        raise _invalid_source_error("url_port")
    if "\\" in parsed_url.netloc or not parsed_url.netloc:
        raise _invalid_source_error("invalid_url_authority")

    try:
        query_pairs = parse_qsl(parsed_url.query, keep_blank_values=True)
    except ValueError as exc:
        raise _invalid_source_error("malformed_query") from exc

    host = hostname.casefold()
    if "%" in parsed_url.path:
        raise _invalid_source_error("encoded_path")
    return parsed_url, host, query_pairs


def _reject_video_query(query_pairs: list[tuple[str, str]]) -> None:
    if any(key.casefold() == "v" for key, _ in query_pairs):
        raise _invalid_source_error("video_query_not_supported")


def _minimal_url(parsed_url: SplitResult) -> str:
    return urlunsplit(
        (
            "https",
            parsed_url.netloc.casefold(),
            parsed_url.path.rstrip("/"),
            _minimal_query(parsed_url.query),
            "",
        )
    )


def _minimal_query(query: str) -> str:
    pairs = parse_qsl(query, keep_blank_values=True)
    identity_pairs = [(key, value) for key, value in pairs if key.casefold() == "v"]
    if len(identity_pairs) > 1:
        raise _invalid_source_error("duplicate_video_query_parameter")
    return f"v={identity_pairs[0][1]}" if identity_pairs else ""


def _path_segments(path: str) -> list[str]:
    if not path or not path.startswith("/"):
        raise _invalid_source_error("missing_path")
    normalized_path = path.removesuffix("/")
    if not normalized_path or normalized_path == "/":
        raise _invalid_source_error("empty_path")
    segments = normalized_path[1:].split("/")
    if any(not segment for segment in segments):
        raise _invalid_source_error("empty_path_segment")
    return segments


def _youtube_video_id(
    host: str,
    path_segments: list[str],
    query_pairs: list[tuple[str, str]],
) -> str:
    video_query_values = [value for key, value in query_pairs if key == "v"]
    if len(video_query_values) > 1:
        raise _invalid_source_error("duplicate_video_query_parameter")

    if host == "youtu.be":
        if len(path_segments) != 1:
            raise _invalid_source_error("invalid_short_video_path")
        video_id = path_segments[0]
        if video_query_values and video_query_values[0] not in {"", video_id}:
            raise _invalid_source_error("conflicting_video_query")
    elif path_segments[0] == "watch":
        if len(path_segments) != 1 or len(video_query_values) != 1:
            raise _invalid_source_error("invalid_watch_path")
        video_id = video_query_values[0]
    elif path_segments[0] in _YOUTUBE_PATH_FORMS:
        if len(path_segments) != 2:
            raise _invalid_source_error("invalid_video_path")
        video_id = path_segments[1]
        if video_query_values and video_query_values[0] not in {"", video_id}:
            raise _invalid_source_error("conflicting_video_query")
    else:
        raise _invalid_source_error("unsupported_video_path")

    if _VIDEO_ID_PATTERN.fullmatch(video_id) is None:
        raise _invalid_source_error("invalid_video_id")
    return video_id


def _instagram_shortcode(path_segments: list[str]) -> str:
    if len(path_segments) != 2 or path_segments[0] not in {
        "p",
        "tv",
        "reel",
        "reels",
    }:
        raise _invalid_source_error("invalid_instagram_path")
    shortcode = path_segments[1]
    if _SAFE_SEGMENT_PATTERN.fullmatch(shortcode) is None:
        raise _invalid_source_error("invalid_instagram_shortcode")
    return shortcode


def _facebook_identity(
    host: str,
    path_segments: list[str],
    query_pairs: list[tuple[str, str]],
) -> str:
    query_values = [value for key, value in query_pairs if key.casefold() == "v"]
    if len(query_values) > 1:
        raise _invalid_source_error("duplicate_facebook_query_parameter")

    if host in _FACEBOOK_SHORT_HOSTS:
        if len(path_segments) != 1 or query_values:
            raise _invalid_source_error("invalid_facebook_short_path")
        return _safe_segment(path_segments[0])

    path = tuple(path_segments)
    if path == ("watch",):
        if len(query_values) != 1:
            raise _invalid_source_error("missing_facebook_watch_id")
        return _facebook_id(query_values[0])
    if path == ("video.php",) or path == ("video", "video.php"):
        if len(query_values) != 1:
            raise _invalid_source_error("missing_facebook_video_id")
        return _facebook_id(query_values[0])
    if len(path) == 2 and path[0] == "reel":
        if query_values:
            raise _invalid_source_error("facebook_reel_query")
        return _facebook_id(path[1])
    if len(path) == 3 and path[0] == "share":
        if query_values:
            raise _invalid_source_error("facebook_share_query")
        return path[2]  # share ID, not facebook ID
    if len(path) in {3, 4} and path[1] in {"videos", "posts"}:
        if path[1] == "posts" and len(path) != 3:
            raise _invalid_source_error("invalid_facebook_posts_path")
        if query_values:
            raise _invalid_source_error("facebook_query_not_supported")
        if _FACEBOOK_OWNER_PATTERN.fullmatch(path[0]) is None:
            raise _invalid_source_error("invalid_facebook_owner")
        if len(path) == 4 and _SAFE_SEGMENT_PATTERN.fullmatch(path[2]) is None:
            raise _invalid_source_error("invalid_facebook_path_segment")
        return _facebook_id(path[-1])
    raise _invalid_source_error("unsupported_facebook_path")


def _facebook_id(value: str) -> str:
    if _FACEBOOK_ID_PATTERN.fullmatch(value) is None:
        raise _invalid_source_error("invalid_facebook_id")
    return value


def _tiktok_identity(host: str, path_segments: list[str]) -> str:
    if host in _TIKTOK_SHORT_HOSTS:
        if len(path_segments) != 1:
            raise _invalid_source_error("invalid_tiktok_short_path")
        return _safe_segment(path_segments[0])
    if path_segments[0] == "t" and len(path_segments) == 2:
        return _safe_segment(path_segments[1])
    if path_segments[0] == "embed" and len(path_segments) == 2:
        if _TIKTOK_ID_PATTERN.fullmatch(path_segments[1]) is None:
            raise _invalid_source_error("invalid_tiktok_embed_id")
        return path_segments[1]
    if (
        len(path_segments) == 3
        and path_segments[0].startswith("@")
        and path_segments[1] == "video"
    ):
        user = path_segments[0][1:]
        video_id = path_segments[2]
        if _TIKTOK_USER_PATTERN.fullmatch(user) is None:
            raise _invalid_source_error("invalid_tiktok_user")
        if _TIKTOK_ID_PATTERN.fullmatch(video_id) is None:
            raise _invalid_source_error("invalid_tiktok_video_id")
        return video_id
    raise _invalid_source_error("unsupported_tiktok_path")


def _x_identity(host: str, path_segments: list[str]) -> str:
    if host in _X_SHORT_HOSTS:
        if len(path_segments) != 1:
            raise _invalid_source_error("invalid_x_short_path")
        return _safe_segment(path_segments[0])

    base_segments = path_segments
    if len(path_segments) >= 2 and path_segments[-2] == "video":
        if len(path_segments) < 3 or not path_segments[-1].isdigit():
            raise _invalid_source_error("invalid_x_video_path")
        if int(path_segments[-1]) <= 0:
            raise _invalid_source_error("invalid_x_video_id")
        base_segments = path_segments[:-2]

    if len(base_segments) == 3 and base_segments[1] == "status":
        if _X_USER_PATTERN.fullmatch(base_segments[0]) is None:
            raise _invalid_source_error("invalid_x_user")
        return _x_status_id(base_segments[2])
    if len(base_segments) == 4 and base_segments[:3] == ["i", "web", "status"]:
        return _x_status_id(base_segments[3])
    if len(base_segments) == 2 and base_segments[0] == "statuses":
        return _x_status_id(base_segments[1])
    raise _invalid_source_error("unsupported_x_path")


def _x_status_id(value: str) -> str:
    if _X_STATUS_PATTERN.fullmatch(value) is None:
        raise _invalid_source_error("invalid_x_status_id")
    return value


def _safe_segment(value: str) -> str:
    if _SAFE_SEGMENT_PATTERN.fullmatch(value) is None:
        raise _invalid_source_error("invalid_safe_segment")
    return value


def _reject_collections_and_live(
    metadata: Mapping[str, object],
    platform: Platform,
) -> None:
    if "entries" in metadata:
        raise _invalid_source_error("collection_entries")

    content_type = metadata.get("_type")
    if content_type is not None:
        if not isinstance(content_type, str):
            raise _metadata_provider_error("invalid_content_type")
        if content_type in _COLLECTION_TYPES:
            raise _invalid_source_error("collection_type")

    if platform is Platform.YOUTUBE:
        return

    is_live = metadata.get("is_live", _MISSING)
    if is_live is not _MISSING and not isinstance(is_live, bool):
        raise _metadata_provider_error("invalid_is_live")
    if is_live is True:
        raise _invalid_source_error("live_video")

    is_upcoming = metadata.get("is_upcoming", _MISSING)
    if is_upcoming is not _MISSING and not isinstance(is_upcoming, bool):
        raise _metadata_provider_error("invalid_is_upcoming")
    if is_upcoming is True:
        raise _invalid_source_error("upcoming_video")

    live_status = metadata.get("live_status", _MISSING)
    if live_status is not _MISSING:
        if not isinstance(live_status, str):
            raise _metadata_provider_error("invalid_live_status")
        if live_status != "not_live":
            raise _invalid_source_error("unsupported_live_status")


def _social_video_id(metadata: Mapping[str, object]) -> str:
    video_id = metadata.get("id", _MISSING)
    if video_id is _MISSING:
        raise _metadata_provider_error("missing_video_id")
    if (
        not isinstance(video_id, str)
        or not video_id
        or _contains_control_character(video_id)
    ):
        raise _metadata_provider_error("invalid_video_id")
    return video_id


def _canonical_social_url(metadata: Mapping[str, object], platform: Platform) -> str:
    webpage_url = metadata.get("webpage_url", _MISSING)
    if webpage_url is _MISSING:
        raise _metadata_provider_error("missing_webpage_url")
    if (
        not isinstance(webpage_url, str)
        or not webpage_url
        or _contains_control_character(webpage_url)
        or _contains_malformed_percent_encoding(webpage_url)
    ):
        raise _metadata_provider_error("invalid_webpage_url")

    try:
        parsed_url = urlsplit(webpage_url)
        hostname = parsed_url.hostname
        username = parsed_url.username
        password = parsed_url.password
        port = parsed_url.port
    except ValueError as exc:
        raise _metadata_provider_error("malformed_webpage_url") from exc
    if (
        parsed_url.scheme.casefold() != "https"
        or hostname is None
        or username is not None
        or password is not None
        or port is not None
        or _authority_has_port(parsed_url.netloc)
        or "\\" in parsed_url.netloc
        or "%" in parsed_url.path
    ):
        raise _metadata_provider_error("unsafe_webpage_url")

    host = hostname.casefold()
    expected_hosts = _platform_hosts(platform)
    if host in _short_hosts():
        is_expected_short_host = (
            (platform is Platform.FACEBOOK and host in _FACEBOOK_SHORT_HOSTS)
            or (platform is Platform.TIKTOK and host in _TIKTOK_SHORT_HOSTS)
            or (platform is Platform.X and host in _X_SHORT_HOSTS)
        )
        if is_expected_short_host:
            raise _metadata_provider_error("unexpected_short_webpage_url")

    if host not in _SUPPORTED_HOSTS:
        raise _invalid_source_error("redirected_to_unsupported_host")
    if host not in expected_hosts:
        raise _invalid_source_error("cross_platform_redirect")

    try:
        classify_submitted_url(webpage_url)
    except UnsupportedPlatformError as exc:
        raise _invalid_source_error("cross_platform_redirect") from exc
    except InvalidUrlError:
        raise
    return urlunsplit(
        (parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.query, "")
    )


def _platform_hosts(platform: Platform) -> frozenset[str]:
    if platform is Platform.INSTAGRAM:
        return _INSTAGRAM_HOSTS
    if platform is Platform.FACEBOOK:
        return _FACEBOOK_HOSTS
    if platform is Platform.TIKTOK:
        return _TIKTOK_HOSTS
    if platform is Platform.X:
        return _X_HOSTS
    return _YOUTUBE_HOSTS


def _short_hosts() -> frozenset[str]:
    return frozenset({*_FACEBOOK_SHORT_HOSTS, *_TIKTOK_SHORT_HOSTS, *_X_SHORT_HOSTS})


def _normalize_title(metadata: Mapping[str, object], platform: Platform) -> str:
    title = metadata.get("title", _MISSING)
    if title is _MISSING or title is None:
        if platform is Platform.YOUTUBE:
            raise _metadata_provider_error("missing_youtube_title")
        return ""
    if not isinstance(title, str):
        raise _metadata_provider_error("invalid_title")
    if not title.strip() and platform is Platform.YOUTUBE:
        raise _metadata_provider_error("empty_youtube_title")
    return title if title.strip() else ""


def _optional_text(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _metadata_provider_error(f"invalid_optional_metadata={key}")
    return value


def _normalize_duration(metadata: Mapping[str, object]) -> int:
    duration_value = metadata.get("duration", _MISSING)
    if duration_value is _MISSING:
        raise _invalid_media_duration_error("missing_duration")
    if isinstance(duration_value, bool) or not isinstance(
        duration_value, (Real, Decimal)
    ):
        raise _invalid_media_duration_error("invalid_duration_type")
    try:
        if duration_value <= 0:
            raise _invalid_media_duration_error("nonpositive_duration")
        if not isfinite(duration_value):
            raise _invalid_media_duration_error("non_finite_duration")
        return cast(int, ceil(duration_value))
    except (InvalidOperation, OverflowError, TypeError, ValueError) as exc:
        raise _invalid_media_duration_error("duration_normalization_failed") from exc


def _validate_social_formats(metadata: Mapping[str, object]) -> None:
    formats = metadata.get("formats", _MISSING)
    if formats is _MISSING:
        raise _metadata_provider_error("missing_formats")
    if not isinstance(formats, Sequence) or isinstance(
        formats, (str, bytes, bytearray)
    ):
        raise _metadata_provider_error("invalid_formats")
    if not formats:
        raise _unsupported_media_error("empty_formats")

    has_video = False
    for item in formats:
        if not isinstance(item, Mapping):
            raise _metadata_provider_error("invalid_format_item")
        video_codec = item.get("vcodec")
        if video_codec is not None and not isinstance(video_codec, str):
            raise _metadata_provider_error("invalid_video_codec")
        if video_codec and video_codec.casefold() != "none":
            has_video = True
    if not has_video:
        raise _unsupported_media_error("no_video_format")


def _looks_like_supported_host_trick(host: str) -> bool:
    return any(
        host.startswith(f"{supported}.") or host.endswith(f".{supported}")
        for supported in _SUPPORTED_HOSTS
    )


def _contains_control_character(value: str) -> bool:
    return any(character.isspace() or ord(character) < 0x20 for character in value)


def _contains_malformed_percent_encoding(value: str) -> bool:
    for index, character in enumerate(value):
        if character != "%":
            continue
        if index + 2 >= len(value):
            return True
        if not all(
            digit in "0123456789abcdefABCDEF" for digit in value[index + 1 : index + 3]
        ):
            return True
    return False


def _authority_has_port(netloc: str) -> bool:
    authority = netloc.rsplit("@", maxsplit=1)[-1]
    if authority.startswith("["):
        closing_bracket = authority.find("]")
        if closing_bracket == -1:
            return True
        return authority[closing_bracket + 1 :].startswith(":")
    return ":" in authority


def _is_timeout_exception(error: BaseException) -> bool:
    """Return whether a provider exception contains a typed timeout."""
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if isinstance(candidate, (Timeout, TimeoutError, socket.timeout)):
            return True
        if isinstance(candidate, DownloadError):
            exc_info = getattr(candidate, "exc_info", None)
            if isinstance(exc_info, tuple) and len(exc_info) > 1:
                nested = exc_info[1]
                if isinstance(nested, BaseException):
                    pending.append(nested)
        for attribute in ("__cause__", "__context__", "reason"):
            nested = getattr(candidate, attribute, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _is_unavailable_error(message: str) -> bool:
    normalized_message = message.casefold()
    return any(marker in normalized_message for marker in _UNAVAILABLE_MARKERS)


def safe_submitted_url(submitted_url: str) -> str:
    """Redact sensitive query values and fragments before structured logging.

    Args:
        submitted_url: URL submitted by the API caller.

    Returns:
        str: Submitted URL with sensitive query values and fragments removed.
    """
    try:
        parsed_url = urlsplit(submitted_url)
        query_pairs = parse_qsl(parsed_url.query, keep_blank_values=True)
    except ValueError:
        return _REDACTED_LOG_VALUE

    has_sensitive_query = any(_is_sensitive_query_key(key) for key, _ in query_pairs)
    if not parsed_url.fragment and not has_sensitive_query:
        return submitted_url

    safe_pairs = [
        (key, _REDACTED_LOG_VALUE if _is_sensitive_query_key(key) else value)
        for key, value in query_pairs
    ]
    return urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            urlencode(safe_pairs),
            "",
        )
    )


def _is_sensitive_query_key(key: str) -> bool:
    normalized_key = key.casefold().replace("-", "_")
    return normalized_key in _SENSITIVE_QUERY_PARTS or any(
        part in normalized_key for part in ("api_key", "token", "secret", "password")
    )
