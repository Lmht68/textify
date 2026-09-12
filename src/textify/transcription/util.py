"""Retry the yt-dlp operation that performs external data retrieval."""

import threading
import time
from collections.abc import Callable, Mapping
from typing import Final, cast

import yt_dlp
from yt_dlp.extractor.facebook import FacebookIE
from yt_dlp.utils import DownloadCancelled

_MAX_YTDLP_ATTEMPTS: Final[int] = 10
_ALLOWED_EXTRACTORS: Final[list[str]] = [
    r"^(?:facebook|facebook:reel|instagram|tiktok|twitter|twitter:shortener|vm\.tiktok|youtube)$"
]


class _TextifyFacebookIE(FacebookIE):  # type: ignore[misc]
    """Extend Facebook extraction to the accepted short and share URL families."""

    _VALID_URL = (
        FacebookIE._VALID_URL,
        r"https://fb\.watch/(?P<id>[A-Za-z0-9_-]+)$",
        (
            r"https://(?:facebook\.com|www\.facebook\.com|m\.facebook\.com)"
            r"/share/(?:v/|[^/?#]+/[^/?#]+/)(?P<id>[A-Za-z0-9_-]+)$"
        ),
    )

    @classmethod
    def ie_key(cls) -> str:
        """Return Facebook's stable yt-dlp extractor key."""
        return cast(str, FacebookIE.ie_key())


def create_yt_dlp(
    operation_options: Mapping[str, object] | None = None,
) -> yt_dlp.YoutubeDL:
    """Create a yt-dlp instance with Textify's non-overridable extractor policy.

    Args:
        operation_options: Optional output template, match filter, and progress hooks.

    Returns:
        yt-dlp instance limited to Textify's approved platform extraction paths.
    """
    options: dict[str, object] = {}
    if operation_options is not None:
        for option_name in ("outtmpl", "match_filter", "progress_hooks"):
            if option_name not in operation_options:
                continue
            option = operation_options[option_name]
            if option_name == "progress_hooks" and isinstance(option, list):
                options[option_name] = list(option)
            else:
                options[option_name] = option

    options.update(
        {
            "allowed_extractors": list(_ALLOWED_EXTRACTORS),
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
    )
    youtube_dl = yt_dlp.YoutubeDL(options)  # pyright: ignore[reportArgumentType]
    youtube_dl.add_info_extractor(_TextifyFacebookIE())
    return youtube_dl


class MediaByteLimitExceeded(DownloadCancelled):  # type: ignore[misc]
    """Stop yt-dlp when a media transfer exceeds the configured byte limit."""

    def __init__(self) -> None:
        """Initialize the stable safe policy-rejection message."""
        super().__init__("Downloaded media exceeds the configured byte limit.")


def raise_if_cancelled_or_expired(
    *,
    deadline: float,
    cancellation_event: threading.Event,
) -> None:
    """Stop a yt-dlp operation after cancellation or its absolute deadline.

    Args:
        deadline: Monotonic absolute deadline for the complete operation.
        cancellation_event: Signal set when request work must stop cooperatively.

    Raises:
        DownloadCancelled: If the request has been cancelled.
        TimeoutError: If the operation has exceeded its deadline.
    """
    if cancellation_event.is_set():
        raise DownloadCancelled("yt-dlp operation cancelled")
    if time.monotonic() >= deadline:
        raise TimeoutError("yt-dlp operation deadline expired")


def build_yt_dlp_progress_hook(
    *,
    deadline: float,
    cancellation_event: threading.Event,
    max_media_bytes: int,
) -> Callable[[dict[str, object]], None]:
    """Build a transfer hook that enforces cancellation, deadline, and byte limits.

    Args:
        deadline: Monotonic absolute deadline for the provider operation.
        cancellation_event: Signal set when request work must stop.
        max_media_bytes: Inclusive limit for the selected downloaded media.

    Returns:
        yt-dlp progress callback that interrupts oversized transfers.
    """

    def _check_progress(status: dict[str, object]) -> None:
        raise_if_cancelled_or_expired(
            deadline=deadline,
            cancellation_event=cancellation_event,
        )
        if status.get("status") not in {"downloading", "finished"}:
            return
        for field_name in ("downloaded_bytes", "total_bytes"):
            byte_count = status.get(field_name)
            if (
                isinstance(byte_count, int)
                and not isinstance(byte_count, bool)
                and byte_count > max_media_bytes
            ):
                raise MediaByteLimitExceeded

    return _check_progress


def extract_info_with_retries[T](
    extract_info: Callable[..., T],
    source_url: str,
    *,
    download: bool,
    deadline: float,
    cancellation_event: threading.Event,
) -> T:
    """Call yt-dlp's data-retrieval API at most ten times.

    Args:
        extract_info: Bound ``yt_dlp.YoutubeDL.extract_info`` method.
        deadline: Monotonic absolute deadline for all retry attempts.
        cancellation_event: Signal set when request work must stop cooperatively.

    Returns:
        The result of the first successful yt-dlp request.

    Raises:
        DownloadCancelled: If yt-dlp cancels the operation or request is cancelled.
        TimeoutError: If the complete operation exceeds its deadline.
        Exception: Final retryable yt-dlp exception after ten attempts.
    """
    for attempt in range(_MAX_YTDLP_ATTEMPTS):
        raise_if_cancelled_or_expired(
            deadline=deadline,
            cancellation_event=cancellation_event,
        )
        try:
            result = extract_info(source_url, download=download)
        except DownloadCancelled:
            raise
        except Exception:
            if attempt == _MAX_YTDLP_ATTEMPTS - 1:
                raise_if_cancelled_or_expired(
                    deadline=deadline,
                    cancellation_event=cancellation_event,
                )
                raise
        else:
            raise_if_cancelled_or_expired(
                deadline=deadline,
                cancellation_event=cancellation_event,
            )
            return result

    raise AssertionError("yt-dlp retry loop completed without a result")
