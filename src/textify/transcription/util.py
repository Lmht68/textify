"""Retry the yt-dlp operation that performs external data retrieval."""

from collections.abc import Callable
from typing import Final

from yt_dlp.utils import DownloadCancelled

_MAX_YTDLP_ATTEMPTS: Final[int] = 10


def extract_info_with_retries[T](
    extract_info: Callable[..., T],
    source_url: str,
    *,
    download: bool,
) -> T:
    """Call yt-dlp's data-retrieval API at most ten times.

    Args:
        extract_info: Bound ``yt_dlp.YoutubeDL.extract_info`` method.
        source_url: Validated provider URL to retrieve.
        download: Whether yt-dlp should download media as part of retrieval.

    Returns:
        The result of the first successful yt-dlp request.

    Raises:
        DownloadCancelled: If yt-dlp cancels the operation.
        Exception: The final retryable exception raised by yt-dlp after ten attempts.
    """
    for attempt in range(_MAX_YTDLP_ATTEMPTS):
        try:
            return extract_info(source_url, download=download)
        except DownloadCancelled:
            raise
        except Exception:
            if attempt == _MAX_YTDLP_ATTEMPTS - 1:
                raise

    raise AssertionError("yt-dlp retry loop completed without a result")
