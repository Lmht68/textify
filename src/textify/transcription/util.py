"""Retry the yt-dlp operation that performs external data retrieval."""

import threading
import time
from collections.abc import Callable
from typing import Final

from yt_dlp.utils import DownloadCancelled

_MAX_YTDLP_ATTEMPTS: Final[int] = 10


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
) -> Callable[[dict[str, object]], None]:
    """Build a transfer hook that enforces one operation's deadline.

    Args:
        deadline: Monotonic absolute deadline for the complete operation.
        cancellation_event: Signal set when request work must stop cooperatively.

    Returns:
        Progress callback suitable for yt-dlp request options.
    """

    def _check_progress(_status: dict[str, object]) -> None:
        raise_if_cancelled_or_expired(
            deadline=deadline,
            cancellation_event=cancellation_event,
        )

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
