"""Configure Textify's safe structured request logging convention."""

import logging
import logging.config
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_ALLOWED_STRUCTURED_FIELDS = frozenset(
    {
        "request_id",
        "platform",
        "source_id",
        "stage",
        "method",
        "duration_seconds",
        "elapsed_seconds",
        "code",
    }
)


@dataclass(slots=True)
class _RequestLogContext:
    """Carry safe fields shared by one HTTP request's log records."""

    request_id: str
    platform: str | None = None
    source_id: str | None = None
    method: str | None = None
    duration_seconds: int | None = None
    code: str | None = None


_request_log_context: ContextVar[_RequestLogContext | None] = ContextVar(
    "textify_request_log_context",
    default=None,
)


class _KeyValueFormatter(logging.Formatter):
    """Append approved structured fields as sorted key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with its approved structured fields.

        Args:
            record: Log record to format.

        Returns:
            The rendered log message and approved structured fields.
        """
        fields = [
            f"{key}={record.__dict__[key]}"
            for key in sorted(_ALLOWED_STRUCTURED_FIELDS)
            if record.__dict__.get(key) is not None
        ]
        return " ".join((super().format(record), *fields))


@contextmanager
def request_log_context(request_id: str) -> Iterator[None]:
    """Bind one correlation ID to the current request context.

    Args:
        request_id: Server-generated request correlation identifier.

    Yields:
        None: The active request scope.
    """
    token = _request_log_context.set(_RequestLogContext(request_id=request_id))
    try:
        yield
    finally:
        _request_log_context.reset(token)


def bind_request_log_fields(
    *,
    platform: str | None = None,
    source_id: str | None = None,
    method: str | None = None,
    duration_seconds: int | None = None,
    code: str | None = None,
) -> None:
    """Bind authoritative safe fields to the current HTTP request.

    Args:
        platform: Normalized source platform.
        source_id: Normalized source identifier.
        method: Successful transcript acquisition method.
        duration_seconds: Validated source duration.
        code: Stable terminal error code.
    """
    context = _request_log_context.get()
    if context is None:
        return
    if platform is not None:
        context.platform = platform
    if source_id is not None:
        context.source_id = source_id
    if method is not None:
        context.method = method
    if duration_seconds is not None:
        context.duration_seconds = duration_seconds
    if code is not None:
        context.code = code


class _SafeContextFilter(logging.Filter):
    """Restrict structured output to Textify records and safe request context."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Enrich an application record without replacing explicit safe fields."""
        if record.name != "textify" and not record.name.startswith("textify."):
            return False
        context = _request_log_context.get()
        if context is None:
            return True
        for field_name, field_value in (
            ("request_id", context.request_id),
            ("platform", context.platform),
            ("source_id", context.source_id),
            ("method", context.method),
            ("duration_seconds", context.duration_seconds),
            ("code", context.code),
        ):
            if field_value is not None and field_name not in record.__dict__:
                record.__dict__[field_name] = field_value
        return True


def configure_logging(level: str) -> None:
    """Configure the Textify logger namespace and third-party log levels.

    Args:
        level: A standard logging level name such as ``INFO`` or ``DEBUG``.

    Returns:
        None: The logging configuration is applied globally.

    Raises:
        ValueError: If ``level`` is not recognized by the logging subsystem.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "key_value": {
                    "()": "textify.logging._KeyValueFormatter",
                    "format": "%(levelname)s:   [%(name)s] %(message)s",
                }
            },
            "filters": {
                "safe_context": {
                    "()": "textify.logging._SafeContextFilter",
                }
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "key_value",
                    "filters": ["safe_context"],
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "textify": {
                    "handlers": ["stderr"],
                    "level": level.upper(),
                    "propagate": False,
                },
                "httpx": {"level": "WARNING"},
                "yt_dlp": {"level": "WARNING"},
                "faster_whisper": {"level": "WARNING"},
            },
            "root": {"handlers": ["stderr"], "level": "WARNING"},
        }
    )
