"""Configure Textify's safe structured logging convention.

INFO records describe stage transitions, durations, and counts.
Structured fields use lowercase snake_case names passed through ``extra``.
Fields that can reveal supplied URLs, media, model output, request material,
local paths, commands, credentials, or secrets are excluded.
"""

import logging
import logging.config

_STANDARD_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)

_SENSITIVE_FIELD_MARKERS = frozenset(
    {
        "command",
        "cookie",
        "credential",
        "header",
        "model",
        "output",
        "path",
        "secret",
        "token",
        "transcript",
        "url",
    }
)


class _KeyValueFormatter(logging.Formatter):
    """Append non-standard log record fields as key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with its structured fields.

        Args:
            record: Log record to format.

        Returns:
            str: The rendered log message and structured fields.
        """
        fields = [
            f"{key}={record.__dict__[key]}"
            for key in sorted(record.__dict__)
            if key not in _STANDARD_LOG_RECORD_FIELDS
            and not key.startswith("_")
            and _is_safe_log_field(key)
        ]
        message = super().format(record)
        return " ".join((message, *fields))


def _is_safe_log_field(key: str) -> bool:
    """Return whether a structured field name is safe to render."""
    normalized_key = key.casefold()
    return not any(marker in normalized_key for marker in _SENSITIVE_FIELD_MARKERS)


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
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "key_value",
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
