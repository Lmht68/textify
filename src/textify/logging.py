"""Configure Textify's safe structured request logging convention."""

import json
import logging
import logging.config
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import TextIO, TypedDict, Unpack
from uuid import uuid4

from textify.identifiers import parse_canonical_uuid4

_ALLOWED_STRUCTURED_FIELDS = frozenset(
    {
        "request_id",
        "internal_job_id",
        "platform",
        "source_id",
        "stage",
        "method",
        "duration_seconds",
        "elapsed_seconds",
        "code",
        "job_status",
        "job_outcome",
        "http_status_code",
        "http_method",
        "http_route",
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

_transcription_job_log_context: ContextVar[int | None] = ContextVar(
    "textify_transcription_job_log_context",
    default=None,
)


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


@contextmanager
def worker_log_context() -> Iterator[None]:
    """Isolate worker records from an inherited HTTP request context.

    Yields:
        None: The active worker scope.
    """
    token = _request_log_context.set(None)
    try:
        yield
    finally:
        _request_log_context.reset(token)


@contextmanager
def transcription_job_log_context(internal_job_id: int) -> Iterator[None]:
    """Bind one Transcription Job identifier to the current execution scope.

    Args:
        internal_job_id: Private PostgreSQL Transcription Job identifier.

    Yields:
        None: The active Transcription Job scope.
    """
    token = _transcription_job_log_context.set(internal_job_id)
    try:
        yield
    finally:
        _transcription_job_log_context.reset(token)


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


def _inherited_log_fields() -> dict[str, object]:
    context = _request_log_context.get()
    fields: dict[str, object] = {}
    if context is not None:
        for field_name, field_value in (
            ("request_id", context.request_id),
            ("platform", context.platform),
            ("source_id", context.source_id),
            ("method", context.method),
            ("duration_seconds", context.duration_seconds),
            ("code", context.code),
        ):
            if field_value is not None:
                fields[field_name] = field_value
    internal_job_id = _transcription_job_log_context.get()
    if internal_job_id is not None:
        fields["internal_job_id"] = internal_job_id
    return fields


class LogServiceRole(StrEnum):
    """Identify a Textify process that emits registered events."""

    API = "api"
    RECONCILER = "reconciler"
    GPU_WORKER = "gpu_worker"


class LogEvent(StrEnum):
    """Identify a registered safe Textify logging event."""

    SERVICE_STARTING = "service_starting"
    SERVICE_READY = "service_ready"
    SERVICE_FAILED = "service_failed"
    SERVICE_STOPPING = "service_stopping"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_INTERRUPTED = "request_interrupted"
    REQUEST_FAILED = "request_failed"
    TRANSCRIPTION_JOB_ADMITTED = "transcription_job_admitted"
    TRANSCRIPTION_JOB_CANCELLATION_ACCEPTED = "transcription_job_cancellation_accepted"
    EXECUTION_ATTEMPT_CLAIMED = "execution_attempt_claimed"
    EXECUTION_ATTEMPT_OWNERSHIP_LOST = "execution_attempt_ownership_lost"
    TRANSCRIPTION_JOB_FINISHED = "transcription_job_finished"
    SOURCE_INSPECTION_FAILED = "source_inspection_failed"
    CAPTION_TRACK_UNAVAILABLE = "caption_track_unavailable"
    CAPTION_ACQUISITION_FAILED = "caption_acquisition_failed"
    AUDIO_DOWNLOAD_FAILED = "audio_download_failed"
    NATIVE_TRANSCRIPTION_FAILED = "native_transcription_failed"
    TRANSCRIPT_ACQUIRED = "transcript_acquired"
    RESOURCE_CLEANUP_FAILED = "resource_cleanup_failed"
    TRANSCRIPTION_JOB_PROCESSING_FAILED = "transcription_job_processing_failed"
    JOB_STORE_UNAVAILABLE = "job_store_unavailable"
    JOB_DISPATCH_UNAVAILABLE = "job_dispatch_unavailable"
    MALFORMED_JOB_DISPATCH_REJECTED = "malformed_job_dispatch_rejected"
    WORKER_RUNTIME_UNAVAILABLE = "worker_runtime_unavailable"
    GPU_OWNERSHIP_FAILED = "gpu_ownership_failed"
    SERVICE_READINESS_FAILED = "service_readiness_failed"


class LogCode(StrEnum):
    """Identify a fixed safe code for job-infrastructure diagnostics."""

    TRANSCRIPTION_JOB_STORAGE_IS_UNAVAILABLE_TO_CLAIM_MONITOR = (
        "transcription_job_storage_is_unavailable_to_claim_monitor"
    )
    TRANSCRIPTION_JOB_DISPATCH_BROKER_IS_UNAVAILABLE = (
        "transcription_job_dispatch_broker_is_unavailable"
    )
    TRANSCRIPTION_JOB_STORAGE_IS_UNAVAILABLE_TO_RECONCILER = (
        "transcription_job_storage_is_unavailable_to_reconciler"
    )
    RECONCILER_SERVICE_READINESS_STORAGE_IS_UNAVAILABLE = (
        "reconciler_service_readiness_storage_is_unavailable"
    )
    REJECTED_MALFORMED_TRANSCRIPTION_JOB_DISPATCH = (
        "rejected_malformed_transcription_job_dispatch"
    )
    TRANSCRIPTION_WORKER_RUNTIME_IS_UNAVAILABLE = (
        "transcription_worker_runtime_is_unavailable"
    )
    TRANSCRIPTION_JOB_STORAGE_IS_UNAVAILABLE_TO_WORKER = (
        "transcription_job_storage_is_unavailable_to_worker"
    )
    GPU_OWNERSHIP_CONNECTION_CLOSE_FAILED = "gpu_ownership_connection_close_failed"
    GPU_OWNERSHIP_CONNECTION_INVALIDATION_FAILED = (
        "gpu_ownership_connection_invalidation_failed"
    )
    SERVICE_READINESS_HEARTBEAT_REFRESH_FAILED = (
        "service_readiness_heartbeat_refresh_failed"
    )
    SERVICE_READINESS_HEARTBEAT_DELETION_FAILED = (
        "service_readiness_heartbeat_deletion_failed"
    )
    BROKER_READINESS_PROBE_FAILED = "broker_readiness_probe_failed"
    WORKER_SERVICE_HEARTBEAT_STARTUP_CLEANUP_FAILED = (
        "worker_service_heartbeat_startup_cleanup_failed"
    )
    WORKER_CLAIM_MANAGER_STARTUP_CLEANUP_FAILED = (
        "worker_claim_manager_startup_cleanup_failed"
    )
    TRANSCRIPTION_EXECUTOR_STARTUP_CLEANUP_FAILED = (
        "transcription_executor_startup_cleanup_failed"
    )
    GPU_OWNERSHIP_STARTUP_CLEANUP_FAILED = "gpu_ownership_startup_cleanup_failed"
    WORKER_ENGINE_STARTUP_CLEANUP_FAILED = "worker_engine_startup_cleanup_failed"


class StructuredLogFields(TypedDict, total=False):
    """Contain the optional fields allowed on a registered event."""

    request_id: str
    internal_job_id: int
    platform: str
    source_id: str
    stage: str
    method: str
    duration_seconds: int
    elapsed_seconds: int | float
    code: str | None
    job_status: str
    job_outcome: str
    http_status_code: int
    http_method: str
    http_route: str


@dataclass(frozen=True, slots=True)
class _EventDefinition:
    """Store the fixed presentation for one registered event."""

    message: str
    level: int


_EVENT_REGISTRY: Mapping[LogEvent, _EventDefinition] = MappingProxyType(
    {
        LogEvent.SERVICE_STARTING: _EventDefinition("service starting", logging.INFO),
        LogEvent.SERVICE_READY: _EventDefinition("service ready", logging.INFO),
        LogEvent.SERVICE_FAILED: _EventDefinition("service failed", logging.ERROR),
        LogEvent.SERVICE_STOPPING: _EventDefinition("service stopping", logging.INFO),
        LogEvent.REQUEST_COMPLETED: _EventDefinition("request completed", logging.INFO),
        LogEvent.REQUEST_INTERRUPTED: _EventDefinition(
            "request interrupted", logging.INFO
        ),
        LogEvent.REQUEST_FAILED: _EventDefinition("request failed", logging.ERROR),
        LogEvent.TRANSCRIPTION_JOB_ADMITTED: _EventDefinition(
            "transcription job admitted", logging.INFO
        ),
        LogEvent.TRANSCRIPTION_JOB_CANCELLATION_ACCEPTED: _EventDefinition(
            "transcription job cancellation accepted", logging.INFO
        ),
        LogEvent.EXECUTION_ATTEMPT_CLAIMED: _EventDefinition(
            "execution attempt claimed", logging.INFO
        ),
        LogEvent.EXECUTION_ATTEMPT_OWNERSHIP_LOST: _EventDefinition(
            "execution attempt ownership lost", logging.WARNING
        ),
        LogEvent.TRANSCRIPTION_JOB_FINISHED: _EventDefinition(
            "transcription job finished", logging.INFO
        ),
        LogEvent.SOURCE_INSPECTION_FAILED: _EventDefinition(
            "source inspection failed", logging.DEBUG
        ),
        LogEvent.CAPTION_TRACK_UNAVAILABLE: _EventDefinition(
            "caption track unavailable", logging.DEBUG
        ),
        LogEvent.CAPTION_ACQUISITION_FAILED: _EventDefinition(
            "caption acquisition failed", logging.DEBUG
        ),
        LogEvent.AUDIO_DOWNLOAD_FAILED: _EventDefinition(
            "audio download failed", logging.DEBUG
        ),
        LogEvent.NATIVE_TRANSCRIPTION_FAILED: _EventDefinition(
            "native transcription failed", logging.DEBUG
        ),
        LogEvent.TRANSCRIPT_ACQUIRED: _EventDefinition(
            "transcript acquired", logging.DEBUG
        ),
        LogEvent.RESOURCE_CLEANUP_FAILED: _EventDefinition(
            "resource cleanup failed", logging.ERROR
        ),
        LogEvent.TRANSCRIPTION_JOB_PROCESSING_FAILED: _EventDefinition(
            "transcription job processing failed", logging.ERROR
        ),
        LogEvent.JOB_STORE_UNAVAILABLE: _EventDefinition(
            "transcription job storage is unavailable", logging.ERROR
        ),
        LogEvent.JOB_DISPATCH_UNAVAILABLE: _EventDefinition(
            "transcription job dispatch is unavailable", logging.ERROR
        ),
        LogEvent.MALFORMED_JOB_DISPATCH_REJECTED: _EventDefinition(
            "malformed transcription job dispatch rejected", logging.WARNING
        ),
        LogEvent.WORKER_RUNTIME_UNAVAILABLE: _EventDefinition(
            "transcription worker runtime is unavailable", logging.ERROR
        ),
        LogEvent.GPU_OWNERSHIP_FAILED: _EventDefinition(
            "gpu ownership failed", logging.ERROR
        ),
        LogEvent.SERVICE_READINESS_FAILED: _EventDefinition(
            "service readiness failed", logging.ERROR
        ),
    }
)
_EVENT_MARKER = object()
_EVENT_MARKER_FIELD = "_textify_registered_event_marker"
_EVENT_FIELD = "event"
_STANDARD_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)
_ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_structured_service_role: LogServiceRole | None = None
_structured_instance_id: str | None = None

_STRING_FIELDS = frozenset(
    {
        "platform",
        "source_id",
        "method",
        "code",
        "http_route",
    }
)
_STAGE_VALUES = frozenset(
    {
        "request",
        "metadata",
        "captions",
        "audio_download",
        "transcription",
        "processing",
        "dispatch",
        "readiness",
        "startup",
        "runtime",
        "shutdown",
        "cleanup",
    }
)
_JOB_STATUS_VALUES = frozenset({"queued", "processing", "finished"})
_JOB_OUTCOME_VALUES = frozenset({"succeeded", "failed", "cancelled"})
_REQUIRED_EVENT_FIELDS: Mapping[LogEvent, frozenset[str]] = MappingProxyType(
    {
        LogEvent.SERVICE_FAILED: frozenset({"stage", "code"}),
        LogEvent.REQUEST_COMPLETED: frozenset(
            {"request_id", "http_status_code", "http_method", "elapsed_seconds"}
        ),
        LogEvent.REQUEST_INTERRUPTED: frozenset(
            {"request_id", "http_method", "elapsed_seconds"}
        ),
        LogEvent.REQUEST_FAILED: frozenset(
            {
                "request_id",
                "http_status_code",
                "http_method",
                "elapsed_seconds",
                "code",
            }
        ),
        LogEvent.TRANSCRIPTION_JOB_ADMITTED: frozenset(
            {"request_id", "internal_job_id", "job_status"}
        ),
        LogEvent.TRANSCRIPTION_JOB_CANCELLATION_ACCEPTED: frozenset(
            {"request_id", "internal_job_id", "job_status"}
        ),
        LogEvent.EXECUTION_ATTEMPT_CLAIMED: frozenset(
            {"internal_job_id", "job_status"}
        ),
        LogEvent.EXECUTION_ATTEMPT_OWNERSHIP_LOST: frozenset({"internal_job_id"}),
        LogEvent.TRANSCRIPTION_JOB_FINISHED: frozenset(
            {"internal_job_id", "job_status", "job_outcome"}
        ),
        LogEvent.SOURCE_INSPECTION_FAILED: frozenset({"stage", "code"}),
        LogEvent.CAPTION_TRACK_UNAVAILABLE: frozenset({"stage", "code"}),
        LogEvent.CAPTION_ACQUISITION_FAILED: frozenset({"stage", "code"}),
        LogEvent.AUDIO_DOWNLOAD_FAILED: frozenset({"stage", "code"}),
        LogEvent.NATIVE_TRANSCRIPTION_FAILED: frozenset({"stage", "code"}),
        LogEvent.TRANSCRIPT_ACQUIRED: frozenset({"stage", "method"}),
        LogEvent.RESOURCE_CLEANUP_FAILED: frozenset({"stage", "code"}),
        LogEvent.TRANSCRIPTION_JOB_PROCESSING_FAILED: frozenset(
            {"internal_job_id", "code"}
        ),
        LogEvent.JOB_STORE_UNAVAILABLE: frozenset({"code"}),
        LogEvent.JOB_DISPATCH_UNAVAILABLE: frozenset({"code", "internal_job_id"}),
        LogEvent.MALFORMED_JOB_DISPATCH_REJECTED: frozenset({"code"}),
        LogEvent.WORKER_RUNTIME_UNAVAILABLE: frozenset({"code", "internal_job_id"}),
        LogEvent.GPU_OWNERSHIP_FAILED: frozenset({"stage", "code"}),
        LogEvent.SERVICE_READINESS_FAILED: frozenset({"stage", "code"}),
    }
)


def _is_canonical_uuid4(value: object) -> bool:
    try:
        parse_canonical_uuid4(value)
    except ValueError:
        return False
    return True


def _is_supported_field_value(field_name: str, value: object) -> bool:
    if field_name == "request_id":
        return _is_canonical_uuid4(value)
    if field_name in _STRING_FIELDS:
        return type(value) is str and bool(value)
    if field_name == "internal_job_id":
        return type(value) is int and value > 0
    if field_name == "duration_seconds":
        return type(value) is int and value >= 0
    if field_name == "elapsed_seconds":
        return (type(value) is int and value >= 0) or (
            type(value) is float and isfinite(value) and value >= 0
        )
    if field_name == "http_status_code":
        return type(value) is int and 100 <= value <= 599
    if field_name == "http_method":
        return type(value) is str and bool(value) and value == value.upper()
    if field_name == "stage":
        return type(value) is str and value in _STAGE_VALUES
    if field_name == "job_status":
        return type(value) is str and value in _JOB_STATUS_VALUES
    if field_name == "job_outcome":
        return type(value) is str and value in _JOB_OUTCOME_VALUES
    return False


def _validate_event_fields(
    event: LogEvent,
    candidate_fields: Mapping[str, object],
) -> dict[str, object] | None:
    if set(candidate_fields) - _ALLOWED_STRUCTURED_FIELDS:
        return None
    fields = {
        field_name: field_value
        for field_name, field_value in candidate_fields.items()
        if field_value is not None
    }
    if not all(
        _is_supported_field_value(field_name, field_value)
        for field_name, field_value in fields.items()
    ):
        return None
    if not _REQUIRED_EVENT_FIELDS.get(event, frozenset()).issubset(fields):
        return None
    if event in {LogEvent.SERVICE_FAILED, LogEvent.REQUEST_FAILED}:
        return fields if fields["code"] == "internal_error" else None
    if event is LogEvent.TRANSCRIPTION_JOB_ADMITTED:
        return fields if fields["job_status"] == "queued" else None
    if event is LogEvent.EXECUTION_ATTEMPT_CLAIMED:
        return fields if fields["job_status"] == "processing" else None
    if (
        event is LogEvent.TRANSCRIPTION_JOB_CANCELLATION_ACCEPTED
        and fields["job_status"] == "finished"
        and "job_outcome" not in fields
    ):
        return None
    if event is LogEvent.TRANSCRIPTION_JOB_FINISHED:
        if fields["job_status"] != "finished":
            return None
        if fields["job_outcome"] == "failed" and "code" not in fields:
            return None
    return fields


class _StructuredRecordFilter(logging.Filter):
    """Accept only complete records emitted through the registered event seam."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether one record can be serialized as a safe JSON event."""
        if _structured_service_role is None or _structured_instance_id is None:
            return False
        if record.name != "textify" and not record.name.startswith("textify."):
            return False
        if record.__dict__.get(_EVENT_MARKER_FIELD) is not _EVENT_MARKER:
            return False
        event = record.__dict__.get(_EVENT_FIELD)
        if not isinstance(event, LogEvent):
            return False
        definition = _EVENT_REGISTRY.get(event)
        if definition is None:
            return False
        if (
            record.msg != definition.message
            or record.levelno != definition.level
            or record.args != ()
            or record.exc_info is not None
            or record.exc_text is not None
            or record.stack_info is not None
        ):
            return False
        allowed_fields = (
            _STANDARD_LOG_RECORD_FIELDS
            | _ALLOWED_STRUCTURED_FIELDS
            | {
                _EVENT_MARKER_FIELD,
                _EVENT_FIELD,
            }
        )
        if not set(record.__dict__) <= allowed_fields:
            return False
        record_fields = {
            field_name: record.__dict__[field_name]
            for field_name in _ALLOWED_STRUCTURED_FIELDS
            if field_name in record.__dict__
        }
        return _validate_event_fields(event, record_fields) is not None


class _JsonFormatter(logging.Formatter):
    """Render a validated structured record as one compact JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Format one validated structured log record.

        Args:
            record: A record accepted by ``_StructuredRecordFilter``.

        Returns:
            One compact JSON object.
        """
        service_role = _structured_service_role
        instance_id = _structured_instance_id
        event = record.__dict__.get(_EVENT_FIELD)
        if (
            service_role is None
            or instance_id is None
            or not isinstance(event, LogEvent)
        ):
            return ""
        definition = _EVENT_REGISTRY.get(event)
        if definition is None:
            return ""
        payload: dict[str, str | int | float] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": definition.message,
            "event": event.value,
            "service": "textify",
            "service_role": service_role.value,
            "instance_id": instance_id,
        }
        for field_name in _ALLOWED_STRUCTURED_FIELDS:
            field_value = record.__dict__.get(field_name)
            if isinstance(field_value, (str, int, float)) and not isinstance(
                field_value, bool
            ):
                payload[field_name] = field_value
        return json.dumps(payload, allow_nan=False, separators=(",", ":"))


class _SilentStreamHandler(logging.StreamHandler[TextIO]):
    """Suppress logging's internal errors for non-fatal diagnostics."""

    def handleError(self, record: logging.LogRecord) -> None:
        """Ignore formatter and stream errors from safe diagnostic output.

        Args:
            record: The record whose handling failed.
        """


def configure_structured_logging(
    service_role: LogServiceRole,
    level: str = "INFO",
) -> None:
    """Configure registered-event JSON logging for one Textify process.

    Args:
        service_role: The process role that owns this logging configuration.
        level: Minimum accepted registered-event severity.

    Raises:
        TypeError: If the role is not a ``LogServiceRole``.
        ValueError: If the level is invalid, or the role changes.
    """
    global _structured_instance_id, _structured_service_role

    if not isinstance(service_role, LogServiceRole):
        raise TypeError("service_role must be a LogServiceRole")
    normalized_level = level.upper() if isinstance(level, str) else ""
    if normalized_level not in _ALLOWED_LOG_LEVELS:
        raise ValueError("level must be a standard Textify logging level")
    if (
        _structured_service_role is not None
        and _structured_service_role is not service_role
    ):
        raise ValueError("structured logging is already configured for another role")
    if _structured_service_role is None:
        _structured_service_role = service_role
        _structured_instance_id = str(uuid4())

    logging.raiseExceptions = False
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "textify.logging._JsonFormatter",
                }
            },
            "filters": {
                "structured_record": {
                    "()": "textify.logging._StructuredRecordFilter",
                }
            },
            "handlers": {
                "stderr": {
                    "class": "textify.logging._SilentStreamHandler",
                    "formatter": "json",
                    "filters": ["structured_record"],
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "textify": {
                    "handlers": ["stderr"],
                    "level": normalized_level,
                    "propagate": False,
                },
                "httpx": {"level": "WARNING"},
                "yt_dlp": {"level": "WARNING"},
                "faster_whisper": {"level": "WARNING"},
            },
            "root": {"handlers": ["stderr"], "level": "WARNING"},
        }
    )


def emit_event(
    logger: logging.Logger,
    event: LogEvent,
    **fields: Unpack[StructuredLogFields],
) -> None:
    """Emit one fixed registered event without exposing arbitrary log content.

    Args:
        logger: A logger within the Textify namespace.
        event: The registered event to emit.
        **fields: Optional approved event context.
    """
    if not isinstance(logger, logging.Logger):
        return
    if logger.name != "textify" and not logger.name.startswith("textify."):
        return
    if not isinstance(event, LogEvent):
        return
    definition = _EVENT_REGISTRY.get(event)
    if definition is None:
        return
    merged_fields = _inherited_log_fields()
    merged_fields.update(fields)
    validated_fields = _validate_event_fields(event, merged_fields)
    if validated_fields is None:
        return
    try:
        logger.log(
            definition.level,
            definition.message,
            extra={
                _EVENT_MARKER_FIELD: _EVENT_MARKER,
                _EVENT_FIELD: event,
                **validated_fields,
            },
        )
    except Exception:  # noqa: BLE001
        return
