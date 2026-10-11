"""Tests for Textify's safe structured logging boundary."""

import json
import logging
import subprocess
import sys
import textwrap
from contextlib import redirect_stderr
from datetime import UTC, datetime
from uuid import UUID

import pytest

from textify.logging import (
    LogEvent,
    LogServiceRole,
    StructuredLogFields,
    bind_request_log_fields,
    configure_logging,
    configure_structured_logging,
    emit_event,
    request_log_context,
    transcription_job_log_context,
)


def test_configured_logging_renders_only_safe_request_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Configured output retains only approved request fields and Textify records.

    Args:
        capsys: Captures formatted standard-error log output.
    """
    sensitive_values = {
        "submitted_url": "url-sentinel",
        "transcript_text": "text-sentinel",
        "output": "output-sentinel",
        "command": "command-sentinel",
        "headers": "header-sentinel",
        "credential": "credential-sentinel",
        "cookie": "cookie-sentinel",
        "media_path": "path-sentinel",
        "reason": "reason-sentinel",
        "postgresql_url": "postgresql-url-sentinel",
        "postgresql_password": "postgresql-password-sentinel",
        "redis_url": "redis-url-sentinel",
        "redis_password": "redis-password-sentinel",
        "public_capability": "capability-sentinel",
        "cancellation_link": "cancellation-link-sentinel",
        "execution_attempt_token": "execution-attempt-token-sentinel",
        "hugging_face_credential": "hugging-face-credential-sentinel",
        "provider_detail": "provider-detail-sentinel",
        "segment_count": "segment-count-sentinel",
    }
    configure_logging("INFO")

    with request_log_context("request-id-sentinel"):
        bind_request_log_fields(
            platform="tiktok",
            source_id="source-id-sentinel",
            method="faster_whisper",
            duration_seconds=12,
            code="safe-code",
        )
        logging.getLogger("textify.test_logging").info(
            "safe event",
            extra={
                "stage": "request",
                "elapsed_seconds": 0.25,
                "internal_job_id": 73,
                **sensitive_values,
            },
        )
    logging.getLogger("dependency.test_logging").warning("dependency-sentinel")

    rendered_output = capsys.readouterr().err
    rendered_fields = rendered_output.split("] safe event ", maxsplit=1)[1].split()
    rendered_field_names = {field.partition("=")[0] for field in rendered_fields}
    assert rendered_field_names == {
        "code",
        "duration_seconds",
        "elapsed_seconds",
        "method",
        "platform",
        "request_id",
        "internal_job_id",
        "source_id",
        "stage",
    }
    for field in (
        "code=safe-code",
        "duration_seconds=12",
        "elapsed_seconds=0.25",
        "method=faster_whisper",
        "internal_job_id=73",
        "platform=tiktok",
        "request_id=request-id-sentinel",
        "source_id=source-id-sentinel",
        "stage=request",
    ):
        assert field in rendered_output
    assert "dependency-sentinel" not in rendered_output
    for sensitive_value in sensitive_values.values():
        assert sensitive_value not in rendered_output


def test_celery_setup_preserves_safe_textify_logging() -> None:
    """Keep Celery records and private execution tokens out of rendered output."""
    child_program = textwrap.dedent(
        """
        import logging

        from textify.jobs.celery_app import create_celery_app
        from textify.jobs.config import JobDispatchConfig
        from textify.logging import configure_logging

        configure_logging("INFO")
        celery_app = create_celery_app(
            JobDispatchConfig(
                broker_url="redis://127.0.0.1:6379/15",
                _env_file=None,
            )
        )
        celery_app.log.setup_logging_subsystem(loglevel="INFO")
        logging.getLogger("celery.test").error(
            "celery-message-sentinel execution-attempt-token-sentinel",
            extra={"execution_attempt_token": "execution-attempt-token-sentinel"},
        )
        logging.getLogger("textify.test").info("textify-message-sentinel")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", child_program],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert "textify-message-sentinel" in completed.stderr
    assert "celery-message-sentinel" not in completed.stderr
    assert "execution-attempt-token-sentinel" not in completed.stderr


def test_structured_logging_emits_registered_event_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Render a fixed registered event as one safe JSON object.

    Args:
        capsys: Captures formatted standard-error log output.
    """
    configure_structured_logging(LogServiceRole.API)

    emit_event(logging.getLogger("textify.test_logging"), LogEvent.SERVICE_READY)

    rendered_event = json.loads(capsys.readouterr().err)
    assert rendered_event == {
        "timestamp": rendered_event["timestamp"],
        "level": "INFO",
        "logger": "textify.test_logging",
        "message": "service ready",
        "event": "service_ready",
        "service": "textify",
        "service_role": "api",
        "instance_id": rendered_event["instance_id"],
    }
    assert datetime.fromisoformat(rendered_event["timestamp"]).tzinfo is UTC
    assert rendered_event["timestamp"].endswith("Z")
    assert UUID(rendered_event["instance_id"]).version == 4


def test_structured_logging_drops_incomplete_registered_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drop a registered event whose required context is absent.

    Args:
        capsys: Captures formatted standard-error log output.
    """
    configure_structured_logging(LogServiceRole.API)

    emit_event(logging.getLogger("textify.test_logging"), LogEvent.REQUEST_COMPLETED)

    assert capsys.readouterr().err == ""


def test_structured_logging_inherits_and_overrides_request_correlation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Inherit the request identifier unless an event supplies its own.

    Args:
        capsys: Captures formatted standard-error log output.
    """
    request_identifier = "f8b2d0be-09be-4de9-bb79-f8c26dcbd568"
    explicit_request_identifier = "4c4c4f2b-0913-4adb-b4d8-2c8c863c8575"
    configure_structured_logging(LogServiceRole.API)

    with request_log_context(request_identifier):
        emit_event(
            logging.getLogger("textify.test_logging"),
            LogEvent.REQUEST_COMPLETED,
            http_status_code=200,
            http_method="GET",
            elapsed_seconds=0.25,
        )
        emit_event(
            logging.getLogger("textify.test_logging"),
            LogEvent.REQUEST_COMPLETED,
            request_id=explicit_request_identifier,
            http_status_code=200,
            http_method="GET",
            elapsed_seconds=0.25,
        )

    rendered_events = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    ]
    assert [event["request_id"] for event in rendered_events] == [
        request_identifier,
        explicit_request_identifier,
    ]


def test_structured_logging_restores_nested_transcription_job_correlation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Restore nested Transcription Job identifiers after an inner scope exits.

    Args:
        capsys: Captures formatted standard-error log output.
    """
    configure_structured_logging(LogServiceRole.API)
    logger = logging.getLogger("textify.test_logging")

    with transcription_job_log_context(47):
        emit_event(logger, LogEvent.JOB_DISPATCH_PUBLISHED)
        with transcription_job_log_context(53):
            emit_event(logger, LogEvent.JOB_DISPATCH_PUBLISHED)
            emit_event(
                logger,
                LogEvent.JOB_DISPATCH_PUBLISHED,
                internal_job_id=59,
            )
        emit_event(logger, LogEvent.JOB_DISPATCH_PUBLISHED)

    rendered_events = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    ]
    assert [event["internal_job_id"] for event in rendered_events] == [47, 53, 59, 47]


@pytest.mark.parametrize(
    ("event", "fields", "message", "level"),
    [
        pytest.param(
            LogEvent.SERVICE_STARTING,
            {},
            "service starting",
            "INFO",
            id="service-starting",
        ),
        pytest.param(
            LogEvent.SERVICE_READY,
            {},
            "service ready",
            "INFO",
            id="service-ready",
        ),
        pytest.param(
            LogEvent.SERVICE_FAILED,
            {"stage": "startup", "code": "internal_error"},
            "service failed",
            "ERROR",
            id="service-failed",
        ),
        pytest.param(
            LogEvent.SERVICE_STOPPING,
            {},
            "service stopping",
            "INFO",
            id="service-stopping",
        ),
        pytest.param(
            LogEvent.REQUEST_COMPLETED,
            {
                "request_id": "f8b2d0be-09be-4de9-bb79-f8c26dcbd568",
                "http_status_code": 200,
                "http_method": "GET",
                "elapsed_seconds": 0.25,
            },
            "request completed",
            "INFO",
            id="request-completed",
        ),
        pytest.param(
            LogEvent.REQUEST_INTERRUPTED,
            {
                "request_id": "f8b2d0be-09be-4de9-bb79-f8c26dcbd568",
                "http_method": "GET",
                "elapsed_seconds": 0.25,
            },
            "request interrupted",
            "INFO",
            id="request-interrupted",
        ),
        pytest.param(
            LogEvent.REQUEST_FAILED,
            {
                "request_id": "f8b2d0be-09be-4de9-bb79-f8c26dcbd568",
                "http_status_code": 500,
                "http_method": "GET",
                "elapsed_seconds": 0.25,
                "code": "internal_error",
            },
            "request failed",
            "ERROR",
            id="request-failed",
        ),
        pytest.param(
            LogEvent.TRANSCRIPTION_JOB_ADMITTED,
            {
                "request_id": "f8b2d0be-09be-4de9-bb79-f8c26dcbd568",
                "internal_job_id": 41,
                "job_status": "queued",
            },
            "transcription job admitted",
            "INFO",
            id="transcription-job-admitted",
        ),
        pytest.param(
            LogEvent.TRANSCRIPTION_JOB_CANCELLATION_ACCEPTED,
            {
                "request_id": "f8b2d0be-09be-4de9-bb79-f8c26dcbd568",
                "internal_job_id": 41,
                "job_status": "queued",
            },
            "transcription job cancellation accepted",
            "INFO",
            id="transcription-job-cancellation-accepted",
        ),
        pytest.param(
            LogEvent.JOB_DISPATCH_PUBLISHED,
            {"internal_job_id": 41},
            "job dispatch published",
            "INFO",
            id="job-dispatch-published",
        ),
        pytest.param(
            LogEvent.EXECUTION_ATTEMPT_CLAIMED,
            {"internal_job_id": 41, "job_status": "processing"},
            "execution attempt claimed",
            "INFO",
            id="execution-attempt-claimed",
        ),
        pytest.param(
            LogEvent.EXECUTION_ATTEMPT_OWNERSHIP_LOST,
            {"internal_job_id": 41},
            "execution attempt ownership lost",
            "WARNING",
            id="execution-attempt-ownership-lost",
        ),
        pytest.param(
            LogEvent.TRANSCRIPTION_JOB_FINISHED,
            {
                "internal_job_id": 41,
                "job_status": "finished",
                "job_outcome": "succeeded",
            },
            "transcription job finished",
            "INFO",
            id="transcription-job-finished",
        ),
        pytest.param(
            LogEvent.TRANSCRIPTION_JOB_RETENTION_EXPIRED,
            {"internal_job_id": 41, "job_outcome": "succeeded"},
            "transcription job retention expired",
            "INFO",
            id="transcription-job-retention-expired",
        ),
        pytest.param(
            LogEvent.SOURCE_INSPECTION_FAILED,
            {"stage": "metadata", "code": "source_invalid"},
            "source inspection failed",
            "DEBUG",
            id="source-inspection-failed",
        ),
        pytest.param(
            LogEvent.CAPTION_TRACK_UNAVAILABLE,
            {"stage": "captions", "code": "caption_unavailable"},
            "caption track unavailable",
            "DEBUG",
            id="caption-track-unavailable",
        ),
        pytest.param(
            LogEvent.CAPTION_ACQUISITION_FAILED,
            {"stage": "captions", "code": "caption_failed"},
            "caption acquisition failed",
            "DEBUG",
            id="caption-acquisition-failed",
        ),
        pytest.param(
            LogEvent.AUDIO_DOWNLOAD_FAILED,
            {"stage": "audio_download", "code": "download_failed"},
            "audio download failed",
            "DEBUG",
            id="audio-download-failed",
        ),
        pytest.param(
            LogEvent.NATIVE_TRANSCRIPTION_FAILED,
            {"stage": "transcription", "code": "native_failed"},
            "native transcription failed",
            "DEBUG",
            id="native-transcription-failed",
        ),
        pytest.param(
            LogEvent.TRANSCRIPT_ACQUIRED,
            {"stage": "transcription", "method": "captions"},
            "transcript acquired",
            "DEBUG",
            id="transcript-acquired",
        ),
        pytest.param(
            LogEvent.RESOURCE_CLEANUP_FAILED,
            {"stage": "cleanup", "code": "cleanup_failed"},
            "resource cleanup failed",
            "ERROR",
            id="resource-cleanup-failed",
        ),
        pytest.param(
            LogEvent.TRANSCRIPTION_JOB_PROCESSING_FAILED,
            {"internal_job_id": 41, "code": "processing_failed"},
            "transcription job processing failed",
            "ERROR",
            id="transcription-job-processing-failed",
        ),
        pytest.param(
            LogEvent.JOB_STORE_UNAVAILABLE,
            {"code": "storage_unavailable"},
            "transcription job storage is unavailable",
            "ERROR",
            id="job-store-unavailable",
        ),
        pytest.param(
            LogEvent.JOB_DISPATCH_UNAVAILABLE,
            {"internal_job_id": 41, "code": "dispatch_unavailable"},
            "transcription job dispatch is unavailable",
            "ERROR",
            id="job-dispatch-unavailable",
        ),
        pytest.param(
            LogEvent.MALFORMED_JOB_DISPATCH_REJECTED,
            {"code": "malformed_dispatch"},
            "malformed transcription job dispatch rejected",
            "WARNING",
            id="malformed-job-dispatch-rejected",
        ),
        pytest.param(
            LogEvent.WORKER_RUNTIME_UNAVAILABLE,
            {"internal_job_id": 41, "code": "runtime_unavailable"},
            "transcription worker runtime is unavailable",
            "ERROR",
            id="worker-runtime-unavailable",
        ),
        pytest.param(
            LogEvent.GPU_OWNERSHIP_FAILED,
            {"stage": "runtime", "code": "gpu_unavailable"},
            "gpu ownership failed",
            "ERROR",
            id="gpu-ownership-failed",
        ),
        pytest.param(
            LogEvent.SERVICE_READINESS_FAILED,
            {"stage": "readiness", "code": "readiness_failed"},
            "service readiness failed",
            "ERROR",
            id="service-readiness-failed",
        ),
    ],
)
def test_structured_logging_emits_each_registered_event(
    capsys: pytest.CaptureFixture[str],
    event: LogEvent,
    fields: StructuredLogFields,
    message: str,
    level: str,
) -> None:
    """Emit each registered event with its public contract's minimal context.

    Args:
        capsys: Captures formatted standard-error log output.
        event: Registered event under test.
        fields: Minimal valid context for the event.
        message: Fixed registry-owned message.
        level: Fixed registry-owned severity.
    """
    configure_structured_logging(LogServiceRole.API, "DEBUG")

    emit_event(logging.getLogger("textify.test_logging"), event, **fields)

    rendered_event = json.loads(capsys.readouterr().err)
    assert rendered_event["event"] == event.value
    assert rendered_event["message"] == message
    assert rendered_event["level"] == level
    assert rendered_event["timestamp"].endswith("Z")


def test_structured_logging_preserves_optional_scalar_types_and_omits_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve approved scalar types and omit unavailable optional fields.

    Args:
        capsys: Captures formatted standard-error log output.
    """
    optional_fields: StructuredLogFields = {
        "request_id": "f8b2d0be-09be-4de9-bb79-f8c26dcbd568",
        "internal_job_id": 41,
        "platform": "youtube",
        "source_id": "source-41",
        "stage": "request",
        "method": "captions",
        "duration_seconds": 12,
        "elapsed_seconds": 0.25,
        "code": "safe_code",
        "job_status": "queued",
        "job_outcome": "succeeded",
        "http_status_code": 200,
        "http_method": "GET",
        "http_route": "/transcriptions/{id}",
    }
    configure_structured_logging(LogServiceRole.API)
    logger = logging.getLogger("textify.test_logging")

    emit_event(logger, LogEvent.SERVICE_READY, **optional_fields)
    emit_event(logger, LogEvent.SERVICE_READY, platform=None)  # type: ignore[arg-type]

    rendered_events = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    ]
    rendered_optional_fields = {
        field_name: rendered_events[0][field_name] for field_name in optional_fields
    }
    assert rendered_optional_fields == optional_fields
    assert type(rendered_events[0]["internal_job_id"]) is int
    assert type(rendered_events[0]["duration_seconds"]) is int
    assert type(rendered_events[0]["elapsed_seconds"]) is float
    assert type(rendered_events[0]["http_status_code"]) is int
    assert "platform" not in rendered_events[1]


def test_structured_logging_reuses_identity_for_one_service_role(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep one process identifier while allowing same-role level updates.

    Args:
        capsys: Captures formatted standard-error log output.
    """
    logger = logging.getLogger("textify.test_logging")
    configure_structured_logging(LogServiceRole.API)
    emit_event(logger, LogEvent.SERVICE_READY)
    first_instance_id = json.loads(capsys.readouterr().err)["instance_id"]

    configure_structured_logging(LogServiceRole.API, "DEBUG")
    emit_event(logger, LogEvent.SERVICE_READY)
    second_instance_id = json.loads(capsys.readouterr().err)["instance_id"]

    assert second_instance_id == first_instance_id
    with pytest.raises(ValueError, match="another role"):
        configure_structured_logging(LogServiceRole.RECONCILER)


def test_structured_logging_drops_unsafe_calls_without_stringifying_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drop invalid calls and direct logger bypasses without leaking sentinel text.

    Args:
        capsys: Captures formatted standard-error log output.
    """

    class StringificationSentinel:
        def __str__(self) -> str:
            raise AssertionError("sentinel stringification must not run")

    sentinel = StringificationSentinel()
    logger = logging.getLogger("textify.test_logging")
    configure_structured_logging(LogServiceRole.API, "DEBUG")

    emit_event(logger, "unregistered-event")  # type: ignore[arg-type]
    emit_event(logger, LogEvent.SERVICE_READY, unknown_field=sentinel)  # type: ignore[call-arg]
    emit_event(
        logger,
        LogEvent.SERVICE_READY,
        code=sentinel,  # type: ignore[arg-type]
    )
    emit_event(
        logger,
        LogEvent.REQUEST_COMPLETED,
        request_id="uppercase-uuid-sentinel",
        http_status_code=200,
        http_method="GET",
        elapsed_seconds=0.25,
    )
    emit_event(logger, LogEvent.SERVICE_READY, internal_job_id=True)
    emit_event(logger, LogEvent.SERVICE_READY, duration_seconds=True)
    emit_event(logger, LogEvent.SERVICE_READY, elapsed_seconds=float("nan"))
    emit_event(logger, LogEvent.SERVICE_READY, elapsed_seconds=float("inf"))
    emit_event(logger, LogEvent.SERVICE_READY, http_status_code=99)
    emit_event(logger, LogEvent.SERVICE_READY, http_method="get")
    emit_event(logger, LogEvent.SERVICE_READY, stage="unknown-stage")
    emit_event(logger, LogEvent.SERVICE_READY, job_status="unknown-status")
    emit_event(logger, LogEvent.SERVICE_READY, job_outcome="unknown-outcome")
    emit_event(logging.getLogger("dependency.test_logging"), LogEvent.SERVICE_READY)
    emit_event(object(), LogEvent.SERVICE_READY)  # type: ignore[arg-type]
    logger.info("formatting-sentinel %s", sentinel)
    logger.error(
        "exception-sentinel",
        exc_info=RuntimeError("exception-detail-sentinel"),
    )
    logger.error("stack-sentinel", stack_info=True)
    logger.info(
        "direct-bypass-sentinel",
        extra={"unapproved_extra": "unapproved-extra-sentinel"},
    )

    assert capsys.readouterr().err == ""


def test_structured_logging_suppresses_stream_failures() -> None:
    """Keep registered-event emission non-fatal when the configured stream fails."""

    class FailingStderr:
        def __init__(self) -> None:
            self.writes: list[str] = []

        def write(self, message: str) -> int:
            self.writes.append(message)
            raise OSError("write failed")

        def flush(self) -> None:
            return None

    failing_stderr = FailingStderr()
    logger = logging.getLogger("textify.test_logging")

    with redirect_stderr(failing_stderr):
        configure_structured_logging(LogServiceRole.API)
        emit_event(logger, LogEvent.SERVICE_READY)

    assert len(failing_stderr.writes) == 1
    assert "Logging error" not in failing_stderr.writes[0]
