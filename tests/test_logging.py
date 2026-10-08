"""Tests for Textify's safe structured logging boundary."""

import logging
import subprocess
import sys
import textwrap

import pytest

from textify.logging import (
    bind_request_log_fields,
    configure_logging,
    request_log_context,
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
