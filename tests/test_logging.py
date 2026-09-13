"""Tests for Textify's safe structured logging boundary."""

import logging

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
        "source_id",
        "stage",
    }
    for field in (
        "code=safe-code",
        "duration_seconds=12",
        "elapsed_seconds=0.25",
        "method=faster_whisper",
        "platform=tiktok",
        "request_id=request-id-sentinel",
        "source_id=source-id-sentinel",
        "stage=request",
    ):
        assert field in rendered_output
    assert "dependency-sentinel" not in rendered_output
    for sensitive_value in sensitive_values.values():
        assert sensitive_value not in rendered_output
