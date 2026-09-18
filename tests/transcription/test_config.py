"""Tests for transcription environment configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from textify.config import AppConfig
from textify.transcription.config import TranscriptionConfig


def test_cpu_device_setting_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject CPU execution before transcription adapters are constructed."""
    monkeypatch.setenv("TEXTIFY_WHISPER_DEVICE", "cpu")

    with pytest.raises(ValidationError):
        TranscriptionConfig(_env_file=None)  # type: ignore[call-arg]


def test_database_path_is_optional_only_before_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep application imports and OpenAPI generation independent of runtime storage."""
    monkeypatch.delenv("TEXTIFY_DATABASE_PATH", raising=False)

    settings = AppConfig(_env_file=None)  # type: ignore[call-arg]

    assert settings.database_path is None


def test_database_path_loads_explicit_filesystem_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the configured durable SQLite filesystem path from the environment."""
    monkeypatch.setenv("TEXTIFY_DATABASE_PATH", "./textify.sqlite3")

    settings = AppConfig(_env_file=None)  # type: ignore[call-arg]

    assert settings.database_path == Path("textify.sqlite3")


def test_durable_job_settings_use_bounded_defaults() -> None:
    """Default durable admission, worker ownership, and polling cadence."""
    settings = TranscriptionConfig(_env_file=None)  # type: ignore[call-arg]

    assert settings.max_outstanding_jobs == 8
    assert settings.job_queue_timeout_seconds == 20
    assert settings.job_worker_count == 1
    assert "max_pending_transcriptions" not in TranscriptionConfig.model_fields
    assert "transcription_queue_timeout_seconds" not in TranscriptionConfig.model_fields


@pytest.mark.parametrize(
    ("setting_name", "setting_value"),
    (
        ("TEXTIFY_MAX_OUTSTANDING_JOBS", "0"),
        ("TEXTIFY_MAX_OUTSTANDING_JOBS", "-1"),
        ("TEXTIFY_JOB_QUEUE_TIMEOUT_SECONDS", "0"),
        ("TEXTIFY_JOB_QUEUE_TIMEOUT_SECONDS", "-1"),
        ("TEXTIFY_JOB_WORKER_COUNT", "0"),
        ("TEXTIFY_JOB_WORKER_COUNT", "-1"),
    ),
)
def test_nonpositive_queued_job_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    setting_name: str,
    setting_value: str,
) -> None:
    """Reject nonpositive durable admission and deadline configuration."""
    monkeypatch.setenv(setting_name, setting_value)

    with pytest.raises(ValidationError):
        TranscriptionConfig(_env_file=None)  # type: ignore[call-arg]
