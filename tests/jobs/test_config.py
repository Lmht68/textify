"""Tests for durable PostgreSQL Transcription Job environment configuration."""

import pytest
from pydantic import ValidationError

from textify.jobs.config import JobConfig, JobDispatchConfig

_DATABASE_URL = "postgresql+asyncpg://textify:change-me@127.0.0.1:5432/textify"


def test_database_url_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject configuration that omits durable PostgreSQL storage."""
    monkeypatch.delenv("TEXTIFY_DATABASE_URL", raising=False)

    with pytest.raises(ValidationError):
        JobConfig(_env_file=None)  # type: ignore[call-arg]


def test_database_url_loads_explicit_async_postgresql_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the configured async PostgreSQL URL from the environment."""
    monkeypatch.setenv("TEXTIFY_DATABASE_URL", _DATABASE_URL)

    settings = JobConfig(_env_file=None)  # type: ignore[call-arg]

    assert str(settings.database_url) == _DATABASE_URL


@pytest.mark.parametrize(
    "database_url",
    (
        "sqlite+aiosqlite:///textify.sqlite3",
        "postgresql://textify:change-me@127.0.0.1:5432/textify",
    ),
)
def test_database_url_rejects_non_async_postgresql_values(
    database_url: str,
) -> None:
    """Reject SQLite and synchronous PostgreSQL durable-store URLs."""
    with pytest.raises(ValidationError):
        JobConfig(database_url=database_url, _env_file=None)  # type: ignore[call-arg]


def test_durable_job_settings_use_bounded_defaults() -> None:
    """Default durable admission, deadline, and retention policy."""
    settings = JobConfig(
        database_url=_DATABASE_URL,
        _env_file=None,  # type: ignore[call-arg]
    )

    assert settings.max_outstanding_jobs == 8
    assert settings.job_queue_timeout_seconds == 20
    assert settings.job_retention_seconds == 86_400
    assert "job_worker_count" not in JobConfig.model_fields
    assert "max_pending_transcriptions" not in JobConfig.model_fields
    assert "transcription_queue_timeout_seconds" not in JobConfig.model_fields


def test_broker_url_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject configuration that omits the dedicated Redis broker."""
    monkeypatch.delenv("TEXTIFY_BROKER_URL", raising=False)

    with pytest.raises(ValidationError):
        JobDispatchConfig(_env_file=None)  # type: ignore[call-arg]


def test_dispatch_settings_use_bounded_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read Redis delivery configuration independently from job storage."""
    monkeypatch.setenv("TEXTIFY_BROKER_URL", "redis://127.0.0.1:6379/15")

    settings = JobDispatchConfig(_env_file=None)  # type: ignore[call-arg]

    assert str(settings.broker_url) == "redis://127.0.0.1:6379/15"
    assert settings.worker_concurrency == 4
    assert settings.reconciler_interval_seconds == 1.0


@pytest.mark.parametrize(
    ("setting_name", "setting_value"),
    (
        ("TEXTIFY_MAX_OUTSTANDING_JOBS", "0"),
        ("TEXTIFY_MAX_OUTSTANDING_JOBS", "-1"),
        ("TEXTIFY_JOB_QUEUE_TIMEOUT_SECONDS", "0"),
        ("TEXTIFY_JOB_QUEUE_TIMEOUT_SECONDS", "-1"),
        ("TEXTIFY_JOB_RETENTION_SECONDS", "0"),
        ("TEXTIFY_JOB_RETENTION_SECONDS", "-1"),
    ),
)
def test_nonpositive_job_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    setting_name: str,
    setting_value: str,
) -> None:
    """Reject nonpositive durable admission, deadline, and retention configuration."""
    monkeypatch.setenv("TEXTIFY_DATABASE_URL", _DATABASE_URL)
    monkeypatch.setenv(setting_name, setting_value)

    with pytest.raises(ValidationError):
        JobConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("setting_name", "setting_value"),
    (
        ("TEXTIFY_WORKER_CONCURRENCY", "0"),
        ("TEXTIFY_WORKER_CONCURRENCY", "-1"),
        ("TEXTIFY_RECONCILER_INTERVAL_SECONDS", "0"),
        ("TEXTIFY_RECONCILER_INTERVAL_SECONDS", "-1"),
    ),
)
def test_nonpositive_dispatch_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    setting_name: str,
    setting_value: str,
) -> None:
    """Reject nonpositive worker concurrency and reconciliation intervals."""
    monkeypatch.setenv("TEXTIFY_BROKER_URL", "redis://127.0.0.1:6379/15")
    monkeypatch.setenv(setting_name, setting_value)

    with pytest.raises(ValidationError):
        JobDispatchConfig(_env_file=None)  # type: ignore[call-arg]
