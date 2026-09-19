"""Durable Transcription Job environment configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class JobConfig(BaseSettings):
    """Validate settings used by durable Transcription Job handling."""

    model_config = SettingsConfigDict(
        env_prefix="TEXTIFY_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    database_path: Path | None = None
    job_worker_count: int = Field(default=4, gt=0)
    max_outstanding_jobs: int = Field(default=8, gt=0)
    job_queue_timeout_seconds: int = Field(default=20, gt=0)
