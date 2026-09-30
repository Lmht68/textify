"""Durable Transcription Job environment configuration."""

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JobConfig(BaseSettings):
    """Validate settings used by durable Transcription Job handling."""

    model_config = SettingsConfigDict(
        env_prefix="TEXTIFY_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: PostgresDsn

    @field_validator("database_url")
    @classmethod
    def require_asyncpg_url(cls, value: PostgresDsn) -> PostgresDsn:
        """Require PostgreSQL's asynchronous asyncpg driver.

        Args:
            value: Parsed durable-store database URL.

        Returns:
            The validated PostgreSQL URL.

        Raises:
            ValueError: If the URL does not select the asyncpg PostgreSQL driver.
        """
        if value.scheme != "postgresql+asyncpg":
            raise ValueError(
                "TEXTIFY_DATABASE_URL must use the postgresql+asyncpg scheme."
            )
        return value

    job_worker_count: int = Field(default=4, gt=0)
    max_outstanding_jobs: int = Field(default=8, gt=0)
    job_queue_timeout_seconds: int = Field(default=20, gt=0)
    job_retention_seconds: int = Field(default=86_400, gt=0)
