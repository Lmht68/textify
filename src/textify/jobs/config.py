"""Durable Transcription Job environment configuration."""

from typing import Self

from pydantic import Field, PostgresDsn, RedisDsn, field_validator, model_validator
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

    max_outstanding_jobs: int = Field(default=8, gt=0)
    job_queue_timeout_seconds: int = Field(default=20, gt=0)
    job_retention_seconds: int = Field(default=86_400, gt=0)


class JobDispatchConfig(BaseSettings):
    """Validate settings used to deliver and process Job Dispatches."""

    model_config = SettingsConfigDict(
        env_prefix="TEXTIFY_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    broker_url: RedisDsn
    worker_concurrency: int = Field(default=4, gt=0)
    reconciler_interval_seconds: float = Field(default=1.0, gt=0)
    attempt_heartbeat_seconds: float = Field(default=5.0, gt=0)
    attempt_lease_seconds: float = Field(default=30.0, gt=0)
    service_heartbeat_ttl_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def require_heartbeat_lease_ratio(self) -> Self:
        """Require a lease that tolerates two complete missed heartbeats.

        Returns:
            Validated dispatch configuration.

        Raises:
            ValueError: If the claim lease is less than three heartbeat intervals.
        """
        if self.attempt_lease_seconds < self.attempt_heartbeat_seconds * 3:
            raise ValueError(
                "TEXTIFY_ATTEMPT_LEASE_SECONDS must be at least three times "
                "TEXTIFY_ATTEMPT_HEARTBEAT_SECONDS."
            )
        return self


class CacheConfig(BaseSettings):
    """Validate optional cache endpoint isolation settings."""

    model_config = SettingsConfigDict(
        env_prefix="TEXTIFY_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    cache_url: RedisDsn | None = None


def validate_redis_role_isolation(
    dispatch_config: JobDispatchConfig,
    cache_config: CacheConfig,
) -> None:
    """Reject broker and cache URLs that identify one Redis server endpoint.

    Args:
        dispatch_config: Required Celery broker connection settings.
        cache_config: Optional future cache connection settings.

    Raises:
        ValueError: If both URLs resolve to the same normalized host and port.
    """
    cache_url = cache_config.cache_url
    if cache_url is None:
        return
    if _redis_server_endpoint(dispatch_config.broker_url) == _redis_server_endpoint(
        cache_url
    ):
        raise ValueError(
            "TEXTIFY_BROKER_URL and TEXTIFY_CACHE_URL must identify different "
            "Redis server endpoints."
        )


def _redis_server_endpoint(redis_url: RedisDsn) -> tuple[str, int]:
    """Return Pydantic-normalized physical Redis host and effective port."""
    host = redis_url.host
    if host is None:
        raise RuntimeError("Redis URL must include a host.")
    return host, redis_url.port or 6379
