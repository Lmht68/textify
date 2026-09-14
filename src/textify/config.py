"""Application-wide environment configuration."""

from enum import StrEnum
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environments supported by the application."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class AppConfig(BaseSettings):
    """Validate settings shared by the application composition root."""

    model_config = SettingsConfigDict(
        env_prefix="TEXTIFY_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Environment = Environment.LOCAL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8182, ge=1, le=65535)
