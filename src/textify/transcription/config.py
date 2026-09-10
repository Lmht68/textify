"""Transcription service environment configuration."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class TranscriptionConfig(BaseSettings):
    """Validate settings used by transcription services."""

    model_config = SettingsConfigDict(
        env_prefix="TEXTIFY_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    max_duration_seconds: int = Field(default=1800, gt=0)
    temporary_media_root: Path = Path("/tmp/textify")
    beam_size: int = Field(default=1, gt=0)
    vad_filter: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    condition_on_previous_text: bool = True
    transcription_concurrency: int = Field(default=1, gt=0)
    initial_prompt: str = (
        "This transcript may mention movies, TV shows, directors, actors, songs, "
        "albums, artists, bands, books, and authors. Transcribe proper names "
        "accurately."
    )
    hf_token: SecretStr | None = None
