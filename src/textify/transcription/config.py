"""Transcription service environment configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TranscriptionConfig(BaseSettings):
    """Validate settings used by transcription services."""

    model_config = SettingsConfigDict(
        env_prefix="TEXTIFY_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    gpu_identity: str
    gpu_lock_directory: Path = Path("/tmp/textify/gpu-locks")

    @model_validator(mode="before")
    @classmethod
    def require_gpu_identity(cls, values: object) -> object:
        """Require and normalize a stable physical-GPU identity.

        Args:
            values: Raw configuration values before field validation.

        Returns:
            Configuration values with normalized GPU identity.

        Raises:
            ValueError: If the GPU identity is missing, non-textual, or blank.
        """
        if not isinstance(values, dict):
            return values
        gpu_identity = values.get("gpu_identity")
        if not isinstance(gpu_identity, str) or not gpu_identity.strip():
            raise ValueError(
                "TEXTIFY_GPU_IDENTITY must be a non-empty stable identifier."
            )
        values["gpu_identity"] = gpu_identity.strip()
        return values

    @field_validator("gpu_lock_directory")
    @classmethod
    def require_absolute_non_root_lock_directory(cls, value: Path) -> Path:
        """Reject lock directories that cannot identify a safe lock namespace.

        Args:
            value: Parsed GPU lock-directory path.

        Returns:
            The validated absolute non-root lock directory.

        Raises:
            ValueError: If the path is relative or is the filesystem root.
        """
        if not value.is_absolute() or value == Path(value.root):
            raise ValueError(
                "TEXTIFY_GPU_LOCK_DIRECTORY must be an absolute non-root directory."
            )
        return value

    whisper_model: str = Field(default="large-v3-turbo", min_length=1)
    whisper_revision: str = Field(
        default="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        min_length=1,
    )
    whisper_device: Literal["cuda"] = "cuda"
    whisper_device_index: int = Field(default=0, ge=0)
    whisper_compute_type: str = Field(default="float16", min_length=1)
    max_duration_seconds: int = Field(default=1800, gt=0)
    temporary_media_root: Path = Path("/tmp/textify")
    beam_size: int = Field(default=1, gt=0)
    vad_filter: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    condition_on_previous_text: bool = True
    transcription_concurrency: int = Field(default=2, gt=0)
    max_media_bytes: int = Field(default=536_870_912, gt=0)
    metadata_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        allow_inf_nan=False,
    )
    audio_download_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        allow_inf_nan=False,
    )
    transcription_timeout_seconds: float = Field(
        default=1800.0,
        gt=0,
        allow_inf_nan=False,
    )
    initial_prompt: str = ""
    hf_token: SecretStr | None = None
