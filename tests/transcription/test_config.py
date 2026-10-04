"""Tests for transcription environment configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from textify.transcription.config import TranscriptionConfig


def test_gpu_identity_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject workers without a stable physical-GPU identity."""
    monkeypatch.delenv("TEXTIFY_GPU_IDENTITY", raising=False)

    with pytest.raises(
        ValidationError,
        match="TEXTIFY_GPU_IDENTITY must be a non-empty stable identifier.",
    ):
        TranscriptionConfig(_env_file=None)  # type: ignore[call-arg]


def test_gpu_identity_strips_surrounding_whitespace() -> None:
    """Normalize the configured physical-GPU identity before it becomes ownership."""
    configuration = TranscriptionConfig(
        gpu_identity="  NVIDIA-GPU-0001  ",
        _env_file=None,  # type: ignore[call-arg]
    )

    assert configuration.gpu_identity == "NVIDIA-GPU-0001"


@pytest.mark.parametrize("gpu_identity", ["", "   "])
def test_blank_gpu_identity_is_rejected(gpu_identity: str) -> None:
    """Reject empty stable-identity settings."""
    with pytest.raises(
        ValidationError,
        match="TEXTIFY_GPU_IDENTITY must be a non-empty stable identifier.",
    ):
        TranscriptionConfig(
            gpu_identity=gpu_identity,
            _env_file=None,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("lock_directory", [Path("gpu-locks"), Path("/")])
def test_relative_or_root_gpu_lock_directory_is_rejected(lock_directory: Path) -> None:
    """Require a dedicated absolute GPU lock namespace."""
    with pytest.raises(
        ValidationError,
        match="TEXTIFY_GPU_LOCK_DIRECTORY must be an absolute non-root directory.",
    ):
        TranscriptionConfig(
            gpu_identity="NVIDIA-GPU-0001",
            gpu_lock_directory=lock_directory,
            _env_file=None,  # type: ignore[call-arg]
        )


def test_transcription_concurrency_defaults_to_two() -> None:
    """Preserve two native permits as the default model concurrency."""
    configuration = TranscriptionConfig(
        gpu_identity="NVIDIA-GPU-0001",
        _env_file=None,  # type: ignore[call-arg]
    )

    assert configuration.transcription_concurrency == 2


def test_nonpositive_transcription_concurrency_is_rejected() -> None:
    """Reject configurations that cannot supply native permits."""
    with pytest.raises(ValidationError):
        TranscriptionConfig(
            gpu_identity="NVIDIA-GPU-0001",
            transcription_concurrency=0,
            _env_file=None,  # type: ignore[call-arg]
        )


def test_cpu_device_setting_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject CPU execution before transcription adapters are constructed."""
    monkeypatch.setenv("TEXTIFY_GPU_IDENTITY", "NVIDIA-GPU-0001")
    monkeypatch.setenv("TEXTIFY_WHISPER_DEVICE", "cpu")

    with pytest.raises(ValidationError):
        TranscriptionConfig(_env_file=None)  # type: ignore[call-arg]
