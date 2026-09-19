"""Tests for transcription environment configuration."""

import pytest
from pydantic import ValidationError

from textify.transcription.config import TranscriptionConfig


def test_cpu_device_setting_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject CPU execution before transcription adapters are constructed."""
    monkeypatch.setenv("TEXTIFY_WHISPER_DEVICE", "cpu")

    with pytest.raises(ValidationError):
        TranscriptionConfig(_env_file=None)  # type: ignore[call-arg]
