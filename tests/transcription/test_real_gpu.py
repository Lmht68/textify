"""Opt-in proof that one Faster-Whisper model overlaps two CUDA transcriptions."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from faster_whisper import WhisperModel

from textify.transcription import acquisition
from textify.transcription.config import TranscriptionConfig
from textify.transcription.types import Transcript, TranscriptMethod

logger = logging.getLogger(__name__)

_AUDIO_A = os.environ.get("TEXTIFY_GPU_TEST_AUDIO_A")
_AUDIO_B = os.environ.get("TEXTIFY_GPU_TEST_AUDIO_B")
_DEVICE_INDEX = os.environ.get("TEXTIFY_GPU_TEST_DEVICE_INDEX", "0")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _AUDIO_A and not _AUDIO_B,
        reason=(
            "TEXTIFY_GPU_TEST_AUDIO_A and TEXTIFY_GPU_TEST_AUDIO_B are required "
            "for the real GPU proof."
        ),
    ),
]


def _require_device_index() -> int:
    """Return the configured nonnegative nvidia-smi device index.

    Returns:
        CUDA device index used by Faster-Whisper and nvidia-smi.

    Raises:
        AssertionError: If the configured device index is not a nonnegative integer.
    """
    try:
        device_index = int(_DEVICE_INDEX)
    except ValueError as exc:
        pytest.fail("TEXTIFY_GPU_TEST_DEVICE_INDEX must be a nonnegative integer.")
        raise AssertionError from exc
    if device_index < 0:
        pytest.fail("TEXTIFY_GPU_TEST_DEVICE_INDEX must be a nonnegative integer.")
        raise AssertionError("pytest.fail must stop the GPU test.")
    return device_index


def _require_audio_inputs() -> tuple[Path, Path]:
    """Validate the opt-in pair of distinct local audio inputs.

    Returns:
        Canonical paths for the two configured source files.

    Raises:
        AssertionError: If configuration is partial, inaccessible, or duplicates one file.
    """
    audio_a_value = _AUDIO_A
    audio_b_value = _AUDIO_B
    if audio_a_value is None or audio_b_value is None:
        pytest.fail(
            "TEXTIFY_GPU_TEST_AUDIO_A and TEXTIFY_GPU_TEST_AUDIO_B must both be set."
        )
        raise AssertionError("pytest.fail must stop the GPU test.")
    audio_a = Path(audio_a_value).resolve()
    audio_b = Path(audio_b_value).resolve()
    if not audio_a.is_file() or not audio_b.is_file():
        pytest.fail("GPU test audio settings must both name existing files.")
        raise AssertionError("pytest.fail must stop the GPU test.")
    if audio_a == audio_b:
        pytest.fail("GPU test audio settings must name distinct files.")
        raise AssertionError("pytest.fail must stop the GPU test.")
    return audio_a, audio_b


def _sample_gpu_memory(device_index: int) -> int:
    """Read one physical GPU's allocated memory in MiB.

    Args:
        device_index: NVIDIA device index passed to nvidia-smi.

    Returns:
        Current allocated GPU memory in MiB.

    Raises:
        AssertionError: If nvidia-smi does not provide one nonnegative integer sample.
    """
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(device_index),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    samples = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(samples) != 1:
        pytest.fail("nvidia-smi did not return one GPU memory sample.")
        raise AssertionError("pytest.fail must stop the GPU test.")
    try:
        memory_mib = int(samples[0])
    except ValueError as exc:
        pytest.fail("nvidia-smi returned an invalid GPU memory sample.")
        raise AssertionError from exc
    if memory_mib < 0:
        pytest.fail("nvidia-smi returned an invalid GPU memory sample.")
        raise AssertionError("pytest.fail must stop the GPU test.")
    return memory_mib


@pytest.mark.asyncio
async def test_one_model_overlaps_two_real_gpu_transcriptions(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    tmp_path: Path,
) -> None:
    """Run two distinct local inputs concurrently through one CUDA model load."""
    audio_a, audio_b = _require_audio_inputs()
    device_index = _require_device_index()
    if shutil.which("nvidia-smi") is None:
        pytest.fail("nvidia-smi is required for the real GPU proof.")

    construction_calls = 0
    whisper_model = cast(Callable[..., object], WhisperModel)

    def record_model(
        model_name: str,
        **model_options: str | int,
    ) -> object:
        """Record one production model construction without changing its arguments."""
        nonlocal construction_calls
        construction_calls += 1
        return whisper_model(model_name, **model_options)

    monkeypatch.setattr(acquisition, "WhisperModel", record_model)
    settings = TranscriptionConfig(
        gpu_identity="real-gpu-proof",
        gpu_lock_directory=tmp_path / "gpu-locks",
        temporary_media_root=tmp_path / "media",
        whisper_model="large-v3-turbo",
        whisper_revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        whisper_device="cuda",
        whisper_device_index=device_index,
        whisper_compute_type="float16",
        transcription_concurrency=2,
        _env_file=None,  # type: ignore[call-arg]
    )
    adapter = acquisition.load_whisper_transcriber(settings)
    baseline_memory_mib = _sample_gpu_memory(device_index)
    call_intervals: dict[Path, tuple[float, float]] = {}
    interval_lock = threading.Lock()
    start_barrier = threading.Barrier(2)

    def transcribe(audio_path: Path) -> Transcript:
        """Record one complete production-adapter transcription interval."""
        start_barrier.wait(timeout=60)
        started_at = time.monotonic()
        transcript = adapter.transcribe(audio_path)
        finished_at = time.monotonic()
        with interval_lock:
            call_intervals[audio_path] = (started_at, finished_at)
        return transcript

    first_task = asyncio.create_task(asyncio.to_thread(transcribe, audio_a))
    second_task = asyncio.create_task(asyncio.to_thread(transcribe, audio_b))
    memory_samples_mib: list[int] = []
    try:
        while not first_task.done() or not second_task.done():
            memory_samples_mib.append(
                await asyncio.to_thread(_sample_gpu_memory, device_index)
            )
        first_result, second_result = await asyncio.gather(first_task, second_task)
    finally:
        await asyncio.to_thread(adapter.shutdown)

    assert memory_samples_mib
    peak_memory_mib = max(memory_samples_mib)
    first_interval = call_intervals[audio_a]
    second_interval = call_intervals[audio_b]
    gpu_overlap_seconds = max(
        0.0,
        min(first_interval[1], second_interval[1])
        - max(first_interval[0], second_interval[0]),
    )
    record_property("gpu_memory_baseline_mib", baseline_memory_mib)
    record_property("gpu_memory_peak_mib", peak_memory_mib)
    record_property("gpu_overlap_seconds", gpu_overlap_seconds)
    logger.info(
        "real GPU concurrency observation model=%s revision=%s num_workers=2 "
        "gpu_overlap_seconds=%.6f gpu_memory_baseline_mib=%d "
        "gpu_memory_peak_mib=%d",
        settings.whisper_model,
        settings.whisper_revision,
        gpu_overlap_seconds,
        baseline_memory_mib,
        peak_memory_mib,
    )

    assert construction_calls == 1
    assert first_result.method is TranscriptMethod.FASTER_WHISPER
    assert second_result.method is TranscriptMethod.FASTER_WHISPER
    assert gpu_overlap_seconds > 0
    assert peak_memory_mib > 0
