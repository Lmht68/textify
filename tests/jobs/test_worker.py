"""Tests for the private Celery threads worker runtime."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from textify.jobs.celery_app import create_celery_app
from textify.jobs.config import JobDispatchConfig
from textify.jobs.contracts import TranscriptionJobExecutionRepository
from textify.jobs.types import JobDispatch
from textify.transcription.config import TranscriptionConfig
from textify.transcription.service import TranscriptionAdapters


class RecordingProcessor:
    """Record dispatches processed on the worker-owned asyncio loop."""

    def __init__(self) -> None:
        """Initialize the dispatch record."""
        self.dispatches: list[JobDispatch] = []

    async def process(self, dispatch: JobDispatch) -> None:
        """Record one dispatch received by the long-lived loop.

        Args:
            dispatch: Validated private execution notification.
        """
        self.dispatches.append(dispatch)


class ShutdownExecutor:
    """Record worker-owned executor shutdown."""

    def __init__(self) -> None:
        """Initialize the shutdown observation flag."""
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        """Record one worker lifecycle shutdown."""
        self.shutdown_calls += 1


class CapturingRuntime:
    """Record task payload forwarding without starting a worker loop."""

    def __init__(self) -> None:
        """Initialize received payload records."""
        self.payloads: list[object] = []

    def process_payload(self, payload: object) -> None:
        """Record one Celery task payload.

        Args:
            payload: Deserialized Celery task payload.
        """
        self.payloads.append(payload)


def test_worker_runtime_processes_only_valid_payloads_on_one_loop() -> None:
    """Reject malformed delivery without provider work and reuse one loop for valid work."""
    from textify.jobs.worker import TranscriptionWorkerRuntime

    processor = RecordingProcessor()
    executor = ShutdownExecutor()
    runtime = TranscriptionWorkerRuntime(processor, executor)
    first_dispatch = JobDispatch(1, uuid4())
    second_dispatch = JobDispatch(2, uuid4())

    runtime.start()
    try:
        runtime.process_payload(
            {"internal_job_id": True, "execution_attempt_token": "x"}
        )
        runtime.process_payload(first_dispatch.to_payload())
        runtime.process_payload(second_dispatch.to_payload())
    finally:
        runtime.shutdown()

    assert processor.dispatches == [first_dispatch, second_dispatch]
    assert executor.shutdown_calls == 1


def test_registered_task_is_unbound_and_forwards_one_json_payload() -> None:
    """Register the fixed private task without binding Celery task state to it."""
    from textify.jobs.worker import register_transcription_task

    celery_app = create_celery_app(
        JobDispatchConfig(
            broker_url="redis://127.0.0.1:6379/15",
            _env_file=None,  # type: ignore[call-arg]
        )
    )
    runtime = CapturingRuntime()
    payload = JobDispatch(7, uuid4()).to_payload()

    task = register_transcription_task(celery_app, runtime)
    task.run(payload)

    assert runtime.payloads == [payload]


def test_worker_runtime_checks_media_capacity_before_adapter_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject an undersized worker media root before loading model adapters."""
    from textify.jobs import worker

    adapter_factory_calls = 0

    def adapter_factory(_config: TranscriptionConfig) -> TranscriptionAdapters:
        """Record any unexpected model adapter construction."""
        nonlocal adapter_factory_calls
        adapter_factory_calls += 1
        raise AssertionError("Capacity failure must precede adapter construction.")

    monkeypatch.setattr(
        worker.shutil,
        "disk_usage",
        lambda _root: SimpleNamespace(free=599),
    )
    configuration = TranscriptionConfig(
        temporary_media_root=tmp_path / "media",
        transcription_concurrency=3,
        max_media_bytes=100,
    )

    with pytest.raises(
        RuntimeError,
        match=r"^Temporary media root requires 600 available bytes; 599 are available\.$",
    ):
        worker.TranscriptionWorkerRuntime.from_repository(
            cast(TranscriptionJobExecutionRepository, object()),
            configuration,
            worker_concurrency=2,
            adapters_factory=adapter_factory,
        )

    assert (tmp_path / "media").is_dir()
    assert adapter_factory_calls == 0
