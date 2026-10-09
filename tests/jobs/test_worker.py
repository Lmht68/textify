"""Tests for the private Celery threads worker runtime."""

import asyncio
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from celery.signals import worker_shutting_down
from sqlalchemy.ext.asyncio import AsyncEngine

from textify.jobs.celery_app import create_celery_app
from textify.jobs.config import JobConfig, JobDispatchConfig
from textify.jobs.contracts import TranscriptionJobExecutionRepository
from textify.jobs.types import JobDispatch
from textify.jobs.worker import (
    _DispatchProcessor,
    _ExecutionResources,
    _ShutdownExecutor,
)
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

    def __init__(self, lifecycle_events: list[str] | None = None) -> None:
        """Initialize shutdown observation state.

        Args:
            lifecycle_events: Optional ordered lifecycle event sink.
        """
        self.shutdown_calls = 0
        self._lifecycle_events = lifecycle_events

    async def shutdown(self) -> None:
        """Record one worker lifecycle shutdown."""
        self.shutdown_calls += 1
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("executor_shutdown")


class RecordingClaimManager:
    """Record claim-manager lifecycle calls on the worker-owned event loop."""

    def __init__(self, lifecycle_events: list[str] | None = None) -> None:
        """Initialize lifecycle observations.

        Args:
            lifecycle_events: Optional ordered lifecycle event sink.
        """
        self.start_calls = 0
        self.shutdown_calls = 0
        self.admission_closed = threading.Event()
        self._lifecycle_events = lifecycle_events

    async def start(self) -> None:
        """Record heartbeat monitor startup."""
        self.start_calls += 1
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("claim_manager_start")

    def stop_accepting(self) -> None:
        """Record admission closure."""
        self.admission_closed.set()
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("claim_manager_stop_accepting")

    async def shutdown(self) -> None:
        """Record claim-manager drain completion."""
        self.shutdown_calls += 1
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("claim_manager_shutdown")


class RecordingEngine:
    """Record disposal on the worker-owned event loop."""

    def __init__(self, lifecycle_events: list[str] | None = None) -> None:
        """Initialize disposal observations.

        Args:
            lifecycle_events: Optional ordered lifecycle event sink.
        """
        self.dispose_calls = 0
        self._lifecycle_events = lifecycle_events

    async def dispose(self) -> None:
        """Record one connection-pool disposal."""
        self.dispose_calls += 1
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("engine_dispose")


class RecordingGpuOwnership:
    """Record GPU ownership lifecycle calls without taking external locks."""

    def __init__(self, lifecycle_events: list[str] | None = None) -> None:
        """Initialize unacquired ownership state.

        Args:
            lifecycle_events: Optional ordered lifecycle event sink.
        """
        self.acquire_calls = 0
        self.release_calls = 0
        self._ready = False
        self._lifecycle_events = lifecycle_events

    @property
    def is_ready(self) -> bool:
        """Return whether model ownership currently admits work."""
        return self._ready

    async def acquire(self) -> None:
        """Record host and database lock acquisition."""
        self.acquire_calls += 1
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("gpu_acquire")

    def mark_model_ready(self) -> None:
        """Record model readiness after deferred construction."""
        self._ready = True
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("gpu_model_ready")

    def mark_model_not_ready(self) -> None:
        """Prevent new work while retaining ownership for shutdown."""
        self._ready = False
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("gpu_model_not_ready")

    async def can_claim(self) -> bool:
        """Return current fake ownership readiness."""
        return self._ready

    async def release(self) -> None:
        """Record final GPU ownership release."""
        self.release_calls += 1
        self._ready = False
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("gpu_release")


class RecordingServiceHeartbeat:
    """Record worker service-readiness lifecycle operations."""

    def __init__(
        self,
        lifecycle_events: list[str] | None = None,
        start_error: RuntimeError | None = None,
    ) -> None:
        """Initialize optional lifecycle logging and startup failure."""
        self._lifecycle_events = lifecycle_events
        self._start_error = start_error
        self.start_calls = 0
        self.stop_calls = 0
        self.stopped = threading.Event()

    async def start(self) -> None:
        """Record a first readiness heartbeat or simulate a storage failure."""
        self.start_calls += 1
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("heartbeat_start")
        if self._start_error is not None:
            raise self._start_error

    async def pulse(self) -> None:
        """Reject unexpected reconciler-style worker heartbeat pulses."""
        raise AssertionError("Worker readiness refreshes must be periodic.")

    async def stop(self) -> None:
        """Record idempotent heartbeat removal."""
        self.stop_calls += 1
        self.stopped.set()
        if self._lifecycle_events is not None:
            self._lifecycle_events.append("heartbeat_stop")


def _execution_resources(
    processor: _DispatchProcessor,
    executor: _ShutdownExecutor,
    heartbeat: RecordingServiceHeartbeat | None = None,
) -> _ExecutionResources:
    """Build the worker's deferred private resource bundle for one test."""
    return _ExecutionResources(
        processor,
        executor,
        heartbeat or RecordingServiceHeartbeat(),
    )


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


def test_worker_cli_emits_safe_lifecycle_logs() -> None:
    """Report worker startup, dispatch handling, and shutdown without private data."""
    child_program = textwrap.dedent(
        """
        from types import SimpleNamespace
        from uuid import UUID

        from textify.jobs import worker
        from textify.jobs.celery_app import TRANSCRIPTION_TASK
        from textify.jobs.types import JobDispatch


        class ClaimManager:
            async def start(self):
                pass

            def stop_accepting(self):
                pass

            async def shutdown(self):
                pass


        class Engine:
            async def dispose(self):
                pass


        class Executor:
            async def shutdown(self):
                pass


        class GpuOwnership:
            is_ready = False

            async def acquire(self):
                pass

            def mark_model_ready(self):
                self.is_ready = True

            def mark_model_not_ready(self):
                self.is_ready = False

            async def release(self):
                pass


        class Heartbeat:
            async def start(self):
                pass

            async def stop(self):
                pass


        class Processor:
            async def process(self, dispatch):
                pass


        class CeleryApp:
            def __init__(self):
                self.tasks = {}

            def task(self, **options):
                def register(task):
                    self.tasks[options["name"]] = task
                    return task

                return register

            def worker_main(self, arguments):
                self.tasks[TRANSCRIPTION_TASK](
                    JobDispatch(
                        1,
                        UUID("00000000-0000-4000-8000-000000000001"),
                    ).to_payload()
                )


        async def create_worker_runtime(*_arguments):
            return worker.TranscriptionWorkerRuntime(
                lambda: worker._ExecutionResources(
                    Processor(),
                    Executor(),
                    Heartbeat(),
                ),
                ClaimManager(),
                GpuOwnership(),
                Engine(),
                verify_database=False,
            )


        worker.AppConfig = lambda: SimpleNamespace(log_level="INFO")
        worker.JobConfig = lambda: SimpleNamespace(
            database_url="postgresql+asyncpg://test:test@localhost/test",
        )
        worker.JobDispatchConfig = lambda: SimpleNamespace(
            broker_url="memory://",
            worker_concurrency=1,
        )
        worker.CacheConfig = lambda: SimpleNamespace()
        worker.TranscriptionConfig = lambda: SimpleNamespace()
        worker.validate_redis_role_isolation = lambda *_arguments: None
        worker.create_application_engine = lambda _database_url: Engine()
        worker._create_worker_runtime = create_worker_runtime
        worker.create_celery_app = lambda _config: CeleryApp()
        worker.main()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", child_program],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    for message in (
        "starting transcription worker",
        "transcription worker is ready",
        "processing transcription job dispatch",
        "finished processing transcription job dispatch",
        "stopping transcription worker",
    ):
        assert message in completed.stderr
    assert "00000000-0000-4000-8000-000000000001" not in completed.stderr


def test_worker_runtime_processes_only_valid_payloads_on_one_loop() -> None:
    """Reject malformed delivery without provider work and reuse one loop for valid work."""
    from textify.jobs.worker import TranscriptionWorkerRuntime

    processor = RecordingProcessor()
    executor = ShutdownExecutor()
    claim_manager = RecordingClaimManager()
    engine = RecordingEngine()
    gpu_ownership = RecordingGpuOwnership()
    runtime = TranscriptionWorkerRuntime(
        lambda: _execution_resources(processor, executor),
        claim_manager,
        gpu_ownership,
        engine,
    )
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
    assert claim_manager.start_calls == 1
    assert claim_manager.shutdown_calls == 1
    assert executor.shutdown_calls == 1
    assert engine.dispose_calls == 1


def test_worker_runtime_starts_and_stops_resources_in_gpu_safe_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquire ownership before model load and release it after native cleanup."""
    from textify.jobs import worker

    lifecycle_events: list[str] = []

    async def verify_database(_engine: AsyncEngine) -> None:
        """Record the initial PostgreSQL verification."""
        lifecycle_events.append("database_verify")

    def build_resources() -> _ExecutionResources:
        """Record deferred model-backed processor construction."""
        lifecycle_events.append("resources_factory")
        return _execution_resources(
            RecordingProcessor(),
            ShutdownExecutor(lifecycle_events),
            RecordingServiceHeartbeat(lifecycle_events),
        )

    monkeypatch.setattr(worker, "verify_application_database", verify_database)
    runtime = worker.TranscriptionWorkerRuntime(
        build_resources,
        RecordingClaimManager(lifecycle_events),
        RecordingGpuOwnership(lifecycle_events),
        RecordingEngine(lifecycle_events),
        verify_database=True,
    )

    runtime.start()
    runtime.shutdown()

    assert lifecycle_events == [
        "database_verify",
        "gpu_acquire",
        "resources_factory",
        "gpu_model_ready",
        "claim_manager_start",
        "heartbeat_start",
        "claim_manager_stop_accepting",
        "heartbeat_stop",
        "gpu_model_not_ready",
        "claim_manager_shutdown",
        "executor_shutdown",
        "gpu_release",
        "engine_dispose",
    ]


def test_worker_shutdown_signal_stops_claiming_without_interrupting_work() -> None:
    """Close admission on Celery warm shutdown without cancelling the worker loop."""
    from textify.jobs.worker import (
        TranscriptionWorkerRuntime,
        _connect_worker_shutdown_signal,
        _disconnect_worker_shutdown_signal,
    )

    claim_manager = RecordingClaimManager()
    heartbeat = RecordingServiceHeartbeat()
    runtime = TranscriptionWorkerRuntime(
        lambda: _execution_resources(
            RecordingProcessor(),
            ShutdownExecutor(),
            heartbeat,
        ),
        claim_manager,
        RecordingGpuOwnership(),
        RecordingEngine(),
    )
    runtime.start()
    receiver = _connect_worker_shutdown_signal(runtime)
    try:
        worker_shutting_down.send(
            sender="test-transcription-worker",
            sig="SIGTERM",
            how="Warm",
            exitcode=0,
        )

        assert claim_manager.admission_closed.wait(timeout=1)
        assert heartbeat.stopped.wait(timeout=1)
    finally:
        _disconnect_worker_shutdown_signal(receiver)
        runtime.shutdown()

    assert claim_manager.shutdown_calls == 1


def test_worker_runtime_shutdown_drains_claims_before_owned_resources() -> None:
    """Retain native work and database ownership until claim draining completes."""
    from textify.jobs.worker import TranscriptionWorkerRuntime

    lifecycle_events: list[str] = []
    runtime = TranscriptionWorkerRuntime(
        lambda: _execution_resources(
            RecordingProcessor(),
            ShutdownExecutor(lifecycle_events),
            RecordingServiceHeartbeat(lifecycle_events),
        ),
        RecordingClaimManager(lifecycle_events),
        RecordingGpuOwnership(lifecycle_events),
        RecordingEngine(lifecycle_events),
    )
    runtime.start()
    lifecycle_events.clear()

    runtime.shutdown()

    assert lifecycle_events == [
        "claim_manager_stop_accepting",
        "heartbeat_stop",
        "gpu_model_not_ready",
        "claim_manager_shutdown",
        "executor_shutdown",
        "gpu_release",
        "engine_dispose",
    ]


def test_worker_runtime_rejects_unusable_native_parallelism_before_lock_or_model_work(
    tmp_path: Path,
) -> None:
    """Reject a worker that cannot drive each configured native model worker."""
    from textify.jobs import worker

    adapter_factory_calls = 0

    def adapter_factory(_config: TranscriptionConfig) -> TranscriptionAdapters:
        """Record unexpected deferred model construction."""
        nonlocal adapter_factory_calls
        adapter_factory_calls += 1
        raise AssertionError("Invalid concurrency must precede model construction.")

    configuration = TranscriptionConfig(
        gpu_identity="test-gpu",
        gpu_lock_directory=tmp_path / "gpu-locks",
        temporary_media_root=tmp_path / "media",
        transcription_concurrency=3,
        max_media_bytes=1,
    )
    dispatch_config = JobDispatchConfig(
        broker_url="redis://127.0.0.1:6379/15",  # type: ignore[arg-type]
        worker_concurrency=2,
        _env_file=None,  # type: ignore[call-arg]
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"^TEXTIFY_WORKER_CONCURRENCY must be at least "
            r"TEXTIFY_TRANSCRIPTION_CONCURRENCY\.$"
        ),
    ):
        worker.TranscriptionWorkerRuntime.from_repository(
            cast(TranscriptionJobExecutionRepository, object()),
            cast(AsyncEngine, object()),
            configuration,
            dispatch_config,
            adapters_factory=adapter_factory,
        )

    assert adapter_factory_calls == 0
    assert not configuration.gpu_lock_directory.exists()


def test_worker_runtime_does_not_load_model_or_start_claims_after_lock_conflict() -> (
    None
):
    """Reject startup before deferred model construction when ownership is unavailable."""
    from textify.jobs.gpu_ownership import GpuOwnershipConflictError
    from textify.jobs.worker import TranscriptionWorkerRuntime

    lifecycle_events: list[str] = []
    claim_manager = RecordingClaimManager(lifecycle_events)
    gpu_ownership = RecordingGpuOwnership(lifecycle_events)
    engine = RecordingEngine(lifecycle_events)
    resources_factory_calls = 0
    heartbeat = RecordingServiceHeartbeat(lifecycle_events)

    async def reject_ownership() -> None:
        """Simulate an already-owned physical GPU."""
        gpu_ownership.acquire_calls += 1
        lifecycle_events.append("gpu_acquire")
        raise GpuOwnershipConflictError(
            "GPU ownership is already held for the configured identity."
        )

    def build_resources() -> _ExecutionResources:
        """Fail if lock conflict allows deferred model construction."""
        nonlocal resources_factory_calls
        resources_factory_calls += 1
        return _execution_resources(
            RecordingProcessor(),
            ShutdownExecutor(lifecycle_events),
            heartbeat,
        )

    gpu_ownership.acquire = reject_ownership  # type: ignore[method-assign]
    runtime = TranscriptionWorkerRuntime(
        build_resources,
        claim_manager,
        gpu_ownership,
        engine,
    )

    with pytest.raises(GpuOwnershipConflictError):
        runtime.start()

    assert resources_factory_calls == 0
    assert claim_manager.start_calls == 0
    assert gpu_ownership.release_calls == 1
    assert engine.dispose_calls == 1
    assert heartbeat.start_calls == 0


def test_worker_runtime_rolls_back_claims_and_heartbeat_start_failure() -> None:
    """Delete partial readiness before draining claims and owned worker resources."""
    from textify.jobs.worker import TranscriptionWorkerRuntime

    lifecycle_events: list[str] = []
    heartbeat = RecordingServiceHeartbeat(
        lifecycle_events,
        start_error=RuntimeError("heartbeat write failed"),
    )
    runtime = TranscriptionWorkerRuntime(
        lambda: _execution_resources(
            RecordingProcessor(),
            ShutdownExecutor(lifecycle_events),
            heartbeat,
        ),
        RecordingClaimManager(lifecycle_events),
        RecordingGpuOwnership(lifecycle_events),
        RecordingEngine(lifecycle_events),
    )

    with pytest.raises(RuntimeError, match=r"^heartbeat write failed$"):
        runtime.start()

    assert lifecycle_events == [
        "gpu_acquire",
        "gpu_model_ready",
        "claim_manager_start",
        "heartbeat_start",
        "heartbeat_stop",
        "gpu_model_not_ready",
        "claim_manager_stop_accepting",
        "claim_manager_shutdown",
        "executor_shutdown",
        "gpu_release",
        "engine_dispose",
    ]


def test_worker_runtime_rolls_back_ownership_when_resource_construction_fails() -> None:
    """Release startup ownership without opening claim admission after a model failure."""
    from textify.jobs.worker import TranscriptionWorkerRuntime

    lifecycle_events: list[str] = []
    claim_manager = RecordingClaimManager(lifecycle_events)
    gpu_ownership = RecordingGpuOwnership(lifecycle_events)
    engine = RecordingEngine(lifecycle_events)

    def build_resources() -> _ExecutionResources:
        """Simulate deferred adapter or model construction failure."""
        raise RuntimeError("model initialization failed")

    runtime = TranscriptionWorkerRuntime(
        build_resources,
        claim_manager,
        gpu_ownership,
        engine,
    )

    with pytest.raises(RuntimeError, match=r"^model initialization failed$"):
        runtime.start()

    assert claim_manager.start_calls == 0
    assert gpu_ownership.release_calls == 1
    assert engine.dispose_calls == 1
    assert lifecycle_events == [
        "gpu_acquire",
        "gpu_model_not_ready",
        "gpu_release",
        "engine_dispose",
    ]


def test_worker_runtime_rejects_payloads_before_runtime_or_ownership_readiness() -> (
    None
):
    """Do not schedule valid private payloads until both readiness gates are true."""
    from textify.jobs.worker import TranscriptionWorkerRuntime

    processor = RecordingProcessor()
    gpu_ownership = RecordingGpuOwnership()
    runtime = TranscriptionWorkerRuntime(
        lambda: _execution_resources(processor, ShutdownExecutor()),
        RecordingClaimManager(),
        gpu_ownership,
        RecordingEngine(),
    )
    payload = JobDispatch(1, uuid4()).to_payload()

    runtime.process_payload(payload)
    runtime.start()
    try:
        gpu_ownership.mark_model_not_ready()
        runtime.process_payload(payload)
    finally:
        runtime.shutdown()

    assert processor.dispatches == []


def test_registered_task_is_unbound_and_forwards_one_json_payload() -> None:
    """Register the fixed private task without binding Celery task state to it."""
    from textify.jobs.worker import register_transcription_task

    celery_app = create_celery_app(
        JobDispatchConfig(
            broker_url="redis://127.0.0.1:6379/15",  # type: ignore[arg-type]
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
        "textify.jobs.worker.shutil.disk_usage",
        lambda _root: SimpleNamespace(free=699),
    )
    configuration = TranscriptionConfig(
        gpu_identity="test-gpu",
        gpu_lock_directory=tmp_path / "gpu-locks",
        temporary_media_root=tmp_path / "media",
        transcription_concurrency=3,
        max_media_bytes=100,
    )
    dispatch_config = JobDispatchConfig(
        broker_url="redis://127.0.0.1:6379/15",  # type: ignore[arg-type]
        worker_concurrency=3,
        _env_file=None,  # type: ignore[call-arg]
    )

    with pytest.raises(
        RuntimeError,
        match=r"^Temporary media root requires 700 available bytes; 699 are available\.$",
    ):
        worker.TranscriptionWorkerRuntime.from_repository(
            cast(TranscriptionJobExecutionRepository, object()),
            cast(AsyncEngine, object()),
            configuration,
            dispatch_config,
            adapters_factory=adapter_factory,
        )

    assert (tmp_path / "media").is_dir()
    assert adapter_factory_calls == 0


class DatabaseProbeProcessor:
    """Run one real repository query and record whether it completed."""

    def __init__(self) -> None:
        """Initialize the deferred repository and completion signal."""
        self.repository: object | None = None
        self.completed = threading.Event()

    async def process(self, dispatch: JobDispatch) -> None:
        """Read an absent public job through the runtime-owned database loop.

        Args:
            dispatch: Valid private task payload, unused by this database probe.
        """
        from textify.jobs.repository import PostgresTranscriptionJobRepository

        del dispatch
        repository = self.repository
        assert isinstance(repository, PostgresTranscriptionJobRepository)
        await repository.get_job(uuid4())
        self.completed.set()


@pytest.mark.asyncio
async def test_worker_runtime_uses_its_own_loop_for_database_connections(
    migrated_postgresql_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep pooled asyncpg connections off the closed startup event loop."""
    from textify.jobs import worker
    from textify.jobs.database import create_application_engine

    probe = DatabaseProbeProcessor()
    executor = ShutdownExecutor()
    engine = create_application_engine(migrated_postgresql_database_url)
    runtime: worker.TranscriptionWorkerRuntime | None = None

    claim_manager = RecordingClaimManager()
    gpu_ownership = RecordingGpuOwnership()

    def build_runtime(
        cls: type[worker.TranscriptionWorkerRuntime],
        repository: TranscriptionJobExecutionRepository,
        worker_engine: AsyncEngine,
        configuration: TranscriptionConfig,
        dispatch_config: JobDispatchConfig,
    ) -> worker.TranscriptionWorkerRuntime:
        """Build a deterministic runtime after production startup verification."""
        del cls, configuration, dispatch_config
        probe.repository = repository
        return worker.TranscriptionWorkerRuntime(
            lambda: _execution_resources(probe, executor),
            claim_manager,
            gpu_ownership,
            worker_engine,
        )

    monkeypatch.setattr(
        worker.TranscriptionWorkerRuntime,
        "from_repository",
        classmethod(build_runtime),
    )
    try:
        runtime = await asyncio.to_thread(
            lambda: asyncio.run(
                worker._create_worker_runtime(
                    engine,
                    JobConfig(
                        database_url=migrated_postgresql_database_url,  # type: ignore[arg-type]
                    ),
                    JobDispatchConfig(
                        broker_url="redis://127.0.0.1:6379/0",  # type: ignore[arg-type]
                        worker_concurrency=1,
                        _env_file=None,  # type: ignore[call-arg]
                    ),
                    TranscriptionConfig(
                        gpu_identity="test-gpu",
                        gpu_lock_directory=tmp_path / "gpu-locks",
                        temporary_media_root=tmp_path,
                        transcription_concurrency=1,
                        max_media_bytes=1024,
                    ),
                )
            )
        )
        assert runtime is not None
        runtime.start()
        runtime.process_payload(JobDispatch(1, uuid4()).to_payload())
    finally:
        if runtime is None:
            await engine.dispose()
        else:
            runtime.shutdown()

    assert probe.completed.is_set()
