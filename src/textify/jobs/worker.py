"""Dedicated Celery threads worker for Transcription Job Execution Attempts."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol, cast

from celery import Celery, Task
from celery.signals import worker_shutting_down
from sqlalchemy.ext.asyncio import AsyncEngine

from textify.errors import InternalError
from textify.jobs.celery_app import (
    TRANSCRIPTION_QUEUE,
    TRANSCRIPTION_TASK,
    create_celery_app,
)
from textify.jobs.claim_manager import ExecutionClaimManager
from textify.jobs.config import (
    CacheConfig,
    JobConfig,
    JobDispatchConfig,
    validate_redis_role_isolation,
)
from textify.jobs.contracts import (
    TranscriptionJobExecutionRepository,
    TranscriptionJobStoreUnavailableError,
)
from textify.jobs.database import create_application_engine, verify_application_database
from textify.jobs.gpu_ownership import GpuModelOwnership
from textify.jobs.processor import TranscriptionJobProcessor
from textify.jobs.readiness import (
    PostgresServiceHeartbeatRepository,
    ServiceHeartbeatReporter,
    ServiceRole,
)
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.service import utc_now
from textify.jobs.types import JobDispatch
from textify.transcription.config import TranscriptionConfig
from textify.transcription.service import (
    TranscriptionAdaptersFactory,
    TranscriptionExecutor,
    build_transcription_adapters,
)

logger = logging.getLogger(__name__)


class _DispatchProcessor(Protocol):
    """Process one validated private Job Dispatch on the worker loop."""

    async def process(self, dispatch: JobDispatch) -> None:
        """Process one claimed Execution Attempt.

        Args:
            dispatch: Validated private execution notification.
        """
        ...


class _ShutdownExecutor(Protocol):
    """Release process-lifetime provider resources during worker shutdown."""

    async def shutdown(self) -> None:
        """Wait until all owned provider work has completed."""
        ...


class _ExecutionClaimLifecycle(Protocol):
    """Coordinate manager startup, admission closure, and worker drain."""

    async def start(self) -> None:
        """Start heartbeat monitoring on the worker's event loop."""
        ...

    def stop_accepting(self) -> None:
        """Close claim admission without cancelling registered executions."""
        ...

    async def shutdown(self) -> None:
        """Drain claims before the heartbeat monitor stops."""
        ...


class _EngineOwner(Protocol):
    """Dispose database connections on the worker's event loop."""

    async def dispose(self) -> None:
        """Close all worker-owned database connections."""
        ...


class _GpuOwnershipLifecycle(Protocol):
    """Own model readiness and both physical-GPU locks for one worker."""

    @property
    def is_ready(self) -> bool:
        """Return whether locked ownership and the model are ready for work."""
        ...

    async def acquire(self) -> None:
        """Acquire physical-GPU locks before model construction."""
        ...

    def mark_model_ready(self) -> None:
        """Permit claims after the worker model is loaded."""
        ...

    def mark_model_not_ready(self) -> None:
        """Prevent claims while retaining ownership for cleanup."""
        ...

    async def can_claim(self) -> bool:
        """Probe current ownership before durable claim admission."""
        ...

    async def release(self) -> None:
        """Release both ownership locks after model teardown."""
        ...


class _TaskPayloadProcessor(Protocol):
    """Accept one deserialized private Celery task payload."""

    def process_payload(self, payload: object) -> None:
        """Process or safely reject one private task payload.

        Args:
            payload: Deserialized JSON task payload.
        """
        ...


@dataclass(frozen=True, slots=True)
class _ExecutionResources:
    """Own deferred worker processing, native cleanup, and service readiness."""

    processor: _DispatchProcessor
    executor: _ShutdownExecutor
    heartbeat: ServiceHeartbeatReporter


_ExecutionResourcesFactory = Callable[[], _ExecutionResources]


class TranscriptionWorkerRuntime:
    """Own one model and one long-lived asyncio loop for Celery task threads."""

    def __init__(
        self,
        resources_factory: _ExecutionResourcesFactory,
        claim_manager: _ExecutionClaimLifecycle,
        gpu_ownership: _GpuOwnershipLifecycle,
        engine: _EngineOwner,
        *,
        verify_database: bool = False,
    ) -> None:
        """Initialize unstarted worker runtime dependencies.

        Args:
            resources_factory: Deferred single construction of processor and executor.
            claim_manager: Worker-local claim ownership lifecycle.
            gpu_ownership: Physical-GPU ownership and readiness lifecycle.
            engine: PostgreSQL connection owner disposed on this runtime's loop.
            verify_database: Whether startup validates the owned PostgreSQL engine.
        """
        self._resources_factory = resources_factory
        self._processor: _DispatchProcessor | None = None
        self._executor: _ShutdownExecutor | None = None
        self._heartbeat: ServiceHeartbeatReporter | None = None
        self._claim_manager = claim_manager
        self._gpu_ownership = gpu_ownership
        self._engine = engine
        self._verify_database = verify_database
        self._state_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._started = False
        self._runtime_ready = False
        self._claim_monitoring_started = False
        self._heartbeat_reporting_started = False
        self._heartbeat_stop_task: asyncio.Task[None] | None = None

    @classmethod
    def from_repository(
        cls,
        repository: TranscriptionJobExecutionRepository,
        engine: AsyncEngine,
        transcription_config: TranscriptionConfig,
        dispatch_config: JobDispatchConfig,
        adapters_factory: TranscriptionAdaptersFactory = build_transcription_adapters,
    ) -> TranscriptionWorkerRuntime:
        """Validate worker resources and construct one deferred managed runtime.

        Args:
            repository: PostgreSQL claim and terminal-publication boundary.
            engine: PostgreSQL connection pool owned by the returned runtime.
            transcription_config: Process-local provider and temporary-media settings.
            dispatch_config: Celery delivery, heartbeat, and claim-lease configuration.
            adapters_factory: Process-lifetime adapter and model construction boundary.

        Returns:
            Unstarted runtime that loads its single adapter bundle only after GPU locks.

        Raises:
            RuntimeError: If media, worker capacity, or GPU ownership cannot satisfy
                the configured runtime contract.
        """
        _ensure_writable_temporary_media_root(transcription_config.temporary_media_root)
        _validate_worker_native_concurrency(
            dispatch_config.worker_concurrency,
            transcription_config.transcription_concurrency,
        )
        _validate_temporary_media_capacity(
            transcription_config,
            dispatch_config.worker_concurrency,
        )
        gpu_ownership = GpuModelOwnership(
            engine,
            transcription_config.gpu_identity,
            transcription_config.gpu_lock_directory,
        )
        claim_manager = ExecutionClaimManager(
            repository,
            max_active_claims=dispatch_config.worker_concurrency,
            heartbeat_interval=timedelta(
                seconds=dispatch_config.attempt_heartbeat_seconds
            ),
            claim_lease_duration=timedelta(
                seconds=dispatch_config.attempt_lease_seconds
            ),
            claim_ready=gpu_ownership.can_claim,
        )

        def build_execution_resources() -> _ExecutionResources:
            """Construct model resources after locks and before ready reporting."""
            executor = TranscriptionExecutor(
                adapters_factory(transcription_config),
                transcription_config,
            )
            heartbeat = ServiceHeartbeatReporter(
                PostgresServiceHeartbeatRepository(engine),
                ServiceRole.GPU_WORKER,
                refresh_interval=timedelta(
                    seconds=dispatch_config.service_heartbeat_ttl_seconds / 3
                ),
                readiness_probe=gpu_ownership.can_claim,
            )
            return _ExecutionResources(
                TranscriptionJobProcessor(repository, executor, claim_manager),
                executor,
                heartbeat,
            )

        return cls(
            build_execution_resources,
            claim_manager,
            gpu_ownership,
            engine,
            verify_database=True,
        )

    def start(self) -> None:
        """Start database verification and claim monitoring before Celery receives work.

        Raises:
            RuntimeError: If this runtime has already started.
        """
        with self._state_lock:
            if self._started:
                raise RuntimeError("Transcription worker runtime has already started.")
            loop = asyncio.new_event_loop()
            loop_ready = threading.Event()
            loop_thread = threading.Thread(
                target=_run_worker_loop,
                args=(loop, loop_ready),
                name="textify-transcription-worker-loop",
                daemon=False,
            )
            self._loop = loop
            self._loop_thread = loop_thread
            self._started = True
            loop_thread.start()
        loop_ready.wait()
        try:
            asyncio.run_coroutine_threadsafe(self._start_resources(), loop).result()
        except Exception:
            self._close_after_start_failure(loop, loop_thread)
            raise

    def process_payload(self, payload: object) -> None:
        """Parse and process one Celery JSON payload without exposing it in logs.

        Args:
            payload: Deserialized task message expected to contain one Job Dispatch.
        """
        dispatch = JobDispatch.from_payload(payload)
        if dispatch is None:
            logger.warning("rejected malformed transcription job dispatch")
            return
        with self._state_lock:
            loop = self._loop if self._started and self._runtime_ready else None
            processor = self._processor
        if loop is None or processor is None or not self._gpu_ownership.is_ready:
            logger.error("transcription worker runtime is unavailable")
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                processor.process(dispatch),
                loop,
            )
            future.result()
        except TranscriptionJobStoreUnavailableError:
            logger.error("transcription job storage is unavailable to worker")
        except Exception:  # noqa: BLE001
            logger.error(
                "transcription job task failed",
                extra={"code": InternalError.code},
            )

    def stop_claiming(self) -> None:
        """Close worker admission and remove readiness from any Celery thread."""
        with self._state_lock:
            loop = self._loop if self._started else None
        if loop is not None:
            loop.call_soon_threadsafe(self._stop_claiming_on_worker_loop)

    def shutdown(self) -> None:
        """Drain claims, native work, ownership, and connections before loop teardown."""
        with self._state_lock:
            if not self._started:
                return
            loop = self._loop
            loop_thread = self._loop_thread
            self._started = False
            self._runtime_ready = False
        assert loop is not None
        assert loop_thread is not None
        try:
            asyncio.run_coroutine_threadsafe(
                self._shutdown_resources(),
                loop,
            ).result()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join()
            with self._state_lock:
                self._loop = None
                self._loop_thread = None

    async def _start_resources(self) -> None:
        """Acquire ownership, load the model, and report readiness before work."""
        startup_complete = False
        try:
            if self._verify_database:
                await verify_application_database(cast(AsyncEngine, self._engine))
            await self._gpu_ownership.acquire()
            resources = self._resources_factory()
            self._processor = resources.processor
            self._executor = resources.executor
            self._heartbeat = resources.heartbeat
            self._gpu_ownership.mark_model_ready()
            await self._claim_manager.start()
            self._claim_monitoring_started = True
            await resources.heartbeat.start()
            self._heartbeat_reporting_started = True
            with self._state_lock:
                self._runtime_ready = True
            startup_complete = True
        finally:
            if not startup_complete:
                await self._rollback_startup()

    async def _rollback_startup(self) -> None:
        """Release partial startup resources without masking the startup exception."""
        try:
            await self._stop_service_heartbeat()
        except (OSError, RuntimeError):
            logger.error("worker service heartbeat startup cleanup failed")
        self._gpu_ownership.mark_model_not_ready()
        if self._claim_monitoring_started:
            self._claim_manager.stop_accepting()
            try:
                await self._claim_manager.shutdown()
            except (OSError, RuntimeError):
                logger.error("worker claim manager startup cleanup failed")
            self._claim_monitoring_started = False
        executor = self._executor
        self._processor = None
        self._executor = None
        self._heartbeat = None
        if executor is not None:
            try:
                await executor.shutdown()
            except Exception:  # noqa: BLE001
                logger.error("transcription executor startup cleanup failed")
        try:
            await self._gpu_ownership.release()
        except Exception:  # noqa: BLE001
            logger.error("GPU ownership startup cleanup failed")
        try:
            await self._engine.dispose()
        except Exception:  # noqa: BLE001
            logger.error("worker engine startup cleanup failed")

    async def _shutdown_resources(self) -> None:
        """Drain owned resources on the same loop that used them."""
        self._claim_manager.stop_accepting()
        await self._stop_service_heartbeat()
        self._gpu_ownership.mark_model_not_ready()
        try:
            if self._claim_monitoring_started:
                await self._claim_manager.shutdown()
                self._claim_monitoring_started = False
        finally:
            executor = self._executor
            try:
                if executor is not None:
                    await executor.shutdown()
            finally:
                self._processor = None
                self._executor = None
                self._heartbeat = None
                try:
                    await self._gpu_ownership.release()
                finally:
                    await self._engine.dispose()

    def _stop_claiming_on_worker_loop(self) -> None:
        """Close claims and asynchronously remove readiness on the worker loop."""
        self._claim_manager.stop_accepting()
        if self._heartbeat_reporting_started and self._heartbeat_stop_task is None:
            self._heartbeat_stop_task = asyncio.create_task(
                self._stop_service_heartbeat()
            )

    async def _stop_service_heartbeat(self) -> None:
        """Remove worker readiness, awaiting any warm-shutdown removal in flight."""
        scheduled_stop = self._heartbeat_stop_task
        if scheduled_stop is not None and scheduled_stop is not asyncio.current_task():
            await scheduled_stop
            self._heartbeat_stop_task = None
        heartbeat = self._heartbeat
        if heartbeat is None:
            return
        await heartbeat.stop()
        self._heartbeat_reporting_started = False

    def _close_after_start_failure(
        self,
        loop: asyncio.AbstractEventLoop,
        loop_thread: threading.Thread,
    ) -> None:
        """Stop a loop whose startup rollback already released every resource."""
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join()
        with self._state_lock:
            self._loop = None
            self._loop_thread = None
            self._started = False
            self._runtime_ready = False


def register_transcription_task(
    celery_app: Celery,
    runtime: _TaskPayloadProcessor,
) -> Task:
    """Register the one unbound private Celery task for this worker process.

    Args:
        celery_app: Configured private transcription Celery application.
        runtime: Started worker runtime that owns the long-lived asyncio loop.
    Returns:
        Registered unbound private Celery task bound to ``runtime``.

    """
    celery_app.tasks.pop(TRANSCRIPTION_TASK, None)

    @celery_app.task(  # type: ignore[untyped-decorator]
        name=TRANSCRIPTION_TASK,
        bind=False,
        ignore_result=True,
    )
    def process_transcription(payload: object) -> None:
        """Forward one Celery payload to the worker-owned async runtime.

        Args:
            payload: Deserialized JSON task payload.
        """
        runtime.process_payload(payload)

    return process_transcription


def _run_worker_loop(
    loop: asyncio.AbstractEventLoop,
    loop_ready: threading.Event,
) -> None:
    """Run and drain one worker-owned asyncio loop.

    Args:
        loop: Dedicated event loop that owns processor coroutine execution.
        loop_ready: Event set after the loop becomes ready for thread-safe submission.
    """
    asyncio.set_event_loop(loop)
    loop_ready.set()
    try:
        loop.run_forever()
    finally:
        pending_tasks = asyncio.all_tasks(loop)
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            loop.run_until_complete(
                asyncio.gather(*pending_tasks, return_exceptions=True)
            )
        loop.close()


def _ensure_writable_temporary_media_root(root: Path) -> None:
    """Create and verify the worker temporary-media root is writable.

    Args:
        root: Directory reserved for worker-owned request media.

    Raises:
        RuntimeError: If the directory cannot be created or used for a temporary file.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root):
            pass
    except OSError as exc:
        raise RuntimeError(
            "Temporary media root must be a writable directory."
        ) from exc


def _validate_worker_native_concurrency(
    worker_concurrency: int,
    transcription_concurrency: int,
) -> None:
    """Require enough Celery threads to drive every configured model worker.

    Args:
        worker_concurrency: Celery threads available to this worker process.
        transcription_concurrency: Faster-Whisper native worker and permit count.

    Raises:
        RuntimeError: If Celery cannot drive every configured native worker.
    """
    if worker_concurrency < transcription_concurrency:
        raise RuntimeError(
            "TEXTIFY_WORKER_CONCURRENCY must be at least "
            "TEXTIFY_TRANSCRIPTION_CONCURRENCY."
        )


def _validate_temporary_media_capacity(
    transcription_config: TranscriptionConfig,
    worker_concurrency: int,
) -> None:
    """Reject worker startup when temporary media cannot hold its configured quota.

    Args:
        transcription_config: Provider media and native concurrency configuration.
        worker_concurrency: Concurrent Celery task threads for this worker process.

    Raises:
        RuntimeError: If disk capacity cannot be determined or is insufficient.
    """
    required_bytes = (
        worker_concurrency + transcription_config.transcription_concurrency + 1
    ) * transcription_config.max_media_bytes
    try:
        available_media_bytes = shutil.disk_usage(
            transcription_config.temporary_media_root
        ).free
    except OSError as exc:
        raise RuntimeError(
            "Unable to determine temporary media root capacity."
        ) from exc
    if available_media_bytes < required_bytes:
        raise RuntimeError(
            "Temporary media root requires "
            f"{required_bytes} available bytes; "
            f"{available_media_bytes} are available."
        )


async def _create_worker_runtime(
    engine: AsyncEngine,
    job_config: JobConfig,
    dispatch_config: JobDispatchConfig,
    transcription_config: TranscriptionConfig,
) -> TranscriptionWorkerRuntime:
    """Create the worker's one process-lifetime runtime.

    Args:
        engine: Application PostgreSQL engine owned by the worker process.
        job_config: Durable lifecycle configuration.
        dispatch_config: Celery worker delivery configuration.
        transcription_config: Provider and temporary-media configuration.

    Returns:
        Unstarted worker runtime that verifies PostgreSQL during ``start``.
    """
    repository = PostgresTranscriptionJobRepository(
        engine=engine,
        maximum_outstanding_jobs=job_config.max_outstanding_jobs,
        queue_timeout=timedelta(seconds=job_config.job_queue_timeout_seconds),
        terminal_retention=timedelta(seconds=job_config.job_retention_seconds),
        clock=utc_now,
    )
    return TranscriptionWorkerRuntime.from_repository(
        repository,
        engine,
        transcription_config,
        dispatch_config,
    )


def _connect_worker_shutdown_signal(
    runtime: TranscriptionWorkerRuntime,
) -> Callable[..., None]:
    """Close claim admission synchronously when Celery begins shutdown.

    Args:
        runtime: Started runtime whose manager must reject new claims.

    Returns:
        Strongly registered Celery receiver to disconnect during teardown.
    """

    def stop_claiming_on_shutdown(**_signal_kwargs: object) -> None:
        """Schedule admission closure without blocking Celery's signal handler."""
        runtime.stop_claiming()

    worker_shutting_down.connect(stop_claiming_on_shutdown, weak=False)
    return stop_claiming_on_shutdown


def _disconnect_worker_shutdown_signal(receiver: Callable[..., None]) -> None:
    """Remove a previously registered Celery shutdown receiver.

    Args:
        receiver: Exact signal receiver returned by connection setup.
    """
    worker_shutting_down.disconnect(receiver)


def main() -> None:
    """Start one dedicated threads-pool Transcription Job worker process."""
    job_config = JobConfig()  # type: ignore[call-arg]
    dispatch_config = JobDispatchConfig()  # type: ignore[call-arg]
    cache_config = CacheConfig()
    validate_redis_role_isolation(dispatch_config, cache_config)
    transcription_config = TranscriptionConfig()  # type: ignore[call-arg]
    engine = create_application_engine(str(job_config.database_url))
    runtime: TranscriptionWorkerRuntime | None = None
    shutdown_receiver: Callable[..., None] | None = None
    try:
        runtime = asyncio.run(
            _create_worker_runtime(
                engine,
                job_config,
                dispatch_config,
                transcription_config,
            )
        )
        runtime.start()
        shutdown_receiver = _connect_worker_shutdown_signal(runtime)
        celery_app = create_celery_app(dispatch_config)
        register_transcription_task(celery_app, runtime)
        celery_app.worker_main(
            [
                "worker",
                "--pool=threads",
                f"--concurrency={dispatch_config.worker_concurrency}",
                f"--queues={TRANSCRIPTION_QUEUE}",
                "--loglevel=INFO",
            ]
        )
    finally:
        if shutdown_receiver is not None:
            _disconnect_worker_shutdown_signal(shutdown_receiver)
        if runtime is None:
            asyncio.run(engine.dispose())
        else:
            runtime.shutdown()
