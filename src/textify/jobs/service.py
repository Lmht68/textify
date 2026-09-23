"""Application coordination for durable Transcription Jobs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID

from textify.errors import InternalError
from textify.jobs.exceptions import (
    JobAlreadyFinishedError,
    JobNotFoundError,
    JobStoreUnavailableError,
    QueueTimeoutError,
    TranscriptionCapacityExceededError,
    WorkerInterruptedError,
)
from textify.jobs.repository import (
    SqliteTranscriptionJobRepository,
    TranscriptionJobAlreadyFinishedError,
    TranscriptionJobCapacityError,
    TranscriptionJobStoreUnavailableError,
)
from textify.jobs.types import (
    CancelledTranscriptionJob,
    ClaimedTranscriptionJob,
    NewQueuedTranscriptionJob,
    ProcessingCancellation,
    ProcessingTranscriptionJob,
    QueuedTranscriptionJob,
    TranscriptionJob,
)
from textify.logging import bind_request_log_fields
from textify.transcription import inspection
from textify.transcription.exceptions import (
    TranscriptionCancellationRequestedError,
    TranscriptionError,
)
from textify.transcription.schemas import build_transcription_response
from textify.transcription.service import (
    TranscriptionExecutionControl,
    TranscriptionExecutor,
)
from textify.transcription.types import ResponseFieldPath

logger = logging.getLogger(__name__)

_WORKER_WAKE_INTERVAL_SECONDS = 1.0


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


async def recover_transcription_jobs(
    repository: SqliteTranscriptionJobRepository,
) -> None:
    """Reconcile durable jobs left active by an interrupted application.

    Args:
        repository: SQLite lifecycle boundary that owns the recovery transaction.

    Raises:
        TranscriptionJobStoreUnavailableError: If recovery cannot commit safely.
    """
    recovered_jobs = await repository.recover_interrupted_jobs()
    for internal_job_id in recovered_jobs.expired_queued_internal_ids:
        logger.info(
            "expired queued transcription job during startup recovery",
            extra={"internal_job_id": internal_job_id, "code": QueueTimeoutError.code},
        )
    for internal_job_id in recovered_jobs.interrupted_processing_internal_ids:
        logger.info(
            "failed interrupted transcription job during startup recovery",
            extra={
                "internal_job_id": internal_job_id,
                "code": WorkerInterruptedError.code,
            },
        )
    for internal_job_id in recovered_jobs.cancelled_processing_internal_ids:
        logger.info(
            "cancelled interrupted transcription job during startup recovery",
            extra={"internal_job_id": internal_job_id},
        )


class TranscriptionJobCoordinator:
    """Validate, admit, inspect, and consume durable Transcription Jobs."""

    def __init__(
        self,
        repository: SqliteTranscriptionJobRepository,
        executor: TranscriptionExecutor,
        worker_count: int,
        retention_cleanup_interval: timedelta,
        *,
        on_store_unavailable: Callable[[], None],
    ) -> None:
        """Initialize application-owned Transcription Job consumers.

        Args:
            repository: SQLite adapter that durably owns lifecycle transitions.
            executor: Process-lifetime provider execution boundary.
            worker_count: Number of consumer tasks to own for this application.
            retention_cleanup_interval: Delay between expired terminal-job cleanup runs.
            on_store_unavailable: Synchronous readiness callback for one store failure.

        Raises:
            ValueError: If no consumer task is configured or cleanup is nonpositive.
        """
        if worker_count <= 0:
            raise ValueError("worker_count must be positive.")
        if retention_cleanup_interval <= timedelta():
            raise ValueError("retention_cleanup_interval must be positive.")
        self._repository = repository
        self._executor = executor
        self._worker_count = worker_count
        self._retention_cleanup_interval = retention_cleanup_interval
        self._on_store_unavailable = on_store_unavailable
        self._work_available = asyncio.Event()
        self._retention_cleanup_stop = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
        self._active_executions: dict[int, TranscriptionExecutionControl] = {}
        self._consumer_tasks: tuple[asyncio.Task[None], ...] = ()
        self._retention_cleanup_task: asyncio.Task[None] | None = None
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._store_available = True
        self._started = False
        self._stopping = False
        self._shutdown_started = False

    async def _mark_store_unavailable(self) -> None:
        """Trip the one-way durable-store failure latch and wake owned tasks."""
        async with self._lifecycle_lock:
            if not self._store_available:
                return
            self._store_available = False
            self._retention_cleanup_stop.set()
            self._work_available.set()
        logger.error(
            "transcription job store became unavailable",
            extra={"code": JobStoreUnavailableError.code},
        )
        self._on_store_unavailable()

    def _require_store_available(self) -> None:
        """Raise when the caller holds the lifecycle lock after a store failure."""
        if not self._store_available:
            raise JobStoreUnavailableError()

    async def start(self) -> None:
        """Start every owned task before making durable admission available.

        Raises:
            JobStoreUnavailableError: If a task detects unavailable durable storage.
            RuntimeError: If consumers have already started, shut down, or fail to start.
        """
        if self._started or self._shutdown_started:
            raise RuntimeError("Transcription Job consumers cannot be started again.")

        startup_signals: list[asyncio.Event] = []
        consumer_tasks: list[asyncio.Task[None]] = []
        retention_cleanup_task: asyncio.Task[None] | None = None
        startup_succeeded = False
        self._work_available.set()
        try:
            for worker_number in range(1, self._worker_count + 1):
                startup_signal = asyncio.Event()
                startup_signals.append(startup_signal)
                consumer_coroutine = self._consume_jobs(startup_signal)
                try:
                    consumer_task = asyncio.create_task(
                        consumer_coroutine,
                        name=f"transcription-job-consumer-{worker_number}",
                    )
                except RuntimeError:
                    consumer_coroutine.close()
                    raise
                consumer_tasks.append(consumer_task)

            retention_startup_signal = asyncio.Event()
            startup_signals.append(retention_startup_signal)
            retention_cleanup_coroutine = (
                self._remove_expired_terminal_jobs_periodically(
                    retention_startup_signal
                )
            )
            try:
                retention_cleanup_task = asyncio.create_task(
                    retention_cleanup_coroutine,
                    name="transcription-job-retention-cleanup",
                )
            except RuntimeError:
                retention_cleanup_coroutine.close()
                raise

            await asyncio.gather(*(signal.wait() for signal in startup_signals))
            assert retention_cleanup_task is not None
            await asyncio.sleep(0)
            for task in (*consumer_tasks, retention_cleanup_task):
                if task.cancelled():
                    raise RuntimeError(
                        "Transcription Job task was cancelled during startup."
                    )
                task_error = task.exception() if task.done() else None
                if task_error is not None:
                    raise task_error
            if not self._store_available:
                raise JobStoreUnavailableError()
            self._consumer_tasks = tuple(consumer_tasks)
            self._retention_cleanup_task = retention_cleanup_task
            self._started = True
            startup_succeeded = True
        finally:
            if not startup_succeeded:
                startup_tasks = tuple(consumer_tasks)
                if retention_cleanup_task is not None:
                    startup_tasks += (retention_cleanup_task,)
                await self._abort_startup(startup_tasks)

    async def _abort_startup(
        self,
        startup_tasks: tuple[asyncio.Task[None], ...],
    ) -> None:
        """Cancel partially started tasks and release executor resources."""
        async with self._lifecycle_lock:
            self._shutdown_started = True
            self._stopping = True
            self._retention_cleanup_stop.set()
        self._work_available.set()
        for task in startup_tasks:
            task.cancel()
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)
        try:
            await self._executor.shutdown()
        finally:
            await self._await_cleanup_tasks()

    async def shutdown(self) -> None:
        """Stop admission and claiming before draining owned execution resources."""
        async with self._lifecycle_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self._stopping = True
            self._retention_cleanup_stop.set()
            active_execution_ids = tuple(sorted(self._active_executions))

        logger.info("transcription job coordinator shutdown started")
        for internal_job_id in active_execution_ids:
            logger.info(
                "waiting for active transcription job during shutdown",
                extra={"internal_job_id": internal_job_id},
            )
        self._work_available.set()
        tasks_to_await = self._consumer_tasks
        if self._retention_cleanup_task is not None:
            tasks_to_await += (self._retention_cleanup_task,)
        try:
            if tasks_to_await:
                await asyncio.gather(*tasks_to_await)
        finally:
            try:
                await self._executor.shutdown()
            finally:
                await self._await_cleanup_tasks()
        logger.info("transcription job coordinator shutdown completed")

    async def submit(
        self,
        submitted_url: str,
        exclusions: frozenset[ResponseFieldPath],
    ) -> QueuedTranscriptionJob:
        """Validate and durably queue one Supported Platform transcription.

        Args:
            submitted_url: Source URL supplied by the client.
            exclusions: Validated immutable response projection exclusions.

        Returns:
            Safe committed queued-job snapshot.

        Raises:
            InvalidUrlError: If the submitted URL is malformed.
            UnsupportedPlatformError: If the submitted URL platform is unsupported.
            TranscriptionCapacityExceededError: If durable admission is full.
            JobStoreUnavailableError: If the durable job store cannot commit safely.
        """
        submitted = inspection.classify_submitted_url(submitted_url)
        bind_request_log_fields(platform=submitted.platform.value)
        new_job = NewQueuedTranscriptionJob(
            submitted_url=submitted_url,
            exclusions=tuple(sorted(exclusions)),
        )
        try:
            async with self._lifecycle_lock:
                self._require_store_available()
                if not self._started or self._stopping:
                    raise TranscriptionCapacityExceededError()
                queued_job = await self._repository.create_queued(new_job)
        except TranscriptionJobCapacityError as exc:
            raise TranscriptionCapacityExceededError() from exc
        except TranscriptionJobStoreUnavailableError as exc:
            await self._mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        self._work_available.set()
        return queued_job

    async def get_status(self, capability: str) -> TranscriptionJob:
        """Retrieve one Transcription Job through a canonical UUIDv4 capability.

        Args:
            capability: Opaque client-supplied bearer capability path value.

        Returns:
            Safe current job snapshot.

        Raises:
            JobNotFoundError: If the capability is malformed, noncanonical, or unknown.
            JobStoreUnavailableError: If the durable job store cannot read safely.
        """
        public_id = _parse_capability(capability)
        try:
            async with self._lifecycle_lock:
                self._require_store_available()
                job = await self._repository.get_job(public_id)
        except TranscriptionJobStoreUnavailableError as exc:
            await self._mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        if job is None:
            raise JobNotFoundError()
        return job

    async def cancel(
        self,
        capability: str,
    ) -> ProcessingTranscriptionJob | CancelledTranscriptionJob:
        """Accept cancellation through one canonical Transcription Job capability.

        Args:
            capability: Opaque client-supplied bearer capability path value.

        Returns:
            Current processing snapshot after durable acceptance, or the stable
            cancelled terminal snapshot.

        Raises:
            JobNotFoundError: If the capability is malformed, noncanonical, or unknown.
            JobAlreadyFinishedError: If the job has a non-cancelled terminal outcome.
            JobStoreUnavailableError: If durable cancellation cannot commit safely.
        """
        public_id = _parse_capability(capability)
        job: ProcessingTranscriptionJob | CancelledTranscriptionJob
        try:
            async with self._lifecycle_lock:
                self._require_store_available()
                cancellation = await self._repository.request_cancellation(public_id)
                if cancellation is None:
                    raise JobNotFoundError()
                if isinstance(cancellation, ProcessingCancellation):
                    control = self._active_executions.get(cancellation.internal_id)
                    if control is not None:
                        control.request_cancellation()
                    job = cancellation.job
                else:
                    job = cancellation
        except TranscriptionJobAlreadyFinishedError as exc:
            raise JobAlreadyFinishedError() from exc
        except TranscriptionJobStoreUnavailableError as exc:
            await self._mark_store_unavailable()
            raise JobStoreUnavailableError() from exc
        self._work_available.set()
        return job

    async def _remove_expired_terminal_jobs_periodically(
        self,
        startup_signal: asyncio.Event,
    ) -> None:
        """Delete expired terminal jobs after each configured full interval."""
        startup_signal.set()
        try:
            while True:
                try:
                    async with asyncio.timeout(
                        self._retention_cleanup_interval.total_seconds()
                    ):
                        await self._retention_cleanup_stop.wait()
                except TimeoutError:
                    pass
                if self._retention_cleanup_stop.is_set():
                    return
                await self._repository.delete_expired_terminal_jobs()
        except TranscriptionJobStoreUnavailableError:
            await self._mark_store_unavailable()

    async def _consume_jobs(self, startup_signal: asyncio.Event) -> None:
        """Wait for committed work, then drain eligible jobs with bounded wakes."""
        startup_signal.set()
        try:
            while True:
                await self._wait_for_work()
                async with self._lifecycle_lock:
                    if self._stopping or not self._store_available:
                        return
                while True:
                    claimed_job = await self._claim_next()
                    if claimed_job is None:
                        break
                    await self._execute_claimed_job(claimed_job)
        except TranscriptionJobStoreUnavailableError:
            await self._mark_store_unavailable()

    async def _wait_for_work(self) -> None:
        """Wait for work submission or an interval boundary before draining."""
        if self._stopping:
            return
        try:
            async with asyncio.timeout(_WORKER_WAKE_INTERVAL_SECONDS):
                await self._work_available.wait()
        except TimeoutError:
            pass
        self._work_available.clear()

    async def _claim_next(self) -> ClaimedTranscriptionJob | None:
        """Claim one job and register its process-local execution control."""
        async with self._lifecycle_lock:
            if self._stopping or not self._store_available:
                return None
            claimed_job = await self._repository.claim_next()
            if claimed_job is not None:
                self._active_executions[claimed_job.internal_id] = (
                    TranscriptionExecutionControl()
                )
            return claimed_job

    async def _execute_claimed_job(self, claimed_job: ClaimedTranscriptionJob) -> None:
        """Execute and publish one durable claim without retaining private inputs."""
        async with self._lifecycle_lock:
            control = self._active_executions.get(claimed_job.internal_id)
        if control is None:
            raise RuntimeError("Claimed Transcription Job has no execution control.")

        try:
            submitted = inspection.classify_submitted_url(claimed_job.submitted_url)
            result = await self._executor.execute(
                submitted,
                control=control,
                include_segments="transcript.segments" not in claimed_job.exclusions,
            )
        except TranscriptionCancellationRequestedError:
            await self._publish_cancelled_after_cleanup(claimed_job, control)
        except TranscriptionError as error:
            if control.cancellation_requested.is_set():
                await self._publish_cancelled_after_cleanup(claimed_job, control)
                return
            logger.error("transcription job worker failed", extra={"code": error.code})
            await self._publish_failure_or_cancelled(
                claimed_job,
                control,
                error.code,
                error.message,
            )
        except Exception:  # noqa: BLE001
            if control.cancellation_requested.is_set():
                await self._publish_cancelled_after_cleanup(claimed_job, control)
                return
            logger.error(
                "transcription job worker failed",
                extra={"code": InternalError.code},
            )
            await self._publish_failure_or_cancelled(
                claimed_job,
                control,
                InternalError.code,
                InternalError.message,
            )
        else:
            if control.cancellation_requested.is_set():
                await self._publish_cancelled_after_cleanup(claimed_job, control)
                return
            try:
                projected_result = build_transcription_response(
                    result,
                    frozenset(claimed_job.exclusions),
                )
            except Exception:  # noqa: BLE001
                if control.cancellation_requested.is_set():
                    await self._publish_cancelled_after_cleanup(claimed_job, control)
                    return
                logger.error(
                    "transcription job worker failed",
                    extra={"code": InternalError.code},
                )
                await self._publish_failure_or_cancelled(
                    claimed_job,
                    control,
                    InternalError.code,
                    InternalError.message,
                )
                return
            published = await self._repository.publish_success(
                claimed_job.internal_id,
                projected_result,
            )
            if not published:
                await self._publish_cancelled_after_cleanup(claimed_job, control)
        finally:
            cleanup_task = asyncio.create_task(
                self._remove_active_execution_after_cleanup(
                    claimed_job.internal_id,
                    control,
                ),
                name=f"transcription-job-cleanup-{claimed_job.internal_id}",
            )
            self._cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._cleanup_tasks.discard)

    async def _publish_failure_or_cancelled(
        self,
        claimed_job: ClaimedTranscriptionJob,
        control: TranscriptionExecutionControl,
        error_code: str,
        error_message: str,
    ) -> None:
        """Publish failure unless accepted cancellation wins its race."""
        if control.cancellation_requested.is_set():
            await self._publish_cancelled_after_cleanup(claimed_job, control)
            return
        published = await self._repository.publish_failure(
            claimed_job.internal_id,
            error_code,
            error_message,
        )
        if not published:
            await self._publish_cancelled_after_cleanup(claimed_job, control)

    async def _publish_cancelled_after_cleanup(
        self,
        claimed_job: ClaimedTranscriptionJob,
        control: TranscriptionExecutionControl,
    ) -> None:
        """Publish accepted cancellation only after provider ownership is released."""
        await control.cleanup_complete.wait()
        await self._repository.publish_cancelled(claimed_job.internal_id)

    async def _remove_active_execution_after_cleanup(
        self,
        internal_id: int,
        control: TranscriptionExecutionControl,
    ) -> None:
        """Remove an execution control only after provider cleanup completes."""
        await control.cleanup_complete.wait()
        async with self._lifecycle_lock:
            if self._active_executions.get(internal_id) is control:
                self._active_executions.pop(internal_id)

    async def _await_cleanup_tasks(self) -> None:
        """Wait for all coordinator-owned execution cleanup tasks."""
        while self._cleanup_tasks:
            cleanup_tasks = tuple(self._cleanup_tasks)
            await asyncio.gather(*cleanup_tasks)


def _parse_capability(capability: str) -> UUID:
    """Parse only canonical lowercase UUIDv4 bearer capabilities."""
    try:
        public_id = UUID(capability)
    except (AttributeError, TypeError, ValueError) as exc:
        raise JobNotFoundError() from exc
    if public_id.version != 4 or str(public_id) != capability:
        raise JobNotFoundError()
    return public_id
