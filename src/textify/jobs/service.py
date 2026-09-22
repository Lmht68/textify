"""Application coordination for durable Transcription Jobs."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from uuid import UUID

from textify.errors import InternalError
from textify.jobs.exceptions import (
    JobAlreadyFinishedError,
    JobNotFoundError,
    JobStoreUnavailableError,
    TranscriptionCapacityExceededError,
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


class TranscriptionJobCoordinator:
    """Validate, admit, inspect, and consume durable Transcription Jobs."""

    def __init__(
        self,
        repository: SqliteTranscriptionJobRepository,
        executor: TranscriptionExecutor,
        worker_count: int,
    ) -> None:
        """Initialize application-owned Transcription Job consumers.

        Args:
            repository: SQLite adapter that durably owns lifecycle transitions.
            executor: Process-lifetime provider execution boundary.
            worker_count: Number of consumer tasks to own for this application.

        Raises:
            ValueError: If no consumer task is configured.
        """
        if worker_count <= 0:
            raise ValueError("worker_count must be positive.")
        self._repository = repository
        self._executor = executor
        self._worker_count = worker_count
        self._work_available = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
        self._active_executions: dict[int, TranscriptionExecutionControl] = {}
        self._consumer_tasks: tuple[asyncio.Task[None], ...] = ()
        self._started = False
        self._stopping = False
        self._shutdown_started = False

    async def start(self) -> None:
        """Start consumers and consider durable queued work immediately.

        Raises:
            RuntimeError: If consumers have already started or shut down.
        """
        if self._started or self._shutdown_started:
            raise RuntimeError("Transcription Job consumers cannot be started again.")
        self._started = True
        self._work_available.set()
        self._consumer_tasks = tuple(
            asyncio.create_task(
                self._consume_jobs(),
                name=f"transcription-job-consumer-{worker_number}",
            )
            for worker_number in range(1, self._worker_count + 1)
        )

    async def shutdown(self) -> None:
        """Stop new claims, drain active execution, and release executor resources."""
        if self._shutdown_started:
            return
        self._shutdown_started = True
        async with self._lifecycle_lock:
            self._stopping = True
        self._work_available.set()
        try:
            if self._consumer_tasks:
                await asyncio.gather(*self._consumer_tasks)
        finally:
            await self._executor.shutdown()

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
            queued_job = await self._repository.create_queued(new_job)
        except TranscriptionJobCapacityError as exc:
            raise TranscriptionCapacityExceededError() from exc
        except TranscriptionJobStoreUnavailableError as exc:
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
            job = await self._repository.get_job(public_id)
        except TranscriptionJobStoreUnavailableError as exc:
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
        async with self._lifecycle_lock:
            try:
                cancellation = await self._repository.request_cancellation(public_id)
            except TranscriptionJobAlreadyFinishedError as exc:
                raise JobAlreadyFinishedError() from exc
            except TranscriptionJobStoreUnavailableError as exc:
                raise JobStoreUnavailableError() from exc

            if cancellation is None:
                raise JobNotFoundError()
            if isinstance(cancellation, ProcessingCancellation):
                control = self._active_executions.get(cancellation.internal_id)
                if control is not None:
                    control.request_cancellation()
                return cancellation.job
            return cancellation

    async def _consume_jobs(self) -> None:
        """Wait for committed work, then drain eligible jobs with bounded wakes."""
        try:
            while True:
                await self._wait_for_work()
                if self._stopping:
                    return
                while True:
                    claimed_job = await self._claim_next()
                    if claimed_job is None:
                        break
                    await self._execute_claimed_job(claimed_job)
        except TranscriptionJobStoreUnavailableError:
            logger.error("job_store_unavailable")

    async def _wait_for_work(self) -> None:
        """Wait for work submission or an interval boundary before draining."""
        try:
            async with asyncio.timeout(_WORKER_WAKE_INTERVAL_SECONDS):
                await self._work_available.wait()
        except TimeoutError:
            pass
        self._work_available.clear()

    async def _claim_next(self) -> ClaimedTranscriptionJob | None:
        """Claim one job and register its process-local execution control."""
        async with self._lifecycle_lock:
            if self._stopping:
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
            asyncio.create_task(
                self._remove_active_execution_after_cleanup(
                    claimed_job.internal_id,
                    control,
                ),
                name=f"transcription-job-cleanup-{claimed_job.internal_id}",
            )

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


def _parse_capability(capability: str) -> UUID:
    """Parse only canonical lowercase UUIDv4 bearer capabilities."""
    try:
        public_id = UUID(capability)
    except (AttributeError, TypeError, ValueError) as exc:
        raise JobNotFoundError() from exc
    if public_id.version != 4 or str(public_id) != capability:
        raise JobNotFoundError()
    return public_id
