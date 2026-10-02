"""Reconcile committed Job Dispatches into private Celery delivery."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4

from textify.jobs.celery_app import create_celery_app
from textify.jobs.config import JobConfig, JobDispatchConfig
from textify.jobs.contracts import (
    TranscriptionJobDispatchRepository,
    TranscriptionJobStoreUnavailableError,
)
from textify.jobs.database import create_application_engine, verify_application_database
from textify.jobs.dispatch import (
    CeleryJobDispatchPublisher,
    JobDispatchPublisher,
    JobDispatchUnavailableError,
)
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.service import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconcilerPolicy:
    """Define bounded timing and lease behavior for one reconciler process."""

    interval: timedelta = timedelta(seconds=1)
    publication_lease_duration: timedelta = timedelta(seconds=5)
    dispatch_batch_size: int = 100
    retention_cleanup_interval: timedelta = timedelta(hours=1)

    def __post_init__(self) -> None:
        """Reject timing policies that can strand or spin dispatch recovery."""
        _require_positive_duration(self.interval, "interval")
        _require_positive_duration(
            self.publication_lease_duration,
            "publication_lease_duration",
        )
        if (
            isinstance(self.dispatch_batch_size, bool)
            or not isinstance(self.dispatch_batch_size, int)
            or self.dispatch_batch_size <= 0
        ):
            raise ValueError("dispatch_batch_size must be a positive integer.")
        _require_positive_duration(
            self.retention_cleanup_interval,
            "retention_cleanup_interval",
        )


class TranscriptionJobReconciler:
    """Publish committed queued Execution Attempts and own terminal retention."""

    def __init__(
        self,
        repository: TranscriptionJobDispatchRepository,
        publisher: JobDispatchPublisher,
        policy: ReconcilerPolicy | None = None,
    ) -> None:
        """Initialize PostgreSQL recovery and private delivery dependencies.

        Args:
            repository: PostgreSQL dispatch-recovery and terminal-retention seam.
            publisher: Private Celery delivery seam.
            policy: Optional bounded reconciliation timing and lease policy.
        """
        self._repository = repository
        self._publisher = publisher
        self._policy = policy or ReconcilerPolicy()
        self._lease_owner = uuid4()

    async def reconcile_once(self) -> None:
        """Recover expired claims, expire queued work, then publish due dispatches.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL recovery cannot
                complete an operation.
        """
        await self._recover_expired_claims()
        await self._expire_queued_jobs()
        while True:
            dispatches = await self._repository.lease_due_dispatches(
                self._lease_owner,
                self._policy.publication_lease_duration,
                self._policy.dispatch_batch_size,
            )
            if not dispatches:
                return
            for index, dispatch in enumerate(dispatches):
                try:
                    await self._publisher.publish(dispatch)
                except JobDispatchUnavailableError:
                    logger.error("transcription job dispatch broker is unavailable")
                    await self._repository.release_dispatch_leases(
                        dispatches[index:],
                        self._lease_owner,
                    )
                    return
                await self._repository.record_dispatch_published(
                    dispatch,
                    self._lease_owner,
                    self._policy.interval,
                )

    async def _recover_expired_claims(self) -> None:
        """Drain bounded lost-worker transitions before queued-job recovery."""
        while await self._repository.recover_expired_claims(
            self._policy.dispatch_batch_size
        ):
            continue

    async def _expire_queued_jobs(self) -> None:
        """Drain bounded queue-timeout transitions before dispatch publication."""
        while await self._repository.expire_queued_jobs(
            self._policy.dispatch_batch_size
        ):
            continue

    async def cleanup_once(self) -> None:
        """Delete terminal jobs that reached PostgreSQL retention cutoff.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot delete safely.
        """
        await self._repository.delete_expired_terminal_jobs()

    async def run(self, stop_event: asyncio.Event) -> None:
        """Reconcile until the caller requests shutdown.

        Args:
            stop_event: Event set by the process owner to stop scans cleanly.
        """
        next_cleanup_at = time.monotonic()
        while not stop_event.is_set():
            try:
                await self.reconcile_once()
                if time.monotonic() >= next_cleanup_at:
                    await self.cleanup_once()
                    next_cleanup_at = (
                        time.monotonic()
                        + self._policy.retention_cleanup_interval.total_seconds()
                    )
            except TranscriptionJobStoreUnavailableError:
                logger.error("transcription job storage is unavailable to reconciler")
            try:
                async with asyncio.timeout(self._policy.interval.total_seconds()):
                    await stop_event.wait()
            except TimeoutError:
                continue


async def run_reconciler() -> None:
    """Run the configured reconciler until the process is interrupted.

    Raises:
        RuntimeError: If PostgreSQL startup validation fails.
    """
    job_config = JobConfig()  # type: ignore[call-arg]
    dispatch_config = JobDispatchConfig()  # type: ignore[call-arg]
    engine = create_application_engine(str(job_config.database_url))
    try:
        await verify_application_database(engine)
        repository = PostgresTranscriptionJobRepository(
            engine=engine,
            maximum_outstanding_jobs=job_config.max_outstanding_jobs,
            queue_timeout=timedelta(seconds=job_config.job_queue_timeout_seconds),
            terminal_retention=timedelta(seconds=job_config.job_retention_seconds),
            clock=utc_now,
        )
        publisher = CeleryJobDispatchPublisher(create_celery_app(dispatch_config))
        reconciler = TranscriptionJobReconciler(
            repository,
            publisher,
            policy=ReconcilerPolicy(
                interval=timedelta(seconds=dispatch_config.reconciler_interval_seconds)
            ),
        )
        await reconciler.run(asyncio.Event())
    finally:
        await engine.dispose()


def _require_positive_duration(value: timedelta, name: str) -> None:
    """Reject invalid reconciler policy durations."""
    if not isinstance(value, timedelta) or value <= timedelta():
        raise ValueError(f"{name} must be a positive timedelta.")


def main() -> None:
    """Start the dedicated Transcription Job reconciler process."""
    asyncio.run(run_reconciler())
