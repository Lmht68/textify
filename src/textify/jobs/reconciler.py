"""Reconcile committed Job Dispatches into private Celery delivery."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta

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


class TranscriptionJobReconciler:
    """Publish committed queued Execution Attempts and own terminal retention."""

    def __init__(
        self,
        repository: TranscriptionJobDispatchRepository,
        publisher: JobDispatchPublisher,
        interval_seconds: float = 1.0,
        retention_cleanup_interval: timedelta = timedelta(hours=1),
    ) -> None:
        """Initialize PostgreSQL scanning and private delivery dependencies.

        Args:
            repository: PostgreSQL queued-dispatch and terminal-retention boundary.
            publisher: Private Celery delivery boundary.
            interval_seconds: Delay between reconciliation scans.
            retention_cleanup_interval: Delay between terminal retention passes.

        Raises:
            ValueError: If either interval is nonpositive.
        """
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")
        if retention_cleanup_interval <= timedelta():
            raise ValueError("retention_cleanup_interval must be positive.")
        self._repository = repository
        self._publisher = publisher
        self._interval_seconds = interval_seconds
        self._retention_cleanup_interval_seconds = (
            retention_cleanup_interval.total_seconds()
        )

    async def reconcile_once(self) -> None:
        """Publish every currently queued dispatch until broker delivery fails.

        Repeated delivery is intentional. PostgreSQL claim semantics ensure that only
        one delivery can own an Execution Attempt.

        Raises:
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot list queued
                dispatches safely.
        """
        dispatches = await self._repository.list_queued_dispatches()
        for dispatch in dispatches:
            try:
                await self._publisher.publish(dispatch)
            except JobDispatchUnavailableError:
                logger.error("transcription job dispatch broker is unavailable")
                return

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
                        time.monotonic() + self._retention_cleanup_interval_seconds
                    )
            except TranscriptionJobStoreUnavailableError:
                logger.error("transcription job storage is unavailable to reconciler")
            try:
                async with asyncio.timeout(self._interval_seconds):
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
            interval_seconds=dispatch_config.reconciler_interval_seconds,
        )
        await reconciler.run(asyncio.Event())
    finally:
        await engine.dispose()


def main() -> None:
    """Start the dedicated Transcription Job reconciler process."""
    asyncio.run(run_reconciler())
