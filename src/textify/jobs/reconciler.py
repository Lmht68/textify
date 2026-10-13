"""Reconcile committed Job Dispatches into private Celery delivery."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal
from uuid import uuid4

from textify.config import AppConfig
from textify.errors import InternalError
from textify.jobs.celery_app import create_celery_app
from textify.jobs.config import (
    CacheConfig,
    JobConfig,
    JobDispatchConfig,
    validate_redis_role_isolation,
)
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
from textify.jobs.readiness import (
    PostgresServiceHeartbeatRepository,
    RedisBrokerReadiness,
    ServiceHeartbeatReporter,
    ServiceReadinessUnavailableError,
    ServiceRole,
)
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.service import utc_now
from textify.logging import (
    LogCode,
    LogEvent,
    LogServiceRole,
    configure_structured_logging,
    emit_event,
)

logger = logging.getLogger(__name__)

_ReconcilerFailureStage = Literal["startup", "runtime", "shutdown"]


class _ReconcilerProcessError(Exception):
    """Carry the safe lifecycle stage for an unexpected process failure."""

    stage: _ReconcilerFailureStage

    def __init__(self, stage: _ReconcilerFailureStage) -> None:
        """Initialize one stage-specific reconciler process failure."""
        super().__init__()
        self.stage = stage


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

    async def reconcile_once(self) -> bool:
        """Recover, expire, and publish every currently due Job Dispatch.

        Returns:
            ``True`` when recovery, queue expiry, all publication records, and the
            current due-dispatch scan complete. ``False`` when Redis rejects a
            leased dispatch and its unpublished suffix is released.

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
                return True
            for index, dispatch in enumerate(dispatches):
                try:
                    await self._publisher.publish(dispatch)
                except JobDispatchUnavailableError:
                    emit_event(
                        logger,
                        LogEvent.JOB_DISPATCH_UNAVAILABLE,
                        internal_job_id=dispatch.internal_job_id,
                        code=(
                            LogCode.TRANSCRIPTION_JOB_DISPATCH_BROKER_IS_UNAVAILABLE.value
                        ),
                    )
                    await self._repository.release_dispatch_leases(
                        dispatches[index:],
                        self._lease_owner,
                    )
                    return False
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

    async def run(
        self,
        stop_event: asyncio.Event,
        heartbeat: ServiceHeartbeatReporter,
    ) -> None:
        """Reconcile and report readiness until the caller requests shutdown.

        Args:
            stop_event: Event set by the process owner to stop scans cleanly.
            heartbeat: Reconciler service-readiness lifecycle.
        """
        try:
            await heartbeat.start()
        except Exception as exc:
            raise _ReconcilerProcessError("startup") from exc
        emit_event(logger, LogEvent.SERVICE_READY)
        next_cleanup_at = time.monotonic()
        try:
            while not stop_event.is_set():
                try:
                    reconciliation_succeeded = await self.reconcile_once()
                    if time.monotonic() >= next_cleanup_at:
                        await self.cleanup_once()
                        next_cleanup_at = (
                            time.monotonic()
                            + self._policy.retention_cleanup_interval.total_seconds()
                        )
                    if reconciliation_succeeded:
                        await heartbeat.pulse()
                except TranscriptionJobStoreUnavailableError:
                    emit_event(
                        logger,
                        LogEvent.JOB_STORE_UNAVAILABLE,
                        code=(
                            LogCode.TRANSCRIPTION_JOB_STORAGE_IS_UNAVAILABLE_TO_RECONCILER.value
                        ),
                    )
                except ServiceReadinessUnavailableError:
                    emit_event(
                        logger,
                        LogEvent.SERVICE_READINESS_FAILED,
                        stage="readiness",
                        code=(
                            LogCode.RECONCILER_SERVICE_READINESS_STORAGE_IS_UNAVAILABLE.value
                        ),
                    )
                try:
                    async with asyncio.timeout(self._policy.interval.total_seconds()):
                        await stop_event.wait()
                except TimeoutError:
                    continue
        except Exception as exc:
            raise _ReconcilerProcessError("runtime") from exc
        finally:
            try:
                await heartbeat.stop()
            except Exception as exc:
                raise _ReconcilerProcessError("shutdown") from exc


async def run_reconciler() -> None:
    """Run the configured reconciler until the process is interrupted.

    Raises:
        _ReconcilerProcessError: If application-engine disposal fails.
    """
    stop_event = asyncio.Event()
    running_loop = asyncio.get_running_loop()
    registered_stop_signals: list[signal.Signals] = []
    try:
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            running_loop.add_signal_handler(stop_signal, stop_event.set)
            registered_stop_signals.append(stop_signal)

        job_config = JobConfig()  # type: ignore[call-arg]
        dispatch_config = JobDispatchConfig()  # type: ignore[call-arg]
        cache_config = CacheConfig()
        validate_redis_role_isolation(dispatch_config, cache_config)
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
            broker_readiness = RedisBrokerReadiness(dispatch_config)
            try:
                await publisher.initialize(broker_readiness)
            finally:
                await broker_readiness.aclose()
            heartbeat = ServiceHeartbeatReporter(
                PostgresServiceHeartbeatRepository(engine),
                ServiceRole.RECONCILER,
            )
            reconciler = TranscriptionJobReconciler(
                repository,
                publisher,
                policy=ReconcilerPolicy(
                    interval=timedelta(
                        seconds=dispatch_config.reconciler_interval_seconds
                    )
                ),
            )
            await reconciler.run(stop_event, heartbeat)
        finally:
            try:
                await engine.dispose()
            except Exception as exc:
                raise _ReconcilerProcessError("shutdown") from exc
    finally:
        for stop_signal in registered_stop_signals:
            running_loop.remove_signal_handler(stop_signal)


def _require_positive_duration(value: timedelta, name: str) -> None:
    """Reject invalid reconciler policy durations."""
    if not isinstance(value, timedelta) or value <= timedelta():
        raise ValueError(f"{name} must be a positive timedelta.")


def _raise_service_failed(stage: _ReconcilerFailureStage) -> None:
    """Emit one safe process failure event and terminate unsuccessfully."""
    emit_event(
        logger,
        LogEvent.SERVICE_FAILED,
        stage=stage,
        code=InternalError.code,
    )
    raise SystemExit(1) from None


def main() -> None:
    """Start the dedicated Transcription Job reconciler process."""
    configure_structured_logging(LogServiceRole.RECONCILER)
    emit_event(logger, LogEvent.SERVICE_STARTING)
    try:
        app_config = AppConfig()
        configure_structured_logging(LogServiceRole.RECONCILER, app_config.log_level)
        asyncio.run(run_reconciler())
    except KeyboardInterrupt:
        pass
    except _ReconcilerProcessError as exc:
        _raise_service_failed(exc.stage)
    except Exception:  # noqa: BLE001
        _raise_service_failed("startup")
    emit_event(logger, LogEvent.SERVICE_STOPPING)
