"""Tests for PostgreSQL-driven Job Dispatch reconciliation."""

import asyncio
import json
import subprocess
import sys
import textwrap
from collections import deque
from collections.abc import Callable
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from textify.jobs.contracts import TranscriptionJobStoreUnavailableError
from textify.jobs.dispatch import JobDispatchUnavailableError
from textify.jobs.readiness import ServiceReadinessUnavailableError
from textify.jobs.reconciler import ReconcilerPolicy, TranscriptionJobReconciler
from textify.jobs.types import JobDispatch
from textify.logging import LogServiceRole, configure_structured_logging


class RecordingDispatchRepository:
    """Record the reconciler's durable dispatch state machine operations."""

    def __init__(
        self,
        lease_batches: tuple[tuple[JobDispatch, ...], ...],
        expire_results: tuple[int, ...] = (),
        recovery_results: tuple[int, ...] = (),
    ) -> None:
        """Initialize deterministic recovery, expiry, and dispatch results.

        Args:
            lease_batches: Consecutive batches returned by due-lease operations.
            expire_results: Consecutive counts returned by queue expiry operations.
            recovery_results: Consecutive counts returned by expired claim recovery.
        """
        self._lease_batches = deque(lease_batches)
        self._expire_results = deque(expire_results)
        self._recovery_results = deque(recovery_results)
        self.calls: list[str] = []
        self.recovery_limits: list[int] = []
        self.lease_requests: list[tuple[UUID, timedelta, int]] = []
        self.recorded_dispatches: list[tuple[JobDispatch, UUID, timedelta]] = []
        self.released_dispatches: list[tuple[tuple[JobDispatch, ...], UUID]] = []
        self.cleanup_calls = 0
        self.record_error: TranscriptionJobStoreUnavailableError | None = None

    async def recover_expired_claims(self, limit: int) -> int:
        """Return one configured bounded expired-claim recovery result."""
        self.calls.append("recover")
        self.recovery_limits.append(limit)
        assert limit > 0
        return self._recovery_results.popleft() if self._recovery_results else 0

    async def expire_queued_jobs(self, limit: int) -> int:
        """Return one configured bounded expiry result."""
        self.calls.append("expire")
        assert limit > 0
        return self._expire_results.popleft() if self._expire_results else 0

    async def lease_due_dispatches(
        self,
        lease_owner: UUID,
        lease_duration: timedelta,
        limit: int,
    ) -> tuple[JobDispatch, ...]:
        """Return one configured due-dispatch batch."""
        self.calls.append("lease")
        self.lease_requests.append((lease_owner, lease_duration, limit))
        return self._lease_batches.popleft() if self._lease_batches else ()

    async def record_dispatch_published(
        self,
        dispatch: JobDispatch,
        lease_owner: UUID,
        redispatch_after: timedelta,
    ) -> bool:
        """Record one successful broker publication or raise a configured failure."""
        self.calls.append("record")
        self.recorded_dispatches.append((dispatch, lease_owner, redispatch_after))
        if self.record_error is not None:
            raise self.record_error
        return True

    async def release_dispatch_leases(
        self,
        dispatches: tuple[JobDispatch, ...],
        lease_owner: UUID,
    ) -> None:
        """Record one mapped broker-failure lease release."""
        self.calls.append("release")
        self.released_dispatches.append((dispatches, lease_owner))

    async def delete_expired_terminal_jobs(self) -> None:
        """Record one reconciler-owned retention pass."""
        self.cleanup_calls += 1


class RecordingPublisher:
    """Record dispatch publication and optionally reject one broker write."""

    def __init__(self, unavailable_after: int | None = None) -> None:
        """Initialize publication records and an optional broker failure position.

        Args:
            unavailable_after: Number of accepted publications before failure.
        """
        self.dispatches: list[JobDispatch] = []
        self._unavailable_after = unavailable_after

    async def publish(self, dispatch: JobDispatch) -> None:
        """Record a publication or signal an unavailable broker."""
        if (
            self._unavailable_after is not None
            and len(self.dispatches) >= self._unavailable_after
        ):
            raise JobDispatchUnavailableError()
        self.dispatches.append(dispatch)


class RecordingServiceHeartbeat:
    """Record reconciler readiness lifecycle operations."""

    def __init__(self, stop_event: asyncio.Event | None = None) -> None:
        """Initialize observable calls and an optional pulse-stop event."""
        self._stop_event = stop_event
        self.start_calls = 0
        self.pulse_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        """Record the required first readiness write."""
        self.start_calls += 1

    async def pulse(self) -> None:
        """Record a successful reconciliation-cycle readiness refresh."""
        self.pulse_calls += 1
        if self._stop_event is not None:
            self._stop_event.set()

    async def stop(self) -> None:
        """Record graceful removal of this reconciler heartbeat."""
        self.stop_calls += 1


class StopOnBrokerFailurePublisher:
    """Stop one run-loop test after an unavailable broker publication attempt."""

    def __init__(self, stop_event: asyncio.Event) -> None:
        """Initialize the run-loop stop event."""
        self._stop_event = stop_event

    async def publish(self, dispatch: JobDispatch) -> None:
        """Stop the loop and simulate a rejected broker publication."""
        del dispatch
        self._stop_event.set()
        raise JobDispatchUnavailableError()


class StoreFailureThenSuccessRepository(RecordingDispatchRepository):
    """Fail one store operation before a complete reconciler retry."""

    def __init__(self) -> None:
        """Initialize one scheduled storage interruption."""
        super().__init__(())
        self._failure_pending = True

    async def recover_expired_claims(self, limit: int) -> int:
        """Reject the first recovery pass, then delegate to the recording parent."""
        if self._failure_pending:
            self._failure_pending = False
            raise TranscriptionJobStoreUnavailableError("postgresql-url-sentinel")
        return await super().recover_expired_claims(limit)


class FailingServiceHeartbeat(RecordingServiceHeartbeat):
    """Fail one successful-cycle readiness refresh without masking shutdown."""

    async def pulse(self) -> None:
        """Stop the loop and signal an unavailable readiness store."""
        self.pulse_calls += 1
        if self._stop_event is not None:
            self._stop_event.set()
        raise ServiceReadinessUnavailableError("readiness-error-sentinel")


def _dispatch(internal_job_id: int) -> JobDispatch:
    """Build one valid private dispatch.

    Args:
        internal_job_id: Positive PostgreSQL job identity.

    Returns:
        Private notification for one Execution Attempt.
    """
    return JobDispatch(internal_job_id, uuid4())


def _policy(dispatch_batch_size: int = 100) -> ReconcilerPolicy:
    """Build the recovery policy used by isolated reconciliation tests."""
    return ReconcilerPolicy(
        interval=timedelta(seconds=1),
        publication_lease_duration=timedelta(seconds=5),
        dispatch_batch_size=dispatch_batch_size,
        retention_cleanup_interval=timedelta(hours=1),
    )


def test_reconciler_cli_emits_lifecycle_logs() -> None:
    """Emit each process lifecycle record once in chronological stderr order."""
    child_program = textwrap.dedent(
        """
        import asyncio
        from types import SimpleNamespace

        from textify.jobs import reconciler
        from textify.logging import (
            LogServiceRole,
            configure_structured_logging,
        )



        class Repository:
            async def recover_expired_claims(self, batch_size):
                return 0

            async def expire_queued_jobs(self, batch_size):
                return 0

            async def lease_due_dispatches(self, lease_owner, lease_duration, batch_size):
                return ()

            async def delete_expired_terminal_jobs(self):
                return None


        class Publisher:
            async def publish(self, dispatch):
                return None


        class Heartbeat:
            async def start(self):
                return None

            async def pulse(self):
                stop_event.set()

            async def stop(self):
                return None


        async def run_reconciler():
            await reconciler.TranscriptionJobReconciler(
                Repository(),
                Publisher(),
            ).run(stop_event, Heartbeat())


        stop_event = asyncio.Event()
        reconciler.AppConfig = lambda: SimpleNamespace(log_level="INFO")
        reconciler.configure_logging = lambda level: configure_structured_logging(
            LogServiceRole.RECONCILER,
            level,
        )
        reconciler.run_reconciler = run_reconciler
        reconciler.main()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", child_program],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    rendered_events = [json.loads(line) for line in completed.stderr.splitlines()]
    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "service_ready",
        "service_stopping",
    ]


@pytest.mark.asyncio
async def test_reconciler_recovers_claims_before_expiry_and_dispatch() -> None:
    """Drain expired claims before queued deadlines and new broker publication."""
    dispatch = _dispatch(3)
    repository = RecordingDispatchRepository(
        ((dispatch,), ()),
        expire_results=(1, 0),
        recovery_results=(2, 1, 0),
    )
    publisher = RecordingPublisher()

    assert (
        await TranscriptionJobReconciler(
            repository,
            publisher,
            policy=_policy(dispatch_batch_size=2),
        ).reconcile_once()
        is True
    )

    assert repository.calls == [
        "recover",
        "recover",
        "recover",
        "expire",
        "expire",
        "lease",
        "record",
        "lease",
    ]
    assert repository.recovery_limits == [2, 2, 2]
    assert publisher.dispatches == [dispatch]
    assert [record[0] for record in repository.recorded_dispatches] == [dispatch]


@pytest.mark.asyncio
async def test_reconciler_drains_due_dispatch_batches() -> None:
    """Lease and record every batch until no due Job Dispatch remains."""
    first_dispatch, second_dispatch, third_dispatch = (
        _dispatch(3),
        _dispatch(8),
        _dispatch(13),
    )
    repository = RecordingDispatchRepository(
        ((first_dispatch, second_dispatch), (third_dispatch,), ()),
    )
    publisher = RecordingPublisher()

    assert (
        await TranscriptionJobReconciler(
            repository,
            publisher,
            policy=_policy(dispatch_batch_size=2),
        ).reconcile_once()
        is True
    )

    assert publisher.dispatches == [
        first_dispatch,
        second_dispatch,
        third_dispatch,
    ]
    assert [record[0] for record in repository.recorded_dispatches] == (
        publisher.dispatches
    )
    assert all(request[2] == 2 for request in repository.lease_requests)


@pytest.mark.asyncio
async def test_reconciler_releases_unpublished_suffix_when_broker_is_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Release the failed and unattempted lease suffix after a broker failure."""
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    first_dispatch, second_dispatch, third_dispatch = (
        _dispatch(3),
        _dispatch(8),
        _dispatch(13),
    )
    repository = RecordingDispatchRepository(
        ((first_dispatch, second_dispatch, third_dispatch),),
    )
    publisher = RecordingPublisher(unavailable_after=1)

    assert (
        await TranscriptionJobReconciler(
            repository,
            publisher,
            policy=_policy(),
        ).reconcile_once()
        is False
    )

    assert publisher.dispatches == [first_dispatch]
    assert [record[0] for record in repository.recorded_dispatches] == [first_dispatch]
    assert repository.released_dispatches == [
        ((second_dispatch, third_dispatch), repository.lease_requests[0][0])
    ]
    rendered_event = json.loads(capsys.readouterr().err)
    assert {
        field_name: rendered_event[field_name]
        for field_name in ("event", "internal_job_id", "code")
    } == {
        "event": "job_dispatch_unavailable",
        "internal_job_id": second_dispatch.internal_job_id,
        "code": "transcription_job_dispatch_broker_is_unavailable",
    }
    assert str(second_dispatch.execution_attempt_token) not in str(rendered_event)


@pytest.mark.asyncio
async def test_reconciler_leaves_lease_after_post_publication_store_failure() -> None:
    """Leave the lease to expire when recording fails after broker acceptance."""
    dispatch = _dispatch(3)
    repository = RecordingDispatchRepository(((dispatch,),))
    repository.record_error = TranscriptionJobStoreUnavailableError()
    publisher = RecordingPublisher()

    with pytest.raises(TranscriptionJobStoreUnavailableError):
        await TranscriptionJobReconciler(
            repository,
            publisher,
            policy=_policy(),
        ).reconcile_once()

    assert publisher.dispatches == [dispatch]
    assert repository.released_dispatches == []


@pytest.mark.asyncio
async def test_reconciler_logs_store_failure_then_retries_to_readiness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retain retries while omitting private store-failure context."""
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    stop_event = asyncio.Event()
    repository = StoreFailureThenSuccessRepository()
    reporter = RecordingServiceHeartbeat(stop_event)
    reconciler = TranscriptionJobReconciler(
        repository,
        RecordingPublisher(),
        policy=ReconcilerPolicy(interval=timedelta(milliseconds=1)),
    )

    await reconciler.run(stop_event, reporter)  # type: ignore[arg-type]

    assert reporter.pulse_calls == 1
    rendered_events = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    ]
    failure_events = [
        event for event in rendered_events if event["event"] == "job_store_unavailable"
    ]
    assert len(failure_events) == 1
    rendered_event = failure_events[0]
    assert {
        field_name: rendered_event[field_name] for field_name in ("event", "code")
    } == {
        "event": "job_store_unavailable",
        "code": "transcription_job_storage_is_unavailable_to_reconciler",
    }
    assert "internal_job_id" not in rendered_event
    assert "postgresql-url-sentinel" not in str(rendered_event)


@pytest.mark.asyncio
async def test_reconciler_logs_readiness_store_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retain heartbeat shutdown when readiness storage rejects a refresh."""
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    stop_event = asyncio.Event()
    reporter = FailingServiceHeartbeat(stop_event)
    reconciler = TranscriptionJobReconciler(
        RecordingDispatchRepository(()),
        RecordingPublisher(),
        policy=ReconcilerPolicy(interval=timedelta(milliseconds=1)),
    )

    await reconciler.run(stop_event, reporter)  # type: ignore[arg-type]

    assert reporter.stop_calls == 1
    rendered_events = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    ]
    failure_events = [
        event
        for event in rendered_events
        if event["event"] == "service_readiness_failed"
    ]
    assert len(failure_events) == 1
    rendered_event = failure_events[0]
    assert {
        field_name: rendered_event[field_name]
        for field_name in ("event", "stage", "code")
    } == {
        "event": "service_readiness_failed",
        "stage": "readiness",
        "code": "reconciler_service_readiness_storage_is_unavailable",
    }
    assert "internal_job_id" not in rendered_event
    assert "readiness-error-sentinel" not in str(rendered_event)


@pytest.mark.asyncio
async def test_reconciler_owns_expired_terminal_job_cleanup() -> None:
    """Run terminal retention independently of Job Dispatch publication."""
    repository = RecordingDispatchRepository(())

    await TranscriptionJobReconciler(
        repository,
        RecordingPublisher(),
        policy=_policy(),
    ).cleanup_once()

    assert repository.cleanup_calls == 1


@pytest.mark.asyncio
async def test_reconciler_reports_only_after_a_complete_successful_cycle() -> None:
    """Start, refresh, and remove readiness around one completed reconciliation."""
    stop_event = asyncio.Event()
    repository = RecordingDispatchRepository(())
    reporter = RecordingServiceHeartbeat(stop_event)
    reconciler = TranscriptionJobReconciler(
        repository,
        RecordingPublisher(),
        policy=_policy(),
    )

    await reconciler.run(stop_event, reporter)  # type: ignore[arg-type]

    assert reporter.start_calls == 1
    assert reporter.pulse_calls == 1
    assert reporter.stop_calls == 1
    assert repository.cleanup_calls == 1


@pytest.mark.asyncio
async def test_reconciler_does_not_refresh_readiness_after_broker_failure() -> None:
    """Remove the row without a refresh after a failed broker publication cycle."""
    stop_event = asyncio.Event()
    reporter = RecordingServiceHeartbeat()
    reconciler = TranscriptionJobReconciler(
        RecordingDispatchRepository(((_dispatch(3),),)),
        StopOnBrokerFailurePublisher(stop_event),
        policy=_policy(),
    )

    await reconciler.run(stop_event, reporter)  # type: ignore[arg-type]

    assert reporter.start_calls == 1
    assert reporter.pulse_calls == 0
    assert reporter.stop_calls == 1


@pytest.mark.parametrize(
    "build_policy",
    (
        lambda: ReconcilerPolicy(interval=timedelta()),
        lambda: ReconcilerPolicy(publication_lease_duration=timedelta()),
        lambda: ReconcilerPolicy(dispatch_batch_size=0),
        lambda: ReconcilerPolicy(retention_cleanup_interval=timedelta()),
    ),
)
def test_reconciler_policy_rejects_nonpositive_values(
    build_policy: Callable[[], ReconcilerPolicy],
) -> None:
    """Reject scheduling policies that could strand due dispatches."""
    with pytest.raises(ValueError):
        build_policy()
