"""Tests for PostgreSQL-driven Job Dispatch reconciliation."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
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


def test_reconciler_main_bootstraps_json_logging_before_configured_debug_level() -> (
    None
):
    """Emit lifecycle records before applying the configured debug level."""
    child_program = textwrap.dedent(
        """
        from types import SimpleNamespace

        from textify.jobs import reconciler


        async def run_reconciler():
            reconciler.emit_event(
                reconciler.logger,
                reconciler.LogEvent.SOURCE_INSPECTION_FAILED,
                stage="metadata",
                code="configured-debug-event",
            )


        reconciler.AppConfig = lambda: SimpleNamespace(log_level="DEBUG")
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
        "source_inspection_failed",
        "service_stopping",
    ]
    assert rendered_events[1]["level"] == "DEBUG"
    assert {
        field_name: rendered_events[1][field_name] for field_name in ("stage", "code")
    } == {
        "stage": "metadata",
        "code": "configured-debug-event",
    }
    assert {event["service_role"] for event in rendered_events} == {"reconciler"}
    instance_ids = {event["instance_id"] for event in rendered_events}
    assert len(instance_ids) == 1
    instance_id = instance_ids.pop()
    assert UUID(instance_id).version == 4
    assert str(UUID(instance_id)) == instance_id


def test_reconciler_main_reports_startup_failure_without_exception_output() -> None:
    """Report invalid startup configuration as one safe process failure."""
    completed = subprocess.run(
        [sys.executable, "-c", "from textify.jobs.reconciler import main; main()"],
        capture_output=True,
        check=False,
        env={**os.environ, "TEXTIFY_LOG_LEVEL": "invalid-log-level-sentinel"},
        text=True,
    )

    assert completed.returncode != 0
    rendered_events = [json.loads(line) for line in completed.stderr.splitlines()]
    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "service_failed",
    ]
    failure_events = [
        event for event in rendered_events if event["event"] == "service_failed"
    ]
    assert len(failure_events) == 1
    assert {
        field_name: failure_events[0][field_name] for field_name in ("stage", "code")
    } == {
        "stage": "startup",
        "code": "internal_error",
    }
    assert "service_stopping" not in {event["event"] for event in rendered_events}
    for unsafe_text in (
        "invalid-log-level-sentinel",
        "TEXTIFY_LOG_LEVEL",
        "ValidationError",
        "ValueError",
        "Traceback",
    ):
        assert unsafe_text not in completed.stderr


def test_reconciler_main_reports_runtime_failure_without_exception_output() -> None:
    """Report an unexpected reconciliation failure as a safe runtime failure."""
    child_program = textwrap.dedent(
        """
        import asyncio

        from textify.jobs import reconciler


        class Repository:
            async def recover_expired_claims(self, limit):
                raise RuntimeError("runtime-error-sentinel")


        class Publisher:
            async def publish(self, dispatch):
                return None


        class Heartbeat:
            async def start(self):
                return None

            async def pulse(self):
                return None

            async def stop(self):
                return None


        async def run_reconciler():
            await reconciler.TranscriptionJobReconciler(
                Repository(),
                Publisher(),
            ).run(asyncio.Event(), Heartbeat())


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

    assert completed.returncode != 0
    rendered_events = [json.loads(line) for line in completed.stderr.splitlines()]
    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "service_ready",
        "service_failed",
    ]
    failure_events = [
        event for event in rendered_events if event["event"] == "service_failed"
    ]
    assert len(failure_events) == 1
    assert {
        field_name: failure_events[0][field_name] for field_name in ("stage", "code")
    } == {
        "stage": "runtime",
        "code": "internal_error",
    }
    assert "service_stopping" not in {event["event"] for event in rendered_events}
    for unsafe_text in ("runtime-error-sentinel", "RuntimeError", "Traceback"):
        assert unsafe_text not in completed.stderr


def test_reconciler_main_reports_shutdown_failure_without_exception_output() -> None:
    """Report an unexpected heartbeat removal failure as safe shutdown failure."""
    child_program = textwrap.dedent(
        """
        import asyncio

        from textify.jobs import reconciler


        class Repository:
            async def recover_expired_claims(self, limit):
                return 0

            async def expire_queued_jobs(self, limit):
                return 0

            async def lease_due_dispatches(self, lease_owner, lease_duration, limit):
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
                return None

            async def stop(self):
                raise RuntimeError("shutdown-error-sentinel")


        async def run_reconciler():
            stop_event = asyncio.Event()
            stop_event.set()
            await reconciler.TranscriptionJobReconciler(
                Repository(),
                Publisher(),
            ).run(stop_event, Heartbeat())


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

    assert completed.returncode != 0
    rendered_events = [json.loads(line) for line in completed.stderr.splitlines()]
    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "service_ready",
        "service_failed",
    ]
    failure_events = [
        event for event in rendered_events if event["event"] == "service_failed"
    ]
    assert len(failure_events) == 1
    assert {
        field_name: failure_events[0][field_name] for field_name in ("stage", "code")
    } == {
        "stage": "shutdown",
        "code": "internal_error",
    }
    assert "service_stopping" not in {event["event"] for event in rendered_events}
    for unsafe_text in ("shutdown-error-sentinel", "RuntimeError", "Traceback"):
        assert unsafe_text not in completed.stderr


@pytest.mark.asyncio
async def test_reconciler_process_stops_on_sigterm_with_only_json_stderr(
    migrated_postgresql_database_url: str,
    redis_broker_url: str,
) -> None:
    """Stop the real reconciler cleanly when the process receives SIGTERM."""
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "TEXTIFY_DATABASE_URL": migrated_postgresql_database_url,
            "TEXTIFY_BROKER_URL": redis_broker_url,
            "TEXTIFY_LOG_LEVEL": "INFO",
            "TEXTIFY_RECONCILER_INTERVAL_SECONDS": "0.01",
        }
    )
    child_environment.pop("TEXTIFY_CACHE_URL", None)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "from textify.jobs.reconciler import main; main()",
        env=child_environment,
        stderr=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
    )
    rendered_events: list[dict[str, object]] = []
    try:
        assert process.stderr is not None
        ready_deadline = time.monotonic() + 10
        while True:
            remaining_seconds = ready_deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise AssertionError("Reconciler did not reach service_ready in time.")
            rendered_line = (
                await asyncio.wait_for(
                    process.stderr.readline(),
                    timeout=remaining_seconds,
                )
            ).decode()
            if not rendered_line:
                raise AssertionError("Reconciler exited before emitting service_ready.")
            try:
                rendered_event = json.loads(rendered_line)
            except json.JSONDecodeError:
                raise AssertionError(
                    "Reconciler emitted a non-JSON stderr line."
                ) from None
            if not isinstance(rendered_event, dict) or not all(
                isinstance(field_name, str) for field_name in rendered_event
            ):
                raise AssertionError(
                    "Reconciler emitted an invalid JSON event envelope."
                )
            rendered_events.append(rendered_event)
            if rendered_event.get("event") == "service_ready":
                break

        process.send_signal(signal.SIGTERM)
        _, remaining_stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=10,
        )
        assert remaining_stderr is not None
        remaining_stderr_text = remaining_stderr.decode()
        for rendered_line in remaining_stderr_text.splitlines():
            try:
                rendered_event = json.loads(rendered_line)
            except json.JSONDecodeError:
                raise AssertionError(
                    "Reconciler emitted a non-JSON stderr line."
                ) from None
            if not isinstance(rendered_event, dict) or not all(
                isinstance(field_name, str) for field_name in rendered_event
            ):
                raise AssertionError(
                    "Reconciler emitted an invalid JSON event envelope."
                )
            rendered_events.append(rendered_event)

        assert process.returncode == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()

    assert [event["event"] for event in rendered_events] == [
        "service_starting",
        "service_ready",
        "service_stopping",
    ]
    envelope_fields = {
        "timestamp",
        "level",
        "logger",
        "message",
        "event",
        "service",
        "service_role",
        "instance_id",
    }
    string_fields = {
        "timestamp",
        "level",
        "logger",
        "message",
        "event",
        "service",
        "service_role",
        "instance_id",
        "request_id",
        "platform",
        "source_id",
        "stage",
        "method",
        "code",
        "job_status",
        "job_outcome",
        "http_method",
        "http_route",
    }
    integer_fields = {"internal_job_id", "duration_seconds", "http_status_code"}
    number_fields = {"elapsed_seconds"}
    for event in rendered_events:
        assert envelope_fields <= event.keys()
        for field_name, field_value in event.items():
            if field_name in string_fields:
                assert type(field_value) is str
            elif field_name in integer_fields:
                assert type(field_value) is int
            elif field_name in number_fields:
                assert type(field_value) in {int, float}
            else:
                raise AssertionError("Reconciler emitted an unsupported log field.")
    assert {event["service_role"] for event in rendered_events} == {"reconciler"}
    assert {event["service"] for event in rendered_events} == {"textify"}
    instance_id = rendered_events[0]["instance_id"]
    assert isinstance(instance_id, str)
    assert {event["instance_id"] for event in rendered_events} == {instance_id}
    assert UUID(instance_id).version == 4
    assert str(UUID(instance_id)) == instance_id
    rendered_stderr = "\n".join(json.dumps(event) for event in rendered_events)
    if migrated_postgresql_database_url in rendered_stderr:
        raise AssertionError("Reconciler exposed the database URL in stderr.")
    if redis_broker_url in rendered_stderr:
        raise AssertionError("Reconciler exposed the broker URL in stderr.")
    for unsafe_text in ("Traceback", "Exception", "Error"):
        if unsafe_text in rendered_stderr:
            raise AssertionError("Reconciler emitted exception text.")


@pytest.mark.asyncio
async def test_reconciler_recovers_claims_before_expiry_and_dispatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drain expired claims before queued deadlines and new broker publication."""
    configure_structured_logging(LogServiceRole.API, "DEBUG")
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
    assert capsys.readouterr().err == ""


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
    for dispatch in (first_dispatch, second_dispatch, third_dispatch):
        assert str(dispatch.execution_attempt_token) not in str(rendered_event)
    recorded_lease_owner = repository.recorded_dispatches[0][1]
    assert str(recorded_lease_owner) not in str(rendered_event)
    assert rendered_event["instance_id"] != str(recorded_lease_owner)


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
async def test_reconciler_owns_expired_terminal_job_cleanup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run terminal retention independently of Job Dispatch publication."""
    configure_structured_logging(LogServiceRole.API, "DEBUG")
    repository = RecordingDispatchRepository(())

    await TranscriptionJobReconciler(
        repository,
        RecordingPublisher(),
        policy=_policy(),
    ).cleanup_once()

    assert repository.cleanup_calls == 1
    assert capsys.readouterr().err == ""


@pytest.mark.asyncio
async def test_reconciler_reports_only_after_a_complete_successful_cycle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Start, refresh, and remove readiness around one completed reconciliation."""
    configure_structured_logging(LogServiceRole.API, "DEBUG")
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
    rendered_events = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    ]
    assert [event["event"] for event in rendered_events] == ["service_ready"]


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
