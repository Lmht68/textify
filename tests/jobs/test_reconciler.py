"""Tests for PostgreSQL-driven Job Dispatch reconciliation."""

from uuid import uuid4

import pytest

from textify.jobs.types import JobDispatch


class RecordingDispatchRepository:
    """Expose queued dispatches and record reconciler-owned cleanup."""

    def __init__(self, dispatches: tuple[JobDispatch, ...]) -> None:
        """Initialize a stable private queued dispatch listing."""
        self.dispatches = dispatches
        self.list_calls = 0
        self.cleanup_calls = 0

    async def list_queued_dispatches(self) -> tuple[JobDispatch, ...]:
        """Return the configured delivery candidates."""
        self.list_calls += 1
        return self.dispatches

    async def delete_expired_terminal_jobs(self) -> None:
        """Record one reconciler-owned retention pass."""
        self.cleanup_calls += 1


class RecordingPublisher:
    """Record dispatch publication and optionally reject a broker write."""

    def __init__(self) -> None:
        """Initialize publication records and no broker failure."""
        self.dispatches: list[JobDispatch] = []
        self.fail_after: int | None = None

    async def publish(self, dispatch: JobDispatch) -> None:
        """Record a publish or signal that the delivery broker is unavailable."""
        from textify.jobs.dispatch import JobDispatchUnavailableError

        if self.fail_after is not None and len(self.dispatches) >= self.fail_after:
            raise JobDispatchUnavailableError()
        self.dispatches.append(dispatch)


def _dispatch(internal_job_id: int) -> JobDispatch:
    """Build one valid private dispatch.

    Args:
        internal_job_id: Positive PostgreSQL job identity.

    Returns:
        A private notification for one Execution Attempt.
    """
    return JobDispatch(internal_job_id, uuid4())


@pytest.mark.asyncio
async def test_reconciler_republishes_every_queued_dispatch_in_order() -> None:
    """Publish the stable PostgreSQL dispatch list without loading job inputs."""
    from textify.jobs.reconciler import TranscriptionJobReconciler

    dispatches = (_dispatch(3), _dispatch(8))
    repository = RecordingDispatchRepository(dispatches)
    publisher = RecordingPublisher()

    await TranscriptionJobReconciler(repository, publisher).reconcile_once()

    assert repository.list_calls == 1
    assert publisher.dispatches == list(dispatches)
    assert repository.cleanup_calls == 0


@pytest.mark.asyncio
async def test_reconciler_stops_current_pass_when_broker_is_unavailable() -> None:
    """Leave remaining deliveries for a later idempotent reconciliation pass."""
    from textify.jobs.reconciler import TranscriptionJobReconciler

    first_dispatch, second_dispatch = _dispatch(3), _dispatch(8)
    repository = RecordingDispatchRepository((first_dispatch, second_dispatch))
    publisher = RecordingPublisher()
    publisher.fail_after = 1

    await TranscriptionJobReconciler(repository, publisher).reconcile_once()

    assert publisher.dispatches == [first_dispatch]


@pytest.mark.asyncio
async def test_reconciler_owns_expired_terminal_job_cleanup() -> None:
    """Run terminal retention independently of queued Job Dispatch publication."""
    from textify.jobs.reconciler import TranscriptionJobReconciler

    repository = RecordingDispatchRepository(())

    await TranscriptionJobReconciler(repository, RecordingPublisher()).cleanup_once()

    assert repository.list_calls == 0
    assert repository.cleanup_calls == 1
