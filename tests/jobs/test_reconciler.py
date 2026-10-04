"""Tests for PostgreSQL-driven Job Dispatch reconciliation."""

from collections import deque
from collections.abc import Callable
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from textify.jobs.contracts import TranscriptionJobStoreUnavailableError
from textify.jobs.dispatch import JobDispatchUnavailableError
from textify.jobs.reconciler import ReconcilerPolicy, TranscriptionJobReconciler
from textify.jobs.types import JobDispatch


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

    await TranscriptionJobReconciler(
        repository,
        publisher,
        policy=_policy(dispatch_batch_size=2),
    ).reconcile_once()

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

    await TranscriptionJobReconciler(
        repository,
        publisher,
        policy=_policy(dispatch_batch_size=2),
    ).reconcile_once()

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
async def test_reconciler_releases_unpublished_suffix_when_broker_is_unavailable() -> (
    None
):
    """Release the failed and unattempted lease suffix after a broker failure."""
    first_dispatch, second_dispatch, third_dispatch = (
        _dispatch(3),
        _dispatch(8),
        _dispatch(13),
    )
    repository = RecordingDispatchRepository(
        ((first_dispatch, second_dispatch, third_dispatch),),
    )
    publisher = RecordingPublisher(unavailable_after=1)

    await TranscriptionJobReconciler(
        repository,
        publisher,
        policy=_policy(),
    ).reconcile_once()

    assert publisher.dispatches == [first_dispatch]
    assert [record[0] for record in repository.recorded_dispatches] == [first_dispatch]
    assert repository.released_dispatches == [
        ((second_dispatch, third_dispatch), repository.lease_requests[0][0])
    ]


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
async def test_reconciler_owns_expired_terminal_job_cleanup() -> None:
    """Run terminal retention independently of Job Dispatch publication."""
    repository = RecordingDispatchRepository(())

    await TranscriptionJobReconciler(
        repository,
        RecordingPublisher(),
        policy=_policy(),
    ).cleanup_once()

    assert repository.cleanup_calls == 1


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
