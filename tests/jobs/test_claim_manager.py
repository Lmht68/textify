"""Tests for worker-owned Execution Claim management."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from textify.jobs.claim_manager import ExecutionClaimManager
from textify.jobs.contracts import TranscriptionJobStoreUnavailableError
from textify.jobs.types import (
    ClaimedTranscriptionJob,
    ClaimHeartbeat,
    ExecutionClaim,
    JobDispatch,
)
from textify.transcription.service import TranscriptionExecutionControl


class ClaimManagerRepository:
    """Record claim ownership and control authoritative heartbeat responses."""

    def __init__(self) -> None:
        """Initialize an empty in-memory Execution Claim adapter."""
        self.claim_calls: list[tuple[JobDispatch, UUID, timedelta]] = []
        self.heartbeat_batches: list[tuple[ExecutionClaim, ...]] = []
        self.cancellation_requested: set[ExecutionClaim] = set()
        self.omitted_claims: set[ExecutionClaim] = set()
        self.claim_failures_remaining = 0
        self.heartbeat_failures_remaining = 0

    async def claim_dispatch(
        self,
        dispatch: JobDispatch,
        worker_owner: UUID,
        lease_duration: timedelta,
    ) -> ClaimedTranscriptionJob | None:
        """Return one claimed private input for each accepted dispatch."""
        self.claim_calls.append((dispatch, worker_owner, lease_duration))
        if self.claim_failures_remaining:
            self.claim_failures_remaining -= 1
            raise TranscriptionJobStoreUnavailableError()
        return ClaimedTranscriptionJob(
            claim=ExecutionClaim(dispatch, worker_owner),
            submitted_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            exclusions=(),
        )

    async def heartbeat_claims(
        self,
        claims: tuple[ExecutionClaim, ...],
        lease_duration: timedelta,
    ) -> tuple[ClaimHeartbeat, ...]:
        """Record one batch and return configured durable ownership state."""
        self.heartbeat_batches.append(claims)
        if self.heartbeat_failures_remaining:
            self.heartbeat_failures_remaining -= 1
            raise TranscriptionJobStoreUnavailableError()
        return tuple(
            ClaimHeartbeat(
                claim=claim,
                cancellation_requested=claim in self.cancellation_requested,
            )
            for claim in claims
            if claim not in self.omitted_claims
        )


def _dispatch(internal_job_id: int) -> JobDispatch:
    """Build one valid private Execution Attempt dispatch."""
    return JobDispatch(internal_job_id, uuid4())


def _manager(
    repository: ClaimManagerRepository, max_active_claims: int
) -> ExecutionClaimManager:
    """Construct a manager with fast deterministic test timing."""
    return ExecutionClaimManager(
        repository,
        max_active_claims=max_active_claims,
        heartbeat_interval=timedelta(milliseconds=10),
        claim_lease_duration=timedelta(milliseconds=60),
    )


async def _wait_for_heartbeat_count(
    repository: ClaimManagerRepository,
    expected_count: int,
) -> None:
    """Wait for a deterministic number of monitor batches."""
    async with asyncio.timeout(1):
        while len(repository.heartbeat_batches) < expected_count:
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_claim_manager_heartbeats_one_bounded_active_batch() -> None:
    """Renew at most the configured active Execution Claim count in one operation."""
    repository = ClaimManagerRepository()
    manager = _manager(repository, max_active_claims=2)
    first_control = TranscriptionExecutionControl()
    second_control = TranscriptionExecutionControl()
    first_claim: ClaimedTranscriptionJob | None = None
    second_claim: ClaimedTranscriptionJob | None = None
    rejected_claim: ClaimedTranscriptionJob | None = None
    try:
        await manager.start()
        first_claim = await manager.claim(_dispatch(1), first_control)
        second_claim = await manager.claim(_dispatch(2), second_control)
        rejected_claim = await manager.claim(
            _dispatch(3),
            TranscriptionExecutionControl(),
        )
        assert first_claim is not None
        assert second_claim is not None
        await _wait_for_heartbeat_count(repository, 1)
    finally:
        if first_claim is not None:
            manager.release(first_claim.claim)
        if second_claim is not None:
            manager.release(second_claim.claim)
        await manager.shutdown()

    assert first_claim is not None
    assert second_claim is not None
    assert rejected_claim is None
    assert repository.heartbeat_batches[0] == (
        first_claim.claim,
        second_claim.claim,
    )


@pytest.mark.asyncio
async def test_claim_manager_propagates_durable_cancellation_to_both_controls() -> None:
    """Bridge a heartbeat Cancellation response to asyncio and provider events."""
    repository = ClaimManagerRepository()
    manager = _manager(repository, max_active_claims=1)
    control = TranscriptionExecutionControl()
    claimed_job: ClaimedTranscriptionJob | None = None
    try:
        await manager.start()
        claimed_job = await manager.claim(_dispatch(1), control)
        assert claimed_job is not None
        repository.cancellation_requested.add(claimed_job.claim)
        await _wait_for_heartbeat_count(repository, 1)
        async with asyncio.timeout(1):
            await control.cancellation_requested.wait()
    finally:
        if claimed_job is not None:
            manager.release(claimed_job.claim)
        await manager.shutdown()

    assert control.provider_cancellation.is_set()


@pytest.mark.asyncio
async def test_claim_manager_retries_storage_interruption_without_guessing_state() -> (
    None
):
    """Retry the next scheduled batch when PostgreSQL heartbeat storage is unavailable."""
    repository = ClaimManagerRepository()
    repository.heartbeat_failures_remaining = 1
    manager = _manager(repository, max_active_claims=1)
    control = TranscriptionExecutionControl()
    claimed_job: ClaimedTranscriptionJob | None = None
    try:
        await manager.start()
        claimed_job = await manager.claim(_dispatch(1), control)
        assert claimed_job is not None
        await _wait_for_heartbeat_count(repository, 2)
    finally:
        if claimed_job is not None:
            manager.release(claimed_job.claim)
        await manager.shutdown()

    assert claimed_job is not None
    assert not control.cancellation_requested.is_set()
    assert all(batch == (claimed_job.claim,) for batch in repository.heartbeat_batches)


@pytest.mark.asyncio
async def test_claim_manager_marks_omitted_claim_lost_without_renewing_it_again() -> (
    None
):
    """Stop cooperative work permanently when PostgreSQL omits one requested claim."""
    repository = ClaimManagerRepository()
    manager = _manager(repository, max_active_claims=2)
    lost_control = TranscriptionExecutionControl()
    retained_control = TranscriptionExecutionControl()
    lost_job: ClaimedTranscriptionJob | None = None
    retained_job: ClaimedTranscriptionJob | None = None
    try:
        await manager.start()
        lost_job = await manager.claim(_dispatch(1), lost_control)
        retained_job = await manager.claim(_dispatch(2), retained_control)
        assert lost_job is not None
        assert retained_job is not None
        repository.omitted_claims.add(lost_job.claim)
        await _wait_for_heartbeat_count(repository, 1)
        async with asyncio.timeout(1):
            await lost_control.cancellation_requested.wait()
        await _wait_for_heartbeat_count(repository, 2)
    finally:
        if lost_job is not None:
            manager.release(lost_job.claim)
        if retained_job is not None:
            manager.release(retained_job.claim)
        await manager.shutdown()

    assert lost_job is not None
    assert lost_control.provider_cancellation.is_set()
    assert retained_control.cancellation_requested.is_set() is False
    assert all(
        lost_job.claim not in heartbeat_batch
        for heartbeat_batch in repository.heartbeat_batches[1:]
    )


@pytest.mark.asyncio
async def test_claim_manager_rejects_admission_after_shutdown_starts() -> None:
    """Leave delivery unclaimed when the worker has stopped accepting work."""
    repository = ClaimManagerRepository()
    manager = _manager(repository, max_active_claims=1)
    claimed_job: ClaimedTranscriptionJob | None = None
    try:
        await manager.start()
        manager.stop_accepting()
        claimed_job = await manager.claim(_dispatch(1), TranscriptionExecutionControl())
    finally:
        await manager.shutdown()

    assert claimed_job is None
    assert repository.claim_calls == []


@pytest.mark.asyncio
async def test_claim_manager_recovers_drain_state_after_claim_storage_failure() -> None:
    """Allow shutdown after an unavailable claim write releases pending capacity."""
    repository = ClaimManagerRepository()
    repository.claim_failures_remaining = 1
    manager = _manager(repository, max_active_claims=1)
    await manager.start()
    try:
        with pytest.raises(TranscriptionJobStoreUnavailableError):
            await manager.claim(_dispatch(1), TranscriptionExecutionControl())

        async with asyncio.timeout(1):
            await manager.shutdown()
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_claim_manager_shutdown_waits_for_registered_execution_release() -> None:
    """Keep monitoring an admitted claim until its processor releases it."""
    repository = ClaimManagerRepository()
    manager = _manager(repository, max_active_claims=1)
    claimed_job: ClaimedTranscriptionJob | None = None
    shutdown_task: asyncio.Task[None] | None = None
    try:
        await manager.start()
        claimed_job = await manager.claim(_dispatch(1), TranscriptionExecutionControl())
        assert claimed_job is not None
        shutdown_task = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        await _wait_for_heartbeat_count(repository, 1)
        manager.release(claimed_job.claim)
        await shutdown_task
    finally:
        if shutdown_task is not None and not shutdown_task.done():
            if claimed_job is not None:
                manager.release(claimed_job.claim)
            await shutdown_task

    assert repository.heartbeat_batches
