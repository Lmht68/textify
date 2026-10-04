"""Process-local heartbeat and Cancellation management for Execution Claims."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Protocol
from uuid import UUID, uuid4

from textify.jobs.contracts import TranscriptionJobStoreUnavailableError
from textify.jobs.types import (
    ClaimedTranscriptionJob,
    ClaimHeartbeat,
    ExecutionClaim,
    JobDispatch,
)
from textify.transcription.service import TranscriptionExecutionControl

logger = logging.getLogger(__name__)


class _ExecutionClaimRepository(Protocol):
    """Provide the durable claim operations required by one worker process."""

    async def claim_dispatch(
        self,
        dispatch: JobDispatch,
        worker_owner: UUID,
        lease_duration: timedelta,
    ) -> ClaimedTranscriptionJob | None:
        """Claim one delivered Execution Attempt for one worker owner."""
        ...

    async def heartbeat_claims(
        self,
        claims: tuple[ExecutionClaim, ...],
        lease_duration: timedelta,
    ) -> tuple[ClaimHeartbeat, ...]:
        """Renew a bounded batch and return authoritative Cancellation state."""
        ...


class ExecutionClaimManager:
    """Own bounded worker claims and their durable heartbeat monitor."""

    def __init__(
        self,
        repository: _ExecutionClaimRepository,
        *,
        max_active_claims: int,
        heartbeat_interval: timedelta,
        claim_lease_duration: timedelta,
        claim_ready: Callable[[], Awaitable[bool]],
    ) -> None:
        """Initialize one worker identity and bounded claim registry.

        Args:
            repository: Durable Execution Attempt claim and heartbeat seam.
            max_active_claims: Maximum processing executions accepted by this worker.
            heartbeat_interval: Positive interval between PostgreSQL heartbeat batches.
            claim_lease_duration: Positive lifetime renewed for every live claim.
            claim_ready: Ownership health probe required before each durable claim.

        Raises:
            ValueError: If the capacity or either duration is nonpositive.
        """
        if (
            isinstance(max_active_claims, bool)
            or not isinstance(max_active_claims, int)
            or max_active_claims <= 0
        ):
            raise ValueError("max_active_claims must be a positive integer.")
        _require_positive_duration(heartbeat_interval, "heartbeat_interval")
        _require_positive_duration(claim_lease_duration, "claim_lease_duration")
        self._repository = repository
        self._worker_owner = uuid4()
        self._max_active_claims = max_active_claims
        self._heartbeat_interval = heartbeat_interval
        self._claim_lease_duration = claim_lease_duration
        self._claim_ready = claim_ready
        self._claims: dict[ExecutionClaim, TranscriptionExecutionControl] = {}
        self._ownership_lost: set[ExecutionClaim] = set()
        self._pending_claims = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._accepting = True
        self._started = False
        self._shutdown = False
        self._monitor_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the worker-owned heartbeat monitor.

        Raises:
            RuntimeError: If startup is repeated or shutdown has already started.
        """
        if self._started or self._shutdown:
            raise RuntimeError("Execution claim manager cannot be started.")
        self._started = True
        self._monitor_task = asyncio.create_task(
            self._monitor(),
            name="textify-execution-claim-monitor",
        )

    async def claim(
        self,
        dispatch: JobDispatch,
        control: TranscriptionExecutionControl,
    ) -> ClaimedTranscriptionJob | None:
        """Claim one delivery when this worker can safely execute it.

        Args:
            dispatch: Private delivered Execution Attempt identity.
            control: Local cooperative controls for the prospective execution.

        Returns:
            Claimed private job inputs, or ``None`` when admission is closed or full.

        Raises:
            RuntimeError: If the manager has not started.
            TranscriptionJobStoreUnavailableError: If PostgreSQL cannot claim safely.
        """
        if not self._started:
            raise RuntimeError("Execution claim manager has not started.")
        if (
            not self._accepting
            or len(self._claims) + self._pending_claims >= self._max_active_claims
        ):
            return None
        if not await self._claim_ready():
            return None
        if (
            not self._accepting
            or len(self._claims) + self._pending_claims >= self._max_active_claims
        ):
            return None
        self._pending_claims += 1
        self._update_drained_state()
        try:
            claimed_job = await self._repository.claim_dispatch(
                dispatch,
                self._worker_owner,
                self._claim_lease_duration,
            )
        finally:
            self._pending_claims -= 1
            self._update_drained_state()
        if claimed_job is not None:
            if (
                claimed_job.claim.dispatch != dispatch
                or claimed_job.claim.worker_owner != self._worker_owner
            ):
                raise TranscriptionJobStoreUnavailableError()
            self._claims[claimed_job.claim] = control
        self._update_drained_state()
        return claimed_job

    def release(self, claim: ExecutionClaim) -> None:
        """Remove an execution that completed every terminal publication path.

        Args:
            claim: Claimed Execution Attempt whose coroutine is fully finished.
        """
        self._claims.pop(claim, None)
        self._ownership_lost.discard(claim)
        self._update_drained_state()

    def stop_accepting(self) -> None:
        """Close claim admission while allowing registered executions to drain."""
        self._accepting = False

    async def shutdown(self) -> None:
        """Drain registered work while retaining heartbeats, then stop the monitor."""
        self.stop_accepting()
        self._shutdown = True
        await self._drained.wait()
        monitor_task = self._monitor_task
        self._monitor_task = None
        if monitor_task is None:
            return
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    async def _monitor(self) -> None:
        """Heartbeat active claims and bridge durable state to local controls."""
        while True:
            await asyncio.sleep(self._heartbeat_interval.total_seconds())
            claims = tuple(
                claim for claim in self._claims if claim not in self._ownership_lost
            )
            if not claims:
                continue
            try:
                heartbeats = await self._repository.heartbeat_claims(
                    claims,
                    self._claim_lease_duration,
                )
            except TranscriptionJobStoreUnavailableError:
                logger.error(
                    "transcription job storage is unavailable to claim monitor"
                )
                continue

            heartbeat_by_claim = {
                heartbeat.claim: heartbeat for heartbeat in heartbeats
            }
            for claim in claims:
                control = self._claims.get(claim)
                if control is None:
                    continue
                heartbeat = heartbeat_by_claim.get(claim)
                if heartbeat is None:
                    self._ownership_lost.add(claim)
                    control.request_cancellation()
                    continue
                if heartbeat.cancellation_requested:
                    control.request_cancellation()

    def _update_drained_state(self) -> None:
        """Reflect whether no admitted execution can require a heartbeat."""
        if self._pending_claims == 0 and not self._claims:
            self._drained.set()
            return
        self._drained.clear()


def _require_positive_duration(value: timedelta, name: str) -> None:
    """Reject a nonpositive manager timing value.

    Args:
        value: Duration requiring strictly positive time.
        name: Configuration name used in the validation error.

    Raises:
        ValueError: If the value is not a positive duration.
    """
    if not isinstance(value, timedelta) or value <= timedelta():
        raise ValueError(f"{name} must be a positive timedelta.")
