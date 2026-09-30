"""Tests for FastAPI-owned durable job lifecycle coordination."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from textify.jobs.contracts import TranscriptionJobStoreUnavailableError
from textify.jobs.exceptions import JobStoreUnavailableError
from textify.jobs.service import TranscriptionJobService
from textify.jobs.types import JobStatus, ProcessingTranscriptionJob


class StoreFailingRepository:
    """Raise a durable-store failure for every attempted job admission."""

    def __init__(self) -> None:
        """Initialize an observable admission counter."""
        self.admission_calls = 0

    async def create_queued(self, _job: object) -> object:
        """Record and reject one durable admission."""
        self.admission_calls += 1
        raise TranscriptionJobStoreUnavailableError()


class CancellingRepository:
    """Return one safe processing snapshot for cancellation acceptance."""

    async def request_cancellation(
        self, _public_id: object
    ) -> ProcessingTranscriptionJob:
        """Accept cancellation without exposing private execution ownership."""
        now = datetime.now(UTC)
        return ProcessingTranscriptionJob(
            public_id=uuid4(),
            status=JobStatus.PROCESSING,
            submitted_at=now,
            started_at=now,
            cancellation_requested=True,
        )


@pytest.mark.asyncio
async def test_service_latches_store_failure_and_closes_subsequent_admission() -> None:
    """Fail closed after durable admission becomes unavailable."""
    repository = StoreFailingRepository()
    unavailable_transitions: list[None] = []
    service = TranscriptionJobService(
        repository,
        on_store_unavailable=lambda: unavailable_transitions.append(None),
    )

    with pytest.raises(JobStoreUnavailableError):
        await service.submit("https://youtu.be/AbCdEf12345", frozenset())
    with pytest.raises(JobStoreUnavailableError):
        await service.submit("https://youtu.be/AbCdEf12345", frozenset())

    assert repository.admission_calls == 1
    assert unavailable_transitions == [None]


@pytest.mark.asyncio
async def test_service_returns_processing_cancellation_without_execution_signal() -> (
    None
):
    """Return the persisted processing snapshot without a worker-local side effect."""
    repository = CancellingRepository()
    service = TranscriptionJobService(repository, on_store_unavailable=lambda: None)

    cancellation = await service.cancel(str(uuid4()))

    assert isinstance(cancellation, ProcessingTranscriptionJob)
    assert cancellation.cancellation_requested is True
