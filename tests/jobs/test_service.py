"""Tests for FastAPI-owned durable job lifecycle coordination."""

import json
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

from textify.jobs.contracts import (
    TranscriptionJobApiRepository,
    TranscriptionJobStoreUnavailableError,
)
from textify.jobs.exceptions import JobStoreUnavailableError
from textify.jobs.service import TranscriptionJobService
from textify.jobs.types import (
    CancelledTranscriptionJob,
    JobOutcome,
    JobStatus,
    ProcessingTranscriptionJob,
    QueuedTranscriptionJob,
    TranscriptionJobCancellationResult,
)
from textify.logging import (
    LogServiceRole,
    configure_structured_logging,
    request_log_context,
)


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
    """Return one committed processing Cancellation result."""

    async def request_cancellation(
        self, _public_id: object
    ) -> TranscriptionJobCancellationResult:
        """Accept cancellation without exposing private execution ownership."""
        now = datetime.now(UTC)
        return TranscriptionJobCancellationResult(
            internal_job_id=41,
            job=ProcessingTranscriptionJob(
                public_id=uuid4(),
                status=JobStatus.PROCESSING,
                submitted_at=now,
                started_at=now,
                cancellation_requested=True,
            ),
            first_accepted=True,
        )


class LifecycleLoggingRepository:
    """Return committed admission and Cancellation results for log assertions."""

    def __init__(self) -> None:
        """Initialize one fixed Transcription Job lifecycle."""
        now = datetime.now(UTC)
        public_id = uuid4()
        processing_job = ProcessingTranscriptionJob(
            public_id=public_id,
            status=JobStatus.PROCESSING,
            submitted_at=now,
            started_at=now,
            cancellation_requested=True,
        )
        cancelled_job = CancelledTranscriptionJob(
            public_id=public_id,
            status=JobStatus.FINISHED,
            outcome=JobOutcome.CANCELLED,
            submitted_at=now,
            started_at=None,
            finished_at=now + timedelta(seconds=1),
        )
        self._queued_job = QueuedTranscriptionJob(
            internal_id=41,
            public_id=public_id,
            status=JobStatus.QUEUED,
            submitted_at=now,
            queue_deadline_at=now + timedelta(seconds=20),
        )
        self._cancellations = (
            TranscriptionJobCancellationResult(41, processing_job, True),
            TranscriptionJobCancellationResult(41, processing_job, False),
            TranscriptionJobCancellationResult(41, cancelled_job, True),
        )
        self._cancellation_index = 0

    async def create_queued(self, _job: object) -> QueuedTranscriptionJob:
        """Return the fixed committed queued projection."""
        return self._queued_job

    async def request_cancellation(
        self, _public_id: object
    ) -> TranscriptionJobCancellationResult:
        """Return the next fixed committed Cancellation result."""
        result = self._cancellations[self._cancellation_index]
        self._cancellation_index += 1
        return result


@pytest.mark.asyncio
async def test_service_latches_store_failure_and_closes_subsequent_admission() -> None:
    """Fail closed after durable admission becomes unavailable."""
    repository = StoreFailingRepository()
    unavailable_transitions: list[None] = []
    service = TranscriptionJobService(
        cast(TranscriptionJobApiRepository, repository),
        on_store_unavailable=lambda: unavailable_transitions.append(None),
    )

    with pytest.raises(JobStoreUnavailableError):
        await service.submit("https://youtu.be/AbCdEf12345", frozenset())
    with pytest.raises(JobStoreUnavailableError):
        await service.submit("https://youtu.be/AbCdEf12345", frozenset())

    assert repository.admission_calls == 1
    assert unavailable_transitions == [None]


@pytest.mark.asyncio
async def test_service_returns_committed_processing_cancellation_without_signal() -> (
    None
):
    """Return the persisted result without a worker-local side effect."""
    repository = CancellingRepository()
    service = TranscriptionJobService(
        cast(TranscriptionJobApiRepository, repository),
        on_store_unavailable=lambda: None,
    )

    cancellation = await service.cancel(str(uuid4()))

    assert cancellation.internal_job_id == 41
    assert cancellation.first_accepted is True
    assert isinstance(cancellation.job, ProcessingTranscriptionJob)
    assert cancellation.job.cancellation_requested is True


@pytest.mark.asyncio
async def test_service_emits_committed_admission_and_cancellation_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit each API lifecycle event only after its database result is committed."""
    request_identifier = "f8b2d0be-09be-4de9-bb79-f8c26dcbd568"
    repository = LifecycleLoggingRepository()
    service = TranscriptionJobService(
        cast(TranscriptionJobApiRepository, repository),
        on_store_unavailable=lambda: None,
    )
    configure_structured_logging(LogServiceRole.API, "DEBUG")

    with request_log_context(request_identifier):
        admission = await service.submit(
            "https://www.youtube.com/watch?v=AbCdEf12345",
            frozenset(),
        )
        first_processing = await service.cancel(str(uuid4()))
        repeated_processing = await service.cancel(str(uuid4()))
        first_queued = await service.cancel(str(uuid4()))

    rendered_events = [
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    ]
    assert admission.internal_id == 41
    assert first_processing.first_accepted is True
    assert repeated_processing.first_accepted is False
    assert first_queued.first_accepted is True
    assert [event["event"] for event in rendered_events] == [
        "transcription_job_admitted",
        "transcription_job_cancellation_accepted",
        "transcription_job_cancellation_accepted",
        "transcription_job_finished",
    ]
    assert [event["request_id"] for event in rendered_events] == [
        request_identifier,
        request_identifier,
        request_identifier,
        request_identifier,
    ]
    assert [event["internal_job_id"] for event in rendered_events] == [41, 41, 41, 41]
    assert [event["job_status"] for event in rendered_events] == [
        "queued",
        "processing",
        "finished",
        "finished",
    ]
    assert "job_outcome" not in rendered_events[1]
    assert rendered_events[2]["job_outcome"] == "cancelled"
    assert rendered_events[3]["job_outcome"] == "cancelled"
