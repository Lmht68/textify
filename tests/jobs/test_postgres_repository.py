"""Cross-session PostgreSQL proofs for Transcription Job lifecycle storage."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

from textify.jobs.contracts import TranscriptionJobCapacityError
from textify.jobs.database import create_application_engine
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.types import (
    CancelledTranscriptionJob,
    NewQueuedTranscriptionJob,
    ProcessingCancellation,
    QueuedTranscriptionJob,
    SucceededTranscriptionJob,
)
from textify.transcription.schemas import TranscriptionResponse
from textify.transcription.types import Platform, TranscriptMethod

_INITIAL = datetime(2026, 8, 24, tzinfo=UTC)
_QUEUE_TIMEOUT = timedelta(minutes=1)
_TERMINAL_RETENTION = timedelta(days=1)


def _repository(
    database_url: str,
    *,
    maximum_outstanding_jobs: int,
    clock: list[datetime],
) -> tuple[PostgresTranscriptionJobRepository, AsyncEngine]:
    """Create one repository and its independent PostgreSQL application engine."""
    engine = create_application_engine(database_url)
    return (
        PostgresTranscriptionJobRepository(
            engine,
            maximum_outstanding_jobs=maximum_outstanding_jobs,
            queue_timeout=_QUEUE_TIMEOUT,
            terminal_retention=_TERMINAL_RETENTION,
            clock=lambda: clock[0],
        ),
        engine,
    )


async def _dispose_engines(*engines: AsyncEngine) -> None:
    """Close independent PostgreSQL application engines owned by this test."""
    await asyncio.gather(*(engine.dispose() for engine in engines))


def _queued_job(source_suffix: str) -> NewQueuedTranscriptionJob:
    """Create one valid private queued-job input with a distinct source URL."""
    return NewQueuedTranscriptionJob(
        submitted_url=f"https://www.youtube.com/watch?v=AbCdEf12345&job={source_suffix}",
        exclusions=(),
    )


def _projected_result() -> TranscriptionResponse:
    """Create one complete, independently known successful response projection."""
    return {
        "source": {
            "platform": Platform.YOUTUBE,
            "video_id": "AbCdEf12345",
            "url": "https://www.youtube.com/watch?v=AbCdEf12345",
            "title": "Repository result",
            "description": "",
            "channel": "Textify",
            "duration_seconds": 1,
        },
        "transcript": {
            "method": TranscriptMethod.YOUTUBE_CAPTIONS,
            "language": "en",
            "text": "Repository result.",
            "segments": ({"start": 0.0, "end": 1.0, "text": "Repository result."},),
        },
    }


async def _wait_for_result_read(
    result_read_started: asyncio.Event,
) -> None:
    """Wait until a repeatable-read result lookup has established its snapshot."""
    try:
        async with asyncio.timeout(1):
            await result_read_started.wait()
    except TimeoutError as exc:
        raise AssertionError(
            "The repeatable-read lookup did not reach its result read."
        ) from exc


async def test_postgres_repositories_enforce_capacity_across_sessions(
    migrated_postgresql_database_url: str,
) -> None:
    """Allow only one concurrent admission across independent PostgreSQL engines."""
    clock = [_INITIAL]
    first_repository, first_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    second_repository, second_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    try:
        admissions = await asyncio.gather(
            first_repository.create_queued(_queued_job("first")),
            second_repository.create_queued(_queued_job("second")),
            return_exceptions=True,
        )
    finally:
        await _dispose_engines(first_engine, second_engine)

    admitted_jobs = [
        admission
        for admission in admissions
        if isinstance(admission, QueuedTranscriptionJob)
    ]
    capacity_errors = [
        admission
        for admission in admissions
        if isinstance(admission, TranscriptionJobCapacityError)
    ]
    assert len(admitted_jobs) == 1
    assert len(capacity_errors) == 1


async def test_postgres_repositories_claim_jobs_in_fifo_order_across_sessions(
    migrated_postgresql_database_url: str,
) -> None:
    """Claim submitted jobs by age even when distinct repository sessions claim them."""
    clock = [_INITIAL]
    submitting_repository, submitting_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=3,
        clock=clock,
    )
    first_consumer_repository, first_consumer_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=3,
        clock=clock,
    )
    second_consumer_repository, second_consumer_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=3,
        clock=clock,
    )
    try:
        first_queued = await submitting_repository.create_queued(_queued_job("first"))
        second_queued = await submitting_repository.create_queued(_queued_job("second"))

        first_claim = await first_consumer_repository.claim_next()
        second_claim = await second_consumer_repository.claim_next()
        no_remaining_claim = await first_consumer_repository.claim_next()
    finally:
        await _dispose_engines(
            submitting_engine,
            first_consumer_engine,
            second_consumer_engine,
        )

    assert first_claim is not None
    assert second_claim is not None
    assert first_claim.internal_id == first_queued.internal_id
    assert second_claim.internal_id == second_queued.internal_id
    assert no_remaining_claim is None


async def test_postgres_cancellation_blocks_competing_success_publication(
    migrated_postgresql_database_url: str,
) -> None:
    """Keep accepted cancellation terminal when success and cancellation publish together."""
    clock = [_INITIAL]
    execution_repository, execution_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    cancellation_repository, cancellation_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    try:
        queued_job = await execution_repository.create_queued(_queued_job("race"))
        claimed_job = await execution_repository.claim_next()
        assert claimed_job is not None

        cancellation = await cancellation_repository.request_cancellation(
            queued_job.public_id
        )
        assert isinstance(cancellation, ProcessingCancellation)
        assert cancellation.internal_id == claimed_job.internal_id

        success_published, cancellation_published = await asyncio.gather(
            execution_repository.publish_success(
                claimed_job.internal_id,
                _projected_result(),
            ),
            cancellation_repository.publish_cancelled(claimed_job.internal_id),
        )
        terminal_job = await execution_repository.get_job(queued_job.public_id)
    finally:
        await _dispose_engines(execution_engine, cancellation_engine)

    assert success_published is False
    assert cancellation_published is True
    assert isinstance(terminal_job, CancelledTranscriptionJob)
    assert not isinstance(terminal_job, SucceededTranscriptionJob)


async def test_postgres_repeatable_read_snapshot_survives_concurrent_retention(
    migrated_postgresql_database_url: str,
) -> None:
    """Read one complete success when retention deletes it after the read snapshot begins."""
    reader_clock = [_INITIAL]
    retention_clock = [_INITIAL + _TERMINAL_RETENTION]
    reader_repository, reader_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=reader_clock,
    )
    retention_repository, retention_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=retention_clock,
    )
    pause_listener: Callable[..., object] | None = None
    try:
        queued_job = await reader_repository.create_queued(_queued_job("snapshot"))
        claimed_job = await reader_repository.claim_next()
        assert claimed_job is not None
        assert await reader_repository.publish_success(
            claimed_job.internal_id,
            _projected_result(),
        )

        result_read_started = asyncio.Event()
        pause_listener = _install_result_read_pause(
            reader_engine,
            result_read_started,
        )
        snapshot_task = asyncio.create_task(
            reader_repository.get_job(queued_job.public_id)
        )
        await _wait_for_result_read(result_read_started)
        await retention_repository.delete_expired_terminal_jobs()
        snapshot = await snapshot_task
        retained_job = await retention_repository.get_job(queued_job.public_id)
    finally:
        if pause_listener is not None:
            event.remove(
                reader_engine.sync_engine,
                "before_cursor_execute",
                pause_listener,
            )
        await _dispose_engines(reader_engine, retention_engine)

    assert isinstance(snapshot, SucceededTranscriptionJob)
    assert snapshot.public_id == queued_job.public_id
    assert snapshot.result == _projected_result()
    assert retained_job is None


def _install_result_read_pause(
    engine: AsyncEngine,
    result_read_started: asyncio.Event,
) -> Callable[..., object]:
    """Pause result-row loading after the repository's parent read establishes a snapshot."""

    def pause_result_read(
        _connection: object,
        _cursor: object,
        statement: str,
        parameters: object,
        _context: object,
        _executemany: bool,
    ) -> tuple[str, object]:
        if "FROM transcription_job_result" not in statement:
            return statement, parameters
        result_read_started.set()
        return (
            (
                "WITH paused_result_read AS MATERIALIZED (SELECT pg_sleep(0.2)) "
                f"{statement} AND EXISTS (SELECT 1 FROM paused_result_read)"
            ),
            parameters,
        )

    event.listen(
        engine.sync_engine,
        "before_cursor_execute",
        pause_result_read,
        retval=True,
    )
    return pause_result_read
