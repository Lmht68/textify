"""Cross-session PostgreSQL proofs for Transcription Job lifecycle storage."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine

from textify.jobs.contracts import (
    TranscriptionJobCapacityError,
    TranscriptionJobStoreUnavailableError,
)
from textify.jobs.database import create_application_engine
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.types import (
    CancelledTranscriptionJob,
    FailedTranscriptionJob,
    JobDispatch,
    NewQueuedTranscriptionJob,
    ProcessingTranscriptionJob,
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


async def _lease_due_dispatches(
    repository: PostgresTranscriptionJobRepository,
) -> tuple[JobDispatch, ...]:
    """Lease a valid short-lived dispatch batch for lifecycle test setup."""
    return await repository.lease_due_dispatches(
        uuid4(),
        timedelta(seconds=5),
        limit=100,
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
        async with first_engine.connect() as connection:
            job_count = await connection.scalar(
                text("SELECT count(*) FROM transcription_job")
            )
            attempt_count = await connection.scalar(
                text("SELECT count(*) FROM transcription_job_execution_attempt")
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
    assert job_count == 1
    assert attempt_count == 1


async def test_postgres_repositories_enforce_fifo_claims_after_reversed_delivery(
    migrated_postgresql_database_url: str,
) -> None:
    """Reject a newer delivery until the oldest eligible attempt claims."""
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
        dispatches = await _lease_due_dispatches(submitting_repository)

        newer_claim = await first_consumer_repository.claim_dispatch(dispatches[1])
        first_claim = await second_consumer_repository.claim_dispatch(dispatches[0])
        second_claim = await first_consumer_repository.claim_dispatch(dispatches[1])
        repeated_claim = await second_consumer_repository.claim_dispatch(dispatches[0])
    finally:
        await _dispose_engines(
            submitting_engine,
            first_consumer_engine,
            second_consumer_engine,
        )

    assert [dispatch.internal_job_id for dispatch in dispatches] == [
        first_queued.internal_id,
        second_queued.internal_id,
    ]
    assert newer_claim is None
    assert first_claim is not None
    assert second_claim is not None
    assert first_claim.dispatch == dispatches[0]
    assert second_claim.dispatch == dispatches[1]
    assert repeated_claim is None


async def test_postgres_fifo_uses_job_id_when_submission_times_match(
    migrated_postgresql_database_url: str,
) -> None:
    """Break equal-submission-time FIFO ties with the internal job identifier."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=2,
        clock=clock,
    )
    try:
        first_queued = await repository.create_queued(_queued_job("first-tie"))
        second_queued = await repository.create_queued(_queued_job("second-tie"))
        dispatches = await _lease_due_dispatches(repository)

        newer_claim = await repository.claim_dispatch(dispatches[1])
        first_claim = await repository.claim_dispatch(dispatches[0])
        second_claim = await repository.claim_dispatch(dispatches[1])
    finally:
        await engine.dispose()

    assert [dispatch.internal_job_id for dispatch in dispatches] == [
        first_queued.internal_id,
        second_queued.internal_id,
    ]
    assert newer_claim is None
    assert first_claim is not None
    assert second_claim is not None


async def test_postgres_admission_rolls_back_job_exclusions_and_attempt(
    migrated_postgresql_database_url: str,
) -> None:
    """Leave no relation persisted when private attempt insertion cannot complete."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )

    def reject_attempt_insert(*args: object) -> None:
        """Fail only the attempt insert after the parent job was staged."""
        statement = args[2]
        if (
            isinstance(statement, str)
            and "INSERT INTO transcription_job_execution_attempt" in statement
        ):
            raise OSError("forced execution-attempt insertion failure")

    event.listen(engine.sync_engine, "before_cursor_execute", reject_attempt_insert)
    try:
        with pytest.raises(TranscriptionJobStoreUnavailableError):
            await repository.create_queued(
                NewQueuedTranscriptionJob(
                    submitted_url=(
                        "https://www.youtube.com/watch?v=AbCdEf12345&job=rollback"
                    ),
                    exclusions=("transcript.text",),
                )
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", reject_attempt_insert)

    try:
        async with engine.connect() as connection:
            relation_counts = tuple(
                [
                    await connection.scalar(text(f"SELECT count(*) FROM {table_name}"))
                    for table_name in (
                        "transcription_job",
                        "transcription_job_exclusion",
                        "transcription_job_execution_attempt",
                    )
                ]
            )
    finally:
        await engine.dispose()

    assert relation_counts == (0, 0, 0)


async def test_postgres_dispatch_leases_are_private(
    migrated_postgresql_database_url: str,
) -> None:
    """Expose only leased internal Job Dispatch values to the reconciler."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=2,
        clock=clock,
    )
    try:
        await repository.create_queued(_queued_job("first"))
        await repository.create_queued(_queued_job("second"))

        dispatches = await _lease_due_dispatches(repository)
    finally:
        await engine.dispose()

    assert len({dispatch.execution_attempt_token for dispatch in dispatches}) == 2
    assert all(
        not hasattr(dispatch, private_attribute)
        for dispatch in dispatches
        for private_attribute in ("submitted_url", "exclusions", "public_id")
    )


async def test_postgres_cancellation_and_timeout_unblock_the_next_fifo_dispatch(
    migrated_postgresql_database_url: str,
) -> None:
    """Allow the next dispatch after the older queued job terminalizes."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=4,
        clock=clock,
    )
    try:
        cancelled_job = await repository.create_queued(_queued_job("cancelled"))
        cancellation_successor = await repository.create_queued(
            _queued_job("cancellation-successor")
        )
        cancellation_dispatches = await _lease_due_dispatches(repository)
        cancelled = await repository.request_cancellation(cancelled_job.public_id)
        cancellation_successor_claim = await repository.claim_dispatch(
            cancellation_dispatches[1]
        )

        timeout_job = await repository.create_queued(_queued_job("timeout"))
        timeout_dispatch = (await _lease_due_dispatches(repository))[0]
        clock[0] += _QUEUE_TIMEOUT + timedelta(seconds=1)
        timeout_successor = await repository.create_queued(
            _queued_job("timeout-successor")
        )
        timeout_successor_dispatch = (await _lease_due_dispatches(repository))[0]
        timeout_claim = await repository.claim_dispatch(timeout_dispatch)
        timeout_successor_claim = await repository.claim_dispatch(
            timeout_successor_dispatch
        )
    finally:
        await engine.dispose()

    assert isinstance(cancelled, CancelledTranscriptionJob)
    assert cancellation_successor_claim is not None
    assert cancellation_dispatches[1].internal_job_id == (
        cancellation_successor.internal_id
    )
    assert timeout_dispatch.internal_job_id == timeout_job.internal_id
    assert timeout_successor_dispatch.internal_job_id == timeout_successor.internal_id
    assert timeout_claim is None
    assert timeout_successor_claim is not None


async def test_postgres_leases_due_dispatches_and_recovers_after_redispatch(
    migrated_postgresql_database_url: str,
) -> None:
    """Lease due attempts, record publication, and release failed publications."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    first_lease_owner = uuid4()
    second_lease_owner = uuid4()
    try:
        queued_job = await repository.create_queued(_queued_job("lease-recovery"))
        first_dispatches = await repository.lease_due_dispatches(
            first_lease_owner,
            timedelta(seconds=5),
            limit=1,
        )
        blocked_dispatches = await repository.lease_due_dispatches(
            second_lease_owner,
            timedelta(seconds=5),
            limit=1,
        )
        recorded = await repository.record_dispatch_published(
            first_dispatches[0],
            first_lease_owner,
            timedelta(seconds=1),
        )
        not_yet_due_dispatches = await repository.lease_due_dispatches(
            second_lease_owner,
            timedelta(seconds=5),
            limit=1,
        )
        clock[0] += timedelta(seconds=1)
        recovered_dispatches = await repository.lease_due_dispatches(
            second_lease_owner,
            timedelta(seconds=5),
            limit=1,
        )
        async with engine.connect() as connection:
            dispatch_history = (
                (
                    await connection.execute(
                        text(
                            "SELECT dispatch_count, last_dispatched_at, "
                            "next_dispatch_at, publication_lease_owner "
                            "FROM transcription_job_execution_attempt "
                            "WHERE job_id = :job_id"
                        ),
                        {"job_id": queued_job.internal_id},
                    )
                )
                .mappings()
                .one()
            )
        await repository.release_dispatch_leases(
            recovered_dispatches,
            second_lease_owner,
        )
        released_dispatches = await repository.lease_due_dispatches(
            uuid4(),
            timedelta(seconds=5),
            limit=1,
        )
    finally:
        await engine.dispose()

    assert first_dispatches[0].internal_job_id == queued_job.internal_id
    assert blocked_dispatches == ()
    assert recorded is True
    assert not_yet_due_dispatches == ()
    assert recovered_dispatches == first_dispatches
    assert released_dispatches == first_dispatches
    assert dispatch_history["dispatch_count"] == 1
    assert dispatch_history["last_dispatched_at"] == _INITIAL
    assert dispatch_history["next_dispatch_at"] == _INITIAL + timedelta(seconds=1)
    assert dispatch_history["publication_lease_owner"] == second_lease_owner


async def test_postgres_lifecycle_transitions_prevent_future_dispatch_leases(
    migrated_postgresql_database_url: str,
) -> None:
    """Exclude processing, cancelled, and queue-timeout jobs from future leasing."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=3,
        clock=clock,
    )
    try:
        processing_job = await repository.create_queued(_queued_job("processing"))
        processing_dispatch = (await _lease_due_dispatches(repository))[0]
        assert processing_dispatch.internal_job_id == processing_job.internal_id
        assert await repository.claim_dispatch(processing_dispatch) is not None
        processing_leases = await _lease_due_dispatches(repository)

        cancelled_job = await repository.create_queued(_queued_job("cancelled"))
        cancelled_dispatch = (await _lease_due_dispatches(repository))[0]
        assert cancelled_dispatch.internal_job_id == cancelled_job.internal_id
        assert isinstance(
            await repository.request_cancellation(cancelled_job.public_id),
            CancelledTranscriptionJob,
        )
        cancelled_leases = await _lease_due_dispatches(repository)

        timeout_job = await repository.create_queued(_queued_job("timeout"))
        timeout_dispatch = (await _lease_due_dispatches(repository))[0]
        assert timeout_dispatch.internal_job_id == timeout_job.internal_id
        clock[0] += _QUEUE_TIMEOUT
        assert await repository.expire_queued_jobs(limit=1) == 1
        timeout_leases = await _lease_due_dispatches(repository)
    finally:
        await engine.dispose()

    assert processing_leases == ()
    assert cancelled_leases == ()
    assert timeout_leases == ()


async def test_postgres_expires_queued_jobs_before_their_attempts_are_claimed(
    migrated_postgresql_database_url: str,
) -> None:
    """Finish due queued jobs without relying on a broker delivery."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=2,
        clock=clock,
    )
    try:
        first_job = await repository.create_queued(_queued_job("first-expiry"))
        second_job = await repository.create_queued(_queued_job("second-expiry"))
        clock[0] += _QUEUE_TIMEOUT
        first_expired_count = await repository.expire_queued_jobs(limit=1)
        first_snapshot = await repository.get_job(first_job.public_id)
        second_snapshot = await repository.get_job(second_job.public_id)
        second_expired_count = await repository.expire_queued_jobs(limit=1)
        terminal_snapshot = await repository.get_job(second_job.public_id)
    finally:
        await engine.dispose()

    assert first_expired_count == 1
    assert isinstance(first_snapshot, FailedTranscriptionJob)
    assert first_snapshot.error_code == "queue_timeout"
    assert isinstance(second_snapshot, QueuedTranscriptionJob)
    assert second_expired_count == 1
    assert isinstance(terminal_snapshot, FailedTranscriptionJob)
    assert terminal_snapshot.error_code == "queue_timeout"


async def test_postgres_claims_before_loading_private_inputs(
    migrated_postgresql_database_url: str,
) -> None:
    """Reject stale deliveries without selecting URL or projection exclusions."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    query_phases: list[str] = []

    def record_claim_query(*args: object) -> None:
        """Record claim transition and private-input read ordering."""
        statement = args[2]
        if not isinstance(statement, str):
            return
        if "submitted_url" in statement:
            query_phases.append("private_input")
        elif statement.startswith("UPDATE transcription_job"):
            query_phases.append("transition")

    try:
        queued_job = await repository.create_queued(_queued_job("claim-order"))
        dispatch = (await _lease_due_dispatches(repository))[0]
        stale_dispatch = JobDispatch(
            internal_job_id=queued_job.internal_id,
            execution_attempt_token=uuid4(),
        )
        event.listen(engine.sync_engine, "before_cursor_execute", record_claim_query)
        try:
            stale_claim = await repository.claim_dispatch(stale_dispatch)
            valid_claim = await repository.claim_dispatch(dispatch)
        finally:
            event.remove(
                engine.sync_engine,
                "before_cursor_execute",
                record_claim_query,
            )
    finally:
        await engine.dispose()

    assert stale_claim is None
    assert valid_claim is not None
    assert query_phases.index("private_input") > query_phases.index("transition")


async def test_postgres_stale_dispatches_do_not_load_private_inputs(
    migrated_postgresql_database_url: str,
) -> None:
    """Reject wrong-token, active, cancelled, and terminal deliveries before input IO."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=4,
        clock=clock,
    )
    private_input_reads: list[str] = []

    def record_private_input_reads(*args: object) -> None:
        """Record only SQL that could expose stored URL or exclusions."""
        statement = args[2]
        if isinstance(statement, str) and "submitted_url" in statement:
            private_input_reads.append(statement)

    try:
        queued_jobs = tuple(
            [
                await repository.create_queued(_queued_job(source_suffix))
                for source_suffix in (
                    "processing",
                    "cancelled",
                    "terminal",
                    "wrong-token",
                )
            ]
        )
        dispatches = await _lease_due_dispatches(repository)
        processing_claim = await repository.claim_dispatch(dispatches[0])
        assert processing_claim is not None
        await repository.request_cancellation(queued_jobs[1].public_id)
        terminal_claim = await repository.claim_dispatch(dispatches[2])
        assert terminal_claim is not None
        assert await repository.publish_failure(
            terminal_claim.dispatch,
            "transcription_failed",
            "The transcription provider failed.",
        )
        invalid_dispatches = (
            JobDispatch(
                internal_job_id=dispatches[3].internal_job_id,
                execution_attempt_token=uuid4(),
            ),
            dispatches[0],
            dispatches[1],
            dispatches[2],
        )
        event.listen(
            engine.sync_engine,
            "before_cursor_execute",
            record_private_input_reads,
        )
        try:
            claims = await asyncio.gather(
                *(
                    repository.claim_dispatch(dispatch)
                    for dispatch in invalid_dispatches
                )
            )
        finally:
            event.remove(
                engine.sync_engine,
                "before_cursor_execute",
                record_private_input_reads,
            )
    finally:
        await engine.dispose()

    assert claims == [None, None, None, None]
    assert not private_input_reads


async def test_postgres_allows_only_one_successful_claim_per_dispatch(
    migrated_postgresql_database_url: str,
) -> None:
    """Allow only one concurrently delivered message to claim one attempt."""
    clock = [_INITIAL]
    submitting_repository, submitting_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    first_consumer, first_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    second_consumer, second_engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=1,
        clock=clock,
    )
    try:
        await submitting_repository.create_queued(_queued_job("concurrent-claim"))
        dispatch = (await _lease_due_dispatches(submitting_repository))[0]
        claims = await asyncio.gather(
            first_consumer.claim_dispatch(dispatch),
            second_consumer.claim_dispatch(dispatch),
        )
    finally:
        await _dispose_engines(submitting_engine, first_engine, second_engine)

    assert sum(claim is not None for claim in claims) == 1


async def test_postgres_terminal_publication_requires_claim_token(
    migrated_postgresql_database_url: str,
) -> None:
    """Guard success, failure, and cancellation publication with the attempt token."""
    clock = [_INITIAL]
    repository, engine = _repository(
        migrated_postgresql_database_url,
        maximum_outstanding_jobs=3,
        clock=clock,
    )
    try:
        success_job = await repository.create_queued(_queued_job("token-success"))
        success_dispatch = (await _lease_due_dispatches(repository))[0]
        success_claim = await repository.claim_dispatch(success_dispatch)
        assert success_claim is not None
        wrong_success_dispatch = JobDispatch(
            internal_job_id=success_dispatch.internal_job_id,
            execution_attempt_token=uuid4(),
        )
        assert not await repository.publish_success(
            wrong_success_dispatch,
            _projected_result(),
        )
        assert await repository.publish_success(
            success_claim.dispatch, _projected_result()
        )

        failure_job = await repository.create_queued(_queued_job("token-failure"))
        failure_dispatch = (await _lease_due_dispatches(repository))[0]
        failure_claim = await repository.claim_dispatch(failure_dispatch)
        assert failure_claim is not None
        wrong_failure_dispatch = JobDispatch(
            internal_job_id=failure_dispatch.internal_job_id,
            execution_attempt_token=uuid4(),
        )
        assert not await repository.publish_failure(
            wrong_failure_dispatch,
            "transcription_failed",
            "The transcription provider failed.",
        )
        assert await repository.publish_failure(
            failure_claim.dispatch,
            "transcription_failed",
            "The transcription provider failed.",
        )

        cancellation_job = await repository.create_queued(
            _queued_job("token-cancelled")
        )
        cancellation_dispatch = (await _lease_due_dispatches(repository))[0]
        cancellation_claim = await repository.claim_dispatch(cancellation_dispatch)
        assert cancellation_claim is not None
        cancellation = await repository.request_cancellation(cancellation_job.public_id)
        assert isinstance(cancellation, ProcessingTranscriptionJob)
        wrong_cancellation_dispatch = JobDispatch(
            internal_job_id=cancellation_dispatch.internal_job_id,
            execution_attempt_token=uuid4(),
        )
        assert not await repository.publish_cancelled(wrong_cancellation_dispatch)
        assert await repository.publish_cancelled(cancellation_claim.dispatch)

        success_terminal = await repository.get_job(success_job.public_id)
        failure_terminal = await repository.get_job(failure_job.public_id)
        cancellation_terminal = await repository.get_job(cancellation_job.public_id)
    finally:
        await engine.dispose()

    assert isinstance(success_terminal, SucceededTranscriptionJob)
    assert isinstance(failure_terminal, FailedTranscriptionJob)
    assert isinstance(cancellation_terminal, CancelledTranscriptionJob)


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
        dispatch = (await _lease_due_dispatches(execution_repository))[0]
        claimed_job = await execution_repository.claim_dispatch(dispatch)
        assert claimed_job is not None

        cancellation = await cancellation_repository.request_cancellation(
            queued_job.public_id
        )
        assert isinstance(cancellation, ProcessingTranscriptionJob)
        assert cancellation.public_id == queued_job.public_id

        success_published, cancellation_published = await asyncio.gather(
            execution_repository.publish_success(
                claimed_job.dispatch,
                _projected_result(),
            ),
            cancellation_repository.publish_cancelled(claimed_job.dispatch),
        )
        terminal_job = await execution_repository.get_job(queued_job.public_id)
    finally:
        await _dispose_engines(execution_engine, cancellation_engine)

    assert success_published is False
    assert cancellation_published is True
    assert isinstance(terminal_job, CancelledTranscriptionJob)
    assert not isinstance(terminal_job, SucceededTranscriptionJob)


async def test_postgres_cancellation_blocks_competing_failure_publication(
    migrated_postgresql_database_url: str,
) -> None:
    """Keep accepted cancellation terminal when a provider failure races publication."""
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
        queued_job = await execution_repository.create_queued(
            _queued_job("failure-race")
        )
        dispatch = (await _lease_due_dispatches(execution_repository))[0]
        claimed_job = await execution_repository.claim_dispatch(dispatch)
        assert claimed_job is not None
        cancellation = await cancellation_repository.request_cancellation(
            queued_job.public_id
        )
        assert isinstance(cancellation, ProcessingTranscriptionJob)

        failure_published, cancellation_published = await asyncio.gather(
            execution_repository.publish_failure(
                claimed_job.dispatch,
                "transcription_failed",
                "The transcription provider failed.",
            ),
            cancellation_repository.publish_cancelled(claimed_job.dispatch),
        )
        terminal_job = await execution_repository.get_job(queued_job.public_id)
    finally:
        await _dispose_engines(execution_engine, cancellation_engine)

    assert failure_published is False
    assert cancellation_published is True
    assert isinstance(terminal_job, CancelledTranscriptionJob)
    assert not isinstance(terminal_job, FailedTranscriptionJob)


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
        dispatch = (await _lease_due_dispatches(reader_repository))[0]
        claimed_job = await reader_repository.claim_dispatch(dispatch)
        assert claimed_job is not None
        assert await reader_repository.publish_success(
            claimed_job.dispatch,
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
        async with retention_engine.connect() as connection:
            retained_attempt_count = await connection.scalar(
                text(
                    "SELECT count(*) FROM transcription_job_execution_attempt "
                    "WHERE job_id = :job_id"
                ),
                {"job_id": queued_job.internal_id},
            )
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
    assert retained_attempt_count == 0


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
