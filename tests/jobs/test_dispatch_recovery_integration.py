"""Real PostgreSQL and restartable-Redis proof for durable Job Dispatch recovery."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import AsyncGenerator, Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID, uuid4

import pytest
from celery import Celery
from celery.contrib.testing.worker import start_worker
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from textify.config import AppConfig, Environment
from textify.jobs.celery_app import TRANSCRIPTION_QUEUE, create_celery_app
from textify.jobs.claim_manager import ExecutionClaimManager
from textify.jobs.config import JobConfig, JobDispatchConfig
from textify.jobs.contracts import TranscriptionJobStoreUnavailableError
from textify.jobs.dispatch import CeleryJobDispatchPublisher, JobDispatchPublisher
from textify.jobs.gpu_ownership import GpuModelOwnership
from textify.jobs.processor import TranscriptionJobProcessor
from textify.jobs.reconciler import ReconcilerPolicy, TranscriptionJobReconciler
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.types import (
    JobDispatch,
    NewQueuedTranscriptionJob,
    TranscriptionJob,
)
from textify.jobs.worker import TranscriptionWorkerRuntime, register_transcription_task
from textify.main import create_app
from textify.transcription import inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import NoUsableTranscriptError
from textify.transcription.service import TranscriptionAdapters, TranscriptionExecutor
from textify.transcription.types import TimedTranscript

_YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_QUEUE_TIMEOUT = timedelta(seconds=20)
_TERMINAL_RETENTION = timedelta(days=1)


class _CaptionTrack:
    """Provide one deterministic original-language caption track."""

    @property
    def language_code(self) -> str:
        """Return the declared caption language."""
        return "en"

    @property
    def is_generated(self) -> bool:
        """Report this deterministic track as original-language."""
        return False

    def fetch_segments(self) -> tuple[tuple[float, float, str], ...]:
        """Return one valid caption segment."""
        return ((0.0, 1.0, "One"),)


class _CaptionProvider:
    """Count starts and optionally block or fail real caption processing."""

    def __init__(self) -> None:
        """Initialize a normal deterministic provider."""
        self.mode = "normal"
        self.starts: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def list_tracks(self, video_id: str) -> tuple[_CaptionTrack, ...]:
        """Record one execution and enact the selected test provider behavior."""
        self.starts.append(video_id)
        self.started.set()
        if self.mode == "blocked" and not self.release.wait(timeout=5):
            raise AssertionError("Blocked provider was never released.")
        if self.mode == "failing":
            raise NoUsableTranscriptError()
        return (_CaptionTrack(),)


class _MetadataExtractor:
    """Return valid provider-shaped metadata while preserving execution order."""

    def __init__(self) -> None:
        """Initialize an empty source-start record."""
        self.started_urls: list[str] = []

    def extract(
        self,
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: object,
    ) -> inspection.ExtractedMetadata:
        """Return deterministic source metadata accepted by production code."""
        del deadline, cancellation_event
        self.started_urls.append(provider_url)
        return inspection.ExtractedMetadata(
            {
                "id": "dQw4w9WgXcQ",
                "extractor_key": "Youtube",
                "webpage_url": provider_url,
                "title": "Title",
                "description": "Description",
                "channel": "Creator",
                "duration": 12,
                "language": "en",
            }
        )


class _UnexpectedNativePath:
    """Reject native processing because recovery proofs use captions only."""

    def download(self, *args: object, **kwargs: object) -> Path:
        """Reject an unexpected native download."""
        del args, kwargs
        raise AssertionError("Caption recovery tests must not download audio.")

    def transcribe(self, *args: object, **kwargs: object) -> TimedTranscript:
        """Reject an unexpected native transcription."""
        del args, kwargs
        raise AssertionError("Caption recovery tests must not transcribe audio.")


class _SyntheticProcessExit(BaseException):
    """Model an uncaught process exit in a publication crash window."""


class _CrashBeforePublisher:
    """Crash after the durable lease but before broker acceptance."""

    async def publish(self, dispatch: JobDispatch) -> None:
        """Raise without adding a broker message."""
        del dispatch
        raise _SyntheticProcessExit()


class _CrashAfterPublisher:
    """Crash after real broker acceptance and before durable publication recording."""

    def __init__(self, delegate: JobDispatchPublisher) -> None:
        """Store the real broker publisher that accepts the message first."""
        self._delegate = delegate

    async def publish(self, dispatch: JobDispatch) -> None:
        """Publish to Redis then model an uncaught exit."""
        await self._delegate.publish(dispatch)
        raise _SyntheticProcessExit()


class _BarrierPublisher:
    """Use a barrier after each reconciler's first lease for concurrency proof."""

    def __init__(
        self, barrier: asyncio.Barrier, delegate: JobDispatchPublisher
    ) -> None:
        """Initialize a publisher with one barrier and real broker delegate."""
        self._barrier = barrier
        self._delegate = delegate
        self.dispatches: list[JobDispatch] = []

    async def publish(self, dispatch: JobDispatch) -> None:
        """Wait until both independently leased dispatches are observable."""
        self.dispatches.append(dispatch)
        if len(self.dispatches) == 1:
            await self._barrier.wait()
        await self._delegate.publish(dispatch)


class _RedisBroker(Protocol):
    """Represent the fixture-owned Redis child used by restart tests."""

    @property
    def url(self) -> str:
        """Return the private Redis URL."""
        ...

    async def stop(self) -> None:
        """Stop the fixture-owned Redis child."""
        ...

    async def restart(self) -> None:
        """Restart and clear the non-persistent Redis child."""
        ...


def _job_config(database_url: str) -> JobConfig:
    """Build fixed durable lifecycle settings for integration tests."""
    return JobConfig(
        database_url=database_url,  # type: ignore[arg-type]
        max_outstanding_jobs=8,
        job_queue_timeout_seconds=int(_QUEUE_TIMEOUT.total_seconds()),
        job_retention_seconds=int(_TERMINAL_RETENTION.total_seconds()),
    )


def _repository(
    database_url: str,
    clock: list[datetime],
) -> tuple[PostgresTranscriptionJobRepository, AsyncEngine]:
    """Create an independent application engine and mutable-clock repository."""
    from textify.jobs.database import create_application_engine

    engine = create_application_engine(database_url)
    repository = PostgresTranscriptionJobRepository(
        engine=engine,
        maximum_outstanding_jobs=8,
        queue_timeout=_QUEUE_TIMEOUT,
        terminal_retention=_TERMINAL_RETENTION,
        clock=lambda: clock[0],
    )
    return repository, engine


def _worker_repository(
    database_url: str,
    clock: list[datetime],
) -> tuple[PostgresTranscriptionJobRepository, AsyncEngine]:
    """Create a worker-loop-safe repository without pooled cross-loop connections."""
    engine = create_async_engine(
        database_url,
        echo=False,
        hide_parameters=True,
        pool_pre_ping=True,
        poolclass=NullPool,
    )
    repository = PostgresTranscriptionJobRepository(
        engine=engine,
        maximum_outstanding_jobs=8,
        queue_timeout=_QUEUE_TIMEOUT,
        terminal_retention=_TERMINAL_RETENTION,
        clock=lambda: clock[0],
    )
    return repository, engine


def _celery_app(broker_url: str) -> Celery:
    """Create a real non-eager Celery app for an isolated Redis server."""
    return create_celery_app(
        JobDispatchConfig(
            broker_url=broker_url,  # type: ignore[arg-type]
            worker_concurrency=1,
            _env_file=None,  # type: ignore[call-arg]
        )
    )


def _policy() -> ReconcilerPolicy:
    """Create the issue's bounded recovery policy."""
    return ReconcilerPolicy(
        interval=timedelta(seconds=1),
        publication_lease_duration=timedelta(seconds=5),
        dispatch_batch_size=1,
        retention_cleanup_interval=timedelta(hours=1),
    )


def _runtime(
    repository: PostgresTranscriptionJobRepository,
    engine: AsyncEngine,
    temporary_media_root: Path,
    provider: _CaptionProvider,
    metadata_extractor: _MetadataExtractor,
) -> TranscriptionWorkerRuntime:
    """Build the production worker with deterministic caption dependencies."""
    transcription_config = TranscriptionConfig(
        gpu_identity="test-gpu-dispatch-recovery",
        gpu_lock_directory=temporary_media_root / "gpu-locks",
        temporary_media_root=temporary_media_root,
        transcription_concurrency=1,
        max_media_bytes=1024,
    )
    gpu_ownership = GpuModelOwnership(
        engine,
        transcription_config.gpu_identity,
        transcription_config.gpu_lock_directory,
    )
    claim_manager = ExecutionClaimManager(
        repository,
        max_active_claims=1,
        heartbeat_interval=timedelta(seconds=5),
        claim_lease_duration=timedelta(seconds=30),
        claim_ready=gpu_ownership.can_claim,
    )

    def build_resources() -> tuple[TranscriptionJobProcessor, TranscriptionExecutor]:
        """Build deterministic captions after physical-GPU ownership is acquired."""
        executor = TranscriptionExecutor(
            TranscriptionAdapters(
                metadata_extractor=metadata_extractor,
                caption_provider=provider,
                audio_downloader=_UnexpectedNativePath(),
                whisper_transcriber=_UnexpectedNativePath(),
            ),
            transcription_config,
        )
        return (
            TranscriptionJobProcessor(repository, executor, claim_manager),
            executor,
        )

    return TranscriptionWorkerRuntime(
        build_resources,
        claim_manager,
        gpu_ownership,
        engine,
    )


async def _wait_for_finished(
    repository: PostgresTranscriptionJobRepository,
    public_id: UUID,
) -> TranscriptionJob:
    """Wait until one durable public job reaches any terminal outcome."""
    final_snapshot: TranscriptionJob | None = None
    for _ in range(100):
        final_snapshot = await repository.get_job(public_id)
        if final_snapshot is not None and final_snapshot.status == "finished":
            return final_snapshot
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"Timed out waiting for a Redis-Celery terminal outcome: {final_snapshot!r}"
    )


async def _wait_for_empty_queue(redis_client: Redis) -> None:
    """Wait until the real worker has consumed each Redis queue message."""
    for _ in range(100):
        if await cast(Awaitable[int], redis_client.llen(TRANSCRIPTION_QUEUE)) == 0:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for Celery to consume Redis messages.")


async def _lease_one(repository: PostgresTranscriptionJobRepository) -> JobDispatch:
    """Lease exactly one current due dispatch for direct Celery publication."""
    dispatches = await repository.lease_due_dispatches(
        uuid4(),
        timedelta(seconds=5),
        limit=1,
    )
    assert len(dispatches) == 1
    return dispatches[0]


@asynccontextmanager
async def _public_client(database_url: str) -> AsyncGenerator[AsyncClient]:
    """Provide the real public HTTP surface over the same PostgreSQL database."""
    application = create_app(
        AppConfig(
            environment=Environment.LOCAL,
            log_level="INFO",
            host="127.0.0.1",
            port=8182,
        ),
        job_config=_job_config(database_url),
    )
    async with (
        application.router.lifespan_context(application),
        AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as client,
    ):
        yield client


@asynccontextmanager
async def _real_worker(
    broker_url: str,
    repository: PostgresTranscriptionJobRepository,
    engine: AsyncEngine,
    temporary_media_root: Path,
    provider: _CaptionProvider,
    metadata_extractor: _MetadataExtractor,
) -> AsyncGenerator[None]:
    """Run one worker runtime with one Celery consumer for a bounded assertion."""
    worker_celery_app = _celery_app(broker_url)
    runtime = _runtime(
        repository,
        engine,
        temporary_media_root,
        provider,
        metadata_extractor,
    )
    runtime.start()
    try:
        register_transcription_task(worker_celery_app, runtime)
        with start_worker(
            worker_celery_app,
            pool="threads",
            concurrency=1,
            perform_ping_check=False,
            queues=TRANSCRIPTION_QUEUE,
        ):
            yield
    finally:
        runtime.shutdown()


async def _task_payloads(redis_client: Redis) -> list[dict[str, object]]:
    """Decode real queued Celery payloads without retaining message metadata."""
    payloads: list[dict[str, object]] = []
    while message := await cast(
        Awaitable[str | bytes | None],
        redis_client.lpop(TRANSCRIPTION_QUEUE),
    ):
        envelope = json.loads(message)
        encoded_body = envelope["body"]
        assert isinstance(encoded_body, str)
        decoded_body = json.loads(base64.b64decode(encoded_body))
        payload = decoded_body[0][0]
        assert isinstance(payload, dict)
        payloads.append(payload)
    return payloads


@pytest.mark.asyncio
async def test_broker_outage_expires_public_job_without_false_execution(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RedisBroker,
) -> None:
    """Keep API admission durable and let the absolute deadline end a broker outage."""
    clock = [datetime.now(UTC)]
    repository, engine = _repository(migrated_postgresql_database_url, clock)
    reconciler = TranscriptionJobReconciler(
        repository,
        CeleryJobDispatchPublisher(_celery_app(restartable_redis_broker.url)),
        policy=_policy(),
    )
    try:
        await restartable_redis_broker.stop()
        async with _public_client(migrated_postgresql_database_url) as client:
            accepted = await client.post(
                "/api/transcription-jobs",
                json={"url": f"{_YOUTUBE_URL}&job=broker-outage"},
            )
            assert accepted.status_code == 202
            location = accepted.headers["Location"]
            await reconciler.reconcile_once()
            assert (await client.get(location)).json()["status"] == "queued"
            clock[0] = datetime.now(UTC) + _QUEUE_TIMEOUT
            await reconciler.reconcile_once()
            terminal = await client.get(location)
    finally:
        await engine.dispose()

    assert terminal.json()["status"] == "finished"
    assert terminal.json()["outcome"] == "failed"
    assert terminal.json()["error"]["code"] == "queue_timeout"


@pytest.mark.asyncio
async def test_broker_restart_recovers_lost_delivery(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RedisBroker,
    tmp_path: Path,
) -> None:
    """Republish a committed queued job after Redis loses its first delivery."""
    redis_client = Redis.from_url(restartable_redis_broker.url)
    clock = [datetime.now(UTC)]
    first_repository, first_engine = _repository(
        migrated_postgresql_database_url,
        clock,
    )
    second_repository, second_engine = _repository(
        migrated_postgresql_database_url,
        clock,
    )
    worker_repository, worker_engine = _worker_repository(
        migrated_postgresql_database_url,
        clock,
    )
    publisher = CeleryJobDispatchPublisher(_celery_app(restartable_redis_broker.url))
    provider = _CaptionProvider()
    try:
        lost_job = await first_repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=lost", ())
        )
        await TranscriptionJobReconciler(
            first_repository,
            publisher,
            policy=_policy(),
        ).reconcile_once()
        assert await redis_client.llen(TRANSCRIPTION_QUEUE) == 1
        await restartable_redis_broker.restart()
        assert await redis_client.llen(TRANSCRIPTION_QUEUE) == 0

        clock[0] += timedelta(seconds=1)
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            _MetadataExtractor(),
        ):
            await TranscriptionJobReconciler(
                second_repository,
                publisher,
                policy=_policy(),
            ).reconcile_once()
            terminal = await _wait_for_finished(
                second_repository,
                lost_job.public_id,
            )
    finally:
        await redis_client.aclose()
        await worker_engine.dispose()
        await second_engine.dispose()
        await first_engine.dispose()

    assert terminal.status == "finished"
    assert len(provider.starts) == 1


@pytest.mark.asyncio
async def test_pre_publication_crash_recovers_expired_lease(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RedisBroker,
    tmp_path: Path,
) -> None:
    """Republish only after an interrupted publisher's lease expires."""
    redis_client = Redis.from_url(restartable_redis_broker.url)
    clock = [datetime.now(UTC)]
    first_repository, first_engine = _repository(
        migrated_postgresql_database_url,
        clock,
    )
    second_repository, second_engine = _repository(
        migrated_postgresql_database_url,
        clock,
    )
    worker_repository, worker_engine = _worker_repository(
        migrated_postgresql_database_url,
        clock,
    )
    publisher = CeleryJobDispatchPublisher(_celery_app(restartable_redis_broker.url))
    provider = _CaptionProvider()
    try:
        pre_crash_job = await first_repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=pre-crash", ())
        )
        with pytest.raises(_SyntheticProcessExit):
            await TranscriptionJobReconciler(
                first_repository,
                _CrashBeforePublisher(),
                policy=_policy(),
            ).reconcile_once()
        assert await redis_client.llen(TRANSCRIPTION_QUEUE) == 0

        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            _MetadataExtractor(),
        ):
            await TranscriptionJobReconciler(
                second_repository,
                publisher,
                policy=_policy(),
            ).reconcile_once()
            assert await redis_client.llen(TRANSCRIPTION_QUEUE) == 0
            clock[0] += timedelta(seconds=5)
            await TranscriptionJobReconciler(
                second_repository,
                publisher,
                policy=_policy(),
            ).reconcile_once()
            terminal = await _wait_for_finished(
                second_repository,
                pre_crash_job.public_id,
            )
    finally:
        await redis_client.aclose()
        await worker_engine.dispose()
        await second_engine.dispose()
        await first_engine.dispose()

    assert terminal.status == "finished"
    assert len(provider.starts) == 1


@pytest.mark.asyncio
async def test_post_publication_crash_recovers_lost_delivery(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RedisBroker,
    tmp_path: Path,
) -> None:
    """Recover a publish-then-crash delivery once its durable lease expires."""
    redis_client = Redis.from_url(restartable_redis_broker.url)
    clock = [datetime.now(UTC)]
    first_repository, first_engine = _repository(
        migrated_postgresql_database_url,
        clock,
    )
    second_repository, second_engine = _repository(
        migrated_postgresql_database_url,
        clock,
    )
    worker_repository, worker_engine = _worker_repository(
        migrated_postgresql_database_url,
        clock,
    )
    publisher = CeleryJobDispatchPublisher(_celery_app(restartable_redis_broker.url))
    provider = _CaptionProvider()
    try:
        post_crash_job = await first_repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=post-crash", ())
        )
        with pytest.raises(_SyntheticProcessExit):
            await TranscriptionJobReconciler(
                first_repository,
                _CrashAfterPublisher(publisher),
                policy=_policy(),
            ).reconcile_once()
        assert await redis_client.llen(TRANSCRIPTION_QUEUE) == 1
        await restartable_redis_broker.restart()
        assert await redis_client.llen(TRANSCRIPTION_QUEUE) == 0

        clock[0] += timedelta(seconds=5)
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            _MetadataExtractor(),
        ):
            await TranscriptionJobReconciler(
                second_repository,
                publisher,
                policy=_policy(),
            ).reconcile_once()
            terminal = await _wait_for_finished(
                second_repository,
                post_crash_job.public_id,
            )
    finally:
        await redis_client.aclose()
        await worker_engine.dispose()
        await second_engine.dispose()
        await first_engine.dispose()

    assert terminal.status == "finished"
    assert len(provider.starts) == 1


@pytest.mark.asyncio
async def test_concurrent_reconcilers_lease_disjoint_real_broker_batches(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RedisBroker,
) -> None:
    """Prove separate engines lease disjoint FIFO batches and publish every due job."""
    clock = [datetime.now(UTC)]
    submitting_repository, submitting_engine = _repository(
        migrated_postgresql_database_url, clock
    )
    first_repository, first_engine = _repository(
        migrated_postgresql_database_url, clock
    )
    second_repository, second_engine = _repository(
        migrated_postgresql_database_url, clock
    )
    publisher = CeleryJobDispatchPublisher(_celery_app(restartable_redis_broker.url))
    barrier = asyncio.Barrier(2)
    first_publisher = _BarrierPublisher(barrier, publisher)
    second_publisher = _BarrierPublisher(barrier, publisher)
    try:
        queued_jobs = tuple(
            [
                await submitting_repository.create_queued(
                    NewQueuedTranscriptionJob(
                        f"{_YOUTUBE_URL}&job=concurrent-{index}",
                        (),
                    )
                )
                for index in range(2)
            ]
        )
        await asyncio.gather(
            TranscriptionJobReconciler(
                first_repository,
                first_publisher,
                policy=_policy(),
            ).reconcile_once(),
            TranscriptionJobReconciler(
                second_repository,
                second_publisher,
                policy=_policy(),
            ).reconcile_once(),
        )
        stranded = await submitting_repository.lease_due_dispatches(
            uuid4(),
            timedelta(seconds=5),
            limit=1,
        )
    finally:
        await second_engine.dispose()
        await first_engine.dispose()
        await submitting_engine.dispose()

    published = set(first_publisher.dispatches) | set(second_publisher.dispatches)
    assert set(first_publisher.dispatches).isdisjoint(second_publisher.dispatches)
    assert {dispatch.internal_job_id for dispatch in published} == {
        job.internal_id for job in queued_jobs
    }
    assert stranded == ()


@pytest.mark.asyncio
async def test_real_redis_payload_and_sql_interruptions_preserve_recovery(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RedisBroker,
) -> None:
    """Keep broker payload private and recover one lease and record SQL interruption."""
    clock = [datetime.now(UTC)]
    repository, engine = _repository(migrated_postgresql_database_url, clock)
    redis_client = Redis.from_url(restartable_redis_broker.url)
    publisher = CeleryJobDispatchPublisher(_celery_app(restartable_redis_broker.url))
    lease_failure = True
    record_failure = True

    def fail_once(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        """Interrupt exactly one lease query and one publication record update."""
        nonlocal lease_failure, record_failure
        if (
            lease_failure
            and "FOR UPDATE OF transcription_job_execution_attempt" in statement
        ):
            lease_failure = False
            raise OSError("synthetic lease interruption")
        if record_failure and "last_dispatched_at" in statement:
            record_failure = False
            raise OSError("synthetic publication record interruption")

    try:
        queued_job = await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=payload", ())
        )
        await TranscriptionJobReconciler(
            repository, publisher, policy=_policy()
        ).reconcile_once()
        payloads = await _task_payloads(redis_client)
        assert len(payloads) == 1
        payload_dispatch = JobDispatch.from_payload(payloads[0])
        assert payload_dispatch is not None
        assert payloads[0].keys() == {
            "internal_job_id",
            "execution_attempt_token",
        }
        assert payload_dispatch.internal_job_id == queued_job.internal_id
        assert (
            await repository.claim_dispatch(
                payload_dispatch,
                uuid4(),
                timedelta(seconds=30),
            )
            is not None
        )

        interrupted_job = await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=sql-interruption", ())
        )
        event.listen(engine.sync_engine, "before_cursor_execute", fail_once)
        try:
            with pytest.raises(TranscriptionJobStoreUnavailableError):
                await repository.lease_due_dispatches(
                    uuid4(), timedelta(seconds=5), limit=1
                )
            lease_owner = uuid4()
            dispatch = (
                await repository.lease_due_dispatches(
                    lease_owner,
                    timedelta(seconds=5),
                    limit=1,
                )
            )[0]
            with pytest.raises(TranscriptionJobStoreUnavailableError):
                await repository.record_dispatch_published(
                    dispatch,
                    lease_owner,
                    timedelta(seconds=1),
                )
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", fail_once)
        clock[0] += timedelta(seconds=5)
        recovered = await repository.lease_due_dispatches(
            uuid4(),
            timedelta(seconds=5),
            limit=1,
        )
    finally:
        await redis_client.aclose()
        await engine.dispose()

    assert interrupted_job.internal_id == dispatch.internal_job_id
    assert recovered == (dispatch,)


@pytest.mark.asyncio
async def test_real_worker_rejects_reverse_fifo_delivery_before_private_execution(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RedisBroker,
    tmp_path: Path,
) -> None:
    """Keep two public jobs queued when Redis delivers the newer dispatch first."""
    clock = [datetime.now(UTC)]
    repository, repository_engine = _repository(migrated_postgresql_database_url, clock)
    worker_repository, worker_engine = _worker_repository(
        migrated_postgresql_database_url,
        clock,
    )
    publisher = CeleryJobDispatchPublisher(_celery_app(restartable_redis_broker.url))
    redis_client = Redis.from_url(restartable_redis_broker.url)
    provider = _CaptionProvider()
    metadata_extractor = _MetadataExtractor()
    try:
        older_job = await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=fifo-older", ())
        )
        newer_job = await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=fifo-newer", ())
        )
        dispatches = await repository.lease_due_dispatches(
            uuid4(),
            timedelta(seconds=5),
            limit=2,
        )
        assert [dispatch.internal_job_id for dispatch in dispatches] == [
            older_job.internal_id,
            newer_job.internal_id,
        ]
        await publisher.publish(dispatches[1])
        await publisher.publish(dispatches[1])
        async with (
            _public_client(migrated_postgresql_database_url) as client,
            _real_worker(
                restartable_redis_broker.url,
                worker_repository,
                worker_engine,
                tmp_path,
                provider,
                metadata_extractor,
            ),
        ):
            await _wait_for_empty_queue(redis_client)
            assert (
                await client.get(f"/api/transcription-jobs/{older_job.public_id}")
            ).json()["status"] == "queued"
            assert (
                await client.get(f"/api/transcription-jobs/{newer_job.public_id}")
            ).json()["status"] == "queued"
            assert provider.starts == []

        await publisher.publish(dispatches[0])
        await publisher.publish(dispatches[1])
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            metadata_extractor,
        ):
            older_terminal = await _wait_for_finished(
                repository,
                older_job.public_id,
            )
            newer_terminal = await _wait_for_finished(
                repository,
                newer_job.public_id,
            )
    finally:
        await redis_client.aclose()
        await worker_engine.dispose()
        await repository_engine.dispose()

    assert older_terminal.status == "finished"
    assert newer_terminal.status == "finished"
    assert metadata_extractor.started_urls == [
        f"{_YOUTUBE_URL}&job=fifo-older",
        f"{_YOUTUBE_URL}&job=fifo-newer",
    ]


@pytest.mark.asyncio
async def test_duplicate_deliveries_start_attempts_once_across_terminal_outcomes(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RedisBroker,
    tmp_path: Path,
) -> None:
    """Ignore duplicates before claim, while blocked, and after terminal transitions."""
    clock = [datetime.now(UTC)]
    repository, repository_engine = _repository(migrated_postgresql_database_url, clock)
    worker_repository, worker_engine = _worker_repository(
        migrated_postgresql_database_url,
        clock,
    )
    publisher = CeleryJobDispatchPublisher(_celery_app(restartable_redis_broker.url))
    redis_client = Redis.from_url(restartable_redis_broker.url)
    provider = _CaptionProvider()
    metadata_extractor = _MetadataExtractor()
    try:
        before_claim_job = await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=before-claim", ())
        )
        before_claim_dispatch = await _lease_one(repository)
        await publisher.publish(before_claim_dispatch)
        await publisher.publish(before_claim_dispatch)
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            metadata_extractor,
        ):
            succeeded = await _wait_for_finished(
                repository,
                before_claim_job.public_id,
            )
            await _wait_for_empty_queue(redis_client)

        await publisher.publish(before_claim_dispatch)
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            metadata_extractor,
        ):
            await _wait_for_empty_queue(redis_client)

        provider.mode = "blocked"
        provider.started.clear()
        provider.release.clear()
        blocked_job = await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=blocked", ())
        )
        blocked_dispatch = await _lease_one(repository)
        await publisher.publish(blocked_dispatch)
        await publisher.publish(blocked_dispatch)
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            metadata_extractor,
        ):
            assert await asyncio.to_thread(provider.started.wait, 5)
            provider.release.set()
            blocked = await _wait_for_finished(repository, blocked_job.public_id)
            await _wait_for_empty_queue(redis_client)

        provider.mode = "failing"
        provider.started.clear()
        failed_job = await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=failed", ())
        )
        failed_dispatch = await _lease_one(repository)
        await publisher.publish(failed_dispatch)
        await publisher.publish(failed_dispatch)
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            metadata_extractor,
        ):
            failed = await _wait_for_finished(repository, failed_job.public_id)
            await _wait_for_empty_queue(redis_client)

        provider.mode = "normal"
        cancelled_job = await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=cancelled", ())
        )
        cancelled_dispatch = await _lease_one(repository)
        await repository.request_cancellation(cancelled_job.public_id)
        await publisher.publish(cancelled_dispatch)
        await publisher.publish(cancelled_dispatch)
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            metadata_extractor,
        ):
            await _wait_for_empty_queue(redis_client)

        await repository.create_queued(
            NewQueuedTranscriptionJob(f"{_YOUTUBE_URL}&job=timeout", ())
        )
        timeout_dispatch = await _lease_one(repository)
        clock[0] += _QUEUE_TIMEOUT
        assert await repository.expire_queued_jobs(limit=1) == 1
        await publisher.publish(timeout_dispatch)
        await publisher.publish(timeout_dispatch)
        async with _real_worker(
            restartable_redis_broker.url,
            worker_repository,
            worker_engine,
            tmp_path,
            provider,
            metadata_extractor,
        ):
            await _wait_for_empty_queue(redis_client)
    finally:
        await redis_client.aclose()
        await worker_engine.dispose()
        await repository_engine.dispose()

    assert succeeded.status == "finished"
    assert blocked.status == "finished"
    assert failed.status == "finished"
    assert getattr(failed, "outcome", None) == "failed"
    assert len(provider.starts) == 3
