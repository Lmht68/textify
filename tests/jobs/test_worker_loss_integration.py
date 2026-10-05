"""Process-level PostgreSQL claim recovery proof through the public API."""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

import pytest
from celery.signals import worker_ready, worker_shutting_down
from httpx import ASGITransport, AsyncClient
from pydantic import PostgresDsn, RedisDsn
from sqlalchemy.ext.asyncio import AsyncEngine

from textify.config import AppConfig, Environment
from textify.jobs.celery_app import TRANSCRIPTION_QUEUE, create_celery_app
from textify.jobs.claim_manager import ExecutionClaimManager
from textify.jobs.config import CacheConfig, JobConfig, JobDispatchConfig
from textify.jobs.contracts import TranscriptionJobStoreUnavailableError
from textify.jobs.database import create_application_engine
from textify.jobs.dispatch import CeleryJobDispatchPublisher
from textify.jobs.gpu_ownership import GpuModelOwnership
from textify.jobs.processor import TranscriptionJobProcessor
from textify.jobs.readiness import (
    PostgresServiceHeartbeatRepository,
    ServiceHeartbeatReporter,
    ServiceRole,
)
from textify.jobs.reconciler import ReconcilerPolicy, TranscriptionJobReconciler
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.service import utc_now
from textify.jobs.types import (
    ClaimedTranscriptionJob,
    ClaimHeartbeat,
    ExecutionClaim,
    JobDispatch,
)
from textify.jobs.worker import (
    TranscriptionWorkerRuntime,
    _connect_worker_shutdown_signal,
    _disconnect_worker_shutdown_signal,
    _ExecutionResources,
    register_transcription_task,
)
from textify.main import create_app
from textify.transcription import inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.schemas import TranscriptionResponse
from textify.transcription.service import TranscriptionAdapters, TranscriptionExecutor
from textify.transcription.types import Segment, TimedTranscript, TranscriptMethod

_YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_HEARTBEAT_SECONDS = 0.1
_LEASE_SECONDS = 0.6

_WorkerMode = Literal[
    "before_claim_block",
    "caption_success",
    "caption_block",
    "metadata_block",
    "heartbeat_outage",
    "native_block",
]


class _RestartableRedisBroker(Protocol):
    """Expose the isolated broker URL required by process-loss scenarios."""

    @property
    def url(self) -> str:
        """Return the isolated Redis broker URL."""
        ...


@dataclass(frozen=True)
class _WorkerControl:
    """Share deterministic worker lifecycle barriers with a spawned child."""

    ready: ProcessEvent
    before_claim_entered: ProcessEvent
    before_claim_release: ProcessEvent
    execution_started: ProcessEvent
    cancellation_observed: ProcessEvent
    heartbeat_enabled: ProcessEvent
    native_release: ProcessEvent
    cancellation_release: ProcessEvent
    warm_shutdown: ProcessEvent


class _CaptionTrack:
    """Provide one deterministic original-language caption transcript."""

    @property
    def language_code(self) -> str:
        """Return the declared original caption language."""
        return "en"

    @property
    def is_generated(self) -> bool:
        """Return that the deterministic track is not machine-generated."""
        return False

    def fetch_segments(self) -> tuple[tuple[float, float, str], ...]:
        """Return a minimal valid caption segment."""
        return ((0.0, 1.0, "One"),)


class _WorkerMetadataExtractor:
    """Provide metadata and observe cooperative Cancellation in a child worker."""

    def __init__(self, mode: _WorkerMode, control: _WorkerControl) -> None:
        """Initialize the child control mode.

        Args:
            mode: Worker behavior selected by the integration scenario.
            control: Shared process lifecycle events.
        """
        self._mode = mode
        self._control = control

    def extract(
        self,
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> inspection.ExtractedMetadata:
        """Return valid metadata, optionally waiting for cooperative Cancellation.

        Args:
            provider_url: Validated source URL.
            deadline: Metadata deadline, unused by this deterministic provider.
            cancellation_event: Worker event set when durable Cancellation is observed.

        Returns:
            Valid provider-shaped metadata for the configured YouTube source.
        """
        del deadline
        if self._mode in {"metadata_block", "heartbeat_outage"}:
            self._control.execution_started.set()
            cancellation_event.wait()
            self._control.cancellation_observed.set()
            if self._mode == "metadata_block":
                self._control.cancellation_release.wait()
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


class _WorkerCaptionProvider:
    """Expose caption execution barriers and cross-process start counts."""

    def __init__(
        self,
        mode: _WorkerMode,
        control: _WorkerControl,
        caption_starts: Synchronized[int],
    ) -> None:
        """Initialize the deterministic caption-provider behavior.

        Args:
            mode: Worker behavior selected by the integration scenario.
            control: Shared process lifecycle events.
            caption_starts: Process-safe counter for provider starts.
        """
        self._mode = mode
        self._control = control
        self._caption_starts = caption_starts

    def list_tracks(self, video_id: str) -> tuple[_CaptionTrack, ...]:
        """Count caption work, block it, or force native fallback by scenario.

        Args:
            video_id: Expected stable YouTube identity.

        Returns:
            A caption track for caption scenarios, otherwise no candidate tracks.
        """
        assert video_id == "dQw4w9WgXcQ"
        if self._mode == "caption_block":
            with self._caption_starts.get_lock():
                self._caption_starts.value += 1
            self._control.execution_started.set()
            self._control.native_release.wait()
            return (_CaptionTrack(),)
        if self._mode == "caption_success":
            with self._caption_starts.get_lock():
                self._caption_starts.value += 1
            return (_CaptionTrack(),)
        return ()


class _WorkerAudioDownloader:
    """Write a small request-owned audio file for native scenarios."""

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Create one valid local audio file.

        Args:
            source_url: Validated provider source URL.
            destination: Worker-created request directory.
            deadline: Download deadline, unused by this deterministic provider.
            cancellation_event: Cooperative event, unused before native work starts.

        Returns:
            Request-owned local audio path.
        """
        del source_url, deadline, cancellation_event
        audio_path = destination / "audio.webm"
        audio_path.write_bytes(b"x")
        return audio_path


class _WorkerNativeTranscriber:
    """Retain native work until a parent process explicitly releases it."""

    def __init__(self, control: _WorkerControl) -> None:
        """Initialize shared start and native-release barriers.

        Args:
            control: Shared process lifecycle events.
        """
        self._control = control

    def transcribe(
        self,
        audio_path: Path,
        *,
        include_segments: bool = True,
    ) -> TimedTranscript:
        """Block native inference until release, then return a valid transcript.

        Args:
            audio_path: Request-owned local media retained during native inference.
            include_segments: Whether callers need timed segments.

        Returns:
            Stable native transcript after the parent releases execution.
        """
        del audio_path, include_segments
        self._control.execution_started.set()
        self._control.native_release.wait()
        return TimedTranscript(
            TranscriptMethod.FASTER_WHISPER,
            "en",
            "One",
            (Segment(0.0, 1.0, "One"),),
        )


class _WorkerRepository:
    """Inject one process-loss fault while delegating PostgreSQL authority."""

    def __init__(
        self,
        delegate: PostgresTranscriptionJobRepository,
        mode: _WorkerMode,
        control: _WorkerControl,
    ) -> None:
        """Initialize a real repository boundary with scenario-local fault injection.

        Args:
            delegate: Real PostgreSQL lifecycle implementation.
            mode: Worker behavior selected by the integration scenario.
            control: Shared process lifecycle events.
        """
        self._delegate = delegate
        self._mode = mode
        self._control = control

    async def claim_dispatch(
        self,
        dispatch: JobDispatch,
        worker_owner: UUID,
        lease_duration: timedelta,
    ) -> ClaimedTranscriptionJob | None:
        """Optionally block before the durable claim, otherwise delegate it.

        Args:
            dispatch: Delivered private Execution Attempt identity.
            worker_owner: Worker-local owner UUID.
            lease_duration: Positive durable claim duration.

        Returns:
            Claimed private execution inputs, or ``None`` for stale delivery.
        """
        if self._mode == "before_claim_block":
            self._control.before_claim_entered.set()
            await asyncio.to_thread(self._control.before_claim_release.wait)
        return await self._delegate.claim_dispatch(
            dispatch, worker_owner, lease_duration
        )

    async def heartbeat_claims(
        self,
        claims: tuple[ExecutionClaim, ...],
        lease_duration: timedelta,
    ) -> tuple[ClaimHeartbeat, ...]:
        """Simulate PostgreSQL interruption only while the scenario requests it.

        Args:
            claims: Active exact Execution Claims to renew.
            lease_duration: Positive durable claim duration.

        Returns:
            Authoritative heartbeat outcomes for live claims.

        Raises:
            TranscriptionJobStoreUnavailableError: While the outage gate is closed.
        """
        if (
            self._mode == "heartbeat_outage"
            and not self._control.heartbeat_enabled.is_set()
        ):
            raise TranscriptionJobStoreUnavailableError()
        return await self._delegate.heartbeat_claims(claims, lease_duration)

    async def publish_cancelled(self, claim: ExecutionClaim) -> bool:
        """Delegate terminal Cancellation publication for an exact claim."""
        return await self._delegate.publish_cancelled(claim)

    async def publish_success(
        self,
        claim: ExecutionClaim,
        projected_result: TranscriptionResponse,
    ) -> bool:
        """Delegate terminal success publication for an exact claim."""
        return await self._delegate.publish_success(claim, projected_result)

    async def publish_failure(
        self,
        claim: ExecutionClaim,
        error_code: str,
        error_message: str,
    ) -> bool:
        """Delegate terminal failure publication for an exact claim."""
        return await self._delegate.publish_failure(claim, error_code, error_message)


def _worker_transcription_config(
    media_root: Path, mode: _WorkerMode
) -> TranscriptionConfig:
    """Build deterministic process-worker provider settings.

    Args:
        media_root: Worker-local temporary-media root.
        mode: Worker behavior selected by the integration scenario.

    Returns:
        Valid configuration with a short timeout for retained native work only.
    """
    return TranscriptionConfig(
        gpu_identity="test-gpu-worker-loss",
        gpu_lock_directory=media_root / "gpu-locks",
        temporary_media_root=media_root,
        transcription_concurrency=1,
        max_media_bytes=1024,
        transcription_timeout_seconds=0.1 if mode == "native_block" else 30.0,
    )


def _run_worker_process(
    database_url: str,
    broker_url: str,
    media_root: str,
    mode: _WorkerMode,
    control: _WorkerControl,
    caption_starts: Synchronized[int],
) -> None:
    """Run one actual Celery threads worker with deterministic provider barriers.

    Args:
        database_url: Disposable migrated PostgreSQL URL.
        broker_url: Isolated Redis broker URL.
        media_root: Process-visible temporary-media root.
        mode: Worker behavior selected by the integration scenario.
        control: Shared process lifecycle events.
        caption_starts: Process-safe counter for caption provider starts.
    """
    dispatch_config = _dispatch_config(broker_url)
    engine = create_application_engine(database_url)
    repository = PostgresTranscriptionJobRepository(
        engine=engine,
        maximum_outstanding_jobs=8,
        queue_timeout=timedelta(seconds=20),
        terminal_retention=timedelta(days=1),
        clock=utc_now,
    )
    worker_repository = _WorkerRepository(repository, mode, control)
    transcription_config = _worker_transcription_config(Path(media_root), mode)
    gpu_ownership = GpuModelOwnership(
        engine,
        transcription_config.gpu_identity,
        transcription_config.gpu_lock_directory,
    )
    claim_manager = ExecutionClaimManager(
        worker_repository,
        max_active_claims=1,
        heartbeat_interval=timedelta(seconds=_HEARTBEAT_SECONDS),
        claim_lease_duration=timedelta(seconds=_LEASE_SECONDS),
        claim_ready=gpu_ownership.can_claim,
    )

    def build_resources() -> _ExecutionResources:
        """Construct process-worker providers after GPU ownership is acquired."""
        adapters = TranscriptionAdapters(
            metadata_extractor=_WorkerMetadataExtractor(mode, control),
            caption_provider=_WorkerCaptionProvider(mode, control, caption_starts),
            audio_downloader=_WorkerAudioDownloader(),
            whisper_transcriber=_WorkerNativeTranscriber(control),
        )
        executor = TranscriptionExecutor(adapters, transcription_config)
        return _ExecutionResources(
            TranscriptionJobProcessor(worker_repository, executor, claim_manager),
            executor,
            ServiceHeartbeatReporter(
                PostgresServiceHeartbeatRepository(engine),
                ServiceRole.GPU_WORKER,
                refresh_interval=timedelta(
                    seconds=dispatch_config.service_heartbeat_ttl_seconds / 3
                ),
                readiness_probe=gpu_ownership.can_claim,
            ),
        )

    runtime = TranscriptionWorkerRuntime(
        build_resources,
        claim_manager,
        gpu_ownership,
        engine,
    )
    celery_app = create_celery_app(dispatch_config)
    register_transcription_task(celery_app, runtime)

    def mark_ready(**_signal_kwargs: object) -> None:
        """Notify the parent after the real Celery consumer is ready."""
        control.ready.set()

    def mark_warm_shutdown(**_signal_kwargs: object) -> None:
        """Notify the parent that Celery accepted its warm shutdown signal."""
        control.warm_shutdown.set()

    worker_ready.connect(mark_ready, weak=False)
    worker_shutting_down.connect(mark_warm_shutdown, weak=False)
    shutdown_receiver = None
    try:
        runtime.start()
        shutdown_receiver = _connect_worker_shutdown_signal(runtime)
        celery_app.worker_main(
            [
                "worker",
                "--pool=threads",
                "--concurrency=1",
                f"--queues={TRANSCRIPTION_QUEUE}",
                "--loglevel=WARNING",
            ]
        )
    finally:
        if shutdown_receiver is not None:
            _disconnect_worker_shutdown_signal(shutdown_receiver)
        worker_ready.disconnect(mark_ready)
        worker_shutting_down.disconnect(mark_warm_shutdown)
        runtime.shutdown()


def _app_config() -> AppConfig:
    """Build application configuration independent of environment files.

    Returns:
        Local FastAPI application configuration for ASGI integration tests.
    """
    return AppConfig(
        environment=Environment.LOCAL,
        log_level="WARNING",
        host="127.0.0.1",
        port=8182,
    )


def _job_config(database_url: str) -> JobConfig:
    """Build durable public-API configuration for a disposable database.

    Args:
        database_url: Disposable migrated PostgreSQL URL.

    Returns:
        Fixed bounded Transcription Job configuration.
    """
    return JobConfig(
        database_url=PostgresDsn(database_url),
        max_outstanding_jobs=8,
        job_queue_timeout_seconds=20,
        job_retention_seconds=86_400,
    )


def _dispatch_config(broker_url: str) -> JobDispatchConfig:
    """Build fast but valid worker heartbeat configuration.

    Args:
        broker_url: Isolated Redis broker URL.

    Returns:
        Dispatch configuration with a three-to-one lease ratio.
    """
    return JobDispatchConfig(
        broker_url=RedisDsn(broker_url),
        worker_concurrency=1,
        reconciler_interval_seconds=_HEARTBEAT_SECONDS,
        attempt_heartbeat_seconds=_HEARTBEAT_SECONDS,
        attempt_lease_seconds=_LEASE_SECONDS,
        _env_file=None,  # type: ignore[call-arg]
    )


def _repository(
    database_url: str,
) -> tuple[PostgresTranscriptionJobRepository, AsyncEngine]:
    """Create a parent-loop PostgreSQL repository and its owned engine.

    Args:
        database_url: Disposable migrated PostgreSQL URL.

    Returns:
        Real repository and engine owned by the parent pytest event loop.
    """
    engine = create_application_engine(database_url)
    return (
        PostgresTranscriptionJobRepository(
            engine=engine,
            maximum_outstanding_jobs=8,
            queue_timeout=timedelta(seconds=20),
            terminal_retention=timedelta(days=1),
            clock=utc_now,
        ),
        engine,
    )


@asynccontextmanager
async def _public_client(database_url: str) -> AsyncGenerator[AsyncClient]:
    """Provide the real public FastAPI surface for one disposable database.

    Args:
        database_url: Disposable migrated PostgreSQL URL.

    Yields:
        ASGI client connected to the production public API composition.
    """
    application = create_app(
        _app_config(),
        job_config=_job_config(database_url),
        dispatch_config=JobDispatchConfig(
            broker_url=RedisDsn("redis://127.0.0.1:1/15"),
            _env_file=None,  # type: ignore[call-arg]
        ),
        cache_config=CacheConfig(_env_file=None),  # type: ignore[call-arg]
    )
    async with (
        application.router.lifespan_context(application),
        AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client,
    ):
        yield client


class _RecordingPublisher:
    """Delegate real broker publication while retaining exact dispatch identities."""

    def __init__(self, delegate: CeleryJobDispatchPublisher) -> None:
        """Initialize real broker delegation and private publication records.

        Args:
            delegate: Production Redis-Celery publication boundary.
        """
        self._delegate = delegate
        self.dispatches: list[JobDispatch] = []

    async def publish(self, dispatch: JobDispatch) -> None:
        """Record and publish one exact private Job Dispatch.

        Args:
            dispatch: Exact attempt identity to deliver through Celery.
        """
        self.dispatches.append(dispatch)
        await self._delegate.publish(dispatch)


def _reconciler(
    repository: PostgresTranscriptionJobRepository,
    broker_url: str,
    publisher: _RecordingPublisher | None = None,
) -> tuple[TranscriptionJobReconciler, _RecordingPublisher]:
    """Build one fast reconciler using real private Celery publication.

    Args:
        repository: Parent-owned PostgreSQL lifecycle repository.
        broker_url: Isolated Redis broker URL.
        publisher: Optional retained publisher used to inspect a prior dispatch.

    Returns:
        Reconciler and its recording real-broker publisher.
    """
    resolved_publisher = publisher or _RecordingPublisher(
        CeleryJobDispatchPublisher(create_celery_app(_dispatch_config(broker_url)))
    )
    return (
        TranscriptionJobReconciler(
            repository,
            resolved_publisher,
            policy=ReconcilerPolicy(
                interval=timedelta(seconds=_HEARTBEAT_SECONDS),
                publication_lease_duration=timedelta(seconds=_HEARTBEAT_SECONDS),
                dispatch_batch_size=8,
                retention_cleanup_interval=timedelta(days=1),
            ),
        ),
        resolved_publisher,
    )


def _new_control(context: SpawnContext) -> _WorkerControl:
    """Create shared parent-child barriers for one process scenario.

    Args:
        context: Spawn context required for isolated worker processes.

    Returns:
        Fresh lifecycle controls with normal heartbeat and native release enabled.
    """
    heartbeat_enabled = context.Event()
    heartbeat_enabled.set()
    native_release = context.Event()
    native_release.set()
    cancellation_release = context.Event()
    cancellation_release.set()
    before_claim_release = context.Event()
    before_claim_release.set()
    return _WorkerControl(
        ready=context.Event(),
        before_claim_entered=context.Event(),
        before_claim_release=before_claim_release,
        execution_started=context.Event(),
        cancellation_observed=context.Event(),
        heartbeat_enabled=heartbeat_enabled,
        native_release=native_release,
        cancellation_release=cancellation_release,
        warm_shutdown=context.Event(),
    )


async def _start_worker(
    context: SpawnContext,
    database_url: str,
    broker_url: str,
    media_root: Path,
    mode: _WorkerMode,
    control: _WorkerControl,
    caption_starts: Synchronized[int],
) -> BaseProcess:
    """Spawn a real Celery worker and wait for its worker-ready signal.

    Args:
        context: Spawn context required for isolated worker processes.
        database_url: Disposable migrated PostgreSQL URL.
        broker_url: Isolated Redis broker URL.
        media_root: Temporary-media root visible to parent and child.
        mode: Worker behavior selected by the integration scenario.
        control: Shared process lifecycle events.
        caption_starts: Process-safe caption start counter.

    Returns:
        Running spawned worker process.

    Raises:
        AssertionError: If Celery does not become ready within the bounded timeout.
    """
    process = context.Process(
        target=_run_worker_process,
        args=(
            database_url,
            broker_url,
            str(media_root),
            mode,
            control,
            caption_starts,
        ),
    )
    process.start()
    if not await asyncio.to_thread(control.ready.wait, 10):
        await _kill_worker(process)
        raise AssertionError("Timed out waiting for spawned Celery worker readiness.")
    return process


async def _kill_worker(process: BaseProcess) -> None:
    """Force process loss and wait until its child process exits.

    Args:
        process: Known test-owned worker process to kill.

    Raises:
        AssertionError: If the known child cannot be reaped promptly.
    """
    if process.is_alive():
        process.kill()
    await asyncio.to_thread(process.join, 10)
    assert not process.is_alive()


async def _stop_worker(process: BaseProcess) -> None:
    """Stop one test-owned idle worker without hiding a failed clean-shutdown check.

    Args:
        process: Known test-owned worker process to stop.
    """
    if process.is_alive():
        process.terminate()
        await asyncio.to_thread(process.join, 10)
    if process.is_alive():
        process.kill()
        await asyncio.to_thread(process.join, 10)


async def _submit_job(client: AsyncClient, marker: str) -> str:
    """Submit one unique public source and return its bearer status location.

    Args:
        client: Real public ASGI client.
        marker: Unique harmless query parameter for distinct durable jobs.

    Returns:
        Relative bearer-capability status location.
    """
    response = await client.post(
        "/api/transcription-jobs",
        json={"url": f"{_YOUTUBE_URL}&scenario={marker}"},
    )
    assert response.status_code == 202
    return response.headers["Location"]


async def _wait_for_finished(client: AsyncClient, location: str) -> dict[str, object]:
    """Poll the public API until one job reaches its terminal outcome.

    Args:
        client: Real public ASGI client.
        location: Relative bearer-capability status location.

    Returns:
        Terminal public job representation.

    Raises:
        AssertionError: If the public surface does not become terminal promptly.
    """
    for _ in range(200):
        response = await client.get(location)
        payload = response.json()
        assert isinstance(payload, dict)
        if payload.get("status") == "finished":
            return payload
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for a terminal public Transcription Job.")


async def _wait_for_queued(client: AsyncClient, location: str) -> dict[str, object]:
    """Read a public job after a bounded wait for a rejected warm-shutdown delivery.

    Args:
        client: Real public ASGI client.
        location: Relative bearer-capability status location.

    Returns:
        Current public queued-job representation.
    """
    await asyncio.sleep(0.3)
    response = await client.get(location)
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def _caption_start_count(caption_starts: Synchronized[int]) -> int:
    """Read the process-safe caption provider start count.

    Args:
        caption_starts: Shared process-safe provider start counter.

    Returns:
        Current number of caption-provider starts.
    """
    with caption_starts.get_lock():
        return caption_starts.value


@pytest.mark.asyncio
async def test_worker_loss_before_claim_redelivers_once(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RestartableRedisBroker,
    tmp_path: Path,
) -> None:
    """Replace loss before PostgreSQL claim with one subsequent execution."""
    broker_url = restartable_redis_broker.url
    context = multiprocessing.get_context("spawn")
    caption_starts = context.Value("i", 0)
    repository, engine = _repository(migrated_postgresql_database_url)
    blocked_control = _new_control(context)
    blocked_control.before_claim_release.clear()
    blocked_worker: BaseProcess | None = None
    healthy_worker: BaseProcess | None = None
    try:
        blocked_worker = await _start_worker(
            context,
            migrated_postgresql_database_url,
            broker_url,
            tmp_path / "before-claim-blocked",
            "before_claim_block",
            blocked_control,
            caption_starts,
        )
        reconciler, _ = _reconciler(repository, broker_url)
        async with _public_client(migrated_postgresql_database_url) as client:
            location = await _submit_job(client, "before-claim")
            await reconciler.reconcile_once()
            assert await asyncio.to_thread(blocked_control.before_claim_entered.wait, 5)
            await _kill_worker(blocked_worker)
            blocked_worker = None

            healthy_worker = await _start_worker(
                context,
                migrated_postgresql_database_url,
                broker_url,
                tmp_path / "before-claim-healthy",
                "caption_success",
                _new_control(context),
                caption_starts,
            )
            await asyncio.sleep(_HEARTBEAT_SECONDS * 2)
            await reconciler.reconcile_once()
            terminal = await _wait_for_finished(client, location)
    finally:
        if blocked_worker is not None:
            await _stop_worker(blocked_worker)
        if healthy_worker is not None:
            await _stop_worker(healthy_worker)
        await engine.dispose()

    assert terminal["outcome"] == "succeeded"
    assert _caption_start_count(caption_starts) == 1


@pytest.mark.asyncio
async def test_worker_loss_after_claim_finishes_worker_interrupted_without_rerun(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RestartableRedisBroker,
    tmp_path: Path,
) -> None:
    """Reject duplicate delivery after a claimed worker is lost."""
    broker_url = restartable_redis_broker.url
    context = multiprocessing.get_context("spawn")
    caption_starts = context.Value("i", 0)
    repository, engine = _repository(migrated_postgresql_database_url)
    control = _new_control(context)
    control.native_release.clear()
    lost_worker: BaseProcess | None = None
    replacement_worker: BaseProcess | None = None
    try:
        lost_worker = await _start_worker(
            context,
            migrated_postgresql_database_url,
            broker_url,
            tmp_path / "after-claim-lost",
            "caption_block",
            control,
            caption_starts,
        )
        reconciler, publisher = _reconciler(repository, broker_url)
        async with _public_client(migrated_postgresql_database_url) as client:
            location = await _submit_job(client, "after-claim")
            await reconciler.reconcile_once()
            assert await asyncio.to_thread(control.execution_started.wait, 5)
            assert _caption_start_count(caption_starts) == 1
            await _kill_worker(lost_worker)
            lost_worker = None

            await asyncio.sleep(_LEASE_SECONDS + _HEARTBEAT_SECONDS)
            await reconciler.reconcile_once()
            terminal = await _wait_for_finished(client, location)

            replacement_worker = await _start_worker(
                context,
                migrated_postgresql_database_url,
                broker_url,
                tmp_path / "after-claim-replacement",
                "caption_success",
                _new_control(context),
                caption_starts,
            )
            assert publisher.dispatches
            await publisher.publish(publisher.dispatches[0])
            await asyncio.sleep(0.3)
    finally:
        if lost_worker is not None:
            await _stop_worker(lost_worker)
        if replacement_worker is not None:
            await _stop_worker(replacement_worker)
        await engine.dispose()

    assert terminal["outcome"] == "failed"
    error = terminal["error"]
    assert isinstance(error, dict)
    assert error["code"] == "worker_interrupted"
    assert _caption_start_count(caption_starts) == 1


@pytest.mark.asyncio
async def test_worker_loss_after_accepted_cancellation_finishes_cancelled(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RestartableRedisBroker,
    tmp_path: Path,
) -> None:
    """Keep accepted Cancellation authoritative after its worker process is lost."""
    broker_url = restartable_redis_broker.url
    context = multiprocessing.get_context("spawn")
    caption_starts = context.Value("i", 0)
    repository, engine = _repository(migrated_postgresql_database_url)
    control = _new_control(context)
    control.cancellation_release.clear()
    worker: BaseProcess | None = None
    try:
        worker = await _start_worker(
            context,
            migrated_postgresql_database_url,
            broker_url,
            tmp_path / "cancelled-loss",
            "metadata_block",
            control,
            caption_starts,
        )
        reconciler, _ = _reconciler(repository, broker_url)
        async with _public_client(migrated_postgresql_database_url) as client:
            location = await _submit_job(client, "cancelled-loss")
            await reconciler.reconcile_once()
            assert await asyncio.to_thread(control.execution_started.wait, 5)
            cancellation = await client.put(f"{location}/cancellation")
            assert cancellation.status_code == 202
            assert await asyncio.to_thread(control.cancellation_observed.wait, 5)
            await _kill_worker(worker)
            worker = None

            await asyncio.sleep(_LEASE_SECONDS + _HEARTBEAT_SECONDS)
            await reconciler.reconcile_once()
            terminal = await _wait_for_finished(client, location)
    finally:
        if worker is not None:
            await _stop_worker(worker)
        await engine.dispose()

    assert terminal["outcome"] == "cancelled"
    assert "result" not in terminal
    assert "error" not in terminal


@pytest.mark.asyncio
async def test_stale_heartbeat_reconnection_cannot_replace_worker_interrupted(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RestartableRedisBroker,
    tmp_path: Path,
) -> None:
    """Reject a released late worker result after PostgreSQL claim expiry."""
    broker_url = restartable_redis_broker.url
    context = multiprocessing.get_context("spawn")
    caption_starts = context.Value("i", 0)
    repository, engine = _repository(migrated_postgresql_database_url)
    control = _new_control(context)
    control.heartbeat_enabled.clear()
    worker: BaseProcess | None = None
    try:
        worker = await _start_worker(
            context,
            migrated_postgresql_database_url,
            broker_url,
            tmp_path / "stale-heartbeat",
            "heartbeat_outage",
            control,
            caption_starts,
        )
        reconciler, _ = _reconciler(repository, broker_url)
        async with _public_client(migrated_postgresql_database_url) as client:
            location = await _submit_job(client, "stale-heartbeat")
            await reconciler.reconcile_once()
            assert await asyncio.to_thread(control.execution_started.wait, 5)
            await asyncio.sleep(_LEASE_SECONDS + _HEARTBEAT_SECONDS)
            control.heartbeat_enabled.set()
            assert await asyncio.to_thread(control.cancellation_observed.wait, 5)
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            await reconciler.reconcile_once()
            terminal = await _wait_for_finished(client, location)
            await asyncio.sleep(_HEARTBEAT_SECONDS * 2)
            late_terminal = await client.get(location)
    finally:
        if worker is not None:
            await _stop_worker(worker)
        await engine.dispose()

    assert terminal["outcome"] == "failed"
    error = terminal["error"]
    assert isinstance(error, dict)
    assert error["code"] == "worker_interrupted"
    assert late_terminal.json() == terminal


@pytest.mark.asyncio
async def test_warm_shutdown_rejects_new_claims_while_native_cleanup_drains(
    migrated_postgresql_database_url: str,
    restartable_redis_broker: _RestartableRedisBroker,
    tmp_path: Path,
) -> None:
    """Keep the worker alive through native cleanup after warm admission closure."""
    broker_url = restartable_redis_broker.url
    context = multiprocessing.get_context("spawn")
    caption_starts = context.Value("i", 0)
    repository, engine = _repository(migrated_postgresql_database_url)
    control = _new_control(context)
    control.native_release.clear()
    worker: BaseProcess | None = None
    try:
        media_root = tmp_path / "warm-native"
        worker = await _start_worker(
            context,
            migrated_postgresql_database_url,
            broker_url,
            media_root,
            "native_block",
            control,
            caption_starts,
        )
        reconciler, _ = _reconciler(repository, broker_url)
        async with _public_client(migrated_postgresql_database_url) as client:
            first_location = await _submit_job(client, "warm-native-first")
            await reconciler.reconcile_once()
            assert await asyncio.to_thread(control.execution_started.wait, 5)
            first_terminal = await _wait_for_finished(client, first_location)

            worker.terminate()
            assert await asyncio.to_thread(control.warm_shutdown.wait, 5)
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            second_location = await _submit_job(client, "warm-native-second")
            await reconciler.reconcile_once()
            second_snapshot = await _wait_for_queued(client, second_location)
            assert worker.is_alive()
            assert media_root.is_dir()
            assert any(media_root.iterdir())

            control.native_release.set()
            await asyncio.to_thread(worker.join, 10)
            assert not worker.is_alive()
            worker = None
    finally:
        if worker is not None:
            await _stop_worker(worker)
        await engine.dispose()

    assert first_terminal["outcome"] == "failed"
    first_error = first_terminal["error"]
    assert isinstance(first_error, dict)
    assert first_error["code"] == "transcription_timeout"
    assert second_snapshot["status"] == "queued"
    assert {entry.name for entry in media_root.iterdir()} == {"gpu-locks"}
