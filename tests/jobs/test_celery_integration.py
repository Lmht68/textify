"""Real PostgreSQL, Redis, and Celery proof for durable Transcription Jobs."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from celery.contrib.testing.worker import start_worker
from httpx import ASGITransport, AsyncClient

from textify.config import AppConfig, Environment
from textify.jobs.celery_app import TRANSCRIPTION_QUEUE, create_celery_app
from textify.jobs.claim_manager import ExecutionClaimManager
from textify.jobs.config import JobConfig, JobDispatchConfig
from textify.jobs.database import create_application_engine
from textify.jobs.dispatch import CeleryJobDispatchPublisher
from textify.jobs.gpu_ownership import GpuModelOwnership
from textify.jobs.processor import TranscriptionJobProcessor
from textify.jobs.reconciler import TranscriptionJobReconciler
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.service import utc_now
from textify.jobs.worker import TranscriptionWorkerRuntime, register_transcription_task
from textify.main import create_app
from textify.transcription import inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.service import TranscriptionAdapters, TranscriptionExecutor
from textify.transcription.types import TimedTranscript, TranscriptMethod

_YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class _CaptionTrack:
    """Return one deterministic original-language caption transcript."""

    @property
    def language_code(self) -> str:
        """Return the declared source language."""
        return "en"

    @property
    def is_generated(self) -> bool:
        """Return that this integration caption is original-language."""
        return False

    def fetch_segments(self) -> tuple[tuple[float, float, str], ...]:
        """Return one valid caption segment."""
        return ((0.0, 1.0, "One"),)


class _CaptionProvider:
    """Serve the deterministic caption track for the submitted video."""

    def list_tracks(self, video_id: str) -> tuple[_CaptionTrack, ...]:
        """Return one track only for the expected public YouTube video.

        Args:
            video_id: Validated source video identity.

        Returns:
            The known original-language caption track.
        """
        assert video_id == "dQw4w9WgXcQ"
        return (_CaptionTrack(),)


class _MetadataExtractor:
    """Return valid provider-shaped metadata without network access."""

    def extract(
        self,
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: object,
    ) -> inspection.ExtractedMetadata:
        """Return valid YouTube metadata used by the production executor.

        Args:
            provider_url: Validated public source URL.
            deadline: Provider operation deadline, unused by this deterministic source.
            cancellation_event: Cooperative provider cancellation signal, unused here.

        Returns:
            Provider-shaped metadata accepted by the normalizing boundary.
        """
        del deadline, cancellation_event
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
    """Fail the integration proof if caption processing tries native execution."""

    def download(self, *args: object, **kwargs: object) -> Path:
        """Reject an unexpected native audio download path."""
        del args, kwargs
        raise AssertionError("Caption integration path must not download audio.")

    def transcribe(self, *args: object, **kwargs: object) -> TimedTranscript:
        """Reject an unexpected native transcription path."""
        del args, kwargs
        raise AssertionError(
            "Caption integration path must not load native transcription."
        )


def _app_config() -> AppConfig:
    """Build isolated FastAPI configuration.

    Returns:
        Application configuration independent of environment files.
    """
    return AppConfig(
        environment=Environment.LOCAL,
        log_level="INFO",
        host="127.0.0.1",
        port=8182,
    )


def _job_config(database_url: str) -> JobConfig:
    """Build shared PostgreSQL lifecycle configuration.

    Args:
        database_url: Disposable migrated PostgreSQL database URL.

    Returns:
        Durable job storage configuration.
    """
    return JobConfig(
        database_url=database_url,  # type: ignore[arg-type]
        max_outstanding_jobs=8,
        job_queue_timeout_seconds=20,
        job_retention_seconds=86_400,
    )


def _transcription_config(temporary_media_root: Path) -> TranscriptionConfig:
    """Build minimal worker provider configuration for the caption path.

    Args:
        temporary_media_root: Per-test worker media root.

    Returns:
        Normal production executor configuration without native model construction.
    """
    return TranscriptionConfig(
        gpu_identity="test-gpu-celery-integration",
        gpu_lock_directory=temporary_media_root / "gpu-locks",
        temporary_media_root=temporary_media_root,
        transcription_concurrency=1,
        max_media_bytes=1024,
    )


async def _wait_for_succeeded_status(
    client: AsyncClient,
    location: str,
) -> dict[str, object]:
    """Poll status until the real worker publishes a terminal response.

    Args:
        client: ASGI client connected to the PostgreSQL-only API.
        location: Bearer capability returned by durable submission.

    Returns:
        Final succeeded public job response.

    Raises:
        AssertionError: If real Redis-Celery delivery does not finish promptly.
    """
    for _ in range(80):
        response = await client.get(location)
        payload = response.json()
        assert isinstance(payload, dict)
        if payload["status"] == "finished":
            assert payload["outcome"] == "succeeded"
            return payload
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for real Redis-Celery job completion.")


@pytest.mark.asyncio
async def test_api_submission_reconciles_and_retrieves_celery_caption_result(
    migrated_postgresql_database_url: str,
    redis_broker_url: str,
    tmp_path: Path,
) -> None:
    """Prove PostgreSQL admission, Redis delivery, Celery processing, and API retrieval."""
    job_config = _job_config(migrated_postgresql_database_url)
    worker_engine = create_application_engine(migrated_postgresql_database_url)
    worker_repository = PostgresTranscriptionJobRepository(
        engine=worker_engine,
        maximum_outstanding_jobs=job_config.max_outstanding_jobs,
        queue_timeout=timedelta(seconds=job_config.job_queue_timeout_seconds),
        terminal_retention=timedelta(seconds=job_config.job_retention_seconds),
        clock=utc_now,
    )
    reconciler_engine = create_application_engine(migrated_postgresql_database_url)
    reconciler_repository = PostgresTranscriptionJobRepository(
        engine=reconciler_engine,
        maximum_outstanding_jobs=job_config.max_outstanding_jobs,
        queue_timeout=timedelta(seconds=job_config.job_queue_timeout_seconds),
        terminal_retention=timedelta(seconds=job_config.job_retention_seconds),
        clock=utc_now,
    )
    transcription_config = _transcription_config(tmp_path)
    adapters = TranscriptionAdapters(
        metadata_extractor=_MetadataExtractor(),
        caption_provider=_CaptionProvider(),
        audio_downloader=_UnexpectedNativePath(),
        whisper_transcriber=_UnexpectedNativePath(),
    )
    gpu_ownership = GpuModelOwnership(
        worker_engine,
        transcription_config.gpu_identity,
        transcription_config.gpu_lock_directory,
    )
    claim_manager = ExecutionClaimManager(
        worker_repository,
        max_active_claims=1,
        heartbeat_interval=timedelta(seconds=5),
        claim_lease_duration=timedelta(seconds=30),
        claim_ready=gpu_ownership.can_claim,
    )

    def build_resources() -> tuple[TranscriptionJobProcessor, TranscriptionExecutor]:
        """Construct deterministic worker resources after GPU ownership is acquired."""
        executor = TranscriptionExecutor(adapters, transcription_config)
        return (
            TranscriptionJobProcessor(worker_repository, executor, claim_manager),
            executor,
        )

    runtime = TranscriptionWorkerRuntime(
        build_resources,
        claim_manager,
        gpu_ownership,
        worker_engine,
    )
    dispatch_config = JobDispatchConfig(
        broker_url=redis_broker_url,  # type: ignore[arg-type]
        worker_concurrency=1,
        _env_file=None,  # type: ignore[call-arg]
    )
    celery_app = create_celery_app(dispatch_config)
    register_transcription_task(celery_app, runtime)
    runtime.start()
    try:
        with start_worker(
            celery_app,
            pool="threads",
            concurrency=1,
            perform_ping_check=False,
            queues=TRANSCRIPTION_QUEUE,
        ):
            application = create_app(_app_config(), job_config=job_config)
            reconciler = TranscriptionJobReconciler(
                reconciler_repository,
                CeleryJobDispatchPublisher(celery_app),
            )
            async with application.router.lifespan_context(application):
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://test",
                ) as client:
                    submission = await client.post(
                        "/api/transcription-jobs",
                        json={"url": _YOUTUBE_URL},
                    )
                    assert submission.status_code == 202
                    location = submission.headers["Location"]
                    await reconciler.reconcile_once()
                    completed = await _wait_for_succeeded_status(client, location)
    finally:
        runtime.shutdown()
        await reconciler_engine.dispose()

    completed_id = completed["id"]
    assert isinstance(completed_id, str)
    assert UUID(completed_id).version == 4
    assert completed["result"] == {
        "source": {
            "platform": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "url": _YOUTUBE_URL,
            "title": "Title",
            "description": "Description",
            "channel": "Creator",
            "duration_seconds": 12,
        },
        "transcript": {
            "method": TranscriptMethod.YOUTUBE_CAPTIONS,
            "language": "en",
            "text": "One",
            "segments": [{"start": 0.0, "end": 1.0, "text": "One"}],
        },
    }
