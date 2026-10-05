"""Deterministic distributed proofs for one two-worker Faster-Whisper model."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from celery.contrib.testing.worker import start_worker
from httpx import ASGITransport, AsyncClient

from textify.config import AppConfig, Environment
from textify.jobs.celery_app import TRANSCRIPTION_QUEUE, create_celery_app
from textify.jobs.config import CacheConfig, JobConfig, JobDispatchConfig
from textify.jobs.database import create_application_engine
from textify.jobs.dispatch import CeleryJobDispatchPublisher
from textify.jobs.reconciler import TranscriptionJobReconciler
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.service import utc_now
from textify.jobs.worker import TranscriptionWorkerRuntime, register_transcription_task
from textify.main import create_app
from textify.transcription import inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.service import TranscriptionAdapters
from textify.transcription.types import (
    Segment,
    TimedTranscript,
    Transcript,
    TranscriptMethod,
)

_NATIVE_A = "dQw4w9WgXcQ"
_NATIVE_B = "M7lc1UVf-VE"
_NATIVE_C = "aqz-KE-bpKQ"
_CAPTION_D = "3JZ_D3ELwOQ"
_VIDEO_IDS = frozenset({_NATIVE_A, _NATIVE_B, _NATIVE_C, _CAPTION_D})


def _youtube_url(video_id: str) -> str:
    """Return a valid deterministic YouTube URL for one test marker."""
    return f"https://www.youtube.com/watch?v={video_id}"


def _video_id_from_url(provider_url: str) -> str:
    """Extract one expected marker from a deterministic provider URL."""
    video_ids = parse_qs(urlsplit(provider_url).query).get("v", [])
    assert len(video_ids) == 1
    video_id = video_ids[0]
    assert video_id in _VIDEO_IDS
    return video_id


class _CaptionTrack:
    """Return one deterministic original caption transcript."""

    @property
    def language_code(self) -> str:
        """Return the source language for caption acquisition."""
        return "en"

    @property
    def is_generated(self) -> bool:
        """Report the deterministic track as original-language captions."""
        return False

    def fetch_segments(self) -> tuple[tuple[float, float, str], ...]:
        """Return one valid timed caption segment."""
        return ((0.0, 1.0, "caption"),)


class _CaptionProvider:
    """Provide captions only for the caption-bypass marker."""

    def list_tracks(self, video_id: str) -> tuple[_CaptionTrack, ...]:
        """Return a caption track only for the configured caption job."""
        assert video_id in _VIDEO_IDS
        if video_id == _CAPTION_D:
            return (_CaptionTrack(),)
        return ()


class _MetadataExtractor:
    """Return marker-preserving YouTube metadata without network access."""

    def extract(
        self,
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> inspection.ExtractedMetadata:
        """Return valid source metadata preserving the requested video marker."""
        del deadline, cancellation_event
        video_id = _video_id_from_url(provider_url)
        return inspection.ExtractedMetadata(
            {
                "id": video_id,
                "extractor_key": "Youtube",
                "webpage_url": provider_url,
                "title": f"Title {video_id}",
                "description": "Description",
                "channel": "Creator",
                "duration": 12,
                "language": "en",
            }
        )


class _MarkerAudioDownloader:
    """Write native markers into request media and expose preparation events."""

    def __init__(self) -> None:
        """Initialize download records for every native source marker."""
        self.markers: list[str] = []
        self.request_directories: list[Path] = []
        self.prepared = {video_id: threading.Event() for video_id in _VIDEO_IDS}
        self._lock = threading.Lock()

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Write a source marker into one request-owned audio file."""
        del deadline, cancellation_event
        video_id = _video_id_from_url(source_url)
        assert video_id != _CAPTION_D
        audio_path = destination / "audio.webm"
        audio_path.write_text(video_id)
        with self._lock:
            self.markers.append(video_id)
            self.request_directories.append(destination)
        self.prepared[video_id].set()
        return audio_path


class _BarrierTranscriber:
    """Hold two marked native calls and record model-level overlap precisely."""

    def __init__(self) -> None:
        """Initialize per-marker entry, release, activity, and interval controls."""
        self.entered = {video_id: threading.Event() for video_id in _VIDEO_IDS}
        self.release = {
            _NATIVE_A: threading.Event(),
            _NATIVE_B: threading.Event(),
        }
        self.markers: list[str] = []
        self.intervals: dict[str, tuple[float, float]] = {}
        self.maximum_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def transcribe(
        self,
        audio_path: Path,
        *,
        include_segments: bool = True,
    ) -> Transcript:
        """Block A and B, admit C after a release, and return native output."""
        video_id = audio_path.read_text()
        assert video_id in {_NATIVE_A, _NATIVE_B, _NATIVE_C}
        started_at = time.monotonic()
        with self._lock:
            self._active += 1
            self.maximum_active = max(self.maximum_active, self._active)
            self.markers.append(video_id)
        self.entered[video_id].set()
        try:
            release = self.release.get(video_id)
            if release is not None:
                release.wait()
            if include_segments:
                return TimedTranscript(
                    TranscriptMethod.FASTER_WHISPER,
                    "en",
                    "native",
                    (Segment(0.0, 1.0, "native"),),
                )
            return Transcript(TranscriptMethod.FASTER_WHISPER, "en", "native")
        finally:
            finished_at = time.monotonic()
            with self._lock:
                self._active -= 1
                self.intervals[video_id] = (started_at, finished_at)


@dataclass(slots=True)
class _ConcurrencyHarness:
    """Expose the running public job system and deterministic provider controls."""

    client: AsyncClient
    reconciler: TranscriptionJobReconciler
    downloader: _MarkerAudioDownloader
    transcriber: _BarrierTranscriber
    adapters_factory_calls: list[None]


@asynccontextmanager
async def _running_harness(
    migrated_postgresql_database_url: str,
    redis_broker_url: str,
    tmp_path: Path,
) -> AsyncGenerator[_ConcurrencyHarness]:
    """Run the public API, reconciler, and production worker composition."""
    job_config = JobConfig(
        database_url=migrated_postgresql_database_url,  # type: ignore[arg-type]
        max_outstanding_jobs=8,
        job_queue_timeout_seconds=20,
        job_retention_seconds=86_400,
    )
    dispatch_config = JobDispatchConfig(
        broker_url=redis_broker_url,  # type: ignore[arg-type]
        worker_concurrency=4,
        _env_file=None,  # type: ignore[call-arg]
    )
    transcription_config = TranscriptionConfig(
        gpu_identity="test-gpu-concurrency",
        gpu_lock_directory=tmp_path / "gpu-locks",
        temporary_media_root=tmp_path / "media",
        transcription_concurrency=2,
        max_media_bytes=1024,
        _env_file=None,  # type: ignore[call-arg]
    )
    worker_engine = create_application_engine(migrated_postgresql_database_url)
    reconciler_engine = create_application_engine(migrated_postgresql_database_url)
    worker_repository = PostgresTranscriptionJobRepository(
        engine=worker_engine,
        maximum_outstanding_jobs=job_config.max_outstanding_jobs,
        queue_timeout=timedelta(seconds=job_config.job_queue_timeout_seconds),
        terminal_retention=timedelta(seconds=job_config.job_retention_seconds),
        clock=utc_now,
    )
    reconciler_repository = PostgresTranscriptionJobRepository(
        engine=reconciler_engine,
        maximum_outstanding_jobs=job_config.max_outstanding_jobs,
        queue_timeout=timedelta(seconds=job_config.job_queue_timeout_seconds),
        terminal_retention=timedelta(seconds=job_config.job_retention_seconds),
        clock=utc_now,
    )
    downloader = _MarkerAudioDownloader()
    transcriber = _BarrierTranscriber()
    adapters_factory_calls: list[None] = []

    def build_adapters(_config: TranscriptionConfig) -> TranscriptionAdapters:
        """Construct the shared deterministic transcriber after GPU ownership."""
        adapters_factory_calls.append(None)
        return TranscriptionAdapters(
            metadata_extractor=_MetadataExtractor(),
            caption_provider=_CaptionProvider(),
            audio_downloader=downloader,
            whisper_transcriber=transcriber,
        )

    runtime = TranscriptionWorkerRuntime.from_repository(
        worker_repository,
        worker_engine,
        transcription_config,
        dispatch_config,
        adapters_factory=build_adapters,
    )
    celery_app = create_celery_app(dispatch_config)
    register_transcription_task(celery_app, runtime)
    reconciler = TranscriptionJobReconciler(
        reconciler_repository,
        CeleryJobDispatchPublisher(celery_app),
    )
    application = create_app(
        AppConfig(
            environment=Environment.LOCAL,
            log_level="INFO",
            host="127.0.0.1",
            port=8182,
        ),
        job_config=job_config,
        dispatch_config=dispatch_config,
        cache_config=CacheConfig(_env_file=None),  # type: ignore[call-arg]
    )

    runtime.start()
    try:
        with start_worker(
            celery_app,
            pool="threads",
            concurrency=4,
            perform_ping_check=False,
            queues=TRANSCRIPTION_QUEUE,
        ):
            async with application.router.lifespan_context(application):
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://test",
                ) as client:
                    yield _ConcurrencyHarness(
                        client,
                        reconciler,
                        downloader,
                        transcriber,
                        adapters_factory_calls,
                    )
    finally:
        runtime.shutdown()
        await reconciler_engine.dispose()


async def _submit_and_reconcile(
    harness: _ConcurrencyHarness,
    video_id: str,
) -> str:
    """Submit one public job and publish its durable Job Dispatch."""
    response = await harness.client.post(
        "/api/transcription-jobs",
        json={"url": _youtube_url(video_id)},
    )
    assert response.status_code == 202
    location = response.headers["Location"]
    await harness.reconciler.reconcile_once()
    return location


async def _wait_for_finished(
    client: AsyncClient,
    location: str,
) -> dict[str, object]:
    """Poll one public job representation until its terminal result is published."""
    for _ in range(100):
        response = await client.get(location)
        payload = response.json()
        assert isinstance(payload, dict)
        if payload["status"] == "finished":
            assert payload["outcome"] == "succeeded"
            return payload
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for a public terminal job response.")


def _transcript_method(finished_job: dict[str, object]) -> object:
    """Return the public transcript method from one terminal job representation."""
    result = finished_job["result"]
    assert isinstance(result, dict)
    transcript = result["transcript"]
    assert isinstance(transcript, dict)
    return transcript["method"]


@pytest.mark.asyncio
async def test_two_native_jobs_overlap_while_a_third_waits_for_native_capacity(
    migrated_postgresql_database_url: str,
    redis_broker_url: str,
    tmp_path: Path,
) -> None:
    """Drive A and B concurrently, then admit C only after one native call exits."""
    async with _running_harness(
        migrated_postgresql_database_url,
        redis_broker_url,
        tmp_path,
    ) as harness:
        try:
            location_a = await _submit_and_reconcile(harness, _NATIVE_A)
            location_b = await _submit_and_reconcile(harness, _NATIVE_B)
            assert await asyncio.to_thread(
                harness.transcriber.entered[_NATIVE_A].wait,
                1,
            )
            assert await asyncio.to_thread(
                harness.transcriber.entered[_NATIVE_B].wait,
                1,
            )

            location_c = await _submit_and_reconcile(harness, _NATIVE_C)
            assert await asyncio.to_thread(
                harness.downloader.prepared[_NATIVE_C].wait,
                1,
            )
            assert not harness.transcriber.entered[_NATIVE_C].is_set()
            processing_response = await harness.client.get(location_c)
            processing_payload = processing_response.json()
            assert isinstance(processing_payload, dict)
            assert processing_payload["status"] == "processing"

            harness.transcriber.release[_NATIVE_A].set()
            assert await asyncio.to_thread(
                harness.transcriber.entered[_NATIVE_C].wait,
                1,
            )
            harness.transcriber.release[_NATIVE_B].set()

            finished_a, finished_b, finished_c = await asyncio.gather(
                _wait_for_finished(harness.client, location_a),
                _wait_for_finished(harness.client, location_b),
                _wait_for_finished(harness.client, location_c),
            )
        finally:
            harness.transcriber.release[_NATIVE_A].set()
            harness.transcriber.release[_NATIVE_B].set()

    assert all(
        _transcript_method(finished) == TranscriptMethod.FASTER_WHISPER
        for finished in (finished_a, finished_b, finished_c)
    )
    assert harness.transcriber.maximum_active == 2
    interval_a = harness.transcriber.intervals[_NATIVE_A]
    interval_b = harness.transcriber.intervals[_NATIVE_B]
    interval_c = harness.transcriber.intervals[_NATIVE_C]
    assert max(interval_a[0], interval_b[0]) < min(interval_a[1], interval_b[1])
    assert interval_c[0] >= interval_a[1]
    assert harness.adapters_factory_calls == [None]
    assert all(
        not request_directory.exists()
        for request_directory in harness.downloader.request_directories
    )


@pytest.mark.asyncio
async def test_caption_job_finishes_while_both_native_permits_are_occupied(
    migrated_postgresql_database_url: str,
    redis_broker_url: str,
    tmp_path: Path,
) -> None:
    """Publish caption D before either blocked native call is released."""
    async with _running_harness(
        migrated_postgresql_database_url,
        redis_broker_url,
        tmp_path,
    ) as harness:
        try:
            await _submit_and_reconcile(harness, _NATIVE_A)
            await _submit_and_reconcile(harness, _NATIVE_B)
            assert await asyncio.to_thread(
                harness.transcriber.entered[_NATIVE_A].wait,
                1,
            )
            assert await asyncio.to_thread(
                harness.transcriber.entered[_NATIVE_B].wait,
                1,
            )

            caption_location = await _submit_and_reconcile(harness, _CAPTION_D)
            caption_finished = await _wait_for_finished(
                harness.client,
                caption_location,
            )
            assert not harness.transcriber.release[_NATIVE_A].is_set()
            assert not harness.transcriber.release[_NATIVE_B].is_set()
            assert _CAPTION_D not in harness.downloader.markers
            assert _CAPTION_D not in harness.transcriber.markers
        finally:
            harness.transcriber.release[_NATIVE_A].set()
            harness.transcriber.release[_NATIVE_B].set()

    assert _transcript_method(caption_finished) == TranscriptMethod.YOUTUBE_CAPTIONS
    assert harness.adapters_factory_calls == [None]
