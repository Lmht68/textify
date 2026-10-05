"""Tests for one worker's physical-GPU ownership lifecycle."""

import asyncio
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from textify.jobs.config import JobDispatchConfig
from textify.jobs.database import create_application_engine
from textify.jobs.gpu_ownership import GpuModelOwnership, GpuOwnershipConflictError
from textify.jobs.repository import PostgresTranscriptionJobRepository
from textify.jobs.service import utc_now
from textify.jobs.worker import TranscriptionWorkerRuntime
from textify.transcription import acquisition, inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.service import TranscriptionAdapters


@pytest.mark.asyncio
async def test_symlinked_lock_directory_is_rejected_before_database_access(
    tmp_path: Path,
) -> None:
    """Keep host-lock files out of an attacker-controlled symlink target."""
    target_directory = tmp_path / "lock-target"
    target_directory.mkdir()
    lock_directory = tmp_path / "gpu-locks"
    lock_directory.symlink_to(target_directory, target_is_directory=True)
    ownership = GpuModelOwnership(
        cast(AsyncEngine, object()),
        "NVIDIA-GPU-SYMLINK",
        lock_directory,
    )

    with pytest.raises(
        RuntimeError,
        match=r"^GPU lock directory must be a writable non-symlink directory\.$",
    ):
        await ownership.acquire()

    assert list(target_directory.iterdir()) == []


@pytest.mark.asyncio
async def test_same_identity_and_shared_host_lock_directory_conflicts(
    postgresql_database_url: str,
    tmp_path: Path,
) -> None:
    """Reject a second owner before it can load another model."""
    engine = create_async_engine(postgresql_database_url)
    first_owner = GpuModelOwnership(
        engine,
        "NVIDIA-GPU-0001",
        tmp_path / "gpu-locks",
    )
    second_owner = GpuModelOwnership(
        engine,
        "NVIDIA-GPU-0001",
        tmp_path / "gpu-locks",
    )

    try:
        await first_owner.acquire()

        with pytest.raises(
            GpuOwnershipConflictError,
            match="GPU ownership is already held for the configured identity.",
        ):
            await second_owner.acquire()

        await first_owner.release()
        await second_owner.acquire()
    finally:
        await first_owner.release()
        await second_owner.release()
        await engine.dispose()


@pytest.mark.asyncio
async def test_same_identity_and_distinct_host_lock_directories_conflicts_in_postgresql(
    postgresql_database_url: str,
    tmp_path: Path,
) -> None:
    """Release a partial host acquisition after PostgreSQL rejects ownership."""
    first_engine = create_async_engine(postgresql_database_url)
    second_engine = create_async_engine(postgresql_database_url)
    first_owner = GpuModelOwnership(
        first_engine,
        "NVIDIA-GPU-0002",
        tmp_path / "first-gpu-locks",
    )
    second_owner = GpuModelOwnership(
        second_engine,
        "NVIDIA-GPU-0002",
        tmp_path / "second-gpu-locks",
    )

    try:
        await first_owner.acquire()

        with pytest.raises(
            GpuOwnershipConflictError,
            match="GPU ownership is already held for the configured identity.",
        ):
            await second_owner.acquire()

        await first_owner.release()
        await second_owner.acquire()
    finally:
        await first_owner.release()
        await second_owner.release()
        await first_engine.dispose()
        await second_engine.dispose()


def _runtime_repository(
    engine: AsyncEngine,
) -> PostgresTranscriptionJobRepository:
    """Build the production repository for one real ownership runtime."""
    return PostgresTranscriptionJobRepository(
        engine=engine,
        maximum_outstanding_jobs=8,
        queue_timeout=timedelta(seconds=20),
        terminal_retention=timedelta(days=1),
        clock=utc_now,
    )


def _runtime_transcription_config(
    temporary_media_root: Path,
    lock_directory: Path,
) -> TranscriptionConfig:
    """Build ownership-ready settings without constructing a real model."""
    return TranscriptionConfig(
        gpu_identity="NVIDIA-GPU-0003",
        gpu_lock_directory=lock_directory,
        temporary_media_root=temporary_media_root,
        transcription_concurrency=1,
        max_media_bytes=1,
        _env_file=None,  # type: ignore[call-arg]
    )


def _recording_adapters_factory(
    factory_calls: list[None],
) -> Callable[[TranscriptionConfig], TranscriptionAdapters]:
    """Build inert provider adapters and record model-factory construction."""

    def build_adapters(_config: TranscriptionConfig) -> TranscriptionAdapters:
        """Record deferred model construction after dual-lock ownership."""
        factory_calls.append(None)
        return TranscriptionAdapters(
            metadata_extractor=cast(inspection.MetadataExtractor, object()),
            caption_provider=cast(acquisition.CaptionProvider, object()),
            audio_downloader=cast(acquisition.AudioDownloader, object()),
            whisper_transcriber=cast(acquisition.WhisperTranscriber, object()),
        )

    return build_adapters


def test_runtime_conflict_prevents_second_model_factory_before_claim_start(
    migrated_postgresql_database_url: str,
    tmp_path: Path,
) -> None:
    """Fail a conflicting runtime before it constructs a second model executor."""
    first_engine = create_application_engine(migrated_postgresql_database_url)
    second_engine = create_application_engine(migrated_postgresql_database_url)
    third_engine = create_application_engine(migrated_postgresql_database_url)
    media_root = tmp_path / "media"
    lock_directory = tmp_path / "gpu-locks"
    dispatch_config = JobDispatchConfig(
        broker_url="redis://127.0.0.1:6379/15",  # type: ignore[arg-type]
        worker_concurrency=1,
        _env_file=None,  # type: ignore[call-arg]
    )
    factory_calls: list[None] = []
    adapters_factory = _recording_adapters_factory(factory_calls)
    first_runtime = TranscriptionWorkerRuntime.from_repository(
        _runtime_repository(first_engine),
        first_engine,
        _runtime_transcription_config(media_root, lock_directory),
        dispatch_config,
        adapters_factory=adapters_factory,
    )
    second_runtime = TranscriptionWorkerRuntime.from_repository(
        _runtime_repository(second_engine),
        second_engine,
        _runtime_transcription_config(media_root, lock_directory),
        dispatch_config,
        adapters_factory=adapters_factory,
    )
    third_runtime = TranscriptionWorkerRuntime.from_repository(
        _runtime_repository(third_engine),
        third_engine,
        _runtime_transcription_config(media_root, lock_directory),
        dispatch_config,
        adapters_factory=adapters_factory,
    )

    async def heartbeat_count() -> int:
        """Count every current worker row through an independent engine loop."""
        readiness_engine = create_application_engine(migrated_postgresql_database_url)
        try:
            async with readiness_engine.connect() as connection:
                count = await connection.scalar(
                    text(
                        "SELECT count(*) FROM service_heartbeat "
                        "WHERE service_role = 'gpu_worker'"
                    )
                )
        finally:
            await readiness_engine.dispose()
        assert isinstance(count, int)
        return count

    try:
        first_runtime.start()
        assert asyncio.run(heartbeat_count()) == 1
        with pytest.raises(
            GpuOwnershipConflictError,
            match="GPU ownership is already held for the configured identity.",
        ):
            second_runtime.start()
        assert asyncio.run(heartbeat_count()) == 1

        assert factory_calls == [None]
        first_runtime.shutdown()
        assert asyncio.run(heartbeat_count()) == 0

        third_runtime.start()
        assert factory_calls == [None, None]
        assert asyncio.run(heartbeat_count()) == 1
    finally:
        first_runtime.shutdown()
        second_runtime.shutdown()
        third_runtime.shutdown()
        asyncio.run(first_engine.dispose())
        asyncio.run(second_engine.dispose())
        asyncio.run(third_engine.dispose())
