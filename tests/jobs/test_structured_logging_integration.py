"""Three-process structured logging contract proof for Transcription Jobs."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient, HTTPError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from textify.jobs.database import create_application_engine
from textify.transcription import inspection
from textify.transcription.service import TranscriptionAdapters
from textify.transcription.types import Transcript

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_YOUTUBE_VIDEO_ID = "dQw4w9WgXcQ"
_SUBMITTED_URL = (
    f"https://www.youtube.com/watch?v={_YOUTUBE_VIDEO_ID}"
    "&request-body-sentinel=submitted-url-sentinel"
)
_PROVIDER_DETAIL_SENTINEL = "provider-detail-sentinel"
_TRANSCRIPT_CONTENT_SENTINEL = "transcript-content-sentinel"

_ENVELOPE_FIELDS = frozenset(
    {
        "timestamp",
        "level",
        "logger",
        "message",
        "event",
        "service",
        "service_role",
        "instance_id",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {
        "request_id",
        "internal_job_id",
        "platform",
        "source_id",
        "stage",
        "method",
        "duration_seconds",
        "elapsed_seconds",
        "code",
        "job_status",
        "job_outcome",
        "http_status_code",
        "http_method",
        "http_route",
    }
)
_APPROVED_TOP_LEVEL_FIELDS = _ENVELOPE_FIELDS | _OPTIONAL_FIELDS
_EVENT_LEVELS = {
    "service_starting": "INFO",
    "service_ready": "INFO",
    "service_failed": "ERROR",
    "service_stopping": "INFO",
    "request_completed": "INFO",
    "request_interrupted": "INFO",
    "request_failed": "ERROR",
    "transcription_job_admitted": "INFO",
    "transcription_job_cancellation_accepted": "INFO",
    "execution_attempt_claimed": "INFO",
    "execution_attempt_ownership_lost": "WARNING",
    "transcription_job_finished": "INFO",
    "source_inspection_failed": "DEBUG",
    "caption_track_unavailable": "DEBUG",
    "caption_acquisition_failed": "DEBUG",
    "audio_download_failed": "DEBUG",
    "native_transcription_failed": "DEBUG",
    "transcript_acquired": "DEBUG",
    "resource_cleanup_failed": "ERROR",
    "transcription_job_processing_failed": "ERROR",
    "job_store_unavailable": "ERROR",
    "job_dispatch_unavailable": "ERROR",
    "malformed_job_dispatch_rejected": "WARNING",
    "worker_runtime_unavailable": "ERROR",
    "gpu_ownership_failed": "ERROR",
    "service_readiness_failed": "ERROR",
}
_REQUIRED_EVENT_FIELDS = {
    "service_failed": frozenset({"stage", "code"}),
    "request_completed": frozenset(
        {"request_id", "http_status_code", "http_method", "elapsed_seconds"}
    ),
    "request_interrupted": frozenset({"request_id", "http_method", "elapsed_seconds"}),
    "request_failed": frozenset(
        {
            "request_id",
            "http_status_code",
            "http_method",
            "elapsed_seconds",
            "code",
        }
    ),
    "transcription_job_admitted": frozenset(
        {"request_id", "internal_job_id", "job_status"}
    ),
    "transcription_job_cancellation_accepted": frozenset(
        {"request_id", "internal_job_id", "job_status"}
    ),
    "execution_attempt_claimed": frozenset({"internal_job_id", "job_status"}),
    "execution_attempt_ownership_lost": frozenset({"internal_job_id"}),
    "transcription_job_finished": frozenset(
        {"internal_job_id", "job_status", "job_outcome"}
    ),
    "source_inspection_failed": frozenset({"stage", "code"}),
    "caption_track_unavailable": frozenset({"stage", "code"}),
    "caption_acquisition_failed": frozenset({"stage", "code"}),
    "audio_download_failed": frozenset({"stage", "code"}),
    "native_transcription_failed": frozenset({"stage", "code"}),
    "transcript_acquired": frozenset({"stage", "method"}),
    "resource_cleanup_failed": frozenset({"stage", "code"}),
    "transcription_job_processing_failed": frozenset({"internal_job_id", "code"}),
    "job_store_unavailable": frozenset({"code"}),
    "job_dispatch_unavailable": frozenset({"internal_job_id", "code"}),
    "malformed_job_dispatch_rejected": frozenset({"code"}),
    "worker_runtime_unavailable": frozenset({"internal_job_id", "code"}),
    "gpu_ownership_failed": frozenset({"stage", "code"}),
    "service_readiness_failed": frozenset({"stage", "code"}),
}
_STAGE_VALUES = frozenset(
    {
        "request",
        "metadata",
        "captions",
        "audio_download",
        "transcription",
        "processing",
        "dispatch",
        "readiness",
        "startup",
        "runtime",
        "shutdown",
        "cleanup",
    }
)
_SAFE_ROUTES = frozenset(
    {
        "/health",
        "/api/v1/transcription-jobs",
        "/api/v1/transcription-jobs/{job_id}",
        "/api/v1/transcription-jobs/{job_id}/cancellation",
    }
)


@dataclass
class _CapturedProcess:
    """Own one child process and its continuously validated stderr stream."""

    name: str
    role: str
    process: subprocess.Popen[str]
    raw_lines: list[str] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    parser_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _reader: threading.Thread = field(init=False, repr=False)

    def start_draining(self) -> None:
        """Start draining and validating stderr before the child can write logs."""
        self._reader = threading.Thread(
            target=self._drain_stderr,
            name=f"{self.name}-structured-log-drainer",
            daemon=True,
        )
        self._reader.start()

    def has_event(self, event_name: str) -> bool:
        """Return whether the process has emitted one named registered event."""
        with self._lock:
            return any(event["event"] == event_name for event in self.events)

    def events_snapshot(self) -> tuple[dict[str, object], ...]:
        """Return a stable copy of the validated parsed stream."""
        with self._lock:
            return tuple(self.events)

    def raw_stderr(self) -> str:
        """Return the complete raw stream after the child process has stopped."""
        with self._lock:
            return "\n".join(self.raw_lines)

    async def wait_for_reader(self) -> None:
        """Join the stderr drainer after its pipe has closed."""
        await asyncio.to_thread(self._reader.join, 10)
        assert not self._reader.is_alive()

    def _drain_stderr(self) -> None:
        """Parse and validate every nonempty stderr line as soon as it arrives."""
        stderr = self.process.stderr
        if stderr is None:
            self._set_parser_error("stderr pipe was not created")
            return
        for raw_line in stderr:
            rendered_line = raw_line.rstrip("\r\n")
            with self._lock:
                self.raw_lines.append(rendered_line)
            if not rendered_line:
                continue
            try:
                decoded = json.loads(rendered_line)
                if not isinstance(decoded, dict) or not all(
                    isinstance(key, str) for key in decoded
                ):
                    raise ValueError("structured event must be a JSON object")
                event = cast(dict[str, object], decoded)
                _validate_log_event(event, self.role)
            except (AssertionError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._set_parser_error(type(exc).__name__)
                continue
            with self._lock:
                self.events.append(event)

    def _set_parser_error(self, error_name: str) -> None:
        """Record the first opaque reader failure without retaining raw log content."""
        with self._lock:
            if self.parser_error is None:
                self.parser_error = error_name


class _FailingCaptionTrack:
    """Raise a provider failure without exposing its provider detail to logging."""

    @property
    def language_code(self) -> str:
        """Return the declared source language."""
        return "en"

    @property
    def is_generated(self) -> bool:
        """Return that this deterministic track is original language."""
        return False

    def fetch_segments(self) -> tuple[tuple[float, float, str], ...]:
        """Raise the first-track provider failure required by the smoke flow."""
        from textify.transcription.acquisition import _CaptionProviderFailure

        raise _CaptionProviderFailure(_PROVIDER_DETAIL_SENTINEL)


class _BlockingCaptionTrack:
    """Block successful captions until the parent proves durable ownership."""

    @property
    def language_code(self) -> str:
        """Return the declared source language."""
        return "en"

    @property
    def is_generated(self) -> bool:
        """Return that this deterministic track is original language."""
        return False

    def fetch_segments(self) -> tuple[tuple[float, float, str], ...]:
        """Wait for the parent and then return one deterministic caption segment."""
        _signal_child_control("caption_track_blocked")
        _wait_for_child_control("release_caption")
        return ((0.0, 1.0, _TRANSCRIPT_CONTENT_SENTINEL),)


class _ProcessCaptionProvider:
    """Supply a failed first track and a parent-controlled successful second track."""

    def list_tracks(
        self,
        video_id: str,
    ) -> tuple[_FailingCaptionTrack, _BlockingCaptionTrack]:
        """Return tracks only for the submitted public YouTube video."""
        if video_id != _YOUTUBE_VIDEO_ID:
            raise AssertionError("Unexpected video ID in process smoke.")
        return (_FailingCaptionTrack(), _BlockingCaptionTrack())


class _ProcessMetadataExtractor:
    """Return provider-shaped metadata without network access."""

    def extract(
        self,
        provider_url: str,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> inspection.ExtractedMetadata:
        """Return deterministic metadata accepted by normal production processing."""
        del deadline, cancellation_event
        return inspection.ExtractedMetadata(
            {
                "id": _YOUTUBE_VIDEO_ID,
                "extractor_key": "Youtube",
                "webpage_url": provider_url,
                "title": "Structured logging smoke title",
                "description": "Structured logging smoke description",
                "channel": "Structured logging smoke channel",
                "duration": 12,
                "language": "en",
            }
        )


class _UnexpectedNativePath:
    """Fail the smoke if the successful captions path tries native inference."""

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Reject an unexpected audio download."""
        del source_url, destination, deadline, cancellation_event
        raise AssertionError("Caption processing must not download audio.")

    def transcribe(
        self,
        audio_path: Path,
        *,
        include_segments: bool = True,
    ) -> Transcript:
        """Reject an unexpected native transcription."""
        del audio_path, include_segments
        raise AssertionError("Caption processing must not load a model.")


def _validate_log_event(event: Mapping[str, object], expected_role: str) -> None:
    """Assert one parsed stderr record conforms to the public logging contract."""
    assert set(event) <= _APPROVED_TOP_LEVEL_FIELDS
    assert _ENVELOPE_FIELDS <= set(event)
    assert all(value is not None for value in event.values())

    timestamp = event["timestamp"]
    assert type(timestamp) is str
    assert timestamp.endswith("Z")
    assert datetime.fromisoformat(timestamp).tzinfo is UTC

    event_name = event["event"]
    assert type(event_name) is str
    assert event_name in _EVENT_LEVELS
    assert event["level"] == _EVENT_LEVELS[event_name]
    assert event["service"] == "textify"
    assert event["service_role"] == expected_role

    logger_name = event["logger"]
    assert type(logger_name) is str
    assert logger_name == "textify" or logger_name.startswith("textify.")
    assert type(event["message"]) is str
    assert bool(event["message"])
    assert _is_canonical_uuid4(event["instance_id"])

    assert _REQUIRED_EVENT_FIELDS.get(event_name, frozenset()) <= set(event)
    for field_name in _OPTIONAL_FIELDS & set(event):
        _validate_optional_field(field_name, event[field_name])


def _validate_optional_field(field_name: str, value: object) -> None:
    """Assert one non-null optional field has its approved native JSON type."""
    if field_name == "request_id":
        assert _is_canonical_uuid4(value)
        return
    if field_name in {"platform", "source_id", "method", "code"}:
        assert type(value) is str
        assert bool(value)
        return
    if field_name == "internal_job_id":
        assert type(value) is int
        assert value > 0
        return
    if field_name == "stage":
        assert value in _STAGE_VALUES
        return
    if field_name == "duration_seconds":
        assert type(value) is int
        assert value >= 0
        return
    if field_name == "elapsed_seconds":
        assert (type(value) is int and value >= 0) or (
            type(value) is float and isfinite(value) and value >= 0
        )
        return
    if field_name == "job_status":
        assert value in {"queued", "processing", "finished"}
        return
    if field_name == "job_outcome":
        assert value in {"succeeded", "failed", "cancelled"}
        return
    if field_name == "http_status_code":
        assert type(value) is int
        assert 100 <= value <= 599
        return
    if field_name == "http_method":
        assert type(value) is str
        assert value and value == value.upper()
        return
    assert field_name == "http_route"
    assert value in _SAFE_ROUTES


def _is_canonical_uuid4(value: object) -> bool:
    """Return whether a JSON scalar is a lowercase canonical UUIDv4."""
    if type(value) is not str:
        return False
    try:
        parsed_value = UUID(value)
    except ValueError:
        return False
    return parsed_value.version == 4 and str(parsed_value) == value


def _child_control_directory() -> Path:
    """Return the process-shared control directory required by child modes."""
    raw_path = os.environ.get("TEXTIFY_PROCESS_SMOKE_CONTROL_DIRECTORY")
    if raw_path is None:
        raise RuntimeError("Process smoke control directory is required.")
    return Path(raw_path)


def _signal_child_control(name: str) -> None:
    """Create one parent-observable signal file from a child process."""
    _child_control_directory().joinpath(name).touch()


def _wait_for_child_control(name: str) -> None:
    """Wait for a parent-controlled release while allowing test cleanup to unblock it."""
    release_path = _child_control_directory().joinpath(name)
    while not release_path.exists():
        time.sleep(0.01)


def _build_process_adapters() -> TranscriptionAdapters:
    """Build the deterministic adapter bundle installed only in worker child mode."""
    return TranscriptionAdapters(
        metadata_extractor=_ProcessMetadataExtractor(),
        caption_provider=_ProcessCaptionProvider(),
        audio_downloader=_UnexpectedNativePath(),
        whisper_transcriber=_UnexpectedNativePath(),
    )


def _run_api_child() -> None:
    """Run the production API main entrypoint in one isolated interpreter."""
    from textify.main import main

    main()


def _run_reconciler_child() -> None:
    """Patch only publication delivery to expose the leased-before-publish barrier."""
    from textify.jobs import reconciler
    from textify.jobs.dispatch import CeleryJobDispatchPublisher
    from textify.jobs.types import JobDispatch

    original_publish = CeleryJobDispatchPublisher.publish

    async def publish_after_parent_release(
        publisher: CeleryJobDispatchPublisher,
        dispatch: JobDispatch,
    ) -> None:
        """Pause exactly after the reconciler owns the first durable lease."""
        _signal_child_control("publication_leased")
        release_path = _child_control_directory().joinpath("release_publication")
        while not release_path.exists():
            await asyncio.sleep(0.01)
        await original_publish(publisher, dispatch)

    publish_method_name = "publish"
    setattr(
        CeleryJobDispatchPublisher,
        publish_method_name,
        publish_after_parent_release,
    )
    reconciler.main()


def _run_worker_child() -> None:
    """Patch only worker runtime construction with deterministic normal adapters."""
    from datetime import timedelta

    from textify.jobs import worker
    from textify.jobs.config import JobConfig, JobDispatchConfig
    from textify.jobs.repository import PostgresTranscriptionJobRepository
    from textify.jobs.service import utc_now
    from textify.jobs.worker import TranscriptionWorkerRuntime
    from textify.transcription.config import TranscriptionConfig

    async def create_deterministic_runtime(
        engine: AsyncEngine,
        job_config: JobConfig,
        dispatch_config: JobDispatchConfig,
        transcription_config: TranscriptionConfig,
    ) -> TranscriptionWorkerRuntime:
        """Build production ownership with only deterministic provider adapters."""
        repository = PostgresTranscriptionJobRepository(
            engine=engine,
            maximum_outstanding_jobs=job_config.max_outstanding_jobs,
            queue_timeout=timedelta(seconds=job_config.job_queue_timeout_seconds),
            terminal_retention=timedelta(seconds=job_config.job_retention_seconds),
            clock=utc_now,
        )
        return TranscriptionWorkerRuntime.from_repository(
            repository,
            engine,
            transcription_config,
            dispatch_config,
            adapters_factory=create_deterministic_adapters,
        )

    def create_deterministic_adapters(
        configuration: TranscriptionConfig,
    ) -> TranscriptionAdapters:
        """Return deterministic adapters while consuming the factory configuration."""
        del configuration
        return _build_process_adapters()

    runtime_factory_name = "_create_worker_runtime"
    setattr(worker, runtime_factory_name, create_deterministic_runtime)
    worker.main()


def _run_child_mode(arguments: list[str]) -> None:
    """Dispatch the direct-interpreter process mode selected by the parent test."""
    if arguments == ["api"]:
        _run_api_child()
        return
    if arguments == ["reconciler"]:
        _run_reconciler_child()
        return
    if arguments == ["worker"]:
        _run_worker_child()
        return
    raise SystemExit("Expected one process mode: api, reconciler, or worker.")


def _reserve_loopback_port() -> int:
    """Reserve an ephemeral loopback port number for the API child process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved_socket:
        reserved_socket.bind(("127.0.0.1", 0))
        return cast(int, reserved_socket.getsockname()[1])


def _child_environment(
    database_url: str,
    broker_url: str,
    api_port: int,
    control_directory: Path,
    media_root: Path,
    gpu_lock_directory: Path,
) -> dict[str, str]:
    """Build one shared process environment for all production child entrypoints."""
    environment = os.environ.copy()
    environment.pop("TEXTIFY_CACHE_URL", None)
    environment.pop("TEXTIFY_HF_TOKEN", None)
    existing_python_path = environment.get("PYTHONPATH")
    environment.update(
        {
            "TEXTIFY_DATABASE_URL": database_url,
            "TEXTIFY_BROKER_URL": broker_url,
            "TEXTIFY_LOG_LEVEL": "DEBUG",
            "TEXTIFY_HOST": "127.0.0.1",
            "TEXTIFY_PORT": str(api_port),
            "TEXTIFY_WORKER_CONCURRENCY": "1",
            "TEXTIFY_RECONCILER_INTERVAL_SECONDS": "0.05",
            "TEXTIFY_ATTEMPT_HEARTBEAT_SECONDS": "0.25",
            "TEXTIFY_ATTEMPT_LEASE_SECONDS": "5",
            "TEXTIFY_SERVICE_HEARTBEAT_TTL_SECONDS": "10",
            "TEXTIFY_GPU_IDENTITY": "structured-logging-process-smoke",
            "TEXTIFY_GPU_LOCK_DIRECTORY": str(gpu_lock_directory),
            "TEXTIFY_TEMPORARY_MEDIA_ROOT": str(media_root),
            "TEXTIFY_TRANSCRIPTION_CONCURRENCY": "1",
            "TEXTIFY_MAX_MEDIA_BYTES": "1024",
            "TEXTIFY_PROCESS_SMOKE_CONTROL_DIRECTORY": str(control_directory),
        }
    )
    python_path_parts = [str(_PROJECT_ROOT)]
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    return environment


def _start_child(
    name: str,
    role: str,
    environment: Mapping[str, str],
) -> _CapturedProcess:
    """Launch one production main entrypoint and immediately drain its stderr."""
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), name],
        cwd=_PROJECT_ROOT,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    captured_process = _CapturedProcess(name, role, process)
    captured_process.start_draining()
    return captured_process


async def _wait_until(
    condition: Callable[[], bool],
    message: str,
    processes: tuple[_CapturedProcess, ...],
) -> None:
    """Wait with bounded polling while surfacing parser and child-exit failures."""
    deadline = asyncio.get_running_loop().time() + 20
    while asyncio.get_running_loop().time() < deadline:
        _assert_children_healthy(processes)
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(message)


def _assert_children_healthy(processes: tuple[_CapturedProcess, ...]) -> None:
    """Fail on malformed output or an unexpectedly exited child without echoing logs."""
    for captured_process in processes:
        assert captured_process.parser_error is None
        assert captured_process.process.poll() is None


async def _wait_for_api_health(client: AsyncClient) -> None:
    """Wait until the real loopback API accepts ready topology health requests."""
    deadline = asyncio.get_running_loop().time() + 20
    while asyncio.get_running_loop().time() < deadline:
        try:
            response = await client.get("/health")
        except HTTPError:
            await asyncio.sleep(0.02)
            continue
        if response.status_code == 200:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("Timed out waiting for the loopback API health endpoint.")


async def _wait_for_succeeded_status(
    client: AsyncClient,
    status_link: str,
) -> dict[str, object]:
    """Poll the real API until the dispatched job commits its success outcome."""
    for _ in range(200):
        response = await client.get(status_link)
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload, dict)
        if payload.get("status") == "finished":
            assert payload.get("outcome") == "succeeded"
            return cast(dict[str, object], payload)
        await asyncio.sleep(0.025)
    raise AssertionError("Timed out waiting for the Transcription Job success outcome.")


async def _read_execution_identities(
    engine: AsyncEngine,
    submitted_url: str,
) -> tuple[int, UUID, UUID]:
    """Read the current leased attempt identity before broker publication resumes."""
    statement = text(
        "SELECT job.id, attempt.publication_lease_owner, "
        "attempt.execution_attempt_token "
        "FROM transcription_job AS job "
        "JOIN transcription_job_execution_attempt AS attempt "
        "ON attempt.job_id = job.id "
        "WHERE job.submitted_url = :submitted_url"
    )
    async with engine.connect() as connection:
        row = (
            (await connection.execute(statement, {"submitted_url": submitted_url}))
            .mappings()
            .one()
        )
    internal_job_id = row["id"]
    publication_lease_owner = row["publication_lease_owner"]
    execution_attempt_token = row["execution_attempt_token"]
    assert type(internal_job_id) is int
    assert internal_job_id > 0
    assert isinstance(publication_lease_owner, UUID)
    assert publication_lease_owner.version == 4
    assert isinstance(execution_attempt_token, UUID)
    assert execution_attempt_token.version == 4
    return internal_job_id, publication_lease_owner, execution_attempt_token


async def _read_claim_and_readiness_identities(
    engine: AsyncEngine,
    internal_job_id: int,
) -> tuple[UUID, dict[str, UUID]]:
    """Read the active worker claim and both live process readiness identities."""
    claim_statement = text(
        "SELECT worker_owner FROM transcription_job_execution_attempt "
        "WHERE job_id = :internal_job_id"
    )
    readiness_statement = text(
        "SELECT service_role, instance_id FROM service_heartbeat "
        "WHERE service_role IN ('reconciler', 'gpu_worker')"
    )
    async with engine.connect() as connection:
        worker_owner = await connection.scalar(
            claim_statement,
            {"internal_job_id": internal_job_id},
        )
        readiness_rows = (
            (await connection.execute(readiness_statement)).mappings().all()
        )
    assert isinstance(worker_owner, UUID)
    assert worker_owner.version == 4
    readiness_ids: dict[str, UUID] = {}
    for row in readiness_rows:
        service_role = row["service_role"]
        instance_id = row["instance_id"]
        assert service_role in {"reconciler", "gpu_worker"}
        assert isinstance(instance_id, UUID)
        assert instance_id.version == 4
        readiness_ids[service_role] = instance_id
    assert set(readiness_ids) == {"reconciler", "gpu_worker"}
    return worker_owner, readiness_ids


async def _read_terminal_outcome(
    engine: AsyncEngine,
    internal_job_id: int,
) -> tuple[str, str]:
    """Read the committed terminal status and outcome from PostgreSQL."""
    statement = text(
        "SELECT status, outcome FROM transcription_job WHERE id = :internal_job_id"
    )
    async with engine.connect() as connection:
        row = (
            (await connection.execute(statement, {"internal_job_id": internal_job_id}))
            .mappings()
            .one()
        )
    status = row["status"]
    outcome = row["outcome"]
    assert type(status) is str
    assert type(outcome) is str
    return status, outcome


async def _read_heartbeat_count(engine: AsyncEngine) -> int:
    """Return the number of process-owned readiness rows remaining after shutdown."""
    async with engine.connect() as connection:
        count = await connection.scalar(text("SELECT count(*) FROM service_heartbeat"))
    assert type(count) is int
    return count


async def _stop_child(captured_process: _CapturedProcess) -> None:
    """Request SIGTERM, bound graceful exit, and use kill only as cleanup fallback."""
    process = captured_process.process
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            await asyncio.to_thread(process.wait, 20)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 10)
    await captured_process.wait_for_reader()


def _release_control(control_directory: Path, name: str) -> None:
    """Unblock a known child barrier during either success or failure cleanup."""
    control_directory.joinpath(name).touch()


def _assert_absent(raw_stderr: str, candidates: Mapping[str, str]) -> None:
    """Fail by label only if a sensitive sentinel appears in any captured stream."""
    leaked_labels = [
        label
        for label, candidate in candidates.items()
        if candidate and candidate in raw_stderr
    ]
    assert not leaked_labels, (
        f"Sensitive logging values leaked: {', '.join(leaked_labels)}"
    )


@pytest.mark.asyncio
async def test_three_process_structured_logging_contract(
    migrated_postgresql_database_url: str,
    redis_broker_url: str,
    tmp_path: Path,
) -> None:
    """Prove real API, reconciler, and worker logs stay safe across one success flow."""
    control_directory = tmp_path / "process-control"
    media_root = tmp_path / "media"
    gpu_lock_directory = tmp_path / "gpu-locks"
    control_directory.mkdir()
    media_root.mkdir()
    gpu_lock_directory.mkdir()
    api_port = _reserve_loopback_port()
    environment = _child_environment(
        migrated_postgresql_database_url,
        redis_broker_url,
        api_port,
        control_directory,
        media_root,
        gpu_lock_directory,
    )
    engine = create_application_engine(migrated_postgresql_database_url)
    reconciler_process: _CapturedProcess | None = None
    worker_process: _CapturedProcess | None = None
    api_process: _CapturedProcess | None = None
    normally_stopped = False

    try:
        reconciler_process = _start_child("reconciler", "reconciler", environment)
        worker_process = _start_child("worker", "gpu_worker", environment)
        processing_processes = (reconciler_process, worker_process)
        await _wait_until(
            lambda: (
                reconciler_process.has_event("service_ready")
                and worker_process.has_event("service_ready")
            ),
            "Timed out waiting for reconciler and worker service readiness.",
            processing_processes,
        )

        api_process = _start_child("api", "api", environment)
        all_processes = (api_process, reconciler_process, worker_process)
        await _wait_until(
            lambda: api_process.has_event("service_ready"),
            "Timed out waiting for API service readiness.",
            all_processes,
        )

        async with AsyncClient(base_url=f"http://127.0.0.1:{api_port}") as client:
            await _wait_for_api_health(client)
            submission = await client.post(
                "/api/v1/transcription-jobs",
                json={"url": _SUBMITTED_URL},
                headers={"Idempotency-Key": str(uuid4())},
            )
            assert submission.status_code == 202
            submission_request_id = submission.headers["X-Request-ID"]
            assert _is_canonical_uuid4(submission_request_id)
            public_response = submission.json()
            assert isinstance(public_response, dict)
            status_link = public_response["links"]["self"]
            cancellation_link = public_response["links"]["cancel"]
            public_job_id = public_response["id"]
            assert type(status_link) is str
            assert type(cancellation_link) is str
            assert type(public_job_id) is str

            await _wait_until(
                lambda: control_directory.joinpath("publication_leased").exists(),
                "Timed out waiting for the reconciler publication lease barrier.",
                all_processes,
            )
            (
                internal_job_id,
                publication_lease_owner,
                execution_attempt_token,
            ) = await _read_execution_identities(engine, _SUBMITTED_URL)
            _release_control(control_directory, "release_publication")

            await _wait_until(
                lambda: (
                    worker_process.has_event("execution_attempt_claimed")
                    and control_directory.joinpath("caption_track_blocked").exists()
                ),
                "Timed out waiting for durable claim and blocked caption processing.",
                all_processes,
            )
            worker_owner, readiness_ids = await _read_claim_and_readiness_identities(
                engine,
                internal_job_id,
            )
            _release_control(control_directory, "release_caption")
            completed_response = await _wait_for_succeeded_status(client, status_link)

        assert completed_response["status"] == "finished"
        assert completed_response["outcome"] == "succeeded"
        result = completed_response["result"]
        assert isinstance(result, dict)
        transcript = result["transcript"]
        assert isinstance(transcript, dict)
        assert transcript["text"] == _TRANSCRIPT_CONTENT_SENTINEL
        assert await _read_terminal_outcome(engine, internal_job_id) == (
            "finished",
            "succeeded",
        )

        await _stop_child(worker_process)
        await _stop_child(reconciler_process)
        await _stop_child(api_process)
        normally_stopped = True

        assert await _read_heartbeat_count(engine) == 0
        for captured_process in all_processes:
            assert captured_process.parser_error is None

        api_events = api_process.events_snapshot()
        reconciler_events = reconciler_process.events_snapshot()
        worker_events = worker_process.events_snapshot()
        all_streams = {
            "api": api_events,
            "reconciler": reconciler_events,
            "worker": worker_events,
        }
        for stream_events in all_streams.values():
            assert stream_events
            assert len({event["instance_id"] for event in stream_events}) == 1
            assert {event["event"] for event in stream_events} >= {
                "service_starting",
                "service_ready",
                "service_stopping",
            }
            assert not any(
                event["event"] == "service_failed" for event in stream_events
            )

        api_instance_id = next(iter({event["instance_id"] for event in api_events}))
        reconciler_instance_id = next(
            iter({event["instance_id"] for event in reconciler_events})
        )
        worker_instance_id = next(
            iter({event["instance_id"] for event in worker_events})
        )
        assert (
            len(
                {
                    api_instance_id,
                    reconciler_instance_id,
                    worker_instance_id,
                    str(publication_lease_owner),
                    str(worker_owner),
                    str(readiness_ids["reconciler"]),
                    str(readiness_ids["gpu_worker"]),
                    str(execution_attempt_token),
                }
            )
            == 8
        )

        admission_events = [
            event
            for event in api_events
            if event["event"] == "transcription_job_admitted"
        ]
        assert len(admission_events) == 1
        assert admission_events[0]["request_id"] == submission_request_id
        assert admission_events[0]["internal_job_id"] == internal_job_id
        submission_completed_events = [
            event
            for event in api_events
            if event["event"] == "request_completed"
            and event.get("request_id") == submission_request_id
        ]
        assert len(submission_completed_events) == 1
        assert all(
            "request_id" not in event for event in (*reconciler_events, *worker_events)
        )

        required_job_events = {
            "execution_attempt_claimed",
            "caption_track_unavailable",
            "transcript_acquired",
            "transcription_job_finished",
        }
        job_events = [
            event
            for event in (*api_events, *reconciler_events, *worker_events)
            if event["event"]
            in required_job_events
            | {"transcription_job_admitted", "job_dispatch_unavailable"}
        ]
        assert required_job_events <= {event["event"] for event in job_events}
        assert all(event["internal_job_id"] == internal_job_id for event in job_events)
        assert all(
            event.get("job_status") == "processing"
            for event in job_events
            if event["event"] == "execution_attempt_claimed"
        )
        assert all(
            event.get("job_status") == "finished"
            and event.get("job_outcome") == "succeeded"
            for event in job_events
            if event["event"] == "transcription_job_finished"
        )

        raw_stderr = "\n".join(
            captured_process.raw_stderr()
            for captured_process in (api_process, reconciler_process, worker_process)
        )
        database_parts = urlsplit(migrated_postgresql_database_url)
        broker_parts = urlsplit(redis_broker_url)
        _assert_absent(
            raw_stderr,
            {
                "submitted URL": _SUBMITTED_URL,
                "request body marker": "request-body-sentinel",
                "media root": str(media_root),
                "GPU lock directory": str(gpu_lock_directory),
                "public job identifier": public_job_id,
                "status capability": status_link,
                "cancellation capability": cancellation_link,
                "transcript content": _TRANSCRIPT_CONTENT_SENTINEL,
                "database URL": migrated_postgresql_database_url,
                "database username": database_parts.username or "",
                "database password": database_parts.password or "",
                "broker URL": redis_broker_url,
                "broker username": broker_parts.username or "",
                "broker password": broker_parts.password or "",
                "provider detail": _PROVIDER_DETAIL_SENTINEL,
                "execution attempt token": str(execution_attempt_token),
                "publication lease owner": str(publication_lease_owner),
                "worker claim owner": str(worker_owner),
                "reconciler readiness identity": str(readiness_ids["reconciler"]),
                "worker readiness identity": str(readiness_ids["gpu_worker"]),
                "provider exception class": "_CaptionProviderFailure",
                "traceback": "Traceback",
            },
        )
    finally:
        _release_control(control_directory, "release_publication")
        _release_control(control_directory, "release_caption")
        if not normally_stopped:
            for process_for_cleanup in (
                worker_process,
                reconciler_process,
                api_process,
            ):
                if process_for_cleanup is not None:
                    await _stop_child(process_for_cleanup)
        await engine.dispose()


if __name__ == "__main__":
    _run_child_mode(sys.argv[1:])
