"""Tests for one claimed Execution Attempt processor."""

from uuid import uuid4

import pytest

from textify.jobs.types import ClaimedTranscriptionJob, JobDispatch
from textify.transcription.exceptions import TranscriptionFailedError
from textify.transcription.types import (
    Platform,
    Segment,
    Source,
    TimedTranscript,
    TranscriptionResult,
    TranscriptMethod,
)


class RecordingExecutionRepository:
    """Expose one claimed job and record terminal publication attempts."""

    def __init__(self, claimed_job: ClaimedTranscriptionJob | None) -> None:
        """Initialize the private claim and empty terminal publication records."""
        self._claimed_job = claimed_job
        self.claimed_dispatches: list[JobDispatch] = []
        self.successes: list[tuple[JobDispatch, dict[str, object]]] = []
        self.failures: list[tuple[JobDispatch, str, str]] = []
        self.cancelled_dispatches: list[JobDispatch] = []
        self.success_published = True
        self.failure_published = True

    async def claim_dispatch(
        self, dispatch: JobDispatch
    ) -> ClaimedTranscriptionJob | None:
        """Return the configured claim only for its exact dispatch."""
        self.claimed_dispatches.append(dispatch)
        if self._claimed_job is None or dispatch != self._claimed_job.dispatch:
            return None
        return self._claimed_job

    async def publish_success(
        self,
        dispatch: JobDispatch,
        projected_result: dict[str, object],
    ) -> bool:
        """Record the requested success publication outcome."""
        self.successes.append((dispatch, projected_result))
        return self.success_published

    async def publish_failure(
        self,
        dispatch: JobDispatch,
        error_code: str,
        error_message: str,
    ) -> bool:
        """Record the requested failure publication outcome."""
        self.failures.append((dispatch, error_code, error_message))
        return self.failure_published

    async def publish_cancelled(self, dispatch: JobDispatch) -> bool:
        """Record cancellation publication after provider cleanup."""
        self.cancelled_dispatches.append(dispatch)
        return True


class ResultExecutor:
    """Return one normalized result and release executor cleanup."""

    def __init__(self) -> None:
        """Initialize invocation observation."""
        self.calls = 0

    async def execute(
        self, _submitted: object, *, control: object, **_kwargs: object
    ) -> TranscriptionResult:
        """Return a fixed result after exposing provider cleanup completion."""
        self.calls += 1
        control.cleanup_complete.set()  # type: ignore[attr-defined]
        return _result()


class FailingExecutor:
    """Raise a safe provider failure and release executor cleanup."""

    def __init__(self) -> None:
        """Initialize invocation observation."""
        self.calls = 0

    async def execute(
        self, _submitted: object, *, control: object, **_kwargs: object
    ) -> TranscriptionResult:
        """Raise the configured safe provider error."""
        self.calls += 1
        control.cleanup_complete.set()  # type: ignore[attr-defined]
        raise TranscriptionFailedError()


def _dispatch() -> JobDispatch:
    """Build one valid private Execution Attempt dispatch.

    Returns:
        Canonical internal dispatch identity.
    """
    return JobDispatch(internal_job_id=1, execution_attempt_token=uuid4())


def _claimed_job(dispatch: JobDispatch) -> ClaimedTranscriptionJob:
    """Build one private claimed job for a valid public source URL.

    Args:
        dispatch: Attempt identity claimed by the processor.

    Returns:
        Private inputs for one execution.
    """
    return ClaimedTranscriptionJob(
        dispatch=dispatch,
        submitted_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        exclusions=(),
    )


def _result() -> TranscriptionResult:
    """Build one normalized result independently of processor projection.

    Returns:
        Stable source and timed transcript for terminal publication.
    """
    return TranscriptionResult(
        Source(
            platform=Platform.YOUTUBE,
            video_id="dQw4w9WgXcQ",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Title",
            description="Description",
            channel="Creator",
            duration_seconds=12,
        ),
        TimedTranscript(
            TranscriptMethod.FASTER_WHISPER,
            "en",
            "One",
            (Segment(0.0, 1.0, "One"),),
        ),
    )


@pytest.mark.asyncio
async def test_processor_ignores_stale_dispatch_before_provider_execution() -> None:
    """Do not run providers or terminal writes when an attempt cannot be claimed."""
    from textify.jobs.processor import TranscriptionJobProcessor

    dispatch = _dispatch()
    repository = RecordingExecutionRepository(claimed_job=None)
    executor = ResultExecutor()

    await TranscriptionJobProcessor(repository, executor).process(dispatch)

    assert repository.claimed_dispatches == [dispatch]
    assert executor.calls == 0
    assert repository.successes == []
    assert repository.failures == []
    assert repository.cancelled_dispatches == []


@pytest.mark.asyncio
async def test_processor_publishes_projected_success_for_its_claim() -> None:
    """Project immutable exclusions and publish success with the claimed token."""
    from textify.jobs.processor import TranscriptionJobProcessor

    dispatch = _dispatch()
    repository = RecordingExecutionRepository(_claimed_job(dispatch))
    executor = ResultExecutor()

    await TranscriptionJobProcessor(repository, executor).process(dispatch)

    assert executor.calls == 1
    assert repository.successes == [
        (
            dispatch,
            {
                "source": {
                    "platform": Platform.YOUTUBE,
                    "video_id": "dQw4w9WgXcQ",
                    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "title": "Title",
                    "description": "Description",
                    "channel": "Creator",
                    "duration_seconds": 12,
                },
                "transcript": {
                    "method": TranscriptMethod.FASTER_WHISPER,
                    "language": "en",
                    "text": "One",
                    "segments": ({"start": 0.0, "end": 1.0, "text": "One"},),
                },
            },
        )
    ]
    assert repository.failures == []
    assert repository.cancelled_dispatches == []


@pytest.mark.asyncio
async def test_processor_publishes_safe_failure_for_provider_error() -> None:
    """Persist only the stable error code and message for expected failures."""
    from textify.jobs.processor import TranscriptionJobProcessor

    dispatch = _dispatch()
    repository = RecordingExecutionRepository(_claimed_job(dispatch))
    executor = FailingExecutor()

    await TranscriptionJobProcessor(repository, executor).process(dispatch)

    assert executor.calls == 1
    assert repository.successes == []
    assert repository.failures == [
        (
            dispatch,
            "transcription_failed",
            "The source could not be transcribed.",
        )
    ]
    assert repository.cancelled_dispatches == []


@pytest.mark.asyncio
async def test_processor_publishes_cancellation_when_terminal_write_loses_race() -> (
    None
):
    """Wait for executor cleanup before finalizing a concurrent cancellation."""
    from textify.jobs.processor import TranscriptionJobProcessor

    dispatch = _dispatch()
    repository = RecordingExecutionRepository(_claimed_job(dispatch))
    repository.success_published = False
    executor = ResultExecutor()

    await TranscriptionJobProcessor(repository, executor).process(dispatch)

    assert repository.successes[0][0] == dispatch
    assert repository.cancelled_dispatches == [dispatch]
