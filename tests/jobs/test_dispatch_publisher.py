"""Tests for private Celery Job Dispatch publication."""

from uuid import uuid4

import pytest

from textify.jobs.types import JobDispatch


class CapturingCeleryApp:
    """Record private Celery publication without connecting to a broker."""

    def __init__(self) -> None:
        """Initialize recorded sends and optional delivery failure."""
        self.calls: list[tuple[str, tuple[object, ...], str, str]] = []
        self.delivery_error: OSError | None = None

    def send_task(
        self,
        task_name: str,
        args: tuple[object, ...],
        *,
        queue: str,
        argsrepr: str,
    ) -> None:
        """Record a private task send or raise the configured broker failure."""
        if self.delivery_error is not None:
            raise self.delivery_error
        self.calls.append((task_name, args, queue, argsrepr))


class RecordingBrokerReadiness:
    """Return one fixed broker readiness result without network I/O."""

    def __init__(self, ready: bool) -> None:
        """Initialize the broker probe result."""
        self._ready = ready
        self.calls = 0

    async def is_ready(self) -> bool:
        """Return the configured broker readiness result."""
        self.calls += 1
        return self._ready


@pytest.mark.asyncio
async def test_publisher_sends_exact_private_dispatch_payload() -> None:
    """Publish only the Execution Attempt identity to the dedicated task queue."""
    from textify.jobs.dispatch import CeleryJobDispatchPublisher

    celery_app = CapturingCeleryApp()
    dispatch = JobDispatch(41, uuid4())

    await CeleryJobDispatchPublisher(celery_app).publish(dispatch)

    assert celery_app.calls == [
        (
            "textify.jobs.process_transcription",
            (dispatch.to_payload(),),
            "textify-transcription",
            "(<private-job-dispatch>,)",
        )
    ]
    assert str(dispatch.execution_attempt_token) not in celery_app.calls[0][3]


@pytest.mark.asyncio
async def test_publisher_translates_broker_delivery_failure() -> None:
    """Expose an unavailable broker without leaking dispatch values in the error."""
    from textify.jobs.dispatch import (
        CeleryJobDispatchPublisher,
        JobDispatchUnavailableError,
    )

    celery_app = CapturingCeleryApp()
    celery_app.delivery_error = OSError("connection refused")

    with pytest.raises(JobDispatchUnavailableError):
        await CeleryJobDispatchPublisher(celery_app).publish(JobDispatch(41, uuid4()))


@pytest.mark.asyncio
async def test_publisher_initialization_requires_a_ready_shared_broker_probe() -> None:
    """Reject reconciler startup before it can publish to the configured broker."""
    from textify.jobs.dispatch import (
        CeleryJobDispatchPublisher,
        JobDispatchUnavailableError,
    )

    celery_app = CapturingCeleryApp()
    publisher = CeleryJobDispatchPublisher(celery_app)
    ready_broker = RecordingBrokerReadiness(True)
    unavailable_broker = RecordingBrokerReadiness(False)

    await publisher.initialize(ready_broker)

    with pytest.raises(JobDispatchUnavailableError):
        await publisher.initialize(unavailable_broker)

    assert ready_broker.calls == 1
    assert unavailable_broker.calls == 1
    assert celery_app.calls == []
