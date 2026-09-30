"""Private Job Dispatch publication through Celery."""

from __future__ import annotations

import asyncio
from typing import Protocol

from celery import Celery
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from textify.jobs.celery_app import TRANSCRIPTION_QUEUE, TRANSCRIPTION_TASK
from textify.jobs.types import JobDispatch


class JobDispatchUnavailableError(RuntimeError):
    """Indicate that the delivery broker could not accept a Job Dispatch."""


class JobDispatchPublisher(Protocol):
    """Publish private Job Dispatches without owning job lifecycle state."""

    async def publish(self, dispatch: JobDispatch) -> None:
        """Publish one private Execution Attempt notification.

        Args:
            dispatch: Exact internal job and Execution Attempt identity.

        Raises:
            JobDispatchUnavailableError: If the delivery broker is unavailable.
        """
        ...


class CeleryJobDispatchPublisher:
    """Publish exact Job Dispatch payloads to the private Celery queue."""

    def __init__(self, celery_app: Celery) -> None:
        """Initialize publication using one configured Celery application.

        Args:
            celery_app: Private transcription Celery delivery application.
        """
        self._celery_app = celery_app

    async def publish(self, dispatch: JobDispatch) -> None:
        """Publish one dispatch without blocking the reconciler event loop.

        Args:
            dispatch: Exact internal job and Execution Attempt identity.

        Raises:
            JobDispatchUnavailableError: If the delivery broker is unavailable.
        """
        try:
            await asyncio.to_thread(self._send_task, dispatch)
        except (OSError, OperationalError, RedisError) as exc:
            raise JobDispatchUnavailableError() from exc

    def _send_task(self, dispatch: JobDispatch) -> None:
        """Send one JSON-serializable private task message.

        Args:
            dispatch: Exact internal job and Execution Attempt identity.
        """
        self._celery_app.send_task(
            TRANSCRIPTION_TASK,
            args=(dispatch.to_payload(),),
            queue=TRANSCRIPTION_QUEUE,
            argsrepr="(<private-job-dispatch>,)",
        )
