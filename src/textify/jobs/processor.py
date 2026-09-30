"""Process one claimed Transcription Job Execution Attempt."""

from __future__ import annotations

import logging

from textify.errors import InternalError
from textify.jobs.contracts import TranscriptionJobExecutionRepository
from textify.jobs.types import ClaimedTranscriptionJob, JobDispatch
from textify.transcription import inspection
from textify.transcription.exceptions import (
    TranscriptionCancellationRequestedError,
    TranscriptionError,
)
from textify.transcription.schemas import build_transcription_response
from textify.transcription.service import (
    TranscriptionExecutionControl,
    TranscriptionExecutor,
)

logger = logging.getLogger(__name__)


class TranscriptionJobProcessor:
    """Claim and process one delivered Execution Attempt through PostgreSQL."""

    def __init__(
        self,
        repository: TranscriptionJobExecutionRepository,
        executor: TranscriptionExecutor,
    ) -> None:
        """Initialize one processor with durable state and provider execution boundaries.

        Args:
            repository: Execution Attempt claim and terminal publication boundary.
            executor: Process-lifetime provider execution boundary.
        """
        self._repository = repository
        self._executor = executor

    async def process(self, dispatch: JobDispatch) -> None:
        """Claim and process one dispatch without exposing private inputs to Celery.

        Args:
            dispatch: Validated private Execution Attempt notification.

        Raises:
            TranscriptionJobStoreUnavailableError: If a durable claim or terminal write
                cannot complete safely.
        """
        claimed_job = await self._repository.claim_dispatch(dispatch)
        if claimed_job is None:
            return

        control = TranscriptionExecutionControl()
        try:
            submitted = inspection.classify_submitted_url(claimed_job.submitted_url)
            result = await self._executor.execute(
                submitted,
                control=control,
                include_segments="transcript.segments" not in claimed_job.exclusions,
            )
        except TranscriptionCancellationRequestedError:
            await self._publish_cancelled_after_cleanup(claimed_job, control)
        except TranscriptionError as error:
            if control.cancellation_requested.is_set():
                await self._publish_cancelled_after_cleanup(claimed_job, control)
                return
            logger.error(
                "transcription job processor failed", extra={"code": error.code}
            )
            await self._publish_failure_or_cancelled(
                claimed_job,
                control,
                error.code,
                error.message,
            )
        except Exception:  # noqa: BLE001
            if control.cancellation_requested.is_set():
                await self._publish_cancelled_after_cleanup(claimed_job, control)
                return
            logger.error(
                "transcription job processor failed",
                extra={"code": InternalError.code},
            )
            await self._publish_failure_or_cancelled(
                claimed_job,
                control,
                InternalError.code,
                InternalError.message,
            )
        else:
            if control.cancellation_requested.is_set():
                await self._publish_cancelled_after_cleanup(claimed_job, control)
                return
            try:
                projected_result = build_transcription_response(
                    result,
                    frozenset(claimed_job.exclusions),
                )
            except Exception:  # noqa: BLE001
                if control.cancellation_requested.is_set():
                    await self._publish_cancelled_after_cleanup(claimed_job, control)
                    return
                logger.error(
                    "transcription job processor failed",
                    extra={"code": InternalError.code},
                )
                await self._publish_failure_or_cancelled(
                    claimed_job,
                    control,
                    InternalError.code,
                    InternalError.message,
                )
                return
            published = await self._repository.publish_success(
                claimed_job.dispatch,
                projected_result,
            )
            if not published:
                await self._publish_cancelled_after_cleanup(claimed_job, control)

    async def _publish_failure_or_cancelled(
        self,
        claimed_job: ClaimedTranscriptionJob,
        control: TranscriptionExecutionControl,
        error_code: str,
        error_message: str,
    ) -> None:
        """Publish failure unless a concurrent cancellation owns the terminal outcome.

        Args:
            claimed_job: Private inputs and dispatch identity from the durable claim.
            control: Provider cancellation and cleanup ownership for this execution.
            error_code: Stable safe public failure code.
            error_message: Stable safe public failure message.
        """
        if control.cancellation_requested.is_set():
            await self._publish_cancelled_after_cleanup(claimed_job, control)
            return
        published = await self._repository.publish_failure(
            claimed_job.dispatch,
            error_code,
            error_message,
        )
        if not published:
            await self._publish_cancelled_after_cleanup(claimed_job, control)

    async def _publish_cancelled_after_cleanup(
        self,
        claimed_job: ClaimedTranscriptionJob,
        control: TranscriptionExecutionControl,
    ) -> None:
        """Publish accepted cancellation only after provider ownership is released.

        Args:
            claimed_job: Private inputs and dispatch identity from the durable claim.
            control: Provider cancellation and cleanup ownership for this execution.
        """
        await control.cleanup_complete.wait()
        await self._repository.publish_cancelled(claimed_job.dispatch)
