"""Private Celery application configuration for transcription dispatches."""

from celery import Celery
from kombu import Queue

from textify.jobs.config import JobDispatchConfig

TRANSCRIPTION_QUEUE = "textify-transcription"
TRANSCRIPTION_TASK = "textify.jobs.process_transcription"


def create_celery_app(config: JobDispatchConfig) -> Celery:
    """Create the Celery application used only for Job Dispatch delivery.

    Args:
        config: Validated Redis delivery and worker configuration.

    Returns:
        A Celery application configured for private transcription dispatches.
    """
    celery_app = Celery("textify", broker=str(config.broker_url))
    celery_app.conf.update(
        result_backend=None,
        task_serializer="json",
        accept_content=("json",),
        task_acks_late=False,
        task_reject_on_worker_lost=False,
        worker_prefetch_multiplier=1,
        task_ignore_result=True,
        task_store_errors_even_if_ignored=False,
        task_always_eager=False,
        worker_pool="threads",
        task_default_queue=TRANSCRIPTION_QUEUE,
        task_queues=(Queue(TRANSCRIPTION_QUEUE),),
        task_routes={TRANSCRIPTION_TASK: {"queue": TRANSCRIPTION_QUEUE}},
    )
    return celery_app
