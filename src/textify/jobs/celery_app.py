"""Private Celery application configuration for transcription dispatches."""

from celery import Celery
from celery.signals import setup_logging
from kombu import Queue

from textify.jobs.config import JobDispatchConfig

TRANSCRIPTION_QUEUE = "textify-transcription"
TRANSCRIPTION_TASK = "textify.jobs.process_transcription"


def _preserve_textify_logging(**_signal_kwargs: object) -> None:
    """Prevent Celery from replacing Textify's process-wide safe logging setup."""


setup_logging.connect(_preserve_textify_logging, weak=False)


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
        task_time_limit=None,
        task_soft_time_limit=None,
        worker_hijack_root_logger=False,
        worker_send_task_events=False,
        task_send_sent_event=False,
        worker_pool="threads",
        task_default_queue=TRANSCRIPTION_QUEUE,
        task_queues=(Queue(TRANSCRIPTION_QUEUE),),
        task_routes={TRANSCRIPTION_TASK: {"queue": TRANSCRIPTION_QUEUE}},
    )
    return celery_app
