"""Tests for the private transcription Celery application."""

from textify.jobs.celery_app import (
    TRANSCRIPTION_QUEUE,
    TRANSCRIPTION_TASK,
    create_celery_app,
)
from textify.jobs.config import JobDispatchConfig


def test_celery_app_uses_private_redis_delivery_configuration() -> None:
    """Configure only JSON dispatch delivery and no result backend."""
    config = JobDispatchConfig(
        broker_url="redis://127.0.0.1:6379/15",
        _env_file=None,  # type: ignore[call-arg]
    )

    app = create_celery_app(config)

    assert app.conf.broker_url == "redis://127.0.0.1:6379/15"
    assert app.conf.result_backend is None
    assert app.conf.task_serializer == "json"
    assert app.conf.accept_content == ("json",)
    assert app.conf.task_acks_late is False
    assert app.conf.task_reject_on_worker_lost is False
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_ignore_result is True
    assert app.conf.task_store_errors_even_if_ignored is False
    assert app.conf.task_always_eager is False
    assert app.conf.task_routes == {TRANSCRIPTION_TASK: {"queue": TRANSCRIPTION_QUEUE}}
    assert app.conf.task_default_queue == TRANSCRIPTION_QUEUE
    assert tuple(queue.name for queue in app.conf.task_queues) == (TRANSCRIPTION_QUEUE,)

    app.amqp.queues.select([TRANSCRIPTION_QUEUE])

    assert tuple(app.amqp.queues.consume_from) == (TRANSCRIPTION_QUEUE,)
