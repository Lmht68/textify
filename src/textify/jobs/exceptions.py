"""Safe domain errors specific to durable Transcription Jobs."""

from textify.errors import TextifyError


class TranscriptionCapacityExceededError(TextifyError):
    """Indicate exhausted durable Transcription Job admission capacity."""

    status_code = 503
    code = "transcription_capacity_exceeded"
    message = "Transcription capacity is currently unavailable."


class JobNotFoundError(TextifyError):
    """Indicate that a Transcription Job bearer capability is unavailable."""

    status_code = 404
    code = "job_not_found"
    message = "The transcription job was not found."


class JobStoreUnavailableError(TextifyError):
    """Indicate that durable Transcription Job storage is unavailable."""

    status_code = 503
    code = "job_store_unavailable"
    message = "Transcription job storage is currently unavailable."


class QueueTimeoutError(TextifyError):
    """Indicate a job expiring before a worker claims it."""

    code = "queue_timeout"
    status_code = 504
    message = (
        "The transcription job did not begin processing before its queue deadline."
    )
