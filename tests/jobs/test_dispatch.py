"""Tests for private Job Dispatch serialization and redaction."""

import logging
from uuid import uuid1, uuid4

import pytest

from textify.jobs.types import JobDispatch


def test_job_dispatch_serializes_and_parses_exact_payload() -> None:
    """Round-trip the only JSON payload permitted on the broker."""
    dispatch = JobDispatch(internal_job_id=123, execution_attempt_token=uuid4())

    payload = dispatch.to_payload()

    assert payload == {
        "internal_job_id": 123,
        "execution_attempt_token": str(dispatch.execution_attempt_token),
    }
    assert JobDispatch.from_payload(payload) == dispatch


@pytest.mark.parametrize(
    "payload",
    (
        None,
        {},
        {"internal_job_id": 1},
        {"execution_attempt_token": str(uuid4())},
        {"internal_job_id": True, "execution_attempt_token": str(uuid4())},
        {"internal_job_id": 0, "execution_attempt_token": str(uuid4())},
        {"internal_job_id": -1, "execution_attempt_token": str(uuid4())},
        {"internal_job_id": 1.0, "execution_attempt_token": str(uuid4())},
        {"internal_job_id": "1", "execution_attempt_token": str(uuid4())},
        {"internal_job_id": 1, "execution_attempt_token": uuid4()},
        {"internal_job_id": 1, "execution_attempt_token": str(uuid1())},
        {
            "internal_job_id": 1,
            "execution_attempt_token": str(uuid4()).upper(),
        },
        {
            "internal_job_id": 1,
            "execution_attempt_token": str(uuid4()),
            "extra": "rejected",
        },
    ),
)
def test_job_dispatch_rejects_noncanonical_or_incomplete_payloads(
    payload: object,
) -> None:
    """Reject data that cannot be one exact private broker dispatch."""
    assert JobDispatch.from_payload(payload) is None


def test_job_dispatch_representation_redacts_execution_attempt_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep the private token out of object representation and rendered logs."""
    dispatch = JobDispatch(internal_job_id=123, execution_attempt_token=uuid4())

    with caplog.at_level(logging.INFO):
        logging.getLogger("test.dispatch").info("dispatch=%r", dispatch)

    assert str(dispatch.execution_attempt_token) not in repr(dispatch)
    assert str(dispatch.execution_attempt_token) not in caplog.text
