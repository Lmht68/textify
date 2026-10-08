"""Opt-in checks against an operator-configured running Textify API."""

import asyncio
import math
import os
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import UUID

import pytest
from httpx import AsyncClient, Response, Timeout

LIVE_BASE_URL = os.environ.get("TEXTIFY_LIVE_BASE_URL")
LIVE_SOURCES = tuple(
    (platform, source_url)
    for platform, environment_name in (
        ("youtube", "TEXTIFY_LIVE_YOUTUBE_URL"),
        ("facebook", "TEXTIFY_LIVE_FACEBOOK_URL"),
        ("instagram", "TEXTIFY_LIVE_INSTAGRAM_URL"),
        ("tiktok", "TEXTIFY_LIVE_TIKTOK_URL"),
        ("x", "TEXTIFY_LIVE_X_URL"),
    )
    if (source_url := os.environ.get(environment_name))
)
LIVE_YOUTUBE_URL_A = os.environ.get("TEXTIFY_LIVE_YOUTUBE_URL_A")
LIVE_YOUTUBE_URL_B = os.environ.get("TEXTIFY_LIVE_YOUTUBE_URL_B")

pytestmark = [pytest.mark.live]


def _mapping(value: object) -> dict[str, object]:
    """Return a JSON object after asserting its structural contract."""
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _parse_utc_timestamp(value: object) -> datetime:
    """Return one required RFC 3339 UTC timestamp."""
    assert isinstance(value, str)
    parsed_timestamp = datetime.fromisoformat(value)
    assert parsed_timestamp.tzinfo is not None
    assert parsed_timestamp.utcoffset() == UTC.utcoffset(parsed_timestamp)
    return parsed_timestamp


def _assert_canonical_job_id(value: object) -> str:
    """Assert that a public job identifier is a canonical lowercase UUIDv4."""
    assert isinstance(value, str)
    parsed_job_id = UUID(value)
    assert parsed_job_id.version == 4
    assert str(parsed_job_id) == value
    return value


def _assert_relative_capability_link(link: object, expected_path: str) -> None:
    """Assert that a capability link is same-origin, relative, and canonical."""
    assert isinstance(link, str)
    parsed_link = urlsplit(link)
    assert parsed_link.scheme == ""
    assert parsed_link.netloc == ""
    assert parsed_link.query == ""
    assert parsed_link.fragment == ""
    assert parsed_link.path == expected_path


def _assert_active_job(
    payload: dict[str, object],
    location: str,
    expected_job_id: str | None = None,
) -> str:
    """Assert the queued or processing representation and its active links."""
    job_id = _assert_canonical_job_id(payload["id"])
    if expected_job_id is not None:
        assert job_id == expected_job_id
    _assert_relative_capability_link(location, f"/api/transcription-jobs/{job_id}")
    links = _mapping(payload["links"])
    assert set(links) == {"self", "cancel"}
    assert links["self"] == location
    _assert_relative_capability_link(links["self"], location)
    _assert_relative_capability_link(links["cancel"], f"{location}/cancellation")
    submitted_at = _parse_utc_timestamp(payload["submitted_at"])

    status = payload["status"]
    assert status in {"queued", "processing"}
    if status == "queued":
        assert set(payload) == {"id", "status", "submitted_at", "links"}
        return job_id

    assert set(payload) == {
        "id",
        "status",
        "submitted_at",
        "started_at",
        "cancellation_requested",
        "links",
    }
    started_at = _parse_utc_timestamp(payload["started_at"])
    assert submitted_at <= started_at
    assert isinstance(payload["cancellation_requested"], bool)
    return job_id


def _assert_succeeded_job(
    payload: dict[str, object],
    location: str,
    expected_job_id: str,
) -> dict[str, object]:
    """Assert a terminal successful representation and return its result."""
    assert set(payload) == {
        "id",
        "status",
        "outcome",
        "submitted_at",
        "started_at",
        "finished_at",
        "result",
        "links",
    }
    assert _assert_canonical_job_id(payload["id"]) == expected_job_id
    assert payload["status"] == "finished"
    assert payload["outcome"] == "succeeded"
    submitted_at = _parse_utc_timestamp(payload["submitted_at"])
    started_at = _parse_utc_timestamp(payload["started_at"])
    finished_at = _parse_utc_timestamp(payload["finished_at"])
    assert submitted_at <= started_at <= finished_at
    links = _mapping(payload["links"])
    assert set(links) == {"self"}
    assert links["self"] == location
    _assert_relative_capability_link(links["self"], location)
    return _mapping(payload["result"])


def _assert_transcription_response(
    result: dict[str, object],
    expected_platform: str,
) -> None:
    """Assert the full successful projected transcript response contract."""
    assert set(result) == {"source", "transcript"}
    source = _mapping(result["source"])
    transcript = _mapping(result["transcript"])
    assert set(source) == {
        "platform",
        "video_id",
        "url",
        "title",
        "description",
        "channel",
        "duration_seconds",
    }
    assert set(transcript) == {"method", "language", "text", "segments"}

    assert source["platform"] == expected_platform
    for field_name in ("title", "description", "channel"):
        assert isinstance(source[field_name], str)
    assert isinstance(source["video_id"], str)
    assert source["video_id"]
    assert isinstance(source["url"], str)
    canonical_url = urlsplit(source["url"])
    assert canonical_url.scheme == "https"
    assert canonical_url.netloc
    assert isinstance(source["duration_seconds"], int)
    assert not isinstance(source["duration_seconds"], bool)
    assert 0 < source["duration_seconds"] <= 1800

    assert transcript["method"] in {"youtube_captions", "faster_whisper"}
    assert isinstance(transcript["language"], str)
    assert transcript["language"]
    assert isinstance(transcript["text"], str)
    segments = transcript["segments"]
    assert isinstance(segments, list)
    segment_texts: list[str] = []

    previous_start = 0.0
    for segment in segments:
        normalized_segment = _mapping(segment)
        assert set(normalized_segment) == {"start", "end", "text"}
        start = normalized_segment["start"]
        end = normalized_segment["end"]
        text = normalized_segment["text"]
        assert isinstance(start, (int, float))
        assert not isinstance(start, bool)
        assert isinstance(end, (int, float))
        assert not isinstance(end, bool)
        assert isinstance(text, str)
        assert text == " ".join(text.split())
        assert text
        segment_texts.append(text)
        assert math.isfinite(start)
        assert math.isfinite(end)
        assert start >= previous_start >= 0.0
        assert end >= start
        previous_start = start
    assert transcript["text"] == " ".join(segment_texts)


async def _poll_until_succeeded(
    client: AsyncClient,
    location: str,
    job_id: str,
) -> Response:
    """Poll a job capability until it reaches the successful terminal state.

    Args:
        client: Connected live API client.
        location: Relative job capability URL returned at submission.
        job_id: Canonical public capability identifier returned at submission.

    Returns:
        The terminal successful polling response.

    Raises:
        AssertionError: If the API returns an invalid lifecycle representation.
    """
    for _ in range(1250):
        response = await client.get(location)
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID")
        assert response.headers["Cache-Control"] == "no-store"
        payload = _mapping(response.json())
        if payload["status"] in {"queued", "processing"}:
            _assert_active_job(payload, location, job_id)
            retry_after = response.headers["Retry-After"]
            assert retry_after.isdecimal()
            assert int(retry_after) > 0
            await asyncio.sleep(int(retry_after))
            continue

        result = _assert_succeeded_job(payload, location, job_id)
        assert "Retry-After" not in response.headers
        assert result
        return response

    raise AssertionError(
        "Transcription Job did not finish within the live polling limit."
    )


@pytest.mark.skipif(
    not LIVE_BASE_URL or (LIVE_YOUTUBE_URL_A is None and LIVE_YOUTUBE_URL_B is None),
    reason=(
        "TEXTIFY_LIVE_BASE_URL and both TEXTIFY_LIVE_YOUTUBE_URL_A and "
        "TEXTIFY_LIVE_YOUTUBE_URL_B are required for the captionless live check."
    ),
)
async def test_two_captionless_youtube_videos_complete_through_native_worker() -> None:
    """Run two distinct captionless YouTube jobs concurrently through native work."""
    assert LIVE_YOUTUBE_URL_A is not None, (
        "TEXTIFY_LIVE_YOUTUBE_URL_A and TEXTIFY_LIVE_YOUTUBE_URL_B must both be set."
    )
    assert LIVE_YOUTUBE_URL_B is not None, (
        "TEXTIFY_LIVE_YOUTUBE_URL_A and TEXTIFY_LIVE_YOUTUBE_URL_B must both be set."
    )
    assert LIVE_YOUTUBE_URL_A != LIVE_YOUTUBE_URL_B, (
        "TEXTIFY_LIVE_YOUTUBE_URL_A and TEXTIFY_LIVE_YOUTUBE_URL_B must differ."
    )
    assert LIVE_BASE_URL is not None

    async with AsyncClient(
        base_url=LIVE_BASE_URL,
        timeout=Timeout(2500.0),
    ) as client:
        health_response = await client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}
        assert health_response.headers.get("X-Request-ID")

        submissions = await asyncio.gather(
            *(
                client.post("/api/transcription-jobs", json={"url": source_url})
                for source_url in (LIVE_YOUTUBE_URL_A, LIVE_YOUTUBE_URL_B)
            )
        )
        locations: list[str] = []
        job_ids: list[str] = []
        submission_delays: list[int] = []
        for submission in submissions:
            assert submission.status_code == 202
            assert submission.headers.get("X-Request-ID")
            assert submission.headers["Cache-Control"] == "no-store"
            retry_after = submission.headers["Retry-After"]
            assert retry_after.isdecimal()
            assert int(retry_after) > 0
            location = submission.headers["Location"]
            job_id = _assert_active_job(_mapping(submission.json()), location)
            locations.append(location)
            job_ids.append(job_id)
            submission_delays.append(int(retry_after))

        await asyncio.gather(*(asyncio.sleep(delay) for delay in submission_delays))
        completed = await asyncio.gather(
            *(
                _poll_until_succeeded(client, location, job_id)
                for location, job_id in zip(locations, job_ids, strict=True)
            )
        )

        for response, location, job_id in zip(
            completed,
            locations,
            job_ids,
            strict=True,
        ):
            result = _assert_succeeded_job(_mapping(response.json()), location, job_id)
            _assert_transcription_response(result, "youtube")
            transcript = _mapping(result["transcript"])
            assert transcript["method"] == "faster_whisper"


@pytest.mark.skipif(
    not LIVE_BASE_URL or not LIVE_SOURCES,
    reason=(
        "TEXTIFY_LIVE_BASE_URL and at least one TEXTIFY_LIVE_<PLATFORM>_URL "
        "are required for live API checks."
    ),
)
async def test_configured_public_videos_match_transcription_job_contract() -> None:
    """Exercise each configured public video through submission and polling."""
    assert LIVE_BASE_URL is not None
    async with AsyncClient(
        base_url=LIVE_BASE_URL,
        timeout=Timeout(2500.0),
    ) as client:
        health_response = await client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}
        assert health_response.headers.get("X-Request-ID")

        for expected_platform, source_url in LIVE_SOURCES:
            submission = await client.post(
                "/api/transcription-jobs",
                json={"url": source_url},
            )
            assert submission.status_code == 202
            assert submission.headers.get("X-Request-ID")
            assert submission.headers["Cache-Control"] == "no-store"
            assert submission.headers["Retry-After"] == "2"
            location = submission.headers["Location"]
            submission_payload = _mapping(submission.json())
            job_id = _assert_active_job(submission_payload, location)
            await asyncio.sleep(int(submission.headers["Retry-After"]))

            response = await _poll_until_succeeded(client, location, job_id)
            result = _assert_succeeded_job(_mapping(response.json()), location, job_id)
            _assert_transcription_response(result, expected_platform)
