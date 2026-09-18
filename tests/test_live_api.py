"""Opt-in checks against an operator-configured running Textify API."""

import asyncio
import math
import os

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

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_BASE_URL or not LIVE_SOURCES,
        reason=(
            "TEXTIFY_LIVE_BASE_URL and at least one TEXTIFY_LIVE_<PLATFORM>_URL "
            "are required for live API checks."
        ),
    ),
]


async def _poll_until_succeeded(
    client: AsyncClient,
    location: str,
) -> Response:
    """Poll a job capability until it reaches the successful terminal state.

    Args:
        client: Connected live API client.
        location: Relative job capability URL returned at submission.

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
        payload = response.json()
        if payload["status"] in {"queued", "processing"}:
            retry_after = response.headers["Retry-After"]
            assert retry_after.isdecimal()
            await asyncio.sleep(int(retry_after))
            continue

        assert payload["status"] == "finished"
        assert payload["outcome"] == "succeeded"
        assert "Retry-After" not in response.headers
        return response

    raise AssertionError(
        "Transcription Job did not finish within the live polling limit."
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
            assert submission.json()["links"]["self"] == location

            response = await _poll_until_succeeded(client, location)
            result = response.json()["result"]
            source = result["source"]
            transcript = result["transcript"]
            segments = transcript["segments"]

            assert source["platform"] == expected_platform
            assert isinstance(source["duration_seconds"], int)
            assert not isinstance(source["duration_seconds"], bool)
            assert 0 < source["duration_seconds"] <= 1800
            assert transcript["method"] in {"youtube_captions", "faster_whisper"}
            assert transcript["text"] == " ".join(
                segment["text"] for segment in segments
            )

            previous_start = 0.0
            for segment in segments:
                start = segment["start"]
                end = segment["end"]
                assert isinstance(start, (int, float))
                assert not isinstance(start, bool)
                assert isinstance(end, (int, float))
                assert not isinstance(end, bool)
                assert math.isfinite(start)
                assert math.isfinite(end)
                assert start >= previous_start >= 0.0
                assert end >= start
                previous_start = start
