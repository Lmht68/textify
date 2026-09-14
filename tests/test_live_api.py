"""Opt-in checks against an operator-configured running Textify API."""

import math
import os

import pytest
from httpx import AsyncClient, Timeout

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


async def test_configured_public_videos_match_transcript_contract() -> None:
    """Exercise each configured public video through the running API."""
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
            response = await client.post("/api/transcripts", json={"url": source_url})
            assert response.status_code == 200
            assert response.headers.get("X-Request-ID")

            payload = response.json()
            source = payload["source"]
            transcript = payload["transcript"]
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
