"""Consumer-visible OpenAPI contract tests."""

from httpx import ASGITransport, AsyncClient

from textify.main import app

EXPECTED_ERROR_CODES = {
    "invalid_url",
    "unsupported_platform",
    "invalid_request",
    "unsupported_content",
    "video_too_long",
    "invalid_media_duration",
    "unsupported_media",
    "no_usable_transcript",
    "metadata_retrieval_failed",
    "audio_download_failed",
    "transcription_failed",
    "transcription_capacity_exceeded",
    "metadata_timeout",
    "audio_download_timeout",
    "transcription_timeout",
    "internal_error",
}
EXPECTED_PLATFORMS = {"youtube", "instagram", "facebook", "tiktok", "x"}
EXPECTED_TRANSCRIPT_METHODS = {"youtube_captions", "faster_whisper"}
DOCUMENTED_ERROR_STATUSES = {"400", "422", "500", "502", "503", "504"}


async def test_openapi_describes_public_transcript_contract() -> None:
    """Publish the typed request, success, health, and error API contracts."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert set(document["paths"]) == {"/health", "/api/transcripts"}
    assert set(document["paths"]["/health"]) == {"get"}
    assert set(document["paths"]["/api/transcripts"]) == {"post"}

    components = document["components"]["schemas"]
    health_operation = document["paths"]["/health"]["get"]
    transcript_operation = document["paths"]["/api/transcripts"]["post"]

    assert health_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/HealthResponse"}
    assert components["HealthResponse"]["examples"] == [{"status": "ok"}]

    request_schema = transcript_operation["requestBody"]["content"]["application/json"][
        "schema"
    ]
    success_schema = transcript_operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert request_schema == {"$ref": "#/components/schemas/TranscriptRequest"}
    assert success_schema == {"$ref": "#/components/schemas/TranscriptionResponse"}
    assert components["TranscriptRequest"]["examples"]
    assert components["TranscriptionResponse"]["examples"]

    assert set(components["Platform"]["enum"]) == EXPECTED_PLATFORMS
    assert set(components["TranscriptMethod"]["enum"]) == EXPECTED_TRANSCRIPT_METHODS
    assert set(components["ErrorDetail"]["properties"]["code"]["enum"]) == (
        EXPECTED_ERROR_CODES
    )
    assert components["ErrorResponse"]["examples"]

    for status_code in DOCUMENTED_ERROR_STATUSES:
        error_response = transcript_operation["responses"][status_code]
        content = error_response["content"]["application/json"]
        assert content["schema"] == {"$ref": "#/components/schemas/ErrorResponse"}
        assert content["example"]
