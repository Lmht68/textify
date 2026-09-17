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
    "job_not_found",
    "job_store_unavailable",
    "metadata_timeout",
    "audio_download_timeout",
    "transcription_timeout",
    "internal_error",
}
EXPECTED_PLATFORMS = {"youtube", "instagram", "facebook", "tiktok", "x"}
EXPECTED_TRANSCRIPT_METHODS = {"youtube_captions", "faster_whisper"}
DOCUMENTED_ERROR_STATUSES = {"400", "422", "500", "502", "503", "504"}
EXPECTED_RESPONSE_FIELD_PATHS = {
    "source.platform",
    "source.video_id",
    "source.url",
    "source.title",
    "source.description",
    "source.channel",
    "source.duration_seconds",
    "transcript.method",
    "transcript.language",
    "transcript.text",
    "transcript.segments",
}


async def test_openapi_describes_public_transcript_contract() -> None:
    """Publish the typed request, success, health, and error API contracts."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")
    assert response.status_code == 200
    document = response.json()

    assert set(document["paths"]) == {
        "/health",
        "/api/transcripts",
        "/api/transcription-jobs",
        "/api/transcription-jobs/{job_id}",
    }
    assert set(document["paths"]["/health"]) == {"get"}
    assert set(document["paths"]["/api/transcripts"]) == {"post"}
    assert set(document["paths"]["/api/transcription-jobs"]) == {"post"}
    assert set(document["paths"]["/api/transcription-jobs/{job_id}"]) == {"get"}

    components = document["components"]["schemas"]
    health_operation = document["paths"]["/health"]["get"]
    transcript_operation = document["paths"]["/api/transcripts"]["post"]
    queued_submission_operation = document["paths"]["/api/transcription-jobs"]["post"]
    queued_status_operation = document["paths"]["/api/transcription-jobs/{job_id}"][
        "get"
    ]

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
    request_component = components["TranscriptRequest"]
    assert "exclude" not in request_component["required"]
    assert (
        set(request_component["properties"]["exclude"]["items"]["enum"])
        == EXPECTED_RESPONSE_FIELD_PATHS
    )

    response_component = components["TranscriptionResponse"]
    source_component = components["SourceResponse"]
    transcript_component = components["TranscriptResponse"]
    segment_component = components["SegmentResponse"]
    assert response_component["required"] == ["source", "transcript"]
    for component in (source_component, transcript_component):
        assert component["minProperties"] == 1
        assert "required" not in component
        assert all(
            property_schema.get("type") != "null"
            for property_schema in component["properties"].values()
        )
    assert segment_component["required"] == ["start", "end", "text"]
    assert segment_component["properties"]["text"]["minLength"] == 1
    full_example, projected_example = response_component["examples"]
    assert "segments" in full_example["transcript"]
    assert "segments" not in projected_example["transcript"]
    assert "description" not in projected_example["source"]

    assert set(components["Platform"]["enum"]) == EXPECTED_PLATFORMS
    assert set(components["TranscriptMethod"]["enum"]) == EXPECTED_TRANSCRIPT_METHODS
    assert set(components["ErrorDetail"]["properties"]["code"]["enum"]) == (
        EXPECTED_ERROR_CODES
    )
    assert components["ErrorResponse"]["examples"]

    queued_request_schema = queued_submission_operation["requestBody"]["content"][
        "application/json"
    ]["schema"]
    queued_submission_schema = queued_submission_operation["responses"]["202"][
        "content"
    ]["application/json"]["schema"]
    queued_status_schema = queued_status_operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert queued_request_schema == {"$ref": "#/components/schemas/TranscriptRequest"}
    assert queued_submission_schema == {
        "$ref": "#/components/schemas/QueuedTranscriptionJobResponse"
    }
    assert queued_status_schema == queued_submission_schema

    queued_component = components["QueuedTranscriptionJobResponse"]
    assert queued_component["required"] == ["id", "status", "submitted_at", "links"]
    assert queued_component["properties"]["id"]["format"] == "uuid4"
    assert queued_component["properties"]["status"]["const"] == "queued"
    assert queued_component["properties"]["submitted_at"]["format"] == "date-time"
    assert queued_component["properties"]["links"] == {
        "$ref": "#/components/schemas/TranscriptionJobLinks"
    }
    links_component = components["TranscriptionJobLinks"]
    assert links_component["required"] == ["self", "cancel"]

    for header_name in ("Location", "Retry-After", "Cache-Control"):
        assert header_name in queued_submission_operation["responses"]["202"]["headers"]
    for header_name in ("Retry-After", "Cache-Control"):
        assert header_name in queued_status_operation["responses"]["200"]["headers"]

    assert set(queued_submission_operation["responses"]) >= {
        "202",
        "400",
        "422",
        "500",
        "503",
    }
    assert set(queued_status_operation["responses"]) >= {"200", "404", "500", "503"}

    for status_code in DOCUMENTED_ERROR_STATUSES:
        error_response = transcript_operation["responses"][status_code]
        content = error_response["content"]["application/json"]
        assert content["schema"] == {"$ref": "#/components/schemas/ErrorResponse"}
        assert content["example"]
