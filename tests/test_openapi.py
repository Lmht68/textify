"""Consumer-visible OpenAPI contract tests."""

from httpx import ASGITransport, AsyncClient

from textify.main import app

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


async def test_openapi_describes_job_only_transcription_contract() -> None:
    """Publish health plus durable Transcription Job submission and polling APIs."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")
    assert response.status_code == 200
    document = response.json()

    assert set(document["paths"]) == {
        "/health",
        "/api/transcription-jobs",
        "/api/transcription-jobs/{job_id}",
    }
    assert set(document["paths"]["/health"]) == {"get"}
    assert set(document["paths"]["/api/transcription-jobs"]) == {"post"}
    assert set(document["paths"]["/api/transcription-jobs/{job_id}"]) == {"get"}
    assert "synchronous" not in document["info"]["description"].casefold()

    components = document["components"]["schemas"]
    health_operation = document["paths"]["/health"]["get"]
    submission_operation = document["paths"]["/api/transcription-jobs"]["post"]
    status_operation = document["paths"]["/api/transcription-jobs/{job_id}"]["get"]

    assert health_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/HealthResponse"}
    assert components["HealthResponse"]["examples"] == [{"status": "ok"}]

    request_schema = submission_operation["requestBody"]["content"]["application/json"][
        "schema"
    ]
    submission_schema = submission_operation["responses"]["202"]["content"][
        "application/json"
    ]["schema"]
    status_schema = status_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    assert request_schema == {"$ref": "#/components/schemas/TranscriptRequest"}
    assert submission_schema == {
        "$ref": "#/components/schemas/QueuedTranscriptionJobResponse"
    }
    assert status_schema == {"$ref": "#/components/schemas/TranscriptionJobResponse"}

    request_component = components["TranscriptRequest"]
    assert "exclude" not in request_component["required"]
    assert (
        set(request_component["properties"]["exclude"]["items"]["enum"])
        == EXPECTED_RESPONSE_FIELD_PATHS
    )

    queued_component = components["QueuedTranscriptionJobResponse"]
    processing_component = components["ProcessingTranscriptionJobResponse"]
    succeeded_component = components["SucceededTranscriptionJobResponse"]
    failed_component = components["FailedTranscriptionJobResponse"]
    assert queued_component["required"] == ["id", "status", "submitted_at", "links"]
    assert queued_component["properties"]["id"]["format"] == "uuid4"
    assert queued_component["properties"]["status"]["const"] == "queued"
    assert queued_component["properties"]["links"] == {
        "$ref": "#/components/schemas/ActiveTranscriptionJobLinks"
    }

    assert processing_component["required"] == [
        "id",
        "status",
        "submitted_at",
        "started_at",
        "cancellation_requested",
        "links",
    ]
    assert processing_component["properties"]["status"]["const"] == "processing"
    assert (
        processing_component["properties"]["cancellation_requested"]["type"]
        == "boolean"
    )

    assert succeeded_component["required"] == [
        "id",
        "status",
        "outcome",
        "submitted_at",
        "started_at",
        "finished_at",
        "result",
        "links",
    ]
    assert succeeded_component["properties"]["status"]["const"] == "finished"
    assert succeeded_component["properties"]["outcome"]["const"] == "succeeded"
    assert succeeded_component["properties"]["result"] == {
        "$ref": "#/components/schemas/TranscriptionResponse"
    }
    assert succeeded_component["properties"]["links"] == {
        "$ref": "#/components/schemas/FinishedTranscriptionJobLinks"
    }

    assert failed_component["required"] == [
        "id",
        "status",
        "outcome",
        "submitted_at",
        "started_at",
        "finished_at",
        "error",
        "links",
    ]
    assert failed_component["properties"]["status"]["const"] == "finished"
    assert failed_component["properties"]["outcome"]["const"] == "failed"
    assert failed_component["properties"]["started_at"]["anyOf"] == [
        {"type": "string", "format": "date-time"},
        {"type": "null"},
    ]
    assert failed_component["properties"]["error"] == {
        "$ref": "#/components/schemas/ErrorDetail"
    }
    assert failed_component["properties"]["links"] == {
        "$ref": "#/components/schemas/FinishedTranscriptionJobLinks"
    }
    assert "result" not in failed_component["properties"]

    assert components["TranscriptionJobResponse"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/QueuedTranscriptionJobResponse"},
            {"$ref": "#/components/schemas/ProcessingTranscriptionJobResponse"},
            {"$ref": "#/components/schemas/SucceededTranscriptionJobResponse"},
            {"$ref": "#/components/schemas/FailedTranscriptionJobResponse"},
        ]
    }
    assert components["ActiveTranscriptionJobLinks"]["required"] == ["self", "cancel"]
    assert components["FinishedTranscriptionJobLinks"]["required"] == ["self"]
    assert "queue_timeout" in components["ErrorDetail"]["properties"]["code"]["enum"]

    for header_name in ("Location", "Retry-After", "Cache-Control"):
        assert header_name in submission_operation["responses"]["202"]["headers"]
    for header_name in ("Retry-After", "Cache-Control"):
        assert header_name in status_operation["responses"]["200"]["headers"]

    assert set(submission_operation["responses"]) >= {"202", "400", "422", "500", "503"}
    assert set(status_operation["responses"]) >= {"200", "404", "500", "503"}
    assert submission_operation["responses"]["400"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ErrorResponse"}
    assert status_operation["responses"]["404"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ErrorResponse"}
