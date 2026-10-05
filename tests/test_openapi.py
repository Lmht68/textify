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

EXPECTED_PUBLIC_ERROR_CODES = {
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
    "job_already_finished",
    "job_store_unavailable",
    "queue_timeout",
    "worker_interrupted",
    "metadata_timeout",
    "audio_download_timeout",
    "transcription_timeout",
    "internal_error",
}


async def test_openapi_describes_job_only_transcription_contract() -> None:
    """Publish complete health and durable Transcription Job API semantics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert set(document["paths"]) == {
        "/health",
        "/api/transcription-jobs",
        "/api/transcription-jobs/{job_id}",
        "/api/transcription-jobs/{job_id}/cancellation",
    }
    assert {path: set(path_item) for path, path_item in document["paths"].items()} == {
        "/health": {"get"},
        "/api/transcription-jobs": {"post"},
        "/api/transcription-jobs/{job_id}": {"get"},
        "/api/transcription-jobs/{job_id}/cancellation": {"put"},
    }
    assert "synchronous" not in document["info"]["description"].casefold()

    components = document["components"]["schemas"]
    health_operation = document["paths"]["/health"]["get"]
    submission_operation = document["paths"]["/api/transcription-jobs"]["post"]
    status_operation = document["paths"]["/api/transcription-jobs/{job_id}"]["get"]
    cancellation_operation = document["paths"][
        "/api/transcription-jobs/{job_id}/cancellation"
    ]["put"]
    operations = (
        health_operation,
        submission_operation,
        status_operation,
        cancellation_operation,
    )

    assert {
        name: set(operation["responses"])
        for name, operation in {
            "health": health_operation,
            "submission": submission_operation,
            "status": status_operation,
            "cancellation": cancellation_operation,
        }.items()
    } == {
        "health": {"200", "503"},
        "submission": {"202", "400", "422", "500", "503"},
        "status": {"200", "404", "422", "500", "503"},
        "cancellation": {"200", "202", "404", "409", "422", "500", "503"},
    }
    assert all("security" not in operation for operation in operations)
    assert "securitySchemes" not in document["components"]

    assert health_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/HealthResponse"}
    assert components["HealthResponse"]["examples"] == [{"status": "ok"}]
    for health_response in health_operation["responses"].values():
        assert set(health_response["headers"]) == {"X-Request-ID"}

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
    assert "requestBody" not in cancellation_operation
    assert cancellation_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/CancelledTranscriptionJobResponse"}
    assert cancellation_operation["responses"]["202"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ProcessingTranscriptionJobResponse"}

    request_component = components["TranscriptRequest"]
    assert "exclude" not in request_component["required"]
    assert set(request_component["properties"]["exclude"]["items"]["enum"]) == (
        EXPECTED_RESPONSE_FIELD_PATHS
    )

    expected_state_fields = {
        "QueuedTranscriptionJobResponse": {
            "id",
            "status",
            "submitted_at",
            "links",
        },
        "ProcessingTranscriptionJobResponse": {
            "id",
            "status",
            "submitted_at",
            "started_at",
            "cancellation_requested",
            "links",
        },
        "SucceededTranscriptionJobResponse": {
            "id",
            "status",
            "outcome",
            "submitted_at",
            "started_at",
            "finished_at",
            "result",
            "links",
        },
        "FailedTranscriptionJobResponse": {
            "id",
            "status",
            "outcome",
            "submitted_at",
            "started_at",
            "finished_at",
            "error",
            "links",
        },
        "CancelledTranscriptionJobResponse": {
            "id",
            "status",
            "outcome",
            "submitted_at",
            "started_at",
            "finished_at",
            "links",
        },
    }
    for schema_name, expected_fields in expected_state_fields.items():
        component = components[schema_name]
        assert component["additionalProperties"] is False
        assert set(component["required"]) == expected_fields
        assert set(component["properties"]) == expected_fields

    queued_component = components["QueuedTranscriptionJobResponse"]
    processing_component = components["ProcessingTranscriptionJobResponse"]
    succeeded_component = components["SucceededTranscriptionJobResponse"]
    failed_component = components["FailedTranscriptionJobResponse"]
    cancelled_component = components["CancelledTranscriptionJobResponse"]
    assert queued_component["properties"]["id"]["format"] == "uuid4"
    assert queued_component["properties"]["status"]["const"] == "queued"
    assert processing_component["properties"]["status"]["const"] == "processing"
    assert (
        processing_component["properties"]["cancellation_requested"]["type"]
        == "boolean"
    )
    assert succeeded_component["properties"]["status"]["const"] == "finished"
    assert succeeded_component["properties"]["outcome"]["const"] == "succeeded"
    assert succeeded_component["properties"]["result"] == {
        "$ref": "#/components/schemas/TranscriptionResponse"
    }
    assert failed_component["properties"]["status"]["const"] == "finished"
    assert failed_component["properties"]["outcome"]["const"] == "failed"
    assert failed_component["properties"]["error"] == {
        "$ref": "#/components/schemas/ErrorDetail"
    }
    assert cancelled_component["properties"]["status"]["const"] == "finished"
    assert cancelled_component["properties"]["outcome"]["const"] == "cancelled"
    assert "result" not in failed_component["properties"]
    assert "error" not in cancelled_component["properties"]
    assert "result" not in cancelled_component["properties"]
    assert failed_component["properties"]["started_at"]["anyOf"] == [
        {"type": "string", "format": "date-time"},
        {"type": "null"},
    ]
    assert cancelled_component["properties"]["started_at"]["anyOf"] == [
        {"type": "string", "format": "date-time"},
        {"type": "null"},
    ]

    assert components["TranscriptionJobResponse"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/QueuedTranscriptionJobResponse"},
            {"$ref": "#/components/schemas/ProcessingTranscriptionJobResponse"},
            {"$ref": "#/components/schemas/SucceededTranscriptionJobResponse"},
            {"$ref": "#/components/schemas/FailedTranscriptionJobResponse"},
            {"$ref": "#/components/schemas/CancelledTranscriptionJobResponse"},
        ]
    }

    active_links = components["ActiveTranscriptionJobLinks"]
    finished_links = components["FinishedTranscriptionJobLinks"]
    assert active_links["required"] == ["self", "cancel"]
    assert finished_links["required"] == ["self"]
    for link_component in (active_links, finished_links):
        self_description = link_component["properties"]["self"][
            "description"
        ].casefold()
        assert "same-origin relative" in self_description
        assert "bearer capability" in self_description
    assert (
        "same-origin relative"
        in active_links["properties"]["cancel"]["description"].casefold()
    )

    for operation in (status_operation, cancellation_operation):
        parameter = operation["parameters"][0]
        assert parameter["name"] == "job_id"
        assert parameter["in"] == "path"
        assert parameter["required"] is True
        assert parameter["schema"]["type"] == "string"
        assert "format" not in parameter["schema"]
        assert "bearer capability" in parameter["description"]

    assert set(components["ErrorDetail"]["properties"]["code"]["enum"]) == (
        EXPECTED_PUBLIC_ERROR_CODES
    )
    for operation, immediate_codes in (
        (
            submission_operation,
            ("invalid_url", "unsupported_platform", "invalid_request"),
        ),
        (status_operation, ("job_not_found", "invalid_request")),
        (
            cancellation_operation,
            ("job_not_found", "job_already_finished", "invalid_request"),
        ),
    ):
        descriptions = " ".join(
            response["description"] for response in operation["responses"].values()
        )
        for code in immediate_codes:
            assert code in descriptions
    assert "queue_timeout" in status_operation["description"]
    assert "worker_interrupted" in status_operation["description"]
    assert "internal_error" in status_operation["description"]

    job_error_responses = (
        *(
            submission_operation["responses"][status_code]
            for status_code in ("400", "422", "500", "503")
        ),
        *(
            status_operation["responses"][status_code]
            for status_code in ("404", "422", "500", "503")
        ),
        *(
            cancellation_operation["responses"][status_code]
            for status_code in ("404", "409", "422", "500", "503")
        ),
    )
    for job_error_response in job_error_responses:
        assert set(job_error_response["headers"]) == {
            "X-Request-ID",
            "Cache-Control",
        }
        assert job_error_response["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }

    assert set(submission_operation["responses"]["202"]["headers"]) == {
        "X-Request-ID",
        "Cache-Control",
        "Retry-After",
        "Location",
    }
    assert set(status_operation["responses"]["200"]["headers"]) == {
        "X-Request-ID",
        "Cache-Control",
        "Retry-After",
    }
    assert set(cancellation_operation["responses"]["200"]["headers"]) == {
        "X-Request-ID",
        "Cache-Control",
    }
    assert set(cancellation_operation["responses"]["202"]["headers"]) == {
        "X-Request-ID",
        "Cache-Control",
        "Retry-After",
    }
    assert (
        "same-origin relative"
        in submission_operation["responses"]["202"]["headers"]["Location"][
            "description"
        ].casefold()
    )
    assert (
        "Present only while work is active"
        in status_operation["responses"]["200"]["headers"]["Retry-After"]["description"]
    )
