"""Integration tests for durable PostgreSQL Transcription Job migrations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid1, uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from textify.jobs.database import EXPECTED_DATABASE_REVISION, create_application_engine

PROJECT_ROOT = Path(__file__).parent.parent
APPLICATION_TABLES = {
    "transcription_job",
    "transcription_job_execution_attempt",
    "transcription_job_exclusion",
    "transcription_job_result",
    "transcription_job_segment",
}
_STORAGE_TRIGGER_NAMES = {
    "transcription_job_storage_execution_attempt_check",
    "transcription_job_storage_job_check",
    "transcription_job_storage_exclusion_check",
    "transcription_job_storage_result_check",
    "transcription_job_storage_segment_check",
}
_CHECK_CONSTRAINT_NAMES = {
    "transcription_job_public_id_uuid4_check",
    "transcription_job_submitted_url_length_check",
    "transcription_job_status_check",
    "transcription_job_outcome_check",
    "transcription_job_timestamp_order_check",
    "transcription_job_error_pair_check",
    "transcription_job_lifecycle_check",
    "transcription_job_execution_attempt_token_uuid4_check",
    "transcription_job_exclusion_field_path_check",
    "transcription_job_result_platform_check",
    "transcription_job_result_method_check",
    "transcription_job_result_duration_check",
    "transcription_job_result_video_id_length_check",
    "transcription_job_result_source_url_length_check",
    "transcription_job_result_language_length_check",
    "transcription_job_segment_values_check",
}


InvalidJobValues = Callable[[datetime], dict[str, object]]


@pytest.fixture
async def migrated_engine(
    migrated_postgresql_database_url: str,
) -> AsyncIterator[AsyncEngine]:
    """Create an engine for one PostgreSQL database migrated to head.

    Args:
        migrated_postgresql_database_url: Isolated database upgraded to Alembic head.

    Yields:
        Disposable engine connected to the migrated PostgreSQL database.
    """
    engine = create_application_engine(migrated_postgresql_database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


async def test_initial_migration_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    postgresql_database_url: str,
) -> None:
    """Upgrade, downgrade, and re-upgrade the PostgreSQL baseline."""
    monkeypatch.setenv("TEXTIFY_DATABASE_URL", postgresql_database_url)
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_application_engine(postgresql_database_url)
    try:
        await asyncio.to_thread(command.upgrade, alembic_config, "head")

        script_directory = ScriptDirectory.from_config(alembic_config)
        assert script_directory.get_current_head() == EXPECTED_DATABASE_REVISION
        assert await _application_table_names(engine) == APPLICATION_TABLES
        assert await _revision_rows(engine) == [EXPECTED_DATABASE_REVISION]

        await asyncio.to_thread(command.downgrade, alembic_config, "base")
        assert not await _application_table_names(engine)

        await asyncio.to_thread(command.upgrade, alembic_config, "head")
        assert await _application_table_names(engine) == APPLICATION_TABLES
        assert await _revision_rows(engine) == [EXPECTED_DATABASE_REVISION]
    finally:
        await engine.dispose()


async def test_execution_attempt_migration_backfills_and_round_trips_populated_jobs(
    monkeypatch: pytest.MonkeyPatch,
    postgresql_database_url: str,
) -> None:
    """Backfill one UUIDv4 attempt per existing job and retain downgrade support."""
    monkeypatch.setenv("TEXTIFY_DATABASE_URL", postgresql_database_url)
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    engine = create_application_engine(postgresql_database_url)
    now = datetime.now(UTC)
    try:
        await asyncio.to_thread(
            command.upgrade,
            alembic_config,
            "0001_postgresql_jobs",
        )
        async with engine.begin() as connection:
            first_job_id = await _insert_job(
                connection,
                _queued_values(now),
                with_execution_attempt=False,
            )
            second_job_id = await _insert_job(
                connection,
                _queued_values(now),
                with_execution_attempt=False,
            )

        await asyncio.to_thread(command.upgrade, alembic_config, "head")
        async with engine.connect() as connection:
            attempt_rows = (
                (
                    await connection.execute(
                        text(
                            "SELECT job_id, execution_attempt_token "
                            "FROM transcription_job_execution_attempt "
                            "ORDER BY job_id"
                        )
                    )
                )
                .mappings()
                .all()
            )
        assert await _revision_rows(engine) == [EXPECTED_DATABASE_REVISION]
        assert [row["job_id"] for row in attempt_rows] == [
            first_job_id,
            second_job_id,
        ]
        for attempt_row in attempt_rows:
            token = attempt_row["execution_attempt_token"]
            assert isinstance(token, UUID)
            assert token.version == 4

        await asyncio.to_thread(
            command.downgrade,
            alembic_config,
            "0001_postgresql_jobs",
        )
        assert await _revision_rows(engine) == ["0001_postgresql_jobs"]
        assert await _application_table_names(engine) == (
            APPLICATION_TABLES - {"transcription_job_execution_attempt"}
        )

        await asyncio.to_thread(command.upgrade, alembic_config, "head")
        async with engine.connect() as connection:
            reupgraded_attempt_count = await connection.scalar(
                text("SELECT count(*) FROM transcription_job_execution_attempt")
            )
        assert reupgraded_attempt_count == 2
    finally:
        await engine.dispose()


async def test_postgresql_baseline_uses_required_native_schema(
    migrated_engine: AsyncEngine,
) -> None:
    """Persist only durable-job tables with PostgreSQL-native identifiers and types."""
    async with migrated_engine.connect() as connection:
        columns = {
            row["column_name"]: row
            for row in (
                (
                    await connection.execute(
                        text(
                            "SELECT column_name, data_type, udt_name, is_identity, "
                            "identity_generation "
                            "FROM information_schema.columns "
                            "WHERE table_schema = 'public' "
                            "AND table_name = 'transcription_job'"
                        )
                    )
                )
                .mappings()
                .all()
            )
        }
        execution_attempt_columns = {
            row["column_name"]: row
            for row in (
                (
                    await connection.execute(
                        text(
                            "SELECT column_name, data_type, udt_name "
                            "FROM information_schema.columns "
                            "WHERE table_schema = 'public' "
                            "AND table_name = "
                            "'transcription_job_execution_attempt'"
                        )
                    )
                )
                .mappings()
                .all()
            )
        }
        index_names = set(
            (
                await connection.scalars(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE schemaname = 'public' "
                        "AND tablename = 'transcription_job'"
                    )
                )
            ).all()
        )
        check_constraint_names = set(
            (
                await connection.scalars(
                    text(
                        "SELECT check_constraint.conname "
                        "FROM pg_constraint AS check_constraint "
                        "JOIN pg_class AS table_class "
                        "ON table_class.oid = check_constraint.conrelid "
                        "JOIN pg_namespace AS table_schema "
                        "ON table_schema.oid = table_class.relnamespace "
                        "WHERE check_constraint.contype = 'c' "
                        "AND table_schema.nspname = 'public' "
                        "AND table_class.relname = ANY(:table_names)"
                    ),
                    {"table_names": list(APPLICATION_TABLES)},
                )
            ).all()
        )
        foreign_keys = {
            (row["table_name"], row["referenced_table"], row["delete_action"])
            for row in (
                (
                    await connection.execute(
                        text(
                            "SELECT child.relname AS table_name, "
                            "parent.relname AS referenced_table, "
                            "foreign_key.confdeltype::text AS delete_action "
                            "FROM pg_constraint AS foreign_key "
                            "JOIN pg_class AS child "
                            "ON child.oid = foreign_key.conrelid "
                            "JOIN pg_class AS parent "
                            "ON parent.oid = foreign_key.confrelid "
                            "WHERE foreign_key.contype = 'f'"
                        )
                    )
                )
                .mappings()
                .all()
            )
        }
        trigger_rows = (
            (
                await connection.execute(
                    text(
                        "SELECT trigger.tgname, trigger.tgdeferrable, "
                        "trigger.tginitdeferred, procedure.proname "
                        "FROM pg_trigger AS trigger "
                        "JOIN pg_proc AS procedure ON procedure.oid = trigger.tgfoid "
                        "WHERE trigger.tgname = ANY(:trigger_names)"
                    ),
                    {"trigger_names": list(_STORAGE_TRIGGER_NAMES)},
                )
            )
            .mappings()
            .all()
        )
        application_columns = set(
            (
                await connection.scalars(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = 'public' "
                        "AND table_name = ANY(:table_names)"
                    ),
                    {"table_names": list(APPLICATION_TABLES)},
                )
            ).all()
        )

    assert columns["id"]["udt_name"] == "int8"
    assert columns["id"]["is_identity"] == "YES"
    assert columns["id"]["identity_generation"] == "BY DEFAULT"
    assert columns["public_id"]["udt_name"] == "uuid"
    assert columns["cancellation_requested"]["udt_name"] == "bool"
    assert execution_attempt_columns["job_id"]["udt_name"] == "int8"
    assert execution_attempt_columns["execution_attempt_token"]["udt_name"] == "uuid"
    for timestamp_column in (
        "submitted_at",
        "queue_deadline_at",
        "started_at",
        "finished_at",
    ):
        assert columns[timestamp_column]["data_type"] == "timestamp with time zone"
    assert index_names >= {
        "transcription_job_status_submitted_at_id_idx",
        "transcription_job_status_queue_deadline_at_idx",
        "transcription_job_status_finished_at_idx",
    }
    assert check_constraint_names == _CHECK_CONSTRAINT_NAMES
    assert foreign_keys == {
        ("transcription_job_execution_attempt", "transcription_job", "c"),
        ("transcription_job_exclusion", "transcription_job", "c"),
        ("transcription_job_result", "transcription_job", "c"),
        ("transcription_job_segment", "transcription_job_result", "c"),
    }
    assert {row["tgname"] for row in trigger_rows} == _STORAGE_TRIGGER_NAMES
    assert all(
        row["tgdeferrable"]
        and row["tginitdeferred"]
        and row["proname"] == "enforce_transcription_job_storage_invariants"
        for row in trigger_rows
    )
    assert not APPLICATION_TABLES & {
        "user",
        "account",
        "billing",
        "owner",
    }
    assert not application_columns & {
        "user_id",
        "account_id",
        "billing_id",
        "owner_id",
    }


async def test_postgresql_baseline_accepts_all_legal_lifecycle_origins(
    migrated_engine: AsyncEngine,
) -> None:
    """Commit every legal queued, processing, failed, cancelled, and succeeded shape."""
    now = datetime.now(UTC)
    async with migrated_engine.begin() as connection:
        await _insert_job(connection, _queued_values(now))
        await _insert_job(
            connection,
            _queued_values(now)
            | {
                "status": "processing",
                "started_at": now,
            },
        )
        await _insert_job(
            connection,
            _queued_values(now)
            | {
                "status": "finished",
                "outcome": "failed",
                "started_at": now,
                "finished_at": now,
                "error_code": "worker_interrupted",
                "error_message": "The worker was interrupted.",
            },
        )
        await _insert_job(
            connection,
            _queued_values(now)
            | {
                "status": "finished",
                "outcome": "failed",
                "finished_at": now,
                "error_code": "queue_timeout",
                "error_message": "The job exceeded its queue timeout.",
            },
        )
        await _insert_job(
            connection,
            _queued_values(now)
            | {
                "status": "finished",
                "outcome": "cancelled",
                "finished_at": now,
            },
        )
        await _insert_job(
            connection,
            _queued_values(now)
            | {
                "status": "finished",
                "outcome": "cancelled",
                "started_at": now,
                "finished_at": now,
                "cancellation_requested": True,
            },
        )
        succeeded_job_id = await _insert_job(
            connection,
            _queued_values(now)
            | {
                "status": "finished",
                "outcome": "succeeded",
                "started_at": now,
                "finished_at": now,
            },
        )
        await _insert_result(connection, succeeded_job_id)

    async with migrated_engine.connect() as connection:
        job_count = await connection.scalar(
            text("SELECT count(*) FROM transcription_job")
        )
    assert job_count == 7


async def test_postgresql_rejects_invalid_and_duplicate_execution_attempt_tokens(
    migrated_engine: AsyncEngine,
) -> None:
    """Require unique UUIDv4 tokens for each persisted Execution Attempt."""
    now = datetime.now(UTC)
    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            job_id = await _insert_job(
                connection,
                _queued_values(now),
                with_execution_attempt=False,
            )
            await _insert_execution_attempt(connection, job_id, uuid1())

    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            first_job_id = await _insert_job(
                connection,
                _queued_values(now),
                with_execution_attempt=False,
            )
            second_job_id = await _insert_job(
                connection,
                _queued_values(now),
                with_execution_attempt=False,
            )
            execution_attempt_token = uuid4()
            await _insert_execution_attempt(
                connection,
                first_job_id,
                execution_attempt_token,
            )
            await _insert_execution_attempt(
                connection,
                second_job_id,
                execution_attempt_token,
            )


async def test_postgresql_requires_exactly_one_execution_attempt_per_job(
    migrated_engine: AsyncEngine,
) -> None:
    """Reject a committed Transcription Job missing its Execution Attempt."""
    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            await _insert_job(
                connection,
                _queued_values(datetime.now(UTC)),
                with_execution_attempt=False,
            )


@pytest.mark.parametrize(
    "invalid_values",
    (
        lambda now: _queued_values(now) | {"public_id": uuid1()},
        lambda now: (
            _queued_values(now)
            | {
                "status": "processing",
            }
        ),
        lambda now: (
            _queued_values(now)
            | {
                "status": "finished",
                "outcome": "failed",
                "started_at": now,
                "finished_at": now,
            }
        ),
        lambda now: (
            _queued_values(now)
            | {
                "status": "finished",
                "outcome": "cancelled",
                "finished_at": now,
                "cancellation_requested": True,
            }
        ),
        lambda now: (
            _queued_values(now)
            | {
                "queue_deadline_at": now,
            }
        ),
        lambda now: (
            _queued_values(now)
            | {
                "status": "finished",
                "outcome": "failed",
                "started_at": now,
                "finished_at": now,
                "error_code": "not_safe",
                "error_message": "Unvalidated error.",
            }
        ),
    ),
)
async def test_postgresql_baseline_rejects_invalid_job_snapshots(
    migrated_engine: AsyncEngine,
    invalid_values: InvalidJobValues,
) -> None:
    """Reject invalid UUID, lifecycle, timestamp, cancellation, and error snapshots."""
    values = invalid_values(datetime.now(UTC))

    await _assert_job_rejected(migrated_engine, values)


async def test_postgresql_baseline_rejects_inconsistent_result_storage(
    migrated_engine: AsyncEngine,
) -> None:
    """Reject result rows for non-success jobs and success jobs lacking one result."""
    now = datetime.now(UTC)
    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            processing_job_id = await _insert_job(
                connection,
                _queued_values(now)
                | {
                    "status": "processing",
                    "started_at": now,
                },
            )
            await _insert_result(connection, processing_job_id)

    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            await _insert_job(
                connection,
                _queued_values(now)
                | {
                    "status": "finished",
                    "outcome": "succeeded",
                    "started_at": now,
                    "finished_at": now,
                },
            )


async def test_postgresql_baseline_rejects_projection_and_segment_mismatches(
    migrated_engine: AsyncEngine,
) -> None:
    """Reject exclusion conflicts, persisted excluded segments, and ordinal gaps."""
    now = datetime.now(UTC)
    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            job_id = await _insert_succeeded_job(connection, now)
            await _insert_result(connection, job_id)
            await connection.execute(
                text(
                    "INSERT INTO transcription_job_exclusion (job_id, field_path) "
                    "VALUES (:job_id, 'source.title')"
                ),
                {"job_id": job_id},
            )

    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            job_id = await _insert_succeeded_job(connection, now)
            await _insert_result(connection, job_id)
            await connection.execute(
                text(
                    "INSERT INTO transcription_job_exclusion (job_id, field_path) "
                    "VALUES (:job_id, 'transcript.segments')"
                ),
                {"job_id": job_id},
            )
            await _insert_segment(connection, job_id, ordinal=0)

    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            job_id = await _insert_succeeded_job(connection, now)
            await _insert_result(connection, job_id)
            await _insert_segment(connection, job_id, ordinal=1)

    with pytest.raises(IntegrityError):
        async with migrated_engine.begin() as connection:
            job_id = await _insert_succeeded_job(connection, now)
            await _insert_result(connection, job_id)
            await connection.execute(
                text(
                    "INSERT INTO transcription_job_segment ("
                    "job_id, ordinal, start_seconds, end_seconds, text) VALUES ("
                    ":job_id, 0, 'NaN'::double precision, 1.0, 'segment')"
                ),
                {"job_id": job_id},
            )


async def test_postgresql_cascades_execution_attempt_and_success_projection_deletion(
    migrated_engine: AsyncEngine,
) -> None:
    """Delete private attempts and normalized results with their retained job."""
    now = datetime.now(UTC)
    async with migrated_engine.begin() as connection:
        job_id = await _insert_succeeded_job(connection, now)
        await _insert_result(connection, job_id)
        await _insert_segment(connection, job_id, ordinal=0)

    async with migrated_engine.begin() as connection:
        await connection.execute(
            text("DELETE FROM transcription_job WHERE id = :job_id"),
            {"job_id": job_id},
        )
        execution_attempt_count = await connection.scalar(
            text(
                "SELECT count(*) FROM transcription_job_execution_attempt "
                "WHERE job_id = :job_id"
            ),
            {"job_id": job_id},
        )
        result_count = await connection.scalar(
            text(
                "SELECT count(*) FROM transcription_job_result WHERE job_id = :job_id"
            ),
            {"job_id": job_id},
        )
        segment_count = await connection.scalar(
            text(
                "SELECT count(*) FROM transcription_job_segment WHERE job_id = :job_id"
            ),
            {"job_id": job_id},
        )

    assert execution_attempt_count == 0
    assert result_count == 0
    assert segment_count == 0


async def _application_table_names(engine: AsyncEngine) -> set[str]:
    """Return Transcription Job tables present in one PostgreSQL database."""
    async with engine.connect() as connection:
        table_names = set(
            (
                await connection.scalars(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
            ).all()
        )
    return table_names & APPLICATION_TABLES


async def _revision_rows(engine: AsyncEngine) -> list[str]:
    """Return exact Alembic revision values from an upgraded PostgreSQL database."""
    async with engine.connect() as connection:
        return list(
            (
                await connection.scalars(
                    text("SELECT version_num FROM alembic_version")
                )
            ).all()
        )


def _queued_values(now: datetime) -> dict[str, object]:
    """Build one valid queued Transcription Job storage row."""
    return {
        "public_id": uuid4(),
        "submitted_url": f"https://example.test/{uuid4()}",
        "status": "queued",
        "outcome": None,
        "submitted_at": now,
        "queue_deadline_at": now + timedelta(seconds=20),
        "started_at": None,
        "finished_at": None,
        "cancellation_requested": False,
        "error_code": None,
        "error_message": None,
    }


async def _insert_job(
    connection: AsyncConnection,
    values: Mapping[str, object],
    *,
    with_execution_attempt: bool = True,
) -> int:
    """Insert one job and, by default, its required Execution Attempt.

    Args:
        connection: Open transaction receiving the durable storage rows.
        values: Complete values for the Transcription Job row.
        with_execution_attempt: Whether to create the required attempt in this
            transaction.

    Returns:
        The generated private Transcription Job identifier.
    """
    job_id = await connection.scalar(
        text(
            "INSERT INTO transcription_job ("
            "public_id, submitted_url, status, outcome, submitted_at, "
            "queue_deadline_at, started_at, finished_at, cancellation_requested, "
            "error_code, error_message) VALUES ("
            ":public_id, :submitted_url, :status, :outcome, :submitted_at, "
            ":queue_deadline_at, :started_at, :finished_at, "
            ":cancellation_requested, :error_code, :error_message) "
            "RETURNING id"
        ),
        values,
    )
    assert not isinstance(job_id, bool)
    assert isinstance(job_id, int)
    if with_execution_attempt:
        await _insert_execution_attempt(connection, job_id, uuid4())
    return job_id


async def _insert_execution_attempt(
    connection: AsyncConnection,
    job_id: int,
    execution_attempt_token: UUID,
) -> None:
    """Insert one private Execution Attempt for an existing Transcription Job."""
    await connection.execute(
        text(
            "INSERT INTO transcription_job_execution_attempt "
            "(job_id, execution_attempt_token) VALUES "
            "(:job_id, :execution_attempt_token)"
        ),
        {
            "job_id": job_id,
            "execution_attempt_token": execution_attempt_token,
        },
    )


async def _insert_succeeded_job(
    connection: AsyncConnection,
    now: datetime,
) -> int:
    """Insert one success-shaped job row before its required result row."""
    return await _insert_job(
        connection,
        _queued_values(now)
        | {
            "status": "finished",
            "outcome": "succeeded",
            "started_at": now,
            "finished_at": now,
        },
    )


async def _insert_result(connection: AsyncConnection, job_id: int) -> None:
    """Insert one complete unexcluded successful result row."""
    await connection.execute(
        text(
            "INSERT INTO transcription_job_result ("
            "job_id, source_platform, source_video_id, source_url, source_title, "
            "source_description, source_channel, source_duration_seconds, "
            "transcript_method, transcript_language, transcript_text) VALUES ("
            ":job_id, 'youtube', 'video', 'https://example.test/source', "
            "'title', 'description', 'channel', 1, 'youtube_captions', 'en', "
            "'segment')"
        ),
        {"job_id": job_id},
    )


async def _insert_segment(
    connection: AsyncConnection,
    job_id: int,
    *,
    ordinal: int,
) -> None:
    """Insert one valid normalized segment with the requested ordinal."""
    await connection.execute(
        text(
            "INSERT INTO transcription_job_segment ("
            "job_id, ordinal, start_seconds, end_seconds, text) VALUES ("
            ":job_id, :ordinal, 0.0, 1.0, 'segment')"
        ),
        {"job_id": job_id, "ordinal": ordinal},
    )


async def _assert_job_rejected(
    engine: AsyncEngine,
    values: Mapping[str, object],
) -> None:
    """Assert PostgreSQL rejects one invalid job snapshot at transaction commit."""
    with pytest.raises(IntegrityError):
        async with engine.begin() as connection:
            await _insert_job(connection, values)
