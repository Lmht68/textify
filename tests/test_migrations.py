"""Integration tests for durable Transcription Job migrations and SQLite policy."""

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from textify.jobs.database import (
    EXPECTED_DATABASE_REVISION,
    SQLITE_BUSY_TIMEOUT_MS,
    create_application_engine,
)

PROJECT_ROOT = Path(__file__).parent.parent
APPLICATION_TABLES = {
    "transcription_job",
    "transcription_job_exclusion",
    "transcription_job_result",
    "transcription_job_segment",
}


def test_initial_migration_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Upgrade and downgrade the complete normalized durable job schema.

    Args:
        monkeypatch: Fixture used to configure Alembic's target database.
        tmp_path: Per-test temporary filesystem root.
    """
    database_path = tmp_path / "migration.sqlite3"
    monkeypatch.setenv("TEXTIFY_DATABASE_PATH", str(database_path))
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))

    command.upgrade(alembic_config, "head")

    script_directory = ScriptDirectory.from_config(alembic_config)
    assert script_directory.get_current_head() == EXPECTED_DATABASE_REVISION
    with sqlite3.connect(database_path) as connection:
        table_names = _table_names(connection)
        assert APPLICATION_TABLES <= table_names
        assert _revision_rows(connection) == [EXPECTED_DATABASE_REVISION]
        job_table_sql = _table_sql(connection, "transcription_job")
        assert "transcription_job_public_id_uuid4_check" in job_table_sql
        assert "transcription_job_lifecycle_check" in job_table_sql
        assert "transcription_job_timestamp_order_check" in job_table_sql
        assert "transcription_job_error_pair_check" in job_table_sql
        assert "AUTOINCREMENT" not in job_table_sql
        assert _index_names(connection, "transcription_job") >= {
            "transcription_job_status_submitted_at_id_idx",
            "transcription_job_status_queue_deadline_at_idx",
        }
        assert _foreign_key_targets(connection, "transcription_job_exclusion") == {
            "transcription_job"
        }
        assert _foreign_key_targets(connection, "transcription_job_result") == {
            "transcription_job"
        }
        assert _foreign_key_targets(connection, "transcription_job_segment") == {
            "transcription_job_result"
        }

    command.downgrade(alembic_config, "base")

    with sqlite3.connect(database_path) as connection:
        assert not APPLICATION_TABLES & _table_names(connection)

    command.upgrade(alembic_config, "head")

    with sqlite3.connect(database_path) as connection:
        assert APPLICATION_TABLES <= _table_names(connection)
        assert _revision_rows(connection) == [EXPECTED_DATABASE_REVISION]


async def test_application_database_connections_enforce_sqlite_policy(
    migrated_database_path: Path,
) -> None:
    """Apply every configured SQLite safety PRAGMA to an application connection.

    Args:
        migrated_database_path: Fresh database upgraded through Alembic.
    """
    engine = create_application_engine(migrated_database_path)
    try:
        async with engine.connect() as connection:
            foreign_keys = await connection.scalar(text("PRAGMA foreign_keys"))
            journal_mode = await connection.scalar(text("PRAGMA journal_mode"))
            synchronous = await connection.scalar(text("PRAGMA synchronous"))
            secure_delete = await connection.scalar(text("PRAGMA secure_delete"))
            busy_timeout = await connection.scalar(text("PRAGMA busy_timeout"))
    finally:
        await engine.dispose()

    assert foreign_keys == 1
    assert journal_mode == "wal"
    assert synchronous == 2
    assert secure_delete == 1
    assert busy_timeout == SQLITE_BUSY_TIMEOUT_MS


def _table_names(connection: sqlite3.Connection) -> set[str]:
    """Return ordinary SQLite table names from one temporary database."""
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _revision_rows(connection: sqlite3.Connection) -> list[str]:
    """Return exact Alembic version values from an upgraded SQLite database."""
    rows = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    return [str(row[0]) for row in rows]


def _table_sql(connection: sqlite3.Connection, table_name: str) -> str:
    """Return one SQLite table's persisted creation statement."""
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    assert row is not None
    return str(row[0])


def _index_names(connection: sqlite3.Connection, table_name: str) -> set[str]:
    """Return SQLite index names attached to one table."""
    rows = connection.execute(f"PRAGMA index_list({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _foreign_key_targets(connection: sqlite3.Connection, table_name: str) -> set[str]:
    """Return referenced table names from one SQLite table's foreign keys."""
    rows = connection.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
    return {str(row[2]) for row in rows}
