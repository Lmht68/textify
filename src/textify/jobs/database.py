"""Verified SQLite engine construction for durable application storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

EXPECTED_DATABASE_REVISION = "0001_transcription_jobs"
SQLITE_BUSY_TIMEOUT_MS = 5_000


class _DbapiCursor(Protocol):
    """Describe the synchronous DBAPI cursor operations used by the hook."""

    def close(self) -> None:
        """Close the database cursor."""
        ...

    def execute(self, operation: str) -> object:
        """Execute one SQLite PRAGMA statement."""
        ...

    def fetchone(self) -> tuple[object, ...] | None:
        """Read one SQLite PRAGMA result row."""
        ...


class _DbapiConnection(Protocol):
    """Describe the synchronous DBAPI connection operation used by the hook."""

    def cursor(self) -> _DbapiCursor:
        """Create one DBAPI cursor."""
        ...


def create_application_engine(database_path: Path) -> AsyncEngine:
    """Create a non-creating SQLite engine for the migrated application store.

    Args:
        database_path: Configured persistent SQLite filesystem path.

    Returns:
        Async engine that opens the resolved SQLite file only in ``mode=rw``.

    Raises:
        ValueError: If the configured value is a SQLite URI or in-memory store.
    """
    resolved_path = _resolve_database_path(database_path)
    database_url = _application_database_url(resolved_path)
    engine = create_async_engine(
        database_url,
        echo=False,
        hide_parameters=True,
    )
    event.listen(engine.sync_engine, "connect", _configure_sqlite_connection)
    return engine


async def verify_application_database(engine: AsyncEngine) -> None:
    """Prove the existing database is writable and at the required revision.

    Args:
        engine: Application SQLite engine created by :func:`create_application_engine`.

    Raises:
        RuntimeError: If storage is unavailable, misconfigured, or not migrated.
    """
    try:
        async with engine.connect() as connection:
            await connection.execute(text("BEGIN IMMEDIATE"))
            versions = await _read_database_revisions(connection)
            await connection.rollback()
    except _DatabaseRevisionError as exc:
        raise RuntimeError("Transcription job database schema is not current.") from exc
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        raise RuntimeError("Transcription job database is unavailable.") from exc

    if versions != [EXPECTED_DATABASE_REVISION]:
        raise RuntimeError("Transcription job database schema is not current.")


async def _read_database_revisions(connection: AsyncConnection) -> list[object]:
    """Read all Alembic revision rows while the writeability transaction is held."""
    try:
        result = await connection.execute(
            text("SELECT version_num FROM alembic_version")
        )
    except SQLAlchemyError as exc:
        raise _DatabaseRevisionError() from exc
    return list(result.scalars())


def _resolve_database_path(database_path: Path) -> Path:
    """Reject SQLite URI forms and resolve a filesystem path once."""
    raw_path = str(database_path)
    if raw_path == ":memory:" or raw_path.startswith(("file:", "sqlite:")):
        raise ValueError("TEXTIFY_DATABASE_PATH must be a filesystem path.")
    return database_path.expanduser().resolve(strict=False)


def _application_database_url(database_path: Path) -> str:
    """Build a SQLite URI that refuses to create a missing database file."""
    return (
        "sqlite+aiosqlite:///file:"
        f"{quote(str(database_path), safe='/')}?mode=rw&uri=true"
    )


def _configure_sqlite_connection(
    dbapi_connection: _DbapiConnection,
    _connection_record: object,
) -> None:
    """Apply and verify SQLite safety settings for every DBAPI connection."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys = ON")
        _require_pragma(cursor, "foreign_keys", 1)
        cursor.execute("PRAGMA journal_mode = WAL")
        _require_pragma(cursor, "journal_mode", "wal")
        cursor.execute("PRAGMA synchronous = FULL")
        _require_pragma(cursor, "synchronous", 2)
        cursor.execute("PRAGMA secure_delete = ON")
        _require_pragma(cursor, "secure_delete", 1)
        cursor.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        _require_pragma(cursor, "busy_timeout", SQLITE_BUSY_TIMEOUT_MS)
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise RuntimeError("Unable to configure SQLite database connection.") from exc
    finally:
        cursor.close()


def _require_pragma(
    cursor: _DbapiCursor,
    name: str,
    expected_value: int | str,
) -> None:
    """Require the current connection's SQLite PRAGMA to equal its policy value."""
    cursor.execute(f"PRAGMA {name}")
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"SQLite PRAGMA {name} did not return a value.")
    actual_value = row[0]
    if isinstance(expected_value, str):
        if not isinstance(actual_value, str) or actual_value.lower() != expected_value:
            raise RuntimeError(f"SQLite PRAGMA {name} is not configured.")
        return
    if actual_value != expected_value:
        raise RuntimeError(f"SQLite PRAGMA {name} is not configured.")


class _DatabaseRevisionError(RuntimeError):
    """Indicate that Alembic revision inspection could not succeed."""
