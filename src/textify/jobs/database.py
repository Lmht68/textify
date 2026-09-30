"""Verified PostgreSQL engine construction for durable application storage."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

EXPECTED_DATABASE_REVISION = "0002_execution_attempts"


def create_application_engine(database_url: str) -> AsyncEngine:
    """Create an application PostgreSQL engine.

    Args:
        database_url: Validated asynchronous PostgreSQL connection URL.

    Returns:
        Async engine configured for durable application storage.
    """
    return create_async_engine(
        database_url,
        echo=False,
        hide_parameters=True,
        pool_pre_ping=True,
    )


async def verify_application_database(engine: AsyncEngine) -> None:
    """Prove PostgreSQL storage is writable and at the required revision.

    Args:
        engine: Application PostgreSQL engine created by
            :func:`create_application_engine`.

    Raises:
        RuntimeError: If storage is unavailable, read-only, or not migrated.
    """
    try:
        async with engine.connect() as connection:
            await _require_writable_session(connection)
            versions = await _read_database_revisions(connection)
    except _DatabaseRevisionError as exc:
        raise RuntimeError("Transcription job database schema is not current.") from exc
    except (_DatabaseReadOnlyError, OSError, SQLAlchemyError) as exc:
        raise RuntimeError("Transcription job database is unavailable.") from exc

    if versions != [EXPECTED_DATABASE_REVISION]:
        raise RuntimeError("Transcription job database schema is not current.")


async def _require_writable_session(connection: AsyncConnection) -> None:
    """Reject an application connection configured as read-only.

    Args:
        connection: Open PostgreSQL application connection.

    Raises:
        _DatabaseReadOnlyError: If the current PostgreSQL transaction is read-only.
    """
    transaction_read_only = await connection.scalar(text("SHOW transaction_read_only"))
    if transaction_read_only != "off":
        raise _DatabaseReadOnlyError()


async def _read_database_revisions(connection: AsyncConnection) -> list[object]:
    """Read all Alembic revision rows from PostgreSQL.

    Args:
        connection: Open PostgreSQL application connection.

    Returns:
        Ordered Alembic revision values.

    Raises:
        _DatabaseRevisionError: If the Alembic version table cannot be read.
    """
    try:
        result = await connection.execute(
            text("SELECT version_num FROM alembic_version")
        )
    except SQLAlchemyError as exc:
        raise _DatabaseRevisionError() from exc
    return list(result.scalars())


class _DatabaseRevisionError(RuntimeError):
    """Indicate that Alembic revision inspection could not succeed."""


class _DatabaseReadOnlyError(RuntimeError):
    """Indicate that a PostgreSQL session cannot accept durable writes."""
