"""Alembic environment for Textify's durable Transcription Job schema."""

import asyncio
from pathlib import Path
from urllib.parse import quote

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from textify.config import AppConfig
from textify.transcription.job_repository import metadata

config = context.config
target_metadata = metadata


def _database_url(database_path: Path) -> str:
    """Build an async SQLite URL that can create a migration target file."""
    raw_path = str(database_path)
    if raw_path == ":memory:" or raw_path.startswith(("file:", "sqlite:")):
        raise RuntimeError("TEXTIFY_DATABASE_PATH must be a filesystem path.")
    resolved_path = database_path.expanduser().resolve(strict=False)
    return f"sqlite+aiosqlite:///{quote(str(resolved_path), safe='/')}"


def _configured_database_url() -> str:
    """Load the migration target from application settings."""
    database_path = AppConfig().database_path
    if database_path is None:
        raise RuntimeError("TEXTIFY_DATABASE_PATH must be configured for migrations.")
    return _database_url(database_path)


def run_migrations_offline() -> None:
    """Run migrations without a live database connection."""
    context.configure(
        url=_configured_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: AsyncConnection) -> None:
    """Configure and execute migrations through one synchronous bridge."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against the configured asynchronous SQLite database."""
    engine = create_async_engine(_configured_database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
