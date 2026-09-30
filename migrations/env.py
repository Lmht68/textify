"""Alembic environment for Textify's durable Transcription Job schema."""

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from textify.jobs.config import JobConfig
from textify.jobs.repository import metadata

config = context.config
target_metadata = metadata


def _configured_database_url() -> str:
    """Load the PostgreSQL migration target from application settings."""
    return str(JobConfig().database_url)


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
    """Run migrations against the configured asynchronous PostgreSQL database."""
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
