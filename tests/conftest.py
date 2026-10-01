"""Shared real-PostgreSQL fixtures for durable Transcription Job tests."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

PROJECT_ROOT = Path(__file__).parent.parent
_TEST_DATABASE_PREFIX = "textify_test_"


class RestartableRedisBroker:
    """Own an isolated non-persistent Redis child for restart fault tests."""

    def __init__(
        self,
        redis_server_path: str,
        port: int,
        data_directory: Path,
    ) -> None:
        """Initialize one stopped Redis child process controller.

        Args:
            redis_server_path: Absolute path to the required Redis server binary.
            port: Reserved loopback port for this child only.
            data_directory: Per-test directory for Redis data and logs.
        """
        self._redis_server_path = redis_server_path
        self._port = port
        self._data_directory = data_directory
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        """Return the private broker URL for this isolated Redis child."""
        return f"redis://127.0.0.1:{self._port}/0"

    async def start(self) -> None:
        """Start Redis and wait until it accepts PING requests.

        Raises:
            AssertionError: If the required Redis child cannot start.
        """
        if self._process is not None and self._process.poll() is None:
            return
        self._data_directory.mkdir(exist_ok=True)
        log_path = self._data_directory / "redis.log"
        self._process = await asyncio.to_thread(
            subprocess.Popen,
            [
                self._redis_server_path,
                "--port",
                str(self._port),
                "--bind",
                "127.0.0.1",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(self._data_directory),
                "--logfile",
                str(log_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        redis_client = Redis.from_url(self.url)
        try:
            for _ in range(100):
                if self._process.poll() is not None:
                    break
                try:
                    await redis_client.ping()
                except RedisError:
                    await asyncio.sleep(0.05)
                else:
                    return
            log_output = log_path.read_text() if log_path.exists() else ""
            await self.stop()
            pytest.fail(
                "redis-server could not start for restart fault tests. "
                f"Review {log_path}: {log_output}"
            )
        finally:
            await redis_client.aclose()

    async def stop(self) -> None:
        """Terminate only this fixture's Redis child process."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 5)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 5)

    async def restart(self) -> None:
        """Discard this non-persistent broker state and start the same child again."""
        await self.stop()
        await self.start()


def _test_admin_database_url() -> URL:
    """Load and validate the PostgreSQL admin URL used for isolated databases.

    Returns:
        Async PostgreSQL admin URL.

    """
    raw_url = os.getenv("TEXTIFY_TEST_DATABASE_URL")
    if raw_url is None:
        pytest.fail(
            "TEXTIFY_TEST_DATABASE_URL is required for durable PostgreSQL tests. "
            "Provide a reachable postgresql+asyncpg admin URL whose role has CREATEDB."
        )
        raise AssertionError("pytest.fail must stop fixture setup.")
    try:
        admin_url = make_url(raw_url)
    except ArgumentError as exc:
        pytest.fail("TEXTIFY_TEST_DATABASE_URL must be a valid postgresql+asyncpg URL.")
        raise AssertionError("pytest.fail must stop fixture setup.") from exc
    if admin_url.drivername != "postgresql+asyncpg":
        pytest.fail("TEXTIFY_TEST_DATABASE_URL must use the postgresql+asyncpg driver.")
    return admin_url


async def _create_generated_database(admin_url: URL, database_name: str) -> None:
    """Create one empty generated test database through an admin connection.

    Args:
        admin_url: Validated async PostgreSQL admin URL.
        database_name: Generated database name containing only trusted identifier text.

    Raises:
        SQLAlchemyError: If the server is unreachable or creation is not permitted.
    """
    engine = create_async_engine(
        admin_url,
        hide_parameters=True,
        poolclass=NullPool,
    )
    try:
        async with engine.connect() as connection:
            autocommit_connection = await connection.execution_options(
                isolation_level="AUTOCOMMIT"
            )
            await autocommit_connection.execute(
                text(f'CREATE DATABASE "{database_name}"')
            )
    finally:
        await engine.dispose()


async def _drop_generated_database(admin_url: URL, database_name: str) -> None:
    """Terminate generated-database sessions and drop the database.

    Args:
        admin_url: Validated async PostgreSQL admin URL.
        database_name: Generated database name containing only trusted identifier text.

    Raises:
        SQLAlchemyError: If generated-database teardown cannot complete.
    """
    engine = create_async_engine(
        admin_url,
        hide_parameters=True,
        poolclass=NullPool,
    )
    try:
        async with engine.connect() as connection:
            autocommit_connection = await connection.execution_options(
                isolation_level="AUTOCOMMIT"
            )
            await autocommit_connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity "
                    "WHERE datname = :database_name "
                    "AND pid <> pg_backend_pid()"
                ),
                {"database_name": database_name},
            )
            await autocommit_connection.execute(
                text(f'DROP DATABASE IF EXISTS "{database_name}"')
            )
    finally:
        await engine.dispose()


@pytest.fixture
async def postgresql_database_url() -> AsyncIterator[str]:
    """Create and remove one uniquely named empty PostgreSQL test database.

    Yields:
        Async PostgreSQL URL for an empty generated test database.
    """
    admin_url = _test_admin_database_url()
    database_name = f"{_TEST_DATABASE_PREFIX}{uuid4().hex}"
    try:
        await _create_generated_database(admin_url, database_name)
    except (OSError, SQLAlchemyError) as exc:
        pytest.fail(
            "Unable to create an isolated PostgreSQL test database. "
            "Check TEXTIFY_TEST_DATABASE_URL reachability and CREATEDB permission."
        )
        raise AssertionError from exc

    database_url = admin_url.set(database=database_name).render_as_string(
        hide_password=False
    )
    try:
        yield database_url
    finally:
        try:
            await _drop_generated_database(admin_url, database_name)
        except (OSError, SQLAlchemyError) as exc:
            pytest.fail(
                "Unable to remove the generated PostgreSQL test database. "
                "Check TEXTIFY_TEST_DATABASE_URL reachability and permissions."
            )
            raise AssertionError from exc


@pytest.fixture
async def migrated_postgresql_database_url(
    monkeypatch: pytest.MonkeyPatch,
    postgresql_database_url: str,
) -> AsyncIterator[str]:
    """Upgrade one generated PostgreSQL database to the Alembic head revision.

    Args:
        monkeypatch: Fixture used to supply the migration target configuration.
        postgresql_database_url: Empty generated PostgreSQL database URL.

    Yields:
        Async PostgreSQL URL upgraded through the real Alembic migration.
    """
    monkeypatch.setenv("TEXTIFY_DATABASE_URL", postgresql_database_url)
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    await asyncio.to_thread(command.upgrade, alembic_config, "head")
    yield postgresql_database_url


@pytest.fixture
async def redis_broker_url() -> AsyncIterator[str]:
    """Require a reachable Redis broker reserved for real Celery delivery tests.

    Yields:
        Redis broker URL supplied through ``TEXTIFY_TEST_BROKER_URL``.
    """
    raw_url = os.getenv("TEXTIFY_TEST_BROKER_URL")
    if raw_url is None:
        pytest.fail(
            "TEXTIFY_TEST_BROKER_URL is required for Redis-Celery integration tests."
        )
        raise AssertionError("pytest.fail must stop fixture setup.")
    if urlsplit(raw_url).scheme not in {"redis", "rediss"}:
        pytest.fail("TEXTIFY_TEST_BROKER_URL must use the redis or rediss scheme.")
        raise AssertionError("pytest.fail must stop fixture setup.")

    redis_client = Redis.from_url(raw_url)
    try:
        await redis_client.ping()
    except RedisError as exc:
        pytest.fail("TEXTIFY_TEST_BROKER_URL must identify a reachable Redis broker.")
        raise AssertionError("pytest.fail must stop fixture setup.") from exc
    try:
        yield raw_url
    finally:
        await redis_client.aclose()


@pytest.fixture
async def restartable_redis_broker(
    tmp_path: Path,
) -> AsyncIterator[RestartableRedisBroker]:
    """Start an isolated redis-server child that tests may stop and restart.

    Args:
        tmp_path: Per-test directory for ephemeral Redis logs and disabled persistence.

    Yields:
        Started Redis child controller with a stable isolated broker URL.
    """
    redis_server_path = shutil.which("redis-server")
    if redis_server_path is None:
        pytest.fail(
            "redis-server is required for restart fault tests. "
            "Install the Redis server binary and rerun the test suite."
        )
        raise AssertionError("pytest.fail must stop fixture setup.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved_socket:
        reserved_socket.bind(("127.0.0.1", 0))
        reserved_port = reserved_socket.getsockname()[1]
    broker = RestartableRedisBroker(
        redis_server_path,
        reserved_port,
        tmp_path / "restartable-redis",
    )
    try:
        await broker.start()
        yield broker
    finally:
        await broker.stop()
