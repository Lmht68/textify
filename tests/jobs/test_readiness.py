"""Tests for PostgreSQL-backed service readiness reporting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import cast
from uuid import UUID, uuid4

import pytest
from redis.exceptions import RedisError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine


class RecordingHeartbeatRepository:
    """Record service-heartbeat writes without PostgreSQL."""

    def __init__(self) -> None:
        """Initialize observable heartbeat operations."""
        self.recorded: list[tuple[object, UUID]] = []
        self.deleted: list[tuple[object, UUID]] = []
        self.first_recorded = asyncio.Event()
        self.periodic_recorded = asyncio.Event()

    async def record_ready(self, role: object, instance_id: UUID) -> None:
        """Record one ready heartbeat."""
        self.recorded.append((role, instance_id))
        if len(self.recorded) == 1:
            self.first_recorded.set()
        else:
            self.periodic_recorded.set()

    async def delete(self, role: object, instance_id: UUID) -> None:
        """Record removal of one ready heartbeat."""
        self.deleted.append((role, instance_id))

    async def fresh_roles(self, _freshness: timedelta) -> frozenset[object]:
        """Return no roles because this fake is only used by reporter tests."""
        return frozenset()


class FixedHeartbeatRepository:
    """Return a fixed role set for aggregate readiness tests."""

    def __init__(self, roles: frozenset[object]) -> None:
        """Initialize roles returned by the storage seam."""
        self._roles = roles
        self.freshness: timedelta | None = None
        self.unavailable = False

    async def record_ready(self, _role: object, _instance_id: UUID) -> None:
        """Reject unused writes in aggregate readiness tests."""
        raise AssertionError("Aggregate readiness must not write service heartbeats.")

    async def delete(self, _role: object, _instance_id: UUID) -> None:
        """Reject unused deletes in aggregate readiness tests."""
        raise AssertionError("Aggregate readiness must not delete service heartbeats.")

    async def fresh_roles(self, freshness: timedelta) -> frozenset[object]:
        """Return configured roles or simulate unavailable readiness storage."""
        self.freshness = freshness
        if self.unavailable:
            from textify.jobs.readiness import ServiceReadinessUnavailableError

            raise ServiceReadinessUnavailableError()
        return self._roles


class FixedBrokerReadiness:
    """Return a fixed broker readiness result without network I/O."""

    def __init__(self, ready: bool) -> None:
        """Initialize one broker result."""
        self.ready = ready
        self.calls = 0

    async def is_ready(self) -> bool:
        """Return the configured broker readiness result."""
        self.calls += 1
        return self.ready


class RecordingRedisClient:
    """Record readiness PING and close calls from the Redis adapter."""

    def __init__(self) -> None:
        """Initialize a successful fake Redis client."""
        self.ping_error: RedisError | OSError | None = None
        self.ping_calls = 0
        self.close_calls = 0

    async def ping(self) -> bool:
        """Record PING and optionally simulate a Redis client failure."""
        self.ping_calls += 1
        if self.ping_error is not None:
            raise self.ping_error
        return True

    async def aclose(self) -> None:
        """Record one owned-client shutdown."""
        self.close_calls += 1


@pytest.fixture
async def heartbeat_repository(
    migrated_postgresql_database_url: str,
) -> AsyncEngine:
    """Create an engine for PostgreSQL service-heartbeat tests."""
    from textify.jobs.database import create_application_engine

    engine = create_application_engine(migrated_postgresql_database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


async def test_postgres_heartbeat_repository_upserts_each_role_instance_and_deletes_exact_row(
    heartbeat_repository: AsyncEngine,
) -> None:
    """Keep one current server-timed readiness row per service process."""
    from textify.jobs.readiness import (
        PostgresServiceHeartbeatRepository,
        ServiceRole,
        service_heartbeat,
    )

    repository = PostgresServiceHeartbeatRepository(heartbeat_repository)
    reconciler_instance = uuid4()
    first_worker_instance = uuid4()
    second_worker_instance = uuid4()

    await repository.record_ready(ServiceRole.RECONCILER, reconciler_instance)
    await repository.record_ready(ServiceRole.GPU_WORKER, first_worker_instance)
    await repository.record_ready(ServiceRole.GPU_WORKER, second_worker_instance)
    await repository.record_ready(ServiceRole.GPU_WORKER, first_worker_instance)

    async with heartbeat_repository.connect() as connection:
        rows = (
            await connection.execute(
                select(
                    service_heartbeat.c.service_role,
                    service_heartbeat.c.instance_id,
                    service_heartbeat.c.last_ready_at,
                ).order_by(
                    service_heartbeat.c.service_role,
                    service_heartbeat.c.instance_id,
                )
            )
        ).all()

    assert {(row.service_role, row.instance_id) for row in rows} == {
        (ServiceRole.GPU_WORKER.value, first_worker_instance),
        (ServiceRole.GPU_WORKER.value, second_worker_instance),
        (ServiceRole.RECONCILER.value, reconciler_instance),
    }
    assert all(row.last_ready_at.tzinfo is not None for row in rows)
    assert await repository.fresh_roles(timedelta(seconds=30)) == frozenset(
        {ServiceRole.RECONCILER, ServiceRole.GPU_WORKER}
    )

    await repository.delete(ServiceRole.GPU_WORKER, first_worker_instance)

    async with heartbeat_repository.connect() as connection:
        remaining_instances = set(
            (
                await connection.scalars(
                    select(service_heartbeat.c.instance_id).order_by(
                        service_heartbeat.c.instance_id
                    )
                )
            ).all()
        )
    assert remaining_instances == {reconciler_instance, second_worker_instance}


async def test_postgres_heartbeat_repository_uses_database_time_for_freshness(
    heartbeat_repository: AsyncEngine,
) -> None:
    """Exclude stale rows without relying on any process-local clock."""
    from textify.jobs.readiness import (
        PostgresServiceHeartbeatRepository,
        ServiceRole,
        service_heartbeat,
    )

    repository = PostgresServiceHeartbeatRepository(heartbeat_repository)
    fresh_instance = uuid4()
    stale_instance = uuid4()
    async with heartbeat_repository.begin() as connection:
        await connection.execute(
            service_heartbeat.insert().values(
                service_role=ServiceRole.RECONCILER.value,
                instance_id=fresh_instance,
                last_ready_at=func.now(),
            )
        )
        await connection.execute(
            service_heartbeat.insert().values(
                service_role=ServiceRole.GPU_WORKER.value,
                instance_id=stale_instance,
                last_ready_at=func.now() - timedelta(seconds=31),
            )
        )

    assert await repository.fresh_roles(timedelta(seconds=30)) == frozenset(
        {ServiceRole.RECONCILER}
    )


async def test_heartbeat_reporter_writes_periodically_and_removes_its_exact_instance() -> (
    None
):
    """Report one ready process until its readiness probe becomes false."""
    from textify.jobs.readiness import ServiceHeartbeatReporter, ServiceRole

    repository = RecordingHeartbeatRepository()
    probe_ready = True

    async def readiness_probe() -> bool:
        """Return the current simulated worker readiness state."""
        return probe_ready

    reporter = ServiceHeartbeatReporter(
        repository,  # type: ignore[arg-type]
        ServiceRole.GPU_WORKER,
        refresh_interval=timedelta(milliseconds=1),
        readiness_probe=readiness_probe,
    )

    await reporter.start()
    await asyncio.wait_for(repository.first_recorded.wait(), timeout=1)
    await asyncio.wait_for(repository.periodic_recorded.wait(), timeout=1)
    probe_ready = False

    for _ in range(100):
        if repository.deleted:
            break
        await asyncio.sleep(0.001)

    await reporter.stop()

    assert len(repository.recorded) >= 2
    assert repository.deleted
    role, instance_id = repository.deleted[0]
    assert role is ServiceRole.GPU_WORKER
    assert instance_id.version == 4


async def test_heartbeat_reporter_rejects_unready_initial_prerequisites() -> None:
    """Do not insert a row for a process that cannot serve its role."""
    from textify.jobs.readiness import ServiceHeartbeatReporter, ServiceRole

    repository = RecordingHeartbeatRepository()

    async def unavailable_probe() -> bool:
        """Declare startup prerequisites unavailable."""
        return False

    reporter = ServiceHeartbeatReporter(
        repository,  # type: ignore[arg-type]
        ServiceRole.GPU_WORKER,
        refresh_interval=timedelta(seconds=1),
        readiness_probe=unavailable_probe,
    )

    with pytest.raises(
        RuntimeError, match="Service readiness prerequisites are unavailable"
    ):
        await reporter.start()

    await reporter.stop()
    assert repository.recorded == []
    assert repository.deleted == []


@pytest.mark.parametrize(
    ("refresh_interval", "readiness_probe"),
    (
        (timedelta(seconds=1), None),
        (None, lambda: _true()),
        (timedelta(), lambda: _true()),
    ),
)
def test_heartbeat_reporter_requires_a_positive_matched_periodic_policy(
    refresh_interval: timedelta | None,
    readiness_probe: Callable[[], Awaitable[bool]] | None,
) -> None:
    """Reject incomplete or nonpositive periodic refresh configuration."""
    from textify.jobs.readiness import ServiceHeartbeatReporter, ServiceRole

    with pytest.raises(ValueError):
        ServiceHeartbeatReporter(
            RecordingHeartbeatRepository(),  # type: ignore[arg-type]
            ServiceRole.RECONCILER,
            refresh_interval=refresh_interval,
            readiness_probe=readiness_probe,
        )


async def test_redis_broker_readiness_uses_one_broker_client_and_hides_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe only the configured Celery broker with PING and close it once."""
    from textify.jobs import readiness
    from textify.jobs.config import JobDispatchConfig

    client = RecordingRedisClient()
    broker_url = "redis://broker.example:6379/15"
    monkeypatch.setattr(readiness.Redis, "from_url", lambda _url: client)
    broker = readiness.RedisBrokerReadiness(
        JobDispatchConfig(
            broker_url=broker_url,  # type: ignore[arg-type]
            _env_file=None,  # type: ignore[call-arg]
        )
    )

    assert await broker.is_ready() is True
    client.ping_error = RedisError("connection refused")
    assert await broker.is_ready() is False
    await broker.aclose()

    assert client.ping_calls == 2
    assert client.close_calls == 1


@pytest.mark.parametrize(
    "roles",
    (
        frozenset(),
        frozenset({"reconciler"}),
        frozenset({"gpu_worker"}),
    ),
)
async def test_application_readiness_requires_every_processing_role(
    monkeypatch: pytest.MonkeyPatch,
    roles: frozenset[object],
) -> None:
    """Reject aggregate health when either processing role lacks a fresh row."""
    from textify.jobs import readiness

    async def verify_database(_engine: AsyncEngine) -> None:
        """Treat the isolated database prerequisite as available."""

    monkeypatch.setattr(readiness, "verify_application_database", verify_database)
    repository = FixedHeartbeatRepository(roles)
    broker = FixedBrokerReadiness(True)
    checker = readiness.ApplicationReadiness(
        cast(AsyncEngine, object()),
        repository,  # type: ignore[arg-type]
        broker,  # type: ignore[arg-type]
        timedelta(seconds=30),
    )

    assert await checker.is_ready() is False
    assert broker.calls == 1
    assert repository.freshness == timedelta(seconds=30)


@pytest.mark.parametrize(
    ("broker_ready", "expected_ready"),
    (
        (True, True),
        (False, False),
    ),
)
async def test_application_readiness_checks_broker_after_database(
    monkeypatch: pytest.MonkeyPatch,
    broker_ready: bool,
    expected_ready: bool,
) -> None:
    """Require a responding broker after PostgreSQL and both role heartbeats."""
    from textify.jobs import readiness
    from textify.jobs.readiness import ServiceRole

    async def verify_database(_engine: AsyncEngine) -> None:
        """Treat the isolated database prerequisite as available."""

    monkeypatch.setattr(readiness, "verify_application_database", verify_database)
    repository = FixedHeartbeatRepository(
        frozenset({ServiceRole.RECONCILER, ServiceRole.GPU_WORKER})
    )
    broker = FixedBrokerReadiness(broker_ready)
    checker = readiness.ApplicationReadiness(
        cast(AsyncEngine, object()),
        repository,  # type: ignore[arg-type]
        broker,  # type: ignore[arg-type]
        timedelta(seconds=30),
    )

    assert await checker.is_ready() is expected_ready
    assert broker.calls == 1
    assert repository.freshness == (timedelta(seconds=30) if broker_ready else None)


async def test_application_readiness_returns_false_for_private_dependency_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep aggregate readiness false without exposing PostgreSQL failure details."""
    from textify.jobs import readiness
    from textify.jobs.readiness import ServiceRole

    async def unavailable_database(_engine: AsyncEngine) -> None:
        """Simulate unavailable or stale PostgreSQL readiness validation."""
        raise RuntimeError("database details must not escape")

    monkeypatch.setattr(readiness, "verify_application_database", unavailable_database)
    repository = FixedHeartbeatRepository(
        frozenset({ServiceRole.RECONCILER, ServiceRole.GPU_WORKER})
    )
    broker = FixedBrokerReadiness(True)
    checker = readiness.ApplicationReadiness(
        cast(AsyncEngine, object()),
        repository,  # type: ignore[arg-type]
        broker,  # type: ignore[arg-type]
        timedelta(seconds=30),
    )

    assert await checker.is_ready() is False
    assert broker.calls == 0


async def _true() -> bool:
    """Return a ready test probe result."""
    return True
