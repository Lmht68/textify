"""Readiness storage and lifecycle adapters for processing service roles."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Interval,
    PrimaryKeyConstraint,
    String,
    Table,
    bindparam,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from textify.jobs.config import JobDispatchConfig
from textify.jobs.database import verify_application_database
from textify.jobs.repository import metadata

logger = logging.getLogger(__name__)


class ServiceRole(StrEnum):
    """Identify a processing role required for application readiness."""

    RECONCILER = "reconciler"
    GPU_WORKER = "gpu_worker"


service_heartbeat = Table(
    "service_heartbeat",
    metadata,
    Column("service_role", String(32), nullable=False),
    Column("instance_id", PostgreSQLUUID(as_uuid=True), nullable=False),
    Column("last_ready_at", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint(
        "service_role",
        "instance_id",
        name="service_heartbeat_pkey",
    ),
    CheckConstraint(
        "service_role IN ('reconciler', 'gpu_worker')",
        name="service_heartbeat_role_check",
    ),
    CheckConstraint(
        "(get_byte(uuid_send(instance_id), 6) & 240) = 64 "
        "AND (get_byte(uuid_send(instance_id), 8) & 192) = 128",
        name="service_heartbeat_instance_id_uuid4_check",
    ),
)

Index(
    "service_heartbeat_role_last_ready_at_idx",
    service_heartbeat.c.service_role,
    service_heartbeat.c.last_ready_at.desc(),
)


class ServiceReadinessUnavailableError(RuntimeError):
    """Indicate that service-readiness storage cannot be used safely."""


class ServiceHeartbeatRepository(Protocol):
    """Persist ready-state heartbeats for independently deployed services."""

    async def record_ready(self, role: ServiceRole, instance_id: UUID) -> None:
        """Record one current ready heartbeat using the storage clock.

        Args:
            role: Processing role proven ready by the caller.
            instance_id: UUIDv4 identifying this process instance.

        Raises:
            ServiceReadinessUnavailableError: If the heartbeat cannot be stored.
        """
        ...

    async def delete(self, role: ServiceRole, instance_id: UUID) -> None:
        """Remove the exact ready heartbeat owned by one process.

        Args:
            role: Processing role represented by the heartbeat.
            instance_id: UUIDv4 identifying the process instance.

        Raises:
            ServiceReadinessUnavailableError: If the heartbeat cannot be removed.
        """
        ...

    async def fresh_roles(self, freshness: timedelta) -> frozenset[ServiceRole]:
        """Read every role with at least one current server-timed heartbeat.

        Args:
            freshness: Maximum permitted heartbeat age.

        Returns:
            Roles represented by one or more fresh process instances.

        Raises:
            ServiceReadinessUnavailableError: If freshness cannot be queried.
            ValueError: If ``freshness`` is not positive.
        """
        ...


class PostgresServiceHeartbeatRepository:
    """Store service-role readiness against PostgreSQL server time."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize storage with the application PostgreSQL engine.

        Args:
            engine: Application connection pool that owns readiness storage.
        """
        self._engine = engine

    async def record_ready(self, role: ServiceRole, instance_id: UUID) -> None:
        """Upsert one process readiness timestamp using PostgreSQL server time.

        Args:
            role: Processing role proven ready by the caller.
            instance_id: UUIDv4 identifying this process instance.

        Raises:
            ServiceReadinessUnavailableError: If the heartbeat cannot be stored.
            ValueError: If a role or process identity is invalid.
        """
        _require_service_role(role)
        _require_uuid4(instance_id)
        statement = postgresql_insert(service_heartbeat).values(
            service_role=role.value,
            instance_id=instance_id,
            last_ready_at=func.now(),
        )
        statement = statement.on_conflict_do_update(
            index_elements=(
                service_heartbeat.c.service_role,
                service_heartbeat.c.instance_id,
            ),
            set_={"last_ready_at": func.now()},
        )
        try:
            async with self._engine.begin() as connection:
                await connection.execute(statement)
        except (OSError, SQLAlchemyError) as exc:
            raise ServiceReadinessUnavailableError() from exc

    async def delete(self, role: ServiceRole, instance_id: UUID) -> None:
        """Delete exactly one process readiness heartbeat.

        Args:
            role: Processing role represented by the heartbeat.
            instance_id: UUIDv4 identifying the process instance.

        Raises:
            ServiceReadinessUnavailableError: If the heartbeat cannot be removed.
            ValueError: If a role or process identity is invalid.
        """
        _require_service_role(role)
        _require_uuid4(instance_id)
        statement = delete(service_heartbeat).where(
            service_heartbeat.c.service_role == role.value,
            service_heartbeat.c.instance_id == instance_id,
        )
        try:
            async with self._engine.begin() as connection:
                await connection.execute(statement)
        except (OSError, SQLAlchemyError) as exc:
            raise ServiceReadinessUnavailableError() from exc

    async def fresh_roles(self, freshness: timedelta) -> frozenset[ServiceRole]:
        """Read roles that have a heartbeat newer than a server-time cutoff.

        Args:
            freshness: Maximum permitted heartbeat age.

        Returns:
            Roles represented by at least one fresh process instance.

        Raises:
            ServiceReadinessUnavailableError: If freshness cannot be queried.
            ValueError: If ``freshness`` is not positive.
        """
        _require_positive_duration(freshness, "freshness")
        cutoff = func.now() - bindparam(
            "freshness",
            freshness,
            type_=Interval(),
        )
        statement = select(service_heartbeat.c.service_role).where(
            service_heartbeat.c.last_ready_at >= cutoff
        )
        try:
            async with self._engine.connect() as connection:
                roles = (await connection.scalars(statement)).all()
        except (OSError, SQLAlchemyError) as exc:
            raise ServiceReadinessUnavailableError() from exc
        return frozenset(ServiceRole(role) for role in roles)


ReadinessProbe = Callable[[], Awaitable[bool]]


class ServiceHeartbeatReporter:
    """Own one process heartbeat and optional prerequisite-checked refreshes."""

    def __init__(
        self,
        repository: ServiceHeartbeatRepository,
        role: ServiceRole,
        *,
        refresh_interval: timedelta | None = None,
        readiness_probe: ReadinessProbe | None = None,
    ) -> None:
        """Initialize one process-instance heartbeat lifecycle.

        Args:
            repository: PostgreSQL readiness storage boundary.
            role: Processing role represented by this process.
            refresh_interval: Optional interval for automatic refreshing.
            readiness_probe: Required current-prerequisite probe when refreshing.

        Raises:
            ValueError: If automatic refresh configuration is incomplete or invalid.
        """
        _require_service_role(role)
        if (refresh_interval is None) != (readiness_probe is None):
            raise ValueError(
                "refresh_interval and readiness_probe must be supplied together."
            )
        if refresh_interval is not None:
            _require_positive_duration(refresh_interval, "refresh_interval")
        self._repository = repository
        self._role = role
        self._refresh_interval = refresh_interval
        self._readiness_probe = readiness_probe
        self._instance_id = uuid4()
        self._started = False
        self._row_may_exist = False
        self._periodic_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Write the first heartbeat only after current prerequisites pass.

        Raises:
            RuntimeError: If prerequisites are unavailable or already started.
            ServiceReadinessUnavailableError: If PostgreSQL cannot store readiness.
        """
        if self._started:
            raise RuntimeError("Service heartbeat reporter has already started.")
        if self._readiness_probe is not None and not await self._readiness_probe():
            raise RuntimeError("Service readiness prerequisites are unavailable.")
        await self._repository.record_ready(self._role, self._instance_id)
        self._row_may_exist = True
        self._started = True
        if self._refresh_interval is not None:
            self._periodic_task = asyncio.create_task(self._refresh_periodically())

    async def pulse(self) -> None:
        """Refresh readiness after a caller-proven successful service cycle.

        Raises:
            ServiceReadinessUnavailableError: If PostgreSQL cannot store readiness.
        """
        if not self._started:
            return
        await self._pulse_once()

    async def stop(self) -> None:
        """Stop automatic refreshes and remove this process heartbeat.

        Later deletion failures are logged so shutdown can continue.
        """
        periodic_task = self._periodic_task
        self._periodic_task = None
        self._started = False
        if periodic_task is not None:
            periodic_task.cancel()
            try:
                await periodic_task
            except asyncio.CancelledError:
                pass
        await self._remove_row(log_failures=True)

    async def _refresh_periodically(self) -> None:
        """Refresh readiness until cancellation or a prerequisite becomes false."""
        assert self._refresh_interval is not None
        while True:
            await asyncio.sleep(self._refresh_interval.total_seconds())
            try:
                if not await self._pulse_once():
                    return
            except (OSError, RuntimeError, ServiceReadinessUnavailableError):
                logger.error("service readiness heartbeat refresh failed")

    async def _pulse_once(self) -> bool:
        """Write once when prerequisites pass and remove the row otherwise."""
        if self._readiness_probe is not None and not await self._readiness_probe():
            await self._remove_row(log_failures=True)
            return False
        await self._repository.record_ready(self._role, self._instance_id)
        self._row_may_exist = True
        return True

    async def _remove_row(self, *, log_failures: bool) -> None:
        """Remove this process row without masking ordinary shutdown failures."""
        if not self._row_may_exist:
            return
        try:
            await self._repository.delete(self._role, self._instance_id)
        except ServiceReadinessUnavailableError:
            if log_failures:
                logger.error("service readiness heartbeat deletion failed")
            return
        self._row_may_exist = False


def _require_service_role(value: ServiceRole) -> None:
    """Reject a value that is not one configured processing role."""
    if not isinstance(value, ServiceRole):
        raise TypeError("role must be a ServiceRole.")


def _require_uuid4(value: UUID) -> None:
    """Reject a non-v4 process identity before reaching PostgreSQL."""
    if not isinstance(value, UUID) or value.version != 4:
        raise ValueError("instance_id must be a UUIDv4.")


def _require_positive_duration(value: timedelta, name: str) -> None:
    """Reject a duration that cannot safely schedule readiness work."""
    if not isinstance(value, timedelta) or value <= timedelta():
        raise ValueError(f"{name} must be a positive timedelta.")


class BrokerReadiness(Protocol):
    """Probe the configured Celery broker without publishing Job Dispatches."""

    async def is_ready(self) -> bool:
        """Return whether the delivery broker currently responds to PING."""
        ...


class RedisBrokerReadiness:
    """Probe one configured Redis broker through an owner-managed client."""

    def __init__(self, dispatch_config: JobDispatchConfig) -> None:
        """Create a Redis client exclusively for the configured Celery broker.

        Args:
            dispatch_config: Required Job Dispatch broker configuration.
        """
        self._client = Redis.from_url(str(dispatch_config.broker_url))

    async def is_ready(self) -> bool:
        """Return whether Redis responds within the bounded readiness probe.

        Returns:
            ``True`` when the configured broker responds to PING within one second.
        """
        try:
            async with asyncio.timeout(1):
                return bool(await self._client.ping())
        except (TimeoutError, OSError, RedisError):
            logger.error("broker readiness probe failed")
            return False

    async def aclose(self) -> None:
        """Close the Redis client owned by this broker readiness adapter."""
        await self._client.aclose()


class ApplicationReadiness:
    """Aggregate PostgreSQL, broker, and processing-role readiness checks."""

    def __init__(
        self,
        engine: AsyncEngine,
        heartbeat_repository: ServiceHeartbeatRepository,
        broker_readiness: BrokerReadiness,
        freshness: timedelta,
    ) -> None:
        """Initialize dependency readiness checks for one API process.

        Args:
            engine: Application PostgreSQL engine to verify for each health request.
            heartbeat_repository: PostgreSQL server-time heartbeat reader.
            broker_readiness: Redis broker PING boundary.
            freshness: Maximum permitted age of each service heartbeat.

        Raises:
            ValueError: If ``freshness`` is not positive.
        """
        _require_positive_duration(freshness, "freshness")
        self._engine = engine
        self._heartbeat_repository = heartbeat_repository
        self._broker_readiness = broker_readiness
        self._freshness = freshness

    async def is_ready(self) -> bool:
        """Return whether every dependency can process committed work.

        Returns:
            ``True`` only when PostgreSQL, Redis, and both required service roles
            are currently ready.
        """
        try:
            await verify_application_database(self._engine)
            if not await self._broker_readiness.is_ready():
                return False
            fresh_roles = await self._heartbeat_repository.fresh_roles(self._freshness)
        except (
            OSError,
            RedisError,
            RuntimeError,
            ServiceReadinessUnavailableError,
        ):
            return False
        return {
            ServiceRole.RECONCILER,
            ServiceRole.GPU_WORKER,
        }.issubset(fresh_roles)
