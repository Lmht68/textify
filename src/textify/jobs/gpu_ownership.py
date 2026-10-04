"""Physical-GPU ownership for one Transcription Job worker process."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import hashlib
import logging
import os
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

logger = logging.getLogger(__name__)

_LOCK_NAMESPACE = b"textify:gpu-ownership:v1:"
_HOST_LOCK_PREFIX = "textify-gpu-"
_HOST_LOCK_SUFFIX = ".lock"


class GpuOwnershipConflictError(RuntimeError):
    """Raise when another worker already owns the configured physical GPU."""


class GpuModelOwnership:
    """Retain host and PostgreSQL ownership while one loaded model is usable.

    Args:
        engine: PostgreSQL engine used to retain one dedicated advisory-lock session.
        gpu_identity: Stable identifier for one physical GPU.
        lock_directory: Shared host-local directory for deterministic lock files.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        gpu_identity: str,
        lock_directory: Path,
    ) -> None:
        """Initialize an unacquired physical-GPU ownership lifecycle.

        Args:
            engine: PostgreSQL engine used to retain one dedicated advisory-lock
                session.
            gpu_identity: Stable identifier for one physical GPU.
            lock_directory: Shared host-local directory for deterministic lock files.
        """
        digest = hashlib.sha256(_LOCK_NAMESPACE + gpu_identity.encode()).digest()
        self._engine = engine
        self._lock_directory = lock_directory
        self._advisory_lock_key = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=True,
        )
        self._host_lock_name = f"{_HOST_LOCK_PREFIX}{digest.hex()}{_HOST_LOCK_SUFFIX}"
        self._host_lock_descriptor: int | None = None
        self._connection: AsyncConnection | None = None
        self._model_ready = False
        self._connection_lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        """Return whether both locks and the loaded model remain ready for claims."""
        connection = self._connection
        return (
            self._host_lock_descriptor is not None
            and connection is not None
            and not connection.closed
            and self._model_ready
        )

    async def acquire(self) -> None:
        """Acquire host and PostgreSQL ownership before model construction.

        Raises:
            GpuOwnershipConflictError: If another worker owns this GPU identity.
            RuntimeError: If a safe ownership lifecycle cannot be established.
        """
        if self._host_lock_descriptor is not None or self._connection is not None:
            raise RuntimeError("GPU ownership could not be established.")

        try:
            lock_path = self._create_host_lock_path()
        except OSError as exc:
            raise RuntimeError(
                "GPU lock directory must be a writable non-symlink directory."
            ) from exc

        try:
            descriptor = os.open(
                lock_path,
                os.O_NOFOLLOW | os.O_CLOEXEC | os.O_CREAT | os.O_RDWR,
                0o600,
            )
        except OSError as exc:
            raise RuntimeError("GPU ownership could not be established.") from exc

        self._host_lock_descriptor = descriptor
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._release_host_lock()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise GpuOwnershipConflictError(
                    "GPU ownership is already held for the configured identity."
                ) from exc
            raise RuntimeError("GPU ownership could not be established.") from exc

        try:
            connection = await self._engine.connect()
            self._connection = connection
            acquired = (
                await connection.execute(
                    text("SELECT pg_try_advisory_lock(CAST(:lock_key AS BIGINT))"),
                    {"lock_key": self._advisory_lock_key},
                )
            ).scalar_one()
            if acquired is not True:
                raise GpuOwnershipConflictError(
                    "GPU ownership is already held for the configured identity."
                )
            await connection.commit()
        except GpuOwnershipConflictError:
            await self._release_connection_after_failure()
            self._release_host_lock()
            raise
        except (OSError, SQLAlchemyError) as exc:
            await self._release_connection_after_failure()
            self._release_host_lock()
            raise RuntimeError("GPU ownership could not be established.") from exc

    def mark_model_ready(self) -> None:
        """Allow claims after the owned Faster-Whisper model has loaded.

        Raises:
            RuntimeError: If either ownership lock is not currently retained.
        """
        connection = self._connection
        if (
            self._host_lock_descriptor is None
            or connection is None
            or connection.closed
        ):
            raise RuntimeError("GPU ownership could not be established.")
        self._model_ready = True

    def mark_model_not_ready(self) -> None:
        """Prevent new claims while retaining ownership for orderly cleanup."""
        self._model_ready = False

    async def can_claim(self) -> bool:
        """Probe the original PostgreSQL session before durable claim admission.

        Returns:
            ``True`` when host ownership, model readiness, and the retained
            PostgreSQL session are all live. ``False`` otherwise.
        """
        async with self._connection_lock:
            if not self.is_ready:
                return False
            connection = self._connection
            assert connection is not None
            try:
                await connection.execute(text("SELECT 1"))
                await connection.commit()
            except (OSError, SQLAlchemyError):
                self._model_ready = False
                await self._release_connection_after_failure()
                return False
            return True

    async def release(self) -> None:
        """Release PostgreSQL ownership before closing the host lock descriptor."""
        async with self._connection_lock:
            self._model_ready = False
            connection = self._connection
            self._connection = None
            if connection is not None:
                try:
                    unlocked = (
                        await connection.execute(
                            text(
                                "SELECT pg_advisory_unlock(CAST(:lock_key AS BIGINT))"
                            ),
                            {"lock_key": self._advisory_lock_key},
                        )
                    ).scalar_one()
                    if unlocked is not True:
                        await connection.invalidate()
                    else:
                        await connection.commit()
                except (OSError, SQLAlchemyError):
                    await self._invalidate_connection(connection)
                finally:
                    try:
                        await connection.close()
                    except (OSError, SQLAlchemyError):
                        logger.error("GPU ownership connection close failed")
            self._release_host_lock()

    def _create_host_lock_path(self) -> Path:
        """Create and validate the configured physical-GPU lock directory.

        Returns:
            Deterministic lock-file path inside the validated directory.

        Raises:
            OSError: If the directory cannot become a safe lock namespace.
        """
        self._lock_directory.mkdir(parents=True, exist_ok=True)
        resolved_directory = self._lock_directory.resolve(strict=True)
        if (
            self._lock_directory != resolved_directory
            or not resolved_directory.is_dir()
        ):
            raise OSError("GPU lock directory is not a direct directory path.")
        return resolved_directory / self._host_lock_name

    async def _release_connection_after_failure(self) -> None:
        """Invalidate and close a partially acquired retained database session."""
        connection = self._connection
        self._connection = None
        if connection is None:
            return
        await self._invalidate_connection(connection)
        try:
            await connection.close()
        except (OSError, SQLAlchemyError):
            logger.error("GPU ownership connection close failed")

    async def _invalidate_connection(self, connection: AsyncConnection) -> None:
        """Invalidate a failed retained database session without exposing its URL."""
        try:
            await connection.invalidate()
        except (OSError, SQLAlchemyError):
            logger.error("GPU ownership connection invalidation failed")

    def _release_host_lock(self) -> None:
        """Unlock and close the retained descriptor without removing its pathname."""
        descriptor = self._host_lock_descriptor
        self._host_lock_descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
