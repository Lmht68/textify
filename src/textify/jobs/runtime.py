"""Shared in-process lifecycle state for Transcription Job roles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class TranscriptionJobLifecycleState:
    """Coordinate admission and durable-store health across Transcription Job roles."""

    def __init__(self, on_store_unavailable: Callable[[], None]) -> None:
        """Initialize a closed-admission lifecycle state.

        Args:
            on_store_unavailable: Readiness callback invoked on the first durable-store
                failure transition.
        """
        self._on_store_unavailable = on_store_unavailable
        self._lock = asyncio.Lock()
        self._store_available = True
        self._admission_open = False

    @property
    def lock(self) -> asyncio.Lock:
        """Return the lock callers must hold before accessing lifecycle state.

        Returns:
            Shared lifecycle lock.
        """
        return self._lock

    @property
    def store_available(self) -> bool:
        """Return whether durable Transcription Job storage remains available.

        Callers must hold :attr:`lock` before reading this property.

        Returns:
            ``True`` until the first observed durable-store failure.
        """
        return self._store_available

    @property
    def admission_open(self) -> bool:
        """Return whether new Transcription Jobs may be admitted.

        Callers must hold :attr:`lock` before reading this property.

        Returns:
            ``True`` only after runner startup and before shutdown or store failure.
        """
        return self._admission_open

    def open_admission(self) -> None:
        """Permit new Transcription Job admission.

        Callers must hold :attr:`lock` before calling this method.
        """
        self._admission_open = True

    def close_admission(self) -> None:
        """Reject new Transcription Job admission.

        Callers must hold :attr:`lock` before calling this method.
        """
        self._admission_open = False

    def mark_store_unavailable(self) -> bool:
        """Latch a durable-store failure and close admission.

        Callers must hold :attr:`lock` before calling this method.

        Returns:
            ``True`` only when this call performed the first failure transition.
        """
        if not self._store_available:
            return False
        self._store_available = False
        self._admission_open = False
        self._on_store_unavailable()
        return True
