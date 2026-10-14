"""Shared canonical identifier parsing helpers."""

from __future__ import annotations

from uuid import UUID


def parse_canonical_uuid4(value: object) -> UUID:
    """Parse one canonical lowercase UUIDv4 value.

    Args:
        value: Value received at an application boundary.

    Returns:
        Parsed UUIDv4 value.

    Raises:
        ValueError: If ``value`` is not canonical lowercase UUIDv4 text.
    """
    if not isinstance(value, str):
        raise ValueError("Value must be a canonical lowercase UUIDv4.")  # noqa: TRY004
    try:
        parsed = UUID(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Value must be a canonical lowercase UUIDv4.") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("Value must be a canonical lowercase UUIDv4.")
    return parsed
