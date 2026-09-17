"""Shared fixtures for tests that enter the application lifespan."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def migrated_database_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Create one fresh SQLite database upgraded through the real migration.

    Args:
        monkeypatch: Fixture used to supply the migration target configuration.
        tmp_path: Per-test temporary filesystem root.

    Returns:
        Filesystem path of an Alembic-upgraded SQLite database.
    """
    database_path = tmp_path / "textify.sqlite3"
    monkeypatch.setenv("TEXTIFY_DATABASE_PATH", str(database_path))
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(alembic_config, "head")
    return database_path
