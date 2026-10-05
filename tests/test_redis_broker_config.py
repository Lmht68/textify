"""Integration test for the production Redis broker persistence policy."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError


async def _wait_for_broker_startup(
    process: subprocess.Popen[bytes],
    client: Redis,
    log_path: Path,
) -> None:
    """Wait until one configured Redis broker accepts PING.

    Args:
        process: Redis server process started with the production configuration.
        client: Client connected to the configured broker port.
        log_path: Process log used only to diagnose failed startup.

    Raises:
        AssertionError: If Redis exits or does not accept connections promptly.
    """
    for _ in range(100):
        if process.poll() is not None:
            log_output = log_path.read_text() if log_path.exists() else ""
            raise AssertionError(
                "redis-server exited while loading deploy/redis-broker.conf: "
                f"{log_output}"
            )
        try:
            await client.ping()
        except RedisError:
            await asyncio.sleep(0.05)
        else:
            return
    raise AssertionError("redis-server did not accept PING within five seconds.")


async def _stop_broker(process: subprocess.Popen[bytes]) -> None:
    """Stop one broker process without leaving a test child behind.

    Args:
        process: Broker process to terminate.
    """
    if process.poll() is not None:
        return
    process.terminate()
    try:
        await asyncio.to_thread(process.wait, 5)
    except subprocess.TimeoutExpired:
        process.kill()
        await asyncio.to_thread(process.wait, 5)


async def test_redis_broker_config_enables_persistence_without_eviction(
    tmp_path: Path,
) -> None:
    """Run Redis with the deployment policy and inspect its effective settings."""
    redis_server_path = shutil.which("redis-server")
    if redis_server_path is None:
        pytest.fail("redis-server is required to validate the broker policy.")
        raise AssertionError("pytest.fail must stop test setup.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved_socket:
        reserved_socket.bind(("127.0.0.1", 0))
        port = reserved_socket.getsockname()[1]

    broker_config = Path(__file__).parent.parent / "deploy" / "redis-broker.conf"
    log_path = tmp_path / "redis.log"
    process = await asyncio.to_thread(
        subprocess.Popen,
        [
            redis_server_path,
            str(broker_config),
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--dir",
            str(tmp_path),
            "--logfile",
            str(log_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = Redis.from_url(f"redis://127.0.0.1:{port}/0", decode_responses=True)
    settings: dict[str, str] = {}
    try:
        await _wait_for_broker_startup(process, client, log_path)
        settings.update(await client.config_get("appendonly"))
        settings.update(await client.config_get("appendfsync"))
        settings.update(await client.config_get("maxmemory-policy"))
    finally:
        await client.aclose()
        await _stop_broker(process)

    assert settings == {
        "appendonly": "yes",
        "appendfsync": "everysec",
        "maxmemory-policy": "noeviction",
    }
