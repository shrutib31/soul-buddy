"""
Redis Connection Manager

Mirrors the SQLAlchemy DB pattern used in sqlalchemy_db.py for consistency.
Provides an async Redis connection pool with graceful silent fallback when
Redis is unavailable — callers never need to guard against connection errors.

Environment variables:
    REDIS_URL                — Redis connection URL (default: redis://localhost:6379)
    REDIS_MAX_CONNECTIONS    — Connection pool size (default: 20)
    REDIS_RECONNECT_INTERVAL — Seconds between reconnect attempts (default: 30)
"""

import asyncio
import os
import logging
from typing import Callable, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisConfig:
    """
    Async Redis connection manager with graceful degradation and auto-reconnect.

    If Redis is unreachable at startup, or if it becomes unavailable later,
    all operations will silently return None / be skipped by CacheService.
    This ensures Redis is strictly optional — the app falls back to the DB.

    The background reconnect loop periodically retries the connection when
    Redis is marked unavailable and calls ``on_reconnect(client)`` once it
    comes back, so CacheService resumes caching automatically.
    """

    def __init__(self):
        self.url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        self._reconnect_interval: int = int(os.getenv("REDIS_RECONNECT_INTERVAL", "30"))
        self._client: Optional[aioredis.Redis] = None
        self._available: bool = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_callback: Optional[Callable] = None

    async def connect(self) -> bool:
        """
        Create the connection pool and verify connectivity with PING.

        Returns:
            True if Redis is reachable, False otherwise (fail-silent).
        """
        try:
            self._client = aioredis.from_url(
                self.url,
                max_connections=self.max_connections,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await self._client.ping()
            self._available = True
            # Log host only, not credentials
            safe_url = self.url.split("@")[-1] if "@" in self.url else self.url
            logger.info("✅ Redis connected: %s", safe_url)
            return True
        except Exception as exc:
            logger.warning(
                "⚠️  Redis unavailable (%s) — caching disabled, falling back to DB",
                exc,
            )
            self._available = False
            self._client = None
            return False

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
            self._available = False
            logger.debug("✅ Redis connection closed")

    def mark_unavailable(self) -> None:
        """
        Mark Redis as unavailable after a mid-operation connection error.

        Called by CacheService when it catches a connection-level exception.
        The background reconnect loop will restore the connection when Redis
        comes back up.
        """
        if self._available:
            logger.warning(
                "⚠️  Redis marked unavailable mid-operation — DB fallback active until reconnect"
            )
        self._available = False
        self._client = None

    async def start_reconnect_loop(self, on_reconnect: Optional[Callable] = None) -> None:
        """
        Start a background task that periodically retries the Redis connection
        when it is unavailable.

        Args:
            on_reconnect: Callable invoked with the live client once Redis
                          reconnects — typically ``cache_service.set_client``.
        """
        self._reconnect_callback = on_reconnect
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        logger.debug(
            "Redis reconnect loop started (interval=%ds)", self._reconnect_interval
        )

    async def _reconnect_loop(self) -> None:
        """Background loop: sleep → retry connection whenever Redis is down."""
        while True:
            try:
                await asyncio.sleep(self._reconnect_interval)
                if not self._available:
                    logger.info(
                        "Redis reconnect loop: attempting to reconnect (interval=%ds)...",
                        self._reconnect_interval,
                    )
                    connected = await self.connect()
                    if connected and self._reconnect_callback is not None:
                        logger.info("✅ Redis reconnected — re-enabling cache")
                        self._reconnect_callback(self.client)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Redis reconnect loop error (will retry): %s", exc)

    async def stop_reconnect_loop(self) -> None:
        """Cancel the reconnect background task and wait for it to finish."""
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            logger.debug("Redis reconnect loop stopped")

    @property
    def client(self) -> Optional[aioredis.Redis]:
        """Return the Redis client, or None if unavailable."""
        return self._client if self._available else None

    @property
    def is_available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# Global instance — initialised in server.py lifespan, used by CacheService
# ---------------------------------------------------------------------------
redis_config: Optional[RedisConfig] = None
