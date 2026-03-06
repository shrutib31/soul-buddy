"""
Unit tests for RedisConfig.

All Redis connections are fully mocked — no real Redis required.
Covers: connect, close, mark_unavailable, properties, and the reconnect loop.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.redis import RedisConfig


# ============================================================================
# Helpers
# ============================================================================

def make_mock_client(ping_raises=None):
    """Return a mock aioredis client whose ping() either succeeds or raises."""
    client = AsyncMock()
    if ping_raises:
        client.ping.side_effect = ping_raises
    else:
        client.ping.return_value = True
    client.aclose = AsyncMock()
    return client


# ============================================================================
# connect()
# ============================================================================

class TestRedisConfigConnect:

    @pytest.mark.asyncio
    async def test_connect_success_sets_available(self):
        config = RedisConfig()
        mock_client = make_mock_client()

        with patch("config.redis.aioredis.from_url", return_value=mock_client):
            result = await config.connect()

        assert result is True
        assert config.is_available is True
        assert config.client is mock_client

    @pytest.mark.asyncio
    async def test_connect_failure_returns_false_and_stays_unavailable(self):
        config = RedisConfig()
        mock_client = make_mock_client(ping_raises=ConnectionRefusedError("refused"))

        with patch("config.redis.aioredis.from_url", return_value=mock_client):
            result = await config.connect()

        assert result is False
        assert config.is_available is False
        assert config.client is None

    @pytest.mark.asyncio
    async def test_connect_success_on_url_with_credentials(self):
        """URL containing @ should not expose credentials in logs."""
        config = RedisConfig()
        config.url = "redis://:secret@localhost:6379"
        mock_client = make_mock_client()

        with patch("config.redis.aioredis.from_url", return_value=mock_client):
            result = await config.connect()

        assert result is True


# ============================================================================
# close()
# ============================================================================

class TestRedisConfigClose:

    @pytest.mark.asyncio
    async def test_close_clears_client_and_availability(self):
        config = RedisConfig()
        mock_client = make_mock_client()
        config._client = mock_client
        config._available = True

        await config.close()

        mock_client.aclose.assert_awaited_once()
        assert config._client is None
        assert config.is_available is False

    @pytest.mark.asyncio
    async def test_close_when_no_client_is_safe(self):
        config = RedisConfig()
        await config.close()  # should not raise


# ============================================================================
# mark_unavailable()
# ============================================================================

class TestRedisConfigMarkUnavailable:

    def test_mark_unavailable_clears_state(self):
        config = RedisConfig()
        config._available = True
        config._client = MagicMock()

        config.mark_unavailable()

        assert config.is_available is False
        assert config._client is None

    def test_mark_unavailable_when_already_unavailable_is_safe(self):
        config = RedisConfig()
        config._available = False
        config._client = None

        config.mark_unavailable()  # should not raise

        assert config.is_available is False


# ============================================================================
# Properties
# ============================================================================

class TestRedisConfigProperties:

    def test_client_returns_none_when_unavailable(self):
        config = RedisConfig()
        config._available = False
        config._client = MagicMock()  # client exists but marked unavailable

        assert config.client is None

    def test_client_returns_client_when_available(self):
        config = RedisConfig()
        mock_client = MagicMock()
        config._available = True
        config._client = mock_client

        assert config.client is mock_client

    def test_is_available_reflects_internal_flag(self):
        config = RedisConfig()
        assert config.is_available is False
        config._available = True
        assert config.is_available is True


# ============================================================================
# Reconnect loop
# ============================================================================

class TestRedisConfigReconnectLoop:

    @pytest.mark.asyncio
    async def test_start_reconnect_loop_creates_background_task(self):
        config = RedisConfig()

        async def sleep_forever(_):
            await asyncio.Event().wait()

        with patch("asyncio.sleep", side_effect=sleep_forever):
            await config.start_reconnect_loop(on_reconnect=MagicMock())

        assert config._reconnect_task is not None
        assert not config._reconnect_task.done()
        config._reconnect_task.cancel()

    @pytest.mark.asyncio
    async def test_stop_reconnect_loop_cancels_task(self):
        config = RedisConfig()

        async def sleep_forever(_):
            await asyncio.Event().wait()

        with patch("asyncio.sleep", side_effect=sleep_forever):
            await config.start_reconnect_loop()

        await config.stop_reconnect_loop()

        assert config._reconnect_task.done()

    @pytest.mark.asyncio
    async def test_stop_reconnect_loop_when_no_task_is_safe(self):
        config = RedisConfig()
        await config.stop_reconnect_loop()  # should not raise

    @pytest.mark.asyncio
    async def test_reconnect_loop_calls_connect_and_callback_when_unavailable(self):
        """When Redis is unavailable the loop should try to reconnect and invoke the callback."""
        config = RedisConfig()
        config._available = False

        mock_client = MagicMock()
        callback = MagicMock()
        config._reconnect_callback = callback

        iteration = 0

        async def fast_sleep(_):
            nonlocal iteration
            iteration += 1
            if iteration > 1:
                raise asyncio.CancelledError()

        async def mock_connect():
            config._available = True
            config._client = mock_client
            return True

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with patch.object(config, "connect", side_effect=mock_connect):
                task = asyncio.create_task(config._reconnect_loop())
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        callback.assert_called_once_with(mock_client)

    @pytest.mark.asyncio
    async def test_reconnect_loop_skips_connect_when_already_available(self):
        """When Redis is already available the loop should not call connect()."""
        config = RedisConfig()
        config._available = True

        iteration = 0

        async def fast_sleep(_):
            nonlocal iteration
            iteration += 1
            if iteration >= 1:
                raise asyncio.CancelledError()

        mock_connect = AsyncMock()
        with patch("asyncio.sleep", side_effect=fast_sleep):
            with patch.object(config, "connect", side_effect=mock_connect):
                task = asyncio.create_task(config._reconnect_loop())
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        mock_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconnect_loop_handles_connect_exception_and_retries(self):
        """If connect() raises unexpectedly the loop should continue, not crash."""
        config = RedisConfig()
        config._available = False

        iteration = 0

        async def fast_sleep(_):
            nonlocal iteration
            iteration += 1
            if iteration > 2:
                raise asyncio.CancelledError()

        connect_calls = 0

        async def flaky_connect():
            nonlocal connect_calls
            connect_calls += 1
            raise OSError("network error")

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with patch.object(config, "connect", side_effect=flaky_connect):
                task = asyncio.create_task(config._reconnect_loop())
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        assert connect_calls >= 1
        assert config.is_available is False
