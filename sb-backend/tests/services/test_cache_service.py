"""
Unit tests for CacheService.

All Redis interactions are fully mocked — no real Redis required.
Covers: lifecycle, connection-error handling, low-level primitives,
key builders, and the public API methods.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

from services.cache_service import CacheService, CONV_HISTORY_MAX_TURNS


# ============================================================================
# Helpers
# ============================================================================

def make_service_with_client():
    """Return a (CacheService, mock_redis_client, mock_redis_config) tuple."""
    service = CacheService()
    mock_client = AsyncMock()
    mock_redis_config = MagicMock()
    service.set_client(mock_client)
    service.set_redis_config(mock_redis_config)
    return service, mock_client, mock_redis_config


# ============================================================================
# Lifecycle
# ============================================================================

class TestCacheServiceLifecycle:

    def test_is_available_false_initially(self):
        service = CacheService()
        assert service.is_available is False

    def test_set_client_makes_service_available(self):
        service = CacheService()
        service.set_client(AsyncMock())
        assert service.is_available is True

    def test_set_client_none_makes_service_unavailable(self):
        service = CacheService()
        service.set_client(AsyncMock())
        service.set_client(None)
        assert service.is_available is False

    def test_set_redis_config_stores_reference(self):
        service = CacheService()
        mock_config = MagicMock()
        service.set_redis_config(mock_config)
        assert service._redis_config is mock_config


# ============================================================================
# _handle_connection_error()
# ============================================================================

class TestCacheServiceHandleConnectionError:

    def test_nulls_client_reference(self):
        service, _, mock_config = make_service_with_client()
        assert service._redis is not None

        service._handle_connection_error(RedisConnectionError("down"), "GET key=x")

        assert service._redis is None

    def test_calls_mark_unavailable_on_redis_config(self):
        service, _, mock_config = make_service_with_client()

        service._handle_connection_error(RedisConnectionError("down"), "GET key=x")

        mock_config.mark_unavailable.assert_called_once()

    def test_safe_when_redis_config_not_set(self):
        service = CacheService()
        service.set_client(AsyncMock())
        # no set_redis_config() call

        service._handle_connection_error(RedisConnectionError("down"), "GET key=x")  # should not raise

        assert service._redis is None


# ============================================================================
# _get()
# ============================================================================

class TestCacheServiceGet:

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self):
        service = CacheService()
        result = await service._get("any-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        service, mock_client, _ = make_service_with_client()
        mock_client.get.return_value = None

        result = await service._get("missing-key")

        assert result is None
        mock_client.get.assert_awaited_once_with("missing-key")

    @pytest.mark.asyncio
    async def test_cache_hit_returns_decoded_value(self):
        service, mock_client, _ = make_service_with_client()
        payload = {"name": "Alice", "score": 42}
        mock_client.get.return_value = json.dumps(payload)

        result = await service._get("user:1:profile")

        assert result == payload

    @pytest.mark.asyncio
    async def test_connection_error_marks_unavailable_and_returns_none(self):
        service, mock_client, mock_config = make_service_with_client()
        mock_client.get.side_effect = RedisConnectionError("connection refused")

        result = await service._get("key")

        assert result is None
        assert service._redis is None
        mock_config.mark_unavailable.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_error_marks_unavailable_and_returns_none(self):
        service, mock_client, mock_config = make_service_with_client()
        mock_client.get.side_effect = RedisTimeoutError("timed out")

        result = await service._get("key")

        assert result is None
        mock_config.mark_unavailable.assert_called_once()

    @pytest.mark.asyncio
    async def test_generic_error_returns_none_without_marking_unavailable(self):
        service, mock_client, mock_config = make_service_with_client()
        mock_client.get.side_effect = ValueError("unexpected error")

        result = await service._get("key")

        assert result is None
        mock_config.mark_unavailable.assert_not_called()


# ============================================================================
# _set()
# ============================================================================

class TestCacheServiceSet:

    @pytest.mark.asyncio
    async def test_skips_when_no_client(self):
        service = CacheService()
        await service._set("key", {"data": 1}, 60)  # should not raise

    @pytest.mark.asyncio
    async def test_serializes_and_stores_with_ttl(self):
        service, mock_client, _ = make_service_with_client()
        data = {"x": 1}

        await service._set("my-key", data, 300)

        mock_client.setex.assert_awaited_once_with("my-key", 300, json.dumps(data, default=str))

    @pytest.mark.asyncio
    async def test_connection_error_marks_unavailable(self):
        service, mock_client, mock_config = make_service_with_client()
        mock_client.setex.side_effect = RedisConnectionError("broken pipe")

        await service._set("key", {}, 60)

        assert service._redis is None
        mock_config.mark_unavailable.assert_called_once()

    @pytest.mark.asyncio
    async def test_generic_error_does_not_mark_unavailable(self):
        service, mock_client, mock_config = make_service_with_client()
        mock_client.setex.side_effect = TypeError("bad value")

        await service._set("key", object(), 60)

        mock_config.mark_unavailable.assert_not_called()


# ============================================================================
# _delete()
# ============================================================================

class TestCacheServiceDelete:

    @pytest.mark.asyncio
    async def test_skips_when_no_client(self):
        service = CacheService()
        await service._delete("key")  # should not raise

    @pytest.mark.asyncio
    async def test_calls_redis_delete(self):
        service, mock_client, _ = make_service_with_client()

        await service._delete("conv:abc:history")

        mock_client.delete.assert_awaited_once_with("conv:abc:history")

    @pytest.mark.asyncio
    async def test_connection_error_marks_unavailable(self):
        service, mock_client, mock_config = make_service_with_client()
        mock_client.delete.side_effect = RedisConnectionError("gone")

        await service._delete("key")

        mock_config.mark_unavailable.assert_called_once()


# ============================================================================
# _delete_pattern()
# ============================================================================

class TestCacheServiceDeletePattern:

    @pytest.mark.asyncio
    async def test_skips_when_no_client(self):
        service = CacheService()
        await service._delete_pattern("user:*")  # should not raise

    @pytest.mark.asyncio
    async def test_single_page_scan_deletes_all_keys(self):
        service, mock_client, _ = make_service_with_client()
        mock_client.scan.return_value = (0, ["key1", "key2"])

        await service._delete_pattern("user:1:*")

        mock_client.scan.assert_awaited_once()
        mock_client.delete.assert_awaited_once_with("key1", "key2")

    @pytest.mark.asyncio
    async def test_multi_page_scan_deletes_across_pages(self):
        service, mock_client, _ = make_service_with_client()
        mock_client.scan.side_effect = [
            (5, ["key1", "key2"]),
            (0, ["key3"]),
        ]

        await service._delete_pattern("user:1:*")

        assert mock_client.scan.await_count == 2
        assert mock_client.delete.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_scan_result_does_not_call_delete(self):
        service, mock_client, _ = make_service_with_client()
        mock_client.scan.return_value = (0, [])

        await service._delete_pattern("user:99:*")

        mock_client.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connection_error_marks_unavailable(self):
        service, mock_client, mock_config = make_service_with_client()
        mock_client.scan.side_effect = RedisConnectionError("broken")

        await service._delete_pattern("user:*")

        mock_config.mark_unavailable.assert_called_once()


# ============================================================================
# Key builders
# ============================================================================

class TestCacheServiceKeyBuilders:

    def test_key_personality(self):
        assert CacheService._key_personality("u1") == "user:u1:personality_profile"

    def test_key_profile(self):
        assert CacheService._key_profile("u1") == "user:u1:profile"

    def test_key_session_summary(self):
        assert CacheService._key_session_summary("conv1") == "conv:conv1:session_summary"

    def test_key_ui_state(self):
        assert CacheService._key_ui_state("u1") == "user:u1:ui_state"

    def test_key_user_config(self):
        assert CacheService._key_user_config("u1", "theme") == "user:u1:config:theme"

    def test_key_conv_history(self):
        assert CacheService._key_conv_history("conv-42") == "conv:conv-42:history"

    def test_key_global_config(self):
        assert CacheService._key_global_config("feature_flags") == "config:feature_flags"


# ============================================================================
# Public API — personality profile
# ============================================================================

class TestCacheServicePersonalityProfile:

    @pytest.mark.asyncio
    async def test_get_personality_profile_uses_correct_key(self):
        service, mock_client, _ = make_service_with_client()
        mock_client.get.return_value = json.dumps({"trait": "openness"})

        result = await service.get_personality_profile("u1")

        mock_client.get.assert_awaited_once_with("user:u1:personality_profile")
        assert result == {"trait": "openness"}

    @pytest.mark.asyncio
    async def test_set_personality_profile_uses_profile_ttl(self):
        service, mock_client, _ = make_service_with_client()

        await service.set_personality_profile("u1", {"trait": "openness"})

        mock_client.setex.assert_awaited_once()
        args = mock_client.setex.call_args[0]
        assert args[0] == "user:u1:personality_profile"
        assert args[1] == CacheService.TTL_PROFILE

    @pytest.mark.asyncio
    async def test_invalidate_personality_profile_deletes_key(self):
        service, mock_client, _ = make_service_with_client()

        await service.invalidate_personality_profile("u1")

        mock_client.delete.assert_awaited_once_with("user:u1:personality_profile")


# ============================================================================
# Public API — conversation history (trim logic)
# ============================================================================

class TestCacheServiceConversationHistory:

    @pytest.mark.asyncio
    async def test_get_conversation_history_uses_correct_key(self):
        turns = [{"speaker": "user", "message": "hi", "turn_index": 0}]
        service, mock_client, _ = make_service_with_client()
        mock_client.get.return_value = json.dumps(turns)

        result = await service.get_conversation_history("conv-1")

        mock_client.get.assert_awaited_once_with("conv:conv-1:history")
        assert result == turns

    @pytest.mark.asyncio
    async def test_set_conversation_history_stores_within_limit(self):
        service, mock_client, _ = make_service_with_client()
        turns = [{"turn_index": i} for i in range(5)]

        await service.set_conversation_history("conv-1", turns)

        args = mock_client.setex.call_args[0]
        stored = json.loads(args[2])
        assert len(stored) == 5

    @pytest.mark.asyncio
    async def test_set_conversation_history_trims_to_max_turns(self):
        service, mock_client, _ = make_service_with_client()
        # More turns than the maximum
        turns = [{"turn_index": i} for i in range(CONV_HISTORY_MAX_TURNS + 10)]

        await service.set_conversation_history("conv-1", turns)

        args = mock_client.setex.call_args[0]
        stored = json.loads(args[2])
        assert len(stored) == CONV_HISTORY_MAX_TURNS
        # Should keep the most recent turns
        assert stored[0]["turn_index"] == 10

    @pytest.mark.asyncio
    async def test_invalidate_conversation_history_deletes_key(self):
        service, mock_client, _ = make_service_with_client()

        await service.invalidate_conversation_history("conv-1")

        mock_client.delete.assert_awaited_once_with("conv:conv-1:history")


# ============================================================================
# Public API — bulk invalidation
# ============================================================================

class TestCacheServiceBulkInvalidation:

    @pytest.mark.asyncio
    async def test_invalidate_all_user_data_uses_wildcard_pattern(self):
        service, mock_client, _ = make_service_with_client()
        mock_client.scan.return_value = (0, [])

        await service.invalidate_all_user_data("u42")

        mock_client.scan.assert_awaited_once()
        scan_args = mock_client.scan.call_args
        assert scan_args[1]["match"] == "user:u42:*"

    @pytest.mark.asyncio
    async def test_invalidate_all_user_config_uses_config_pattern(self):
        service, mock_client, _ = make_service_with_client()
        mock_client.scan.return_value = (0, [])

        await service.invalidate_all_user_config("u42")

        scan_args = mock_client.scan.call_args
        assert scan_args[1]["match"] == "user:u42:config:*"
