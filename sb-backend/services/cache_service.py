"""
CacheService — Centralised Redis caching wrapper.

Design principles
-----------------
* Single entry point: all cache operations go through this class.
  No scattered ``redis`` calls anywhere else in the codebase.

* Fail-silent: every public method catches exceptions and returns None /
  does nothing when Redis is unavailable, so callers never need to guard.

* Cache-aside ready: the ``get`` / ``set`` / ``delete`` methods are the
  only place that touch Redis.  An in-memory L1 cache can be layered in
  front of ``_get`` / ``_set`` / ``_delete`` later without changing any
  callers.

Key schema
----------
  user:<userId>:personality_profile   — personality traits (TTL_PROFILE)
  user:<userId>:profile               — user profile / preferences (TTL_PROFILE)
  user:<userId>:user_memory           — long-term user memory / narrative (TTL_CONFIG = 24h)
  user:<userId>:ui_state              — last UI / navigation state (TTL_CONV)
  user:<userId>:config:<configKey>    — per-user config entry (TTL_CONFIG)
  conv:<conversationId>:history       — last N conversation turns (TTL_CONV)
  conv:<conversationId>:session_summary — current session summary JSONB (TTL_PROFILE = 2h)
  config:<configKey>                  — global config entry (TTL_CONFIG)

Deprecated keys (kept for transition period):
  user:<userId>:conversation_summary  — replaced by conv:<id>:session_summary

TTLs (seconds, overrideable via env vars)
-----------------------------------------
  REDIS_TTL_PROFILE      default 7 200   (2 h)
  REDIS_TTL_CONFIG       default 86 400  (24 h)
  REDIS_TTL_CONVERSATION default 1 800   (30 min)
"""

import json
import logging
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTL constants — sourced from centralised settings
# ---------------------------------------------------------------------------
_TTL_PROFILE: int = settings.redis.ttl_profile
_TTL_CONFIG: int = settings.redis.ttl_config
_TTL_CONVERSATION: int = settings.redis.ttl_conversation

CONV_HISTORY_MAX_TURNS: int = settings.redis.conv_history_max_turns


class CacheService:
    """
    Centralised cache service.

    Instantiate once (module level below) and inject the Redis client after
    the server connects to Redis in ``server.py``.

    All public methods are safe to call even when Redis is down — they will
    log a warning and return ``None`` / do nothing.
    """

    TTL_PROFILE = _TTL_PROFILE
    TTL_CONFIG = _TTL_CONFIG
    TTL_CONVERSATION = _TTL_CONVERSATION

    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None
        self._redis_config = None  # set via set_redis_config(); avoids circular import

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_client(self, client: Optional[aioredis.Redis]) -> None:
        """Inject the Redis client after startup or reconnect (called from server.py)."""
        self._redis = client

    def set_redis_config(self, redis_config) -> None:
        """
        Inject the RedisConfig instance so CacheService can call
        ``mark_unavailable()`` when it detects a connection-level error.
        """
        self._redis_config = redis_config

    @property
    def is_available(self) -> bool:
        return self._redis is not None

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _handle_connection_error(self, exc: Exception, context: str) -> None:
        """
        Null the local client reference and notify RedisConfig to begin
        the reconnect loop.  Called on RedisConnectionError / TimeoutError.
        """
        logger.warning("cache %s failed (connection lost) error=%s — Redis marked unavailable", context, exc)
        self._redis = None
        if self._redis_config is not None:
            self._redis_config.mark_unavailable()

    # ------------------------------------------------------------------
    # Low-level primitives — override these to add an L1 in-memory layer
    # ------------------------------------------------------------------

    async def _get(self, key: str) -> Optional[Any]:
        """
        Fetch a cached value and JSON-decode it.

        Returns None on miss, error, or Redis unavailability.
        """
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(key)
            if raw is None:
                logger.debug("cache MISS key=%s", key)
                return None
            logger.debug("cache HIT  key=%s", key)
            return json.loads(raw)
        except (RedisConnectionError, RedisTimeoutError) as exc:
            self._handle_connection_error(exc, f"GET key={key}")
            return None
        except Exception as exc:
            logger.warning("cache GET failed key=%s error=%s", key, exc)
            return None

    async def _set(self, key: str, value: Any, ttl: int) -> None:
        """
        JSON-encode and store a value with the given TTL (seconds).

        Silently ignores errors / Redis unavailability.
        """
        if self._redis is None:
            return
        try:
            serialised = json.dumps(value, default=str)
            await self._redis.setex(key, ttl, serialised)
            logger.debug("cache SET  key=%s ttl=%ds", key, ttl)
        except (RedisConnectionError, RedisTimeoutError) as exc:
            self._handle_connection_error(exc, f"SET key={key}")
        except Exception as exc:
            logger.warning("cache SET failed key=%s error=%s", key, exc)

    async def _delete(self, key: str) -> None:
        """Delete a key. Silently ignores errors."""
        if self._redis is None:
            return
        try:
            await self._redis.delete(key)
            logger.debug("cache DEL  key=%s", key)
        except (RedisConnectionError, RedisTimeoutError) as exc:
            self._handle_connection_error(exc, f"DEL key={key}")
        except Exception as exc:
            logger.warning("cache DEL failed key=%s error=%s", key, exc)

    async def _delete_pattern(self, pattern: str) -> None:
        """Delete all keys matching a glob pattern (uses SCAN to avoid blocking)."""
        if self._redis is None:
            return
        try:
            cursor = 0
            total_deleted = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                    total_deleted += len(keys)
                if cursor == 0:
                    break
            logger.debug("cache DEL  pattern=%s keys_removed=%d", pattern, total_deleted)
        except (RedisConnectionError, RedisTimeoutError) as exc:
            self._handle_connection_error(exc, f"DEL pattern={pattern}")
        except Exception as exc:
            logger.warning("cache DEL pattern failed pattern=%s error=%s", pattern, exc)

    # ------------------------------------------------------------------
    # Key builders (pure, testable)
    # ------------------------------------------------------------------

    @staticmethod
    def _key_personality(user_id: str) -> str:
        return f"user:{user_id}:personality_profile"

    @staticmethod
    def _key_profile(user_id: str) -> str:
        return f"user:{user_id}:profile"

    @staticmethod
    def _key_conversation_summary(user_id: str) -> str:
        # Deprecated — kept for transition period only
        return f"user:{user_id}:conversation_summary"

    @staticmethod
    def _key_user_memory(user_id: str) -> str:
        return f"user:{user_id}:user_memory"

    @staticmethod
    def _key_session_summary(conversation_id: str) -> str:
        return f"conv:{conversation_id}:session_summary"

    @staticmethod
    def _key_ui_state(user_id: str) -> str:
        return f"user:{user_id}:ui_state"

    @staticmethod
    def _key_user_config(user_id: str, config_key: str) -> str:
        return f"user:{user_id}:config:{config_key}"

    @staticmethod
    def _key_conv_history(conversation_id: str) -> str:
        return f"conv:{conversation_id}:history"

    @staticmethod
    def _key_global_config(config_key: str) -> str:
        return f"config:{config_key}"

    # ------------------------------------------------------------------
    # Personality profile
    # ------------------------------------------------------------------

    async def get_personality_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._get(self._key_personality(user_id))

    async def set_personality_profile(self, user_id: str, data: Dict[str, Any]) -> None:
        await self._set(self._key_personality(user_id), data, self.TTL_PROFILE)

    async def invalidate_personality_profile(self, user_id: str) -> None:
        await self._delete(self._key_personality(user_id))

    # ------------------------------------------------------------------
    # User profile (preferences, settings)
    # ------------------------------------------------------------------

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._get(self._key_profile(user_id))

    async def set_user_profile(self, user_id: str, data: Dict[str, Any]) -> None:
        await self._set(self._key_profile(user_id), data, self.TTL_PROFILE)

    async def invalidate_user_profile(self, user_id: str) -> None:
        await self._delete(self._key_profile(user_id))

    # ------------------------------------------------------------------
    # Conversation summary
    # ------------------------------------------------------------------

    async def get_conversation_summary(self, user_id: str) -> Optional[str]:
        return await self._get(self._key_conversation_summary(user_id))

    async def set_conversation_summary(self, user_id: str, summary: str) -> None:
        await self._set(self._key_conversation_summary(user_id), summary, self.TTL_CONVERSATION)

    async def invalidate_conversation_summary(self, user_id: str) -> None:
        await self._delete(self._key_conversation_summary(user_id))

    # ------------------------------------------------------------------
    # Session summary  (conv:<conversationId>:session_summary, TTL 2h)
    # Stores the JSONB summary dict for the current session.
    # ------------------------------------------------------------------

    async def get_session_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return await self._get(self._key_session_summary(conversation_id))

    async def set_session_summary(self, conversation_id: str, summary: Dict[str, Any]) -> None:
        await self._set(self._key_session_summary(conversation_id), summary, self.TTL_PROFILE)

    async def invalidate_session_summary(self, conversation_id: str) -> None:
        await self._delete(self._key_session_summary(conversation_id))

    # ------------------------------------------------------------------
    # User memory  (user:<userId>:user_memory, TTL 24h)
    # Stores the full user_memory row dict (growth_summary, themes, etc.)
    # ------------------------------------------------------------------

    async def get_user_memory(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._get(self._key_user_memory(user_id))

    async def set_user_memory(self, user_id: str, memory: Dict[str, Any]) -> None:
        await self._set(self._key_user_memory(user_id), memory, self.TTL_CONFIG)

    async def invalidate_user_memory(self, user_id: str) -> None:
        await self._delete(self._key_user_memory(user_id))

    # ------------------------------------------------------------------
    # Conversation history (last N turns)
    # ------------------------------------------------------------------

    async def get_conversation_history(
        self, conversation_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        return await self._get(self._key_conv_history(conversation_id))

    async def set_conversation_history(
        self, conversation_id: str, turns: List[Dict[str, Any]]
    ) -> None:
        # Trim to the most recent N turns before caching
        trimmed = turns[-CONV_HISTORY_MAX_TURNS:] if len(turns) > CONV_HISTORY_MAX_TURNS else turns
        await self._set(self._key_conv_history(conversation_id), trimmed, self.TTL_CONVERSATION)

    async def invalidate_conversation_history(self, conversation_id: str) -> None:
        await self._delete(self._key_conv_history(conversation_id))

    # ------------------------------------------------------------------
    # UI / navigation state
    # ------------------------------------------------------------------

    async def get_ui_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._get(self._key_ui_state(user_id))

    async def set_ui_state(self, user_id: str, state: Dict[str, Any]) -> None:
        await self._set(self._key_ui_state(user_id), state, self.TTL_CONVERSATION)

    async def invalidate_ui_state(self, user_id: str) -> None:
        await self._delete(self._key_ui_state(user_id))

    # ------------------------------------------------------------------
    # Global config  (config:<configKey>)
    # ------------------------------------------------------------------

    async def get_config(self, config_key: str) -> Optional[Any]:
        return await self._get(self._key_global_config(config_key))

    async def set_config(self, config_key: str, data: Any) -> None:
        await self._set(self._key_global_config(config_key), data, self.TTL_CONFIG)

    async def invalidate_config(self, config_key: str) -> None:
        await self._delete(self._key_global_config(config_key))

    # ------------------------------------------------------------------
    # Per-user config  (user:<userId>:config:<configKey>)
    # ------------------------------------------------------------------

    async def get_user_config(self, user_id: str, config_key: str) -> Optional[Any]:
        return await self._get(self._key_user_config(user_id, config_key))

    async def set_user_config(self, user_id: str, config_key: str, data: Any) -> None:
        await self._set(self._key_user_config(user_id, config_key), data, self.TTL_CONFIG)

    async def invalidate_user_config(self, user_id: str, config_key: str) -> None:
        await self._delete(self._key_user_config(user_id, config_key))

    async def invalidate_all_user_config(self, user_id: str) -> None:
        await self._delete_pattern(f"user:{user_id}:config:*")

    # ------------------------------------------------------------------
    # Bulk user invalidation (e.g. on account deletion)
    # ------------------------------------------------------------------

    async def invalidate_all_user_data(self, user_id: str) -> None:
        """Remove every cached key belonging to a user."""
        await self._delete_pattern(f"user:{user_id}:*")


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
cache_service = CacheService()
