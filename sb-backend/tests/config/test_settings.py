"""
Unit tests for centralised AppSettings (config/settings.py).

Validates structure, types, and defaults without requiring any external service.
Environment variable overrides are tested by temporarily patching os.environ.
"""

import os
import pytest
from unittest.mock import patch

from config.settings import settings, AppSettings


# ============================================================================
# Structure
# ============================================================================

class TestAppSettingsStructure:

    def test_settings_is_app_settings_instance(self):
        assert isinstance(settings, AppSettings)

    def test_all_sub_settings_are_present(self):
        assert hasattr(settings, "logging")
        assert hasattr(settings, "data_db")
        assert hasattr(settings, "auth_db")
        assert hasattr(settings, "supabase")
        assert hasattr(settings, "redis")
        assert hasattr(settings, "ollama")
        assert hasattr(settings, "openai")
        assert hasattr(settings, "llm")
        assert hasattr(settings, "server")


# ============================================================================
# Redis settings
# ============================================================================

class TestRedisSettings:

    def test_url_has_redis_scheme(self):
        assert settings.redis.url.startswith("redis://")

    def test_max_connections_is_positive_int(self):
        assert isinstance(settings.redis.max_connections, int)
        assert settings.redis.max_connections > 0

    def test_ttl_profile_is_positive_int(self):
        assert isinstance(settings.redis.ttl_profile, int)
        assert settings.redis.ttl_profile > 0

    def test_ttl_config_is_positive_int(self):
        assert isinstance(settings.redis.ttl_config, int)
        assert settings.redis.ttl_config > 0

    def test_ttl_conversation_is_positive_int(self):
        assert isinstance(settings.redis.ttl_conversation, int)
        assert settings.redis.ttl_conversation > 0

    def test_conv_history_max_turns_is_positive_int(self):
        assert isinstance(settings.redis.conv_history_max_turns, int)
        assert settings.redis.conv_history_max_turns > 0


# ============================================================================
# Ollama settings
# ============================================================================

class TestOllamaSettings:

    def test_base_url_is_non_empty_string(self):
        assert isinstance(settings.ollama.base_url, str)
        assert len(settings.ollama.base_url) > 0

    def test_model_is_non_empty_string(self):
        assert isinstance(settings.ollama.model, str)
        assert len(settings.ollama.model) > 0

    def test_timeout_is_positive_int(self):
        assert isinstance(settings.ollama.timeout, int)
        assert settings.ollama.timeout > 0


# ============================================================================
# LLM flags
# ============================================================================

class TestLLMSettings:

    def test_flags_are_booleans(self):
        assert isinstance(settings.llm.compare_results, bool)
        assert isinstance(settings.llm.ollama_flag, bool)
        assert isinstance(settings.llm.openai_flag, bool)


# ============================================================================
# Server settings
# ============================================================================

class TestServerSettings:

    def test_host_is_non_empty_string(self):
        assert isinstance(settings.server.host, str)
        assert len(settings.server.host) > 0

    def test_port_is_positive_int(self):
        assert isinstance(settings.server.port, int)
        assert settings.server.port > 0

    def test_environment_is_string(self):
        assert isinstance(settings.server.environment, str)


# ============================================================================
# Environment variable override
# ============================================================================

class TestSettingsEnvOverride:
    """Confirm that env vars are actually read (tested by creating a fresh instance)."""

    def test_redis_url_reads_from_env(self):
        from config.settings import RedisSettings
        custom_url = "redis://custom-host:9999"
        with patch.dict(os.environ, {"REDIS_URL": custom_url}):
            s = RedisSettings()
        assert s.url == custom_url

    def test_ollama_timeout_reads_from_env(self):
        from config.settings import OllamaSettings
        with patch.dict(os.environ, {"OLLAMA_TIMEOUT": "999"}):
            s = OllamaSettings()
        assert s.timeout == 999

    def test_compare_results_true_from_env(self):
        from config.settings import LLMSettings
        with patch.dict(os.environ, {"COMPARE_RESULTS": "true"}):
            s = LLMSettings()
        assert s.compare_results is True

    def test_compare_results_false_from_env(self):
        from config.settings import LLMSettings
        with patch.dict(os.environ, {"COMPARE_RESULTS": "false"}):
            s = LLMSettings()
        assert s.compare_results is False

    def test_conv_history_max_turns_reads_from_env(self):
        from config.settings import RedisSettings
        with patch.dict(os.environ, {"CACHE_CONV_HISTORY_MAX_TURNS": "50"}):
            s = RedisSettings()
        assert s.conv_history_max_turns == 50
