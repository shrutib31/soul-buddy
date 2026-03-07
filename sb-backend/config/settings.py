"""
Centralised application settings.

All environment variables are read exactly once here, grouped into typed
dataclasses by concern.  Every other module should import ``settings``
from this module instead of calling ``os.getenv`` directly.

Usage:
    from config.settings import settings

    url = settings.redis.url
    key = settings.openai.api_key
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Per-concern config dataclasses
# ============================================================================

@dataclass(frozen=True)
class LoggingSettings:
    config_path: str = field(default_factory=lambda: os.getenv("LOGGING_CONFIG_PATH", "logging.yaml"))
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))


@dataclass(frozen=True)
class DataDBSettings:
    url: str = field(default_factory=lambda: os.getenv("DATA_DB_URL", ""))
    host: str = field(default_factory=lambda: os.getenv("DATA_DB_HOST", "localhost"))
    port: str = field(default_factory=lambda: os.getenv("DATA_DB_PORT", "5432"))
    name: str = field(default_factory=lambda: os.getenv("DATA_DB_NAME", "soulbuddy_data_db"))
    user: str = field(default_factory=lambda: os.getenv("DATA_DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DATA_DB_PASSWORD", ""))


@dataclass(frozen=True)
class AuthDBSettings:
    # Supports both AUTH_DB_* and legacy RBAC_DB_* variable names.
    url: str = field(default_factory=lambda: os.getenv("AUTH_DB_URL", os.getenv("RBAC_DB_URL", "")))
    host: str = field(default_factory=lambda: os.getenv("AUTH_DB_HOST", os.getenv("RBAC_DB_HOST", "localhost")))
    port: str = field(default_factory=lambda: os.getenv("AUTH_DB_PORT", os.getenv("RBAC_DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("AUTH_DB_NAME", os.getenv("RBAC_DB_NAME", "souloxy-db")))
    user: str = field(default_factory=lambda: os.getenv("AUTH_DB_USER", os.getenv("RBAC_DB_USER", "postgres")))
    password: str = field(default_factory=lambda: os.getenv("AUTH_DB_PASSWORD", os.getenv("RBAC_DB_PASSWORD", "")))


@dataclass(frozen=True)
class SupabaseSettings:
    url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    service_role_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))
    anon_key: str = field(default_factory=lambda: os.getenv("SUPABASE_ANON_KEY", ""))


@dataclass(frozen=True)
class RedisSettings:
    url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    max_connections: int = field(default_factory=lambda: int(os.getenv("REDIS_MAX_CONNECTIONS", "20")))
    reconnect_interval: int = field(default_factory=lambda: int(os.getenv("REDIS_RECONNECT_INTERVAL", "30")))
    ttl_profile: int = field(default_factory=lambda: int(os.getenv("REDIS_TTL_PROFILE", "7200")))
    ttl_config: int = field(default_factory=lambda: int(os.getenv("REDIS_TTL_CONFIG", "86400")))
    ttl_conversation: int = field(default_factory=lambda: int(os.getenv("REDIS_TTL_CONVERSATION", "1800")))
    conv_history_max_turns: int = field(default_factory=lambda: int(os.getenv("CACHE_CONV_HISTORY_MAX_TURNS", "20")))


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://72.60.99.35:11434"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    timeout: int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "120")))


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


@dataclass(frozen=True)
class LLMSettings:
    """
    Controls which LLM backend(s) are used.

    compare_results=True  — call both Ollama + OpenAI in parallel; pick best.
                            Requires ollama.base_url and openai.api_key.
    compare_results=False — use individual flags:
        ollama_flag=True  → Ollama only
        openai_flag=True  → OpenAI only
        both True         → call both, use first successful response
    """
    compare_results: bool = field(default_factory=lambda: os.getenv("COMPARE_RESULTS", "false").lower() == "true")
    ollama_flag: bool = field(default_factory=lambda: os.getenv("OLLAMA_FLAG", "false").lower() == "true")
    openai_flag: bool = field(default_factory=lambda: os.getenv("OPENAI_FLAG", "false").lower() == "true")


@dataclass(frozen=True)
class ServerSettings:
    host: str = field(default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8000")))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))


# ============================================================================
# Root settings object — import this everywhere
# ============================================================================

@dataclass(frozen=True)
class AppSettings:
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    data_db: DataDBSettings = field(default_factory=DataDBSettings)
    auth_db: AuthDBSettings = field(default_factory=AuthDBSettings)
    supabase: SupabaseSettings = field(default_factory=SupabaseSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    server: ServerSettings = field(default_factory=ServerSettings)

settings: AppSettings = AppSettings()
import os
from dotenv import load_dotenv

load_dotenv()

class EncryptionConfig:
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'souloxy-webapp')
    GCP_KMS_LOCATION = os.getenv('GCP_KMS_LOCATION', 'in')
    GCP_KMS_KEYRING = os.getenv('GCP_KMS_KEYRING', 'souloxy-mk-test')
    GCP_KMS_KEY = os.getenv('GCP_KMS_KEY', 'souloxy-mk-test')
    SERVICE_ACCOUNT_KEY = './config/serviceAccountKey.json'
    ENCRYPTION_ENABLED = os.getenv('ENCRYPTION_ENABLED', 'true').lower().strip() == 'true'

encryption_config = EncryptionConfig()