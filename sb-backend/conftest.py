"""
Root conftest.py — stubs out optional/heavy third-party packages that are
not installed in the test environment so that test collection succeeds even
when those packages are absent.
"""

import sys
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# google-cloud-kms  (optional dependency; not installed in test venv)
# ---------------------------------------------------------------------------

def _stub_google_cloud_kms():
    """Insert minimal stubs so `from google.cloud import kms` doesn't crash."""
    google_mock = MagicMock()
    google_cloud_mock = MagicMock()
    kms_mock = MagicMock()

    # Wire the hierarchy together
    google_mock.cloud = google_cloud_mock
    google_cloud_mock.kms = kms_mock

    sys.modules.setdefault("google", google_mock)
    sys.modules.setdefault("google.cloud", google_cloud_mock)
    sys.modules.setdefault("google.cloud.kms", kms_mock)


try:
    import google.cloud.kms  # type: ignore[import-not-found]
except ImportError:
    _stub_google_cloud_kms()


# ---------------------------------------------------------------------------
# redis  (optional dependency; not installed in test venv)
# ---------------------------------------------------------------------------

def _stub_redis():
    """Insert minimal stubs so `import redis` doesn't crash."""
    import types

    redis_module = types.ModuleType("redis")
    redis_asyncio_module = types.ModuleType("redis.asyncio")
    redis_exceptions_module = types.ModuleType("redis.exceptions")
    redis_asyncio_module.Redis = object
    redis_exceptions_module.ConnectionError = RuntimeError
    redis_exceptions_module.TimeoutError = RuntimeError
    redis_module.asyncio = redis_asyncio_module
    redis_module.exceptions = redis_exceptions_module
    sys.modules.setdefault("redis", redis_module)
    sys.modules.setdefault("redis.asyncio", redis_asyncio_module)
    sys.modules.setdefault("redis.exceptions", redis_exceptions_module)


try:
    import redis  # noqa: F401
except ImportError:
    _stub_redis()
