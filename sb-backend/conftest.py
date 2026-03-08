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


_stub_google_cloud_kms()
