"""
Unit tests for KeyManager (services/key_manager.py).

All tests that touch encryption run without GCP KMS — either by
disabling encryption (ENCRYPTION_ENABLED=False) or by exercising the
pure-crypto helpers (encrypt_data / decrypt_data) directly with a
pre-built test key.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Helpers
# ============================================================================

def _make_disabled_config():
    """Return a mock EncryptionConfig with encryption disabled."""
    cfg = MagicMock()
    cfg.ENCRYPTION_ENABLED = False
    return cfg


def _make_enabled_config():
    """Return a mock EncryptionConfig with encryption enabled."""
    cfg = MagicMock()
    cfg.ENCRYPTION_ENABLED = True
    cfg.GCP_PROJECT_ID = "test-project"
    cfg.GCP_KMS_LOCATION = "global"
    cfg.GCP_KMS_KEYRING = "test-ring"
    cfg.GCP_KMS_KEY = "test-key"
    cfg.SERVICE_ACCOUNT_KEY = "/fake/service_account.json"
    return cfg


def _make_key_manager(encryption_enabled: bool = False):
    """Create a KeyManager with GCP KMS mocked out."""
    cfg = _make_enabled_config() if encryption_enabled else _make_disabled_config()
    with patch("services.key_manager.encryption_config", cfg), \
         patch("services.key_manager.kms") as mock_kms:
        mock_client = MagicMock()
        mock_client.crypto_key_path.return_value = "projects/test/..."
        mock_kms.KeyManagementServiceClient.from_service_account_file.return_value = mock_client

        from services.key_manager import KeyManager
        km = KeyManager()
        km._kms_client_mock = mock_client  # expose for assertions if needed
    return km, cfg


# ============================================================================
# is_data_encrypted
# ============================================================================

class TestIsDataEncrypted:

    def test_returns_true_for_enc_prefix(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        assert km.is_data_encrypted("ENC:v1:abc123") is True

    def test_returns_false_for_plaintext(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        assert km.is_data_encrypted("Hello, world!") is False

    def test_returns_false_for_empty_string(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        assert km.is_data_encrypted("") is False

    def test_returns_false_for_partial_prefix(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        assert km.is_data_encrypted("ENC:v1") is False  # missing trailing colon


# ============================================================================
# is_encryption_enabled
# ============================================================================

class TestIsEncryptionEnabled:

    def test_returns_false_when_disabled(self):
        km, cfg = _make_key_manager(encryption_enabled=False)
        with patch("services.key_manager.encryption_config", cfg):
            assert km.is_encryption_enabled() is False

    def test_returns_true_when_enabled(self):
        km, cfg = _make_key_manager(encryption_enabled=True)
        with patch("services.key_manager.encryption_config", cfg):
            assert km.is_encryption_enabled() is True


# ============================================================================
# encrypt_data / decrypt_data  (pure crypto — no KMS)
# ============================================================================

class TestEncryptDataDecryptData:
    """Tests for the pure AES-256-GCM helpers — no KMS involvement."""

    TEST_KEY = os.urandom(32)  # stable within this test session

    def test_encrypt_produces_enc_prefix(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        encrypted = km.encrypt_data(self.TEST_KEY, "hello world")
        assert encrypted.startswith("ENC:v1:")

    def test_decrypt_recovers_original_plaintext(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        plaintext = "This is a secret message"
        encrypted = km.encrypt_data(self.TEST_KEY, plaintext)
        decrypted = km.decrypt_data(self.TEST_KEY, encrypted)
        assert decrypted == plaintext

    def test_different_nonces_produce_different_ciphertext(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        enc1 = km.encrypt_data(self.TEST_KEY, "same plaintext")
        enc2 = km.encrypt_data(self.TEST_KEY, "same plaintext")
        # AES-GCM uses a random 12-byte IV, so ciphertexts must differ
        assert enc1 != enc2

    def test_decrypt_fails_with_wrong_key(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        encrypted = km.encrypt_data(self.TEST_KEY, "secret")
        wrong_key = os.urandom(32)
        with pytest.raises(Exception):
            km.decrypt_data(wrong_key, encrypted)

    def test_roundtrip_with_unicode_text(self):
        km, _ = _make_key_manager(encryption_enabled=False)
        plaintext = "नमस्ते 🙏 こんにちは"
        encrypted = km.encrypt_data(self.TEST_KEY, plaintext)
        assert km.decrypt_data(self.TEST_KEY, encrypted) == plaintext


# ============================================================================
# encrypt / decrypt  (high-level — encryption disabled)
# ============================================================================

class TestEncryptDecryptDisabled:
    """When ENCRYPTION_ENABLED=False the high-level methods are pass-through."""

    @pytest.mark.asyncio
    async def test_encrypt_returns_plaintext_unchanged(self):
        km, cfg = _make_key_manager(encryption_enabled=False)
        with patch("services.key_manager.encryption_config", cfg):
            result = await km.encrypt("conv-123", "hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_decrypt_returns_plaintext_unchanged(self):
        km, cfg = _make_key_manager(encryption_enabled=False)
        with patch("services.key_manager.encryption_config", cfg):
            result = await km.decrypt("conv-123", "hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_decrypt_encrypted_data_when_disabled_returns_placeholder(self):
        """If somehow encrypted data reaches decrypt() but encryption is off,
        return the 'enable encryption' placeholder instead of crashing."""
        km, cfg = _make_key_manager(encryption_enabled=False)
        with patch("services.key_manager.encryption_config", cfg):
            result = await km.decrypt("conv-123", "ENC:v1:somebase64data==")
        assert "[Encrypted" in result

    @pytest.mark.asyncio
    async def test_decrypt_non_encrypted_data_returned_as_is(self):
        km, cfg = _make_key_manager(encryption_enabled=False)
        with patch("services.key_manager.encryption_config", cfg):
            result = await km.decrypt("conv-123", "plain text stored without encryption")
        assert result == "plain text stored without encryption"


# ============================================================================
# get_master_key  (raises when encryption disabled)
# ============================================================================

class TestGetMasterKey:

    @pytest.mark.asyncio
    async def test_raises_when_encryption_disabled(self):
        km, cfg = _make_key_manager(encryption_enabled=False)
        with patch("services.key_manager.encryption_config", cfg):
            with pytest.raises(RuntimeError, match="Encryption is disabled"):
                await km.get_master_key()

    @pytest.mark.asyncio
    async def test_caches_master_key_after_first_call(self):
        """Second call must not hit KMS again — uses cached value."""
        km, cfg = _make_key_manager(encryption_enabled=True)

        # Pre-seed the cache to bypass real KMS call
        test_master_key = os.urandom(32)
        km._master_key = test_master_key

        with patch("services.key_manager.encryption_config", cfg):
            result = await km.get_master_key()

        assert result == test_master_key
