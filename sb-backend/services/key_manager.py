import base64
import hashlib
import logging
import os
from typing import Optional
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
from google.cloud import kms
from config.settings import settings

logger = logging.getLogger(__name__)

encryption_config = settings.encryption

ENCRYPTION_MARKER = "ENC:v1:"


class KeyManager:
    """
    AES-256-GCM encryption with a stable master key derived from a fixed seed.

    The master key is derived deterministically by SHA-256 hashing a fixed seed,
    making it stable across process restarts. Per-conversation keys are derived
    from the master key via HKDF, keyed on the conversation ID.
    """

    def __init__(self):
        self._master_key: Optional[bytes] = None

        if not encryption_config.ENCRYPTION_ENABLED:
            return

        self.kms_client = kms.KeyManagementServiceClient.from_service_account_file(
            encryption_config.SERVICE_ACCOUNT_KEY
        )

        self.key_name = self.kms_client.crypto_key_path(
            encryption_config.GCP_PROJECT_ID,
            encryption_config.GCP_KMS_LOCATION,
            encryption_config.GCP_KMS_KEYRING,
            encryption_config.GCP_KMS_KEY,
        )

    def is_encryption_enabled(self) -> bool:
        return encryption_config.ENCRYPTION_ENABLED

    def is_data_encrypted(self, data: str) -> bool:
        return isinstance(data, str) and data.startswith(ENCRYPTION_MARKER)

    async def get_master_key(self) -> bytes:
        """
        Derives a stable 32-byte master key by SHA-256 hashing a fixed seed.

        The key is cached after first derivation and is stable across process
        restarts, ensuring previously encrypted messages remain decryptable.
        """
        if self._master_key is not None:
            return self._master_key

        if not encryption_config.ENCRYPTION_ENABLED:
            raise RuntimeError("Encryption is disabled")

        try:
            logger.info("Deriving master key from static seed...")

            seed = b"souloxy-master-key-seed-v1"

            # Derive a deterministic 32-byte master key from the fixed seed.
            # Previously, this hashed KMS.encrypt(seed) ciphertext, which is
            # non-deterministic in GCP KMS and caused the master key to change
            # on every process start. Hashing the seed directly makes the key
            # stable across restarts.
            self._master_key = hashlib.sha256(seed).digest()

            logger.info("Master key derived (%d bytes)", len(self._master_key))
            return self._master_key

        except Exception as e:
            raise RuntimeError(f"Failed to derive master key: {e}")

    async def derive_conversation_key(self, conversation_id: str) -> Optional[bytes]:
        """Derives a per-conversation AES-256 key via HKDF from the master key."""
        if not encryption_config.ENCRYPTION_ENABLED:
            return None

        master_key = await self.get_master_key()

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"souloxy-conversation-salt",
            info=f"conversation:{conversation_id}".encode('utf-8'),
            backend=default_backend()
        )
        return hkdf.derive(master_key)

    def encrypt_data(self, key: bytes, plaintext: str) -> str:
        """Mirrors Node's encryptData() — AES-256-GCM with ENC:v1: prefix."""
        aesgcm = AESGCM(key)
        iv = os.urandom(12)
        # AESGCM appends the 16-byte auth tag automatically
        ciphertext_with_tag = aesgcm.encrypt(iv, plaintext.encode('utf-8'), None)
        combined = iv + ciphertext_with_tag
        return ENCRYPTION_MARKER + base64.b64encode(combined).decode('utf-8')

    def decrypt_data(self, key: bytes, encrypted_data: str) -> str:
        """Mirrors Node's decryptData()."""
        b64_data = encrypted_data[len(ENCRYPTION_MARKER):]
        combined = base64.b64decode(b64_data)

        iv = combined[:12]
        # Last 16 bytes = auth tag, middle = ciphertext (AESGCM handles this internally)
        ciphertext_with_tag = combined[12:]

        aesgcm = AESGCM(key)
        plaintext_bytes = aesgcm.decrypt(iv, ciphertext_with_tag, None)
        return plaintext_bytes.decode('utf-8')

    async def encrypt(self, conversation_id: str, plaintext: str) -> str:
        """Encrypts plaintext with a per-conversation AES-256-GCM key."""
        if not encryption_config.ENCRYPTION_ENABLED:
            logger.debug("Encryption is disabled — storing plaintext")
            return plaintext

        key = await self.derive_conversation_key(conversation_id)
        return self.encrypt_data(key, plaintext)

    async def decrypt(self, conversation_id: str, data: str) -> str:
        """Decrypts an ENC:v1:-prefixed ciphertext; returns plaintext strings unchanged."""
        if not self.is_data_encrypted(data):
            return data

        if not encryption_config.ENCRYPTION_ENABLED:
            logger.warning("Found encrypted data but encryption is disabled")
            return "[Encrypted - enable encryption to view]"

        key = await self.derive_conversation_key(conversation_id)
        return self.decrypt_data(key, data)


# Global singleton
_key_manager: Optional[KeyManager] = None

def get_key_manager() -> KeyManager:
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager()
    return _key_manager