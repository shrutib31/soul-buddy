import os
import base64
import hashlib
from typing import Optional
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
from google.cloud import kms
from config.settings import encryption_config

ENCRYPTION_MARKER = "ENC:v1:"


class KeyManager:
    """
    Mirrors the Node server's KMS encryption approach exactly:
      1. Encrypt a fixed seed string with GCP KMS
      2. SHA-256 hash the resulting ciphertext → master key material
      3. Derive per-conversation keys via HKDF
      4. Encrypt/decrypt with AES-256-GCM
    
    Note: master key is stable within a process session (cached),
    but will differ across restarts. All data must be read/written
    within the same process generation — same limitation as Node server.
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
        Mirrors Node's getMasterKey():
        - Encrypt fixed seed with KMS
        - SHA-256 hash the ciphertext → 32-byte master key
        - Cache in memory
        """
        if self._master_key is not None:
            return self._master_key

        if not encryption_config.ENCRYPTION_ENABLED:
            raise RuntimeError("Encryption is disabled")

        try:
            print("Fetching master key from GCP KMS...")

            seed = b"souloxy-master-key-seed-v1"

            response = self.kms_client.encrypt(request={
                "name": self.key_name,
                "plaintext": seed,
            })

            ciphertext = bytes(response.ciphertext)
            # SHA-256 hash of ciphertext → deterministic 32-byte key (within session)
            self._master_key = hashlib.sha256(ciphertext).digest()

            print(f"✅ Master key loaded from GCP KMS ({len(self._master_key)} bytes)")
            return self._master_key

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve master key from GCP KMS: {e}")

    async def derive_conversation_key(self, conversation_id: str) -> bytes:
        """Mirrors Node's deriveConversationKey()."""
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
        """Mirrors Node's encrypt()."""
        if not encryption_config.ENCRYPTION_ENABLED:
            print("⚠️ Encryption is disabled - storing plaintext")
            return plaintext

        key = await self.derive_conversation_key(conversation_id)
        return self.encrypt_data(key, plaintext)

    async def decrypt(self, conversation_id: str, data: str) -> str:
        """Mirrors Node's decrypt()."""
        if not self.is_data_encrypted(data):
            return data

        if not encryption_config.ENCRYPTION_ENABLED:
            print("⚠️ Found encrypted data but encryption is disabled")
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