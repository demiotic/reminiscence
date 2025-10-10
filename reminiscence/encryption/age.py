"""Age encryption backend using pyrage with batch support."""

from typing import List
from concurrent.futures import ThreadPoolExecutor

try:
    from pyrage import x25519, encrypt, decrypt
except ImportError:
    raise ImportError(
        "pyrage is required for age encryption. Install with: pip install pyrage"
    )

from .base import EncryptionBackend, EncryptionError, DecryptionError


class AgeEncryption(EncryptionBackend):
    """
    Age encryption backend with batch support.

    Uses age (https://age-encryption.org/) for encryption.
    Works with bytes directly (no JSON serialization).

    Example:
        >>> enc = AgeEncryption(key="AGE-SECRET-KEY-...")

        >>> # Single encryption
        >>> encrypted = enc.encrypt(b"sensitive data")
        >>> decrypted = enc.decrypt(encrypted)

        >>> # Batch encryption (parallel)
        >>> data_list = [b"data1", b"data2", b"data3"]
        >>> encrypted_list = enc.encrypt_batch(data_list)
        >>> decrypted_list = enc.decrypt_batch(encrypted_list)
    """

    def __init__(self, key: str, max_workers: int = 4):
        """
        Initialize age encryption.

        Args:
            key: Age public/private key string
            max_workers: Max threads for batch operations

        Raises:
            ValueError: If key format is invalid
        """
        self.key = key
        self.max_workers = max_workers

        self.is_public = self.key.startswith("age1")
        self.is_private = self.key.startswith("AGE-SECRET-KEY-")

        if not (self.is_public or self.is_private):
            raise ValueError(
                "Invalid age key format. "
                "Must start with 'age1' (public) or 'AGE-SECRET-KEY-' (private)"
            )

        if self.is_public:
            self.recipient = x25519.Recipient.from_str(self.key)
            self.identity = None
        else:
            self.identity = x25519.Identity.from_str(self.key)
            self.recipient = self.identity.to_public()

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt bytes.

        Args:
            data: Raw bytes to encrypt

        Returns:
            Encrypted bytes

        Raises:
            EncryptionError: If encryption fails
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")

        try:
            encrypted = encrypt(data, [self.recipient])
            return encrypted
        except Exception as e:
            raise EncryptionError(f"Age encryption failed: {e}") from e

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt bytes.

        Args:
            encrypted_data: Encrypted bytes

        Returns:
            Decrypted bytes

        Raises:
            DecryptionError: If decryption fails or no private key
        """
        if not self.is_private:
            raise DecryptionError(
                "Decryption requires a private key (AGE-SECRET-KEY-...)"
            )

        if not isinstance(encrypted_data, bytes):
            raise TypeError(f"Expected bytes, got {type(encrypted_data).__name__}")

        try:
            decrypted_bytes = decrypt(encrypted_data, [self.identity])
            return decrypted_bytes
        except Exception as e:
            raise DecryptionError(f"Age decryption failed: {e}") from e

    def encrypt_batch(self, data_list: List[bytes]) -> List[bytes]:
        """
        Encrypt multiple byte sequences in parallel.

        Args:
            data_list: List of bytes to encrypt

        Returns:
            List of encrypted bytes (same order as input)

        Raises:
            EncryptionError: If any encryption fails
        """
        if not data_list:
            return []

        if len(data_list) <= 3:
            return [self.encrypt(data) for data in data_list]

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                encrypted_list = list(executor.map(self.encrypt, data_list))
            return encrypted_list
        except Exception as e:
            raise EncryptionError(f"Batch encryption failed: {e}") from e

    def decrypt_batch(self, encrypted_list: List[bytes]) -> List[bytes]:
        """
        Decrypt multiple encrypted byte sequences in parallel.

        Args:
            encrypted_list: List of encrypted bytes

        Returns:
            List of decrypted bytes (same order as input)

        Raises:
            DecryptionError: If any decryption fails or no private key
        """
        if not encrypted_list:
            return []

        if not self.is_private:
            raise DecryptionError(
                "Decryption requires a private key (AGE-SECRET-KEY-...)"
            )

        if len(encrypted_list) <= 3:
            return [self.decrypt(data) for data in encrypted_list]

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                decrypted_list = list(executor.map(self.decrypt, encrypted_list))
            return decrypted_list
        except Exception as e:
            raise DecryptionError(f"Batch decryption failed: {e}") from e

    def __repr__(self) -> str:
        key_type = "public" if self.is_public else "private"
        return f"AgeEncryption(key_type={key_type}, max_workers={self.max_workers})"
