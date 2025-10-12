"""Base encryption interface for Reminiscence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class EncryptionBackend(ABC):
    """Abstract base class for encryption backends."""

    @abstractmethod
    def encrypt(self, data: Any) -> bytes:
        """Encrypt single data item.

        Args:
            data: Data to encrypt.

        Returns:
            Encrypted bytes.

        Raises:
            EncryptionError: If encryption fails.
        """
        pass

    @abstractmethod
    def decrypt(self, encrypted_data: bytes) -> Any:
        """Decrypt single data item.

        Args:
            encrypted_data: Encrypted bytes to decrypt.

        Returns:
            Decrypted data.

        Raises:
            DecryptionError: If decryption fails.
        """
        pass

    def encrypt_batch(self, data_list: List[Any]) -> List[bytes]:
        """Encrypt multiple items (default: sequential).

        Override for better performance (e.g., parallel, vectorized).

        Args:
            data_list: List of data items to encrypt.

        Returns:
            List of encrypted bytes.
        """
        return [self.encrypt(data) for data in data_list]

    def decrypt_batch(self, encrypted_list: List[bytes]) -> List[Any]:
        """Decrypt multiple items (default: sequential).

        Override for better performance (e.g., parallel, vectorized).

        Args:
            encrypted_list: List of encrypted bytes to decrypt.

        Returns:
            List of decrypted data items.
        """
        return [self.decrypt(data) for data in encrypted_list]

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Class name as string.
        """
        return f"{self.__class__.__name__}()"


class EncryptionError(Exception):
    """Raised when encryption fails."""

    pass


class DecryptionError(Exception):
    """Raised when decryption fails."""

    pass
