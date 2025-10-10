# reminiscence/encryption/base.py
"""Base encryption interface for Reminiscence."""

from abc import ABC, abstractmethod
from typing import Any, List


class EncryptionBackend(ABC):
    """Abstract base class for encryption backends."""

    @abstractmethod
    def encrypt(self, data: Any) -> bytes:
        """Encrypt single data item."""
        pass

    @abstractmethod
    def decrypt(self, encrypted_data: bytes) -> Any:
        """Decrypt single data item."""
        pass

    def encrypt_batch(self, data_list: List[Any]) -> List[bytes]:
        """
        Encrypt multiple items (default: sequential).

        Override for better performance (e.g., parallel, vectorized).
        """
        return [self.encrypt(data) for data in data_list]

    def decrypt_batch(self, encrypted_list: List[bytes]) -> List[Any]:
        """
        Decrypt multiple items (default: sequential).

        Override for better performance (e.g., parallel, vectorized).
        """
        return [self.decrypt(data) for data in encrypted_list]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class EncryptionError(Exception):
    """Raised when encryption fails."""

    pass


class DecryptionError(Exception):
    """Raised when decryption fails."""

    pass
