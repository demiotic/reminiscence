# reminiscence/encryption/__init__.py
"""Encryption backends for Reminiscence."""

from .base import EncryptionBackend, EncryptionError, DecryptionError

try:
    from .age import AgeEncryption
except ImportError:
    AgeEncryption = None

__all__ = [
    "EncryptionBackend",
    "EncryptionError",
    "DecryptionError",
    "AgeEncryption",
]
