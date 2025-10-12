"""Encryption backends for Reminiscence."""

from __future__ import annotations

from typing import Optional

from .base import DecryptionError, EncryptionBackend, EncryptionError

# Optional import - pyrage might not be installed
try:
    from .age import AgeEncryption
except ImportError:
    AgeEncryption: Optional[type] = None  # type: ignore

__all__ = [
    "EncryptionBackend",
    "EncryptionError",
    "DecryptionError",
    "AgeEncryption",
]
