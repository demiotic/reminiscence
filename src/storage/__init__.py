"""Storage backend abstractions."""

from .base import StorageBackend
from .lancedb import LanceDBBackend


def create_storage_backend(config, embedding_dim: int) -> StorageBackend:
    """Factory to create storage backend from config."""
    return LanceDBBackend(config, embedding_dim)


__all__ = ["StorageBackend", "create_storage_backend"]
