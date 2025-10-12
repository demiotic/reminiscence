"""Storage backends for Reminiscence."""

from __future__ import annotations

from typing import Any

from .base import StorageBackend
from .lancedb import LanceDBBackend
from .schemas import create_exact_schema, create_semantic_schema


def create_storage_backend(
    config, embedding_dim: int, metrics: Any = None
) -> StorageBackend:
    """Factory to create storage backend from config.

    Currently only supports LanceDB. In the future, this can be extended
    to support ChromaDB, Qdrant, Pinecone, etc.

    Args:
        config: Configuration object with db_uri and other settings.
        embedding_dim: Dimension of embedding vectors.
        metrics: Optional CacheMetrics instance for tracking.

    Returns:
        StorageBackend instance (currently LanceDBBackend).

    Raises:
        ValueError: If storage backend type is not supported.

    Example:
        >>> from reminiscence import ReminiscenceConfig
        >>> config = ReminiscenceConfig(db_uri="memory://")
        >>> storage = create_storage_backend(config, embedding_dim=384)
    """
    # Future: detect backend from config.storage_backend or db_uri
    # For now, always use LanceDB
    return LanceDBBackend(config, embedding_dim, metrics)


__all__ = [
    "StorageBackend",
    "LanceDBBackend",
    "create_storage_backend",
    "create_exact_schema",
    "create_semantic_schema",
]
