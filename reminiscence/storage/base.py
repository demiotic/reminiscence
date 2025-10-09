"""Abstract storage interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..types import CacheEntry


class StorageBackend(ABC):
    """Abstract interface for cache storage with hybrid exact/semantic support."""

    @abstractmethod
    def count(self) -> int:
        """Get total number of entries across all tables."""
        pass

    @abstractmethod
    def add(self, entries: List[CacheEntry], query_mode: str = "semantic"):
        """
        Add cache entries to appropriate table based on query_mode.

        Args:
            entries: Cache entries to store
            query_mode: "exact" stores without embeddings, "semantic"/"auto" with embeddings
        """
        pass

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        context: Dict[str, Any],
        limit: int,
        similarity_threshold: float,
        query_mode: str = "semantic",
        query_text: str = None,
    ) -> List[CacheEntry]:
        """
        Search cache with mode-based routing.

        Args:
            embedding: Query embedding vector (None for exact mode)
            context: Context dict for exact matching
            limit: Maximum results to return
            similarity_threshold: Minimum similarity score for semantic search
            query_mode: "exact", "semantic", or "auto"
            query_text: Original query text (required for exact/auto modes)
        """
        pass

    @abstractmethod
    def to_arrow(self):
        """Convert to Arrow table."""
        pass

    @abstractmethod
    def delete_by_filter(self, filter_expr: str):
        """Delete entries matching filter expression."""
        pass

    @abstractmethod
    def has_index(self) -> bool:
        """Check if vector index exists on semantic table."""
        pass

    @abstractmethod
    def create_index(self, num_partitions: int, num_sub_vectors: int):
        """Create vector index on semantic table."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all entries from all tables."""
        pass
