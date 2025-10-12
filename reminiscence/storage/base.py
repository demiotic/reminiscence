"""Abstract storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..types import CacheEntry


class StorageBackend(ABC):
    """Abstract interface for cache storage with hybrid exact/semantic support."""

    @abstractmethod
    def count(self) -> int:
        """Get total number of entries across all tables.

        Returns:
            Total count of cache entries.
        """
        pass

    @abstractmethod
    def add(self, entries: List[CacheEntry]) -> None:
        """Add cache entries to appropriate table based on query_mode.

        Args:
            entries: Cache entries to store. Each entry's metadata should
                    contain 'query_mode' to determine routing.
        """
        pass

    @abstractmethod
    def search(
        self,
        embedding: Optional[List[float]],
        context: Dict[str, Any],
        limit: int,
        similarity_threshold: float,
        query_mode: str,
        query_text: Optional[str] = None,
    ) -> List[CacheEntry]:
        """Search cache with mode-based routing.

        Args:
            embedding: Query embedding vector (None for exact mode).
            context: Context dict for exact matching.
            limit: Maximum results to return.
            similarity_threshold: Minimum similarity score for semantic search.
            query_mode: "exact" or "semantic".
            query_text: Original query text (required for exact mode).

        Returns:
            List of matching cache entries.
        """
        pass

    @abstractmethod
    def to_arrow(self):
        """Convert to Arrow table.

        Returns:
            PyArrow Table with cache entries.
        """
        pass

    @abstractmethod
    def delete_by_filter(self, filter_expr: str) -> None:
        """Delete entries matching filter expression.

        Args:
            filter_expr: SQL-like filter expression.
        """
        pass

    @abstractmethod
    def has_index(self) -> bool:
        """Check if vector index exists on semantic table.

        Returns:
            True if index exists, False otherwise.
        """
        pass

    @abstractmethod
    def create_index(self, num_partitions: int, num_sub_vectors: int) -> None:
        """Create vector index on semantic table.

        Args:
            num_partitions: Number of IVF partitions.
            num_sub_vectors: Number of PQ sub-vectors.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from all tables."""
        pass

    @abstractmethod
    def delete_by_id(self, entry_id: str) -> bool:
        """Delete a single entry by its unique ID.

        Args:
            entry_id: Unique identifier of the entry.

        Returns:
            True if entry was deleted, False if not found.
        """
        pass
