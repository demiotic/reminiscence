"""Abstract storage interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..types import CacheEntry


class StorageBackend(ABC):
    """Abstract interface for cache storage."""

    @abstractmethod
    def count(self) -> int:
        """Get number of entries."""
        pass

    @abstractmethod
    def add(self, entries: List[CacheEntry]):
        """Add cache entries."""
        pass

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        context: Dict[str, Any],
        limit: int,
        similarity_threshold: float,
    ) -> List[CacheEntry]:
        """Search by embedding similarity with exact context matching."""
        pass

    @abstractmethod
    def to_arrow(self):
        """Convert to Arrow table."""
        pass

    @abstractmethod
    def delete_by_filter(self, filter_expr: str):
        """Delete entries matching filter."""
        pass

    @abstractmethod
    def has_index(self) -> bool:
        """Check if vector index exists."""
        pass

    @abstractmethod
    def create_index(self, num_partitions: int, num_sub_vectors: int):
        """Create vector index."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all entries."""
        pass
