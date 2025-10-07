"""Abstract eviction policy."""

from abc import ABC, abstractmethod


class EvictionPolicy(ABC):
    """Base class for eviction policies."""

    @abstractmethod
    def on_access(self, entry_id: str) -> None:
        """Called when entry is accessed (cache hit)."""
        pass

    @abstractmethod
    def on_insert(self, entry_id: str) -> None:
        """Called when entry is inserted."""
        pass

    @abstractmethod
    def select_victim(self) -> str:
        """Select entry to evict. Returns entry_id."""
        pass

    @abstractmethod
    def on_evict(self, entry_id: str) -> None:
        """Called when entry is evicted."""
        pass
