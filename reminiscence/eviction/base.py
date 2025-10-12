"""Abstract eviction policy."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EvictionPolicy(ABC):
    """Base class for eviction policies."""

    @abstractmethod
    def on_access(self, entry_id: str) -> None:
        """Called when entry is accessed (cache hit).

        Args:
            entry_id: Unique identifier for accessed entry.
        """
        pass

    @abstractmethod
    def on_insert(self, entry_id: str) -> None:
        """Called when entry is inserted.

        Args:
            entry_id: Unique identifier for new entry.
        """
        pass

    @abstractmethod
    def select_victim(self) -> str:
        """Select entry to evict.

        Returns:
            Entry ID to evict.

        Raises:
            ValueError: If no entries exist.
        """
        pass

    @abstractmethod
    def on_evict(self, entry_id: str) -> None:
        """Called when entry is evicted.

        Args:
            entry_id: Unique identifier for evicted entry.
        """
        pass
