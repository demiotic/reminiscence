"""FIFO eviction policy."""

from .base import EvictionPolicy


class FIFOPolicy(EvictionPolicy):
    """First In First Out eviction policy."""

    def __init__(self):
        self.queue = []

    def on_access(self, entry_id: str) -> None:
        pass  # FIFO doesn't track accesses

    def on_insert(self, entry_id: str) -> None:
        self.queue.append(entry_id)

    def select_victim(self) -> str:
        if not self.queue:
            raise ValueError("No entries to evict")
        return self.queue[0]

    def on_evict(self, entry_id: str) -> None:
        if entry_id in self.queue:
            self.queue.remove(entry_id)
