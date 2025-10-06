"""LFU eviction policy."""

from typing import Dict

from .base import EvictionPolicy


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""

    def __init__(self):
        self.frequencies: Dict[str, int] = {}

    def on_access(self, entry_id: str) -> None:
        self.frequencies[entry_id] = self.frequencies.get(entry_id, 0) + 1

    def on_insert(self, entry_id: str) -> None:
        self.frequencies[entry_id] = 0

    def select_victim(self) -> str:
        if not self.frequencies:
            raise ValueError("No entries to evict")
        victim = min(self.frequencies.items(), key=lambda x: x[1])
        return victim[0]

    def on_evict(self, entry_id: str) -> None:
        if entry_id in self.frequencies:
            del self.frequencies[entry_id]
