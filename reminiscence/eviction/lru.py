"""LRU eviction policy."""

from typing import Dict
from datetime import datetime

from .base import EvictionPolicy


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    def __init__(self):
        self.access_times: Dict[str, float] = {}

    def on_access(self, entry_id: str) -> None:
        self.access_times[entry_id] = datetime.now().timestamp()

    def on_insert(self, entry_id: str) -> None:
        self.access_times[entry_id] = datetime.now().timestamp()

    def select_victim(self) -> str:
        if not self.access_times:
            raise ValueError("No entries to evict")
        victim = min(self.access_times.items(), key=lambda x: x[1])
        return victim[0]

    def on_evict(self, entry_id: str) -> None:
        if entry_id in self.access_times:
            del self.access_times[entry_id]
