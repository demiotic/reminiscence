"""LRU eviction policy with metrics instrumentation."""

import time
from typing import Dict
from .base import EvictionPolicy


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    def __init__(self, metrics=None):
        """
        Initialize LRU policy.

        Args:
            metrics: Optional CacheMetrics instance for tracking
        """
        self.access_times: Dict[str, float] = {}
        self.insertion_times: Dict[str, float] = {}
        self.metrics = metrics
        self.total_accesses = 0

    def on_access(self, entry_id: str) -> None:
        """
        Update access time on hit.

        Args:
            entry_id: Entry that was accessed
        """
        self.access_times[entry_id] = time.time()
        self.total_accesses += 1

        # Track access pattern metrics
        if self.metrics:
            if not hasattr(self.metrics, "lru_total_accesses"):
                self.metrics.lru_total_accesses = 0
            self.metrics.lru_total_accesses += 1

    def on_insert(self, entry_id: str) -> None:
        """
        Record insertion time for new entry.

        Args:
            entry_id: New entry being inserted
        """
        current_time = time.time()
        self.access_times[entry_id] = current_time
        self.insertion_times[entry_id] = current_time

    def select_victim(self) -> str:
        """
        Select least recently used entry for eviction.

        Returns:
            Entry ID to evict

        Raises:
            ValueError: If no entries exist
        """
        if not self.access_times:
            raise ValueError("No entries to evict")

        victim = min(self.access_times.items(), key=lambda x: x[1])
        victim_id = victim[0]
        victim_last_access = victim[1]

        # Track eviction metrics
        if self.metrics:
            now = time.time()

            # Age since last access (recency)
            victim_recency = now - victim_last_access

            if not hasattr(self.metrics, "lru_evicted_recency_seconds"):
                self.metrics.lru_evicted_recency_seconds = []
            self.metrics.lru_evicted_recency_seconds.append(victim_recency)

            if len(self.metrics.lru_evicted_recency_seconds) > 1000:
                self.metrics.lru_evicted_recency_seconds = (
                    self.metrics.lru_evicted_recency_seconds[-1000:]
                )

            # Total age since insertion
            if victim_id in self.insertion_times:
                victim_age = now - self.insertion_times[victim_id]

                if not hasattr(self.metrics, "evicted_entry_ages"):
                    self.metrics.evicted_entry_ages = []
                self.metrics.evicted_entry_ages.append(victim_age)

                if len(self.metrics.evicted_entry_ages) > 1000:
                    self.metrics.evicted_entry_ages = self.metrics.evicted_entry_ages[
                        -1000:
                    ]

        return victim_id

    def on_evict(self, entry_id: str) -> None:
        """
        Remove evicted entry from tracking.

        Args:
            entry_id: Entry that was evicted
        """
        if entry_id in self.access_times:
            del self.access_times[entry_id]

        if entry_id in self.insertion_times:
            del self.insertion_times[entry_id]

        # Track eviction count by policy
        if self.metrics:
            if not hasattr(self.metrics, "evictions"):
                self.metrics.evictions = 0
            self.metrics.evictions += 1

            if not hasattr(self.metrics, "evictions_by_policy"):
                self.metrics.evictions_by_policy = {}
            if "lru" not in self.metrics.evictions_by_policy:
                self.metrics.evictions_by_policy["lru"] = 0
            self.metrics.evictions_by_policy["lru"] += 1

    def get_policy_stats(self) -> dict:
        """
        Get LRU-specific statistics.

        Returns:
            Dict with policy statistics
        """
        if not self.access_times:
            return {
                "policy": "lru",
                "tracked_entries": 0,
                "oldest_access_age_seconds": 0,
                "newest_access_age_seconds": 0,
                "avg_access_age_seconds": 0,
                "total_accesses": self.total_accesses,
            }

        now = time.time()
        recencies = [now - access_time for access_time in self.access_times.values()]

        return {
            "policy": "lru",
            "tracked_entries": len(self.access_times),
            "oldest_access_age_seconds": round(max(recencies), 2),
            "newest_access_age_seconds": round(min(recencies), 2),
            "avg_access_age_seconds": round(sum(recencies) / len(recencies), 2),
            "total_accesses": self.total_accesses,
        }
