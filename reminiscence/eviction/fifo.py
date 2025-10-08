"""FIFO eviction policy with metrics instrumentation."""

import time
from .base import EvictionPolicy


class FIFOPolicy(EvictionPolicy):
    """First In First Out eviction policy."""

    def __init__(self, metrics=None):
        """
        Initialize FIFO policy.

        Args:
            metrics: Optional CacheMetrics instance for tracking
        """
        self.queue = []
        self.metrics = metrics
        self.insertion_times = {}  # Track when entries were added

    def on_access(self, entry_id: str) -> None:
        """FIFO doesn't track accesses."""
        pass

    def on_insert(self, entry_id: str) -> None:
        """
        Track new entry insertion.

        Args:
            entry_id: Unique identifier for cache entry
        """
        self.queue.append(entry_id)
        self.insertion_times[entry_id] = time.time()

    def select_victim(self) -> str:
        """
        Select oldest entry for eviction.

        Returns:
            Entry ID to evict

        Raises:
            ValueError: If no entries exist
        """
        if not self.queue:
            raise ValueError("No entries to evict")

        victim_id = self.queue[0]

        # Track eviction metrics
        if self.metrics and victim_id in self.insertion_times:
            victim_age = time.time() - self.insertion_times[victim_id]

            if not hasattr(self.metrics, "evicted_entry_ages"):
                self.metrics.evicted_entry_ages = []
            self.metrics.evicted_entry_ages.append(victim_age)

            # Keep only last 1000 samples
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
        if entry_id in self.queue:
            self.queue.remove(entry_id)

        if entry_id in self.insertion_times:
            del self.insertion_times[entry_id]

        # Track eviction count by policy
        if self.metrics:
            if not hasattr(self.metrics, "evictions"):
                self.metrics.evictions = 0
            self.metrics.evictions += 1

            if not hasattr(self.metrics, "evictions_by_policy"):
                self.metrics.evictions_by_policy = {}
            if "fifo" not in self.metrics.evictions_by_policy:
                self.metrics.evictions_by_policy["fifo"] = 0
            self.metrics.evictions_by_policy["fifo"] += 1

    def get_policy_stats(self) -> dict:
        """
        Get FIFO-specific statistics.

        Returns:
            Dict with policy statistics
        """
        if not self.queue:
            return {
                "policy": "fifo",
                "tracked_entries": 0,
                "oldest_entry_age_seconds": 0,
                "newest_entry_age_seconds": 0,
                "avg_entry_age_seconds": 0,
            }

        now = time.time()
        ages = [
            now - self.insertion_times[entry_id]
            for entry_id in self.queue
            if entry_id in self.insertion_times
        ]

        if not ages:
            return {
                "policy": "fifo",
                "tracked_entries": len(self.queue),
                "oldest_entry_age_seconds": 0,
                "newest_entry_age_seconds": 0,
                "avg_entry_age_seconds": 0,
            }

        return {
            "policy": "fifo",
            "tracked_entries": len(self.queue),
            "oldest_entry_age_seconds": round(max(ages), 2),
            "newest_entry_age_seconds": round(min(ages), 2),
            "avg_entry_age_seconds": round(sum(ages) / len(ages), 2),
        }
