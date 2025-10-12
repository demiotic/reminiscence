"""LFU eviction policy with metrics instrumentation."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .base import EvictionPolicy


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""

    def __init__(self, metrics: Optional[Any] = None):
        """Initialize LFU policy.

        Args:
            metrics: Optional CacheMetrics instance for tracking.
        """
        self.frequencies: Dict[str, int] = {}
        self.insertion_times: Dict[str, float] = {}
        self.metrics = metrics
        self.total_accesses = 0
        self.last_inserted: Optional[str] = None  # Track most recent insert

    def on_access(self, entry_id: str) -> None:
        """Increment frequency counter on access.

        Args:
            entry_id: Entry that was accessed.
        """
        self.frequencies[entry_id] = self.frequencies.get(entry_id, 0) + 1
        self.total_accesses += 1

        # Track access pattern metrics
        if self.metrics:
            if not hasattr(self.metrics, "lfu_total_accesses"):
                self.metrics.lfu_total_accesses = 0
            self.metrics.lfu_total_accesses += 1

    def on_insert(self, entry_id: str) -> None:
        """Initialize frequency counter for new entry.

        Args:
            entry_id: New entry being inserted.
        """
        self.frequencies[entry_id] = 0
        self.insertion_times[entry_id] = time.time()
        self.last_inserted = entry_id  # Track for immediate eviction protection

    def select_victim(self) -> str:
        """Select least frequently used entry for eviction.

        Uses standard LFU semantics: evicts entry with lowest frequency.
        Tie-breaker: oldest entry (FIFO among equal frequencies).

        Protects the most recently inserted entry from immediate eviction
        by excluding it from victim selection if it triggered the eviction.

        Returns:
            Entry ID to evict.

        Raises:
            ValueError: If no entries exist.
        """
        if not self.frequencies:
            raise ValueError("No entries to evict")

        # Exclude the most recently inserted entry to prevent immediate eviction
        candidates = {
            k: v
            for k, v in self.frequencies.items()
            if k != self.last_inserted or len(self.frequencies) == 1
        }

        # If we filtered everything out (shouldn't happen), use all entries
        if not candidates:
            candidates = self.frequencies

        # Select victim: lowest frequency first, then oldest
        victim = min(
            candidates.items(),
            key=lambda x: (x[1], self.insertion_times.get(x[0], float("inf"))),
        )

        victim_id = victim[0]
        victim_freq = victim[1]

        # Track eviction metrics
        if self.metrics:
            # Track age of evicted entry
            if victim_id in self.insertion_times:
                victim_age = time.time() - self.insertion_times[victim_id]

                if not hasattr(self.metrics, "evicted_entry_ages"):
                    self.metrics.evicted_entry_ages = []
                self.metrics.evicted_entry_ages.append(victim_age)

                if len(self.metrics.evicted_entry_ages) > 1000:
                    self.metrics.evicted_entry_ages = self.metrics.evicted_entry_ages[
                        -1000:
                    ]

            # Track frequency of evicted entries
            if not hasattr(self.metrics, "lfu_evicted_frequencies"):
                self.metrics.lfu_evicted_frequencies = []
            self.metrics.lfu_evicted_frequencies.append(victim_freq)

            if len(self.metrics.lfu_evicted_frequencies) > 1000:
                self.metrics.lfu_evicted_frequencies = (
                    self.metrics.lfu_evicted_frequencies[-1000:]
                )

        return victim_id

    def on_evict(self, entry_id: str) -> None:
        """Remove evicted entry from tracking.

        Args:
            entry_id: Entry that was evicted.
        """
        if entry_id in self.frequencies:
            del self.frequencies[entry_id]

        if entry_id in self.insertion_times:
            del self.insertion_times[entry_id]

        # Clear last_inserted if we evicted it (shouldn't happen, but defensive)
        if self.last_inserted == entry_id:
            self.last_inserted = None

        # Track eviction count by policy
        if self.metrics:
            if not hasattr(self.metrics, "evictions"):
                self.metrics.evictions = 0
            self.metrics.evictions += 1

            if not hasattr(self.metrics, "evictions_by_policy"):
                self.metrics.evictions_by_policy = {}
            if "lfu" not in self.metrics.evictions_by_policy:
                self.metrics.evictions_by_policy["lfu"] = 0
            self.metrics.evictions_by_policy["lfu"] += 1

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get LFU-specific statistics.

        Returns:
            Dict with policy statistics.
        """
        if not self.frequencies:
            return {
                "policy": "lfu",
                "tracked_entries": 0,
                "min_frequency": 0,
                "max_frequency": 0,
                "avg_frequency": 0,
                "total_accesses": self.total_accesses,
            }

        freqs = list(self.frequencies.values())

        return {
            "policy": "lfu",
            "tracked_entries": len(self.frequencies),
            "min_frequency": min(freqs),
            "max_frequency": max(freqs),
            "avg_frequency": round(sum(freqs) / len(freqs), 2),
            "total_accesses": self.total_accesses,
        }
