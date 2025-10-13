"""Enhanced metrics for Reminiscence."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class CacheMetrics:
    """Cache performance metrics.

    Tracks hits/misses, latencies, result sizes, eviction, storage,
    embedding, and scheduler metrics.
    """

    # Sample limit to avoid memory leak (used to create deque maxlen)
    max_samples: int = 1000

    # Cache-level metrics
    hits: int = 0
    misses: int = 0
    total_latency_saved_ms: float = 0.0

    # Use deque with maxlen for auto-truncation (prevents memory leaks)
    # These are initialized in __post_init__ using max_samples
    lookup_latencies_ms: "deque[float]" = field(init=False)

    store_errors: int = 0
    lookup_errors: int = 0

    result_sizes_bytes: "deque[int]" = field(init=False)

    # Eviction metrics (general)
    evictions: int = 0
    evictions_by_policy: Dict[str, int] = field(default_factory=dict)
    evicted_entry_ages: "deque[float]" = field(init=False)

    # LFU-specific metrics
    lfu_total_accesses: int = 0
    lfu_evicted_frequencies: "deque[int]" = field(init=False)

    # LRU-specific metrics
    lru_total_accesses: int = 0
    lru_evicted_recency_seconds: "deque[float]" = field(init=False)

    # Storage metrics
    storage_searches: int = 0
    storage_adds: int = 0
    storage_search_latencies_ms: "deque[float]" = field(init=False)
    storage_add_latencies_ms: "deque[float]" = field(init=False)
    storage_search_errors: int = 0
    storage_add_errors: int = 0

    # Embedding metrics
    embedding_generations: int = 0
    embedding_latencies_ms: "deque[float]" = field(init=False)
    embedding_errors: int = 0

    # Scheduler metrics
    scheduler_runs: int = 0
    scheduler_cleanup_latencies_ms: "deque[float]" = field(init=False)
    scheduler_errors: int = 0

    def __post_init__(self) -> None:
        """Initialize deques with configurable maxlen and thread lock.

        Uses max_samples to set deque maxlen.
        """
        # Thread safety: Single lock for all metrics operations
        # This ensures accurate counters under concurrent access with minimal overhead
        self._lock = threading.RLock()  # Reentrant lock for nested calls

        self.lookup_latencies_ms = deque(maxlen=self.max_samples)
        self.result_sizes_bytes = deque(maxlen=self.max_samples)
        self.evicted_entry_ages = deque(maxlen=self.max_samples)
        self.lfu_evicted_frequencies = deque(maxlen=self.max_samples)
        self.lru_evicted_recency_seconds = deque(maxlen=self.max_samples)
        self.storage_search_latencies_ms = deque(maxlen=self.max_samples)
        self.storage_add_latencies_ms = deque(maxlen=self.max_samples)
        self.embedding_latencies_ms = deque(maxlen=self.max_samples)
        self.scheduler_cleanup_latencies_ms = deque(maxlen=self.max_samples)

    def record_lookup_latency(self, latency_ms: float) -> None:
        """Record lookup latency (auto-truncates with deque).

        Args:
            latency_ms: Lookup latency in milliseconds.
        """
        with self._lock:
            self.lookup_latencies_ms.append(latency_ms)

    def record_result_size(self, size_bytes: int) -> None:
        """Record cached result size (auto-truncates with deque).

        Args:
            size_bytes: Size of cached result in bytes.
        """
        with self._lock:
            self.result_sizes_bytes.append(size_bytes)

    @property
    def hit_rate(self) -> float:
        """Hit rate (0.0 - 1.0).

        Returns:
            Cache hit rate as a float between 0.0 and 1.0.
        """
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total requests.

        Returns:
            Total number of cache requests (hits + misses).
        """
        with self._lock:
            return self.hits + self.misses

    @property
    def eviction_rate(self) -> float:
        """Evictions per request.

        Returns:
            Eviction rate as a float.
        """
        with self._lock:
            total = self.hits + self.misses
            return self.evictions / total if total > 0 else 0.0

    def get_percentiles(
        self, values: Union["deque[float]", List[float]]
    ) -> Dict[str, float]:
        """Calculate percentiles for a deque or list of values.

        Args:
            values: Collection of numeric values.

        Returns:
            Dict with p50, p95, and p99 percentiles.
        """
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return {
            "p50": sorted_vals[int(n * 0.50)] if n > 0 else 0.0,
            "p95": sorted_vals[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_vals[int(n * 0.99)] if n > 0 else 0.0,
        }

    def report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report.

        Returns:
            Dict with all metrics including cache, eviction, storage,
            embedding, and scheduler metrics.
        """
        # Lock for the entire report to ensure consistent snapshot of all data
        with self._lock:
            # Copy all data while holding lock
            hits = self.hits
            misses = self.misses
            total_latency_saved = self.total_latency_saved_ms
            lookup_errors = self.lookup_errors
            store_errors = self.store_errors
            evictions = self.evictions
            evictions_by_policy_copy = dict(self.evictions_by_policy)

            # Copy deques for percentile calculations
            lookup_latencies = list(self.lookup_latencies_ms)
            result_sizes = list(self.result_sizes_bytes)
            evicted_ages = list(self.evicted_entry_ages)
            storage_search_lats = list(self.storage_search_latencies_ms)
            storage_add_lats = list(self.storage_add_latencies_ms)
            embedding_lats = list(self.embedding_latencies_ms)
            scheduler_lats = list(self.scheduler_cleanup_latencies_ms)

            # Copy policy-specific metrics
            lfu_total = self.lfu_total_accesses
            lfu_freqs = (
                list(self.lfu_evicted_frequencies)
                if self.lfu_evicted_frequencies
                else []
            )
            lru_total = self.lru_total_accesses
            lru_recency = (
                list(self.lru_evicted_recency_seconds)
                if self.lru_evicted_recency_seconds
                else []
            )

            # Copy storage/embedding/scheduler metrics
            storage_searches = self.storage_searches
            storage_adds = self.storage_adds
            storage_search_errors = self.storage_search_errors
            storage_add_errors = self.storage_add_errors
            embedding_generations = self.embedding_generations
            embedding_errors = self.embedding_errors
            scheduler_runs = self.scheduler_runs
            scheduler_errors = self.scheduler_errors

        # Now calculate percentiles without holding lock (expensive operation)
        latency_percentiles = self.get_percentiles(lookup_latencies)
        size_percentiles = self.get_percentiles([float(s) for s in result_sizes])
        evicted_age_percentiles = self.get_percentiles(evicted_ages)
        storage_search_percentiles = self.get_percentiles(storage_search_lats)
        storage_add_percentiles = self.get_percentiles(storage_add_lats)
        embedding_percentiles = self.get_percentiles(embedding_lats)
        scheduler_percentiles = self.get_percentiles(scheduler_lats)

        # Calculate derived metrics
        total_requests = hits + misses
        hit_rate = hits / total_requests if total_requests > 0 else 0.0
        eviction_rate = evictions / total_requests if total_requests > 0 else 0.0

        # Build eviction report with policy-specific metrics
        eviction_report = {
            "total_evictions": evictions,
            "eviction_rate": f"{eviction_rate * 100:.2f}%",
            "by_policy": evictions_by_policy_copy,
            "evicted_entry_age_seconds": {
                "p50": round(evicted_age_percentiles["p50"], 2),
                "p95": round(evicted_age_percentiles["p95"], 2),
                "p99": round(evicted_age_percentiles["p99"], 2),
                "samples": len(evicted_ages),
            },
        }

        # Add LFU-specific metrics if available
        if lfu_freqs:
            lfu_freq_percentiles = self.get_percentiles([float(f) for f in lfu_freqs])
            eviction_report["lfu_metrics"] = {
                "total_accesses": lfu_total,
                "evicted_frequencies": {
                    "p50": round(lfu_freq_percentiles["p50"], 2),
                    "p95": round(lfu_freq_percentiles["p95"], 2),
                    "p99": round(lfu_freq_percentiles["p99"], 2),
                    "samples": len(lfu_freqs),
                },
            }

        # Add LRU-specific metrics if available
        if lru_recency:
            lru_recency_percentiles = self.get_percentiles(lru_recency)
            eviction_report["lru_metrics"] = {
                "total_accesses": lru_total,
                "evicted_recency_seconds": {
                    "p50": round(lru_recency_percentiles["p50"], 2),
                    "p95": round(lru_recency_percentiles["p95"], 2),
                    "p99": round(lru_recency_percentiles["p99"], 2),
                    "samples": len(lru_recency),
                },
            }

        return {
            # Cache metrics
            "hits": hits,
            "misses": misses,
            "total_requests": total_requests,
            "hit_rate": f"{hit_rate * 100:.2f}%",
            "total_latency_saved_ms": round(total_latency_saved, 1),
            "avg_latency_saved_ms": round(
                total_latency_saved / hits if hits > 0 else 0.0, 1
            ),
            "lookup_latency_ms": {
                "p50": round(latency_percentiles["p50"], 2),
                "p95": round(latency_percentiles["p95"], 2),
                "p99": round(latency_percentiles["p99"], 2),
                "samples": len(lookup_latencies),
            },
            "errors": {
                "lookup": lookup_errors,
                "store": store_errors,
            },
            "result_size_bytes": {
                "p50": int(size_percentiles["p50"]),
                "p95": int(size_percentiles["p95"]),
                "p99": int(size_percentiles["p99"]),
                "samples": len(result_sizes),
            },
            # Eviction metrics
            "eviction": eviction_report,
            # Storage metrics
            "storage": {
                "total_searches": storage_searches,
                "total_adds": storage_adds,
                "search_latency_ms": {
                    "p50": round(storage_search_percentiles["p50"], 2),
                    "p95": round(storage_search_percentiles["p95"], 2),
                    "p99": round(storage_search_percentiles["p99"], 2),
                    "samples": len(storage_search_lats),
                },
                "add_latency_ms": {
                    "p50": round(storage_add_percentiles["p50"], 2),
                    "p95": round(storage_add_percentiles["p95"], 2),
                    "p99": round(storage_add_percentiles["p99"], 2),
                    "samples": len(storage_add_lats),
                },
                "errors": {
                    "search": storage_search_errors,
                    "add": storage_add_errors,
                },
            },
            # Embedding metrics
            "embedding": {
                "total_generations": embedding_generations,
                "latency_ms": {
                    "p50": round(embedding_percentiles["p50"], 2),
                    "p95": round(embedding_percentiles["p95"], 2),
                    "p99": round(embedding_percentiles["p99"], 2),
                    "samples": len(embedding_lats),
                },
                "errors": embedding_errors,
            },
            # Scheduler metrics
            "scheduler": {
                "total_runs": scheduler_runs,
                "cleanup_latency_ms": {
                    "p50": round(scheduler_percentiles["p50"], 2),
                    "p95": round(scheduler_percentiles["p95"], 2),
                    "p99": round(scheduler_percentiles["p99"], 2),
                    "samples": len(scheduler_lats),
                },
                "errors": scheduler_errors,
            },
        }

    def reset(self) -> None:
        """Reset all metrics to zero."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.lookup_errors = 0
            self.store_errors = 0
            self.total_latency_saved_ms = 0.0
            self.lookup_latencies_ms.clear()
            self.result_sizes_bytes.clear()

            # Reset eviction metrics
            self.evictions = 0
            self.evictions_by_policy.clear()
            self.evicted_entry_ages.clear()

            # Reset LFU metrics
            self.lfu_total_accesses = 0
            self.lfu_evicted_frequencies.clear()

            # Reset LRU metrics
            self.lru_total_accesses = 0
            self.lru_evicted_recency_seconds.clear()

            # Reset storage metrics
            self.storage_searches = 0
            self.storage_adds = 0
            self.storage_search_latencies_ms.clear()
            self.storage_add_latencies_ms.clear()
            self.storage_search_errors = 0
            self.storage_add_errors = 0

            # Reset embedding metrics
            self.embedding_generations = 0
            self.embedding_latencies_ms.clear()
            self.embedding_errors = 0

            # Reset scheduler metrics
            self.scheduler_runs = 0
            self.scheduler_cleanup_latencies_ms.clear()
            self.scheduler_errors = 0
