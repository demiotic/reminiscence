"""Enhanced metrics for Reminiscence."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CacheMetrics:
    """
    Cache performance metrics.

    Tracks hits/misses, latencies, result sizes, eviction, storage,
    embedding, and scheduler metrics.
    """

    # Cache-level metrics
    hits: int = 0
    misses: int = 0
    total_latency_saved_ms: float = 0.0
    lookup_latencies_ms: List[float] = field(default_factory=list)
    store_errors: int = 0
    lookup_errors: int = 0
    result_sizes_bytes: List[int] = field(default_factory=list)

    # Eviction metrics (general)
    evictions: int = 0
    evictions_by_policy: Dict[str, int] = field(default_factory=dict)
    evicted_entry_ages: List[float] = field(default_factory=list)

    # LFU-specific metrics
    lfu_total_accesses: int = 0
    lfu_evicted_frequencies: List[int] = field(default_factory=list)

    # LRU-specific metrics
    lru_total_accesses: int = 0
    lru_evicted_recency_seconds: List[float] = field(default_factory=list)

    # Storage metrics
    storage_searches: int = 0
    storage_adds: int = 0
    storage_search_latencies_ms: List[float] = field(default_factory=list)
    storage_add_latencies_ms: List[float] = field(default_factory=list)
    storage_search_errors: int = 0
    storage_add_errors: int = 0

    # Embedding metrics
    embedding_generations: int = 0
    embedding_latencies_ms: List[float] = field(default_factory=list)
    embedding_errors: int = 0

    # Scheduler metrics
    scheduler_runs: int = 0
    scheduler_cleanup_latencies_ms: List[float] = field(default_factory=list)
    scheduler_errors: int = 0

    # Sample limit to avoid memory leak
    max_samples: int = 10000

    def record_lookup_latency(self, latency_ms: float):
        """Record lookup latency."""
        self.lookup_latencies_ms.append(latency_ms)

        if len(self.lookup_latencies_ms) > self.max_samples:
            self.lookup_latencies_ms = self.lookup_latencies_ms[-self.max_samples :]

    def record_result_size(self, size_bytes: int):
        """Record cached result size."""
        self.result_sizes_bytes.append(size_bytes)

        if len(self.result_sizes_bytes) > self.max_samples:
            self.result_sizes_bytes = self.result_sizes_bytes[-self.max_samples :]

    @property
    def hit_rate(self) -> float:
        """Hit rate (0.0 - 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total requests."""
        return self.hits + self.misses

    @property
    def eviction_rate(self) -> float:
        """Evictions per request."""
        total = self.total_requests
        return self.evictions / total if total > 0 else 0.0

    def get_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentiles for a list of values."""
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
        """
        Generate comprehensive metrics report.

        Returns:
            Dict with all metrics including cache, eviction, storage,
            embedding, and scheduler metrics
        """
        latency_percentiles = self.get_percentiles(self.lookup_latencies_ms)
        size_percentiles = self.get_percentiles(
            [float(s) for s in self.result_sizes_bytes]
        )

        # Eviction age percentiles
        evicted_age_percentiles = self.get_percentiles(self.evicted_entry_ages)

        # Storage latency percentiles
        storage_search_percentiles = self.get_percentiles(
            self.storage_search_latencies_ms
        )
        storage_add_percentiles = self.get_percentiles(self.storage_add_latencies_ms)

        # Embedding latency percentiles
        embedding_percentiles = self.get_percentiles(self.embedding_latencies_ms)

        # Scheduler latency percentiles
        scheduler_percentiles = self.get_percentiles(
            self.scheduler_cleanup_latencies_ms
        )

        # Build eviction report with policy-specific metrics
        eviction_report = {
            "total_evictions": self.evictions,
            "eviction_rate": f"{self.eviction_rate * 100:.2f}%",
            "by_policy": self.evictions_by_policy,
            "evicted_entry_age_seconds": {
                "p50": round(evicted_age_percentiles["p50"], 2),
                "p95": round(evicted_age_percentiles["p95"], 2),
                "p99": round(evicted_age_percentiles["p99"], 2),
                "samples": len(self.evicted_entry_ages),
            },
        }

        # Add LFU-specific metrics if available
        if self.lfu_evicted_frequencies:
            lfu_freq_percentiles = self.get_percentiles(
                [float(f) for f in self.lfu_evicted_frequencies]
            )
            eviction_report["lfu_metrics"] = {
                "total_accesses": self.lfu_total_accesses,
                "evicted_frequencies": {
                    "p50": round(lfu_freq_percentiles["p50"], 2),
                    "p95": round(lfu_freq_percentiles["p95"], 2),
                    "p99": round(lfu_freq_percentiles["p99"], 2),
                    "samples": len(self.lfu_evicted_frequencies),
                },
            }

        # Add LRU-specific metrics if available
        if self.lru_evicted_recency_seconds:
            lru_recency_percentiles = self.get_percentiles(
                self.lru_evicted_recency_seconds
            )
            eviction_report["lru_metrics"] = {
                "total_accesses": self.lru_total_accesses,
                "evicted_recency_seconds": {
                    "p50": round(lru_recency_percentiles["p50"], 2),
                    "p95": round(lru_recency_percentiles["p95"], 2),
                    "p99": round(lru_recency_percentiles["p99"], 2),
                    "samples": len(self.lru_evicted_recency_seconds),
                },
            }

        return {
            # Cache metrics
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate * 100:.2f}%",
            "total_latency_saved_ms": round(self.total_latency_saved_ms, 1),
            "avg_latency_saved_ms": round(
                self.total_latency_saved_ms / self.hits if self.hits > 0 else 0.0, 1
            ),
            "lookup_latency_ms": {
                "p50": round(latency_percentiles["p50"], 2),
                "p95": round(latency_percentiles["p95"], 2),
                "p99": round(latency_percentiles["p99"], 2),
                "samples": len(self.lookup_latencies_ms),
            },
            "errors": {
                "lookup": self.lookup_errors,
                "store": self.store_errors,
            },
            "result_size_bytes": {
                "p50": int(size_percentiles["p50"]),
                "p95": int(size_percentiles["p95"]),
                "p99": int(size_percentiles["p99"]),
                "samples": len(self.result_sizes_bytes),
            },
            # Eviction metrics
            "eviction": eviction_report,
            # Storage metrics
            "storage": {
                "total_searches": self.storage_searches,
                "total_adds": self.storage_adds,
                "search_latency_ms": {
                    "p50": round(storage_search_percentiles["p50"], 2),
                    "p95": round(storage_search_percentiles["p95"], 2),
                    "p99": round(storage_search_percentiles["p99"], 2),
                    "samples": len(self.storage_search_latencies_ms),
                },
                "add_latency_ms": {
                    "p50": round(storage_add_percentiles["p50"], 2),
                    "p95": round(storage_add_percentiles["p95"], 2),
                    "p99": round(storage_add_percentiles["p99"], 2),
                    "samples": len(self.storage_add_latencies_ms),
                },
                "errors": {
                    "search": self.storage_search_errors,
                    "add": self.storage_add_errors,
                },
            },
            # Embedding metrics
            "embedding": {
                "total_generations": self.embedding_generations,
                "latency_ms": {
                    "p50": round(embedding_percentiles["p50"], 2),
                    "p95": round(embedding_percentiles["p95"], 2),
                    "p99": round(embedding_percentiles["p99"], 2),
                    "samples": len(self.embedding_latencies_ms),
                },
                "errors": self.embedding_errors,
            },
            # Scheduler metrics
            "scheduler": {
                "total_runs": self.scheduler_runs,
                "cleanup_latency_ms": {
                    "p50": round(scheduler_percentiles["p50"], 2),
                    "p95": round(scheduler_percentiles["p95"], 2),
                    "p99": round(scheduler_percentiles["p99"], 2),
                    "samples": len(self.scheduler_cleanup_latencies_ms),
                },
                "errors": self.scheduler_errors,
            },
        }

    def reset(self):
        """Reset all metrics to zero."""
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
