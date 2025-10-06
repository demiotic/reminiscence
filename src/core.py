"""Main Memora class - Facade for all components."""

import time
from typing import Any, Dict, Optional

from .config import CacheConfig
from .types import LookupResult, AvailabilityCheck
from .embeddings import create_embedder
from .storage import create_storage_backend
from .eviction import create_eviction_policy
from .cache import CacheOperations
from .metrics import CacheMetrics
from .utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


class Memora:
    """
    Semantic cache for LLMs and multi-agent systems.

    Hybrid matching: semantic similarity + exact context matching.

    Main API:
    - lookup(): Search for existing result
    - store(): Save new result
    - check_availability(): Verify availability without retrieving data
    - invalidate(): Mark entries as invalid
    - cleanup_expired(): Remove expired entries
    - create_index(): Create vector index for fast searches
    - get_stats(): Return cache statistics
    - health_check(): Perform health checks
    - cached(): Decorator for automatic caching

    Example:
        >>> cache = Memora()
        >>>
        >>> # Manual usage with context
        >>> result = cache.lookup("What is ML?", {"agent": "qa", "model": "gpt-4"})
        >>> if result.is_hit:
        ...     print(result.result)
        ... else:
        ...     data = expensive_llm_call()
        ...     cache.store("What is ML?", {"agent": "qa", "model": "gpt-4"}, data)
        >>>
        >>> # Decorator usage
        >>> @cache.cached(query_param="question", strict_params=["model"])
        >>> def ask_llm(question: str, model: str):
        ...     return expensive_llm_call(question, model)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize Memora with all components.

        Args:
            config: Cache configuration. If None, loads from environment variables.
        """
        self.config = config or CacheConfig.load()

        # Setup logging
        configure_logging(self.config.log_level, self.config.json_logs)

        logger.info(
            "initializing_memora",
            model=self.config.model_name,
            db_uri=self.config.db_uri,
            eviction=self.config.eviction_policy,
        )

        # Initialize components
        self.embedder = create_embedder(self.config)
        self.backend = create_storage_backend(self.config, self.embedder.embedding_dim)
        self.eviction = create_eviction_policy(self.config.eviction_policy)
        self.metrics = CacheMetrics() if self.config.enable_metrics else None

        # Single operations handler (lookup, store, maintenance)
        self.ops = CacheOperations(
            storage=self.backend,
            embedder=self.embedder,
            eviction=self.eviction,
            config=self.config,
            metrics=self.metrics,
        )

        logger.info(
            "memora_ready",
            entries=self.backend.count(),
            max_entries=self.config.max_entries,
            embedding_dim=self.embedder.embedding_dim,
            threshold=self.config.similarity_threshold,
        )

    # ====================
    # PUBLIC API - Delegate to ops
    # ====================

    def clear(self):
        """
        Clear all cache entries and reset metrics.

        Useful for testing or manual cache management.

        Example:
            >>> cache = Memora()
            >>> cache.store("test", {}, "result")
            >>> cache.clear()
            >>> assert cache.backend.count() == 0
        """
        self.backend.clear()
        if self.metrics:
            self.metrics.reset()

    def lookup(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        _track_metrics: Optional[bool] = True,
    ) -> LookupResult:
        """
        Search cache entry by semantic similarity with exact context matching.

        Args:
            query: Query text to search
            context: Context dict for exact matching (agent_id, tools, model, etc)
            similarity_threshold: Minimum similarity score (0-1)
            _track_metrics: Internal flag to control metrics tracking

        Returns:
            LookupResult with hit status and cached data
        """
        return self.ops.lookup(query, context, similarity_threshold, _track_metrics)

    def store(
        self,
        query: str,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store result in cache.

        Args:
            query: Query text
            context: Context dict (will be matched exactly in future lookups)
            result: Result to cache (supports JSON, Arrow, Pandas, Polars)
            metadata: Optional metadata
        """
        self.ops.store(query, context, result, metadata)

    def invalidate(
        self,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """
        Invalidate cache entries by criteria.

        Args:
            query: Query text (not implemented yet)
            context: Context dict for exact matching
            older_than_seconds: Invalidate entries older than this

        Returns:
            Number of invalidated entries
        """
        return self.ops.invalidate(query, context, older_than_seconds)

    def cleanup_expired(self) -> int:
        """
        Clean expired entries according to configured TTL.

        Returns:
            Number of deleted entries
        """
        return self.ops.cleanup_expired()

    def check_availability(
        self,
        query: str,
        context: Dict[str, Any],
        similarity_threshold: Optional[float] = None,
    ) -> AvailabilityCheck:
        """
        Verify availability without retrieving full data.

        Useful for schedulers and pre-checks.
        """
        result = self.lookup(query, context, similarity_threshold, _track_metrics=False)

        if not result.is_hit:
            return AvailabilityCheck(available=False)

        ttl_remaining = None
        if self.config.ttl_seconds and result.age_seconds is not None:
            ttl_remaining = self.config.ttl_seconds - result.age_seconds

        return AvailabilityCheck(
            available=True,
            age_seconds=result.age_seconds,
            ttl_remaining_seconds=ttl_remaining,
            similarity=result.similarity,
        )

    # ====================
    # INDEX & STATS
    # ====================

    def create_index(
        self,
        num_partitions: int = 256,
        num_sub_vectors: Optional[int] = None,
    ) -> None:
        """Create IVF-PQ index for fast vector searches."""
        row_count = self.backend.count()

        if row_count < 256:
            logger.warning(
                "insufficient_entries_for_index", count=row_count, minimum=256
            )
            return

        if num_sub_vectors is None:
            num_sub_vectors = max(1, self.embedder.embedding_dim // 4)

        logger.info(
            "creating_index",
            partitions=num_partitions,
            sub_vectors=num_sub_vectors,
            entries=row_count,
        )

        self.backend.create_index(num_partitions, num_sub_vectors)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        stats = {
            "total_entries": self.backend.count(),
            "max_entries": self.config.max_entries,
            "eviction_policy": self.config.eviction_policy,
            "threshold": self.config.similarity_threshold,
            "embedding_dim": self.embedder.embedding_dim,
            "model": self.config.model_name,
            "ttl_seconds": self.config.ttl_seconds,
            "storage": self.config.db_uri,
            "index_created": self.backend.has_index(),
        }

        if self.metrics:
            stats.update(self.metrics.report())

        return stats

    def get_index_stats(self) -> Dict[str, Any]:
        """Return vector index statistics."""
        return {
            "has_index": self.backend.has_index(),
            "total_entries": self.backend.count(),
            "note": "LanceDB doesn't expose detailed index metrics",
        }

    # ====================
    # HEALTH CHECK
    # ====================

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache components."""
        checks = {
            "embedding": {"ok": True, "error": None},
            "database": {"ok": True, "error": None},
            "error_rate": {"ok": True, "details": "No metrics available"},
        }

        # Test embedding
        try:
            test_embedding = self.embedder.embed("health check test")
            if len(test_embedding) != self.embedder.embedding_dim:
                checks["embedding"]["ok"] = False
                checks["embedding"]["error"] = (
                    f"Embedding dimension mismatch: {len(test_embedding)} != {self.embedder.embedding_dim}"
                )
        except Exception as e:
            checks["embedding"]["ok"] = False
            checks["embedding"]["error"] = str(e)
            logger.error("health_check_embedding_failed", error=str(e), exc_info=True)

        # Test database
        try:
            entry_count = self.backend.count()
            if entry_count > 0:
                _ = self.backend.to_arrow()
        except Exception as e:
            checks["database"]["ok"] = False
            checks["database"]["error"] = str(e)
            logger.error("health_check_database_failed", error=str(e), exc_info=True)

        # Check error rates
        if self.metrics:
            total_requests = self.metrics.total_requests
            lookup_errors = self.metrics.lookup_errors
            store_errors = self.metrics.store_errors
            total_errors = lookup_errors + store_errors

            if total_requests >= 10:
                error_rate = total_errors / total_requests if total_requests > 0 else 0
                if error_rate > 0.10:
                    checks["error_rate"]["ok"] = False
                    checks["error_rate"]["details"] = (
                        f"High error rate: {error_rate * 100:.1f}% "
                        f"({total_errors}/{total_requests} requests)"
                    )
                else:
                    checks["error_rate"]["ok"] = True
                    checks["error_rate"]["details"] = (
                        f"Error rate: {error_rate * 100:.1f}% "
                        f"({total_errors}/{total_requests} requests)"
                    )
            else:
                checks["error_rate"]["ok"] = True
                checks["error_rate"]["details"] = (
                    f"Insufficient data: {total_requests} requests"
                )

        # Overall status
        all_checks_ok = all(check["ok"] for check in checks.values())
        status = "healthy" if all_checks_ok else "unhealthy"

        response = {
            "status": status,
            "checks": checks,
            "metrics": {
                "total_entries": self.backend.count()
                if checks["database"]["ok"]
                else 0,
                "recent_errors": {
                    "lookup": self.metrics.lookup_errors if self.metrics else 0,
                    "store": self.metrics.store_errors if self.metrics else 0,
                },
            },
            "timestamp": int(time.time() * 1000),
        }

        if status == "unhealthy":
            logger.warning("health_check_failed", response=response)
        else:
            logger.debug("health_check_passed")

        return response

    # ====================
    # DECORATOR
    # ====================

    def cached(
        self,
        query_param: str = "query",
        strict_params: Optional[list] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
    ):
        """Decorator to cache function results with hybrid matching."""
        from .decorators import create_cached_decorator

        decorator_factory = create_cached_decorator(self)
        return decorator_factory(
            query_param=query_param,
            strict_params=strict_params,
            static_context=static_context,
            auto_strict=auto_strict,
        )
