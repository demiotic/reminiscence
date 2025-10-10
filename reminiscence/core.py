"""Core Reminiscence class - Facade for all components."""

import time
from typing import Any, Dict, Optional, List, Union

from .config import ReminiscenceConfig
from .types import LookupResult, AvailabilityCheck
from .embeddings.base import EmbeddingModel
from .embeddings import create_embedder
from .storage import create_storage_backend
from .eviction import create_eviction_policy
from .cache import CacheOperations
from .metrics import CacheMetrics
from .metrics.exporters import OpenTelemetryExporter
from .scheduler import SchedulerManager
from .utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


class Reminiscence:
    """
    Semantic cache for LLMs and multi-agent systems.

    Hybrid matching: semantic similarity + exact context matching.

    Query modes:
      - semantic: Normal semantic search (configurable threshold)
      - exact: Very high threshold (0.9999) for near-exact matches
      - auto: Try exact first, fallback to semantic

    Main API:
      - lookup: Search for existing result
      - lookup_batch: Search for multiple results (optimized)
      - store: Save new result
      - store_batch: Save multiple results (optimized)
      - check_availability: Verify availability without retrieving data
      - invalidate: Mark entries as invalid
      - cleanup_expired: Remove expired entries
      - create_index: Create vector index for fast searches
      - get_stats: Return cache statistics
      - health_check: Perform health checks
      - cached: Decorator for automatic caching
      - start_scheduler: Start background cleanup and metrics export
      - stop_scheduler: Stop background schedulers

    Example:
        >>> # Initialize cache
        >>> cache = Reminiscence()
        >>>
        >>> # Manual usage with context
        >>> result = cache.lookup("What is ML?", {"agent": "qa", "model": "gpt-4"})
        >>> if result.is_hit:
        ...     print(result.result)
        ... else:
        ...     data = expensive_llm_call()
        ...     cache.store("What is ML?", {"agent": "qa", "model": "gpt-4"}, data)
        >>>
        >>> # Batch operations (optimized)
        >>> results = cache.lookup_batch(
        ...     queries=["What is AI?", "What is ML?"],
        ...     contexts={"model": "gpt-4"}  # Shared context
        ... )
        >>>
        >>> cache.store_batch(
        ...     queries=["q1", "q2", "q3"],
        ...     contexts={"model": "gpt-4"},  # Shared context
        ...     results=[r1, r2, r3]
        ... )
        >>>
        >>> # Exact mode for SQL caching
        >>> result = cache.lookup("SELECT * FROM users", {"db": "prod"}, query_mode="exact")
        >>>
        >>> # Auto mode (exact then semantic)
        >>> result = cache.lookup("What is Python?", {"agent": "qa"}, query_mode="auto")
        >>>
        >>> # With automatic background tasks
        >>> cache.start_scheduler()
        >>> ... # use cache
        >>> cache.stop_scheduler()
        >>>
        >>> # Decorator usage
        >>> @cache.cached(query="question", context_params=["model"], query_mode="semantic")
        >>> def ask_llm(question: str, model: str):
        ...     return expensive_llm_call(question, model)
    """

    def __init__(
        self,
        config: Optional[ReminiscenceConfig] = None,
        embedder: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize Reminiscence with all components.

        Args:
            config: Cache configuration. If None, loads from environment variables.
        """
        self.config = config or ReminiscenceConfig.load()

        configure_logging(self.config.log_level, self.config.json_logs)

        logger.info(
            "initializing_reminiscence",
            model=self.config.model_name,
            db_uri=self.config.db_uri,
            eviction=self.config.eviction_policy,
        )

        # Initialize components
        if embedder is not None:
            self.embedder = embedder
            logger.info("using_provided_embedder", model_type=type(embedder).__name__)
        else:
            self.embedder = create_embedder(self.config)

        # Warm-up embedder if configured
        if self.config.warm_up_embedder:
            logger.debug("warming_up_embedder")
            warmup_start = time.perf_counter()
            try:
                # Generate a dummy embedding to load model into memory
                _ = self.embedder.embed("warm up query")
                warmup_ms = (time.perf_counter() - warmup_start) * 1000
                logger.info(
                    "embedder_warmed_up",
                    model=self.config.model_name,
                    warmup_ms=round(warmup_ms, 1),
                )
            except Exception as e:
                logger.warning(
                    "embedder_warmup_failed",
                    error=str(e),
                    note="Continuing without warm-up",
                )

        self.backend = create_storage_backend(self.config, self.embedder.embedding_dim)
        self.eviction = create_eviction_policy(self.config.eviction_policy)
        self.metrics = CacheMetrics() if self.config.enable_metrics else None

        # OpenTelemetry exporter
        self.otel_exporter: Optional[OpenTelemetryExporter] = None
        if self.config.otel_enabled and self.metrics:
            try:
                self.otel_exporter = OpenTelemetryExporter.from_config(self.config)
                if self.otel_exporter:
                    logger.info(
                        "opentelemetry_enabled",
                        endpoint=self.config.otel_endpoint,
                        service=self.config.otel_service_name,
                        interval_ms=self.config.otel_export_interval_ms,
                    )
                else:
                    logger.warning("opentelemetry_exporter_disabled")
            except Exception as e:
                logger.error(
                    "opentelemetry_init_failed",
                    error=str(e),
                    exc_info=True,
                )
        elif self.config.otel_enabled and not self.metrics:
            logger.warning(
                "opentelemetry_disabled",
                reason="Metrics are disabled (REMINISCENCE_ENABLE_METRICS=false)",
            )

        # Cache operations
        self._ops = CacheOperations(
            storage=self.backend,
            embedder=self.embedder,
            eviction=self.eviction,
            config=self.config,
            metrics=self.metrics,
        )

        self.scheduler_manager = None

        logger.info(
            "reminiscence_ready",
            entries=self.backend.count(),
            max_entries=self.config.max_entries,
            embedding_dim=self.embedder.embedding_dim,
            threshold=self.config.similarity_threshold,
        )

    def clear(self):
        """Clear all cache entries and reset metrics."""
        self.backend.clear()
        if self.metrics:
            self.metrics.reset()

    def lookup(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
        track_metrics: Optional[bool] = True,
    ) -> LookupResult:
        """
        Search cache entry by semantic similarity with exact context matching.

        Args:
            query: Query text to search
            context: Context dict for exact matching (agent_id, tools, model, etc)
            similarity_threshold: Minimum similarity score (0-1)
            query_mode: Matching strategy ("semantic", "exact", "auto")
            track_metrics: Internal flag to control metrics tracking

        Returns:
            LookupResult with hit status and cached data
        """
        return self._ops.lookup(
            query,
            context,
            similarity_threshold,
            query_mode,
            track_metrics,
        )

    def lookup_batch(
        self,
        queries: List[str],
        contexts: Union[Dict[str, Any], List[Dict[str, Any]]],
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
        track_metrics: bool = True,
    ) -> List[LookupResult]:
        """
        Lookup multiple queries in batch (optimized for embeddings).

        Args:
            queries: List of query texts to search
            contexts: Single context dict (shared) OR list of contexts (one per query)
            similarity_threshold: Minimum similarity score (overrides config)
            query_mode: Matching strategy ("semantic", "exact", "auto")
            track_metrics: Whether to track metrics for these lookups

        Returns:
            List of LookupResult objects (one per query)

        Examples:
            >>> # Shared context (most common)
            >>> results = cache.lookup_batch(
            ...     queries=["What is AI?", "What is ML?"],
            ...     contexts={"model": "gpt-4"}  # Single dict
            ... )
            >>>
            >>> # Individual contexts (when needed)
            >>> results = cache.lookup_batch(
            ...     queries=["q1", "q2"],
            ...     contexts=[{"model": "gpt-4"}, {"model": "claude"}]
            ... )
        """
        # Normalize contexts to list
        if isinstance(contexts, dict):
            contexts_list = [contexts] * len(queries)
        else:
            contexts_list = contexts
            if len(contexts_list) != len(queries):
                raise ValueError(
                    f"Length mismatch: {len(queries)} queries but {len(contexts_list)} contexts"
                )

        return self._ops.lookup_batch(
            queries,
            contexts_list,
            similarity_threshold,
            query_mode,
            track_metrics,
        )

    def store(
        self,
        query: str,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        query_mode: str = "semantic",
        allow_errors: bool = False,
    ):
        """
        Store result in cache.

        Args:
            query: Query text
            context: Context dict (will be matched exactly in future lookups)
            result: Result to cache (supports JSON, Arrow, Pandas, Polars)
            metadata: Optional metadata
            query_mode: For tracking purposes (all modes generate embeddings)
            allow_errors: If False (default), don't cache error results
        """
        self._ops.store(query, context, result, metadata, query_mode, allow_errors)

    def store_batch(
        self,
        queries: List[str],
        contexts: Union[Dict[str, Any], List[Dict[str, Any]]],
        results: List[Any],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        query_mode: str = "semantic",
        allow_errors: bool = False,
    ):
        """
        Store multiple results in batch (optimized for embeddings).

        This is 3-5x faster than calling store() in a loop because
        embeddings are generated in batch.

        Args:
            queries: List of query texts
            contexts: Single context dict (shared) OR list of contexts (one per query)
            results: List of results to cache
            metadata: Single metadata dict (shared) OR list of metadata dicts
            query_mode: Storage mode ("semantic"/"exact"/"auto")
            allow_errors: If False (default), skip error results

        Examples:
            >>> # Shared context (most common)
            >>> cache.store_batch(
            ...     queries=["q1", "q2", "q3"],
            ...     contexts={"model": "gpt-4"},  # Single dict
            ...     results=[r1, r2, r3]
            ... )
            >>>
            >>> # Individual contexts (advanced)
            >>> cache.store_batch(
            ...     queries=["q1", "q2", "q3"],
            ...     contexts=[{"model": "gpt-4"}, {"model": "claude"}, {"model": "gpt-4"}],
            ...     results=[r1, r2, r3]
            ... )
        """
        # Normalize contexts to list
        if isinstance(contexts, dict):
            contexts_list = [contexts] * len(queries)
        else:
            contexts_list = contexts
            if len(contexts_list) != len(queries):
                raise ValueError(
                    f"Length mismatch: {len(queries)} queries but {len(contexts_list)} contexts"
                )

        # Normalize metadata to list
        if metadata is None:
            metadata_list = [None] * len(queries)
        elif isinstance(metadata, dict):
            metadata_list = [metadata] * len(queries)
        else:
            metadata_list = metadata
            if len(metadata_list) != len(queries):
                raise ValueError(
                    f"Length mismatch: {len(queries)} queries but {len(metadata_list)} metadata dicts"
                )

        self._ops.store_batch(
            queries,
            contexts_list,
            results,
            metadata_list,
            query_mode,
            allow_errors,
        )

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
        return self._ops.invalidate(query, context, older_than_seconds)

    def cleanup_expired(self) -> int:
        """
        Clean expired entries according to configured TTL.

        Returns:
            Number of deleted entries
        """
        return self._ops.cleanup_expired()

    def check_availability(
        self,
        query: str,
        context: Dict[str, Any],
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
    ) -> AvailabilityCheck:
        """
        Verify availability without retrieving full data.

        Args:
            query: Query text
            context: Context dict
            similarity_threshold: Minimum similarity score
            query_mode: Matching strategy

        Returns:
            AvailabilityCheck with availability info
        """
        result = self.lookup(
            query,
            context,
            similarity_threshold,
            query_mode,
            track_metrics=False,
        )

        if not result.is_hit:
            return AvailabilityCheck(available=False, ttl_remaining_seconds=None)

        ttl_remaining = None
        if self.config.ttl_seconds and result.age_seconds is not None:
            ttl_remaining = self.config.ttl_seconds - result.age_seconds

        return AvailabilityCheck(
            available=True,
            age_seconds=result.age_seconds,
            ttl_remaining_seconds=ttl_remaining,
            similarity=result.similarity,
        )

    def start_scheduler(
        self,
        interval_seconds: Optional[int] = None,
        initial_delay_seconds: int = 60,
        metrics_export_interval_seconds: Optional[int] = None,
        metrics_initial_delay_seconds: int = 0,
    ):
        """
        Start background schedulers for cleanup and metrics export.

        Args:
            interval_seconds: Interval for cache cleanup (default: 3600)
            initial_delay_seconds: Initial delay before first cleanup run (default: 60)
            metrics_export_interval_seconds: Interval for metrics export (default: from config)
            metrics_initial_delay_seconds: Initial delay before first metrics export (default: 0)

        Example:
            >>> cache = Reminiscence()
            >>> cache.start_scheduler(
            ...     interval_seconds=1800,
            ...     metrics_export_interval_seconds=10
            ... )
            >>> # ... use cache ...
            >>> cache.stop_scheduler()
        """
        if self.scheduler_manager and self.scheduler_manager.schedulers:
            logger.warning("schedulers_already_running")
            return

        self.scheduler_manager = SchedulerManager(metrics=self.metrics)

        # Cleanup scheduler (if TTL configured)
        if self.config.ttl_seconds is not None:
            cleanup_interval = interval_seconds or 3600
            self.scheduler_manager.add_scheduler(
                name="cache_cleanup",
                cleanup_func=self.cleanup_expired,
                interval_seconds=cleanup_interval,
                initial_delay_seconds=initial_delay_seconds,
            )
            logger.info(
                "cleanup_scheduler_configured",
                interval_seconds=cleanup_interval,
                ttl_seconds=self.config.ttl_seconds,
            )
        else:
            logger.warning(
                "cleanup_scheduler_skipped",
                reason="No TTL configured (REMINISCENCE_TTL_SECONDS not set)",
            )

        # Metrics export scheduler (if OTEL enabled)
        if self.otel_exporter and self.metrics:
            default_interval = self.config.otel_export_interval_ms / 1000
            metrics_interval = metrics_export_interval_seconds or default_interval

            def export_metrics() -> int:
                """Export current metrics to OpenTelemetry."""
                try:
                    metrics_data = self.metrics.report()
                    self.otel_exporter.export(metrics_data)
                    logger.debug(
                        "metrics_exported",
                        hits=metrics_data["hits"],
                        misses=metrics_data["misses"],
                        hit_rate=metrics_data["hit_rate"],
                    )
                    return 0
                except Exception as e:
                    logger.error(
                        "metrics_export_failed",
                        error=str(e),
                        exc_info=True,
                    )
                    return 0

            self.scheduler_manager.add_scheduler(
                name="metrics_export",
                cleanup_func=export_metrics,
                interval_seconds=metrics_interval,
                initial_delay_seconds=metrics_initial_delay_seconds,
            )
            logger.info(
                "metrics_export_scheduler_configured",
                interval_seconds=metrics_interval,
                endpoint=self.config.otel_endpoint,
            )

        # Start all schedulers
        if self.scheduler_manager.schedulers:
            self.scheduler_manager.start_all()
            logger.info(
                "schedulers_started",
                active_schedulers=list(self.scheduler_manager.schedulers.keys()),
            )
        else:
            logger.warning(
                "no_schedulers_configured",
                reason="Neither TTL nor OpenTelemetry are enabled",
            )

    def stop_scheduler(self, timeout: float = 5.0):
        """
        Stop all background schedulers.

        Args:
            timeout: Maximum time to wait for schedulers to stop (seconds)

        Example:
            >>> cache.stop_scheduler()
        """
        if self.scheduler_manager is None:
            logger.warning("schedulers_not_initialized")
            return

        for name, scheduler in self.scheduler_manager.schedulers.items():
            logger.debug("stopping_scheduler", name=name)
            scheduler.stop(timeout=timeout)

        logger.info("schedulers_stopped")

    def get_scheduler_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics for all schedulers.

        Returns:
            Dict with stats for each scheduler or None if no schedulers

        Example:
            >>> stats = cache.get_scheduler_stats()
            >>> if stats:
            ...     print(f"Cache cleanup runs: {stats['cache_cleanup']['total_runs']}")
            ...     print(f"Metrics exports: {stats['metrics_export']['total_runs']}")
        """
        if self.scheduler_manager is None:
            return None
        return self.scheduler_manager.get_stats()

    def create_index(
        self, num_partitions: int = 256, num_subvectors: Optional[int] = None
    ):
        """
        Create IVF-PQ index for fast vector searches.

        Args:
            num_partitions: Number of IVF partitions
            num_subvectors: Number of PQ subvectors (default: embedding_dim / 4)
        """
        row_count = self.backend.count()
        if row_count < 256:
            logger.warning(
                "insufficient_entries_for_index",
                count=row_count,
                minimum=256,
            )
            return

        if num_subvectors is None:
            num_subvectors = max(1, self.embedder.embedding_dim // 4)

        logger.info(
            "creating_index",
            partitions=num_partitions,
            subvectors=num_subvectors,
            entries=row_count,
        )
        self.backend.create_index(num_partitions, num_subvectors)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        stats = {
            "cache_entries": self.backend.count(),
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

        if self.scheduler_manager:
            stats["schedulers"] = self.scheduler_manager.get_stats()

        return stats

    def get_index_stats(self) -> Dict[str, Any]:
        """Return vector index statistics."""
        return {
            "has_index": self.backend.has_index(),
            "total_entries": self.backend.count(),
            "note": "LanceDB doesn't expose detailed index metrics",
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache components."""
        checks = {
            "embedding": {"ok": True, "error": None},
            "database": {"ok": True, "error": None},
            "error_rate": {"ok": True, "details": "No metrics available"},
            "schedulers": {"ok": True, "details": "Not running"},
            "opentelemetry": {"ok": True, "details": "Disabled"},
        }

        # Test embeddings
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
                self.backend.to_arrow()
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
                error_rate = (
                    (total_errors / total_requests) if total_requests > 0 else 0
                )
                if error_rate > 0.10:
                    checks["error_rate"]["ok"] = False
                    checks["error_rate"]["details"] = (
                        f"High error rate: {error_rate * 100:.1f}% ({total_errors}/{total_requests} requests)"
                    )
                else:
                    checks["error_rate"]["ok"] = True
                    checks["error_rate"]["details"] = (
                        f"Error rate: {error_rate * 100:.1f}% ({total_errors}/{total_requests} requests)"
                    )
            else:
                checks["error_rate"]["ok"] = True
                checks["error_rate"]["details"] = (
                    f"Insufficient data ({total_requests} requests)"
                )

        # Check schedulers
        if self.scheduler_manager and self.scheduler_manager.schedulers:
            all_stats = self.scheduler_manager.get_stats()
            running_count = sum(1 for s in all_stats.values() if s["running"])
            total_errors = sum(s["errors"] for s in all_stats.values())

            if total_errors > 0:
                checks["schedulers"]["ok"] = False
                checks["schedulers"]["details"] = (
                    f"{running_count}/{len(all_stats)} running with {total_errors} errors"
                )
            else:
                checks["schedulers"]["ok"] = True
                checks["schedulers"]["details"] = (
                    f"{running_count}/{len(all_stats)} schedulers running"
                )

        # Check OpenTelemetry
        if self.otel_exporter:
            checks["opentelemetry"]["ok"] = True
            checks["opentelemetry"]["details"] = (
                f"Enabled (service: {self.otel_exporter.service_name}, "
                f"endpoint: {self.otel_exporter.endpoint})"
            )
        elif self.config.otel_enabled:
            checks["opentelemetry"]["ok"] = False
            checks["opentelemetry"]["details"] = (
                "Enabled but exporter failed to initialize"
            )

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

    def cached(
        self,
        query: str = "query",
        context_params: Optional[list] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
        query_mode: str = "semantic",
        similarity_threshold: Optional[float] = None,
    ):
        """
        Decorator to cache function results with hybrid matching.

        Args:
            query: Name of parameter to use as query text (default: "query")
            context_params: Parameter names to include in context
            static_context: Static context dict
            auto_strict: Auto-detect non-string params as context
            query_mode: Matching strategy ("semantic", "exact", "auto")
            similarity_threshold: Minimum similarity score
        """
        from .decorators import create_cached_decorator

        decorator_factory = create_cached_decorator(self)
        return decorator_factory(
            query=query,
            context_params=context_params,
            static_context=static_context,
            auto_strict=auto_strict,
            query_mode=query_mode,
            similarity_threshold=similarity_threshold,
        )

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - stops all schedulers."""
        if self.scheduler_manager and self.scheduler_manager.schedulers:
            self.stop_scheduler()
        return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Reminiscence(entries={self.backend.count()}, "
            f"dim={self.embedder.embedding_dim}, "
            f"threshold={self.config.similarity_threshold})"
        )
