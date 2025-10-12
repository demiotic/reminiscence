"""Core Reminiscence class - Facade for all components."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, cast

from .cache import CacheOperations
from .config import ReminiscenceConfig
from .embeddings import create_embedder
from .embeddings.base import EmbeddingModel
from .eviction import create_eviction_policy
from .metrics import CacheMetrics
from .metrics.exporters import OpenTelemetryExporter
from .scheduler import SchedulerManager
from .storage import create_storage_backend
from .types import (
    AvailabilityCheck,
    EvictionPolicy,
    LookupRequest,
    LookupResult,
    MultiModalInput,
    QueryMode,
    StoreRequest,
)
from .utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


class Reminiscence:
    """Semantic cache for LLMs and multi-agent systems with multimodal support.

    Hybrid matching: semantic similarity + exact context matching.
    Supports text, image, video, and audio inputs through MultiModalInput.

    Query modes:
      - QueryMode.SEMANTIC: Semantic similarity search with configurable threshold
      - QueryMode.EXACT: Exact text matching only (no semantic similarity)
      - QueryMode.AUTO: Try exact match first, fallback to semantic if no match

    Main API:
      - lookup: Search for existing result
      - lookup_batch: Search for multiple results (optimized, uses LookupRequest objects)
      - store: Save new result
      - store_batch: Save multiple results (optimized, uses StoreRequest objects)
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
        >>> from reminiscence import Reminiscence, QueryMode
        >>> from reminiscence.types import MultiModalInput, StoreRequest
        >>>
        >>> # Initialize cache
        >>> cache = Reminiscence()
        >>>
        >>> # Manual usage with context
        >>> query = MultiModalInput(text="What is ML?")
        >>> result = cache.lookup(query, {"agent": "qa", "model": "gpt-4"})
        >>> if result.is_hit:
        ...     print(result.result)
        ... else:
        ...     data = expensive_llm_call()
        ...     cache.store(query, {"agent": "qa", "model": "gpt-4"}, data)
        >>>
        >>> # Batch operations with typed requests
        >>> from reminiscence import LookupRequest, StoreRequest
        >>>
        >>> lookup_requests = [
        ...     LookupRequest(
        ...         query=MultiModalInput(text="What is AI?"),
        ...         context={"model": "gpt-4"}
        ...     ),
        ...     LookupRequest(
        ...         query=MultiModalInput(text="What is ML?"),
        ...         context={"model": "gpt-4"}
        ...     ),
        ... ]
        >>> results = cache.lookup_batch(lookup_requests)
        >>>
        >>> store_requests = [
        ...     StoreRequest(
        ...         query=MultiModalInput(text="q1"),
        ...         context={"model": "gpt-4"},
        ...         result="r1"
        ...     ),
        ...     StoreRequest(
        ...         query=MultiModalInput(text="q2"),
        ...         context={"model": "gpt-4"},
        ...         result="r2"
        ...     ),
        ... ]
        >>> cache.store_batch(store_requests)
        >>>
        >>> # Multimodal queries
        >>> img_query = MultiModalInput(text="What's in this image?", image=img_bytes)
        >>> result = cache.lookup(img_query, {"model": "gpt-4o"})
        >>>
        >>> # Exact mode for SQL caching
        >>> sql_query = MultiModalInput(text="SELECT * FROM users")
        >>> result = cache.lookup(
        ...     sql_query,
        ...     {"db": "prod"},
        ...     mode=QueryMode.EXACT
        ... )
        >>>
        >>> # With automatic background tasks
        >>> cache.start_scheduler()
        >>> # ... use cache ...
        >>> cache.stop_scheduler()
        >>>
        >>> # Decorator usage
        >>> @cache.cached(
        ...     query="question",
        ...     context=["model"],
        ...     mode=QueryMode.SEMANTIC
        ... )
        >>> def ask_llm(question: str, model: str):
        ...     return expensive_llm_call(question, model)
    """

    def __init__(
        self,
        config: Optional[ReminiscenceConfig] = None,
        embedder: Optional[EmbeddingModel] = None,
    ):
        """Initialize Reminiscence with all components.

        Args:
            config: Cache configuration. If None, loads from environment variables.
            embedder: Optional custom embedder. If None, creates from config.
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
        self.eviction = create_eviction_policy(EvictionPolicy(self.config.eviction_policy))
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

        # Cache operations facade (internal)
        self._ops = CacheOperations(
            storage=self.backend,
            embedder=self.embedder,
            eviction=self.eviction,
            config=self.config,
            metrics=self.metrics,
        )

        self.scheduler_manager: Optional[SchedulerManager] = None

        logger.info(
            "reminiscence_ready",
            entries=self.backend.count(),
            max_entries=self.config.max_entries,
            embedding_dim=self.embedder.embedding_dim,
            threshold=self.config.similarity_threshold,
        )

    # ========================================================================
    # Core Cache Operations
    # ========================================================================

    def clear(self) -> None:
        """Clear all cache entries and reset metrics."""
        self.backend.clear()
        if self.metrics:
            self.metrics.reset()

    def lookup(
        self,
        query: MultiModalInput,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        mode: QueryMode = QueryMode.AUTO,
        track_metrics: bool = True,
    ) -> LookupResult:
        """Search cache entry by semantic similarity with exact context matching.

        Args:
            query: MultiModalInput containing text, image, video, or audio.
            context: Context dict for exact matching (default: {}).
            similarity_threshold: Minimum similarity score (overrides config).
            mode: Matching strategy (default: QueryMode.AUTO).
            track_metrics: Whether to track cache metrics (default: True).

        Returns:
            LookupResult with hit status and cached data.

        Example:
            >>> from reminiscence import QueryMode
            >>> from reminiscence.types import MultiModalInput
            >>> query = MultiModalInput(text="What is ML?")
            >>> result = cache.lookup(
            ...     query,
            ...     context={"agent": "qa"},
            ...     mode=QueryMode.SEMANTIC
            ... )
        """
        return self._ops.lookup(
            query,
            context,
            similarity_threshold,
            mode,
            track_metrics,
        )

    def lookup_batch(
        self,
        requests: List[LookupRequest],
        track_metrics: bool = True,
    ) -> List[LookupResult]:
        """Lookup multiple queries in batch (optimized for embeddings).

        Args:
            requests: List of LookupRequest objects with query, context, threshold, mode.
            track_metrics: Whether to track metrics for these lookups (default: True).

        Returns:
            List of LookupResult objects (one per request, same order).

        Example:
            >>> from reminiscence import LookupRequest, QueryMode
            >>> from reminiscence.types import MultiModalInput
            >>>
            >>> requests = [
            ...     LookupRequest(
            ...         query=MultiModalInput(text="What is AI?"),
            ...         context={"model": "gpt-4"},
            ...         mode=QueryMode.SEMANTIC
            ...     ),
            ...     LookupRequest(
            ...         query=MultiModalInput(text="What is ML?"),
            ...         context={"model": "gpt-4"},
            ...         mode=QueryMode.EXACT
            ...     ),
            ... ]
            >>> results = cache.lookup_batch(requests)
        """
        return self._ops.lookup_batch(requests, track_metrics)

    def store(
        self,
        query: MultiModalInput,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        context_threshold: Optional[float] = None,
        allow_errors: bool = False,
        mode: QueryMode = QueryMode.AUTO,
    ) -> None:
        """Store result in cache.

        Args:
            query: MultiModalInput containing text, image, video, or audio.
            context: Context dict (will be matched exactly in future lookups).
            result: Result to cache (supports JSON, Arrow, Pandas, Polars).
            metadata: Optional metadata to store with entry.
            ttl_seconds: Time-to-live in seconds (overrides global, None = no expiration).
            context_threshold: Override similarity threshold for this entry.
            allow_errors: If True, store error results (default: False).
            mode: Query matching strategy (default: QueryMode.AUTO).

        Example:
            >>> from reminiscence.types import MultiModalInput
            >>> query = MultiModalInput(text="What is ML?")
            >>> cache.store(
            ...     query,
            ...     context={"agent": "qa"},
            ...     result="Machine Learning is...",
            ...     ttl_seconds=3600
            ... )
        """
        self._ops.store(
            query,
            context,
            result,
            metadata,
            ttl_seconds,
            context_threshold,
            allow_errors,
            mode,
        )

    def store_batch(
        self,
        requests: List[StoreRequest],
        allow_errors: bool = False,
        mode: QueryMode = QueryMode.AUTO,
    ) -> None:
        """Store multiple results in batch (optimized for embeddings).

        This is 3-5x faster than calling store() in a loop because
        embeddings are generated in batch.

        Args:
            requests: List of StoreRequest objects with query, context, result, metadata, ttl.
            allow_errors: If True, store error results (default: False).
            mode: Query matching strategy (default: QueryMode.AUTO).

        Example:
            >>> from reminiscence import StoreRequest
            >>> from reminiscence.types import MultiModalInput
            >>>
            >>> requests = [
            ...     StoreRequest(
            ...         query=MultiModalInput(text="What is AI?"),
            ...         context={"model": "gpt-4"},
            ...         result="AI is...",
            ...         ttl_seconds=3600
            ...     ),
            ...     StoreRequest(
            ...         query=MultiModalInput(text="What is ML?"),
            ...         context={"model": "gpt-4"},
            ...         result="ML is...",
            ...         ttl_seconds=3600
            ...     ),
            ... ]
            >>> cache.store_batch(requests)
        """
        self._ops.store_batch(requests, allow_errors, mode)

    def check_availability(
        self,
        query: MultiModalInput,
        context: Dict[str, Any],
        similarity_threshold: Optional[float] = None,
        mode: QueryMode = QueryMode.AUTO,
    ) -> AvailabilityCheck:
        """Verify availability without retrieving full data.

        Args:
            query: MultiModalInput to check.
            context: Context dict.
            similarity_threshold: Minimum similarity score (overrides config).
            mode: Matching strategy (default: QueryMode.AUTO).

        Returns:
            AvailabilityCheck with availability info.
        """
        result = self.lookup(
            query,
            context,
            similarity_threshold,
            mode,
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

    # ========================================================================
    # Invalidation & Cleanup
    # ========================================================================

    def invalidate(
        self,
        query: Optional[MultiModalInput] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """Invalidate cache entries by criteria.

        Args:
            query: Exact multimodal query to invalidate (optional).
            context: Exact context to match (optional).
            older_than_seconds: Invalidate entries older than this (optional).

        Returns:
            Number of invalidated entries.
        """
        return self._ops.invalidate(query, context, older_than_seconds)

    def cleanup_expired(self) -> int:
        """Clean expired entries according to configured TTL.

        Returns:
            Number of deleted entries.
        """
        return self._ops.cleanup_expired()

    # ========================================================================
    # Scheduler Management
    # ========================================================================

    def start_scheduler(
        self,
        interval_seconds: Optional[int] = None,
        initial_delay_seconds: int = 60,
        metrics_export_interval_seconds: Optional[int] = None,
        metrics_initial_delay_seconds: int = 0,
    ) -> None:
        """Start background schedulers for cleanup and metrics export.

        Args:
            interval_seconds: Interval for cache cleanup (default: 3600).
            initial_delay_seconds: Initial delay before first cleanup run (default: 60).
            metrics_export_interval_seconds: Interval for metrics export (default: from config).
            metrics_initial_delay_seconds: Initial delay before first metrics export (default: 0).

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

            # Capture non-None values for closure (mypy type narrowing)
            metrics = self.metrics
            otel_exporter = self.otel_exporter

            def export_metrics() -> int:
                """Export current metrics to OpenTelemetry."""
                try:
                    metrics_data = metrics.report()
                    otel_exporter.export(metrics_data)
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
                interval_seconds=int(metrics_interval),
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

    def stop_scheduler(self, timeout: float = 5.0) -> None:
        """Stop all background schedulers.

        Args:
            timeout: Maximum time to wait for schedulers to stop (seconds).

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
        """Get statistics for all schedulers.

        Returns:
            Dict with stats for each scheduler or None if no schedulers.

        Example:
            >>> stats = cache.get_scheduler_stats()
            >>> if stats:
            ...     print(f"Cache cleanup runs: {stats['cache_cleanup']['total_runs']}")
            ...     print(f"Metrics exports: {stats['metrics_export']['total_runs']}")
        """
        if self.scheduler_manager is None:
            return None
        return self.scheduler_manager.get_stats()

    # ========================================================================
    # Index & Stats
    # ========================================================================

    def create_index(
        self, num_partitions: int = 256, num_subvectors: Optional[int] = None
    ) -> None:
        """Create IVF-PQ index for fast vector searches.

        Args:
            num_partitions: Number of IVF partitions (default: 256).
            num_subvectors: Number of PQ subvectors (default: embedding_dim / 4).
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
            # Type narrowing: get_stats() returns Dict[str, Dict[str, Any]]
            stats["schedulers"] = cast(Any, self.scheduler_manager.get_stats())

        return stats

    def get_index_stats(self) -> Dict[str, Any]:
        """Return vector index statistics."""
        return {
            "has_index": self.backend.has_index(),
            "total_entries": self.backend.count(),
            "note": "LanceDB doesn't expose detailed index metrics",
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache components.

        Returns:
            Dict with status and component health checks.
        """
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
                embedding_check = cast(Dict[str, Any], checks["embedding"])
                embedding_check["ok"] = False
                embedding_check["error"] = (
                    f"Embedding dimension mismatch: {len(test_embedding)} != {self.embedder.embedding_dim}"
                )
        except Exception as e:
            embedding_check = cast(Dict[str, Any], checks["embedding"])
            embedding_check["ok"] = False
            embedding_check["error"] = str(e)
            logger.error("health_check_embedding_failed", error=str(e), exc_info=True)

        # Test database
        try:
            entry_count = self.backend.count()
            if entry_count > 0:
                self.backend.to_arrow()
        except Exception as e:
            database_check = cast(Dict[str, Any], checks["database"])
            database_check["ok"] = False
            database_check["error"] = str(e)
            logger.error("health_check_database_failed", error=str(e), exc_info=True)

        # Check error rates
        if self.metrics:
            total_requests = self.metrics.total_requests
            lookup_errors = self.metrics.lookup_errors
            store_errors = self.metrics.store_errors
            total_errors = lookup_errors + store_errors

            error_rate_check = cast(Dict[str, Any], checks["error_rate"])
            if total_requests >= 10:
                error_rate = (
                    (total_errors / total_requests) if total_requests > 0 else 0
                )
                if error_rate > 0.10:
                    error_rate_check["ok"] = False
                    error_rate_check["details"] = (
                        f"High error rate: {error_rate * 100:.1f}% ({total_errors}/{total_requests} requests)"
                    )
                else:
                    error_rate_check["ok"] = True
                    error_rate_check["details"] = (
                        f"Error rate: {error_rate * 100:.1f}% ({total_errors}/{total_requests} requests)"
                    )
            else:
                error_rate_check["ok"] = True
                error_rate_check["details"] = (
                    f"Insufficient data ({total_requests} requests)"
                )

        # Check schedulers
        if self.scheduler_manager and self.scheduler_manager.schedulers:
            all_stats = self.scheduler_manager.get_stats()
            running_count = sum(1 for s in all_stats.values() if s["running"])
            total_errors = sum(s["errors"] for s in all_stats.values())

            schedulers_check = cast(Dict[str, Any], checks["schedulers"])
            if total_errors > 0:
                schedulers_check["ok"] = False
                schedulers_check["details"] = (
                    f"{running_count}/{len(all_stats)} running with {total_errors} errors"
                )
            else:
                schedulers_check["ok"] = True
                schedulers_check["details"] = (
                    f"{running_count}/{len(all_stats)} schedulers running"
                )

        # Check OpenTelemetry
        otel_check = cast(Dict[str, Any], checks["opentelemetry"])
        if self.otel_exporter:
            otel_check["ok"] = True
            otel_check["details"] = (
                f"Enabled (service: {self.otel_exporter.service_name}, "
                f"endpoint: {self.otel_exporter.endpoint})"
            )
        elif self.config.otel_enabled:
            otel_check["ok"] = False
            otel_check["details"] = (
                "Enabled but exporter failed to initialize"
            )

        all_checks_ok = all(cast(Dict[str, Any], check)["ok"] for check in checks.values())
        status = "healthy" if all_checks_ok else "unhealthy"

        database_check_final = cast(Dict[str, Any], checks["database"])
        response = {
            "status": status,
            "checks": checks,
            "metrics": {
                "total_entries": self.backend.count()
                if database_check_final["ok"]
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

    # ========================================================================
    # Decorator Support
    # ========================================================================

    def cached(
        self,
        query: str = "query",
        context: Optional[list] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
        mode: QueryMode = QueryMode.AUTO,
        similarity_threshold: Optional[float] = None,
    ):
        """Decorator to cache function results with hybrid matching.

        Args:
            query: Name of parameter to use as query text (default: "query").
            context: Parameter names to include in context.
            static_context: Static context dict.
            auto_strict: Auto-detect non-string params as context.
            mode: Matching strategy (default: QueryMode.AUTO).
            similarity_threshold: Minimum similarity score (overrides config).

        Example:
            >>> from reminiscence import QueryMode
            >>>
            >>> @cache.cached(
            ...     query="question",
            ...     context=["model"],
            ...     mode=QueryMode.SEMANTIC
            ... )
            >>> def ask_llm(question: str, model: str):
            ...     return expensive_llm_call(question, model)
        """
        from .decorators import create_cached_decorator

        decorator_factory = create_cached_decorator(self)
        return decorator_factory(
            query=query,
            context=context,
            static_context=static_context,
            auto_strict=auto_strict,
            mode=mode,
            similarity_threshold=similarity_threshold,
        )

    # ========================================================================
    # Context Manager & Repr
    # ========================================================================

    def __enter__(self) -> Reminiscence:
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager support - stops all schedulers."""
        if self.scheduler_manager and self.scheduler_manager.schedulers:
            self.stop_scheduler()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Reminiscence(entries={self.backend.count()}, "
            f"dim={self.embedder.embedding_dim}, "
            f"threshold={self.config.similarity_threshold})"
        )
