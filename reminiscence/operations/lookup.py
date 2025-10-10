"""Lookup and matching operations."""

import time
import json
from typing import Optional, Dict, Any, List

from ..types import LookupResult, CacheEntry
from ..utils.logging import get_logger
from ..utils.query_detection import should_use_exact_mode

logger = get_logger(__name__)


class LookupOperations:
    """Handles cache lookup and matching operations."""

    def __init__(self, storage, embedder, eviction, config, metrics):
        """Initialize lookup operations.

        Args:
            storage: Storage backend instance
            embedder: Embedding model instance
            eviction: Eviction policy instance
            config: Configuration object
            metrics: Optional metrics tracker
        """
        self.storage = storage
        self.embedder = embedder
        self.eviction = eviction
        self.config = config
        self.metrics = metrics
        self.otel_exporter = None
        self._otel_export_counter = 0

        # Initialize OpenTelemetry if enabled
        if config.otel_enabled:
            try:
                from ..metrics.exporters import OpenTelemetryExporter

                self.otel_exporter = OpenTelemetryExporter.from_config(config)
                logger.info("opentelemetry_enabled", endpoint=config.otel_endpoint)
            except Exception as e:
                logger.warning("opentelemetry_init_failed", error=str(e))

        self._sync_eviction_state()

    def _sync_eviction_state(self):
        """Sync eviction policy with existing entries on startup."""
        sync_start = time.time()
        try:
            arrow_table = self.storage.to_arrow()
            if len(arrow_table) > 0:
                rows = arrow_table.to_pylist()
                logger.debug("syncing_eviction_state", existing_entries=len(rows))

                for row in rows:
                    entry_id = self._generate_entry_id(
                        row.get("query_text", ""), row.get("context", "{}")
                    )
                    self.eviction.on_insert(entry_id)

                    if hasattr(self.eviction, "access_times"):
                        self.eviction.access_times[entry_id] = row.get(
                            "timestamp", time.time()
                        )

                    if hasattr(self.eviction, "frequencies"):
                        self.eviction.frequencies[entry_id] = 0

                sync_ms = (time.time() - sync_start) * 1000
                logger.info(
                    "synced_eviction_state",
                    entries=len(rows),
                    policy=self.config.eviction_policy,
                    latency_ms=round(sync_ms, 1),
                )
            else:
                logger.debug("eviction_sync_skipped_empty_cache")
        except Exception as e:
            logger.warning("failed_to_sync_eviction_state", error=str(e), exc_info=True)

    def _generate_entry_id(self, query: str, context: Any) -> str:
        """Generate consistent entry ID for eviction tracking.

        Args:
            query: Query text
            context: Context dict or JSON string

        Returns:
            Unique entry identifier string
        """
        if isinstance(context, str):
            context_str = context
        else:
            context_str = json.dumps(context, sort_keys=True)
        return f"{query[:30]}:{context_str[:30]}"

    def _export_metrics_to_otel(self):
        """Export metrics to OpenTelemetry periodically."""
        if self.otel_exporter and self.metrics:
            try:
                report = self.metrics.report()
                self.otel_exporter.export(report)
                logger.debug("otel_metrics_exported", hit_rate=report.get("hit_rate"))
            except Exception as e:
                logger.warning("otel_export_failed", error=str(e))

    def lookup(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
        track_metrics: bool = True,
    ) -> LookupResult:
        """Search cache by query with exact context matching.

        Supports context-specific similarity thresholds and per-entry TTL checking.

        Args:
            query: Query text to search
            context: Context dict for exact matching
            similarity_threshold: Minimum similarity score (overrides config)
            query_mode: Query matching strategy (semantic, exact, auto)
            track_metrics: Internal flag to control metrics tracking

        Returns:
            LookupResult with hit status and cached data if found
        """
        start_time = time.time()
        context = context or {}

        logger.debug(
            "lookup_start",
            query_preview=query[:50],
            query_length=len(query),
            context_keys=list(context.keys()),
            query_mode=query_mode,
        )

        try:
            cache_count = self.storage.count()
            if cache_count == 0:
                logger.debug("lookup_miss_empty_cache")
                return self._miss(
                    "cache_empty", start_time, track_metrics=track_metrics
                )

            logger.debug("cache_size", entries=cache_count)

            # Auto-detect query mode if requested
            actual_mode = query_mode
            if query_mode == "auto":
                use_exact = should_use_exact_mode(query)
                actual_mode = "exact" if use_exact else "semantic"
                logger.debug(
                    "auto_mode_detected",
                    query_preview=query[:50],
                    detected_mode=actual_mode,
                    query_length=len(query),
                )

            # Generate embedding for semantic search
            embedding = None
            if actual_mode == "semantic":
                embed_start = time.time()
                embedding = self.embedder.embed(query)
                embed_ms = (time.time() - embed_start) * 1000
                logger.debug(
                    "embedding_generated",
                    latency_ms=round(embed_ms, 1),
                    text_length=len(query),
                    embedding_dim=len(embedding) if embedding else 0,
                )
            else:
                logger.debug("embedding_skipped_exact_mode")

            # Determine similarity threshold
            if similarity_threshold is None:
                threshold = self.config.get_threshold_for_context(context)
            else:
                threshold = similarity_threshold

            logger.debug(
                "using_threshold",
                threshold=threshold,
                source="context_specific"
                if similarity_threshold is None and self.config.context_thresholds
                else "default",
            )

            # Search storage
            search_start = time.time()
            candidates = self.storage.search(
                embedding=embedding,
                context=context,
                limit=1,
                similarity_threshold=threshold,
                query_mode=actual_mode,
                query_text=query,
            )
            search_ms = (time.time() - search_start) * 1000

            if not candidates:
                reason = "no_exact_match" if actual_mode == "exact" else "no_match"
                logger.debug(
                    "lookup_miss",
                    reason=reason,
                    search_ms=round(search_ms, 1),
                    threshold=threshold,
                )
                return self._miss(reason, start_time, track_metrics=track_metrics)

            logger.debug(
                "lookup_candidates_found",
                count=len(candidates),
                search_ms=round(search_ms, 1),
            )

            return self._process_hit(candidates[0], start_time, track_metrics)

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                "cache_lookup_error",
                error_type=type(e).__name__,
                error_message=str(e),
                query_preview=query[:50],
                query_mode=query_mode,
                latency_ms=round(elapsed_ms, 1),
                exc_info=True,
            )

            if self.metrics and track_metrics:
                self.metrics.misses += 1
                self.metrics.record_lookup_latency(elapsed_ms)
                self.metrics.lookup_errors += 1

            return LookupResult(hit=False)

    def _process_hit(
        self, best: CacheEntry, start_time: float, track_metrics: bool
    ) -> LookupResult:
        """Process cache hit with per-entry TTL check and metrics.

        Args:
            best: Matched cache entry
            start_time: Lookup start timestamp
            track_metrics: Whether to track metrics

        Returns:
            LookupResult with cached data or miss if expired
        """
        entry_ttl = (
            best.ttl_seconds
            if best.ttl_seconds is not None
            else self.config.ttl_seconds
        )

        # Check if entry has expired
        if entry_ttl is not None:
            age = best.age_seconds if best.age_seconds else 0
            if age > entry_ttl:
                logger.debug(
                    "lookup_miss_expired",
                    query_preview=best.query_text[:50],
                    age_seconds=round(age, 1),
                    ttl_seconds=entry_ttl,
                )
                return self._miss("expired", start_time, track_metrics=track_metrics)

        # Update eviction policy
        entry_id = self._generate_entry_id(best.query_text, best.context)
        self.eviction.on_access(entry_id)

        elapsed_ms = (time.time() - start_time) * 1000

        ttl_remaining = None
        if entry_ttl is not None:
            ttl_remaining = max(0.0, entry_ttl - best.age_seconds)

        logger.info(
            "cache_hit",
            similarity=round(best.similarity, 3) if best.similarity else 1.0,
            query_preview=best.query_text[:50],
            age_seconds=round(best.age_seconds, 1) if best.age_seconds else 0,
            ttl_remaining=round(ttl_remaining, 1)
            if ttl_remaining is not None
            else None,
            latency_ms=round(elapsed_ms, 1),
        )

        if self.metrics and track_metrics:
            self.metrics.hits += 1
            self.metrics.total_latency_saved_ms += 2000
            self.metrics.record_lookup_latency(elapsed_ms)

        # Export metrics periodically
        self._otel_export_counter += 1
        if self._otel_export_counter % 100 == 0:
            self._export_metrics_to_otel()

        return LookupResult(
            hit=True,
            result=best.result,
            similarity=best.similarity,
            matched_query=best.query_text,
            age_seconds=best.age_seconds,
            entry_id=getattr(best, "_id", None),
            context=best.context,
            ttl_remaining=ttl_remaining,
        )

    def _miss(
        self, reason: str, start_time: float, track_metrics: bool = True
    ) -> LookupResult:
        """Handle cache miss with metrics tracking.

        Args:
            reason: Miss reason for logging
            start_time: Lookup start timestamp
            track_metrics: Whether to track metrics

        Returns:
            LookupResult indicating miss
        """
        elapsed_ms = (time.time() - start_time) * 1000

        if self.metrics and track_metrics:
            self.metrics.misses += 1
            self.metrics.record_lookup_latency(elapsed_ms)

        logger.debug("cache_miss", reason=reason, latency_ms=round(elapsed_ms, 2))
        return LookupResult(hit=False)

    def lookup_batch(
        self,
        queries: List[str],
        contexts: List[Dict[str, Any]],
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
        track_metrics: bool = True,
    ) -> List[LookupResult]:
        """Batch lookup for multiple queries optimized for embeddings.

        Main optimization: generates all embeddings in a single batch call,
        which is 2-3x faster than generating them individually.

        Supports context-specific thresholds per query.

        Args:
            queries: List of query texts
            contexts: List of context dicts (one per query)
            similarity_threshold: Minimum similarity threshold (overrides config)
            query_mode: Query matching strategy (semantic, exact, auto)
            track_metrics: Whether to track metrics

        Returns:
            List of LookupResult objects (one per query)
        """
        batch_start = time.time()

        logger.debug(
            "batch_lookup_start",
            total_items=len(queries),
            query_mode=query_mode,
        )

        try:
            cache_count = self.storage.count()
            if cache_count == 0:
                logger.debug("batch_lookup_miss_empty_cache")
                return [
                    self._miss("cache_empty", batch_start, track_metrics=track_metrics)
                    for _ in queries
                ]

            # Determine actual mode for each query
            actual_modes = []
            for query in queries:
                if query_mode == "auto":
                    use_exact = should_use_exact_mode(query)
                    actual_modes.append("exact" if use_exact else "semantic")
                else:
                    actual_modes.append(query_mode)

            # Separate queries by mode
            semantic_indices = [
                i for i, mode in enumerate(actual_modes) if mode == "semantic"
            ]
            exact_indices = [
                i for i, mode in enumerate(actual_modes) if mode == "exact"
            ]

            results = [None] * len(queries)

            # Batch process semantic queries
            if semantic_indices:
                semantic_queries = [queries[i] for i in semantic_indices]

                embed_start = time.time()
                embeddings = self.embedder.embed_batch(semantic_queries)
                embed_ms = (time.time() - embed_start) * 1000
                logger.debug(
                    "batch_embeddings_generated",
                    count=len(semantic_queries),
                    latency_ms=round(embed_ms, 1),
                    per_item_ms=round(embed_ms / len(semantic_queries), 2),
                )

                for idx, embedding in zip(semantic_indices, embeddings):
                    threshold = similarity_threshold
                    if threshold is None:
                        threshold = self.config.get_threshold_for_context(contexts[idx])

                    result = self._lookup_with_embedding(
                        queries[idx],
                        contexts[idx],
                        embedding,
                        threshold,
                        track_metrics=track_metrics,
                    )
                    results[idx] = result

            # Process exact queries individually
            if exact_indices:
                for idx in exact_indices:
                    result = self.lookup(
                        queries[idx],
                        contexts[idx],
                        similarity_threshold,
                        query_mode="exact",
                        track_metrics=track_metrics,
                    )
                    results[idx] = result

            batch_ms = (time.time() - batch_start) * 1000
            hits = sum(1 for r in results if r.is_hit)

            logger.info(
                "batch_lookup_complete",
                total=len(queries),
                hits=hits,
                misses=len(queries) - hits,
                hit_rate=round((hits / len(queries)) * 100, 1),
                total_ms=round(batch_ms, 1),
                per_item_ms=round(batch_ms / len(queries), 2),
            )

            return results

        except Exception as e:
            elapsed_ms = (time.time() - batch_start) * 1000
            logger.error(
                "batch_lookup_error",
                error_type=type(e).__name__,
                error_message=str(e),
                total_items=len(queries),
                latency_ms=round(elapsed_ms, 1),
                exc_info=True,
            )
            return [LookupResult(hit=False) for _ in queries]

    def _lookup_with_embedding(
        self,
        query: str,
        context: Dict[str, Any],
        embedding: List[float],
        similarity_threshold: Optional[float] = None,
        track_metrics: bool = True,
    ) -> LookupResult:
        """Internal lookup with pre-generated embedding.

        Used by lookup_batch to avoid regenerating embeddings.
        This is a performance optimization for batch operations.

        Args:
            query: Query text (for logging/context only)
            context: Context dict for exact matching
            embedding: Pre-computed embedding vector
            similarity_threshold: Minimum similarity score
            track_metrics: Whether to track metrics

        Returns:
            LookupResult with hit status and data
        """
        start_time = time.time()
        context = context or {}

        try:
            if similarity_threshold is None:
                threshold = self.config.get_threshold_for_context(context)
            else:
                threshold = similarity_threshold

            search_start = time.time()
            candidates = self.storage.search(
                embedding=embedding,
                context=context,
                limit=1,
                similarity_threshold=threshold,
                query_mode="semantic",
                query_text=query,
            )
            search_ms = (time.time() - search_start) * 1000

            if not candidates:
                logger.debug(
                    "lookup_with_embedding_miss",
                    query_preview=query[:50],
                    search_ms=round(search_ms, 1),
                    threshold=threshold,
                )
                return self._miss("no_match", start_time, track_metrics=track_metrics)

            return self._process_hit(candidates[0], start_time, track_metrics)

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                "lookup_with_embedding_error",
                error=str(e),
                query_preview=query[:50],
                latency_ms=round(elapsed_ms, 1),
                exc_info=True,
            )

            if self.metrics and track_metrics:
                self.metrics.misses += 1
                self.metrics.lookup_errors += 1

            return LookupResult(hit=False)

    def check_availability(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
    ) -> bool:
        """Check if cached result exists without retrieving it.

        Lightweight check used by schedulers for availability verification.

        Args:
            query: Query text
            context: Context dict
            similarity_threshold: Minimum similarity threshold
            query_mode: Query mode (semantic/exact/auto)

        Returns:
            True if cache entry exists and is valid
        """
        start_time = time.time()
        context = context or {}

        try:
            cache_count = self.storage.count()
            if cache_count == 0:
                return False

            # Auto-detect mode
            actual_mode = query_mode
            if query_mode == "auto":
                use_exact = should_use_exact_mode(query)
                actual_mode = "exact" if use_exact else "semantic"

            # Generate embedding if needed
            embedding = None
            if actual_mode == "semantic":
                embedding = self.embedder.embed(query)

            # Determine threshold
            if similarity_threshold is None:
                threshold = self.config.get_threshold_for_context(context)
            else:
                threshold = similarity_threshold

            # Search for candidates
            candidates = self.storage.search(
                embedding=embedding,
                context=context,
                limit=1,
                similarity_threshold=threshold,
                query_mode=actual_mode,
                query_text=query,
            )

            if not candidates:
                return False

            # Check TTL
            best = candidates[0]
            entry_ttl = (
                best.ttl_seconds
                if best.ttl_seconds is not None
                else self.config.ttl_seconds
            )

            if entry_ttl is not None:
                age = best.age_seconds if best.age_seconds else 0
                if age > entry_ttl:
                    return False

            check_ms = (time.time() - start_time) * 1000
            logger.debug(
                "availability_check_complete",
                available=True,
                latency_ms=round(check_ms, 1),
            )
            return True

        except Exception as e:
            logger.error(
                "availability_check_failed",
                error=str(e),
                query_preview=query[:50],
                exc_info=True,
            )
            return False
