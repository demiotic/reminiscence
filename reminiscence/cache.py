"""Core cache operations - lookup, store, maintenance."""

import time
import json
from typing import Optional, Dict, Any, List

from .types import LookupResult, CacheEntry, BulkInvalidatePattern
from .utils.logging import get_logger
from .utils.query_detection import should_use_exact_mode
from .utils.fingerprint import create_fingerprint

logger = get_logger(__name__)


class CacheOperations:
    """Handles all cache operations with hybrid matching."""

    def __init__(
        self,
        storage,
        embedder,
        eviction,
        config,
        metrics=None,
    ):
        """
        Initialize cache operations.

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

        logger.debug(
            "cache_operations_initialized",
            eviction_policy=config.eviction_policy,
            ttl_seconds=config.ttl_seconds,
            max_entries=config.max_entries,
        )

        if config.otel_enabled:
            try:
                from .metrics.exporters import OpenTelemetryExporter

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
        """
        Generate consistent entry ID for eviction tracking.

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
        """
        Search cache by query with exact context matching.

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
        """
        Process cache hit with per-entry TTL check and metrics.

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

    def _is_error_result(self, result: Any) -> bool:
        """
        Check if result represents an error that shouldn't be cached.

        Args:
            result: Result object to check

        Returns:
            True if result is an error
        """
        if isinstance(result, Exception):
            logger.debug("error_detected_exception", error_type=type(result).__name__)
            return True

        if isinstance(result, dict):
            error_keys = {"error", "exception", "traceback", "error_message", "failed"}
            found_keys = [k for k in error_keys if k in result]
            if found_keys:
                logger.debug("error_detected_dict", error_keys=found_keys)
                return True

        if isinstance(result, str):
            error_patterns = ["error:", "exception:", "traceback:", "failed:"]
            if any(result.lower().startswith(pattern) for pattern in error_patterns):
                logger.debug("error_detected_string_pattern")
                return True

        if result is None:
            logger.debug("error_detected_none_result")
            return True

        return False

    def lookup_batch(
        self,
        queries: List[str],
        contexts: List[Dict[str, Any]],
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
        track_metrics: bool = True,
    ) -> List[LookupResult]:
        """
        Batch lookup for multiple queries optimized for embeddings.

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

            actual_modes = []
            for query in queries:
                if query_mode == "auto":
                    use_exact = should_use_exact_mode(query)
                    actual_modes.append("exact" if use_exact else "semantic")
                else:
                    actual_modes.append(query_mode)

            semantic_indices = [
                i for i, mode in enumerate(actual_modes) if mode == "semantic"
            ]
            exact_indices = [
                i for i, mode in enumerate(actual_modes) if mode == "exact"
            ]

            results = [None] * len(queries)

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
        """
        Internal lookup with pre-generated embedding.

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

    def store(
        self,
        query: str,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        query_mode: str = "semantic",
        allow_errors: bool = False,
        ttl_seconds: Optional[int] = None,
        context_threshold: Optional[float] = None,
    ):
        """
        Store result in cache with context.

        Args:
            query: Query text
            context: Context dict
            result: Result to cache
            metadata: Additional metadata
            query_mode: Query mode (semantic/exact/auto)
            allow_errors: Whether to cache error results
            ttl_seconds: Per-entry TTL (overrides global config)
            context_threshold: Per-entry similarity threshold
        """
        store_start = time.time()

        logger.debug(
            "store_start",
            query_preview=query[:50],
            query_length=len(query),
            context_keys=list(context.keys()),
            query_mode=query_mode,
            allow_errors=allow_errors,
            ttl_seconds=ttl_seconds,
            context_threshold=context_threshold,
        )

        try:
            if not allow_errors and self._is_error_result(result):
                logger.debug(
                    "skipping_error_cache",
                    query_preview=query[:50],
                    result_type=type(result).__name__,
                    reason="error_result_detected",
                )
                if self.metrics:
                    self.metrics.store_errors += 1
                return

            current_count = self.storage.count()
            if self.config.max_entries and current_count >= self.config.max_entries:
                logger.debug(
                    "cache_eviction_triggered",
                    reason="max_entries_reached",
                    current_count=current_count,
                    max_entries=self.config.max_entries,
                )
                self._evict_one()

            embedding = None
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

            if actual_mode == "semantic":
                embed_start = time.time()
                embedding = self.embedder.embed(query)
                embed_ms = (time.time() - embed_start) * 1000
                logger.debug(
                    "store_embedding_generated",
                    latency_ms=round(embed_ms, 1),
                    embedding_dim=len(embedding) if embedding else 0,
                )
            else:
                logger.debug("store_embedding_skipped_exact_mode")

            timestamp = time.time()
            metadata_dict = metadata or {}
            metadata_dict["query_mode"] = actual_mode

            entry = CacheEntry(
                query_text=query,
                context=context,
                embedding=embedding,
                result=result,
                timestamp=timestamp,
                metadata=metadata_dict,
                ttl_seconds=ttl_seconds,
                context_threshold=context_threshold,
            )

            storage_start = time.time()
            self.storage.add([entry])
            storage_ms = (time.time() - storage_start) * 1000
            logger.debug("store_entry_added", storage_ms=round(storage_ms, 1))

            entry_id = self._generate_entry_id(query, context)
            self.eviction.on_insert(entry_id)

            if self.config.auto_create_index:
                index_start = time.time()
                self.storage.maybe_auto_create_index(
                    self.config.index_threshold_entries,
                    self.config.index_num_partitions,
                )
                index_ms = (time.time() - index_start) * 1000
                if index_ms > 1.0:
                    logger.debug("store_index_check", latency_ms=round(index_ms, 1))

            if self.metrics:
                try:
                    result_str = json.dumps(result)
                    result_size = len(result_str.encode("utf-8"))
                    self.metrics.record_result_size(result_size)
                except Exception:
                    pass

            total_ms = (time.time() - store_start) * 1000
            logger.info(
                "cache_store_success",
                query_preview=query[:50],
                context_keys=list(context.keys()),
                cache_entries=self.storage.count(),
                query_mode=actual_mode,
                total_ms=round(total_ms, 1),
                storage_ms=round(storage_ms, 1),
                ttl_seconds=ttl_seconds,
            )

        except Exception as e:
            elapsed_ms = (time.time() - store_start) * 1000
            logger.error(
                "cache_store_error",
                error_type=type(e).__name__,
                error_message=str(e),
                query_preview=query[:50],
                context_preview=str(context)[:100],
                query_mode=query_mode,
                latency_ms=round(elapsed_ms, 1),
                exc_info=True,
            )
            if self.metrics:
                self.metrics.store_errors += 1

    def store_batch(
        self,
        queries: List[str],
        contexts: List[Dict[str, Any]],
        results: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
        query_mode: str = "semantic",
        allow_errors: bool = False,
        ttl_seconds: Optional[List[Optional[int]]] = None,
        context_thresholds: Optional[List[Optional[float]]] = None,
    ):
        """
        Store multiple results in batch optimized for embeddings.

        Args:
            queries: List of query texts
            contexts: List of context dicts
            results: List of results to cache
            metadata: Optional list of metadata dicts
            query_mode: Query mode (semantic/exact/auto)
            allow_errors: Whether to cache error results
            ttl_seconds: Per-entry TTL overrides (one per query)
            context_thresholds: Per-entry thresholds (one per query)
        """
        batch_start = time.time()

        logger.info(
            "batch_store_start",
            total_items=len(queries),
            query_mode=query_mode,
            allow_errors=allow_errors,
        )

        valid_entries = []
        for i, result in enumerate(results):
            if not allow_errors and self._is_error_result(result):
                logger.debug(
                    "skipping_error_in_batch",
                    query_preview=queries[i][:50],
                    index=i,
                    result_type=type(result).__name__,
                )
                if self.metrics:
                    self.metrics.store_errors += 1
                continue
            valid_entries.append(i)

        if not valid_entries:
            logger.warning("batch_store_skipped_all_errors", total=len(results))
            return

        valid_queries = [queries[i] for i in valid_entries]
        valid_contexts = [contexts[i] for i in valid_entries]
        valid_results = [results[i] for i in valid_entries]
        valid_metadata = [metadata[i] if metadata else None for i in valid_entries]
        valid_ttls = [ttl_seconds[i] if ttl_seconds else None for i in valid_entries]
        valid_thresholds = [
            context_thresholds[i] if context_thresholds else None for i in valid_entries
        ]

        logger.debug(
            "batch_store_valid_entries",
            valid=len(valid_entries),
            skipped=len(results) - len(valid_entries),
        )

        actual_modes = []
        for query in valid_queries:
            if query_mode == "auto":
                use_exact = should_use_exact_mode(query)
                detected_mode = "exact" if use_exact else "semantic"
                actual_modes.append(detected_mode)
                logger.debug(
                    "batch_auto_mode_detected_per_query",
                    query_preview=query[:50],
                    detected_mode=detected_mode,
                )
            else:
                actual_modes.append(query_mode)

        if query_mode == "auto":
            exact_count = sum(1 for m in actual_modes if m == "exact")
            semantic_count = sum(1 for m in actual_modes if m == "semantic")
            logger.info(
                "batch_auto_mode_summary",
                total=len(actual_modes),
                exact=exact_count,
                semantic=semantic_count,
            )

        embed_start = time.time()
        embeddings = [None] * len(valid_queries)

        semantic_indices = [
            i for i, mode in enumerate(actual_modes) if mode == "semantic"
        ]

        if semantic_indices:
            semantic_queries = [valid_queries[i] for i in semantic_indices]
            semantic_embeddings = self.embedder.embed_batch(semantic_queries)

            for idx, emb in zip(semantic_indices, semantic_embeddings):
                embeddings[idx] = emb

            embed_ms = (time.time() - embed_start) * 1000
            logger.debug(
                "batch_embeddings_generated",
                count=len(semantic_queries),
                latency_ms=round(embed_ms, 1),
                per_item_ms=round(embed_ms / len(semantic_queries), 2)
                if semantic_queries
                else 0,
            )
        else:
            logger.debug("batch_embeddings_skipped", reason="all_exact_mode")
            embed_ms = 0

        entries_start = time.time()
        entries = []

        for i, query in enumerate(valid_queries):
            meta = valid_metadata[i] or {}
            meta["query_mode"] = actual_modes[i]

            entry = CacheEntry(
                query_text=query,
                context=valid_contexts[i],
                embedding=embeddings[i],
                result=valid_results[i],
                timestamp=time.time(),
                metadata=meta,
                ttl_seconds=valid_ttls[i],
                context_threshold=valid_thresholds[i],
            )
            entries.append(entry)

        entries_ms = (time.time() - entries_start) * 1000
        logger.debug(
            "batch_entries_created",
            count=len(entries),
            latency_ms=round(entries_ms, 1),
        )

        storage_start = time.time()
        logger.debug("batch_storage_add_start", entries=len(entries))

        self.storage.add(entries)

        storage_ms = (time.time() - storage_start) * 1000
        logger.debug(
            "batch_storage_add_complete",
            entries=len(entries),
            latency_ms=round(storage_ms, 1),
            per_item_ms=round(storage_ms / len(entries), 2) if entries else 0,
        )

        batch_total_ms = (time.time() - batch_start) * 1000
        logger.info(
            "batch_store_complete",
            total_entries=len(entries),
            skipped_errors=len(results) - len(entries),
            embed_ms=round(embed_ms, 1),
            storage_ms=round(storage_ms, 1),
            total_ms=round(batch_total_ms, 1),
            per_item_ms=round(batch_total_ms / len(entries), 1) if entries else 0,
        )

    def cleanup_expired(self) -> int:
        """
        Remove expired entries based on TTL.

        Returns:
            Number of entries deleted
        """
        if self.config.ttl_seconds is None:
            logger.warning("cleanup_called_no_ttl")
            return 0

        cleanup_start = time.time()
        logger.debug("cleanup_expired_start", ttl_seconds=self.config.ttl_seconds)

        try:
            import pyarrow.compute as pc

            exact_table = self.storage.exact_table.to_arrow()
            semantic_table = self.storage.semantic_table.to_arrow()

            cutoff = time.time() - self.config.ttl_seconds
            deleted_total = 0

            if len(exact_table) > 0:
                before = len(exact_table)
                expired_mask = pc.less_equal(exact_table["timestamp"], cutoff)
                expired_rows = exact_table.filter(expired_mask).to_pylist()

                for row in expired_rows:
                    entry_id = self._generate_entry_id(
                        row.get("query_text", ""), row.get("context", "{}")
                    )
                    try:
                        self.eviction.on_evict(entry_id)
                    except Exception:
                        pass

                mask = pc.greater(exact_table["timestamp"], cutoff)

                if self.config.db_uri == "memory://":
                    filtered = exact_table.filter(mask)
                    self.storage.exact_table = self.storage.db.create_table(
                        self.storage._exact_table_name,
                        data=filtered if len(filtered) > 0 else None,
                        schema=self.storage.exact_schema
                        if len(filtered) == 0
                        else None,
                        mode="overwrite",
                    )
                else:
                    self.storage.exact_table.delete(f"timestamp <= {cutoff}")

                deleted_total += before - len(self.storage.exact_table.to_arrow())

            if len(semantic_table) > 0:
                before = len(semantic_table)
                expired_mask = pc.less_equal(semantic_table["timestamp"], cutoff)
                expired_rows = semantic_table.filter(expired_mask).to_pylist()

                for row in expired_rows:
                    entry_id = self._generate_entry_id(
                        row.get("query_text", ""), row.get("context", "{}")
                    )
                    try:
                        self.eviction.on_evict(entry_id)
                    except Exception:
                        pass

                mask = pc.greater(semantic_table["timestamp"], cutoff)

                if self.config.db_uri == "memory://":
                    filtered = semantic_table.filter(mask)
                    self.storage.semantic_table = self.storage.db.create_table(
                        self.storage._semantic_table_name,
                        data=filtered if len(filtered) > 0 else None,
                        schema=self.storage.semantic_schema
                        if len(filtered) == 0
                        else None,
                        mode="overwrite",
                    )
                    self.storage.table = self.storage.semantic_table
                else:
                    self.storage.semantic_table.delete(f"timestamp <= {cutoff}")

                deleted_total += before - len(self.storage.semantic_table.to_arrow())

            cleanup_ms = (time.time() - cleanup_start) * 1000
            logger.info(
                "cleaned_up_expired",
                deleted=deleted_total,
                latency_ms=round(cleanup_ms, 1),
            )
            return deleted_total

        except Exception as e:
            logger.error("cleanup_failed", error=str(e), exc_info=True)
            return 0

    def invalidate(
        self,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """
        Invalidate cache entries by criteria.

        Args:
            query: Query text pattern to match
            context: Context dict to match exactly
            older_than_seconds: Delete entries older than this

        Returns:
            Number of entries deleted
        """
        if query is None and context is None and older_than_seconds is None:
            logger.warning("invalidate_called_without_criteria")
            return 0

        invalidate_start = time.time()
        logger.debug(
            "invalidate_start",
            has_query=query is not None,
            has_context=context is not None,
            older_than_seconds=older_than_seconds,
        )

        try:
            import pyarrow.compute as pc

            before = self.storage.count()
            entries_to_remove = []

            if older_than_seconds is not None:
                cutoff = time.time() - older_than_seconds

                for table, table_name, schema in [
                    (
                        self.storage.exact_table,
                        self.storage._exact_table_name,
                        self.storage.exact_schema,
                    ),
                    (
                        self.storage.semantic_table,
                        self.storage._semantic_table_name,
                        self.storage.semantic_schema,
                    ),
                ]:
                    arrow_table = table.to_arrow()
                    if len(arrow_table) == 0:
                        continue

                    old_mask = pc.less_equal(arrow_table["timestamp"], cutoff)
                    old_rows = arrow_table.filter(old_mask).to_pylist()
                    entries_to_remove.extend(old_rows)

                    if self.config.db_uri == "memory://":
                        mask = pc.greater(arrow_table["timestamp"], cutoff)
                        filtered = arrow_table.filter(mask)
                        new_table = self.storage.db.create_table(
                            table_name,
                            data=filtered if len(filtered) > 0 else None,
                            schema=schema if len(filtered) == 0 else None,
                            mode="overwrite",
                        )
                        if table_name == self.storage._exact_table_name:
                            self.storage.exact_table = new_table
                        else:
                            self.storage.semantic_table = new_table
                            self.storage.table = new_table
                    else:
                        table.delete(f"timestamp <= {cutoff}")

            elif context is not None:
                context_json = json.dumps(context, sort_keys=True)

                for table, table_name, schema in [
                    (
                        self.storage.exact_table,
                        self.storage._exact_table_name,
                        self.storage.exact_schema,
                    ),
                    (
                        self.storage.semantic_table,
                        self.storage._semantic_table_name,
                        self.storage.semantic_schema,
                    ),
                ]:
                    arrow_table = table.to_arrow()
                    if len(arrow_table) == 0:
                        continue

                    context_mask = pc.equal(arrow_table["context"], context_json)
                    context_rows = arrow_table.filter(context_mask).to_pylist()
                    entries_to_remove.extend(context_rows)

                    if self.config.db_uri == "memory://":
                        mask = pc.not_equal(arrow_table["context"], context_json)
                        filtered = arrow_table.filter(mask)
                        new_table = self.storage.db.create_table(
                            table_name,
                            data=filtered if len(filtered) > 0 else None,
                            schema=schema if len(filtered) == 0 else None,
                            mode="overwrite",
                        )
                        if table_name == self.storage._exact_table_name:
                            self.storage.exact_table = new_table
                        else:
                            self.storage.semantic_table = new_table
                            self.storage.table = new_table
                    else:
                        table.delete(f"context = '{context_json}'")

            elif query is not None:
                logger.warning("semantic_invalidation_not_implemented")
                return 0

            for row in entries_to_remove:
                entry_id = self._generate_entry_id(
                    row.get("query_text", ""), row.get("context", "{}")
                )
                try:
                    self.eviction.on_evict(entry_id)
                except Exception:
                    pass

            deleted = before - self.storage.count()
            invalidate_ms = (time.time() - invalidate_start) * 1000
            logger.info(
                "invalidated",
                deleted=deleted,
                latency_ms=round(invalidate_ms, 1),
            )
            return deleted

        except Exception as e:
            logger.error("invalidation_failed", error=str(e), exc_info=True)
            return 0

    def invalidate_bulk(self, pattern: BulkInvalidatePattern) -> int:
        """
        Bulk invalidate entries matching pattern using efficient batch deletion.

        This method scans all entries once, collects IDs of entries matching
        the pattern, and then deletes them in batch. Much more efficient than
        deleting one-by-one.

        Args:
            pattern: BulkInvalidatePattern with matching criteria

        Returns:
            Number of entries invalidated

        Raises:
            Exception: Re-raises any exception after logging

        Performance:
            - Single-pass scan: O(n) where n = total entries
            - Batch deletion: O(k) where k = matched entries
            - Total: O(n + k) vs O(n * k) for naive approach
        """
        bulk_start = time.time()
        invalidated_count = 0
        entry_ids_to_delete = []
        eviction_ids_to_notify = []

        try:
            # Scan both tables once to collect matching entry IDs
            for table in [self.storage.exact_table, self.storage.semantic_table]:
                arrow_table = table.to_arrow()
                if len(arrow_table) == 0:
                    continue

                rows = arrow_table.to_pylist()

                for row in rows:
                    # Extract entry data
                    entry_id = row.get("id")
                    query_text = row.get("query_text", "")
                    context_str = row.get("context", "{}")
                    timestamp = row.get("timestamp", 0)

                    # Parse context JSON
                    try:
                        context = json.loads(context_str) if context_str != "{}" else {}
                    except json.JSONDecodeError:
                        logger.warning(
                            "invalid_context_json_in_bulk",
                            entry_id=entry_id[:16] if entry_id else "unknown",
                        )
                        context = {}

                    # Calculate entry age
                    age_seconds = time.time() - timestamp

                    # Apply pattern matching filters
                    if not pattern.matches_query(query_text):
                        continue
                    if not pattern.matches_context(context):
                        continue
                    if not pattern.matches_age(age_seconds):
                        continue

                    # Entry matches all criteria - mark for deletion
                    entry_ids_to_delete.append(entry_id)

                    # Generate eviction tracking ID for policy notification
                    eviction_id = self._generate_entry_id(query_text, context)
                    eviction_ids_to_notify.append(eviction_id)

                    invalidated_count += 1

            # Early exit if nothing to delete
            if not entry_ids_to_delete:
                logger.debug("bulk_invalidation_no_matches")
                return 0

            # Delete all matched entries in batch
            logger.debug(
                "bulk_deletion_start",
                entries_to_delete=len(entry_ids_to_delete),
            )

            deletion_start = time.time()
            deleted_count = 0

            for entry_id in entry_ids_to_delete:
                if self.storage.delete_by_id(entry_id):
                    deleted_count += 1

            deletion_ms = (time.time() - deletion_start) * 1000

            logger.debug(
                "bulk_deletion_complete",
                attempted=len(entry_ids_to_delete),
                deleted=deleted_count,
                latency_ms=round(deletion_ms, 1),
            )

            # Notify eviction policy of all removed entries
            for eviction_id in eviction_ids_to_notify:
                try:
                    self.eviction.on_evict(eviction_id)
                except Exception as e:
                    logger.warning(
                        "eviction_notification_failed",
                        eviction_id=eviction_id,
                        error=str(e),
                    )

            # Update metrics
            if self.metrics:
                if not hasattr(self.metrics, "invalidations"):
                    self.metrics.invalidations = 0
                self.metrics.invalidations += invalidated_count

                if not hasattr(self.metrics, "bulk_invalidations"):
                    self.metrics.bulk_invalidations = 0
                self.metrics.bulk_invalidations += 1

            bulk_ms = (time.time() - bulk_start) * 1000

            logger.info(
                "bulk_invalidation_completed",
                matched=invalidated_count,
                deleted=deleted_count,
                total_ms=round(bulk_ms, 1),
                scan_ms=round(bulk_ms - deletion_ms, 1),
                deletion_ms=round(deletion_ms, 1),
            )

            return invalidated_count

        except Exception as e:
            elapsed_ms = (time.time() - bulk_start) * 1000
            logger.error(
                "bulk_invalidation_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(elapsed_ms, 1),
                exc_info=True,
            )
            raise

    def invalidate_by_prefix(self, query_prefix: str) -> int:
        """
        Invalidate all entries with query starting with prefix.

        This is a convenience method wrapping invalidate_bulk.

        Args:
            query_prefix: Prefix to match (e.g., "SELECT")

        Returns:
            Number of entries invalidated
        """
        pattern = BulkInvalidatePattern(query_prefix=query_prefix)
        return self.invalidate_bulk(pattern)

    def invalidate_by_regex(self, query_regex: str) -> int:
        """
        Invalidate all entries matching regex pattern.

        Args:
            query_regex: Regular expression pattern

        Returns:
            Number of entries invalidated
        """
        pattern = BulkInvalidatePattern(query_regex=query_regex)
        return self.invalidate_bulk(pattern)

    def invalidate_by_context(self, context_matches: Dict[str, str]) -> int:
        """
        Invalidate entries matching context pattern.

        Supports wildcard matching with asterisk (*).

        Args:
            context_matches: Dict of context patterns to match

        Examples:
            cache.invalidate_by_context({"model": "gpt-4"})
            cache.invalidate_by_context({"agent_*": "*"})

        Returns:
            Number of entries invalidated
        """
        pattern = BulkInvalidatePattern(context_matches=context_matches)
        return self.invalidate_bulk(pattern)

    def invalidate_older_than(self, seconds: float) -> int:
        """
        Invalidate all entries older than specified seconds.

        Args:
            seconds: Age threshold in seconds

        Returns:
            Number of entries invalidated
        """
        pattern = BulkInvalidatePattern(older_than_seconds=seconds)
        return self.invalidate_bulk(pattern)

    def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        clear_start = time.time()
        logger.warning("clearing_all_cache")

        try:
            before = self.storage.count()
            self.storage.exact_table.delete("timestamp > 0")
            self.storage.semantic_table.delete("timestamp > 0")

            if hasattr(self.eviction, "order"):
                self.eviction.order.clear()
            if hasattr(self.eviction, "access_times"):
                self.eviction.access_times.clear()
            if hasattr(self.eviction, "frequencies"):
                self.eviction.frequencies.clear()

            cleared = before
            clear_ms = (time.time() - clear_start) * 1000
            logger.warning(
                "cache_cleared",
                entries_deleted=cleared,
                latency_ms=round(clear_ms, 1),
            )
            return cleared

        except Exception as e:
            logger.error("clear_all_failed", error=str(e), exc_info=True)
            return 0

    def _evict_one(self):
        """
        Evict one entry to make space using the configured eviction policy.

        This method uses the eviction policy to select a victim entry, then
        efficiently deletes it by reconstructing the full entry ID from storage.

        Raises:
            Exception: If eviction fails, logs error but doesn't raise to
                    allow cache operations to continue
        """
        evict_start = time.time()

        try:
            victim_id = self.eviction.select_victim()

            # Parse victim_id format: "query_preview:context_json"
            parts = victim_id.split(":", 1)
            if len(parts) != 2:
                logger.error(
                    "invalid_victim_id_format",
                    victim_id=victim_id,
                    expected_format="query:context",
                )
                return

            query_preview = parts[0]
            context_str = parts[1]

            # Parse context JSON
            try:
                context = json.loads(context_str)
            except json.JSONDecodeError:
                logger.warning(
                    "invalid_context_json",
                    victim_id=victim_id,
                    context_str=context_str[:100],
                )
                context = {}

            # Find the actual entry in storage to get full query text
            # We need to search by context hash to find candidates
            context_hash = create_fingerprint(context)

            deleted = False

            # Search in both tables
            for table in [self.storage.exact_table, self.storage.semantic_table]:
                arrow_table = table.to_arrow()
                if len(arrow_table) == 0:
                    continue

                import pyarrow.compute as pc

                # Filter by context hash and query prefix
                context_mask = pc.equal(arrow_table["context_hash"], context_hash)
                query_mask = pc.starts_with(arrow_table["query_text"], query_preview)
                combined = pc.and_(context_mask, query_mask)

                filtered = arrow_table.filter(combined)

                if len(filtered) > 0:
                    # Found the entry - get its actual ID
                    entry_id = filtered["id"][0].as_py()

                    logger.debug(
                        "evicting_entry",
                        victim_id=victim_id,
                        entry_id=entry_id[:16],
                        policy=self.config.eviction_policy,
                    )

                    # Delete by actual ID
                    deleted = self.storage.delete_by_id(entry_id)
                    break

            if deleted:
                self.eviction.on_evict(victim_id)

                evict_ms = (time.time() - evict_start) * 1000
                logger.info(
                    "entry_evicted",
                    victim_id=victim_id,
                    policy=self.config.eviction_policy,
                    latency_ms=round(evict_ms, 1),
                )

                if self.metrics:
                    self.metrics.evictions += 1
            else:
                logger.warning(
                    "eviction_entry_not_found",
                    victim_id=victim_id,
                )

        except Exception as e:
            logger.error(
                "eviction_failed",
                error=str(e),
                policy=self.config.eviction_policy,
                exc_info=True,
            )

    def _miss(
        self, reason: str, start_time: float, track_metrics: bool = True
    ) -> LookupResult:
        """
        Handle cache miss with metrics tracking.

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

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics and status
        """
        try:
            exact_count = len(self.storage.exact_table.to_arrow())
            semantic_count = len(self.storage.semantic_table.to_arrow())
            total_count = exact_count + semantic_count

            stats_dict = {
                "total_entries": total_count,
                "exact_entries": exact_count,
                "semantic_entries": semantic_count,
                "max_entries": self.config.max_entries,
                "ttl_seconds": self.config.ttl_seconds,
                "eviction_policy": self.config.eviction_policy,
                "similarity_threshold": self.config.similarity_threshold,
            }

            if self.metrics:
                stats_dict.update(self.metrics.report())

            return stats_dict

        except Exception as e:
            logger.error("stats_failed", error=str(e), exc_info=True)
            return {"error": str(e)}

    def check_availability(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
    ) -> bool:
        """
        Check if cached result exists without retrieving it.

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

            actual_mode = query_mode
            if query_mode == "auto":
                use_exact = should_use_exact_mode(query)
                actual_mode = "exact" if use_exact else "semantic"

            embedding = None
            if actual_mode == "semantic":
                embedding = self.embedder.embed(query)

            if similarity_threshold is None:
                threshold = self.config.get_threshold_for_context(context)
            else:
                threshold = similarity_threshold

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

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """
        Get all cache entries as list of dicts.

        Returns:
            List of entry dictionaries
        """
        try:
            arrow_table = self.storage.to_arrow()
            entries = arrow_table.to_pylist()
            return entries
        except Exception as e:
            logger.error("get_all_entries_failed", error=str(e), exc_info=True)
            return []

    def export_to_file(self, filepath: str, format: str = "parquet"):
        """
        Export cache to file.

        Args:
            filepath: Output file path
            format: File format (parquet, json, csv)
        """
        try:
            import pyarrow.parquet as pq
            import pyarrow.json as pj
            import pyarrow.csv as pc

            arrow_table = self.storage.to_arrow()

            if format == "parquet":
                pq.write_table(arrow_table, filepath)
            elif format == "json":
                pj.write_json(arrow_table, filepath)
            elif format == "csv":
                pc.write_csv(arrow_table, filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(
                "cache_exported",
                filepath=filepath,
                format=format,
                entries=len(arrow_table),
            )

        except Exception as e:
            logger.error(
                "export_failed",
                filepath=filepath,
                format=format,
                error=str(e),
                exc_info=True,
            )
            raise

    def import_from_file(self, filepath: str, format: str = "parquet"):
        """
        Import cache from file.

        Args:
            filepath: Input file path
            format: File format (parquet, json, csv)
        """
        try:
            import pyarrow.parquet as pq
            import pyarrow.json as pj
            import pyarrow.csv as pc

            if format == "parquet":
                arrow_table = pq.read_table(filepath)
            elif format == "json":
                arrow_table = pj.read_json(filepath)
            elif format == "csv":
                arrow_table = pc.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            entries = arrow_table.to_pylist()
            for row in entries:
                entry = CacheEntry(
                    query_text=row.get("query_text", ""),
                    context=json.loads(row.get("context", "{}")),
                    embedding=row.get("embedding"),
                    result=row.get("result"),
                    timestamp=row.get("timestamp", time.time()),
                    metadata=json.loads(row.get("metadata", "{}")),
                    ttl_seconds=row.get("ttl_seconds"),
                    context_threshold=row.get("context_threshold"),
                )
                self.storage.add([entry])

            logger.info(
                "cache_imported",
                filepath=filepath,
                format=format,
                entries=len(entries),
            )

        except Exception as e:
            logger.error(
                "import_failed",
                filepath=filepath,
                format=format,
                error=str(e),
                exc_info=True,
            )
            raise
