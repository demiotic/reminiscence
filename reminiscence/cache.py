"""Core cache operations - lookup, store, maintenance."""

import time
import json
from typing import Optional, Dict, Any, List

from .types import LookupResult, CacheEntry
from .utils.logging import get_logger
from .utils.query_detection import should_use_exact_mode

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
        """Generate consistent entry ID for eviction tracking."""
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
        _track_metrics: bool = True,
    ) -> LookupResult:
        """
        Search cache by query with exact context matching.

        Args:
            query: Query text to search
            context: Context dict for exact matching
            similarity_threshold: Minimum similarity score (overrides config)
            query_mode: Query matching strategy
            _track_metrics: Internal flag to control metrics tracking

        Returns:
            LookupResult with hit status and data
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
                    "cache_empty", start_time, track_metrics=_track_metrics
                )

            logger.debug("cache_size", entries=cache_count)

            # Handle auto mode with intelligent detection
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

            # Generate embedding only for semantic mode
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

            threshold = similarity_threshold or self.config.similarity_threshold

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
                return self._miss(reason, start_time, track_metrics=_track_metrics)

            logger.debug(
                "lookup_candidates_found",
                count=len(candidates),
                search_ms=round(search_ms, 1),
            )

            return self._process_hit(candidates[0], start_time, _track_metrics)

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
            if self.metrics and _track_metrics:
                self.metrics.misses += 1
                self.metrics.record_lookup_latency(elapsed_ms)
                self.metrics.lookup_errors += 1
            return LookupResult(hit=False)

    def _process_hit(
        self, best: CacheEntry, start_time: float, track_metrics: bool
    ) -> LookupResult:
        """Process cache hit with TTL check and metrics."""
        if self._is_expired(best):
            logger.debug(
                "lookup_miss_expired",
                query_preview=best.query_text[:50],
                age_seconds=round(best.age_seconds, 1) if best.age_seconds else 0,
                ttl_seconds=self.config.ttl_seconds,
            )
            return self._miss("expired", start_time, track_metrics=track_metrics)

        entry_id = self._generate_entry_id(best.query_text, best.context)
        self.eviction.on_access(entry_id)

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            "cache_hit",
            similarity=round(best.similarity, 3) if best.similarity else 1.0,
            query_preview=best.query_text[:50],
            age_seconds=round(best.age_seconds, 1) if best.age_seconds else 0,
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
            entry_id=getattr(best, "id", None),
            context=best.context,
        )

    def _is_error_result(self, result: Any) -> bool:
        """Check if result represents an error that shouldn't be cached."""
        # 1. Exception objects
        if isinstance(result, Exception):
            logger.debug("error_detected_exception", error_type=type(result).__name__)
            return True

        # 2. Dict with error keys
        if isinstance(result, dict):
            error_keys = {"error", "exception", "traceback", "error_message", "failed"}
            found_keys = [k for k in error_keys if k in result]
            if found_keys:
                logger.debug("error_detected_dict", error_keys=found_keys)
                return True

        # 3. String error patterns
        if isinstance(result, str):
            error_patterns = ["error:", "exception:", "traceback:", "failed:"]
            if any(result.lower().startswith(pattern) for pattern in error_patterns):
                logger.debug("error_detected_string_pattern")
                return True

        # 4. None results
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
        Batch lookup for multiple queries (optimized for embeddings).

        Main optimization: generates all embeddings in a single batch call,
        which is 2-3x faster than generating them individually.

        Args:
            queries: List of query texts
            contexts: List of context dicts (one per query)
            similarity_threshold: Minimum similarity threshold
            query_mode: Query matching strategy ("semantic", "exact", "auto")
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

            # Determine actual mode for each query (handle "auto" mode)
            actual_modes = []
            for query in queries:
                if query_mode == "auto":
                    use_exact = should_use_exact_mode(query)
                    actual_modes.append("exact" if use_exact else "semantic")
                else:
                    actual_modes.append(query_mode)

            # Group indices by mode for optimization
            semantic_indices = [
                i for i, mode in enumerate(actual_modes) if mode == "semantic"
            ]
            exact_indices = [
                i for i, mode in enumerate(actual_modes) if mode == "exact"
            ]

            results = [None] * len(queries)

            # Batch process semantic queries (main optimization)
            if semantic_indices:
                semantic_queries = [queries[i] for i in semantic_indices]

                # Batch embedding generation (THIS IS THE KEY OPTIMIZATION)
                embed_start = time.time()
                embeddings = self.embedder.embed_batch(semantic_queries)
                embed_ms = (time.time() - embed_start) * 1000

                logger.debug(
                    "batch_embeddings_generated",
                    count=len(semantic_queries),
                    latency_ms=round(embed_ms, 1),
                    per_item_ms=round(embed_ms / len(semantic_queries), 2),
                )

                # Lookup each with pre-generated embedding
                for idx, embedding in zip(semantic_indices, embeddings):
                    result = self._lookup_with_embedding(
                        queries[idx],
                        contexts[idx],
                        embedding,
                        similarity_threshold,
                        track_metrics=track_metrics,
                    )
                    results[idx] = result

            # Process exact queries (no embeddings needed)
            if exact_indices:
                for idx in exact_indices:
                    result = self.lookup(
                        queries[idx],
                        contexts[idx],
                        similarity_threshold,
                        query_mode="exact",
                        _track_metrics=track_metrics,
                    )
                    results[idx] = result

            batch_ms = (time.time() - batch_start) * 1000

            hits = sum(1 for r in results if r.is_hit)

            logger.info(
                "batch_lookup_complete",
                total=len(queries),
                hits=hits,
                misses=len(queries) - hits,
                hit_rate=round(hits / len(queries) * 100, 1),
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

            # Return misses for all on error
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
            threshold = similarity_threshold or self.config.similarity_threshold

            # Search with pre-computed embedding
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
                )
                return self._miss("no_match", start_time, track_metrics=track_metrics)

            # Process hit with TTL check
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
    ):
        """Store result in cache with context."""
        store_start = time.time()

        logger.debug(
            "store_start",
            query_preview=query[:50],
            query_length=len(query),
            context_keys=list(context.keys()),
            query_mode=query_mode,
            allow_errors=allow_errors,
        )

        try:
            # Validate result before caching
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

            # Handle auto mode
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

            # Generate embedding only if needed
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
            )

            # Storage add
            storage_start = time.time()
            self.storage.add([entry])
            storage_ms = (time.time() - storage_start) * 1000

            logger.debug(
                "store_entry_added",
                storage_ms=round(storage_ms, 1),
            )

            entry_id = self._generate_entry_id(query, context)
            self.eviction.on_insert(entry_id)

            if self.config.auto_create_index:
                index_start = time.time()
                self.storage.maybe_auto_create_index(
                    self.config.index_threshold_entries,
                    self.config.index_num_partitions,
                )
                index_ms = (time.time() - index_start) * 1000
                if index_ms > 1.0:  # Only log if significant
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
    ):
        """Store multiple results in batch (optimized for embeddings)."""
        batch_start = time.time()

        logger.info(
            "batch_store_start",
            total_items=len(queries),
            query_mode=query_mode,
            allow_errors=allow_errors,
        )

        # Filter out errors if allow_errors=False
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

        # Nothing to store
        if not valid_entries:
            logger.warning("batch_store_skipped_all_errors", total=len(results))
            return

        # Filter to valid entries only
        valid_queries = [queries[i] for i in valid_entries]
        valid_contexts = [contexts[i] for i in valid_entries]
        valid_results = [results[i] for i in valid_entries]
        valid_metadata = [metadata[i] if metadata else None for i in valid_entries]

        logger.debug(
            "batch_store_valid_entries",
            valid=len(valid_entries),
            skipped=len(results) - len(valid_entries),
        )

        # ← FIX: Detect mode for each query when query_mode="auto"
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

        # Log summary of detected modes
        if query_mode == "auto":
            exact_count = sum(1 for m in actual_modes if m == "exact")
            semantic_count = sum(1 for m in actual_modes if m == "semantic")
            logger.info(
                "batch_auto_mode_summary",
                total=len(actual_modes),
                exact=exact_count,
                semantic=semantic_count,
            )

        # Batch embedding - only for queries that need semantic
        embed_start = time.time()
        embeddings = [None] * len(valid_queries)

        semantic_indices = [
            i for i, mode in enumerate(actual_modes) if mode == "semantic"
        ]
        if semantic_indices:
            semantic_queries = [valid_queries[i] for i in semantic_indices]
            semantic_embeddings = self.embedder.embed_batch(semantic_queries)

            # Insert embeddings in correct positions
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

        # Build entries with detected mode in metadata
        entries_start = time.time()
        entries = []
        for i, query in enumerate(valid_queries):
            # Add detected mode to metadata
            meta = valid_metadata[i] or {}
            meta["query_mode"] = actual_modes[i]  # ← Store detected mode

            entry = CacheEntry(
                query_text=query,
                context=valid_contexts[i],
                embedding=embeddings[i],
                result=valid_results[i],
                timestamp=time.time(),
                metadata=meta,
            )
            entries.append(entry)
        entries_ms = (time.time() - entries_start) * 1000

        logger.debug(
            "batch_entries_created",
            count=len(entries),
            latency_ms=round(entries_ms, 1),
        )

        # Storage add (this is where serialization happens)
        storage_start = time.time()
        logger.debug("batch_storage_add_start", entries=len(entries))

        # Pass query_mode for logging, but storage will use individual metadata
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
        """Remove expired entries based on TTL."""
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
        """Invalidate cache entries by criteria."""
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

    def get_metrics_report(self) -> Optional[Dict[str, Any]]:
        """Get current metrics report."""
        if self.metrics:
            return self.metrics.report()
        return None

    def export_metrics_now(self):
        """Force immediate export of metrics to OpenTelemetry."""
        self._export_metrics_to_otel()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        if self.config.ttl_seconds is None:
            return False
        age = entry.age_seconds if entry.age_seconds else 0
        return age > self.config.ttl_seconds

    def _evict_one(self):
        """Evict one entry using configured eviction policy."""
        evict_start = time.time()
        try:
            victim_id = self.eviction.select_victim()
            logger.debug("eviction_victim_selected", victim_id=victim_id[:50])

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

                rows = arrow_table.to_pylist()

                victim_row = None
                for row in rows:
                    entry_id = self._generate_entry_id(
                        row.get("query_text", ""), row.get("context", "{}")
                    )
                    if entry_id == victim_id:
                        victim_row = row
                        break

                if victim_row is not None:
                    victim_ts = victim_row.get("timestamp")

                    if self.config.db_uri == "memory://":
                        import pyarrow.compute as pc

                        mask = pc.not_equal(arrow_table["timestamp"], victim_ts)
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
                        table.delete(f"timestamp = {victim_ts}")

                    self.eviction.on_evict(victim_id)

                    evict_ms = (time.time() - evict_start) * 1000
                    logger.info(
                        "entry_evicted",
                        policy=self.config.eviction_policy,
                        victim_id=victim_id[:50],
                        table=table_name,
                        latency_ms=round(evict_ms, 1),
                    )
                    return

            logger.warning("victim_not_found", victim_id=victim_id)

        except ValueError as e:
            logger.warning("eviction_failed_no_entries", error=str(e))
        except Exception as e:
            logger.error("eviction_failed", error=str(e), exc_info=True)

    def _miss(
        self, reason: str, start_time: float, track_metrics: bool = True
    ) -> LookupResult:
        """Create miss result with logging."""
        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug("lookup_miss", reason=reason, latency_ms=round(elapsed_ms, 1))

        if self.metrics and track_metrics:
            self.metrics.misses += 1
            self.metrics.record_lookup_latency(elapsed_ms)

        return LookupResult(hit=False)
