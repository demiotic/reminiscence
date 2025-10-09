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
        try:
            arrow_table = self.storage.to_arrow()
            if len(arrow_table) > 0:
                rows = arrow_table.to_pylist()
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

                logger.info(
                    "synced_eviction_state",
                    entries=len(rows),
                    policy=self.config.eviction_policy,
                )
        except Exception as e:
            logger.warning("failed_to_sync_eviction_state", error=str(e))

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
            query_mode: Query matching strategy:
                - "semantic": Normal semantic search (default)
                - "exact": Hash-based exact matching (no embeddings)
                - "auto": Intelligent detection (SQL/code → exact, text → semantic)
            _track_metrics: Internal flag to control metrics tracking

        Returns:
            LookupResult with hit status and data
        """
        start_time = time.time()
        context = context or {}

        try:
            if self.storage.count() == 0:
                return self._miss(
                    "cache_empty", start_time, track_metrics=_track_metrics
                )

            # Handle auto mode with intelligent detection
            actual_mode = query_mode
            if query_mode == "auto":
                use_exact = should_use_exact_mode(query)
                actual_mode = "exact" if use_exact else "semantic"

                logger.debug(
                    "auto_mode_lookup",
                    query_preview=query[:50],
                    detected_mode=actual_mode,
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
                )

            threshold = similarity_threshold or self.config.similarity_threshold

            candidates = self.storage.search(
                embedding=embedding,
                context=context,
                limit=1,
                similarity_threshold=threshold,
                query_mode=actual_mode,
                query_text=query,
            )

            if not candidates:
                reason = "no_exact_match" if actual_mode == "exact" else "no_match"
                return self._miss(reason, start_time, track_metrics=_track_metrics)

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

    def store(
        self,
        query: str,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        query_mode: str = "semantic",
    ):
        """
        Store result in cache with context.

        Args:
            query: Query text
            context: Context dict (matched exactly in lookups)
            result: Result to cache
            metadata: Optional metadata
            query_mode: Storage mode:
                - "semantic": Generate embedding (default)
                - "exact": No embedding, hash-based
                - "auto": Intelligent detection (SQL/code → exact, text → semantic)
        """
        try:
            if (
                self.config.max_entries
                and self.storage.count() >= self.config.max_entries
            ):
                logger.debug(
                    "cache_eviction_triggered",
                    reason="max_entries_reached",
                    current_count=self.storage.count(),
                    max_entries=self.config.max_entries,
                )
                self._evict_one()

            # Handle auto mode with intelligent detection
            embedding = None
            actual_mode = query_mode

            if query_mode == "auto":
                # Detect query type intelligently
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
                    f"store_{actual_mode}_mode",
                    query_preview=query[:50],
                    embedding_latency_ms=round(embed_ms, 1),
                )
            else:  # exact
                logger.debug(
                    f"store_{actual_mode}_mode",
                    query_preview=query[:50],
                    embedding_skipped=True,
                )

            timestamp = time.time()

            metadata_dict = metadata or {}
            metadata_dict["query_mode"] = actual_mode

            entry = CacheEntry(
                query_text=query,
                context=context,
                embedding=embedding,  # None for exact mode
                result=result,
                timestamp=timestamp,
                metadata=metadata_dict,
            )

            # Store with resolved mode (exact or semantic, never "auto")
            self.storage.add([entry], query_mode=actual_mode)

            entry_id = self._generate_entry_id(query, context)
            self.eviction.on_insert(entry_id)

            if self.config.auto_create_index:
                self.storage.maybe_auto_create_index(
                    self.config.index_threshold_entries,
                    self.config.index_num_partitions,
                )

            if self.metrics:
                try:
                    result_str = json.dumps(result)
                    result_size = len(result_str.encode("utf-8"))
                    self.metrics.record_result_size(result_size)
                except Exception:
                    pass

            logger.debug(
                "cache_store_success",
                query_preview=query[:50],
                context_keys=list(context.keys()),
                cache_entries=self.storage.count(),
                query_mode=actual_mode,
            )

        except Exception as e:
            logger.error(
                "cache_store_error",
                error_type=type(e).__name__,
                error_message=str(e),
                query_preview=query[:50],
                context_preview=str(context)[:100],
                query_mode=query_mode,
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
    ):
        """Store multiple results in batch (optimized for embeddings)."""
        if query_mode in ("semantic", "auto"):
            # Batch embedding (3-5x faster than sequential)
            embeddings = self.embedder.embed_batch(queries)
        else:
            embeddings = [None] * len(queries)

        entries = []
        for i, query in enumerate(queries):
            entry = CacheEntry(
                query_text=query,
                context=contexts[i],
                embedding=embeddings[i],
                result=results[i],
                timestamp=time.time(),
                metadata=metadata[i] if metadata else None,
            )
            entries.append(entry)

        self.storage.add(entries, query_mode=query_mode)

    def cleanup_expired(self) -> int:
        """Remove expired entries based on TTL."""
        if self.config.ttl_seconds is None:
            logger.warning("no_ttl_configured")
            return 0

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

            logger.info("cleaned_up_expired", deleted=deleted_total)
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
            logger.info("invalidated", deleted=deleted)
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
        try:
            victim_id = self.eviction.select_victim()

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

                    logger.info(
                        "entry_evicted",
                        policy=self.config.eviction_policy,
                        victim_id=victim_id[:50],
                        table=table_name,
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

        logger.debug(reason, latency_ms=round(elapsed_ms, 1))

        if self.metrics and track_metrics:
            self.metrics.misses += 1
            self.metrics.record_lookup_latency(elapsed_ms)

        return LookupResult(hit=False)
