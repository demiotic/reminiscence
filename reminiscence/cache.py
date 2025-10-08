"""Core cache operations - lookup, store, maintenance."""

import time
import json
from typing import Optional, Dict, Any

from .types import LookupResult, CacheEntry
from .utils.logging import get_logger

logger = get_logger(__name__)


class CacheOperations:
    """
    Handles all cache operations with hybrid matching.

    Combines lookup (semantic + exact context), store, and maintenance.
    """

    def __init__(
        self,
        storage,
        embedder,
        eviction,
        config,
        metrics=None,
        enable_otel: bool = True,
        otel_endpoint: str = "http://localhost:4318/v1/metrics",
    ):
        self.storage = storage
        self.embedder = embedder
        self.eviction = eviction
        self.config = config
        self.metrics = metrics

        # OpenTelemetry exporter (optional)
        self.otel_exporter = None
        self._otel_export_counter = 0
        if enable_otel:
            try:
                from .metrics.exporters import OpenTelemetryExporter

                self.otel_exporter = OpenTelemetryExporter(endpoint=otel_endpoint)
                logger.info("opentelemetry_enabled", endpoint=otel_endpoint)
            except Exception as e:
                logger.warning("opentelemetry_init_failed", error=str(e))

        # Sync eviction policy with existing entries
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

                    # For LRU: use creation timestamp
                    if hasattr(self.eviction, "access_times"):
                        self.eviction.access_times[entry_id] = row.get(
                            "timestamp", time.time()
                        )

                    # For LFU: initialize frequency
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

    # ====================
    # LOOKUP
    # ====================

    def lookup(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        _track_metrics: bool = True,
    ) -> LookupResult:
        """
        Search cache by semantic similarity with exact context matching.

        Args:
            query: Query text to embed and search
            context: Context dict for exact matching (agent_id, tools, etc)
            similarity_threshold: Minimum similarity score
            _track_metrics: Internal flag to control metrics tracking

        Returns:
            LookupResult with hit status and data
        """
        start_time = time.time()
        context = context or {}
        threshold = similarity_threshold or self.config.similarity_threshold

        try:
            # Empty cache check
            if self.storage.count() == 0:
                return self._miss(
                    "cache_empty", start_time, track_metrics=_track_metrics
                )

            # Embed query
            query_embedding = self.embedder.embed(query)

            # Hybrid search: exact context + semantic similarity
            candidates = self.storage.search(
                embedding=query_embedding,
                context=context,
                limit=50,
                similarity_threshold=threshold,
            )

            if not candidates:
                return self._miss("no_match", start_time, track_metrics=_track_metrics)

            # Best match (already sorted by similarity)
            best = candidates[0]

            # TTL check
            if self._is_expired(best):
                return self._miss("expired", start_time, track_metrics=_track_metrics)

            # EVICTION: Track access on HIT
            entry_id = self._generate_entry_id(best.query_text, best.context)
            self.eviction.on_access(entry_id)

            # HIT
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "cache_hit",
                similarity=round(best.similarity, 3),
                query_preview=query[:50],
                matched_query_preview=best.query_text[:50],
                age_seconds=round(best.age_seconds, 1),
                latency_ms=round(elapsed_ms, 1),
            )

            if self.metrics and _track_metrics:
                self.metrics.hits += 1
                self.metrics.total_latency_saved_ms += 2000  # Estimated LLM call time
                self.metrics.record_lookup_latency(elapsed_ms)

                # Export to OpenTelemetry every 100 requests
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

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                "cache_lookup_error",
                error_type=type(e).__name__,
                error_message=str(e),
                query_preview=query[:50],
                latency_ms=round(elapsed_ms, 1),
                exc_info=True,
            )
            if self.metrics and _track_metrics:
                self.metrics.misses += 1
                self.metrics.record_lookup_latency(elapsed_ms)
                self.metrics.lookup_errors += 1
            return LookupResult(hit=False)

    # ====================
    # STORE
    # ====================

    def store(
        self,
        query: str,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store result in cache with context.

        Args:
            query: Query text
            context: Context dict (will be matched exactly in lookups)
            result: Result to cache (any JSON-serializable or tabular data)
            metadata: Optional metadata
        """
        try:
            # Evict if needed
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

            # Create cache entry
            embedding = self.embedder.embed(query)
            timestamp = time.time()

            entry = CacheEntry(
                query_text=query,
                context=context,
                embedding=embedding,
                result=result,
                timestamp=timestamp,
                metadata=metadata,
            )

            # Store (serialization happens in storage backend)
            self.storage.add([entry])

            # EVICTION: Track insertion
            entry_id = self._generate_entry_id(query, context)
            self.eviction.on_insert(entry_id)

            # Auto-index
            if self.config.auto_create_index:
                self.storage.maybe_auto_create_index(
                    self.config.index_threshold_entries,
                    self.config.index_num_partitions,
                )

            # Track result size in metrics
            if self.metrics:
                try:
                    result_str = json.dumps(result)
                    result_size = len(result_str.encode("utf-8"))
                    self.metrics.record_result_size(result_size)
                except Exception:
                    pass  # Skip size tracking if serialization fails

            logger.debug(
                "cache_store_success",
                query_preview=query[:50],
                context_keys=list(context.keys()),
                cache_entries=self.storage.count(),
            )

        except Exception as e:
            logger.error(
                "cache_store_error",
                error_type=type(e).__name__,
                error_message=str(e),
                query_preview=query[:50],
                context_preview=str(context)[:100],
                exc_info=True,
            )
            if self.metrics:
                self.metrics.store_errors += 1

    # ====================
    # MAINTENANCE
    # ====================

    def cleanup_expired(self) -> int:
        """Remove expired entries based on TTL."""
        if self.config.ttl_seconds is None:
            logger.warning("no_ttl_configured")
            return 0

        try:
            import pyarrow.compute as pc

            arrow_table = self.storage.to_arrow()

            if len(arrow_table) == 0:
                return 0

            cutoff = time.time() - self.config.ttl_seconds
            before = len(arrow_table)

            # Get expired entries to remove from eviction policy
            expired_mask = pc.less_equal(arrow_table["timestamp"], cutoff)
            expired_rows = arrow_table.filter(expired_mask).to_pylist()

            # Remove from eviction policy
            for row in expired_rows:
                entry_id = self._generate_entry_id(
                    row.get("query_text", ""), row.get("context", "{}")
                )
                try:
                    self.eviction.on_evict(entry_id)
                except Exception:
                    pass  # Entry might not exist in eviction policy

            # Filter expired entries
            mask = pc.greater(arrow_table["timestamp"], cutoff)

            if self.config.db_uri == "memory://":
                filtered = arrow_table.filter(mask)
                self.storage.table = self.storage.db.create_table(
                    self.config.table_name,
                    data=filtered if len(filtered) > 0 else None,
                    schema=self.storage.schema if len(filtered) == 0 else None,
                    mode="overwrite",
                )
            else:
                self.storage.delete_by_filter(f"timestamp <= {cutoff}")

            deleted = before - self.storage.count()
            logger.info("cleaned_up_expired", deleted=deleted)
            return deleted

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
            query: Query text (semantic matching not implemented)
            context: Context dict for exact matching
            older_than_seconds: Invalidate entries older than this

        Returns:
            Number of deleted entries
        """
        if query is None and context is None and older_than_seconds is None:
            logger.warning("invalidate_called_without_criteria")
            return 0

        try:
            import pyarrow.compute as pc

            before = self.storage.count()
            arrow_table = self.storage.to_arrow()

            # Collect entries to remove from eviction policy
            entries_to_remove = []

            # By age
            if older_than_seconds is not None:
                cutoff = time.time() - older_than_seconds
                old_mask = pc.less_equal(arrow_table["timestamp"], cutoff)
                old_rows = arrow_table.filter(old_mask).to_pylist()
                entries_to_remove.extend(old_rows)

                if self.config.db_uri == "memory://":
                    mask = pc.greater(arrow_table["timestamp"], cutoff)
                    filtered = arrow_table.filter(mask)
                    self.storage.table = self.storage.db.create_table(
                        self.config.table_name,
                        data=filtered if len(filtered) > 0 else None,
                        schema=self.storage.schema if len(filtered) == 0 else None,
                        mode="overwrite",
                    )
                else:
                    self.storage.delete_by_filter(f"timestamp <= {cutoff}")

            # By context
            elif context is not None:
                context_json = json.dumps(context, sort_keys=True)
                context_mask = pc.equal(arrow_table["context"], context_json)
                context_rows = arrow_table.filter(context_mask).to_pylist()
                entries_to_remove.extend(context_rows)

                if self.config.db_uri == "memory://":
                    mask = pc.not_equal(arrow_table["context"], context_json)
                    filtered = arrow_table.filter(mask)
                    self.storage.table = self.storage.db.create_table(
                        self.config.table_name,
                        data=filtered if len(filtered) > 0 else None,
                        schema=self.storage.schema if len(filtered) == 0 else None,
                        mode="overwrite",
                    )
                else:
                    self.storage.delete_by_filter(f"context = '{context_json}'")

            # By query (not implemented)
            elif query is not None:
                logger.warning("semantic_invalidation_not_implemented")
                return 0

            # Remove from eviction policy
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
        """
        Get current metrics report.

        Returns:
            Dict with comprehensive metrics or None if metrics disabled
        """
        if self.metrics:
            return self.metrics.report()
        return None

    def export_metrics_now(self):
        """Force immediate export of metrics to OpenTelemetry."""
        self._export_metrics_to_otel()

    # ====================
    # INTERNAL HELPERS
    # ====================

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        if self.config.ttl_seconds is None:
            return False
        return entry.age_seconds > self.config.ttl_seconds

    def _evict_one(self):
        """Evict one entry using configured eviction policy."""
        try:
            # Select victim using eviction policy
            victim_id = self.eviction.select_victim()

            # Parse victim_id to get query and context
            # Format: "query[:30]:context[:30]"
            arrow_table = self.storage.to_arrow()
            rows = arrow_table.to_pylist()

            # Find matching entry
            victim_row = None
            for row in rows:
                entry_id = self._generate_entry_id(
                    row.get("query_text", ""), row.get("context", "{}")
                )
                if entry_id == victim_id:
                    victim_row = row
                    break

            if victim_row is None:
                # Fallback: evict oldest
                logger.warning(
                    "victim_not_found_fallback_to_oldest", victim_id=victim_id
                )
                import pyarrow.compute as pc

                oldest_ts = pc.min(arrow_table["timestamp"]).as_py()

                if self.config.db_uri == "memory://":
                    mask = pc.not_equal(arrow_table["timestamp"], oldest_ts)
                    filtered = arrow_table.filter(mask)
                    self.storage.table = self.storage.db.create_table(
                        self.config.table_name,
                        data=filtered if len(filtered) > 0 else None,
                        schema=self.storage.schema if len(filtered) == 0 else None,
                        mode="overwrite",
                    )
                else:
                    self.storage.delete_by_filter(f"timestamp = {oldest_ts}")
            else:
                # Delete the victim
                victim_ts = victim_row.get("timestamp")

                if self.config.db_uri == "memory://":
                    import pyarrow.compute as pc

                    mask = pc.not_equal(arrow_table["timestamp"], victim_ts)
                    filtered = arrow_table.filter(mask)
                    self.storage.table = self.storage.db.create_table(
                        self.config.table_name,
                        data=filtered if len(filtered) > 0 else None,
                        schema=self.storage.schema if len(filtered) == 0 else None,
                        mode="overwrite",
                    )
                else:
                    self.storage.delete_by_filter(f"timestamp = {victim_ts}")

            # Notify eviction policy
            self.eviction.on_evict(victim_id)

            logger.info(
                "entry_evicted",
                policy=self.config.eviction_policy,
                victim_id=victim_id[:50],
            )

        except ValueError as e:
            # No entries to evict
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
