"""Invalidation and deletion operations."""

import time
import json
from typing import Optional, Dict, Any

from ..types import BulkInvalidatePattern
from ..utils.logging import get_logger

logger = get_logger(__name__)


class InvalidationOperations:
    """Handles cache invalidation and deletion operations."""

    def __init__(self, storage, eviction, config, metrics):
        """Initialize invalidation operations.

        Args:
            storage: Storage backend instance
            eviction: Eviction policy instance
            config: Configuration object
            metrics: Optional metrics tracker
        """
        self.storage = storage
        self.eviction = eviction
        self.config = config
        self.metrics = metrics

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

    def invalidate(
        self,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """Invalidate cache entries by criteria.

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

            # Invalidate by age
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

            # Invalidate by context
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

            # Query-based invalidation not implemented
            elif query is not None:
                logger.warning("semantic_invalidation_not_implemented")
                return 0

            # Notify eviction policy
            for row in entries_to_remove:
                entry_id = self._generate_entry_id(
                    row.get("query_text", ""), row.get("context", "{}")
                )
                try:
                    self.eviction.on_evict(entry_id)
                except Exception as e:
                    logger.debug(
                        "eviction_state_cleanup_failed_during_invalidation",
                        entry_id=entry_id,
                        error=str(e),
                    )

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
        """Bulk invalidate entries matching pattern using efficient batch deletion.

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
        """Invalidate all entries with query starting with prefix.

        This is a convenience method wrapping invalidate_bulk.

        Args:
            query_prefix: Prefix to match (e.g., "SELECT")

        Returns:
            Number of entries invalidated
        """
        pattern = BulkInvalidatePattern(query_prefix=query_prefix)
        return self.invalidate_bulk(pattern)

    def invalidate_by_regex(self, query_regex: str) -> int:
        """Invalidate all entries matching regex pattern.

        Args:
            query_regex: Regular expression pattern

        Returns:
            Number of entries invalidated
        """
        pattern = BulkInvalidatePattern(query_regex=query_regex)
        return self.invalidate_bulk(pattern)

    def invalidate_by_context(self, context_matches: Dict[str, str]) -> int:
        """Invalidate entries matching context pattern.

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
        """Invalidate all entries older than specified seconds.

        Args:
            seconds: Age threshold in seconds

        Returns:
            Number of entries invalidated
        """
        pattern = BulkInvalidatePattern(older_than_seconds=seconds)
        return self.invalidate_bulk(pattern)

    def clear_all(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        clear_start = time.time()
        logger.warning("clearing_all_cache")

        try:
            before = self.storage.count()
            self.storage.exact_table.delete("timestamp > 0")
            self.storage.semantic_table.delete("timestamp > 0")

            # Clear eviction policy state
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
