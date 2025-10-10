"""Maintenance, stats, and I/O operations."""

import time
import json
from typing import Dict, Any, List

from ..types import CacheEntry
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MaintenanceOperations:
    """Handles cache maintenance, statistics, and import/export operations."""

    def __init__(self, storage, eviction, config, metrics):
        """Initialize maintenance operations.

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

    def cleanup_expired(self) -> int:
        """Remove expired entries based on TTL.

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

            # Clean up exact table
            if len(exact_table) > 0:
                before = len(exact_table)
                expired_mask = pc.less_equal(exact_table["timestamp"], cutoff)
                expired_rows = exact_table.filter(expired_mask).to_pylist()

                # Notify eviction policy
                for row in expired_rows:
                    entry_id = self._generate_entry_id(
                        row.get("query_text", ""), row.get("context", "{}")
                    )
                    try:
                        self.eviction.on_evict(entry_id)
                    except Exception as e:
                        logger.debug(
                            "eviction_state_cleanup_failed_during_ttl_cleanup",
                            entry_id=entry_id,
                            error=str(e),
                        )

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

            # Clean up semantic table
            if len(semantic_table) > 0:
                before = len(semantic_table)
                expired_mask = pc.less_equal(semantic_table["timestamp"], cutoff)
                expired_rows = semantic_table.filter(expired_mask).to_pylist()

                # Notify eviction policy
                for row in expired_rows:
                    entry_id = self._generate_entry_id(
                        row.get("query_text", ""), row.get("context", "{}")
                    )
                    try:
                        self.eviction.on_evict(entry_id)
                    except Exception as e:
                        logger.debug(
                            "eviction_state_cleanup_failed_during_age_cleanup",
                            entry_id=entry_id,
                            error=str(e),
                        )

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

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

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

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all cache entries as list of dicts.

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
        """Export cache to file.

        Args:
            filepath: Output file path
            format: File format (parquet, json, csv)

        Raises:
            ValueError: If format is unsupported
            Exception: If export operation fails
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
        """Import cache from file.

        Args:
            filepath: Input file path
            format: File format (parquet, json, csv)

        Raises:
            ValueError: If format is unsupported
            Exception: If import operation fails
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
