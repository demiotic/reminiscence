"""Deletion operations for LanceDB backend."""

from __future__ import annotations

import time
from typing import Callable

import pyarrow.compute as pc

from ...utils.logging import get_logger

logger = get_logger(__name__)


class DeletionMixin:
    """Mixin providing deletion operations for LanceDB backend."""

    def delete_by_id(self, entry_id: str) -> bool:
        """Delete a single entry by its unique ID.

        This method provides efficient single-entry deletion by ID across both
        exact and semantic tables. It handles both memory and persistent storage
        modes appropriately.

        Args:
            entry_id: SHA256 hash ID of the entry to delete.

        Returns:
            True if entry was found and deleted, False otherwise.

        Performance:
            - Memory mode: O(n) table scan with filter
            - Persistent mode: O(1) indexed delete by primary key
        """
        delete_start = time.perf_counter()
        logger.debug("delete_by_id_start", entry_id=entry_id[:16])

        deleted = False

        if self.config.db_uri == "memory://":
            # Memory mode: filter out the target entry and recreate tables
            for table, table_name, schema in [
                (self.exact_table, self._exact_table_name, self.exact_schema),
                (self.semantic_table, self._semantic_table_name, self.semantic_schema),
            ]:
                arrow_table = table.to_arrow()
                if len(arrow_table) == 0:
                    continue

                # Create mask: keep all entries except the one to delete
                mask = pc.not_equal(arrow_table["id"], entry_id)
                filtered = arrow_table.filter(mask)

                # Check if anything was actually deleted
                if len(filtered) < len(arrow_table):
                    deleted = True
                    new_table = self.db.create_table(
                        table_name,
                        data=filtered if len(filtered) > 0 else None,
                        schema=schema if len(filtered) == 0 else None,
                        mode="overwrite",
                    )

                    # Update table references
                    if table_name == self._exact_table_name:
                        self.exact_table = new_table
                    else:
                        self.semantic_table = new_table
                        self.table = new_table

        else:
            # Persistent mode: use SQL-based deletion for efficiency
            filter_expr = f"id = '{entry_id}'"

            # Try deleting from exact table
            try:
                before_exact = self.exact_table.count_rows()
                self.exact_table.delete(filter_expr)
                after_exact = self.exact_table.count_rows()

                if before_exact > after_exact:
                    deleted = True
                    logger.debug("deleted_from_exact_table", entry_id=entry_id[:16])
            except Exception as e:
                logger.debug("exact_table_delete_skipped", error=str(e))

            # Try deleting from semantic table
            try:
                before_semantic = self.semantic_table.count_rows()
                self.semantic_table.delete(filter_expr)
                after_semantic = self.semantic_table.count_rows()

                if before_semantic > after_semantic:
                    deleted = True
                    logger.debug("deleted_from_semantic_table", entry_id=entry_id[:16])
            except Exception as e:
                logger.debug("semantic_table_delete_skipped", error=str(e))

        delete_ms = (time.perf_counter() - delete_start) * 1000

        logger.debug(
            "delete_by_id_complete",
            entry_id=entry_id[:16],
            deleted=deleted,
            latency_ms=round(delete_ms, 1),
        )

        return deleted

    def delete_by_filter(self, filter_expr: str) -> None:
        """Delete entries matching filter from both tables.

        Args:
            filter_expr: SQL-like filter expression.

        Raises:
            NotImplementedError: If called on memory:// storage (use delete_by_condition).
        """
        delete_start = time.perf_counter()
        logger.debug("delete_by_filter_start", filter_expr=filter_expr)

        if self.config.db_uri == "memory://":
            raise NotImplementedError("Use delete_by_condition for memory://")
        else:
            deleted_count = 0
            try:
                before = self.exact_table.count_rows()
                self.exact_table.delete(filter_expr)
                after = self.exact_table.count_rows()
                deleted_count += before - after
                logger.debug("exact_table_deleted", count=before - after)
            except Exception as e:
                logger.debug("exact_table_delete_skipped", error=str(e))

            try:
                before = self.semantic_table.count_rows()
                self.semantic_table.delete(filter_expr)
                after = self.semantic_table.count_rows()
                deleted_count += before - after
                logger.debug("semantic_table_deleted", count=before - after)
            except Exception as e:
                logger.debug("semantic_table_delete_skipped", error=str(e))

            try:
                self.exact_table.compact_files()
                self.semantic_table.compact_files()
                logger.debug("tables_compacted")
            except AttributeError:
                pass

            delete_ms = (time.perf_counter() - delete_start) * 1000
            logger.info(
                "delete_by_filter_complete",
                deleted=deleted_count,
                latency_ms=round(delete_ms, 1),
            )

    def delete_by_condition(self, condition_func: Callable) -> None:
        """Delete by custom condition (for memory mode).

        Args:
            condition_func: Function that takes an Arrow table and returns a boolean mask.

        Raises:
            NotImplementedError: If called on persistent storage (use delete_by_filter).
        """
        delete_start = time.perf_counter()
        logger.debug("delete_by_condition_start")

        if self.config.db_uri == "memory://":
            exact_arrow = self.exact_table.to_arrow()
            mask_exact = condition_func(exact_arrow)
            filtered_exact = exact_arrow.filter(mask_exact)

            self.exact_table = self.db.create_table(
                self._exact_table_name,
                data=filtered_exact if len(filtered_exact) > 0 else None,
                schema=self.exact_schema if len(filtered_exact) == 0 else None,
                mode="overwrite",
            )

            semantic_arrow = self.semantic_table.to_arrow()
            mask_semantic = condition_func(semantic_arrow)
            filtered_semantic = semantic_arrow.filter(mask_semantic)

            self.semantic_table = self.db.create_table(
                self._semantic_table_name,
                data=filtered_semantic if len(filtered_semantic) > 0 else None,
                schema=self.semantic_schema if len(filtered_semantic) == 0 else None,
                mode="overwrite",
            )
            self.table = self.semantic_table

            delete_ms = (time.perf_counter() - delete_start) * 1000
            logger.info("delete_by_condition_complete", latency_ms=round(delete_ms, 1))
        else:
            raise NotImplementedError("Use delete_by_filter for persistent storage")

    def clear(self) -> None:
        """Clear all entries from both tables."""
        clear_start = time.perf_counter()
        before = self.count()

        logger.debug("clear_start", entries=before)

        self.exact_table = self.db.create_table(
            self._exact_table_name,
            schema=self.exact_schema,
            mode="overwrite",
        )
        self.semantic_table = self.db.create_table(
            self._semantic_table_name,
            schema=self.semantic_schema,
            mode="overwrite",
        )
        self.table = self.semantic_table
        self._index_created = False

        clear_ms = (time.perf_counter() - clear_start) * 1000
        logger.info("storage_cleared", deleted=before, latency_ms=round(clear_ms, 1))
