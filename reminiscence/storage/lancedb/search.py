"""Search operations for LanceDB backend."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pyarrow.compute as pc

from ...types import CacheEntry
from ...utils.fingerprint import compute_query_hash, create_fingerprint
from ...utils.logging import get_logger

logger = get_logger(__name__)


class SearchMixin:
    """Mixin providing search operations for LanceDB backend."""

    def search(
        self,
        embedding: Optional[List[float]],
        context: Dict[str, Any],
        limit: int = 50,
        similarity_threshold: float = 0.85,
        query_mode: str = "semantic",
        query_text: Optional[str] = None,
    ) -> List[CacheEntry]:
        """Search with mode-based routing.

        Args:
            embedding: Query embedding vector (None for exact mode).
            context: Context dict for matching.
            limit: Maximum results to return.
            similarity_threshold: Minimum similarity score for semantic search.
            query_mode: "exact" or "semantic".
            query_text: Original query text (required for exact mode).

        Returns:
            List of matching cache entries.
        """
        start = time.perf_counter()

        logger.debug(
            "search_start",
            query_mode=query_mode,
            limit=limit,
            similarity_threshold=similarity_threshold,
            has_embedding=embedding is not None,
            has_query_text=query_text is not None,
        )

        if query_mode == "exact":
            results = self._search_exact(query_text, context)
        elif query_mode == "semantic":
            results = self._search_semantic(
                embedding, context, limit, similarity_threshold
            )
        else:
            logger.warning(
                "unexpected_query_mode",
                mode=query_mode,
                expected=["exact", "semantic"],
                fallback="semantic",
                note="Mode should be resolved upstream in lookup/store",
            )
            results = self._search_semantic(
                embedding, context, limit, similarity_threshold
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if self.metrics:
            if not hasattr(self.metrics, "storage_searches"):
                self.metrics.storage_searches = 0
            self.metrics.storage_searches += 1

            if not hasattr(self.metrics, "storage_search_latencies_ms"):
                from collections import deque

                self.metrics.storage_search_latencies_ms = deque(maxlen=1000)
            self.metrics.storage_search_latencies_ms.append(elapsed_ms)

        logger.debug(
            "search_complete",
            mode=query_mode,
            results_count=len(results),
            latency_ms=round(elapsed_ms, 2),
        )

        return results

    def _search_exact(
        self, query_text: str, context: Dict[str, Any]
    ) -> List[CacheEntry]:
        """Exact match using hash lookup.

        Args:
            query_text: Query text for exact matching.
            context: Context dict.

        Returns:
            List with single matching entry or empty list.
        """
        if not query_text:
            logger.debug("exact_search_skipped_no_query")
            return []

        search_start = time.perf_counter()
        query_hash = compute_query_hash(query_text, context)
        context_hash = create_fingerprint(context)

        # Enhanced debugging: Log table state BEFORE search
        table_rows = self.exact_table.count_rows()
        logger.info(
            "exact_search_debug",
            query_hash=query_hash,  # Full hash for debugging
            context_hash=context_hash,  # Full hash for debugging
            table_id=id(self.exact_table),
            table_rows=table_rows,
            query_text=query_text,
            context=context,
        )

        logger.debug(
            "exact_search_start",
            query_hash=query_hash[:16],
            context_hash=context_hash[:16],
        )

        try:
            where_clause = (
                f"query_hash = '{query_hash}' AND context_hash = '{context_hash}'"
            )
            logger.info("exact_search_where", where_clause=where_clause)

            results = self.exact_table.search().where(where_clause).limit(1).to_arrow()

            search_ms = (time.perf_counter() - search_start) * 1000

            # Enhanced debugging: Log result details
            logger.info(
                "exact_search_results",
                result_count=len(results),
                result_schema=str(results.schema) if len(results) > 0 else "empty",
            )

            if len(results) == 0:
                logger.debug("exact_search_miss", latency_ms=round(search_ms, 1))
                return []

            entry = self._arrow_row_to_cache_entry(results, 0, similarity=1.0)
            logger.debug("exact_search_hit", latency_ms=round(search_ms, 1))
            return [entry] if entry else []

        except Exception as e:
            search_ms = (time.perf_counter() - search_start) * 1000
            logger.error(
                "exact_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(search_ms, 1),
                exc_info=True,  # Add stack trace
            )
            if self.metrics:
                if not hasattr(self.metrics, "storage_search_errors"):
                    self.metrics.storage_search_errors = 0
                self.metrics.storage_search_errors += 1
            return []

    def _search_semantic(
        self,
        embedding: List[float],
        context: Dict[str, Any],
        limit: int,
        similarity_threshold: float,
    ) -> List[CacheEntry]:
        """Semantic search with vectorized filtering.

        Args:
            embedding: Query embedding vector.
            context: Context dict.
            limit: Maximum results.
            similarity_threshold: Minimum similarity score.

        Returns:
            List of matching entries sorted by similarity.
        """
        search_start = time.perf_counter()
        context_hash = (
            create_fingerprint(context) if context else create_fingerprint({})
        )
        where_clause = f"context_hash = '{context_hash}'"

        logger.debug(
            "semantic_search_start",
            context_hash=context_hash[:16],
            limit=limit,
            threshold=similarity_threshold,
        )

        try:
            query = self.semantic_table.search(embedding).metric("cosine").limit(limit)
            query = query.where(where_clause)
            results = query.to_arrow()

            if len(results) == 0:
                search_ms = (time.perf_counter() - search_start) * 1000
                logger.debug(
                    "semantic_search_no_candidates", latency_ms=round(search_ms, 1)
                )
                return []

            distances = results["_distance"]
            similarities = pc.subtract(1.0, distances)

            mask = pc.greater_equal(similarities, similarity_threshold)
            filtered_results = results.filter(mask)

            if len(filtered_results) == 0:
                search_ms = (time.perf_counter() - search_start) * 1000
                logger.debug(
                    "semantic_search_filtered_out",
                    candidates=len(results),
                    latency_ms=round(search_ms, 1),
                )
                return []

            entries = []
            filtered_distances = filtered_results["_distance"]
            for i in range(len(filtered_results)):
                similarity = 1.0 - filtered_distances[i].as_py()
                entry = self._arrow_row_to_cache_entry(filtered_results, i, similarity)
                if entry:
                    entries.append(entry)

            entries.sort(key=lambda x: x.similarity or 0, reverse=True)

            search_ms = (time.perf_counter() - search_start) * 1000
            logger.debug(
                "semantic_search_success",
                candidates=len(results),
                filtered=len(entries),
                top_similarity=round(entries[0].similarity, 3) if entries else 0,
                latency_ms=round(search_ms, 1),
            )
            return entries

        except Exception as e:
            search_ms = (time.perf_counter() - search_start) * 1000
            logger.error(
                "semantic_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(search_ms, 1),
                exc_info=True,
            )
            if self.metrics:
                if not hasattr(self.metrics, "storage_search_errors"):
                    self.metrics.storage_search_errors = 0
                self.metrics.storage_search_errors += 1
            return []
