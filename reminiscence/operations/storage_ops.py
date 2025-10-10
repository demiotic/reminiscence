"""Storage and batch operations."""

import time
import json
from typing import Optional, Dict, Any, List

from ..types import CacheEntry
from ..utils.logging import get_logger
from ..utils.query_detection import should_use_exact_mode
from ..utils.fingerprint import create_fingerprint

logger = get_logger(__name__)


class StorageOperations:
    """Handles cache storage and batch operations."""

    def __init__(self, storage, embedder, eviction, config, metrics):
        """Initialize storage operations.

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

    def _is_error_result(self, result: Any) -> bool:
        """Check if result represents an error that shouldn't be cached.

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
        """Store result in cache with context.

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
            # Skip error results if not allowed
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

            # Check if eviction needed
            current_count = self.storage.count()
            if self.config.max_entries and current_count >= self.config.max_entries:
                logger.debug(
                    "cache_eviction_triggered",
                    reason="max_entries_reached",
                    current_count=current_count,
                    max_entries=self.config.max_entries,
                )
                self._evict_one()

            # Determine query mode
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

            # Generate embedding for semantic mode
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

            # Create entry
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

            # Store entry
            storage_start = time.time()
            self.storage.add([entry])
            storage_ms = (time.time() - storage_start) * 1000
            logger.debug("store_entry_added", storage_ms=round(storage_ms, 1))

            # Update eviction policy
            entry_id = self._generate_entry_id(query, context)
            self.eviction.on_insert(entry_id)

            # Auto-create index if enabled
            if self.config.auto_create_index:
                index_start = time.time()
                self.storage.maybe_auto_create_index(
                    self.config.index_threshold_entries,
                    self.config.index_num_partitions,
                )
                index_ms = (time.time() - index_start) * 1000
                if index_ms > 1.0:
                    logger.debug("store_index_check", latency_ms=round(index_ms, 1))

            # Track result size
            if self.metrics:
                try:
                    result_str = json.dumps(result)
                    result_size = len(result_str.encode("utf-8"))
                    self.metrics.record_result_size(result_size)
                except Exception as e:
                    logger.debug("result_size_measurement_failed", error=str(e))

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
        """Store multiple results in batch optimized for embeddings.

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

        # Filter out errors
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

        # Extract valid entries
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

        # Determine modes
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

        # Generate embeddings in batch
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

        # Create entries
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

        # Store all entries
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

    def _evict_one(self):
        """Evict one entry to make space using the configured eviction policy.

        This method uses the eviction policy to select a victim entry,
        then efficiently deletes it by reconstructing the full entry ID from storage.

        Raises:
            Exception: If eviction fails, logs error but doesn't raise to allow
                      cache operations to continue
        """
        evict_start = time.time()
        try:
            victim_id = self.eviction.select_victim()

            # Parse victim ID format: "query_preview:context_json"
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

            # Generate context hash for search
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
                logger.warning("eviction_entry_not_found", victim_id=victim_id)

        except Exception as e:
            logger.error(
                "eviction_failed",
                error=str(e),
                policy=self.config.eviction_policy,
                exc_info=True,
            )
