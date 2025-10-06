"""Memora core - Queryable semantic cache."""

import time
from typing import Any, Dict, List, Optional

import lancedb
import pyarrow as pa
import pyarrow.compute as pc
from sentence_transformers import SentenceTransformer

from .config import CacheConfig
from .metrics import CacheMetrics
from .types import LookupResult, AvailabilityCheck
from .utils import create_fingerprint, cosine_similarity, serialize, deserialize

from .utils.logging import configure_logging, get_logger


class Memora:
    """
    Semantic cache for multi-agent systems.

    Design: Storage and query only. Does NOT execute logic.

    Main API:
    - lookup(): Search for existing result
    - store(): Save new result
    - check_availability(): Verify availability without retrieving data
    - invalidate(): Mark entries as invalid
    - create_index(): Create vector index for fast searches

    Example:
        >>> memora = Memora(CacheConfig.for_development())
        >>>
        >>> # Query
        >>> result = memora.lookup(
        ...     query="Analyze Q3 sales",
        ...     context={"agent": "sql", "db": "prod"}
        ... )
        >>>
        >>> if result.is_hit:
        ...     print(result.result)
        ... else:
        ...     # Execute agent externally
        ...     data = execute_agent(...)
        ...     memora.store(query, context, data)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize Memora."""
        self.config = config or CacheConfig.load()
        # Setup structured logging
        configure_logging(
            log_level=self.config.log_level, json_logs=self.config.json_logs
        )

        global logger
        logger = get_logger(__name__)

        logger.info(
            "initializing_memora",
            model=self.config.model_name,
            db_uri=self.config.db_uri,
        )

        # Initialize components
        self.model = SentenceTransformer(
            self.config.model_name,
            backend="onnx",
        )

        # Log which ONNX model file was loaded
        try:
            if hasattr(self.model, "_backend") and hasattr(
                self.model._backend, "_model_path"
            ):
                model_path = self.model._backend._model_path
                logger.info(
                    "onnx_model_loaded",
                    model_path=str(model_path),
                    is_quantized="qint8" in str(model_path)
                    or "quint8" in str(model_path),
                )
            else:
                logger.debug(
                    "onnx_model_info", message="Cannot access backend model path"
                )
        except Exception as e:
            logger.debug("onnx_model_info", error=str(e))

        self.db = lancedb.connect(self.config.db_uri)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.metrics = CacheMetrics() if self.config.enable_metrics else None

        # Schema - timestamp in milliseconds, result and metadata as binary
        self.schema = pa.schema(
            [
                pa.field("query_text", pa.string()),
                pa.field("context_hash", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("result", pa.binary()),
                pa.field("timestamp", pa.int64()),
                pa.field("metadata", pa.binary()),
            ]
        )

        self.table = self._init_table()
        self._index_created = False

        logger.info(
            f"Memora ready | dim={self.embedding_dim}, "
            f"threshold={self.config.similarity_threshold}, "
            f"entries={self.table.count_rows()}, "
            f"max_entries={self.config.max_entries}"
        )

        # Auto-create index if configured
        if self.config.auto_create_index:
            self._maybe_auto_create_index()

    def _init_table(self):
        """Initialize or open LanceDB table."""
        try:
            table = self.db.open_table(self.config.table_name)
            logger.debug(f"Table '{self.config.table_name}' opened")
            return table
        except Exception:
            table = self.db.create_table(
                self.config.table_name, schema=self.schema, mode="overwrite"
            )
            logger.debug(f"Table '{self.config.table_name}' created")
            return table

    def _maybe_auto_create_index(self):
        """Create index automatically if threshold is reached."""
        if self._index_created:
            return

        row_count = self.table.count_rows()
        if row_count >= self.config.index_threshold_entries:
            logger.info(
                f"Auto-creating index: {row_count} >= {self.config.index_threshold_entries}"
            )
            self.create_index(num_partitions=self.config.index_num_partitions)
            self._index_created = True

    def _embed(self, text: str) -> List[float]:
        """Generate L2-normalized embedding."""
        try:
            embedding_np = self.model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return embedding_np.tolist()
        except Exception as e:
            logger.error(
                f"Embedding generation failed: {e} | text='{text[:50]}...'",
                exc_info=True,
            )
            # Re-raise because we can't do anything without embedding
            # Will be caught by lookup() or store()
            raise

    def _current_timestamp_ms(self) -> int:
        """Return current timestamp in milliseconds."""
        return int(time.time() * 1000)

    def _is_expired(self, timestamp_ms: int) -> bool:
        """Check expiration according to TTL."""
        if self.config.ttl_seconds is None:
            return False
        age_ms = self._current_timestamp_ms() - timestamp_ms
        age_seconds = age_ms / 1000
        return age_seconds > self.config.ttl_seconds

    def _evict_oldest(self) -> int:
        """
        Evict oldest entry using FIFO policy.

        Returns:
            Number of entries evicted (0 or 1)
        """
        try:
            arrow_table = self.table.to_arrow()
            if len(arrow_table) == 0:
                return 0

            # Find oldest timestamp using pyarrow.compute
            oldest_ts = pc.min(arrow_table["timestamp"]).as_py()

            # Delete oldest entry
            if self.config.db_uri == "memory://":
                mask = pc.not_equal(arrow_table["timestamp"], oldest_ts)
                filtered = arrow_table.filter(mask)
                self.table = self.db.create_table(
                    self.config.table_name,
                    data=filtered if len(filtered) > 0 else None,
                    schema=self.schema if len(filtered) == 0 else None,
                    mode="overwrite",
                )
            else:
                self.table.delete(f"timestamp = {oldest_ts}")
                try:
                    self.table.compact_files()
                except AttributeError:
                    pass

            logger.info(f"Evicted oldest entry (ts={oldest_ts}) via FIFO policy")
            return 1

        except Exception as e:
            logger.error(f"Eviction failed: {e}", exc_info=True)
            return 0

    def cached(
        self,
        context: Optional[Dict[str, Any]] = None,
        query_param: str = "query",
        extract_from_args: bool = True,
        exclude_from_context: Optional[list] = None,
    ):
        """
        Decorator to cache function results using this Memora instance.

        Args:
            context: Static context dict (optional)
            query_param: Name of query parameter (default: "query")
            extract_from_args: Extract function params into context (default: True)
            exclude_from_context: List of params to exclude from context

        Returns:
            Decorator function

        Example:
            >>> memora = Memora()
            >>>
            >>> @memora.cached(context={"model": "gpt-4"})
            >>> def ask_llm(query: str, temperature: float = 0.7):
            >>>     return expensive_api_call(query, temperature)
        """
        from .decorators import create_cached_decorator

        decorator_factory = create_cached_decorator(self)
        return decorator_factory(
            context=context,
            query_param=query_param,
            extract_from_args=extract_from_args,
            exclude_from_context=exclude_from_context,
        )

    def lookup(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> LookupResult:
        """
        Search cache entry by semantic similarity.

        Args:
            query: User query
            context: Context (agent_id, tools, params, etc.)
            similarity_threshold: Override global threshold

        Returns:
            LookupResult with hit/miss and associated data
        """
        start_time = time.time()
        try:
            context = context or {}
            threshold = similarity_threshold or self.config.similarity_threshold

            # Empty cache
            if self.table.count_rows() == 0:
                logger.debug(
                    "cache_empty",
                    operation="lookup",
                    query_preview=query[:50],
                )
                if self.metrics:
                    self.metrics.misses += 1
                return LookupResult(hit=False)

            # Prepare search
            context_hash = create_fingerprint(context)
            query_embedding = self._embed(query)

            # Search candidates (without context filter in search, we'll do it after)
            search_results = (
                self.table.search(query_embedding)
                .limit(50)  # More candidates to filter later
                .to_arrow()
            )

            if len(search_results) == 0:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    "cache_miss",
                    reason="no_results",
                    query_preview=query[:50],
                    context_hash_preview=context_hash[:8],
                    latency_ms=round(elapsed_ms, 1),
                )
                if self.metrics:
                    self.metrics.misses += 1
                    self.metrics.record_lookup_latency(elapsed_ms)
                return LookupResult(hit=False)

            # Filter by context_hash manually (more reliable)
            mask_context = pc.equal(search_results["context_hash"], context_hash)
            search_results = search_results.filter(mask_context)

            if len(search_results) == 0:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    "cache_miss",
                    reason="context_mismatch",
                    query_preview=query[:50],
                    context_hash_preview=context_hash[:8],
                    latency_ms=round(elapsed_ms, 1),
                )
                if self.metrics:
                    self.metrics.misses += 1
                    self.metrics.record_lookup_latency(elapsed_ms)
                return LookupResult(hit=False)

            # Filter by TTL (timestamp in ms)
            if self.config.ttl_seconds is not None:
                cutoff_ms = self._current_timestamp_ms() - int(
                    self.config.ttl_seconds * 1000
                )
                mask_ttl = pc.greater(search_results["timestamp"], cutoff_ms)
                search_results = search_results.filter(mask_ttl)

                if len(search_results) == 0:
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(
                        "cache_miss",
                        reason="expired",
                        query_preview=query[:50],
                        ttl_seconds=self.config.ttl_seconds,
                        latency_ms=round(elapsed_ms, 1),
                    )
                    if self.metrics:
                        self.metrics.misses += 1
                        self.metrics.record_lookup_latency(elapsed_ms)
                    return LookupResult(hit=False)

            # First result is most similar
            best_idx = 0
            best_query = search_results["query_text"][best_idx].as_py()
            best_embedding = search_results["embedding"][best_idx].as_py()
            best_sim = cosine_similarity(query_embedding, best_embedding)

            # Evaluate threshold
            if best_sim < threshold:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    "cache_miss",
                    reason="low_similarity",
                    similarity=round(best_sim, 3),
                    threshold=threshold,
                    query_preview=query[:50],
                    context_hash_preview=context_hash[:8],
                    latency_ms=round(elapsed_ms, 1),
                )
                if self.metrics:
                    self.metrics.misses += 1
                    self.metrics.record_lookup_latency(elapsed_ms)
                return LookupResult(hit=False, similarity=best_sim)

            # HIT - Deserialize result
            result_bytes = search_results["result"][best_idx].as_py()
            result_data = deserialize(result_bytes)

            timestamp_ms = search_results["timestamp"][best_idx].as_py()
            age_ms = self._current_timestamp_ms() - timestamp_ms
            age_seconds = int(age_ms / 1000)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "cache_hit",
                similarity=round(best_sim, 3),
                query_preview=query[:50],
                matched_query_preview=best_query[:50],
                context_hash_preview=context_hash[:8],
                age_seconds=age_seconds,
                result_size_bytes=len(result_bytes),
                latency_ms=round(elapsed_ms, 1),
            )

            if self.metrics:
                self.metrics.hits += 1
                self.metrics.total_latency_saved_ms += 2000
                self.metrics.record_lookup_latency(elapsed_ms)

            return LookupResult(
                hit=True,
                result=result_data,
                similarity=best_sim,
                matched_query=best_query,
                age_seconds=age_seconds,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                "cache_lookup_error",
                error_type=type(e).__name__,
                error_message=str(e),
                query_preview=query[:50],
                context_hash_preview=create_fingerprint(context)[:8],
                latency_ms=round(elapsed_ms, 1),
                exc_info=True,
            )
            if self.metrics:
                self.metrics.misses += 1
                self.metrics.record_lookup_latency(elapsed_ms)
                self.metrics.lookup_errors += 1
            return LookupResult(hit=False)

    def store(
        self,
        query: str,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store result in cache.

        Args:
            query: User query
            context: Agent context
            result: Result to cache (str, dict, etc.)
            metadata: Additional metadata (optional)
        """
        try:
            # 1. Check max_entries BEFORE expensive operations (evict if needed)
            if self.config.max_entries is not None:
                current_count = self.table.count_rows()
                if current_count >= self.config.max_entries:
                    logger.debug(
                        "cache_eviction_triggered",
                        reason="max_entries_reached",
                        current_count=current_count,
                        max_entries=self.config.max_entries,
                        eviction_policy=self.config.eviction_policy,
                    )
                    self._evict_oldest()

            # 2. Generate embedding and fingerprint
            context_hash = create_fingerprint(context)
            embedding = self._embed(query)
            timestamp = self._current_timestamp_ms()

            # 3. Serialize and check size limits
            result_bytes = serialize(result)
            metadata_bytes = serialize(metadata) if metadata else b""
            total_payload_size = len(result_bytes) + len(metadata_bytes)

            # Check payload size
            if len(result_bytes) > self.config.max_result_size_bytes:
                logger.warning(
                    "cache_store_rejected",
                    reason="payload_too_large",
                    payload_size_bytes=len(result_bytes),
                    max_size_bytes=self.config.max_result_size_bytes,
                    query_preview=query[:50],
                    context_hash_preview=context_hash[:8],
                )
                if self.metrics:
                    self.metrics.store_errors += 1
                return  # Don't store oversized payloads

            # 4. Track size in metrics
            if self.metrics:
                self.metrics.record_result_size(len(result_bytes))

            # 5. Store in table
            data = [
                {
                    "query_text": query,
                    "context_hash": context_hash,
                    "embedding": embedding,
                    "result": result_bytes,
                    "timestamp": timestamp,
                    "metadata": metadata_bytes,
                }
            ]

            self.table.add(data)

            # 6. Get ACTUAL count after adding (FIX: don't use stale current_count)
            actual_count = self.table.count_rows()

            logger.debug(
                "cache_store_success",
                query_preview=query[:50],
                context_hash_preview=context_hash[:8],
                result_size_bytes=len(result_bytes),
                metadata_size_bytes=len(metadata_bytes),
                total_payload_bytes=total_payload_size,
                cache_entries=actual_count,
            )

            # 7. Auto-create index if needed
            if self.config.auto_create_index:
                self._maybe_auto_create_index()

        except Exception as e:
            # DO NOT propagate error - app must continue without cache
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

    def check_availability(
        self,
        query: str,
        context: Dict[str, Any],
        similarity_threshold: Optional[float] = None,
    ) -> AvailabilityCheck:
        """
        Verify availability without retrieving full data.

        Useful for planners that only need to know if cache exists.

        Args:
            query: Query to verify
            context: Context
            similarity_threshold: Override threshold

        Returns:
            AvailabilityCheck with minimal metadata
        """
        result = self.lookup(query, context, similarity_threshold)

        if not result.is_hit:
            return AvailabilityCheck(available=False)

        ttl_remaining = None
        if self.config.ttl_seconds and result.age_seconds is not None:
            ttl_remaining = self.config.ttl_seconds - result.age_seconds

        return AvailabilityCheck(
            available=True,
            age_seconds=result.age_seconds,
            ttl_remaining_seconds=ttl_remaining,
            similarity=result.similarity,
        )

    def invalidate(
        self,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            query: If specified, invalidate semantic matches
            context: If specified, invalidate by context_hash
            older_than_seconds: If specified, invalidate old entries (accepts decimals)

        Returns:
            Number of invalidated entries
        """
        if query is None and context is None and older_than_seconds is None:
            logger.warning("invalidate() called without criteria, ignoring")
            return 0

        before = self.table.count_rows()

        # Invalidate by age
        if older_than_seconds is not None:
            # Convert seconds to milliseconds
            cutoff_ms = self._current_timestamp_ms() - int(older_than_seconds * 1000)
            return self._cleanup_by_timestamp(cutoff_ms, before)

        # Invalidate by context
        if context is not None:
            context_hash = create_fingerprint(context)
            return self._delete_by_hash(context_hash, before)

        # Invalidate by query (semantic)
        if query is not None:
            logger.warning("Semantic invalidation not implemented yet")
            return 0

    def _delete_by_hash(self, context_hash: str, before: int) -> int:
        """Delete entries with specific context_hash."""
        if self.config.db_uri == "memory://":
            arrow_table = self.table.to_arrow()
            mask = pc.not_equal(arrow_table["context_hash"], context_hash)
            filtered = arrow_table.filter(mask)

            self.table = self.db.create_table(
                self.config.table_name,
                data=filtered if len(filtered) > 0 else None,
                schema=self.schema if len(filtered) == 0 else None,
                mode="overwrite",
            )
        else:
            self.table.delete(f"context_hash = '{context_hash}'")
            try:
                self.table.compact_files()
            except AttributeError:
                pass

        after = self.table.count_rows()
        deleted = before - after
        logger.info(f"Invalidated {deleted} entries with ctx_hash={context_hash[:8]}")
        return deleted

    def _cleanup_by_timestamp(self, cutoff_ms: int, before: int) -> int:
        """Delete entries with timestamp <= cutoff_ms (older than cutoff)."""
        if self.config.db_uri == "memory://":
            arrow_table = self.table.to_arrow()
            # Keep only NEWER entries than cutoff
            mask = pc.greater(arrow_table["timestamp"], cutoff_ms)
            filtered = arrow_table.filter(mask)

            self.table = self.db.create_table(
                self.config.table_name,
                data=filtered if len(filtered) > 0 else None,
                schema=self.schema if len(filtered) == 0 else None,
                mode="overwrite",
            )
        else:
            # Delete OLDER entries than cutoff
            self.table.delete(f"timestamp <= {cutoff_ms}")
            try:
                self.table.compact_files()
            except AttributeError:
                pass

        after = self.table.count_rows()
        deleted = before - after
        logger.info(f"Cleaned up {deleted} expired entries (cutoff: {cutoff_ms}ms)")
        return deleted

    def cleanup_expired(self) -> int:
        """
        Clean expired entries according to configured TTL.

        Returns:
            Number of deleted entries
        """
        if self.config.ttl_seconds is None:
            logger.warning("No TTL configured, skipping cleanup")
            return 0

        cutoff_ms = self._current_timestamp_ms() - int(self.config.ttl_seconds * 1000)
        before = self.table.count_rows()

        return self._cleanup_by_timestamp(cutoff_ms, before)

    def create_index(
        self,
        num_partitions: int = 256,
        num_sub_vectors: Optional[int] = None,
    ) -> None:
        """
        Create IVF-PQ index for fast vector searches.

        IMPORTANT: Only useful with >256 entries. For less, use linear ANN.

        Args:
            num_partitions: Number of IVF clusters (default: 256)
            num_sub_vectors: Sub-vectors for PQ (default: embedding_dim // 4)

        Example:
            >>> memora = Memora(CacheConfig.for_production())
            >>> # ... add >1000 entries ...
            >>> memora.create_index(num_partitions=512)
        """
        row_count = self.table.count_rows()

        if row_count < 256:
            logger.warning(
                f"Only {row_count} entries - index not recommended. "
                "At least 256 entries required."
            )
            return

        if num_sub_vectors is None:
            num_sub_vectors = max(1, self.embedding_dim // 4)

        logger.info(
            f"Creating vector index: partitions={num_partitions}, "
            f"sub_vectors={num_sub_vectors}, entries={row_count}"
        )

        try:
            self.table.create_index(
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
            self._index_created = True
            logger.info("Index created successfully")
        except Exception as e:
            logger.error(f"Error creating index: {e}", exc_info=True)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        stats = {
            "total_entries": self.table.count_rows(),
            "max_entries": self.config.max_entries,
            "max_result_size_bytes": self.config.max_result_size_bytes,
            "eviction_policy": self.config.eviction_policy,
            "threshold": self.config.similarity_threshold,
            "embedding_dim": self.embedding_dim,
            "model": self.config.model_name,
            "ttl_seconds": self.config.ttl_seconds,
            "storage": self.config.db_uri,
            "index_created": self._index_created,
        }

        if self.metrics:
            stats.update(self.metrics.report())

        return stats

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Return vector index statistics.

        Returns:
            Dict with index metadata (or None if doesn't exist)
        """
        return {
            "has_index": self._index_created,
            "total_entries": self.table.count_rows(),
            "note": "LanceDB doesn't expose detailed index metrics",
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache components.

        Verifies:
        - Embedding model functionality
        - Database accessibility
        - Recent error rates

        Returns:
            Dict with health status:
            {
                "status": "healthy" | "unhealthy",
                "checks": {
                    "embedding": {"ok": bool, "error": str | None},
                    "database": {"ok": bool, "error": str | None},
                    "error_rate": {"ok": bool, "details": str}
                },
                "metrics": {
                    "total_entries": int,
                    "recent_errors": {"lookup": int, "store": int}
                },
                "timestamp": int
            }

        Example:
            >>> memora = Memora()
            >>> health = memora.health_check()
            >>> if health["status"] == "healthy":
            ...     print("Cache is operational")
        """
        checks = {
            "embedding": {"ok": True, "error": None},
            "database": {"ok": True, "error": None},
            "error_rate": {"ok": True, "details": "No metrics available"},
        }

        # 1. Test embedding generation
        try:
            test_embedding = self._embed("health check test")
            if len(test_embedding) != self.embedding_dim:
                checks["embedding"]["ok"] = False
                checks["embedding"]["error"] = (
                    f"Embedding dimension mismatch: {len(test_embedding)} != {self.embedding_dim}"
                )
        except Exception as e:
            checks["embedding"]["ok"] = False
            checks["embedding"]["error"] = str(e)
            logger.error(f"Health check: Embedding test failed: {e}", exc_info=True)

        # 2. Test database access
        try:
            entry_count = self.table.count_rows()
            # Try a basic read operation
            if entry_count > 0:
                _ = self.table.to_arrow()
        except Exception as e:
            checks["database"]["ok"] = False
            checks["database"]["error"] = str(e)
            logger.error(f"Health check: Database test failed: {e}", exc_info=True)

        # 3. Check error rates (if metrics enabled)
        if self.metrics:
            total_requests = self.metrics.total_requests
            lookup_errors = self.metrics.lookup_errors
            store_errors = self.metrics.store_errors
            total_errors = lookup_errors + store_errors

            # Consider unhealthy if error rate > 10% and at least 10 requests
            if total_requests >= 10:
                error_rate = total_errors / total_requests if total_requests > 0 else 0
                if error_rate > 0.10:
                    checks["error_rate"]["ok"] = False
                    checks["error_rate"]["details"] = (
                        f"High error rate: {error_rate * 100:.1f}% "
                        f"({total_errors}/{total_requests} requests)"
                    )
                else:
                    checks["error_rate"]["ok"] = True
                    checks["error_rate"]["details"] = (
                        f"Error rate: {error_rate * 100:.1f}% "
                        f"({total_errors}/{total_requests} requests)"
                    )
            else:
                checks["error_rate"]["ok"] = True
                checks["error_rate"]["details"] = (
                    f"Insufficient data: {total_requests} requests"
                )

        # Determine overall status
        all_checks_ok = all(check["ok"] for check in checks.values())
        status = "healthy" if all_checks_ok else "unhealthy"

        # Build response
        response = {
            "status": status,
            "checks": checks,
            "metrics": {
                "total_entries": self.table.count_rows()
                if checks["database"]["ok"]
                else 0,
                "recent_errors": {
                    "lookup": self.metrics.lookup_errors if self.metrics else 0,
                    "store": self.metrics.store_errors if self.metrics else 0,
                },
            },
            "timestamp": self._current_timestamp_ms(),
        }

        if status == "unhealthy":
            logger.warning(f"Health check FAILED: {response}")
        else:
            logger.debug(f"Health check PASSED: {response}")

        return response
