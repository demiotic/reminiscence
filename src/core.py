"""Memora core - Queryable semantic cache."""

import logging
import time
from typing import Any, Dict, List, Optional

import lancedb
import pyarrow as pa
import pyarrow.compute as pc
from sentence_transformers import SentenceTransformer

from .config import CacheConfig
from .metrics import CacheMetrics
from .types import CacheEntry, LookupResult, AvailabilityCheck
from .utils import create_fingerprint, cosine_similarity, serialize, deserialize

logger = logging.getLogger(__name__)


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
        self.config = config or CacheConfig()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger.info(f"Initializing Memora | model={self.config.model_name}")

        # Initialize components
        self.model = SentenceTransformer(self.config.model_name)
        self.db = lancedb.connect(self.config.db_uri)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.metrics = CacheMetrics() if self.config.enable_metrics else None

        # Schema - timestamp in milliseconds, result and metadata as binary
        self.schema = pa.schema(
            [
                pa.field("query_text", pa.string()),
                pa.field("context_hash", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("result", pa.binary()),  # bytes, not string
                pa.field("timestamp", pa.int64()),  # Milliseconds since epoch
                pa.field("metadata", pa.binary()),  # bytes, not string
            ]
        )

        self.table = self._init_table()
        self._index_created = False

        logger.info(
            f"Memora ready | dim={self.embedding_dim}, "
            f"threshold={self.config.similarity_threshold}, "
            f"entries={self.table.count_rows()}"
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
                logger.debug("Cache empty")
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
                logger.debug(f"MISS | Search no results | {elapsed_ms:.1f}ms")
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
                    f"MISS | No results with matching context | {elapsed_ms:.1f}ms"
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
                    logger.debug("MISS | Results expired")
                    if self.metrics:
                        self.metrics.misses += 1
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
                    f"MISS | sim={best_sim:.3f} < threshold={threshold} | "
                    f"query='{query[:50]}...' | {elapsed_ms:.1f}ms"
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
                f"HIT | sim={best_sim:.3f} | "
                f"query='{query[:50]}...' → '{best_query[:50]}...' | "
                f"age={age_seconds}s | {elapsed_ms:.1f}ms"
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
                f"Cache lookup failed: {e} | query='{query[:50]}...' | "
                f"context_hash={create_fingerprint(context)[:8]} | {elapsed_ms:.1f}ms",
                exc_info=True,
            )
            if self.metrics:
                self.metrics.misses += 1
                self.metrics.record_lookup_latency(elapsed_ms)
                self.metrics.lookup_errors += 1  # ← FIXED: removed hasattr check
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
            context_hash = create_fingerprint(context)
            embedding = self._embed(query)
            timestamp = self._current_timestamp_ms()

            # Serialize result and metadata to bytes
            result_bytes = serialize(result)
            metadata_bytes = serialize(metadata) if metadata else b""

            # Track size in metrics
            if self.metrics:
                self.metrics.record_result_size(len(result_bytes))

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
            logger.debug(
                f"Stored | ctx_hash={context_hash[:8]} | query='{query[:50]}...'"
            )

            # Auto-create index if needed
            if self.config.auto_create_index:
                self._maybe_auto_create_index()

        except Exception as e:
            # DO NOT propagate error - app must continue without cache
            logger.error(
                f"Cache store failed: {e} | query='{query[:50]}...'", exc_info=True
            )
            if self.metrics:
                self.metrics.store_errors += 1  # ← FIXED: removed hasattr check

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
