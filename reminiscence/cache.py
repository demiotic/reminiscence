"""Core cache operations - lookup, store, maintenance."""

from typing import Optional, Dict, Any, List
from .types import LookupResult, BulkInvalidatePattern
from .operations.lookup import LookupOperations
from .operations.storage_ops import StorageOperations
from .operations.invalidation import InvalidationOperations
from .operations.maintenance import MaintenanceOperations


class CacheOperations:
    """Facade for all cache operations with hybrid matching."""

    def __init__(self, storage, embedder, eviction, config, metrics=None):
        """Initialize cache operations with delegated components."""
        self.storage = storage
        self.embedder = embedder
        self.eviction = eviction
        self.config = config
        self.metrics = metrics

        self._lookup = LookupOperations(storage, embedder, eviction, config, metrics)
        self._storage = StorageOperations(storage, embedder, eviction, config, metrics)
        self._invalidation = InvalidationOperations(storage, eviction, config, metrics)
        self._maintenance = MaintenanceOperations(storage, eviction, config, metrics)

    # Lookup Operations
    def lookup(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
        track_metrics: bool = True,
    ) -> LookupResult:
        """Search cache by query with exact context matching."""
        return self._lookup.lookup(
            query, context, similarity_threshold, query_mode, track_metrics
        )

    def lookup_batch(
        self,
        queries: List[str],
        contexts: List[Dict[str, Any]],
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
        track_metrics: bool = True,
    ) -> List[LookupResult]:
        """Batch lookup for multiple queries optimized for embeddings."""
        return self._lookup.lookup_batch(
            queries, contexts, similarity_threshold, query_mode, track_metrics
        )

    def check_availability(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        query_mode: str = "semantic",
    ) -> bool:
        """Check if cached result exists without retrieving it."""
        return self._lookup.check_availability(
            query, context, similarity_threshold, query_mode
        )

    # Storage Operations
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
        """Store result in cache with context."""
        return self._storage.store(
            query,
            context,
            result,
            metadata,
            query_mode,
            allow_errors,
            ttl_seconds,
            context_threshold,
        )

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
        """Store multiple results in batch optimized for embeddings."""
        return self._storage.store_batch(
            queries,
            contexts,
            results,
            metadata,
            query_mode,
            allow_errors,
            ttl_seconds,
            context_thresholds,
        )

    # Invalidation Operations
    def invalidate(
        self,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """Invalidate cache entries by criteria."""
        return self._invalidation.invalidate(query, context, older_than_seconds)

    def invalidate_bulk(self, pattern: BulkInvalidatePattern) -> int:
        """Bulk invalidate entries matching pattern using efficient batch deletion."""
        return self._invalidation.invalidate_bulk(pattern)

    def invalidate_by_prefix(self, query_prefix: str) -> int:
        """Invalidate all entries with query starting with prefix."""
        return self._invalidation.invalidate_by_prefix(query_prefix)

    def invalidate_by_regex(self, query_regex: str) -> int:
        """Invalidate all entries matching regex pattern."""
        return self._invalidation.invalidate_by_regex(query_regex)

    def invalidate_by_context(self, context_matches: Dict[str, str]) -> int:
        """Invalidate entries matching context pattern with wildcard support."""
        return self._invalidation.invalidate_by_context(context_matches)

    def invalidate_older_than(self, seconds: float) -> int:
        """Invalidate all entries older than specified seconds."""
        return self._invalidation.invalidate_older_than(seconds)

    def clear_all(self) -> int:
        """Clear all cache entries."""
        return self._invalidation.clear_all()

    # Maintenance Operations
    def cleanup_expired(self) -> int:
        """Remove expired entries based on TTL."""
        return self._maintenance.cleanup_expired()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._maintenance.stats()

    def export_to_file(self, filepath: str, format: str = "parquet"):
        """Export cache to file."""
        return self._maintenance.export_to_file(filepath, format)

    def import_from_file(self, filepath: str, format: str = "parquet"):
        """Import cache from file."""
        return self._maintenance.import_from_file(filepath, format)

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all cache entries as list of dicts."""
        return self._maintenance.get_all_entries()

    # Internal helper delegation (for testing)
    def _is_error_result(self, result: Any) -> bool:
        """Check if result represents an error.

        Internal method exposed through facade for backward compatibility with tests.
        Delegates to storage operations.

        Args:
            result: Result object to check

        Returns:
            True if result is an error
        """
        return self._storage._is_error_result(result)
