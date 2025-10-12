"""Core cache operations - lookup, store, maintenance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .operations.invalidation import InvalidationOperations
from .operations.lookup import LookupOperations
from .operations.maintenance import MaintenanceOperations
from .operations.store import StorageOperations
from .types import (
    BulkInvalidatePattern,
    LookupRequest,
    LookupResult,
    MultiModalInput,
    QueryMode,
    StoreRequest,
)


class CacheOperations:
    """Internal facade for all cache operations with hybrid matching.

    This class delegates to specialized operation modules and should not
    be used directly. Use the Reminiscence class as the public API.

    Architecture:
        - LookupOperations: Handles cache lookups with hybrid matching
        - StorageOperations: Handles storing and retrieving entries
        - InvalidationOperations: Handles cache invalidation
        - MaintenanceOperations: Handles background maintenance tasks

    Note:
        This class is internal (no leading underscore by convention).
        External users should interact via the Reminiscence facade.
    """

    def __init__(
        self,
        storage: Any,
        embedder: Any,
        eviction: Any,
        config: Any,
        metrics: Any = None,
    ) -> None:
        """Initialize cache operations with delegated components.

        Args:
            storage: Storage backend instance.
            embedder: Embedding model instance.
            eviction: Eviction policy instance.
            config: Configuration object.
            metrics: Optional metrics collector.
        """
        self.storage = storage
        self.embedder = embedder
        self.eviction = eviction
        self.config = config
        self.metrics = metrics

        # Delegate to specialized operation modules
        self._lookup = LookupOperations(storage, embedder, eviction, config, metrics)
        self._storage = StorageOperations(storage, embedder, eviction, config, metrics)
        self._invalidation = InvalidationOperations(storage, eviction, config, metrics)
        self._maintenance = MaintenanceOperations(storage, eviction, config, metrics)

    # ========================================================================
    # Lookup Operations
    # ========================================================================

    def lookup(
        self,
        query: MultiModalInput,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        mode: QueryMode = QueryMode.AUTO,
        track_metrics: bool = True,
    ) -> LookupResult:
        """Look up multimodal query in cache with hybrid matching.

        Args:
            query: Multimodal query to look up.
            context: Contextual parameters for matching (default: {}).
            similarity_threshold: Minimum similarity score (overrides config).
            mode: Query matching strategy (default: QueryMode.AUTO).
            track_metrics: Whether to track metrics (default: True).

        Returns:
            LookupResult with hit status and data.
        """
        return self._lookup.lookup(
            query=query,
            context=context or {},
            similarity_threshold=similarity_threshold,
            query_mode=mode.value,
            track_metrics=track_metrics,
        )

    def lookup_batch(
        self,
        requests: List[LookupRequest],
        track_metrics: bool = True,
    ) -> List[LookupResult]:
        """Lookup multiple queries in batch (optimized for embeddings).

        Args:
            requests: List of LookupRequest objects.
            track_metrics: Whether to track metrics (default: True).

        Returns:
            List of LookupResult objects in same order as requests.

        Examples:
            >>> from reminiscence.types import LookupRequest, MultiModalInput
            >>> requests = [
            ...     LookupRequest(
            ...         query=MultiModalInput(text="hello"),
            ...         context={"lang": "en"}
            ...     ),
            ...     LookupRequest(
            ...         query=MultiModalInput(text="bonjour"),
            ...         context={"lang": "fr"}
            ...     ),
            ... ]
            >>> results = cache.lookup_batch(requests)
        """
        # Extract parallel lists for internal operations
        queries = [req.query for req in requests]
        contexts = [req.context or {} for req in requests]

        # Use first request's threshold/mode as default
        similarity_threshold = requests[0].similarity_threshold if requests else None
        mode = requests[0].mode or QueryMode.AUTO

        return self._lookup.lookup_batch(
            queries=queries,
            contexts=contexts,
            similarity_threshold=similarity_threshold,
            query_mode=mode.value,
            track_metrics=track_metrics,
        )

    def check_availability(
        self,
        query: MultiModalInput,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        mode: QueryMode = QueryMode.AUTO,
    ) -> Any:
        """Check if cached result exists without retrieving it.

        Args:
            query: Multimodal query to check.
            context: Contextual parameters (default: {}).
            similarity_threshold: Minimum similarity score.
            mode: Query matching strategy (default: QueryMode.AUTO).

        Returns:
            AvailabilityCheck with status and metadata.
        """
        return self._lookup.check_availability(
            query=query,
            context=context or {},
            similarity_threshold=similarity_threshold,
            query_mode=mode.value,
        )

    # ========================================================================
    # Storage Operations
    # ========================================================================

    def store(
        self,
        query: MultiModalInput,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        context_threshold: Optional[float] = None,
        allow_errors: bool = False,
        mode: QueryMode = QueryMode.AUTO,
    ) -> None:
        """Store multimodal query result in cache.

        Args:
            query: Multimodal query being cached.
            context: Context dict for exact matching.
            result: Result data to cache.
            metadata: Optional metadata to store with entry.
            ttl_seconds: Time-to-live in seconds (overrides global TTL).
            context_threshold: Override similarity threshold for this entry.
            allow_errors: If True, store error results (default: False).
            mode: Query matching strategy (default: QueryMode.AUTO).
        """
        return self._storage.store(
            query=query,
            context=context,
            result=result,
            metadata=metadata,
            query_mode=mode.value,
            allow_errors=allow_errors,
            ttl_seconds=ttl_seconds,
            context_threshold=context_threshold,
        )

    def store_batch(
        self,
        requests: List[StoreRequest],
        allow_errors: bool = False,
        mode: QueryMode = QueryMode.AUTO,
    ) -> None:
        """Store multiple results in batch optimized for embeddings.

        Args:
            requests: List of StoreRequest objects.
            allow_errors: If True, store error results (default: False).
            mode: Query matching strategy (default: QueryMode.AUTO).

        Examples:
            >>> from reminiscence.types import StoreRequest, MultiModalInput
            >>> requests = [
            ...     StoreRequest(
            ...         query=MultiModalInput(text="hello"),
            ...         context={"lang": "en"},
            ...         result="Hello!",
            ...         ttl_seconds=3600
            ...     ),
            ...     StoreRequest(
            ...         query=MultiModalInput(text="bonjour"),
            ...         context={"lang": "fr"},
            ...         result="Bonjour!",
            ...         ttl_seconds=3600
            ...     ),
            ... ]
            >>> cache.store_batch(requests)
        """
        # Extract parallel lists for internal operations
        queries = [req.query for req in requests]
        contexts = [req.context for req in requests]
        results = [req.result for req in requests]
        metadata_list = [req.metadata for req in requests]
        ttl_seconds = [req.ttl_seconds for req in requests]
        context_thresholds = [req.context_threshold for req in requests]

        return self._storage.store_batch(
            queries=queries,
            contexts=contexts,
            results=results,
            metadata=metadata_list,
            query_mode=mode.value,
            allow_errors=allow_errors,
            ttl_seconds=ttl_seconds,
            context_thresholds=context_thresholds,
        )

    # ========================================================================
    # Invalidation Operations
    # ========================================================================

    def invalidate(
        self,
        query: Optional[MultiModalInput] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """Invalidate cache entries by criteria.

        Args:
            query: Exact multimodal query to invalidate (optional).
            context: Exact context to match (optional).
            older_than_seconds: Invalidate entries older than this (optional).

        Returns:
            Number of entries invalidated.

        Examples:
            Invalidate specific query:
            >>> count = cache.invalidate(
            ...     query=MultiModalInput(text="outdated"),
            ...     context={"model": "gpt-3.5"}
            ... )

            Invalidate by age:
            >>> count = cache.invalidate(older_than_seconds=3600)
        """
        return self._invalidation.invalidate(
            query=query,
            context=context,
            older_than_seconds=older_than_seconds,
        )

    def bulk_invalidate(self, pattern: BulkInvalidatePattern) -> int:
        """Bulk invalidate entries matching pattern using efficient batch deletion.

        Args:
            pattern: BulkInvalidatePattern specification.

        Returns:
            Number of entries invalidated.

        Examples:
            >>> from reminiscence.types import BulkInvalidatePattern
            >>> pattern = BulkInvalidatePattern(
            ...     query_regex="^SELECT.*",
            ...     older_than_seconds=86400
            ... )
            >>> count = cache.bulk_invalidate(pattern)
        """
        return self._invalidation.bulk_invalidate(pattern)

    def invalidate_by_prefix(self, query_prefix: str) -> int:
        """Invalidate all entries with query text starting with prefix.

        Args:
            query_prefix: Prefix to match in query text.

        Returns:
            Number of entries invalidated.
        """
        return self._invalidation.invalidate_by_prefix(query_prefix)

    def invalidate_by_regex(self, query_regex: str) -> int:
        """Invalidate all entries matching regex pattern.

        Args:
            query_regex: Regex pattern to match query text.

        Returns:
            Number of entries invalidated.
        """
        return self._invalidation.invalidate_by_regex(query_regex)

    def invalidate_by_context(self, context_matches: Dict[str, str]) -> int:
        """Invalidate entries matching context pattern with wildcard support.

        Args:
            context_matches: Dict of context key-value patterns (* wildcard).

        Returns:
            Number of entries invalidated.

        Examples:
            >>> count = cache.invalidate_by_context({
            ...     "model": "gpt-3.5",
            ...     "agent_*": "*"
            ... })
        """
        return self._invalidation.invalidate_by_context(context_matches)

    def invalidate_older_than(self, seconds: float) -> int:
        """Invalidate all entries older than specified seconds.

        Args:
            seconds: Age threshold in seconds.

        Returns:
            Number of entries invalidated.
        """
        return self._invalidation.invalidate_older_than(seconds)

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        return self._invalidation.clear_all()

    # ========================================================================
    # Maintenance Operations
    # ========================================================================

    def cleanup_expired(self) -> int:
        """Remove expired entries based on TTL.

        Returns:
            Number of expired entries removed.
        """
        return self._maintenance.cleanup_expired()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with statistics including:
            - total_entries: Total number of cached entries
            - hit_rate: Cache hit rate percentage
            - storage_size: Approximate storage size
            - oldest_entry_age: Age of oldest entry
        """
        return self._maintenance.stats()

    def export_to_file(self, filepath: str, format: str = "parquet") -> None:
        """Export cache to file.

        Args:
            filepath: Path to export file.
            format: Export format (parquet, json, csv).

        Examples:
            >>> cache.export_to_file("/tmp/cache_backup.parquet")
            >>> cache.export_to_file("/tmp/cache.json", format="json")
        """
        return self._maintenance.export_to_file(filepath, format)

    def import_from_file(self, filepath: str, format: str = "parquet") -> None:
        """Import cache from file.

        Args:
            filepath: Path to import file.
            format: Import format (parquet, json, csv).

        Examples:
            >>> cache.import_from_file("/tmp/cache_backup.parquet")
            >>> cache.import_from_file("/tmp/cache.json", format="json")
        """
        return self._maintenance.import_from_file(filepath, format)

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all cache entries as list of dicts.

        Returns:
            List of dictionaries representing cache entries.

        Warning:
            This loads all entries into memory. Use with caution on large caches.
        """
        return self._maintenance.get_all_entries()

    # ========================================================================
    # Internal Helpers (for testing/backward compat)
    # ========================================================================

    def _is_error_result(self, result: Any) -> bool:
        """Check if result represents an error (internal method for tests).

        Args:
            result: Result to check.

        Returns:
            True if result is an error, False otherwise.
        """
        return self._storage._is_error_result(result)
