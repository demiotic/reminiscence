"""gRPC client for Reminiscence semantic cache.

Example:
    >>> from reminiscence.api.client import ReminiscenceClient
    >>> from reminiscence.types import MultiModalInput
    >>>
    >>> client = ReminiscenceClient("localhost:8080")
    >>>
    >>> # Lookup
    >>> query = MultiModalInput(text="What is ML?")
    >>> result = client.lookup(query, {"model": "gpt-4"})
    >>> if result.is_hit:
    ...     print(result.result)
    >>>
    >>> # Store
    >>> client.store(query, {"model": "gpt-4"}, "Machine Learning is...")
    >>>
    >>> # Health check
    >>> health = client.health_check()
    >>> print(health["status"])
    >>>
    >>> # Cleanup
    >>> client.close()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import grpc

from ..types import (
    AvailabilityCheck,
    LookupRequest,
    LookupResult,
    MultiModalInput,
    QueryMode,
    StoreRequest,
)
from . import reminiscence_pb2 as pb2
from . import reminiscence_pb2_grpc as pb2_grpc


class ReminiscenceClient:
    """Python client for Reminiscence gRPC service.

    This client provides a Python-friendly interface to the gRPC service,
    handling all type conversions automatically.

    Args:
        address: Server address in format "host:port" (e.g., "localhost:8080").
        timeout: Default timeout for RPC calls in seconds (default: 10).
        credentials: Optional gRPC credentials for secure connections.
        options: Optional channel options (e.g., max message size).

    Example:
        >>> client = ReminiscenceClient("localhost:8080")
        >>> query = MultiModalInput(text="What is AI?")
        >>> result = client.lookup(query, {"model": "gpt-4"})
        >>> client.close()
        >>>
        >>> # Or use context manager:
        >>> with ReminiscenceClient("localhost:8080") as client:
        ...     result = client.lookup(query, {"model": "gpt-4"})
    """

    def __init__(
        self,
        address: str,
        timeout: float = 10.0,
        credentials: Optional[grpc.ChannelCredentials] = None,
        options: Optional[List[tuple]] = None,
    ):
        """Initialize gRPC client."""
        self.address = address
        self.timeout = timeout

        # Create channel
        if credentials:
            self.channel = grpc.secure_channel(address, credentials, options)
        else:
            self.channel = grpc.insecure_channel(address, options)

        # Create stub
        self.stub = pb2_grpc.ReminiscenceServiceStub(self.channel)

    def close(self) -> None:
        """Close the gRPC channel."""
        self.channel.close()

    def __enter__(self) -> ReminiscenceClient:
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager support - closes channel."""
        self.close()

    # ========================================================================
    # Type Converters
    # ========================================================================

    @staticmethod
    def _query_mode_to_proto(mode: QueryMode) -> pb2.QueryMode:
        """Convert QueryMode to protobuf enum."""
        mapping = {
            QueryMode.AUTO: pb2.QUERY_MODE_AUTO,
            QueryMode.SEMANTIC: pb2.QUERY_MODE_SEMANTIC,
            QueryMode.EXACT: pb2.QUERY_MODE_EXACT,
        }
        return mapping.get(mode, pb2.QUERY_MODE_AUTO)

    @staticmethod
    def _proto_to_query_mode(proto_mode: pb2.QueryMode) -> QueryMode:
        """Convert protobuf enum to QueryMode."""
        mapping = {
            pb2.QUERY_MODE_AUTO: QueryMode.AUTO,
            pb2.QUERY_MODE_SEMANTIC: QueryMode.SEMANTIC,
            pb2.QUERY_MODE_EXACT: QueryMode.EXACT,
        }
        return mapping.get(proto_mode, QueryMode.AUTO)

    @staticmethod
    def _multimodal_input_to_proto(query: MultiModalInput) -> pb2.MultiModalInput:
        """Convert MultiModalInput to protobuf."""
        proto = pb2.MultiModalInput()
        if query.text:
            proto.text = query.text
        if query.image:
            proto.image = query.image
        if query.video:
            proto.video = query.video
        if query.audio:
            proto.audio = query.audio
        if query.metadata:
            proto.metadata.update(query.metadata)
        return proto

    @staticmethod
    def _context_to_proto(context: Dict[str, Any]) -> pb2.Context:
        """Convert context dict to protobuf."""
        proto = pb2.Context()
        for key, value in context.items():
            proto.values[key] = str(value)
        return proto

    @staticmethod
    def _proto_to_context(proto: pb2.Context) -> Dict[str, Any]:
        """Convert protobuf context to dict."""
        return dict(proto.values)

    @staticmethod
    def _serialize_result(result: Any) -> pb2.SerializedResult:
        """Serialize result to protobuf."""
        import orjson
        json_bytes = orjson.dumps(result)
        return pb2.SerializedResult(json_data=json_bytes)

    @staticmethod
    def _deserialize_result(proto: pb2.SerializedResult) -> Any:
        """Deserialize result from protobuf."""
        import orjson
        return orjson.loads(proto.json_data)

    @staticmethod
    def _proto_to_lookup_result(proto: pb2.LookupResponse) -> LookupResult:
        """Convert protobuf LookupResponse to LookupResult."""
        result = None
        if proto.HasField("result"):
            result = ReminiscenceClient._deserialize_result(proto.result)

        context = {}
        if proto.HasField("context"):
            context = ReminiscenceClient._proto_to_context(proto.context)

        return LookupResult(
            hit=proto.hit,
            result=result,
            similarity=proto.similarity if proto.HasField("similarity") else None,
            matched_query=(
                proto.matched_query if proto.HasField("matched_query") else None
            ),
            age_seconds=(
                proto.age_seconds if proto.HasField("age_seconds") else None
            ),
            entry_id=proto.entry_id if proto.HasField("entry_id") else None,
            context=context,
            ttl_remaining=(
                proto.ttl_remaining if proto.HasField("ttl_remaining") else None
            ),
        )

    # ========================================================================
    # Core Cache Operations
    # ========================================================================

    def lookup(
        self,
        query: MultiModalInput,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        mode: QueryMode = QueryMode.AUTO,
        track_metrics: bool = True,
        timeout: Optional[float] = None,
    ) -> LookupResult:
        """Lookup cache entry by semantic similarity.

        Args:
            query: MultiModalInput containing text, image, video, or audio.
            context: Context dict for exact matching (default: {}).
            similarity_threshold: Minimum similarity score (overrides config).
            mode: Matching strategy (default: QueryMode.AUTO).
            track_metrics: Whether to track cache metrics (default: True).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            LookupResult with hit status and cached data.
        """
        request = pb2.LookupRequest(
            query=self._multimodal_input_to_proto(query),
            mode=self._query_mode_to_proto(mode),
            track_metrics=track_metrics,
        )

        if context:
            request.context.CopyFrom(self._context_to_proto(context))

        if similarity_threshold is not None:
            request.similarity_threshold = similarity_threshold

        response = self.stub.Lookup(request, timeout=timeout or self.timeout)
        return self._proto_to_lookup_result(response)

    def lookup_batch(
        self,
        requests: List[LookupRequest],
        track_metrics: bool = True,
        timeout: Optional[float] = None,
    ) -> List[LookupResult]:
        """Lookup multiple queries in batch.

        Args:
            requests: List of LookupRequest objects.
            track_metrics: Whether to track metrics (default: True).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            List of LookupResult objects (same order as requests).
        """
        proto_requests = []
        for req in requests:
            proto_req = pb2.LookupRequest(
                query=self._multimodal_input_to_proto(req.query),
                mode=self._query_mode_to_proto(req.mode),
                track_metrics=track_metrics,
            )

            if req.context:
                proto_req.context.CopyFrom(self._context_to_proto(req.context))

            if req.similarity_threshold is not None:
                proto_req.similarity_threshold = req.similarity_threshold

            proto_requests.append(proto_req)

        batch_request = pb2.LookupBatchRequest(
            requests=proto_requests,
            track_metrics=track_metrics,
        )

        response = self.stub.LookupBatch(
            batch_request,
            timeout=timeout or self.timeout,
        )

        return [self._proto_to_lookup_result(r) for r in response.responses]

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
        timeout: Optional[float] = None,
    ) -> bool:
        """Store result in cache.

        Args:
            query: MultiModalInput containing text, image, video, or audio.
            context: Context dict (will be matched exactly in future lookups).
            result: Result to cache (supports JSON, Arrow, Pandas, Polars).
            metadata: Optional metadata to store with entry.
            ttl_seconds: Time-to-live in seconds (overrides global).
            context_threshold: Override similarity threshold for this entry.
            allow_errors: If True, store error results (default: False).
            mode: Query matching strategy (default: QueryMode.AUTO).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            True if stored successfully, False otherwise.
        """
        request = pb2.StoreRequest(
            query=self._multimodal_input_to_proto(query),
            context=self._context_to_proto(context),
            result=self._serialize_result(result),
            allow_errors=allow_errors,
            mode=self._query_mode_to_proto(mode),
        )

        if metadata:
            request.metadata.update(
                {k: str(v) for k, v in metadata.items()}
            )

        if ttl_seconds is not None:
            request.ttl_seconds = ttl_seconds

        if context_threshold is not None:
            request.context_threshold = context_threshold

        response = self.stub.Store(request, timeout=timeout or self.timeout)
        return response.success

    def store_batch(
        self,
        requests: List[StoreRequest],
        allow_errors: bool = False,
        mode: QueryMode = QueryMode.AUTO,
        timeout: Optional[float] = None,
    ) -> List[bool]:
        """Store multiple results in batch.

        Args:
            requests: List of StoreRequest objects.
            allow_errors: If True, store error results (default: False).
            mode: Query matching strategy (default: QueryMode.AUTO).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            List of success flags (True/False) for each request.
        """
        proto_requests = []
        for req in requests:
            proto_req = pb2.StoreRequest(
                query=self._multimodal_input_to_proto(req.query),
                context=self._context_to_proto(req.context),
                result=self._serialize_result(req.result),
                allow_errors=allow_errors,
                mode=self._query_mode_to_proto(mode),
            )

            if req.metadata:
                proto_req.metadata.update(
                    {k: str(v) for k, v in req.metadata.items()}
                )

            if req.ttl_seconds is not None:
                proto_req.ttl_seconds = req.ttl_seconds

            if req.context_threshold is not None:
                proto_req.context_threshold = req.context_threshold

            proto_requests.append(proto_req)

        batch_request = pb2.StoreBatchRequest(
            requests=proto_requests,
            allow_errors=allow_errors,
            mode=self._query_mode_to_proto(mode),
        )

        response = self.stub.StoreBatch(
            batch_request,
            timeout=timeout or self.timeout,
        )

        return [r.success for r in response.responses]

    def check_availability(
        self,
        query: MultiModalInput,
        context: Dict[str, Any],
        similarity_threshold: Optional[float] = None,
        mode: QueryMode = QueryMode.AUTO,
        timeout: Optional[float] = None,
    ) -> AvailabilityCheck:
        """Verify availability without retrieving full data.

        Args:
            query: MultiModalInput to check.
            context: Context dict.
            similarity_threshold: Minimum similarity score (overrides config).
            mode: Matching strategy (default: QueryMode.AUTO).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            AvailabilityCheck with availability info.
        """
        request = pb2.CheckAvailabilityRequest(
            query=self._multimodal_input_to_proto(query),
            context=self._context_to_proto(context),
            mode=self._query_mode_to_proto(mode),
        )

        if similarity_threshold is not None:
            request.similarity_threshold = similarity_threshold

        response = self.stub.CheckAvailability(
            request,
            timeout=timeout or self.timeout,
        )

        return AvailabilityCheck(
            available=response.available,
            age_seconds=(
                response.age_seconds if response.HasField("age_seconds") else None
            ),
            ttl_remaining_seconds=(
                response.ttl_remaining_seconds
                if response.HasField("ttl_remaining_seconds")
                else None
            ),
            similarity=(
                response.similarity if response.HasField("similarity") else None
            ),
        )

    # ========================================================================
    # Invalidation & Cleanup
    # ========================================================================

    def invalidate(
        self,
        query: Optional[MultiModalInput] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> int:
        """Invalidate cache entries by criteria.

        Args:
            query: Exact multimodal query to invalidate (optional).
            context: Exact context to match (optional).
            older_than_seconds: Invalidate entries older than this (optional).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            Number of invalidated entries.
        """
        request = pb2.InvalidateRequest()

        if query is not None:
            request.query.CopyFrom(self._multimodal_input_to_proto(query))

        if context is not None:
            request.context.CopyFrom(self._context_to_proto(context))

        if older_than_seconds is not None:
            request.older_than_seconds = older_than_seconds

        response = self.stub.Invalidate(request, timeout=timeout or self.timeout)
        return response.invalidated_count

    def cleanup_expired(self, timeout: Optional[float] = None) -> int:
        """Clean expired entries according to configured TTL.

        Args:
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            Number of deleted entries.
        """
        request = pb2.CleanupExpiredRequest()
        response = self.stub.CleanupExpired(
            request,
            timeout=timeout or self.timeout,
        )
        return response.deleted_count

    def clear(self, timeout: Optional[float] = None) -> bool:
        """Clear all cache entries.

        Args:
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            True if cleared successfully.
        """
        request = pb2.ClearRequest()
        response = self.stub.Clear(request, timeout=timeout or self.timeout)
        return response.success

    # ========================================================================
    # Index & Stats
    # ========================================================================

    def create_index(
        self,
        num_partitions: int = 256,
        num_subvectors: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Create IVF-PQ index for fast vector searches.

        Args:
            num_partitions: Number of IVF partitions (default: 256).
            num_subvectors: Number of PQ subvectors (default: embedding_dim / 4).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            True if index created successfully.
        """
        request = pb2.CreateIndexRequest(num_partitions=num_partitions)
        if num_subvectors is not None:
            request.num_subvectors = num_subvectors

        response = self.stub.CreateIndex(request, timeout=timeout or self.timeout)
        return response.success

    def get_stats(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Return cache statistics.

        Args:
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            Dict with cache statistics.
        """
        request = pb2.GetStatsRequest()
        response = self.stub.GetStats(request, timeout=timeout or self.timeout)

        stats = {
            "cache_entries": response.cache_entries,
            "total_entries": response.total_entries,
            "max_entries": response.max_entries,
            "eviction_policy": response.eviction_policy,
            "threshold": response.threshold,
            "embedding_dim": response.embedding_dim,
            "model": response.model,
            "ttl_seconds": (
                response.ttl_seconds if response.HasField("ttl_seconds") else None
            ),
            "storage": response.storage,
            "index_created": response.index_created,
        }

        # Add metrics if available
        if response.HasField("hits"):
            stats["hits"] = response.hits
        if response.HasField("misses"):
            stats["misses"] = response.misses
        if response.HasField("hit_rate"):
            stats["hit_rate"] = response.hit_rate
        if response.HasField("avg_lookup_latency_ms"):
            stats["avg_lookup_latency_ms"] = response.avg_lookup_latency_ms
        if response.HasField("avg_store_latency_ms"):
            stats["avg_store_latency_ms"] = response.avg_store_latency_ms
        if response.HasField("lookup_errors"):
            stats["lookup_errors"] = response.lookup_errors
        if response.HasField("store_errors"):
            stats["store_errors"] = response.store_errors
        if response.HasField("evictions"):
            stats["evictions"] = response.evictions

        # Add scheduler stats
        if response.schedulers:
            stats["schedulers"] = {}
            for name, scheduler_stats in response.schedulers.items():
                stats["schedulers"][name] = {
                    "running": scheduler_stats.running,
                    "total_runs": scheduler_stats.total_runs,
                    "errors": scheduler_stats.errors,
                    "last_run_timestamp": (
                        scheduler_stats.last_run_timestamp
                        if scheduler_stats.HasField("last_run_timestamp")
                        else None
                    ),
                    "next_run_timestamp": (
                        scheduler_stats.next_run_timestamp
                        if scheduler_stats.HasField("next_run_timestamp")
                        else None
                    ),
                }

        return stats

    def get_index_stats(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Return vector index statistics.

        Args:
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            Dict with index statistics.
        """
        request = pb2.GetIndexStatsRequest()
        response = self.stub.GetIndexStats(
            request,
            timeout=timeout or self.timeout,
        )

        return {
            "has_index": response.has_index,
            "total_entries": response.total_entries,
            "note": response.note,
        }

    def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        context_filter: Optional[Dict[str, Any]] = None,
        query_filter: Optional[str] = None,
        sort_by: str = "created_at",
        sort_descending: bool = True,
        include_embeddings: bool = False,
        include_results: bool = True,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """List cache entries with pagination and filtering.

        Args:
            limit: Max entries to return (default: 100, max: 1000).
            offset: Offset for pagination (default: 0).
            context_filter: Filter by exact context match (optional).
            query_filter: Filter by query text substring (optional).
            sort_by: Sort field - "created_at", "last_accessed", "access_count" (default: "created_at").
            sort_descending: Sort order (default: True for descending).
            include_embeddings: Include embedding vectors (default: False).
            include_results: Include cached results (default: True).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            Dict with entries list, total_count, has_more, and returned_count.

        Example:
            >>> entries = client.list_entries(limit=10, context_filter={"model": "gpt-4"})
            >>> print(f"Found {entries['total_count']} entries")
            >>> for entry in entries['entries']:
            ...     print(f"  {entry['query']['text']}: {entry['created_at']}")
        """
        # Map sort field strings to proto enums
        sort_field_map = {
            "created_at": pb2.SORT_FIELD_CREATED_AT,
            "last_accessed": pb2.SORT_FIELD_LAST_ACCESSED,
            "access_count": pb2.SORT_FIELD_ACCESS_COUNT,
            "similarity": pb2.SORT_FIELD_SIMILARITY,
        }
        sort_field_enum = sort_field_map.get(sort_by, pb2.SORT_FIELD_CREATED_AT)

        # Proto3 booleans default to False, so we need to set them explicitly
        request = pb2.ListEntriesRequest(
            limit=limit,
            offset=offset,
            sort_by=sort_field_enum,
            sort_descending=sort_descending,  # Set explicitly (proto3 bool has no presence)
            include_embeddings=include_embeddings,  # Set explicitly
            include_results=include_results,  # Set explicitly
        )

        if context_filter:
            request.context_filter.CopyFrom(self._context_to_proto(context_filter))

        if query_filter:
            request.query_filter = query_filter

        response = self.stub.ListEntries(request, timeout=timeout or self.timeout)

        # Convert entries to dicts
        entries = []
        for entry in response.entries:
            entry_dict = {
                "entry_id": entry.entry_id,
                "query": {
                    "text": entry.query.text if entry.query.HasField("text") else None,
                    "image": entry.query.image if entry.query.HasField("image") else None,
                    "video": entry.query.video if entry.query.HasField("video") else None,
                    "audio": entry.query.audio if entry.query.HasField("audio") else None,
                    "metadata": dict(entry.query.metadata) if entry.query.metadata else None,
                },
                "context": self._proto_to_context(entry.context),
                "metadata": dict(entry.metadata) if entry.metadata else {},
                "created_at": entry.created_at,
                "last_accessed_at": entry.last_accessed_at,
                "access_count": entry.access_count,
                "embedding_dim": entry.embedding_dim,
                "matched_context_key": (
                    entry.matched_context_key
                    if entry.HasField("matched_context_key")
                    else None
                ),
            }

            # Add result if included
            if entry.HasField("result"):
                entry_dict["result"] = self._deserialize_result(entry.result)

            # Add TTL info if available
            if entry.HasField("ttl_seconds"):
                entry_dict["ttl_seconds"] = entry.ttl_seconds
            if entry.HasField("expires_at"):
                entry_dict["expires_at"] = entry.expires_at

            # Add embedding if included
            if entry.HasField("embedding"):
                import struct
                num_floats = len(entry.embedding) // 4
                entry_dict["embedding"] = list(struct.unpack(f'{num_floats}f', entry.embedding))

            entries.append(entry_dict)

        return {
            "entries": entries,
            "total_count": response.total_count,
            "has_more": response.has_more,
            "returned_count": response.returned_count,
        }

    # ========================================================================
    # Health & Capabilities
    # ========================================================================

    def health_check(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Perform health check on cache components.

        Args:
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            Dict with status and component health checks.
        """
        request = pb2.HealthCheckRequest()
        response = self.stub.HealthCheck(request, timeout=timeout or self.timeout)

        checks = {}
        for name, check in response.checks.items():
            checks[name] = {
                "ok": check.ok,
                "error": check.error if check.HasField("error") else None,
                "details": check.details if check.HasField("details") else None,
            }

        return {
            "status": response.status,
            "checks": checks,
            "metrics": {
                "total_entries": response.metrics.total_entries,
                "recent_errors": {
                    "lookup": response.metrics.recent_errors.lookup,
                    "store": response.metrics.recent_errors.store,
                },
            },
            "timestamp": response.timestamp,
        }

    def get_capabilities(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get server capabilities.

        Args:
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            Dict with version, Flight status, and supported features.
        """
        request = pb2.GetCapabilitiesRequest()
        response = self.stub.GetCapabilities(
            request,
            timeout=timeout or self.timeout,
        )

        return {
            "version": response.version,
            "flight_enabled": response.flight_enabled,
            "flight_endpoint": (
                response.flight_endpoint
                if response.HasField("flight_endpoint")
                else None
            ),
            "supported_features": list(response.supported_features),
        }

    # ========================================================================
    # Arrow Flight Data Plane
    # ========================================================================

    def get_flight_ticket(
        self,
        operation: str = "list_entries",
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get Flight ticket for bulk data operations.

        Args:
            operation: Operation name (e.g., "list_entries").
            parameters: Operation parameters (dict will be JSON-encoded).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            Dict with ticket (bytes), flight_endpoint, estimated_rows, estimated_bytes.

        Example:
            >>> ticket_info = client.get_flight_ticket(
            ...     operation="list_entries",
            ...     parameters={"limit": 1000, "include_embeddings": False}
            ... )
            >>> print(f"Flight endpoint: {ticket_info['flight_endpoint']}")
        """
        import json

        request = pb2.FlightTicketRequest(operation=operation)
        if parameters:
            request.parameters = json.dumps(parameters)

        response = self.stub.GetFlightTicket(
            request,
            timeout=timeout or self.timeout,
        )

        return {
            "ticket": response.ticket,
            "flight_endpoint": response.flight_endpoint,
            "estimated_rows": response.estimated_rows,
            "estimated_bytes": response.estimated_bytes,
        }

    def get_flight_client(
        self,
        flight_endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Get Arrow Flight client for streaming operations.

        Args:
            flight_endpoint: Flight server location (e.g., "grpc://localhost:8081").
                If None, fetches from server capabilities.
            timeout: RPC timeout for capability check (default: client timeout).

        Returns:
            pyarrow.flight.FlightClient instance.

        Raises:
            ImportError: If pyarrow is not installed.
            RuntimeError: If Flight is not enabled on server.

        Example:
            >>> flight_client = client.get_flight_client()
            >>> # Use flight_client for streaming operations
        """
        try:
            import pyarrow.flight as flight
        except ImportError:
            raise ImportError(
                "pyarrow is required for Flight client. "
                "Install with: pip install reminiscence[grpc]"
            )

        # Get Flight endpoint from capabilities if not provided
        if flight_endpoint is None:
            caps = self.get_capabilities(timeout=timeout)
            if not caps["flight_enabled"]:
                raise RuntimeError(
                    "Arrow Flight is not enabled on the server. "
                    "Enable with: REMINISCENCE_FLIGHT_ENABLED=true"
                )
            flight_endpoint = caps["flight_endpoint"]

        if not flight_endpoint:
            raise RuntimeError("Flight endpoint not available from server")

        return flight.FlightClient(flight_endpoint)

    def stream_entries_arrow(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        context_filter: Optional[Dict[str, Any]] = None,
        query_filter: Optional[str] = None,
        include_embeddings: bool = False,
        include_results: bool = True,
        batch_size: int = 10000,
        timeout: Optional[float] = None,
    ):
        """Stream cache entries as Arrow tables (high-throughput data plane).

        This method uses Arrow Flight for zero-copy streaming of large datasets.
        Significantly faster than list_entries() for bulk data export.

        Args:
            limit: Maximum entries to stream (None = all).
            offset: Offset for pagination (default: 0).
            context_filter: Filter by exact context match (optional).
            query_filter: Filter by query text substring (optional).
            include_embeddings: Include embedding vectors (default: False).
            include_results: Include cached results (default: True).
            batch_size: Rows per batch (default: 10000).
            timeout: RPC timeout in seconds (default: client timeout).

        Yields:
            pyarrow.RecordBatch objects containing cache entries.

        Example:
            >>> # Stream all entries in batches
            >>> for batch in client.stream_entries_arrow(limit=10000):
            ...     print(f"Batch: {len(batch)} rows")
            ...     # Convert to pandas: df = batch.to_pandas()
            ...     # Convert to polars: df = pl.from_arrow(batch)
            ...
            >>> # Stream with filtering
            >>> for batch in client.stream_entries_arrow(
            ...     context_filter={"model": "gpt-4"},
            ...     include_embeddings=True
            ... ):
            ...     # Process batch
            ...     pass
        """
        try:
            import pyarrow.flight as flight
        except ImportError:
            raise ImportError(
                "pyarrow is required for Flight streaming. "
                "Install with: pip install reminiscence[grpc]"
            )

        # Get Flight ticket
        params = {
            "offset": offset,
            "batch_size": batch_size,
            "include_embeddings": include_embeddings,
            "include_results": include_results,
        }

        if limit is not None:
            params["limit"] = limit

        if context_filter is not None:
            params["context_filter"] = context_filter

        if query_filter is not None:
            params["query_filter"] = query_filter

        ticket_info = self.get_flight_ticket(
            operation="list_entries",
            parameters=params,
            timeout=timeout,
        )

        # Connect to Flight server
        flight_client = self.get_flight_client(
            flight_endpoint=ticket_info["flight_endpoint"],
            timeout=timeout,
        )

        # Stream data
        ticket = flight.Ticket(ticket_info["ticket"])
        reader = flight_client.do_get(ticket)

        # Yield batches
        for batch in reader:
            yield batch.data

    def get_entries_arrow_table(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        context_filter: Optional[Dict[str, Any]] = None,
        query_filter: Optional[str] = None,
        include_embeddings: bool = False,
        include_results: bool = True,
        batch_size: int = 10000,
        timeout: Optional[float] = None,
    ):
        """Get cache entries as a single Arrow table (high-throughput data plane).

        This method uses Arrow Flight for zero-copy streaming and collects
        all batches into a single table. Use stream_entries_arrow() for
        large datasets that don't fit in memory.

        Args:
            limit: Maximum entries to fetch (None = all).
            offset: Offset for pagination (default: 0).
            context_filter: Filter by exact context match (optional).
            query_filter: Filter by query text substring (optional).
            include_embeddings: Include embedding vectors (default: False).
            include_results: Include cached results (default: True).
            batch_size: Rows per batch for streaming (default: 10000).
            timeout: RPC timeout in seconds (default: client timeout).

        Returns:
            pyarrow.Table with cache entries.

        Example:
            >>> # Get table and convert to pandas
            >>> table = client.get_entries_arrow_table(limit=1000)
            >>> df = table.to_pandas()
            >>> print(df.head())
            >>>
            >>> # Get table and convert to polars
            >>> import polars as pl
            >>> table = client.get_entries_arrow_table()
            >>> df = pl.from_arrow(table)
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "pyarrow is required for Flight operations. "
                "Install with: pip install reminiscence[grpc]"
            )

        # Stream batches and collect into table
        batches = list(
            self.stream_entries_arrow(
                limit=limit,
                offset=offset,
                context_filter=context_filter,
                query_filter=query_filter,
                include_embeddings=include_embeddings,
                include_results=include_results,
                batch_size=batch_size,
                timeout=timeout,
            )
        )

        if not batches:
            # Return empty table with schema
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("query_text", pa.string()),
                    pa.field("context", pa.string()),
                    pa.field("timestamp", pa.float64()),
                    pa.field("table_name", pa.string()),
                ]
            )
            return pa.Table.from_batches([], schema=schema)

        return pa.Table.from_batches(batches)

    def __repr__(self) -> str:
        """String representation."""
        return f"ReminiscenceClient(address={self.address!r})"
