"""gRPC server implementation for Reminiscence semantic cache.

This module provides a gRPC server that exposes Reminiscence operations
over the network, enabling remote access to the semantic cache.
"""

from __future__ import annotations

import json
from concurrent import futures
from typing import Any, Dict, Optional

import grpc
from grpc_reflection.v1alpha import reflection

from ..config import ReminiscenceConfig
from ..core import Reminiscence
from ..types import LookupRequest as PyLookupRequest
from ..types import MultiModalInput, QueryMode
from ..types import StoreRequest as PyStoreRequest
from ..utils.logging import get_logger
from . import reminiscence_pb2 as pb2
from . import reminiscence_pb2_grpc as pb2_grpc

logger = get_logger(__name__)


# =============================================================================
# Type Converters (Protobuf <-> Python)
# =============================================================================


def proto_to_query_mode(mode: pb2.QueryMode) -> QueryMode:
    """Convert protobuf QueryMode to Python QueryMode."""
    mapping = {
        pb2.QUERY_MODE_AUTO: QueryMode.AUTO,
        pb2.QUERY_MODE_SEMANTIC: QueryMode.SEMANTIC,
        pb2.QUERY_MODE_EXACT: QueryMode.EXACT,
        pb2.QUERY_MODE_UNSPECIFIED: QueryMode.AUTO,  # Default
    }
    return mapping.get(mode, QueryMode.AUTO)


def proto_to_multimodal_input(proto: pb2.MultiModalInput) -> MultiModalInput:
    """Convert protobuf MultiModalInput to Python MultiModalInput."""
    return MultiModalInput(
        text=proto.text if proto.HasField("text") else None,
        image=proto.image if proto.HasField("image") else None,
        video=proto.video if proto.HasField("video") else None,
        audio=proto.audio if proto.HasField("audio") else None,
        metadata=dict(proto.metadata) if proto.metadata else None,
    )


def proto_to_context(proto: Optional[pb2.Context]) -> Dict[str, Any]:
    """Convert protobuf Context to Python dict."""
    if proto is None:
        return {}
    return dict(proto.values)


def context_to_proto(context: Dict[str, Any]) -> pb2.Context:
    """Convert Python dict to protobuf Context."""
    return pb2.Context(values={str(k): str(v) for k, v in context.items()})


def serialize_result(result: Any) -> pb2.SerializedResult:
    """Serialize Python result to protobuf SerializedResult."""
    json_bytes = json.dumps(result).encode("utf-8")
    return pb2.SerializedResult(json_data=json_bytes)


def deserialize_result(proto: pb2.SerializedResult) -> Any:
    """Deserialize protobuf SerializedResult to Python object."""
    return json.loads(proto.json_data.decode("utf-8"))


# =============================================================================
# gRPC Service Implementation
# =============================================================================


class ReminiscenceServicer(pb2_grpc.ReminiscenceServiceServicer):
    """gRPC servicer implementation for Reminiscence semantic cache."""

    def __init__(self, cache: Reminiscence):
        """Initialize servicer with a Reminiscence instance.

        Args:
            cache: Reminiscence instance to serve over gRPC.
        """
        self.cache = cache
        logger.info(
            "grpc_servicer_initialized",
            entries=cache.backend.count(),
            model=cache.config.model_name,
            flight_enabled=cache.flight_server is not None,
        )

    def Lookup(
        self, request: pb2.LookupRequest, context: grpc.ServicerContext
    ) -> pb2.LookupResponse:
        """Perform cache lookup."""
        try:
            query = proto_to_multimodal_input(request.query)
            ctx = (
                proto_to_context(request.context) if request.HasField("context") else {}
            )
            threshold = (
                request.similarity_threshold
                if request.HasField("similarity_threshold")
                else None
            )
            mode = proto_to_query_mode(request.mode)

            result = self.cache.lookup(
                query=query,
                context=ctx,
                similarity_threshold=threshold,
                mode=mode,
                track_metrics=request.track_metrics,
            )

            response = pb2.LookupResponse(hit=result.hit)

            if result.is_hit:
                response.result.CopyFrom(serialize_result(result.result))
                if result.similarity is not None:
                    response.similarity = result.similarity
                if result.matched_query:
                    response.matched_query = result.matched_query
                if result.age_seconds is not None:
                    response.age_seconds = result.age_seconds
                if result.entry_id:
                    response.entry_id = result.entry_id
                if result.context:
                    response.context.CopyFrom(context_to_proto(result.context))
                if result.ttl_remaining is not None:
                    response.ttl_remaining = result.ttl_remaining

            return response

        except Exception as e:
            logger.error("lookup_failed", error=str(e), exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Lookup failed: {str(e)}")

    def LookupBatch(
        self, request: pb2.LookupBatchRequest, context: grpc.ServicerContext
    ) -> pb2.LookupBatchResponse:
        """Perform batch cache lookup."""
        try:
            py_requests = []
            for req in request.requests:
                query = proto_to_multimodal_input(req.query)
                ctx = proto_to_context(req.context) if req.HasField("context") else {}
                threshold = (
                    req.similarity_threshold
                    if req.HasField("similarity_threshold")
                    else None
                )
                mode = proto_to_query_mode(req.mode)

                py_requests.append(
                    PyLookupRequest(
                        query=query,
                        context=ctx,
                        similarity_threshold=threshold,
                        mode=mode,
                    )
                )

            results = self.cache.lookup_batch(
                py_requests, track_metrics=request.track_metrics
            )

            responses = []
            for result in results:
                response = pb2.LookupResponse(hit=result.hit)

                if result.is_hit:
                    response.result.CopyFrom(serialize_result(result.result))
                    if result.similarity is not None:
                        response.similarity = result.similarity
                    if result.matched_query:
                        response.matched_query = result.matched_query
                    if result.age_seconds is not None:
                        response.age_seconds = result.age_seconds
                    if result.entry_id:
                        response.entry_id = result.entry_id
                    if result.context:
                        response.context.CopyFrom(context_to_proto(result.context))
                    if result.ttl_remaining is not None:
                        response.ttl_remaining = result.ttl_remaining

                responses.append(response)

            return pb2.LookupBatchResponse(responses=responses)

        except Exception as e:
            logger.error("lookup_batch_failed", error=str(e), exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Batch lookup failed: {str(e)}")

    def Store(
        self, request: pb2.StoreRequest, context: grpc.ServicerContext
    ) -> pb2.StoreResponse:
        """Store result in cache."""
        try:
            query = proto_to_multimodal_input(request.query)
            ctx = proto_to_context(request.context)
            result = deserialize_result(request.result)
            metadata = dict(request.metadata) if request.metadata else None
            ttl = request.ttl_seconds if request.HasField("ttl_seconds") else None
            threshold = (
                request.context_threshold
                if request.HasField("context_threshold")
                else None
            )
            mode = proto_to_query_mode(request.mode)

            self.cache.store(
                query=query,
                context=ctx,
                result=result,
                metadata=metadata,
                ttl_seconds=ttl,
                context_threshold=threshold,
                allow_errors=request.allow_errors,
                mode=mode,
            )

            return pb2.StoreResponse(success=True)

        except Exception as e:
            logger.error("store_failed", error=str(e), exc_info=True)
            return pb2.StoreResponse(success=False, error=str(e))

    def StoreBatch(
        self, request: pb2.StoreBatchRequest, context: grpc.ServicerContext
    ) -> pb2.StoreBatchResponse:
        """Store multiple results in batch."""
        try:
            py_requests = []
            for req in request.requests:
                query = proto_to_multimodal_input(req.query)
                ctx = proto_to_context(req.context)
                result = deserialize_result(req.result)
                metadata = dict(req.metadata) if req.metadata else None
                ttl = req.ttl_seconds if req.HasField("ttl_seconds") else None
                threshold = (
                    req.context_threshold if req.HasField("context_threshold") else None
                )

                py_requests.append(
                    PyStoreRequest(
                        query=query,
                        context=ctx,
                        result=result,
                        metadata=metadata,
                        ttl_seconds=ttl,
                        context_threshold=threshold,
                    )
                )

            mode = proto_to_query_mode(request.mode)
            self.cache.store_batch(
                py_requests, allow_errors=request.allow_errors, mode=mode
            )

            # Return success for all requests if batch succeeded
            responses = [pb2.StoreResponse(success=True) for _ in py_requests]
            return pb2.StoreBatchResponse(responses=responses)

        except Exception as e:
            logger.error("store_batch_failed", error=str(e), exc_info=True)
            # Return errors for all requests
            error_responses = [
                pb2.StoreResponse(success=False, error=str(e)) for _ in request.requests
            ]
            return pb2.StoreBatchResponse(responses=error_responses)

    def CheckAvailability(
        self, request: pb2.CheckAvailabilityRequest, context: grpc.ServicerContext
    ) -> pb2.CheckAvailabilityResponse:
        """Check if cache entry is available without retrieving data."""
        try:
            query = proto_to_multimodal_input(request.query)
            ctx = proto_to_context(request.context)
            threshold = (
                request.similarity_threshold
                if request.HasField("similarity_threshold")
                else None
            )
            mode = proto_to_query_mode(request.mode)

            availability = self.cache.check_availability(
                query=query,
                context=ctx,
                similarity_threshold=threshold,
                mode=mode,
            )

            response = pb2.CheckAvailabilityResponse(
                available=availability.available, is_fresh=availability.is_fresh
            )

            if availability.age_seconds is not None:
                response.age_seconds = availability.age_seconds
            if availability.ttl_remaining_seconds is not None:
                response.ttl_remaining_seconds = availability.ttl_remaining_seconds
            if availability.similarity is not None:
                response.similarity = availability.similarity

            return response

        except Exception as e:
            logger.error("check_availability_failed", error=str(e), exc_info=True)
            context.abort(
                grpc.StatusCode.INTERNAL, f"Availability check failed: {str(e)}"
            )

    def Invalidate(
        self, request: pb2.InvalidateRequest, context: grpc.ServicerContext
    ) -> pb2.InvalidateResponse:
        """Invalidate cache entries by criteria."""
        try:
            query = (
                proto_to_multimodal_input(request.query)
                if request.HasField("query")
                else None
            )
            ctx = (
                proto_to_context(request.context)
                if request.HasField("context")
                else None
            )
            older_than = (
                request.older_than_seconds
                if request.HasField("older_than_seconds")
                else None
            )

            count = self.cache.invalidate(
                query=query, context=ctx, older_than_seconds=older_than
            )

            return pb2.InvalidateResponse(invalidated_count=count)

        except Exception as e:
            logger.error("invalidate_failed", error=str(e), exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Invalidation failed: {str(e)}")

    def CleanupExpired(
        self, request: pb2.CleanupExpiredRequest, context: grpc.ServicerContext
    ) -> pb2.CleanupExpiredResponse:
        """Clean up expired cache entries."""
        try:
            count = self.cache.cleanup_expired()
            return pb2.CleanupExpiredResponse(deleted_count=count)

        except Exception as e:
            logger.error("cleanup_expired_failed", error=str(e), exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Cleanup failed: {str(e)}")

    def Clear(
        self, request: pb2.ClearRequest, context: grpc.ServicerContext
    ) -> pb2.ClearResponse:
        """Clear all cache entries."""
        try:
            self.cache.clear()
            return pb2.ClearResponse(success=True)

        except Exception as e:
            logger.error("clear_failed", error=str(e), exc_info=True)
            return pb2.ClearResponse(success=False)

    def CreateIndex(
        self, request: pb2.CreateIndexRequest, context: grpc.ServicerContext
    ) -> pb2.CreateIndexResponse:
        """Create vector index for fast searches."""
        try:
            num_partitions = request.num_partitions or 256
            num_subvectors = (
                request.num_subvectors if request.HasField("num_subvectors") else None
            )

            self.cache.create_index(
                num_partitions=num_partitions, num_subvectors=num_subvectors
            )

            return pb2.CreateIndexResponse(success=True)

        except Exception as e:
            logger.error("create_index_failed", error=str(e), exc_info=True)
            return pb2.CreateIndexResponse(success=False, error=str(e))

    def GetStats(
        self, request: pb2.GetStatsRequest, context: grpc.ServicerContext
    ) -> pb2.GetStatsResponse:
        """Get cache statistics."""
        try:
            stats = self.cache.get_stats()

            response = pb2.GetStatsResponse(
                cache_entries=stats["cache_entries"],
                total_entries=stats["total_entries"],
                max_entries=stats["max_entries"]
                if stats["max_entries"] is not None
                else 0,
                eviction_policy=stats["eviction_policy"],
                threshold=stats["threshold"],
                embedding_dim=stats["embedding_dim"],
                model=stats["model"] or "",
                storage=self.cache.config.db_uri,  # Use DB URI string instead of dict
                index_created=stats["index_created"],
            )

            if stats["ttl_seconds"] is not None:
                response.ttl_seconds = stats["ttl_seconds"]

            # Add metrics if available
            if "hits" in stats:
                response.hits = stats["hits"]
            if "misses" in stats:
                response.misses = stats["misses"]
            if "hit_rate" in stats:
                # Convert percentage string "X.XX%" to float
                hit_rate_str = stats["hit_rate"]
                if isinstance(hit_rate_str, str) and hit_rate_str.endswith("%"):
                    response.hit_rate = float(hit_rate_str.rstrip("%")) / 100.0
                else:
                    response.hit_rate = float(hit_rate_str) if hit_rate_str else 0.0

            # Note: avg_lookup_latency_ms and avg_store_latency_ms don't exist
            # in stats dict. The actual fields are nested, so we'll skip them.

            if "errors" in stats and isinstance(stats["errors"], dict):
                response.lookup_errors = stats["errors"].get("lookup", 0)
                response.store_errors = stats["errors"].get("store", 0)

            if "eviction" in stats and isinstance(stats["eviction"], dict):
                response.evictions = stats["eviction"].get("total_evictions", 0)

            # Add scheduler stats if available
            if "schedulers" in stats and stats["schedulers"]:
                for name, sched_stats in stats["schedulers"].items():
                    # Directly assign fields to map entry (protobuf maps)
                    scheduler_entry = response.schedulers[name]
                    scheduler_entry.running = sched_stats["running"]
                    scheduler_entry.total_runs = sched_stats["total_runs"]
                    scheduler_entry.errors = sched_stats["errors"]

                    if sched_stats.get("last_run_timestamp") is not None:
                        scheduler_entry.last_run_timestamp = sched_stats[
                            "last_run_timestamp"
                        ]
                    if sched_stats.get("next_run_timestamp") is not None:
                        scheduler_entry.next_run_timestamp = sched_stats[
                            "next_run_timestamp"
                        ]

            return response

        except Exception as e:
            logger.error("get_stats_failed", error=str(e), exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Get stats failed: {str(e)}")

    def GetIndexStats(
        self, request: pb2.GetIndexStatsRequest, context: grpc.ServicerContext
    ) -> pb2.GetIndexStatsResponse:
        """Get vector index statistics."""
        try:
            stats = self.cache.get_index_stats()

            return pb2.GetIndexStatsResponse(
                has_index=stats["has_index"],
                total_entries=stats["total_entries"],
                note=stats["note"],
            )

        except Exception as e:
            logger.error("get_index_stats_failed", error=str(e), exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Get index stats failed: {str(e)}")

    def ListEntries(
        self, request: pb2.ListEntriesRequest, context: grpc.ServicerContext
    ) -> pb2.ListEntriesResponse:
        """List cache entries with pagination and filtering."""
        try:
            # Parse request parameters
            limit = min(request.limit if request.HasField("limit") else 100, 1000)
            offset = request.offset if request.HasField("offset") else 0
            include_embeddings = request.include_embeddings
            include_results = request.include_results

            # Get context filter
            context_filter = None
            if request.HasField("context_filter"):
                context_filter = proto_to_context(request.context_filter)

            # Get query filter
            query_filter = (
                request.query_filter if request.HasField("query_filter") else None
            )

            # Get sort parameters
            sort_field_map = {
                pb2.SORT_FIELD_UNSPECIFIED: "timestamp",
                pb2.SORT_FIELD_CREATED_AT: "timestamp",
                pb2.SORT_FIELD_LAST_ACCESSED: "timestamp",  # We'll use timestamp
                pb2.SORT_FIELD_ACCESS_COUNT: "timestamp",  # Not tracked yet
                pb2.SORT_FIELD_SIMILARITY: "timestamp",  # Not applicable for listing
            }
            sort_field = sort_field_map.get(request.sort_by, "timestamp")
            sort_descending = (
                request.sort_descending
            )  # Bool has no presence in proto3, defaults to False

            # Collect entries from both tables
            all_entries = []

            # Query semantic table
            try:
                semantic_table = self.cache.backend.semantic_table.to_arrow()
                for i in range(len(semantic_table)):
                    row = {
                        "id": semantic_table["id"][i].as_py(),
                        "query_text": semantic_table["query_text"][i].as_py(),
                        "context": semantic_table["context"][i].as_py(),
                        "context_hash": semantic_table["context_hash"][i].as_py(),
                        "result": semantic_table["result"][i].as_py(),
                        "result_type": semantic_table["result_type"][i].as_py(),
                        "timestamp": semantic_table["timestamp"][i].as_py(),
                        "metadata": semantic_table["metadata"][i].as_py(),
                        "embedding": semantic_table["embedding"][i].as_py()
                        if include_embeddings
                        else None,
                        "table": "semantic",
                    }
                    all_entries.append(row)
            except Exception as e:
                logger.warning("semantic_table_query_failed", error=str(e))

            # Query exact table
            try:
                exact_table = self.cache.backend.exact_table.to_arrow()
                for i in range(len(exact_table)):
                    row = {
                        "id": exact_table["id"][i].as_py(),
                        "query_text": exact_table["query_text"][i].as_py(),
                        "context": exact_table["context"][i].as_py(),
                        "context_hash": exact_table["context_hash"][i].as_py(),
                        "result": exact_table["result"][i].as_py(),
                        "result_type": exact_table["result_type"][i].as_py(),
                        "timestamp": exact_table["timestamp"][i].as_py(),
                        "metadata": exact_table["metadata"][i].as_py(),
                        "embedding": None,
                        "table": "exact",
                    }
                    all_entries.append(row)
            except Exception as e:
                logger.warning("exact_table_query_failed", error=str(e))

            # Apply filters
            filtered_entries = []
            for entry in all_entries:
                # Context filter (exact match)
                if context_filter is not None:
                    entry_context = json.loads(entry["context"])
                    if entry_context != context_filter:
                        continue

                # Query filter (substring match)
                if query_filter is not None:
                    if query_filter.lower() not in entry["query_text"].lower():
                        continue

                filtered_entries.append(entry)

            total_count = len(filtered_entries)

            # Sort entries
            filtered_entries.sort(
                key=lambda x: x.get(sort_field, 0), reverse=sort_descending
            )

            # Apply pagination
            paginated_entries = filtered_entries[offset : offset + limit]
            has_more = (offset + limit) < total_count

            # Convert to protobuf
            pb_entries = []
            for entry in paginated_entries:
                # Parse metadata
                metadata_dict = {}
                try:
                    metadata_str = entry["metadata"]
                    if metadata_str and metadata_str != "{}":
                        metadata_dict = json.loads(metadata_str)
                except Exception:
                    pass

                # Parse context
                context_dict = json.loads(entry["context"])

                # Create entry
                pb_entry = pb2.CacheEntry(
                    entry_id=entry["id"],
                    query=pb2.MultiModalInput(text=entry["query_text"]),
                    context=context_to_proto(context_dict),
                    metadata={str(k): str(v) for k, v in metadata_dict.items()},
                    created_at=entry["timestamp"],
                    last_accessed_at=entry[
                        "timestamp"
                    ],  # We don't track this separately yet
                    access_count=0,  # We don't track this yet
                    embedding_dim=self.cache.embedder.embedding_dim,
                )

                # Add result if requested
                if include_results:
                    try:
                        result_obj = self.cache.backend.serializer.deserialize(
                            entry["result"], entry["result_type"]
                        )
                        pb_entry.result.CopyFrom(serialize_result(result_obj))
                    except Exception as e:
                        logger.warning("result_deserialization_failed", error=str(e))

                # Add embedding if requested
                if include_embeddings and entry["embedding"] is not None:
                    import struct

                    embedding_bytes = struct.pack(
                        f"{len(entry['embedding'])}f", *entry["embedding"]
                    )
                    pb_entry.embedding = embedding_bytes

                # Add matched context key for debugging
                pb_entry.matched_context_key = entry["table"]

                pb_entries.append(pb_entry)

            return pb2.ListEntriesResponse(
                entries=pb_entries,
                total_count=total_count,
                has_more=has_more,
                returned_count=len(pb_entries),
            )

        except Exception as e:
            logger.error("list_entries_failed", error=str(e), exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"List entries failed: {str(e)}")

    def HealthCheck(
        self, request: pb2.HealthCheckRequest, context: grpc.ServicerContext
    ) -> pb2.HealthCheckResponse:
        """Perform health check on cache components."""
        try:
            health = self.cache.health_check()

            response = pb2.HealthCheckResponse(
                status=health["status"],
                timestamp=health["timestamp"],
            )

            # Add component checks
            for component, check in health["checks"].items():
                # Directly assign fields to map entry (protobuf maps)
                check_entry = response.checks[component]
                check_entry.ok = check["ok"]
                if check.get("error"):
                    check_entry.error = check["error"]
                if check.get("details"):
                    check_entry.details = check["details"]

            # Add metrics
            metrics = health["metrics"]
            proto_metrics = pb2.HealthMetrics(
                total_entries=metrics["total_entries"],
                recent_errors=pb2.ErrorCounts(
                    lookup=metrics["recent_errors"]["lookup"],
                    store=metrics["recent_errors"]["store"],
                ),
            )
            response.metrics.CopyFrom(proto_metrics)

            return response

        except Exception as e:
            logger.error("health_check_failed", error=str(e), exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Health check failed: {str(e)}")

    def GetFlightTicket(
        self, request: pb2.FlightTicketRequest, context: grpc.ServicerContext
    ) -> pb2.FlightTicketResponse:
        """Get Arrow Flight ticket for bulk data operations."""
        try:
            # Check actual Flight server status dynamically
            if self.cache.flight_server is None:
                context.abort(
                    grpc.StatusCode.UNAVAILABLE, "Arrow Flight server not enabled"
                )

            flight_location = (
                f"grpc://{self.cache.config.flight_host}:{self.cache.config.flight_port}"
            )

            operation = request.operation
            parameters = request.parameters if request.HasField("parameters") else "{}"

            # Build Flight command
            command = {
                "operation": operation,
            }

            # Parse and merge parameters
            try:
                params = json.loads(parameters)
                command.update(params)
            except json.JSONDecodeError as e:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, f"Invalid parameters JSON: {e}"
                )

            # Create ticket
            ticket_bytes = json.dumps(command).encode("utf-8")

            # Estimate data size (rough approximation)
            estimated_rows = -1
            estimated_bytes = -1

            if operation == "list_entries":
                total_entries = self.cache.backend.count()
                limit = command.get("limit", 1000)
                estimated_rows = min(total_entries, limit)
                # Rough estimate: ~1KB per entry
                estimated_bytes = estimated_rows * 1024

            logger.info(
                "flight_ticket_issued",
                operation=operation,
                estimated_rows=estimated_rows,
            )

            return pb2.FlightTicketResponse(
                ticket=ticket_bytes,
                flight_endpoint=flight_location,
                estimated_rows=estimated_rows,
                estimated_bytes=estimated_bytes,
            )

        except Exception as e:
            logger.error("get_flight_ticket_failed", error=str(e), exc_info=True)
            context.abort(
                grpc.StatusCode.INTERNAL, f"Get flight ticket failed: {str(e)}"
            )

    def GetCapabilities(
        self, request: pb2.GetCapabilitiesRequest, context: grpc.ServicerContext
    ) -> pb2.GetCapabilitiesResponse:
        """Get server capabilities including Flight status."""
        try:
            # Check actual Flight server status dynamically
            flight_enabled = self.cache.flight_server is not None
            flight_endpoint = None
            if flight_enabled:
                flight_endpoint = (
                    f"grpc://{self.cache.config.flight_host}:{self.cache.config.flight_port}"
                )

            response = pb2.GetCapabilitiesResponse(
                version="0.6.0",
                flight_enabled=flight_enabled,
                supported_features=[
                    "semantic_search",
                    "exact_match",
                    "hybrid_matching",
                    "multimodal",
                    "batch_operations",
                    "ttl",
                    "eviction_policies",
                    "metrics",
                    "health_checks",
                    "list_entries",
                ],
            )

            # Add Flight endpoint if available
            if flight_endpoint:
                response.flight_endpoint = flight_endpoint
                response.supported_features.append("arrow_flight_streaming")

            return response

        except Exception as e:
            logger.error("get_capabilities_failed", error=str(e), exc_info=True)
            context.abort(
                grpc.StatusCode.INTERNAL, f"Get capabilities failed: {str(e)}"
            )


# =============================================================================
# Server Factory
# =============================================================================


def create_server(
    cache: Optional[Reminiscence] = None,
    config: Optional[ReminiscenceConfig] = None,
    port: int = 8080,
    host: str = "127.0.0.1",
    max_workers: int = 10,
    enable_reflection: bool = True,
    enable_flight: bool = False,
    flight_port: int = 8081,
    flight_host: str = "127.0.0.1",
) -> grpc.Server:
    """Create and configure a gRPC server for Reminiscence.

    Args:
        cache: Existing Reminiscence instance (if None, creates from config).
        config: Configuration for Reminiscence (if cache not provided).
        port: Port to listen on (default: 8080).
        host: Host to bind to (default: 127.0.0.1, localhost only).
        max_workers: Maximum number of worker threads (default: 10).
        enable_reflection: Enable gRPC reflection for debugging (default: True).
        enable_flight: Enable Arrow Flight data plane (default: False).
        flight_port: Port for Flight server (default: 8081).
        flight_host: Host for Flight server (default: 127.0.0.1, localhost only).

    Returns:
        Configured gRPC server (not started). Access Flight server
        via server.flight_server if enabled.

    Example:
        >>> from reminiscence.api.server import create_server
        >>> server = create_server(port=8080, enable_flight=True, flight_port=8081)
        >>> server.start()
        >>> # Flight server auto-starts with gRPC server
        >>> # ... use servers ...
        >>> server.stop(grace=5.0)
    """
    if cache is None:
        if config is None:
            config = ReminiscenceConfig.load()
        cache = Reminiscence(config)

    # Create Flight server if enabled
    flight_server = None
    if enable_flight and cache.flight_server is None:
        # Create a new Flight server only if one doesn't exist
        try:
            from .flight_server import create_flight_server

            flight_location = f"grpc://{flight_host}:{flight_port}"
            flight_server = create_flight_server(
                cache, host=flight_host, port=flight_port
            )
            # Update cache's flight_server reference so dynamic checks work
            cache.flight_server = flight_server
            logger.info("flight_server_created", location=flight_location)
        except ImportError as e:
            logger.warning(
                "flight_server_unavailable",
                error=str(e),
                note="Install pyarrow to enable Flight support",
            )
    elif cache.flight_server is not None:
        # Use existing Flight server from cache
        flight_server = cache.flight_server
        flight_location = (
            f"grpc://{cache.config.flight_host}:{cache.config.flight_port}"
        )
        logger.info("using_existing_flight_server", location=flight_location)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    servicer = ReminiscenceServicer(cache)
    pb2_grpc.add_ReminiscenceServiceServicer_to_server(servicer, server)

    # Enable reflection for easier debugging
    if enable_reflection:
        service_names = (
            pb2.DESCRIPTOR.services_by_name["ReminiscenceService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)

    server.add_insecure_port(f"{host}:{port}")

    # Attach Flight server to gRPC server for lifecycle management
    if flight_server:
        server.flight_server = flight_server

        # Override start/stop to manage both servers
        original_start = server.start

        def start_both():
            original_start()
            # Start Flight server in background thread
            import threading

            flight_thread = threading.Thread(
                target=flight_server.serve, daemon=True, name="FlightServer"
            )
            flight_thread.start()
            logger.info("flight_server_started", location=flight_location)

        server.start = start_both

    logger.info(
        "grpc_server_created",
        port=port,
        max_workers=max_workers,
        flight_enabled=enable_flight,
    )

    return server


def serve(
    cache: Optional[Reminiscence] = None,
    config: Optional[ReminiscenceConfig] = None,
    port: int = 8080,
    host: str = "127.0.0.1",
    max_workers: int = 10,
) -> None:
    """Start gRPC server and block until terminated.

    Args:
        cache: Existing Reminiscence instance (if None, creates from config).
        config: Configuration for Reminiscence (if cache not provided).
        port: Port to listen on (default: 8080).
        host: Host to bind to (default: 127.0.0.1, localhost only).
        max_workers: Maximum number of worker threads (default: 10).

    Example:
        >>> from reminiscence.api.server import serve
        >>> serve(port=8080)  # Blocks until Ctrl+C on localhost
    """
    server = create_server(
        cache=cache, config=config, port=port, host=host, max_workers=max_workers
    )
    server.start()

    logger.info("grpc_server_started", port=port)
    print(f"Reminiscence gRPC server listening on port {port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("grpc_server_stopping")
        print("\nShutting down server...")
        server.stop(grace=5.0)
        logger.info("grpc_server_stopped")


if __name__ == "__main__":
    # Run server when executed directly
    serve()
