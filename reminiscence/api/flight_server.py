"""Arrow Flight server for high-throughput data plane operations.

This module provides an Arrow Flight server that runs alongside the gRPC
control plane, enabling efficient bulk data transfer for large-scale operations.

Architecture:
- gRPC (port 8080): Control plane for metadata and cache management
- Flight (port 8081): Data plane for bulk entry streaming and analytics

The Flight server exposes cache entries as Arrow tables for:
- Zero-copy data transfer
- Streaming large datasets without memory overhead
- High-performance analytics and bulk export
- Integration with Arrow-native tools (DuckDB, Polars, pandas)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.flight as flight

from ..core import Reminiscence
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReminiscenceFlightServer(flight.FlightServerBase):
    """Arrow Flight server for Reminiscence data plane.

    Provides high-throughput streaming of cache entries as Arrow tables.
    """

    def __init__(
        self,
        cache: Reminiscence,
        location: str = "grpc://0.0.0.0:8081",
        **kwargs,
    ):
        """Initialize Flight server.

        Args:
            cache: Reminiscence instance to serve.
            location: Flight server location (default: grpc://0.0.0.0:8081).
            **kwargs: Additional arguments for FlightServerBase.
        """
        super().__init__(location, **kwargs)
        self.cache = cache
        self.location = location

        logger.info(
            "flight_server_initialized",
            location=location,
            entries=cache.backend.count(),
        )

    def list_flights(self, context, criteria: bytes):
        """List available flights (datasets).

        Returns:
            Iterator of FlightInfo objects describing available datasets.
        """
        # Return metadata about available datasets
        flights = [
            flight.FlightInfo(
                schema=self._get_schema(),
                descriptor=flight.FlightDescriptor.for_path("cache_entries"),
                endpoints=[
                    flight.FlightEndpoint(
                        ticket=flight.Ticket(b"cache_entries"),
                        locations=[self.location],
                    )
                ],
                total_records=self.cache.backend.count(),
                total_bytes=-1,  # Unknown
            )
        ]

        for f in flights:
            yield f

    def get_flight_info(self, context, descriptor: flight.FlightDescriptor):
        """Get information about a specific flight.

        Args:
            context: Flight server context.
            descriptor: Flight descriptor identifying the dataset.

        Returns:
            FlightInfo describing the dataset.
        """
        if descriptor.path == [b"cache_entries"]:
            return flight.FlightInfo(
                schema=self._get_schema(),
                descriptor=descriptor,
                endpoints=[
                    flight.FlightEndpoint(
                        ticket=flight.Ticket(b"cache_entries"),
                        locations=[self.location],
                    )
                ],
                total_records=self.cache.backend.count(),
                total_bytes=-1,
            )

        raise flight.FlightNotAvailableError(f"Unknown flight: {descriptor.path}")

    def do_get(self, context, ticket: flight.Ticket):
        """Stream dataset to client.

        Args:
            context: Flight server context.
            ticket: Ticket identifying the dataset and parameters.

        Yields:
            RecordBatch objects containing cache entries.
        """
        # Parse ticket (contains request parameters as JSON)
        params = self._parse_ticket(ticket)

        logger.info(
            "flight_do_get_start",
            params=params,
            total_entries=self.cache.backend.count(),
        )

        start_time = time.time()

        # Get entries from both tables
        entries = self._collect_entries(params)

        # Convert to Arrow table
        table = self._entries_to_arrow(entries, params)

        elapsed = time.time() - start_time
        logger.info(
            "flight_do_get_complete",
            rows=len(table),
            elapsed_ms=int(elapsed * 1000),
        )

        # Stream table in batches
        reader = pa.RecordBatchReader.from_batches(
            table.schema,
            table.to_batches(max_chunksize=10000),
        )

        return flight.RecordBatchStream(reader)

    def _parse_ticket(self, ticket: flight.Ticket) -> Dict[str, Any]:
        """Parse ticket data into request parameters.

        Args:
            ticket: Flight ticket containing request parameters.

        Returns:
            Dictionary of request parameters.
        """
        try:
            if ticket.ticket == b"cache_entries":
                # Default parameters
                return {
                    "limit": None,  # No limit for Flight
                    "offset": 0,
                    "context_filter": None,
                    "query_filter": None,
                    "include_embeddings": False,
                    "include_results": True,
                }

            # Parse JSON ticket
            params = json.loads(ticket.ticket.decode("utf-8"))
            return params
        except Exception as e:
            logger.error("ticket_parse_failed", error=str(e))
            return {}

    def _collect_entries(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect entries from storage with filters.

        Args:
            params: Request parameters (filters, limits, etc.).

        Returns:
            List of entry dictionaries.
        """
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
                    "embedding": semantic_table["embedding"][i].as_py() if params.get("include_embeddings") else None,
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
        filtered = self._apply_filters(all_entries, params)

        # Apply limit/offset
        offset = params.get("offset", 0)
        limit = params.get("limit")

        if limit:
            return filtered[offset : offset + limit]
        else:
            return filtered[offset:]

    def _apply_filters(
        self,
        entries: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Apply filters to entry list.

        Args:
            entries: List of entry dictionaries.
            params: Filter parameters.

        Returns:
            Filtered entry list.
        """
        filtered = []

        context_filter = params.get("context_filter")
        query_filter = params.get("query_filter")

        for entry in entries:
            # Context filter (exact match)
            if context_filter:
                entry_context = json.loads(entry["context"])
                if entry_context != context_filter:
                    continue

            # Query filter (substring match)
            if query_filter:
                if query_filter.lower() not in entry["query_text"].lower():
                    continue

            filtered.append(entry)

        return filtered

    def _entries_to_arrow(
        self,
        entries: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> pa.Table:
        """Convert entries to Arrow table.

        Args:
            entries: List of entry dictionaries.
            params: Request parameters.

        Returns:
            PyArrow Table with cache entries.
        """
        if not entries:
            # Return empty table with schema
            return pa.Table.from_pydict(
                {
                    "id": [],
                    "query_text": [],
                    "context": [],
                    "timestamp": [],
                    "table_name": [],
                },
                schema=self._get_schema(),
            )

        # Build columns
        ids = []
        query_texts = []
        contexts = []
        timestamps = []
        table_names = []
        results = [] if params.get("include_results", True) else None
        result_types = [] if params.get("include_results", True) else None
        embeddings = [] if params.get("include_embeddings", False) else None

        for entry in entries:
            ids.append(entry["id"])
            query_texts.append(entry["query_text"])
            contexts.append(entry["context"])
            timestamps.append(entry["timestamp"])
            table_names.append(entry["table"])

            if results is not None:
                results.append(entry["result"])
                result_types.append(entry["result_type"])

            if embeddings is not None and entry["embedding"]:
                embeddings.append(entry["embedding"])
            elif embeddings is not None:
                embeddings.append(None)

        # Build dict for table
        data = {
            "id": ids,
            "query_text": query_texts,
            "context": contexts,
            "timestamp": timestamps,
            "table_name": table_names,
        }

        if results is not None:
            data["result"] = results
            data["result_type"] = result_types

        if embeddings is not None:
            data["embedding"] = embeddings

        return pa.Table.from_pydict(data)

    def _get_schema(self) -> pa.Schema:
        """Get Arrow schema for cache entries.

        Returns:
            PyArrow schema defining the table structure.
        """
        fields = [
            pa.field("id", pa.string()),
            pa.field("query_text", pa.string()),
            pa.field("context", pa.string()),  # JSON string
            pa.field("timestamp", pa.float64()),
            pa.field("table_name", pa.string()),  # 'semantic' or 'exact'
            pa.field("result", pa.string(), nullable=True),  # Optional
            pa.field("result_type", pa.string(), nullable=True),  # Optional
            pa.field("embedding", pa.list_(pa.float32()), nullable=True),  # Optional
        ]

        return pa.schema(fields)


def create_flight_server(
    cache: Reminiscence,
    port: int = 8081,
    host: str = "0.0.0.0",
) -> ReminiscenceFlightServer:
    """Create Arrow Flight server for data plane operations.

    Args:
        cache: Reminiscence instance to serve.
        port: Port to listen on (default: 8081).
        host: Host to bind to (default: 0.0.0.0).

    Returns:
        ReminiscenceFlightServer instance.

    Example:
        >>> from reminiscence import Reminiscence
        >>> from reminiscence.api.flight_server import create_flight_server
        >>>
        >>> cache = Reminiscence()
        >>> flight_server = create_flight_server(cache, port=8081)
        >>> flight_server.serve()  # Blocks until stopped
    """
    location = f"grpc://{host}:{port}"
    server = ReminiscenceFlightServer(cache=cache, location=location)

    logger.info("flight_server_created", location=location)

    return server
