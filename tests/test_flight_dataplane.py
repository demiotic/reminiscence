"""Tests for Arrow Flight data plane server and streaming.

These tests verify that the Flight server works correctly for high-throughput
bulk data operations.
"""

import time
from typing import Any, Dict

import pytest

from reminiscence import Reminiscence, ReminiscenceConfig
from reminiscence.types import MultiModalInput

# Skip all tests if Flight dependencies not installed
pytest.importorskip("grpc")
pytest.importorskip("pyarrow")
pytest.importorskip("reminiscence.api.server")
pytest.importorskip("reminiscence.api.client")
pytest.importorskip("reminiscence.api.flight_server")

import pyarrow as pa  # noqa: E402
import pyarrow.flight as flight  # noqa: E402

from reminiscence.api.client import ReminiscenceClient  # noqa: E402
from reminiscence.api.server import create_server  # noqa: E402


@pytest.fixture(scope="module")
def dual_plane_ports():
    """Ports for control and data plane servers."""
    return {
        "grpc": 50061,  # Control plane port
        "flight": 50062,  # Data plane port
    }


@pytest.fixture(scope="module")
def dual_plane_server(shared_embedder, dual_plane_ports):
    """Start both gRPC control plane and Flight data plane for testing."""
    config = ReminiscenceConfig(
        db_uri="memory://",
        log_level="WARNING",
        enable_metrics=True,
    )

    cache = Reminiscence(config, embedder=shared_embedder)

    # Populate with test data
    for i in range(100):
        query = MultiModalInput(text=f"Test query {i}")
        context = {"model": "gpt-4", "batch": str(i // 10)}
        cache.store(query, context, f"Result for query {i}")

    # Create server with both planes enabled
    server = create_server(
        cache=cache,
        port=dual_plane_ports["grpc"],
        max_workers=5,
        enable_reflection=True,
        enable_flight=True,
        flight_port=dual_plane_ports["flight"],
    )
    server.start()

    yield cache, server

    # Cleanup
    server.stop(grace=1.0)
    cache.backend.clear()


@pytest.fixture(scope="module")
def grpc_client(dual_plane_server, dual_plane_ports):
    """Create gRPC client connected to control plane."""
    _, _ = dual_plane_server  # Ensure server is started

    # Give server a moment to fully start
    time.sleep(0.2)

    client = ReminiscenceClient(
        f"localhost:{dual_plane_ports['grpc']}",
        timeout=5.0,
    )

    yield client

    client.close()


@pytest.fixture(scope="module")
def flight_client(dual_plane_server, dual_plane_ports):
    """Create Arrow Flight client connected to data plane."""
    _, _ = dual_plane_server  # Ensure server is started

    time.sleep(0.2)  # Let Flight server fully start

    location = f"grpc://localhost:{dual_plane_ports['flight']}"
    client = flight.FlightClient(location)

    yield client


# =============================================================================
# Server Lifecycle Tests
# =============================================================================


def test_flight_server_creation(shared_embedder, dual_plane_ports):
    """Test Flight server creation and shutdown."""
    config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
    cache = Reminiscence(config, embedder=shared_embedder)

    server = create_server(
        cache=cache,
        port=dual_plane_ports["grpc"] + 10,
        enable_flight=True,
        flight_port=dual_plane_ports["flight"] + 10,
    )
    assert server is not None
    assert hasattr(server, "flight_server")

    server.start()
    time.sleep(0.1)  # Let server start
    server.stop(grace=0.5)

    cache.backend.clear()


def test_flight_server_capabilities(grpc_client):
    """Test that GetCapabilities exposes Flight endpoint."""
    caps = grpc_client.get_capabilities()

    assert "flight_enabled" in caps
    assert caps["flight_enabled"] is True
    assert "flight_endpoint" in caps
    assert caps["flight_endpoint"] is not None
    assert "grpc://" in caps["flight_endpoint"]
    assert "arrow_flight_streaming" in caps["supported_features"]


# =============================================================================
# Flight Ticket Tests
# =============================================================================


def test_get_flight_ticket(grpc_client):
    """Test getting Flight ticket from control plane."""
    ticket_info = grpc_client.get_flight_ticket(
        operation="list_entries",
        parameters={"limit": 50, "include_embeddings": False},
    )

    assert "ticket" in ticket_info
    assert "flight_endpoint" in ticket_info
    assert "estimated_rows" in ticket_info
    assert "estimated_bytes" in ticket_info

    assert ticket_info["ticket"] is not None
    assert isinstance(ticket_info["ticket"], bytes)
    assert ticket_info["flight_endpoint"] is not None


def test_get_flight_ticket_with_filtering(grpc_client):
    """Test getting Flight ticket with filter parameters."""
    ticket_info = grpc_client.get_flight_ticket(
        operation="list_entries",
        parameters={
            "limit": 10,
            "context_filter": {"model": "gpt-4"},
            "include_results": True,
        },
    )

    assert ticket_info["ticket"] is not None


# =============================================================================
# Direct Flight Server Tests
# =============================================================================


def test_flight_list_flights(flight_client):
    """Test listing available flights."""
    flights = list(flight_client.list_flights())

    # Should have at least one flight (cache_entries)
    assert len(flights) >= 1


def test_flight_get_flight_info(flight_client):
    """Test getting flight info."""
    descriptor = flight.FlightDescriptor.for_path("cache_entries")
    info = flight_client.get_flight_info(descriptor)

    assert info is not None
    assert info.total_records >= 0
    assert len(info.endpoints) >= 1


def test_flight_stream_entries(flight_client, dual_plane_server):
    """Test streaming entries via Flight."""
    cache, _ = dual_plane_server

    # Get Flight info
    descriptor = flight.FlightDescriptor.for_path("cache_entries")
    info = flight_client.get_flight_info(descriptor)

    # Stream data using ticket
    ticket = info.endpoints[0].ticket
    reader = flight_client.do_get(ticket)

    total_rows = 0
    batch_count = 0

    for batch in reader:
        batch_count += 1
        num_rows = len(batch.data)
        total_rows += num_rows

        # Verify batch structure
        assert "id" in batch.data.schema.names
        assert "query_text" in batch.data.schema.names
        assert "context" in batch.data.schema.names
        assert "timestamp" in batch.data.schema.names
        assert "table_name" in batch.data.schema.names

    # Should have streamed all 100 test entries
    assert total_rows == 100
    assert batch_count >= 1


# =============================================================================
# Client Integration Tests
# =============================================================================


def test_client_get_flight_client(grpc_client):
    """Test getting Flight client from gRPC client."""
    flight_client = grpc_client.get_flight_client()

    assert flight_client is not None
    assert isinstance(flight_client, flight.FlightClient)


def test_client_stream_entries_arrow(grpc_client):
    """Test streaming entries using client method."""
    total_rows = 0
    batch_count = 0

    for batch in grpc_client.stream_entries_arrow(limit=50):
        batch_count += 1
        num_rows = len(batch)
        total_rows += num_rows

        # Verify batch is Arrow RecordBatch
        assert isinstance(batch, pa.RecordBatch)

        # Verify schema
        assert "id" in batch.schema.names
        assert "query_text" in batch.schema.names
        assert "context" in batch.schema.names

    # Should have streamed 50 entries (due to limit)
    assert total_rows == 50
    assert batch_count >= 1


def test_client_stream_with_filtering(grpc_client):
    """Test streaming with context filter."""
    total_rows = 0

    for batch in grpc_client.stream_entries_arrow(
        context_filter={"model": "gpt-4", "batch": "0"},
        include_results=True,
    ):
        total_rows += len(batch)

        # Verify all entries match filter
        for i in range(len(batch)):
            context_json = batch["context"][i].as_py()
            import json

            context = json.loads(context_json)
            assert context["model"] == "gpt-4"
            assert context["batch"] == "0"

    # Batch 0 has queries 0-9 (10 entries)
    assert total_rows == 10


def test_client_stream_with_query_filter(grpc_client):
    """Test streaming with query text filter."""
    total_rows = 0

    for batch in grpc_client.stream_entries_arrow(
        query_filter="query 5",  # Matches "Test query 5", "Test query 50-59"
        limit=100,
    ):
        total_rows += len(batch)

        # Verify all entries contain "query 5" in text
        for i in range(len(batch)):
            query_text = batch["query_text"][i].as_py()
            assert "query 5" in query_text.lower()

    # Should match: "Test query 5", "Test query 50-59" (11 entries)
    assert total_rows >= 11


def test_client_get_entries_arrow_table(grpc_client):
    """Test getting entries as Arrow table."""
    table = grpc_client.get_entries_arrow_table(limit=25)

    assert isinstance(table, pa.Table)
    assert len(table) == 25

    # Verify schema
    assert "id" in table.schema.names
    assert "query_text" in table.schema.names
    assert "context" in table.schema.names
    assert "timestamp" in table.schema.names
    assert "table_name" in table.schema.names

    # Verify data
    assert table["query_text"][0].as_py() is not None


def test_client_stream_without_results(grpc_client):
    """Test streaming without including results (reduces bandwidth)."""
    for batch in grpc_client.stream_entries_arrow(
        limit=10,
        include_results=False,
    ):
        # Results column should not be present
        assert "result" not in batch.schema.names or all(
            batch["result"][i].as_py() is None for i in range(len(batch))
        )


def test_client_stream_with_embeddings(grpc_client):
    """Test streaming with embeddings included."""
    for batch in grpc_client.stream_entries_arrow(
        limit=5,
        include_embeddings=True,
    ):
        # Embeddings column should be present
        if "embedding" in batch.schema.names:
            # Check that embeddings are lists of floats
            for i in range(len(batch)):
                embedding = batch["embedding"][i].as_py()
                if embedding is not None:
                    assert isinstance(embedding, list)
                    assert len(embedding) > 0
                    assert isinstance(embedding[0], float)


def test_client_stream_pagination(grpc_client):
    """Test pagination with offset."""
    # Get first page
    page1_batches = list(
        grpc_client.stream_entries_arrow(limit=10, offset=0)
    )
    page1_ids = set()
    for batch in page1_batches:
        for i in range(len(batch)):
            page1_ids.add(batch["id"][i].as_py())

    # Get second page
    page2_batches = list(
        grpc_client.stream_entries_arrow(limit=10, offset=10)
    )
    page2_ids = set()
    for batch in page2_batches:
        for i in range(len(batch)):
            page2_ids.add(batch["id"][i].as_py())

    # Pages should not overlap
    assert len(page1_ids & page2_ids) == 0


# =============================================================================
# Integration with pandas/polars
# =============================================================================


def test_flight_to_pandas(grpc_client):
    """Test converting Flight table to pandas DataFrame."""
    pytest.importorskip("pandas")
    import pandas as pd

    table = grpc_client.get_entries_arrow_table(limit=20)
    df = table.to_pandas()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert "query_text" in df.columns
    assert "context" in df.columns
    assert "timestamp" in df.columns


def test_flight_to_polars(grpc_client):
    """Test converting Flight table to polars DataFrame."""
    polars = pytest.importorskip("polars")

    table = grpc_client.get_entries_arrow_table(limit=20)
    df = polars.from_arrow(table)

    assert len(df) == 20
    assert "query_text" in df.columns
    assert "context" in df.columns


# =============================================================================
# Performance Tests
# =============================================================================


def test_flight_streaming_performance(grpc_client):
    """Test that Flight streaming handles large batches efficiently."""
    start_time = time.time()

    total_rows = 0
    for batch in grpc_client.stream_entries_arrow(
        limit=100,
        batch_size=10000,
    ):
        total_rows += len(batch)

    elapsed = time.time() - start_time

    assert total_rows == 100
    # Streaming should be fast (< 1 second for 100 entries)
    assert elapsed < 1.0


# =============================================================================
# Error Handling
# =============================================================================


def test_flight_client_without_server(dual_plane_ports):
    """Test Flight client behavior when server is not available."""
    # Try to connect to non-existent Flight server
    location = f"grpc://localhost:{dual_plane_ports['flight'] + 100}"
    client = flight.FlightClient(location)

    # Should raise error when trying to list flights
    with pytest.raises(Exception):
        list(client.list_flights())


def test_get_flight_ticket_without_flight_enabled(shared_embedder):
    """Test getting Flight ticket when Flight is not enabled."""
    config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
    cache = Reminiscence(config, embedder=shared_embedder)

    # Create server WITHOUT Flight enabled
    server = create_server(
        cache=cache,
        port=50071,
        enable_flight=False,
    )
    server.start()

    try:
        time.sleep(0.1)
        client = ReminiscenceClient("localhost:50071")

        # Check capabilities shows Flight disabled
        caps = client.get_capabilities()
        assert caps["flight_enabled"] is False
        assert caps["flight_endpoint"] is None

        # Trying to get Flight ticket should fail
        with pytest.raises(Exception):  # gRPC error
            client.get_flight_ticket()

        client.close()
    finally:
        server.stop(grace=0.5)
        cache.backend.clear()


# =============================================================================
# End-to-End Dual-Plane Workflow
# =============================================================================


def test_dual_plane_end_to_end(grpc_client):
    """Test complete dual-plane workflow."""
    # 1. Check capabilities (control plane)
    caps = grpc_client.get_capabilities()
    assert caps["flight_enabled"] is True
    assert caps["flight_endpoint"] is not None

    # 2. Perform lookups via control plane
    result = grpc_client.lookup(
        MultiModalInput(text="Test query 42"),
        {"model": "gpt-4", "batch": "4"},
    )
    assert result.is_hit is True

    # 3. Get stats via control plane
    stats = grpc_client.get_stats()
    assert stats["cache_entries"] >= 100

    # 4. Get Flight ticket from control plane
    ticket_info = grpc_client.get_flight_ticket(
        operation="list_entries",
        parameters={"limit": 50},
    )
    assert ticket_info["ticket"] is not None

    # 5. Stream bulk data via data plane
    total_rows = 0
    for batch in grpc_client.stream_entries_arrow(limit=50):
        total_rows += len(batch)

    assert total_rows == 50

    # 6. Verify control plane and data plane see same data
    # Control plane entry count
    control_plane_count = stats["cache_entries"]

    # Data plane entry count
    data_plane_count = 0
    for batch in grpc_client.stream_entries_arrow():
        data_plane_count += len(batch)

    assert control_plane_count == data_plane_count
