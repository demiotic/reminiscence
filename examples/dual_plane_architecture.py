"""Example: Dual-Plane Architecture with gRPC Control and Arrow Flight Data Planes.

This example demonstrates the complete two-plane architecture:
- Control Plane (gRPC on port 8080): Metadata, operations, health checks
- Data Plane (Arrow Flight on port 8081): High-throughput bulk data streaming

Architecture:
    Client → gRPC (8080) → Get capabilities/tickets
           → Flight (8081) → Stream bulk data

Use Cases:
- Bulk export of cache entries
- Analytics on cache data
- High-performance data migration
- Real-time monitoring dashboards
"""

import json
import time

import pyarrow.flight as flight

from reminiscence import Reminiscence, ReminiscenceConfig
from reminiscence.api.client import ReminiscenceClient
from reminiscence.api.server import create_server
from reminiscence.types import MultiModalInput


def start_dual_plane_server():
    """Start both control and data plane servers."""
    print("=== Starting Dual-Plane Architecture ===\n")

    # Create cache
    config = ReminiscenceConfig(
        db_uri="memory://",
        max_entries=10000,
        enable_metrics=True,
    )
    cache = Reminiscence(config)

    # Populate with test data
    print("Populating cache with test data...")
    for i in range(100):
        cache.store(
            query=MultiModalInput(text=f"Test query {i}"),
            context={"model": "gpt-4", "batch": str(i // 10)},
            result=f"Result for query {i}",
        )
    print(f"✓ Added 100 entries to cache\n")

    # Create server with both planes enabled
    server = create_server(
        cache=cache,
        port=8080,  # Control plane
        enable_flight=True,
        flight_port=8081,  # Data plane
        enable_reflection=True,
    )

    server.start()

    print("✓ Servers started:")
    print("  - Control Plane (gRPC):    grpc://localhost:8080")
    print("  - Data Plane (Flight):     grpc://localhost:8081")
    print()

    return server, cache


def demo_control_plane():
    """Demonstrate control plane operations (gRPC)."""
    print("=== Control Plane (gRPC) Operations ===\n")

    client = ReminiscenceClient("localhost:8080")

    # 1. Get capabilities
    caps = client.get_capabilities()
    print("1. Server Capabilities:")
    print(f"   Version: {caps['version']}")
    print(f"   Flight Enabled: {caps['flight_enabled']}")
    print(f"   Flight Endpoint: {caps.get('flight_endpoint', 'N/A')}")
    print(f"   Features: {len(caps['supported_features'])} features")
    print()

    # 2. Lookup (metadata operation)
    result = client.lookup(
        query=MultiModalInput(text="Test query 42"),
        context={"model": "gpt-4", "batch": "4"}
    )
    print(f"2. Lookup via Control Plane:")
    print(f"   Hit: {result.is_hit}")
    print(f"   Similarity: {result.similarity:.3f}")
    print(f"   Result: {result.result}")
    print()

    # 3. Get stats
    stats = client.get_stats()
    print("3. Cache Statistics:")
    print(f"   Total Entries: {stats['cache_entries']}")
    print(f"   Hit Rate: {stats.get('hit_rate', 0)}")
    print()

    # 4. List entries metadata (small set via gRPC)
    entries = client.list_entries(limit=5)
    print(f"4. List Entries (first 5 via gRPC):")
    print(f"   Total: {entries['total_count']}")
    print(f"   Showing: {entries['returned_count']}")
    for entry in entries['entries']:
        print(f"   - {entry['query']['text']}")
    print()

    client.close()


def demo_data_plane():
    """Demonstrate data plane operations (Arrow Flight)."""
    print("=== Data Plane (Arrow Flight) Operations ===\n")

    # Connect to gRPC control plane to get Flight ticket
    grpc_client = ReminiscenceClient("localhost:8080")

    # Import protobuf for Flight ticket request
    from reminiscence.api import reminiscence_pb2 as pb2

    print("1. Getting Flight ticket from control plane...")

    # Request Flight ticket for bulk data streaming
    ticket_request = pb2.FlightTicketRequest(
        operation="list_entries",
        parameters=json.dumps({
            "limit": 50,
            "offset": 0,
            "batch_size": 25,
            "include_embeddings": False,
        })
    )

    ticket_response = grpc_client.stub.GetFlightTicket(ticket_request)

    print(f"✓ Flight ticket received:")
    print(f"  Endpoint: {ticket_response.flight_endpoint}")
    print(f"  Estimated rows: {ticket_response.estimated_rows}")
    print(f"  Estimated bytes: {ticket_response.estimated_bytes}")
    print()

    grpc_client.close()

    # Connect to Flight data plane
    print("2. Connecting to Flight data plane...")
    flight_client = flight.FlightClient("grpc://localhost:8081")

    # Stream data using the ticket
    print("3. Streaming data from Flight server...\n")

    ticket = flight.Ticket(ticket_response.ticket)
    reader = flight_client.do_get(ticket)

    total_rows = 0
    batch_count = 0

    for batch in reader:
        batch_count += 1
        num_rows = len(batch.data)
        total_rows += num_rows

        print(f"   Batch {batch_count}: {num_rows} rows")

        # Show first few entries from first batch
        if batch_count == 1:
            print(f"   Sample data:")
            for i in range(min(3, num_rows)):
                entry_id = batch.data['entry_id'][i].as_py()
                query_text = batch.data['query_text'][i].as_py()
                table_type = batch.data['table_type'][i].as_py()
                print(f"     [{i}] {entry_id[:12]}... | {query_text} ({table_type})")

    print(f"\n✓ Streamed {total_rows} rows in {batch_count} batches")
    print()

    flight_client.close()


def demo_analytics_use_case():
    """Demonstrate analytics use case with Flight."""
    print("=== Analytics Use Case: Query Pattern Analysis ===\n")

    # Connect to Flight
    flight_client = flight.FlightClient("grpc://localhost:8081")

    # Get Flight ticket for all entries
    grpc_client = ReminiscenceClient("localhost:8080")

    from reminiscence.api import reminiscence_pb2 as pb2

    ticket_request = pb2.FlightTicketRequest(
        operation="list_entries",
        parameters=json.dumps({
            "limit": 1000,
            "batch_size": 100,
        })
    )

    ticket_response = grpc_client.stub.GetFlightTicket(ticket_request)
    grpc_client.close()

    # Stream and analyze
    print("Analyzing query patterns from Flight stream...")

    ticket = flight.Ticket(ticket_response.ticket)
    reader = flight_client.do_get(ticket)

    # Collect query patterns
    contexts = {}
    table_types = {"semantic": 0, "exact": 0}

    for batch in reader:
        for i in range(len(batch.data)):
            # Analyze context
            context_json = batch.data['context'][i].as_py()
            context = json.loads(context_json)
            batch_id = context.get('batch', 'unknown')
            contexts[batch_id] = contexts.get(batch_id, 0) + 1

            # Count table types
            table_type = batch.data['table_type'][i].as_py()
            table_types[table_type] += 1

    print(f"\n✓ Analysis complete:")
    print(f"  Total unique batches: {len(contexts)}")
    print(f"  Semantic entries: {table_types['semantic']}")
    print(f"  Exact entries: {table_types['exact']}")
    print(f"\n  Top 5 batches by entry count:")
    for batch_id, count in sorted(contexts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    Batch {batch_id}: {count} entries")
    print()

    flight_client.close()


def main():
    """Run complete dual-plane architecture demo."""
    # Start servers
    server, cache = start_dual_plane_server()

    try:
        # Wait for servers to be fully ready
        time.sleep(0.5)

        # Demo 1: Control plane operations
        demo_control_plane()

        # Demo 2: Data plane operations
        demo_data_plane()

        # Demo 3: Analytics use case
        demo_analytics_use_case()

        print("=== Demo Complete ===")
        print("\nKey Takeaways:")
        print("1. Control Plane (gRPC): Fast metadata operations, lookups, stores")
        print("2. Data Plane (Flight): Efficient bulk data streaming, analytics")
        print("3. Zero-copy: Arrow Flight transfers data without serialization overhead")
        print("4. Scalable: Stream millions of entries without memory constraints")
        print()
        print("Next Steps for Rust CLI:")
        print("- Use tonic for gRPC control plane")
        print("- Use arrow-flight for data plane streaming")
        print("- Get Flight tickets from GetFlightTicket RPC")
        print("- Stream bulk data for export/analysis commands")

    finally:
        # Cleanup
        print("\nShutting down servers...")
        server.stop(grace=2.0)
        print("✓ Servers stopped")


if __name__ == "__main__":
    main()
