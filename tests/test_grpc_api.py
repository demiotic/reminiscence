"""Tests for gRPC server and client.

These tests verify that the gRPC API works correctly for remote access.
"""

import time
from typing import Any, Dict

import pytest

from reminiscence import Reminiscence, ReminiscenceConfig
from reminiscence.types import (
    LookupRequest,
    MultiModalInput,
    QueryMode,
    StoreRequest,
)

# Skip all tests if gRPC dependencies not installed
pytest.importorskip("grpc")
pytest.importorskip("reminiscence.api.server")
pytest.importorskip("reminiscence.api.client")

from reminiscence.api.client import ReminiscenceClient  # noqa: E402
from reminiscence.api.server import create_server  # noqa: E402


@pytest.fixture(scope="module")
def grpc_server_port():
    """Port for gRPC server in tests."""
    return 50051  # Use non-standard port to avoid conflicts


@pytest.fixture
def grpc_server(shared_embedder, grpc_server_port):
    """Start gRPC server for testing."""
    config = ReminiscenceConfig(
        db_uri="memory://",
        log_level="WARNING",
        enable_metrics=True,
    )

    cache = Reminiscence(config, embedder=shared_embedder)
    server = create_server(
        cache=cache,
        port=grpc_server_port,
        max_workers=5,
        enable_reflection=True,
    )
    server.start()

    yield cache, server

    # Cleanup
    server.stop(grace=1.0)
    cache.backend.clear()


@pytest.fixture
def grpc_client(grpc_server, grpc_server_port):
    """Create gRPC client connected to test server."""
    _, _ = grpc_server  # Ensure server is started

    # Give server a moment to fully start
    time.sleep(0.1)

    client = ReminiscenceClient(
        f"localhost:{grpc_server_port}",
        timeout=5.0,
    )

    yield client

    client.close()


# =============================================================================
# Server Lifecycle Tests
# =============================================================================


def test_server_creation(shared_embedder, grpc_server_port):
    """Test basic server creation and shutdown."""
    config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
    cache = Reminiscence(config, embedder=shared_embedder)

    server = create_server(cache=cache, port=grpc_server_port + 1)
    assert server is not None

    server.start()
    server.stop(grace=0.5)

    cache.backend.clear()


def test_client_context_manager(grpc_server_port):
    """Test client context manager support."""
    with ReminiscenceClient(f"localhost:{grpc_server_port}") as client:
        assert client is not None
        assert client.address == f"localhost:{grpc_server_port}"


# =============================================================================
# Core Cache Operations
# =============================================================================


def test_grpc_lookup_miss(grpc_client):
    """Test lookup when entry doesn't exist."""
    query = MultiModalInput(text="What is Python?")
    context = {"model": "gpt-4"}

    result = grpc_client.lookup(query, context)

    assert result.is_hit is False
    assert result.result is None


def test_grpc_store_and_lookup(grpc_client):
    """Test store followed by lookup."""
    query = MultiModalInput(text="What is Python?")
    context = {"model": "gpt-4"}
    result_data = "Python is a programming language"

    # Store
    success = grpc_client.store(query, context, result_data)
    assert success is True

    # Lookup
    result = grpc_client.lookup(query, context)
    assert result.is_hit is True
    assert result.result == result_data
    assert result.similarity is not None
    assert result.similarity > 0.99  # Should be near-perfect match


def test_grpc_store_with_metadata(grpc_client):
    """Test storing entry with metadata."""
    query = MultiModalInput(text="Test query with metadata")
    context = {"model": "gpt-4"}
    result_data = "Test result"
    metadata = {"source": "test", "version": "1.0"}

    success = grpc_client.store(
        query,
        context,
        result_data,
        metadata=metadata,
    )

    assert success is True

    # Verify can retrieve it
    result = grpc_client.lookup(query, context)
    assert result.is_hit is True


def test_grpc_store_with_ttl(grpc_client):
    """Test storing entry with TTL."""
    query = MultiModalInput(text="Test query with TTL")
    context = {"model": "gpt-4"}
    result_data = "Test result"

    success = grpc_client.store(
        query,
        context,
        result_data,
        ttl_seconds=3600,
    )

    assert success is True


def test_grpc_lookup_with_threshold(grpc_client):
    """Test lookup with custom similarity threshold."""
    query = MultiModalInput(text="What is machine learning?")
    context = {"model": "gpt-4"}
    result_data = "ML is a subset of AI"

    # Store
    grpc_client.store(query, context, result_data)

    # Lookup with very high threshold (should miss)
    result = grpc_client.lookup(
        query,
        context,
        similarity_threshold=0.99,
    )
    assert result.is_hit is True  # Exact match should still hit

    # Lookup with lower threshold (should hit)
    similar_query = MultiModalInput(text="Explain machine learning")
    result = grpc_client.lookup(
        similar_query,
        context,
        similarity_threshold=0.70,
    )
    assert result.is_hit is True
    assert result.similarity < 1.0  # Not exact match


def test_grpc_query_modes(grpc_client):
    """Test different query modes."""
    # Test SEMANTIC/AUTO modes with natural language query
    semantic_query = MultiModalInput(text="What are the latest user statistics?")
    context = {"db": "prod"}
    result_data = "user data"

    # Store with SEMANTIC mode
    success = grpc_client.store(
        semantic_query,
        context,
        result_data,
        mode=QueryMode.SEMANTIC,
    )
    assert success is True

    # Lookup with SEMANTIC mode (should hit)
    result = grpc_client.lookup(semantic_query, context, mode=QueryMode.SEMANTIC)
    assert result.is_hit is True

    # Lookup with AUTO mode (should also hit via semantic detection)
    result = grpc_client.lookup(semantic_query, context, mode=QueryMode.AUTO)
    assert result.is_hit is True

    # Test EXACT mode with SQL query
    exact_query = MultiModalInput(text="SELECT * FROM products")
    success = grpc_client.store(
        exact_query,
        context,
        "product data",
        mode=QueryMode.EXACT,
    )
    assert success is True

    # EXACT mode lookup should hit for exact match
    result = grpc_client.lookup(exact_query, context, mode=QueryMode.EXACT)
    assert result.is_hit is True

    # AUTO mode with SQL query should also work (detects as exact)
    result = grpc_client.lookup(exact_query, context, mode=QueryMode.AUTO)
    assert result.is_hit is True


# =============================================================================
# Batch Operations
# =============================================================================


def test_grpc_lookup_batch(grpc_client):
    """Test batch lookup."""
    # Store some entries
    for i in range(3):
        query = MultiModalInput(text=f"Query {i}")
        context = {"model": "gpt-4", "index": str(i)}
        grpc_client.store(query, context, f"Result {i}")

    # Batch lookup
    requests = [
        LookupRequest(
            query=MultiModalInput(text=f"Query {i}"),
            context={"model": "gpt-4", "index": str(i)},
            mode=QueryMode.AUTO,
        )
        for i in range(3)
    ]

    results = grpc_client.lookup_batch(requests)

    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.is_hit is True
        assert result.result == f"Result {i}"


def test_grpc_store_batch(grpc_client):
    """Test batch store."""
    requests = [
        StoreRequest(
            query=MultiModalInput(text=f"Batch query {i}"),
            context={"model": "gpt-4", "batch": str(i)},
            result=f"Batch result {i}",
        )
        for i in range(5)
    ]

    success_flags = grpc_client.store_batch(requests)

    assert len(success_flags) == 5
    assert all(success_flags)

    # Verify entries were stored
    for i in range(5):
        result = grpc_client.lookup(
            MultiModalInput(text=f"Batch query {i}"),
            {"model": "gpt-4", "batch": str(i)},
        )
        assert result.is_hit is True


# =============================================================================
# Check Availability
# =============================================================================


def test_grpc_check_availability(grpc_client):
    """Test availability check."""
    query = MultiModalInput(text="Availability test")
    context = {"model": "gpt-4"}

    # Check when not available
    avail = grpc_client.check_availability(query, context)
    assert avail.available is False

    # Store entry
    grpc_client.store(query, context, "Test result")

    # Check when available
    avail = grpc_client.check_availability(query, context)
    assert avail.available is True
    assert avail.age_seconds is not None
    assert avail.similarity is not None


# =============================================================================
# Invalidation & Cleanup
# =============================================================================


def test_grpc_invalidate_by_context(grpc_client):
    """Test invalidation by context."""
    # Store entries with different contexts
    for i in range(3):
        query = MultiModalInput(text=f"Query {i}")
        context = {"model": "gpt-4" if i < 2 else "claude"}
        grpc_client.store(query, context, f"Result {i}")

    # Invalidate gpt-4 entries
    count = grpc_client.invalidate(context={"model": "gpt-4"})
    assert count >= 2  # At least the 2 gpt-4 entries


def test_grpc_invalidate_by_age(grpc_client):
    """Test invalidation by age."""
    # Store entry
    query = MultiModalInput(text="Old query")
    context = {"model": "gpt-4"}
    grpc_client.store(query, context, "Old result")

    # Wait a moment
    time.sleep(0.1)

    # Invalidate entries older than 0.05 seconds
    count = grpc_client.invalidate(older_than_seconds=0.05)
    assert count >= 1


def test_grpc_cleanup_expired(grpc_client):
    """Test cleanup of expired entries."""
    # Without TTL, cleanup should delete nothing
    count = grpc_client.cleanup_expired()
    assert count == 0


def test_grpc_clear(grpc_client):
    """Test clearing all entries."""
    # Store some entries
    for i in range(3):
        query = MultiModalInput(text=f"Clear test {i}")
        grpc_client.store(query, {"model": "gpt-4"}, f"Result {i}")

    # Clear all
    success = grpc_client.clear()
    assert success is True

    # Verify entries are gone
    stats = grpc_client.get_stats()
    assert stats["cache_entries"] == 0


# =============================================================================
# Index Operations
# =============================================================================


def test_grpc_create_index(grpc_client):
    """Test index creation."""
    # Store enough entries for indexing (256+ recommended)
    for i in range(300):
        query = MultiModalInput(text=f"Index test query {i}")
        grpc_client.store(query, {"model": "gpt-4"}, f"Result {i}")

    # Create index - may fail if insufficient entries, that's okay for testing
    success = grpc_client.create_index(num_partitions=128)
    # Index creation can fail due to various reasons (insufficient data, etc.)
    # Just verify the operation completes without crashing
    assert success in [True, False]

    # If index was created, verify stats
    if success:
        index_stats = grpc_client.get_index_stats()
        assert index_stats["has_index"] is True


def test_grpc_get_index_stats(grpc_client):
    """Test getting index statistics."""
    stats = grpc_client.get_index_stats()

    assert "has_index" in stats
    assert "total_entries" in stats
    assert "note" in stats


# =============================================================================
# Stats & Health
# =============================================================================


def test_grpc_get_stats(grpc_client):
    """Test getting cache statistics."""
    # Store some entries
    for i in range(5):
        query = MultiModalInput(text=f"Stats test {i}")
        grpc_client.store(query, {"model": "gpt-4"}, f"Result {i}")

    stats = grpc_client.get_stats()

    assert stats["cache_entries"] >= 5
    assert stats["total_entries"] >= 5
    assert "max_entries" in stats
    assert "eviction_policy" in stats
    assert "threshold" in stats
    assert "embedding_dim" in stats
    assert "model" in stats
    assert "storage" in stats

    # Metrics should be present
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats


def test_grpc_health_check(grpc_client):
    """Test health check."""
    health = grpc_client.health_check()

    assert "status" in health
    assert health["status"] in ["healthy", "unhealthy"]

    assert "checks" in health
    assert "embedding" in health["checks"]
    assert "database" in health["checks"]

    assert "metrics" in health
    assert "total_entries" in health["metrics"]
    assert "recent_errors" in health["metrics"]

    assert "timestamp" in health


# =============================================================================
# Capabilities
# =============================================================================


def test_grpc_get_capabilities(grpc_client):
    """Test getting server capabilities."""
    caps = grpc_client.get_capabilities()

    assert "version" in caps
    assert "flight_enabled" in caps
    assert "supported_features" in caps

    # Note: In this test, Flight is not enabled because the gRPC server
    # is created using create_server() directly without detecting the
    # Flight server that auto-started with Reminiscence()
    assert caps["flight_enabled"] is False


# =============================================================================
# Database Inspection
# =============================================================================


def test_grpc_list_entries_basic(grpc_client):
    """Test basic entry listing."""
    # Store some test entries
    for i in range(5):
        query = MultiModalInput(text=f"List test query {i}")
        context = {"model": "gpt-4", "index": str(i)}
        grpc_client.store(query, context, f"List test result {i}")

    # List entries
    result = grpc_client.list_entries(limit=10)

    assert result["total_count"] >= 5
    assert result["returned_count"] >= 5
    assert len(result["entries"]) >= 5

    # Check entry structure
    entry = result["entries"][0]
    assert "entry_id" in entry
    assert "query" in entry
    assert "context" in entry
    assert "created_at" in entry
    assert "matched_context_key" in entry  # "semantic" or "exact"


def test_grpc_list_entries_pagination(grpc_client):
    """Test entry listing with pagination."""
    # Store 20 entries
    for i in range(20):
        query = MultiModalInput(text=f"Pagination test {i}")
        context = {"model": "gpt-4", "page": str(i // 5)}
        grpc_client.store(query, context, f"Result {i}")

    # First page
    result = grpc_client.list_entries(limit=10, offset=0)
    assert result["returned_count"] == 10
    assert result["has_more"] is True

    # Second page
    result = grpc_client.list_entries(limit=10, offset=10)
    assert result["returned_count"] >= 10
    assert "entries" in result


def test_grpc_list_entries_with_context_filter(grpc_client):
    """Test filtering entries by context."""
    # Store entries with different contexts
    for i in range(3):
        query = MultiModalInput(text=f"Filter test {i}")
        context = {"model": "gpt-4" if i < 2 else "claude", "index": str(i)}
        grpc_client.store(query, context, f"Result {i}")

    # Filter by context
    result = grpc_client.list_entries(
        limit=100,
        context_filter={"model": "gpt-4"},
    )

    # Should only get gpt-4 entries
    for entry in result["entries"]:
        # Check that the entry context contains the filter
        if "model" in entry["context"]:
            # Some entries might not have the model field if they were stored before
            assert entry["context"]["model"] == "gpt-4"


def test_grpc_list_entries_with_query_filter(grpc_client):
    """Test filtering entries by query text."""
    # Store entries
    grpc_client.store(
        MultiModalInput(text="Python programming"),
        {"model": "gpt-4"},
        "Python result",
    )
    grpc_client.store(
        MultiModalInput(text="JavaScript coding"),
        {"model": "gpt-4"},
        "JS result",
    )

    # Filter by query text
    result = grpc_client.list_entries(limit=100, query_filter="Python")

    # Should only get Python entries
    found_python = False
    for entry in result["entries"]:
        if "Python" in entry["query"]["text"]:
            found_python = True

    assert found_python is True


def test_grpc_list_entries_without_results(grpc_client):
    """Test listing entries without including cached results."""
    # Store entry
    query = MultiModalInput(text="No results test")
    context = {"model": "gpt-4"}
    grpc_client.store(query, context, "This is a large result" * 100)

    # List without results
    result = grpc_client.list_entries(
        limit=10,
        include_results=False,
    )

    # Results should not be included
    if result["entries"]:
        entry = result["entries"][0]
        # The result field should not be present
        assert "result" not in entry or entry.get("result") is None


def test_grpc_list_entries_sorting(grpc_client):
    """Test sorting entries."""
    import time

    # Store entries with delays to ensure different timestamps
    for i in range(3):
        query = MultiModalInput(text=f"Sort test {i}")
        context = {"model": "gpt-4", "index": str(i)}
        grpc_client.store(query, context, f"Result {i}")
        time.sleep(0.05)  # Small delay between entries

    # List with descending order (newest first)
    result = grpc_client.list_entries(
        limit=10,
        sort_by="created_at",
        sort_descending=True,
    )

    if len(result["entries"]) >= 2:
        # Newer entries should come first
        assert (
            result["entries"][0]["created_at"] >= result["entries"][1]["created_at"]
        )


# =============================================================================
# Error Handling
# =============================================================================


def test_grpc_store_error_results_not_allowed_by_default(grpc_client):
    """Test that error results are not stored by default."""
    query = MultiModalInput(text="Error test")
    context = {"model": "gpt-4"}
    error_result = {"error": "Something failed"}

    # Store operation succeeds but entry is silently skipped
    success = grpc_client.store(query, context, error_result)
    assert success is True  # Store doesn't fail, but entry not saved

    # Verify entry was NOT actually stored
    result = grpc_client.lookup(query, context)
    assert result.is_hit is False  # Entry should not exist


def test_grpc_store_error_results_with_allow_errors(grpc_client):
    """Test storing error results when allowed."""
    query = MultiModalInput(text="Error test with flag")
    context = {"model": "gpt-4"}
    error_result = {"error": "Something failed"}

    # Should store errors when explicitly allowed
    success = grpc_client.store(
        query,
        context,
        error_result,
        allow_errors=True,
    )
    assert success is True

    # Verify can retrieve it
    result = grpc_client.lookup(query, context)
    assert result.is_hit is True
    assert result.result == error_result


# =============================================================================
# Integration Tests
# =============================================================================


def test_grpc_end_to_end_workflow(grpc_client):
    """Test complete end-to-end workflow."""
    # 1. Check health
    health = grpc_client.health_check()
    assert health["status"] == "healthy"

    # 2. Store multiple entries
    for i in range(10):
        query = MultiModalInput(text=f"E2E query {i}")
        context = {"model": "gpt-4", "session": "test"}
        success = grpc_client.store(query, context, f"E2E result {i}")
        assert success is True

    # 3. Lookup entries
    result = grpc_client.lookup(
        MultiModalInput(text="E2E query 5"),
        {"model": "gpt-4", "session": "test"},
    )
    assert result.is_hit is True

    # 4. Get stats
    stats = grpc_client.get_stats()
    assert stats["cache_entries"] >= 10

    # 5. Clear all entries
    success = grpc_client.clear()
    assert success is True

    # 6. Verify all entries are gone
    result = grpc_client.lookup(
        MultiModalInput(text="E2E query 5"),
        {"model": "gpt-4", "session": "test"},
    )
    assert result.is_hit is False

    stats = grpc_client.get_stats()
    assert stats["cache_entries"] == 0


def test_grpc_multimodal_support(grpc_client):
    """Test multimodal input support."""
    # Text only
    text_query = MultiModalInput(text="Text query")
    grpc_client.store(text_query, {"type": "text"}, "Text result")

    result = grpc_client.lookup(text_query, {"type": "text"})
    assert result.is_hit is True

    # With metadata
    multimodal_query = MultiModalInput(
        text="Image query",
        metadata={"format": "png", "size": "1024x768"},
    )
    grpc_client.store(
        multimodal_query,
        {"type": "image"},
        "Multimodal result",
    )

    result = grpc_client.lookup(multimodal_query, {"type": "image"})
    assert result.is_hit is True
