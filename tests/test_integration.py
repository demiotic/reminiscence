"""Integration tests for end-to-end scenarios."""

import time
import pytest
from reminiscence import Reminiscence, ReminiscenceConfig


class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_complete_workflow(self, reminiscence):
        """Test complete cache workflow with exact matches."""
        # 1. Empty cache (fixture ya lo limpió)
        stats = reminiscence.get_stats()
        assert stats["total_entries"] == 0

        # 2. Store multiple entries
        for i in range(5):
            reminiscence.store(f"query {i}", {"agent": "test"}, f"result {i}")

        # 3. Lookup exact
        result = reminiscence.lookup("query 0", {"agent": "test"})
        assert result.is_hit
        assert result.result == "result 0"

        # 4. Check stats
        stats = reminiscence.get_stats()
        assert stats["total_entries"] == 5
        assert stats["hits"] >= 1

        # 5. Invalidate
        deleted = reminiscence.invalidate(context={"agent": "test"})
        assert deleted == 5
        assert reminiscence.backend.count() == 0

    def test_semantic_similarity_workflow(self, reminiscence):
        """Test semantic similarity matching."""
        # Store detailed query
        reminiscence.store(
            "What is machine learning and how does it work?",
            {"agent": "qa"},
            "Machine learning explanation",
        )

        # Lookup with similar wording
        result = reminiscence.lookup(
            "Explain how machine learning works", {"agent": "qa"}
        )

        assert result.is_hit
        assert result.result == "Machine learning explanation"
        assert result.similarity > 0.70

    def test_multi_context_workflow(self, reminiscence):
        """Test with multiple contexts."""
        # Store with different contexts
        contexts = [
            {"agent": "sql", "db": "prod"},
            {"agent": "sql", "db": "dev"},
            {"agent": "api", "service": "payments"},
        ]

        for ctx in contexts:
            reminiscence.store("test query", ctx, f"result for {ctx}")

        # Lookup should respect context
        for ctx in contexts:
            result = reminiscence.lookup("test query", ctx)
            assert result.is_hit
            assert str(ctx) in str(result.result)

    def test_persistence_workflow(self, temp_cache_dir):
        """Test persistence across instances."""
        from pathlib import Path

        db_path = str(Path(temp_cache_dir) / "persist.db")

        # First instance - store data
        config1 = ReminiscenceConfig(db_uri=db_path, log_level="WARNING")
        cache1 = Reminiscence(config1)
        cache1.store("persistent query", {"agent": "test"}, "persistent result")

        # Second instance - should find data
        config2 = ReminiscenceConfig(db_uri=db_path, log_level="WARNING")
        cache2 = Reminiscence(config2)

        result = cache2.lookup("persistent query", {"agent": "test"})
        assert result.is_hit
        assert result.result == "persistent result"

    def test_ttl_workflow(self, reminiscence):
        """Test TTL expiration workflow."""
        # Necesita config específico con TTL
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=1,
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # Store entry
        cache.store("expiring query", {"agent": "test"}, "temporary result")

        # Should hit immediately
        result = cache.lookup("expiring query", {"agent": "test"})
        assert result.is_hit

        # Wait for expiration
        time.sleep(1.2)

        # Should miss after TTL
        result = cache.lookup("expiring query", {"agent": "test"})
        assert result.is_miss

    def test_decorator_workflow(self, reminiscence):
        """Test decorator integration."""
        call_count = 0

        @reminiscence.cached(static_context={"function": "expensive"})
        def expensive_function(query: str):
            nonlocal call_count
            call_count += 1
            return f"Computed: {query}"

        # First call
        result1 = expensive_function("compute this")
        assert call_count == 1

        # Second call (cache hit)
        result2 = expensive_function("compute this")
        assert call_count == 1
        assert result1 == result2

    def test_eviction_workflow(self, reminiscence):
        """Test eviction policy workflow."""
        # Necesita config específico con max_entries
        config = ReminiscenceConfig(
            db_uri="memory://",
            max_entries=3,
            eviction_policy="fifo",
            similarity_threshold=0.95,
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # Store 4 entries con queries MUY diferentes
        queries = [
            ("What is Python programming?", "Python explanation"),
            ("How to cook Italian pasta?", "Pasta recipe"),
            ("Explain quantum mechanics", "Quantum physics"),
            ("Best travel destinations Europe", "Travel guide"),
        ]

        for query, result in queries:
            cache.store(query, {"agent": "test"}, result)
            time.sleep(0.01)

        # Should have evicted oldest
        assert cache.backend.count() == 3

        # First entry should be gone
        result = cache.lookup("What is Python programming?", {"agent": "test"})
        assert result.is_miss

        # Newer entries should exist
        for query, expected_result in queries[1:]:
            result = cache.lookup(query, {"agent": "test"})
            assert result.is_hit
            assert result.result == expected_result


class TestQueryModesEndToEnd:
    """End-to-end tests for query modes (semantic/exact/auto)."""

    def test_semantic_mode_workflow(self, reminiscence):
        """Test semantic mode end-to-end workflow."""
        # Store with semantic mode (default)
        reminiscence.store(
            "What is artificial intelligence?",
            {"agent": "qa"},
            "AI is the simulation of human intelligence",
            query_mode="semantic",
        )

        # Lookup with semantic mode - similar query should hit
        result = reminiscence.lookup(
            "Explain artificial intelligence", {"agent": "qa"}, query_mode="semantic"
        )

        assert result.is_hit
        assert result.similarity > 0.75
        assert "AI is the simulation" in result.result

    def test_exact_mode_workflow(self, reminiscence):
        """Test exact mode end-to-end workflow."""
        # Store with exact mode
        sql_query = "SELECT * FROM users WHERE id = 1"
        reminiscence.store(
            sql_query,
            {"database": "prod"},
            [{"id": 1, "name": "Alice"}],
            query_mode="exact",
        )

        # Exact same query should hit
        result = reminiscence.lookup(
            sql_query, {"database": "prod"}, query_mode="exact"
        )

        assert result.is_hit
        assert result.similarity >= 0.9999
        assert result.result == [{"id": 1, "name": "Alice"}]

        # Slightly different query should miss (exact mode)
        result = reminiscence.lookup(
            "SELECT * FROM users WHERE id = 2", {"database": "prod"}, query_mode="exact"
        )

        assert result.is_miss

    def test_auto_mode_workflow(self, reminiscence):
        """Test auto mode workflow (exact → semantic fallback)."""
        # Store with auto mode (generates embeddings)
        reminiscence.store(
            "What is deep learning?",
            {"agent": "qa"},
            "Deep learning explanation",
            query_mode="auto",
        )

        # Exact same query - should hit via exact match first
        result = reminiscence.lookup(
            "What is deep learning?", {"agent": "qa"}, query_mode="auto"
        )

        assert result.is_hit
        assert result.similarity >= 0.9999

        # Similar query - should hit via semantic fallback
        result = reminiscence.lookup(
            "Explain deep learning concepts", {"agent": "qa"}, query_mode="auto"
        )

        assert result.is_hit
        assert result.similarity < 1.0
        assert result.similarity > 0.70

    def test_mixed_modes_coexistence(self, reminiscence):
        """Test that entries with different modes coexist correctly."""
        # Store semantic entry
        reminiscence.store(
            "What is Python?",
            {"agent": "qa"},
            "Python explanation",
            query_mode="semantic",
        )

        # Store exact entry
        reminiscence.store(
            "SELECT COUNT(*) FROM orders",
            {"database": "analytics"},
            {"count": 1000},
            query_mode="exact",
        )

        assert reminiscence.backend.count() == 2

        # Both should be retrievable
        result1 = reminiscence.lookup(
            "Explain Python", {"agent": "qa"}, query_mode="semantic"
        )
        assert result1.is_hit

        result2 = reminiscence.lookup(
            "SELECT COUNT(*) FROM orders", {"database": "analytics"}, query_mode="exact"
        )
        assert result2.is_hit

    def test_exact_mode_with_complex_results(self, reminiscence):
        """Test exact mode with complex data types (DataFrames, etc)."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Store DataFrame with exact mode
        reminiscence.store(
            "SELECT * FROM products", {"database": "prod"}, df, query_mode="exact"
        )

        # Retrieve with exact mode
        result = reminiscence.lookup(
            "SELECT * FROM products", {"database": "prod"}, query_mode="exact"
        )

        assert result.is_hit
        assert isinstance(result.result, pd.DataFrame)
        assert result.result.equals(df)

    def test_decorator_with_query_modes(self, reminiscence):
        """Test decorator integration with query modes."""
        call_count = 0

        @reminiscence.cached(
            query="question", query_mode="semantic", context_params=["user"]
        )
        def ask_semantic(question: str, user: str):
            nonlocal call_count
            call_count += 1
            return f"Answer for {user}: {question}"

        # First call
        result1 = ask_semantic("What is AI?", "alice")
        assert call_count == 1

        # Similar question - should hit
        result2 = ask_semantic("Explain AI", "alice")
        assert call_count == 1
        assert result1 == result2

    def test_decorator_exact_mode(self, reminiscence):
        """Test decorator with exact mode."""
        call_count = 0

        @reminiscence.cached(
            query="sql", query_mode="exact", context_params=["database"]
        )
        def run_sql(sql: str, database: str):
            nonlocal call_count
            call_count += 1
            return f"Result: {sql}"

        # First call
        result1 = run_sql("SELECT * FROM users", "prod")
        assert call_count == 1

        # Exact same - should hit
        result2 = run_sql("SELECT * FROM users", "prod")
        assert call_count == 1
        assert result1 == result2

        # Different SQL - should miss
        _ = run_sql("SELECT * FROM orders", "prod")
        assert call_count == 2

    def test_performance_exact_vs_semantic(self, reminiscence):
        """Test that exact mode uses high threshold correctly."""
        query = "SELECT COUNT(*) FROM large_table"
        context = {"database": "analytics"}
        result_data = {"count": 10000}

        # Store
        reminiscence.store(query, context, result_data, query_mode="exact")

        # Exact same query should hit
        result = reminiscence.lookup(query, context, query_mode="exact")
        assert result.is_hit
        assert result.similarity >= 0.9999

        # Slightly different query should miss
        result = reminiscence.lookup(
            query + " WHERE id > 5", context, query_mode="exact"
        )
        assert result.is_miss

    def test_query_mode_with_ttl(self, reminiscence):
        """Test query modes work correctly with TTL."""
        # Necesita config específico
        config = ReminiscenceConfig(
            db_uri="memory://", ttl_seconds=1, log_level="WARNING"
        )
        cache = Reminiscence(config)

        cache.store(
            "SELECT * FROM users", {"db": "prod"}, [{"id": 1}], query_mode="exact"
        )

        result = cache.lookup("SELECT * FROM users", {"db": "prod"}, query_mode="exact")
        assert result.is_hit

        time.sleep(1.2)

        result = cache.lookup("SELECT * FROM users", {"db": "prod"}, query_mode="exact")
        assert result.is_miss

    def test_query_mode_with_eviction(self, reminiscence):
        """Test query modes work correctly with eviction."""
        # Necesita config específico
        config = ReminiscenceConfig(
            db_uri="memory://",
            max_entries=3,
            eviction_policy="fifo",
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        cache.store("query1", {"agent": "test"}, "r1", query_mode="exact")
        time.sleep(0.01)
        cache.store("query2", {"agent": "test"}, "r2", query_mode="semantic")
        time.sleep(0.01)
        cache.store("query3", {"agent": "test"}, "r3", query_mode="exact")
        time.sleep(0.01)

        assert cache.backend.count() == 3

        cache.store("query4", {"agent": "test"}, "r4", query_mode="semantic")

        assert cache.backend.count() == 3

        result = cache.lookup("query1", {"agent": "test"}, query_mode="exact")
        assert result.is_miss

    def test_stats_with_query_modes(self, reminiscence):
        """Test stats reporting works with mixed query modes."""
        reminiscence.store(
            "What is Python programming language?",
            {"agent": "qa"},
            "Python info",
            query_mode="semantic",
        )
        reminiscence.store(
            "How to cook pasta carbonara?",
            {"agent": "qa"},
            "Pasta recipe",
            query_mode="exact",
        )
        reminiscence.store(
            "Explain quantum mechanics basics",
            {"agent": "qa"},
            "Quantum info",
            query_mode="semantic",
        )

        reminiscence.lookup(
            "What is Python programming language?",
            {"agent": "qa"},
            query_mode="semantic",
        )
        reminiscence.lookup(
            "How to cook pasta carbonara?", {"agent": "qa"}, query_mode="exact"
        )

        reminiscence.lookup(
            "Best travel destinations in Europe 2025",
            {"agent": "qa"},
            query_mode="auto",
        )

        stats = reminiscence.get_stats()

        assert stats["total_entries"] == 3
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert float(stats["hit_rate"].rstrip("%")) > 50
