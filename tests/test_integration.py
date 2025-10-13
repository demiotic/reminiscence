"""Integration tests for end-to-end scenarios."""

import time

import pytest

from reminiscence import Reminiscence, ReminiscenceConfig
from reminiscence.types import MultiModalInput, QueryMode


class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_complete_workflow(self, reminiscence):
        """Test complete cache workflow with exact matches."""
        # Empty cache (fixture already cleared it)
        stats = reminiscence.get_stats()
        assert stats["total_entries"] == 0

        # 2. Store multiple entries
        for i in range(5):
            reminiscence.store(
                MultiModalInput(text=f"query {i}"), {"agent": "test"}, f"result {i}"
            )

        # 3. Lookup exact
        result = reminiscence.lookup(MultiModalInput(text="query 0"), {"agent": "test"})
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
            MultiModalInput(text="What is machine learning and how does it work?"),
            {"agent": "qa"},
            "Machine learning explanation",
        )

        # Lookup with similar wording
        result = reminiscence.lookup(
            MultiModalInput(text="Explain how machine learning works"), {"agent": "qa"}
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
            reminiscence.store(
                MultiModalInput(text="test query"), ctx, f"result for {ctx}"
            )

        # Lookup should respect context
        for ctx in contexts:
            result = reminiscence.lookup(MultiModalInput(text="test query"), ctx)
            assert result.is_hit
            assert str(ctx) in str(result.result)

    def test_persistence_workflow(self, temp_cache_dir):
        """Test persistence across instances."""
        from pathlib import Path

        db_path = str(Path(temp_cache_dir) / "persist.db")

        # First instance - store data
        config1 = ReminiscenceConfig(db_uri=db_path, log_level="WARNING")
        cache1 = Reminiscence(config1)
        cache1.store(
            MultiModalInput(text="persistent query"),
            {"agent": "test"},
            "persistent result",
        )

        # Second instance - should find data
        config2 = ReminiscenceConfig(db_uri=db_path, log_level="WARNING")
        cache2 = Reminiscence(config2)

        result = cache2.lookup(
            MultiModalInput(text="persistent query"), {"agent": "test"}
        )
        assert result.is_hit
        assert result.result == "persistent result"

    def test_ttl_workflow(self, reminiscence):
        """Test TTL expiration workflow."""
        # Needs specific config with TTL
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=1,
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # Store entry
        cache.store(
            MultiModalInput(text="expiring query"),
            {"agent": "test"},
            "temporary result",
        )

        # Should hit immediately
        result = cache.lookup(MultiModalInput(text="expiring query"), {"agent": "test"})
        assert result.is_hit

        # Wait for expiration
        time.sleep(1.2)

        # Should miss after TTL
        result = cache.lookup(MultiModalInput(text="expiring query"), {"agent": "test"})
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
        # Needs specific config with max_entries
        config = ReminiscenceConfig(
            db_uri="memory://",
            max_entries=3,
            eviction_policy="fifo",
            similarity_threshold=0.95,
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # Store 4 entries with very different queries
        queries = [
            ("What is Python programming?", "Python explanation"),
            ("How to cook Italian pasta?", "Pasta recipe"),
            ("Explain quantum mechanics", "Quantum physics"),
            ("Best travel destinations Europe", "Travel guide"),
        ]

        for query, result in queries:
            cache.store(MultiModalInput(text=query), {"agent": "test"}, result)
            time.sleep(0.01)

        # Should have evicted oldest
        assert cache.backend.count() == 3

        # First entry should be gone
        result = cache.lookup(
            MultiModalInput(text="What is Python programming?"), {"agent": "test"}
        )
        assert result.is_miss

        # Newer entries should exist
        for query, expected_result in queries[1:]:
            result = cache.lookup(MultiModalInput(text=query), {"agent": "test"})
            assert result.is_hit
            assert result.result == expected_result


class TestQueryModesEndToEnd:
    """End-to-end tests for query modes (semantic/exact/auto)."""

    def test_semantic_mode_workflow(self, reminiscence):
        """Test semantic mode end-to-end workflow."""
        # Store with semantic mode (default)
        reminiscence.store(
            MultiModalInput(text="What is artificial intelligence?"),
            {"agent": "qa"},
            "AI is the simulation of human intelligence",
            mode=QueryMode.SEMANTIC,
        )

        # Lookup with semantic mode - similar query should hit
        result = reminiscence.lookup(
            MultiModalInput(text="Explain artificial intelligence"),
            {"agent": "qa"},
            mode=QueryMode.SEMANTIC,
        )

        assert result.is_hit
        assert result.similarity > 0.75
        assert "AI is the simulation" in result.result

    def test_exact_mode_workflow(self, reminiscence):
        """Test exact mode end-to-end workflow."""
        # Store with exact mode
        sql_query = "SELECT * FROM users WHERE id = 1"
        reminiscence.store(
            MultiModalInput(text=sql_query),
            {"database": "prod"},
            [{"id": 1, "name": "Alice"}],
            mode=QueryMode.EXACT,
        )

        # Exact same query should hit
        result = reminiscence.lookup(
            MultiModalInput(text=sql_query), {"database": "prod"}, mode=QueryMode.EXACT
        )

        assert result.is_hit
        assert result.similarity >= 0.9999
        assert result.result == [{"id": 1, "name": "Alice"}]

        # Slightly different query should miss (exact mode)
        result = reminiscence.lookup(
            MultiModalInput(text="SELECT * FROM users WHERE id = 2"),
            {"database": "prod"},
            mode=QueryMode.EXACT,
        )

        assert result.is_miss

    def test_auto_mode_workflow(self, reminiscence):
        """Test auto mode workflow (exact → semantic fallback)."""
        # Store with auto mode (generates embeddings)
        reminiscence.store(
            MultiModalInput(text="What is deep learning?"),
            {"agent": "qa"},
            "Deep learning explanation",
            mode=QueryMode.AUTO,
        )

        # Exact same query - should hit via exact match first
        result = reminiscence.lookup(
            MultiModalInput(text="What is deep learning?"),
            {"agent": "qa"},
            mode=QueryMode.AUTO,
        )

        assert result.is_hit
        assert result.similarity >= 0.9999

        # Similar query - should hit via semantic fallback
        result = reminiscence.lookup(
            MultiModalInput(text="Explain deep learning concepts"),
            {"agent": "qa"},
            mode=QueryMode.AUTO,
        )

        assert result.is_hit
        assert result.similarity < 1.0
        assert result.similarity > 0.70

    def test_mixed_modes_coexistence(self, reminiscence):
        """Test that entries with different modes coexist correctly."""
        # Store semantic entry
        reminiscence.store(
            MultiModalInput(text="What is Python?"),
            {"agent": "qa"},
            "Python explanation",
            mode=QueryMode.SEMANTIC,
        )

        # Store exact entry
        reminiscence.store(
            MultiModalInput(text="SELECT COUNT(*) FROM orders"),
            {"database": "analytics"},
            {"count": 1000},
            mode=QueryMode.EXACT,
        )

        assert reminiscence.backend.count() == 2

        # Both should be retrievable
        result1 = reminiscence.lookup(
            MultiModalInput(text="Explain Python"),
            {"agent": "qa"},
            mode=QueryMode.SEMANTIC,
        )
        assert result1.is_hit

        result2 = reminiscence.lookup(
            MultiModalInput(text="SELECT COUNT(*) FROM orders"),
            {"database": "analytics"},
            mode=QueryMode.EXACT,
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
            MultiModalInput(text="SELECT * FROM products"),
            {"database": "prod"},
            df,
            mode=QueryMode.EXACT,
        )

        # Retrieve with exact mode
        result = reminiscence.lookup(
            MultiModalInput(text="SELECT * FROM products"),
            {"database": "prod"},
            mode=QueryMode.EXACT,
        )

        assert result.is_hit
        assert isinstance(result.result, pd.DataFrame)
        assert result.result.equals(df)

    def test_decorator_with_query_modes(self, reminiscence):
        """Test decorator integration with query modes."""
        call_count = 0

        @reminiscence.cached(
            query="question", mode=QueryMode.SEMANTIC, context=["user"]
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

        @reminiscence.cached(query="sql", mode=QueryMode.EXACT, context=["database"])
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
        reminiscence.store(
            MultiModalInput(text=query), context, result_data, mode=QueryMode.EXACT
        )

        # Exact same query should hit
        result = reminiscence.lookup(
            MultiModalInput(text=query), context, mode=QueryMode.EXACT
        )
        assert result.is_hit
        assert result.similarity >= 0.9999

        # Slightly different query should miss
        result = reminiscence.lookup(
            MultiModalInput(text=query + " WHERE id > 5"), context, mode=QueryMode.EXACT
        )
        assert result.is_miss

    def test_query_mode_with_ttl(self, reminiscence):
        """Test query modes work correctly with TTL."""
        # Needs specific config
        config = ReminiscenceConfig(
            db_uri="memory://", ttl_seconds=1, log_level="WARNING"
        )
        cache = Reminiscence(config)

        cache.store(
            MultiModalInput(text="SELECT * FROM users"),
            {"db": "prod"},
            [{"id": 1}],
            mode=QueryMode.EXACT,
        )

        result = cache.lookup(
            MultiModalInput(text="SELECT * FROM users"),
            {"db": "prod"},
            mode=QueryMode.EXACT,
        )
        assert result.is_hit

        time.sleep(1.2)

        result = cache.lookup(
            MultiModalInput(text="SELECT * FROM users"),
            {"db": "prod"},
            mode=QueryMode.EXACT,
        )
        assert result.is_miss

    def test_query_mode_with_eviction(self, reminiscence):
        """Test query modes work correctly with eviction."""
        # Needs specific config
        config = ReminiscenceConfig(
            db_uri="memory://",
            max_entries=3,
            eviction_policy="fifo",
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        cache.store(
            MultiModalInput(text="query1"),
            {"agent": "test"},
            "r1",
            mode=QueryMode.EXACT,
        )
        time.sleep(0.01)
        cache.store(
            MultiModalInput(text="query2"),
            {"agent": "test"},
            "r2",
            mode=QueryMode.SEMANTIC,
        )
        time.sleep(0.01)
        cache.store(
            MultiModalInput(text="query3"),
            {"agent": "test"},
            "r3",
            mode=QueryMode.EXACT,
        )
        time.sleep(0.01)

        assert cache.backend.count() == 3

        cache.store(
            MultiModalInput(text="query4"),
            {"agent": "test"},
            "r4",
            mode=QueryMode.SEMANTIC,
        )

        assert cache.backend.count() == 3

        result = cache.lookup(
            MultiModalInput(text="query1"), {"agent": "test"}, mode=QueryMode.EXACT
        )
        assert result.is_miss

    def test_stats_with_query_modes(self, reminiscence):
        """Test stats reporting works with mixed query modes."""
        reminiscence.store(
            MultiModalInput(text="What is Python programming language?"),
            {"agent": "qa"},
            "Python info",
            mode=QueryMode.SEMANTIC,
        )
        reminiscence.store(
            MultiModalInput(text="How to cook pasta carbonara?"),
            {"agent": "qa"},
            "Pasta recipe",
            mode=QueryMode.EXACT,
        )
        reminiscence.store(
            MultiModalInput(text="Explain quantum mechanics basics"),
            {"agent": "qa"},
            "Quantum info",
            mode=QueryMode.SEMANTIC,
        )

        reminiscence.lookup(
            MultiModalInput(text="What is Python programming language?"),
            {"agent": "qa"},
            mode=QueryMode.SEMANTIC,
        )
        reminiscence.lookup(
            MultiModalInput(text="How to cook pasta carbonara?"),
            {"agent": "qa"},
            mode=QueryMode.EXACT,
        )

        reminiscence.lookup(
            MultiModalInput(text="Best travel destinations in Europe 2025"),
            {"agent": "qa"},
            mode=QueryMode.AUTO,
        )

        stats = reminiscence.get_stats()

        assert stats["total_entries"] == 3
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert float(stats["hit_rate"].rstrip("%")) > 50
