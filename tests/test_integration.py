"""Integration tests for end-to-end scenarios."""

import pytest
import time

from memora import Memora, CacheConfig


class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_complete_workflow(self):
        """Test complete cache workflow with exact matches."""
        config = CacheConfig(
            db_uri="memory://",
            similarity_threshold=0.75,
            enable_metrics=True,
            log_level="WARNING",
        )
        cache = Memora(config)

        # 1. Empty cache
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

        # 2. Store multiple entries
        for i in range(5):
            cache.store(f"query {i}", {"agent": "test"}, f"result {i}")

        # 3. Lookup exact
        result = cache.lookup("query 0", {"agent": "test"})
        assert result.is_hit
        assert result.result == "result 0"

        # 4. Check stats
        stats = cache.get_stats()
        assert stats["total_entries"] == 5
        assert stats["hits"] >= 1

        # 5. Invalidate
        deleted = cache.invalidate(context={"agent": "test"})
        assert deleted == 5
        assert cache.backend.count() == 0

    def test_semantic_similarity_workflow(self):
        """Test semantic similarity matching."""
        config = CacheConfig(
            db_uri="memory://",
            similarity_threshold=0.70,
            log_level="WARNING",
        )
        cache = Memora(config)

        # Store detailed query
        cache.store(
            "What is machine learning and how does it work?",
            {"agent": "qa"},
            "Machine learning explanation",
        )

        # Lookup with similar wording
        result = cache.lookup("Explain how machine learning works", {"agent": "qa"})

        assert result.is_hit
        assert result.result == "Machine learning explanation"
        assert result.similarity > 0.70

    def test_multi_context_workflow(self):
        """Test with multiple contexts."""
        cache = Memora(CacheConfig(db_uri="memory://", log_level="WARNING"))

        # Store with different contexts
        contexts = [
            {"agent": "sql", "db": "prod"},
            {"agent": "sql", "db": "dev"},
            {"agent": "api", "service": "payments"},
        ]

        for ctx in contexts:
            cache.store("test query", ctx, f"result for {ctx}")

        # Lookup should respect context
        for ctx in contexts:
            result = cache.lookup("test query", ctx)
            assert result.is_hit
            assert str(ctx) in str(result.result)

    def test_persistence_workflow(self, temp_cache_dir):
        """Test persistence across instances."""
        from pathlib import Path

        db_path = str(Path(temp_cache_dir) / "persist.db")

        # First instance - store data
        config1 = CacheConfig(db_uri=db_path, log_level="WARNING")
        cache1 = Memora(config1)
        cache1.store("persistent query", {"agent": "test"}, "persistent result")

        # Second instance - should find data
        config2 = CacheConfig(db_uri=db_path, log_level="WARNING")
        cache2 = Memora(config2)

        result = cache2.lookup("persistent query", {"agent": "test"})
        assert result.is_hit
        assert result.result == "persistent result"

    def test_ttl_workflow(self):
        """Test TTL expiration workflow."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=1,
            log_level="WARNING",
        )
        cache = Memora(config)

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

    def test_decorator_workflow(self):
        """Test decorator integration."""
        cache = Memora(CacheConfig(db_uri="memory://", log_level="WARNING"))

        call_count = 0

        @cache.cached(static_context={"function": "expensive"})
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

    def test_eviction_workflow(self):
        """Test eviction policy workflow."""
        config = CacheConfig(
            db_uri="memory://",
            max_entries=3,
            eviction_policy="fifo",
            log_level="WARNING",
        )
        cache = Memora(config)

        # Store 4 entries
        for i in range(4):
            cache.store(f"query {i}", {"agent": "test"}, f"result {i}")
            time.sleep(0.01)

        # Should have evicted oldest
        assert cache.backend.count() == 3

        # First entry should be gone
        result = cache.lookup("query 0", {"agent": "test"})
        assert result.is_miss

        # Newer entries should exist
        for i in range(1, 4):
            result = cache.lookup(f"query {i}", {"agent": "test"})
            assert result.is_hit
