"""Tests for memora.core.Memora."""

import pytest
import time
from memora import Memora, CacheConfig


# ============================================================================
# SESSION FIXTURES (load model once, reuse across tests)
# ============================================================================


@pytest.fixture(scope="module")
def memora_session():
    """Shared Memora instance for entire test session (loads model once)."""
    config = CacheConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
    )
    return Memora(config)


@pytest.fixture
def memora_memory(memora_session):
    """Clean Memora for each test (reuses instance, only resets state)."""
    memora_session.clear()
    yield memora_session


# ============================================================================
# TESTS
# ============================================================================


class TestMemoraBasics:
    """Basic initialization and operation tests."""

    def test_init_memory(self, memora_memory):
        """Test initialization with memory backend."""
        assert memora_memory is not None
        assert memora_memory.backend.count() == 0

    def test_init_disk(self, temp_cache_dir):
        """Test initialization with disk backend."""
        from pathlib import Path

        config = CacheConfig(
            db_uri=str(Path(temp_cache_dir) / "test.db"), log_level="WARNING"
        )
        memora_disk = Memora(config)
        assert memora_disk is not None
        assert memora_disk.config.db_uri != "memory://"

    def test_get_stats_empty(self, memora_memory):
        """Test statistics with empty cache."""
        stats = memora_memory.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestErrorHandling:
    """Graceful error handling tests."""

    def test_lookup_handles_embedding_failure(self, memora_memory, monkeypatch):
        """Lookup should return MISS if embedding fails."""
        memora_memory.store("dummy query", {"agent": "test"}, "dummy result")

        def failing_embed(text):
            raise RuntimeError("Embedding model crashed")

        monkeypatch.setattr(memora_memory.embedder, "embed", failing_embed)

        result = memora_memory.lookup("test query", {"agent": "test"})

        assert result.is_miss
        assert result.result is None
        assert memora_memory.metrics.lookup_errors >= 1

    def test_store_handles_unserializable_types(self, memora_memory):
        """Storage should handle unserializable types gracefully."""
        import threading

        # Un objeto que no es JSON serializable
        unserializable = threading.Lock()

        # No debería crashear, solo loggear error
        memora_memory.store(
            query="test with unserializable",
            context={"agent": "test"},
            result=unserializable,
        )

        # El entry no debería haberse guardado
        result = memora_memory.lookup("test with unserializable", {"agent": "test"})
        assert not result.is_hit


class TestLookupAndStore:
    """Lookup and store operation tests."""

    def test_lookup_miss_empty_cache(self, memora_memory):
        """Lookup on empty cache should return miss."""
        result = memora_memory.lookup("test query", {"agent": "test"})

        assert result.is_miss
        assert result.result is None
        assert result.similarity is None

    def test_store_and_lookup_exact(self, memora_memory):
        """Store followed by exact lookup should HIT."""
        query = "What is Python?"
        context = {"agent": "llm"}
        expected = "Python is a programming language"

        memora_memory.store(query, context, expected)
        assert memora_memory.backend.count() == 1

        result = memora_memory.lookup(query, context)

        assert result.is_hit
        assert result.result == expected
        assert result.similarity >= 0.99
        assert result.matched_query == query

    def test_store_and_lookup_semantic(self, memora_memory):
        """Semantic similar query should HIT."""
        memora_memory.store(
            "What is machine learning?",
            {"agent": "test"},
            "Machine learning explanation",
        )

        result = memora_memory.lookup("Explain machine learning", {"agent": "test"})

        assert result.is_hit
        assert result.result == "Machine learning explanation"
        assert result.similarity > 0.75

    def test_lookup_different_context_miss(self, memora_memory):
        """Different context should cause MISS."""
        memora_memory.store("test", {"agent": "A"}, "result A")
        result = memora_memory.lookup("test", {"agent": "B"})

        assert result.is_miss

    def test_store_dict_result(self, memora_memory):
        """Store and retrieve dict data."""
        data = {"key": "value", "number": 42}

        memora_memory.store("query", {"agent": "test"}, data)
        result = memora_memory.lookup("query", {"agent": "test"})

        assert result.is_hit
        assert result.result == data

    def test_store_list_result(self, memora_memory):
        """Store and retrieve list data."""
        data = [1, 2, 3, "text", {"nested": "dict"}]

        memora_memory.store("query", {"agent": "test"}, data)
        result = memora_memory.lookup("query", {"agent": "test"})

        assert result.is_hit
        assert result.result == data

    def test_lookup_below_threshold(self, memora_memory):
        """Query below similarity threshold should MISS."""
        memora_memory.store("What is Python?", {"agent": "test"}, "Python explanation")

        result = memora_memory.lookup("What is the weather?", {"agent": "test"})

        assert result.is_miss

    def test_lookup_custom_threshold(self, memora_memory):
        """Custom threshold should be respected."""
        memora_memory.store(
            "What is machine learning?",
            {"agent": "test"},
            "Machine learning explanation",
        )

        result = memora_memory.lookup(
            "Explain machine learning", {"agent": "test"}, similarity_threshold=0.6
        )

        assert result.is_hit

    def test_store_large_dataframe(self, memora_memory):
        """Storage should handle large DataFrames with Arrow IPC."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        # DataFrame grande (> 100KB)
        large_df = pd.DataFrame({"col1": range(10000), "col2": ["text" * 10] * 10000})

        # Debería guardarse sin problemas
        memora_memory.store(
            query="Get large dataset", context={"agent": "test"}, result=large_df
        )

        # Debería recuperarse correctamente
        result = memora_memory.lookup("Get large dataset", {"agent": "test"})
        assert result.is_hit
        assert isinstance(result.result, pd.DataFrame)
        assert len(result.result) == 10000


class TestSizeLimitsAndEviction:
    """Size limits and eviction policy tests."""

    def test_max_entries_triggers_eviction(self):
        """Storing beyond max_entries should evict oldest entry."""
        config = CacheConfig(
            db_uri="memory://",
            max_entries=3,
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        for i in range(4):
            memora.store(f"query {i}", {"agent": "test"}, f"result {i}")
            time.sleep(0.01)

        assert memora.backend.count() == 3

        result = memora.lookup("query 0", {"agent": "test"})
        assert result.is_miss

        for i in range(1, 4):
            result = memora.lookup(f"query {i}", {"agent": "test"})
            assert result.is_hit


class TestTTLAndCleanup:
    """TTL and cleanup tests."""

    def test_cleanup_expired_entries(self):
        """Cleanup should remove expired entries."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=1,
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        memora.store("test", {"agent": "test"}, "result")
        time.sleep(1.1)

        deleted = memora.cleanup_expired()

        assert deleted == 1
        assert memora.backend.count() == 0

    def test_lookup_respects_ttl(self):
        """Lookup should not return expired entries."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=0.5,
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        memora.store("test", {"agent": "test"}, "result")

        result = memora.lookup("test", {"agent": "test"})
        assert result.is_hit

        time.sleep(0.6)

        result = memora.lookup("test", {"agent": "test"})
        assert result.is_miss


class TestInvalidation:
    """Invalidation tests."""

    def test_invalidate_by_context(self, memora_memory):
        """Invalidate by context should remove matching entries."""
        memora_memory.store("q1", {"agent": "A"}, "r1")
        memora_memory.store("q2", {"agent": "B"}, "r2")
        memora_memory.store("q3", {"agent": "A"}, "r3")

        deleted = memora_memory.invalidate(context={"agent": "A"})

        assert deleted == 2
        assert memora_memory.backend.count() == 1

    def test_invalidate_by_age(self, memora_memory):
        """Invalidate by age should remove old entries."""
        memora_memory.store("old query", {"agent": "test"}, "old result")

        time.sleep(0.1)

        deleted = memora_memory.invalidate(older_than_seconds=0.05)

        assert deleted == 1
        assert memora_memory.backend.count() == 0

    def test_invalidate_without_criteria(self, memora_memory):
        """Invalidate without criteria should do nothing."""
        memora_memory.store("test", {"agent": "test"}, "result")

        deleted = memora_memory.invalidate()

        assert deleted == 0
        assert memora_memory.backend.count() == 1


class TestAvailabilityCheck:
    """Availability check tests."""

    def test_check_availability_miss(self, memora_memory):
        """Check availability for missing entry should return unavailable."""
        check = memora_memory.check_availability("test", {"agent": "test"})

        assert not check.available

    def test_check_availability_hit(self, memora_memory):
        """Check availability for existing entry should return available."""
        memora_memory.store("test", {"agent": "test"}, "result")

        check = memora_memory.check_availability("test", {"agent": "test"})

        assert check.available
        assert check.age_seconds is not None
        assert check.similarity is not None

    def test_check_availability_with_ttl(self):
        """Check availability should include TTL remaining."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=10,
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        memora.store("test", {"agent": "test"}, "result")

        check = memora.check_availability("test", {"agent": "test"})

        assert check.available
        assert check.ttl_remaining_seconds is not None
        assert check.ttl_remaining_seconds <= 10


class TestStatistics:
    """Statistics and metrics tests."""

    def test_get_stats_with_data(self, memora_memory):
        """Stats should include hits and misses."""
        memora_memory.store(
            "What is Python programming?", {"agent": "test"}, "it's something cool"
        )
        memora_memory.store("How to bake a cake?", {"agent": "test"}, "With love")
        memora_memory.lookup("Explain me what's python about", {"agent": "test"})
        memora_memory.lookup("What is the weather in Almeria?", {"agent": "test"})

        stats = memora_memory.get_stats()

        assert stats["total_entries"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "50.00%"

    def test_metrics_track_latency(self, memora_memory):
        """Metrics should track lookup latency."""
        memora_memory.store("test", {"agent": "test"}, "result")
        memora_memory.lookup("test", {"agent": "test"})

        stats = memora_memory.get_stats()

        assert "lookup_latency_ms" in stats
        assert stats["lookup_latency_ms"]["samples"] >= 1


class TestHealthCheck:
    """Health check tests."""

    def test_health_check_healthy(self, memora_memory):
        """Health check should report healthy status."""
        health = memora_memory.health_check()

        assert health["status"] == "healthy"
        assert health["checks"]["embedding"]["ok"]
        assert health["checks"]["database"]["ok"]
        assert "timestamp" in health

    def test_health_check_includes_metrics(self, memora_memory):
        """Health check should include metrics."""
        memora_memory.store("test", {"agent": "test"}, "result")
        memora_memory.lookup("test", {"agent": "test"})

        health = memora_memory.health_check()

        assert "metrics" in health
        assert health["metrics"]["total_entries"] == 1
        assert "recent_errors" in health["metrics"]


class TestIndex:
    """Vector index tests."""

    def test_create_index_insufficient_entries(self, memora_memory):
        """Should warn if insufficient entries for index."""
        memora_memory.create_index()

        assert not memora_memory.backend.has_index()

    def test_get_index_stats(self, memora_memory):
        """Should return index stats."""
        stats = memora_memory.get_index_stats()

        assert "has_index" in stats
        assert "total_entries" in stats
