"""Tests for reminiscence.core.Reminiscence."""

import pytest
import time
from reminiscence import Reminiscence, ReminiscenceConfig


# ============================================================================
# SESSION FIXTURES (load model once, reuse across tests)
# ============================================================================


@pytest.fixture(scope="module")
def reminiscence_session():
    """Shared Reminiscence instance for entire test session (loads model once)."""
    config = ReminiscenceConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
    )
    return Reminiscence(config)


# ============================================================================
# TESTS
# ============================================================================


class TestReminiscenceBasics:
    """Basic initialization and operation tests."""

    def test_init_memory(self, reminiscence):
        """Test initialization with memory backend."""
        assert reminiscence is not None
        assert reminiscence.backend.count() == 0

    def test_init_disk(self, temp_cache_dir):
        """Test initialization with disk backend."""
        from pathlib import Path

        config = ReminiscenceConfig(
            db_uri=str(Path(temp_cache_dir) / "test.db"), log_level="WARNING"
        )
        reminiscence_disk = Reminiscence(config)
        assert reminiscence_disk is not None
        assert reminiscence_disk.config.db_uri != "memory://"

    def test_get_stats_empty(self, reminiscence):
        """Test statistics with empty cache."""
        stats = reminiscence.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestErrorHandling:
    """Graceful error handling tests."""

    def test_lookup_handles_embedding_failure(self, reminiscence, monkeypatch):
        """Lookup should return MISS if embedding fails."""
        reminiscence.store("dummy query", {"agent": "test"}, "dummy result")

        def failing_embed(text):
            raise RuntimeError("Embedding model crashed")

        monkeypatch.setattr(reminiscence.embedder, "embed", failing_embed)

        result = reminiscence.lookup("test query", {"agent": "test"})

        assert result.is_miss
        assert result.result is None
        assert reminiscence.metrics.lookup_errors >= 1

    def test_store_handles_unserializable_types(self, reminiscence):
        """Storage should handle unserializable types gracefully."""
        import threading

        # Un objeto que no es JSON serializable
        unserializable = threading.Lock()

        # No debería crashear, solo loggear error
        reminiscence.store(
            query="test with unserializable",
            context={"agent": "test"},
            result=unserializable,
        )

        # El entry no debería haberse guardado
        result = reminiscence.lookup("test with unserializable", {"agent": "test"})
        assert not result.is_hit


class TestLookupAndStore:
    """Lookup and store operation tests."""

    def test_lookup_miss_empty_cache(self, reminiscence):
        """Lookup on empty cache should return miss."""
        result = reminiscence.lookup("test query", {"agent": "test"})

        assert result.is_miss
        assert result.result is None
        assert result.similarity is None

    def test_store_and_lookup_exact(self, reminiscence):
        """Store followed by exact lookup should HIT."""
        query = "What is Python?"
        context = {"agent": "llm"}
        expected = "Python is a programming language"

        reminiscence.store(query, context, expected)
        assert reminiscence.backend.count() == 1

        result = reminiscence.lookup(query, context)

        assert result.is_hit
        assert result.result == expected
        assert result.similarity >= 0.99
        assert result.matched_query == query

    def test_store_and_lookup_semantic(self, reminiscence):
        """Semantic similar query should HIT."""
        reminiscence.store(
            "What is machine learning?",
            {"agent": "test"},
            "Machine learning explanation",
        )

        result = reminiscence.lookup("Explain machine learning", {"agent": "test"})

        assert result.is_hit
        assert result.result == "Machine learning explanation"
        assert result.similarity > 0.75

    def test_lookup_different_context_miss(self, reminiscence):
        """Different context should cause MISS."""
        reminiscence.store("test", {"agent": "A"}, "result A")
        result = reminiscence.lookup("test", {"agent": "B"})

        assert result.is_miss

    def test_store_dict_result(self, reminiscence):
        """Store and retrieve dict data."""
        data = {"key": "value", "number": 42}

        reminiscence.store("query", {"agent": "test"}, data)
        result = reminiscence.lookup("query", {"agent": "test"})

        assert result.is_hit
        assert result.result == data

    def test_store_list_result(self, reminiscence):
        """Store and retrieve list data."""
        data = [1, 2, 3, "text", {"nested": "dict"}]

        reminiscence.store("query", {"agent": "test"}, data)
        result = reminiscence.lookup("query", {"agent": "test"})

        assert result.is_hit
        assert result.result == data

    def test_lookup_below_threshold(self, reminiscence):
        """Query below similarity threshold should MISS."""
        reminiscence.store("What is Python?", {"agent": "test"}, "Python explanation")

        result = reminiscence.lookup("What is the weather?", {"agent": "test"})

        assert result.is_miss

    def test_lookup_custom_threshold(self, reminiscence):
        """Custom threshold should be respected."""
        reminiscence.store(
            "What is machine learning?",
            {"agent": "test"},
            "Machine learning explanation",
        )

        result = reminiscence.lookup(
            "Explain machine learning", {"agent": "test"}, similarity_threshold=0.6
        )

        assert result.is_hit

    def test_store_large_dataframe(self, reminiscence):
        """Storage should handle large DataFrames with Arrow IPC."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        # DataFrame grande (> 100KB)
        large_df = pd.DataFrame({"col1": range(10000), "col2": ["text" * 10] * 10000})

        # Debería guardarse sin problemas
        reminiscence.store(
            query="Get large dataset", context={"agent": "test"}, result=large_df
        )

        # Debería recuperarse correctamente
        result = reminiscence.lookup("Get large dataset", {"agent": "test"})
        assert result.is_hit
        assert isinstance(result.result, pd.DataFrame)
        assert len(result.result) == 10000


class TestSizeLimitsAndEviction:
    """Size limits and eviction policy tests."""

    def test_max_entries_triggers_eviction(self):
        """Storing beyond max_entries should evict oldest entry."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            max_entries=3,
            enable_metrics=True,
            log_level="WARNING",
        )
        reminiscence = Reminiscence(config)

        for i in range(4):
            reminiscence.store(f"query {i}", {"agent": "test"}, f"result {i}")
            time.sleep(0.01)

        assert reminiscence.backend.count() == 3

        result = reminiscence.lookup("query 0", {"agent": "test"})
        assert result.is_miss

        for i in range(1, 4):
            result = reminiscence.lookup(f"query {i}", {"agent": "test"})
            assert result.is_hit


class TestTTLAndCleanup:
    """TTL and cleanup tests."""

    def test_cleanup_expired_entries(self):
        """Cleanup should remove expired entries."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=1,
            enable_metrics=True,
            log_level="WARNING",
        )
        reminiscence = Reminiscence(config)

        reminiscence.store("test", {"agent": "test"}, "result")
        time.sleep(1.1)

        deleted = reminiscence.cleanup_expired()

        assert deleted == 1
        assert reminiscence.backend.count() == 0

    def test_lookup_respects_ttl(self):
        """Lookup should not return expired entries."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=0.5,
            enable_metrics=True,
            log_level="WARNING",
        )
        reminiscence = Reminiscence(config)

        reminiscence.store("test", {"agent": "test"}, "result")

        result = reminiscence.lookup("test", {"agent": "test"})
        assert result.is_hit

        time.sleep(0.6)

        result = reminiscence.lookup("test", {"agent": "test"})
        assert result.is_miss


class TestInvalidation:
    """Invalidation tests."""

    def test_invalidate_by_context(self, reminiscence):
        """Invalidate by context should remove matching entries."""
        reminiscence.store("q1", {"agent": "A"}, "r1")
        reminiscence.store("q2", {"agent": "B"}, "r2")
        reminiscence.store("q3", {"agent": "A"}, "r3")

        deleted = reminiscence.invalidate(context={"agent": "A"})

        assert deleted == 2
        assert reminiscence.backend.count() == 1

    def test_invalidate_by_age(self, reminiscence):
        """Invalidate by age should remove old entries."""
        reminiscence.store("old query", {"agent": "test"}, "old result")

        time.sleep(0.1)

        deleted = reminiscence.invalidate(older_than_seconds=0.05)

        assert deleted == 1
        assert reminiscence.backend.count() == 0

    def test_invalidate_without_criteria(self, reminiscence):
        """Invalidate without criteria should do nothing."""
        reminiscence.store("test", {"agent": "test"}, "result")

        deleted = reminiscence.invalidate()

        assert deleted == 0
        assert reminiscence.backend.count() == 1


class TestAvailabilityCheck:
    """Availability check tests."""

    def test_check_availability_miss(self, reminiscence):
        """Check availability for missing entry should return unavailable."""
        check = reminiscence.check_availability("test", {"agent": "test"})

        assert not check.available

    def test_check_availability_hit(self, reminiscence):
        """Check availability for existing entry should return available."""
        reminiscence.store("test", {"agent": "test"}, "result")

        check = reminiscence.check_availability("test", {"agent": "test"})

        assert check.available
        assert check.age_seconds is not None
        assert check.similarity is not None

    def test_check_availability_with_ttl(self):
        """Check availability should include TTL remaining."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=10,
            enable_metrics=True,
            log_level="WARNING",
        )
        reminiscence = Reminiscence(config)

        reminiscence.store("test", {"agent": "test"}, "result")

        check = reminiscence.check_availability("test", {"agent": "test"})

        assert check.available
        assert check.ttl_remaining_seconds is not None
        assert check.ttl_remaining_seconds <= 10


class TestStatistics:
    """Statistics and metrics tests."""

    def test_get_stats_with_data(self, reminiscence):
        """Stats should include hits and misses."""
        reminiscence.store(
            "What is Python programming?", {"agent": "test"}, "it's something cool"
        )
        reminiscence.store("How to bake a cake?", {"agent": "test"}, "With love")
        reminiscence.lookup("Explain me what's python about", {"agent": "test"})
        reminiscence.lookup("What is the weather in Almeria?", {"agent": "test"})

        stats = reminiscence.get_stats()

        assert stats["total_entries"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "50.00%"

    def test_metrics_track_latency(self, reminiscence):
        """Metrics should track lookup latency."""
        reminiscence.store("test", {"agent": "test"}, "result")
        reminiscence.lookup("test", {"agent": "test"})

        stats = reminiscence.get_stats()

        assert "lookup_latency_ms" in stats
        assert stats["lookup_latency_ms"]["samples"] >= 1


class TestHealthCheck:
    """Health check tests."""

    def test_health_check_healthy(self, reminiscence):
        """Health check should report healthy status."""
        health = reminiscence.health_check()

        assert health["status"] == "healthy"
        assert health["checks"]["embedding"]["ok"]
        assert health["checks"]["database"]["ok"]
        assert "timestamp" in health

    def test_health_check_includes_metrics(self, reminiscence):
        """Health check should include metrics."""
        reminiscence.store("test", {"agent": "test"}, "result")
        reminiscence.lookup("test", {"agent": "test"})

        health = reminiscence.health_check()

        assert "metrics" in health
        assert health["metrics"]["total_entries"] == 1
        assert "recent_errors" in health["metrics"]


class TestIndex:
    """Vector index tests."""

    def test_create_index_insufficient_entries(self, reminiscence):
        """Should warn if insufficient entries for index."""
        reminiscence.create_index()

        assert not reminiscence.backend.has_index()

    def test_get_index_stats(self, reminiscence):
        """Should return index stats."""
        stats = reminiscence.get_index_stats()

        assert "has_index" in stats
        assert "total_entries" in stats
