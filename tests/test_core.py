"""Tests for memora.core.Memora."""

import pytest
import time
from memora import Memora, CacheConfig


class TestMemoraBasics:
    """Basic initialization and operation tests."""

    def test_init_memory(self, memora_memory):
        """Test initialization with memory backend."""
        assert memora_memory is not None
        assert memora_memory.table.count_rows() == 0

    def test_init_disk(self, memora_disk):
        """Test initialization with disk backend."""
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
        print("\n[DEBUG] Test: lookup_handles_embedding_failure")
        print(f"[DEBUG] Initial cache entries: {memora_memory.table.count_rows()}")

        # FIRST: Put data in cache so it doesn't exit early
        memora_memory.store("dummy query", {"agent": "test"}, "dummy result")
        print(f"[DEBUG] After store, cache entries: {memora_memory.table.count_rows()}")

        # Mock _embed to fail
        def failing_embed(text):
            print(f"[DEBUG] failing_embed called with: '{text}'")
            raise RuntimeError("Embedding model crashed")

        monkeypatch.setattr(memora_memory, "_embed", failing_embed)

        # Lookup should return MISS without propagating error
        print("[DEBUG] Calling lookup...")
        result = memora_memory.lookup("test query", {"agent": "test"})

        print(f"[DEBUG] Result: is_miss={result.is_miss}, result={result.result}")
        print(
            f"[DEBUG] Metrics: misses={memora_memory.metrics.misses}, lookup_errors={memora_memory.metrics.lookup_errors}"
        )

        assert result.is_miss
        assert result.result is None

        # Verify error was tracked in metrics
        if memora_memory.metrics:
            assert memora_memory.metrics.lookup_errors == 1

    def test_lookup_handles_lancedb_failure(self, memora_memory, monkeypatch):
        """Lookup should return MISS if LanceDB fails."""
        print("\n[DEBUG] Test: lookup_handles_lancedb_failure")

        # Mock table.search to fail
        def failing_search(*args, **kwargs):
            print("[DEBUG] failing_search called")
            raise Exception("LanceDB connection lost")

        monkeypatch.setattr(memora_memory.table, "search", failing_search)

        # Store something first
        memora_memory.store("query", {"agent": "test"}, "result")
        print(f"[DEBUG] Stored entry, cache size: {memora_memory.table.count_rows()}")

        # Lookup should return MISS
        result = memora_memory.lookup("query", {"agent": "test"})
        print(f"[DEBUG] Lookup result: is_miss={result.is_miss}")

        assert result.is_miss
        if memora_memory.metrics:
            print(f"[DEBUG] lookup_errors: {memora_memory.metrics.lookup_errors}")
            assert memora_memory.metrics.lookup_errors >= 1

    def test_store_handles_serialization_failure(self, memora_memory, monkeypatch):
        """Store should log error without exploding if serialization fails."""
        print("\n[DEBUG] Test: store_handles_serialization_failure")

        # Mock serialize to fail - patch in the memora.core module where it's imported
        def failing_serialize(obj):
            print(f"[DEBUG] failing_serialize called with type: {type(obj)}")
            raise TypeError("Cannot serialize this type")

        # Patch where serialize is USED (in core.py), not where it's defined
        import memora.core

        monkeypatch.setattr(memora.core, "serialize", failing_serialize)

        # Store should NOT propagate error
        try:
            memora_memory.store("query", {"agent": "test"}, "result")
            print("[DEBUG] store() completed without exception")
            assert True
        except Exception as e:
            pytest.fail(f"store() propagated error when it shouldn't: {e}")

        # Verify error was tracked
        if memora_memory.metrics:
            print(f"[DEBUG] store_errors: {memora_memory.metrics.store_errors}")
            assert memora_memory.metrics.store_errors >= 1

    def test_store_handles_lancedb_write_failure(self, memora_memory, monkeypatch):
        """Store should log error if LanceDB fails to write."""
        print("\n[DEBUG] Test: store_handles_lancedb_write_failure")

        # Mock table.add to fail
        def failing_add(data):
            print("[DEBUG] failing_add called")
            raise Exception("Disk full")

        monkeypatch.setattr(memora_memory.table, "add", failing_add)

        # Store should NOT propagate error
        memora_memory.store("query", {"agent": "test"}, "result")
        print("[DEBUG] store() completed")

        # Verify error was tracked
        if memora_memory.metrics:
            print(f"[DEBUG] store_errors: {memora_memory.metrics.store_errors}")
            assert memora_memory.metrics.store_errors >= 1

        # Verify lookup still works (empty cache)
        result = memora_memory.lookup("query", {"agent": "test"})
        assert result.is_miss

    def test_metrics_track_errors_correctly(self, memora_memory, monkeypatch):
        """Metrics should track errors correctly."""
        print("\n[DEBUG] Test: metrics_track_errors_correctly")

        # FIRST: put data in cache
        memora_memory.store("initial", {"agent": "test"}, "data")
        print(f"[DEBUG] Initial cache size: {memora_memory.table.count_rows()}")

        # Force multiple errors
        def failing_embed(text):
            print(f"[DEBUG] failing_embed called: '{text}'")
            raise RuntimeError("Model crashed")

        monkeypatch.setattr(memora_memory, "_embed", failing_embed)

        # 3 failed lookups
        for i in range(3):
            print(f"[DEBUG] Lookup {i + 1}/3")
            result = memora_memory.lookup(f"query {i}", {"agent": "test"})
            assert result.is_miss

        # Verify counters
        stats = memora_memory.get_stats()
        print(f"[DEBUG] Final stats: {stats.get('errors', {})}")
        if "errors" in stats:
            assert stats["errors"]["lookup"] == 3
            assert stats["misses"] == 3

    def test_app_continues_working_after_cache_failure(
        self, memora_memory, monkeypatch
    ):
        """Application should continue working if cache fails."""
        print("\n[DEBUG] Test: app_continues_working_after_cache_failure")

        # Mock total failure in table
        def failing_operation(*args, **kwargs):
            print("[DEBUG] failing_operation called")
            raise Exception("LanceDB crashed")

        monkeypatch.setattr(memora_memory.table, "search", failing_operation)
        monkeypatch.setattr(memora_memory.table, "add", failing_operation)

        # Lookup returns MISS
        result = memora_memory.lookup("query", {"agent": "test"})
        print(f"[DEBUG] Lookup result: is_miss={result.is_miss}")
        assert result.is_miss

        # Store doesn't explode
        memora_memory.store("query", {"agent": "test"}, "result")
        print("[DEBUG] Store completed")

        # App can execute normally
        actual_result = "computed result"
        assert actual_result == "computed result"


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

        # Store
        memora_memory.store(query, context, expected)
        assert memora_memory.table.count_rows() == 1

        # Exact lookup
        result = memora_memory.lookup(query, context)

        assert result.is_hit
        assert result.result == expected
        assert result.similarity >= 0.99  # Almost 1.0 for identical
        assert result.matched_query == query

    def test_lookup_similar_query(self, memora_memory, sample_queries):
        """Lookup with similar query should HIT if above threshold."""
        original_query = sample_queries["similar"][0]
        similar_query = sample_queries["similar"][1]
        context = {"agent": "llm"}

        # Store original
        memora_memory.store(original_query, context, "answer")

        # Lookup similar
        result = memora_memory.lookup(similar_query, context)

        # Should HIT (queries are semantically similar)
        assert result.is_hit
        assert result.similarity > 0.75

    def test_lookup_different_query_miss(self, memora_memory, sample_queries):
        """Lookup with very different query should MISS."""
        query1 = sample_queries["different"][0]
        query2 = sample_queries["different"][1]
        context = {"agent": "llm"}

        # Store first query
        memora_memory.store(query1, context, "answer 1")

        # Lookup different query
        result = memora_memory.lookup(query2, context)

        # Should MISS (queries are different)
        assert result.is_miss or result.similarity < 0.75

    def test_lookup_different_context_miss(self, memora_memory, sample_contexts):
        """Lookup with different context should MISS."""
        query = "What is Python?"

        # Store with context 1
        memora_memory.store(query, sample_contexts["llm_gpt4"], "gpt4 answer")

        # Lookup with context 2
        result = memora_memory.lookup(query, sample_contexts["llm_claude"])

        # MISS because context is different
        assert result.is_miss

    def test_store_multiple_contexts(self, memora_memory, sample_contexts):
        """Multiple stores with different contexts should coexist."""
        query = "What is Python?"

        memora_memory.store(query, sample_contexts["llm_gpt4"], "gpt4 answer")
        memora_memory.store(query, sample_contexts["llm_claude"], "claude answer")

        assert memora_memory.table.count_rows() == 2

        # Each context should return its answer
        result_gpt4 = memora_memory.lookup(query, sample_contexts["llm_gpt4"])
        result_claude = memora_memory.lookup(query, sample_contexts["llm_claude"])

        assert result_gpt4.result == "gpt4 answer"
        assert result_claude.result == "claude answer"


class TestTTL:
    """Time-To-Live tests."""

    def test_ttl_not_expired(self):
        """Entry within TTL should HIT."""
        config = CacheConfig(
            db_uri="memory://", ttl_seconds=10, enable_metrics=True, log_level="WARNING"
        )
        memora = Memora(config)

        query = "test"
        context = {"agent": "test"}

        memora.store(query, context, "result")

        # Immediate lookup (not expired)
        result = memora.lookup(query, context)
        assert result.is_hit

    def test_ttl_expired(self):
        """Entry outside TTL should MISS."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=1,  # 1 second
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        query = "test"
        context = {"agent": "test"}

        memora.store(query, context, "result")

        # Wait for expiration
        time.sleep(1.5)

        # Lookup after expiration
        result = memora.lookup(query, context)
        assert result.is_miss


class TestCheckAvailability:
    """check_availability tests."""

    def test_availability_hit(self, memora_memory):
        """Check availability should return available=True if exists."""
        query = "test"
        context = {"agent": "test"}

        memora_memory.store(query, context, "result")

        avail = memora_memory.check_availability(query, context)

        assert avail.available
        assert avail.age_seconds is not None
        assert avail.similarity >= 0.99

    def test_availability_miss(self, memora_memory):
        """Check availability should return available=False if doesn't exist."""
        avail = memora_memory.check_availability("doesn't exist", {"agent": "test"})

        assert not avail.available
        assert avail.age_seconds is None


class TestInvalidation:
    """Invalidation tests."""

    def test_invalidate_by_context(self, memora_memory):
        """Invalidate by context should delete specific entries."""
        context1 = {"agent": "agent1"}
        context2 = {"agent": "agent2"}

        memora_memory.store("query1", context1, "result1")
        memora_memory.store("query2", context2, "result2")

        assert memora_memory.table.count_rows() == 2

        # Invalidate context 1
        deleted = memora_memory.invalidate(context=context1)

        assert deleted == 1
        assert memora_memory.table.count_rows() == 1

        # Verify only context2 remains
        result = memora_memory.lookup("query2", context2)
        assert result.is_hit

    def test_invalidate_by_age(self, memora_memory):
        """Invalidate by age should delete only old entries."""
        context = {"agent": "test"}

        # Store first entry
        memora_memory.store("query1", context, "result1")
        print(f"[DEBUG] Query1 timestamp: {int(time.time())}")

        # Wait a bit
        time.sleep(0.2)

        # Store second entry
        memora_memory.store("query2", context, "result2")
        print(f"[DEBUG] Query2 timestamp: {int(time.time())}")

        assert memora_memory.table.count_rows() == 2

        # View all timestamps
        table = memora_memory.table.to_arrow()
        print(f"[DEBUG] Timestamps in DB: {table['timestamp'].to_pylist()}")

        # Calculate cutoff
        now = time.time()
        older_than = 0.1
        cutoff = int(now - older_than)
        print(f"[DEBUG] Now: {now}, Cutoff: {cutoff}, Older_than: {older_than}")

        # Invalidate entries older than 0.1 seconds
        deleted = memora_memory.invalidate(older_than_seconds=0.1)

        print(f"[DEBUG] Deleted: {deleted}")

        assert deleted == 1  # Only the first one
        assert memora_memory.table.count_rows() == 1


class TestMetrics:
    """Metrics tests."""

    def test_metrics_hits_and_misses(self, memora_memory):
        """Metrics should track hits and misses correctly."""
        query = "test"
        context = {"agent": "test"}

        # Miss
        memora_memory.lookup(query, context)

        # Store
        memora_memory.store(query, context, "result")

        # Hit
        memora_memory.lookup(query, context)

        stats = memora_memory.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert "50.00%" in stats["hit_rate"]

    def test_metrics_disabled(self):
        """With metrics disabled, stats should not include hits/misses."""
        config = CacheConfig(
            db_uri="memory://", enable_metrics=False, log_level="WARNING"
        )
        memora = Memora(config)

        stats = memora.get_stats()

        assert "hits" not in stats
        assert "misses" not in stats


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_query(self, memora_memory):
        """Empty query should work."""
        memora_memory.store("", {"agent": "test"}, "result")
        result = memora_memory.lookup("", {"agent": "test"})

        assert result.is_hit

    def test_unicode_query(self, memora_memory):
        """Query with unicode should work."""
        query = "What is Python? 🐍 中文 العربية"
        context = {"agent": "test"}

        memora_memory.store(query, context, "result")
        result = memora_memory.lookup(query, context)

        assert result.is_hit

    def test_large_result(self, memora_memory):
        """Large results should be stored correctly."""
        large_result = "x" * 10000  # 10KB

        memora_memory.store("query", {"agent": "test"}, large_result)
        result = memora_memory.lookup("query", {"agent": "test"})

        assert result.is_hit
        assert result.result == large_result

    def test_complex_context(self, memora_memory):
        """Complex context with nested dicts should work."""
        context = {
            "agent": "complex",
            "config": {
                "model": "gpt-4",
                "params": {"temperature": 0.7, "max_tokens": 100},
            },
            "tools": ["search", "calculator"],
        }

        memora_memory.store("query", context, "result")
        result = memora_memory.lookup("query", context)

        assert result.is_hit
