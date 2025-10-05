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


class TestSizeLimitsAndEviction:
    """Size limits and eviction policy tests."""

    def test_max_entries_triggers_eviction(self):
        """Storing beyond max_entries should evict oldest entry."""
        print("\n[DEBUG] Test: max_entries_triggers_eviction")

        config = CacheConfig(
            db_uri="memory://",
            max_entries=3,
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        # Store 4 entries with small delays to ensure different timestamps
        for i in range(4):
            memora.store(f"query {i}", {"agent": "test"}, f"result {i}")
            print(f"[DEBUG] Stored entry {i}, cache size: {memora.table.count_rows()}")
            time.sleep(0.01)  # Ensure different timestamps

        # Should only have 3 (oldest evicted)
        assert memora.table.count_rows() == 3
        print(f"[DEBUG] Final cache size: {memora.table.count_rows()}")

        # First entry (query 0) should be gone
        result = memora.lookup("query 0", {"agent": "test"})
        assert result.is_miss
        print("[DEBUG] Query 0 correctly evicted (MISS)")

        # Entries 1, 2, 3 should still exist
        for i in range(1, 4):
            result = memora.lookup(f"query {i}", {"agent": "test"})
            assert result.is_hit
            assert result.result == f"result {i}"
            print(f"[DEBUG] Query {i} still in cache (HIT)")

    def test_oversized_payload_rejected(self):
        """Payloads larger than max_result_size_bytes should be rejected."""
        print("\n[DEBUG] Test: oversized_payload_rejected")

        config = CacheConfig(
            db_uri="memory://",
            max_result_size_bytes=1000,  # 1KB limit
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        # Try to store 10KB payload (should be rejected)
        large_payload = "x" * 10000
        print(f"[DEBUG] Attempting to store {len(large_payload)} bytes")

        memora.store("query", {"agent": "test"}, large_payload)

        # Should not be stored
        assert memora.table.count_rows() == 0
        print(f"[DEBUG] Cache size after rejected store: {memora.table.count_rows()}")

        # Error should be tracked
        assert memora.metrics.store_errors >= 1
        print(f"[DEBUG] store_errors: {memora.metrics.store_errors}")

        # Lookup should return MISS
        result = memora.lookup("query", {"agent": "test"})
        assert result.is_miss

    def test_normal_sized_payload_accepted(self):
        """Payloads within size limit should be accepted."""
        print("\n[DEBUG] Test: normal_sized_payload_accepted")

        config = CacheConfig(
            db_uri="memory://",
            max_result_size_bytes=10000,  # 10KB limit
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        # Store 5KB payload (should be accepted)
        normal_payload = "x" * 5000
        print(f"[DEBUG] Storing {len(normal_payload)} bytes")

        memora.store("query", {"agent": "test"}, normal_payload)

        # Should be stored
        assert memora.table.count_rows() == 1
        print(f"[DEBUG] Cache size: {memora.table.count_rows()}")

        # Lookup should return HIT
        result = memora.lookup("query", {"agent": "test"})
        assert result.is_hit
        assert result.result == normal_payload

    def test_eviction_preserves_newest_entries(self):
        """FIFO eviction should keep newest entries."""
        print("\n[DEBUG] Test: eviction_preserves_newest_entries")

        config = CacheConfig(
            db_uri="memory://",
            max_entries=5,
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        # Use semantically DISTINCT queries to test eviction logic, not semantic matching
        queries = [
            "Paris is the capital of France",
            "Machine learning uses neural networks",
            "The ocean contains saltwater",
            "Python is a programming language",
            "Photosynthesis converts light to energy",
            "Earth orbits around the Sun",
            "DNA carries genetic information",
            "Shakespeare wrote Hamlet",
            "Mount Everest is the tallest mountain",
            "Water boils at 100 degrees Celsius",
        ]

        for i, query in enumerate(queries):
            memora.store(query, {"agent": "test"}, f"result {i}")
            time.sleep(0.01)
            print(f"[DEBUG] Stored entry {i}, cache size: {memora.table.count_rows()}")

        # Should only have 5 entries
        assert memora.table.count_rows() == 5

        # First 5 entries (0-4) should be evicted
        for i in range(5):
            result = memora.lookup(queries[i], {"agent": "test"})
            assert result.is_miss
            print(f"[DEBUG] Query {i} evicted (MISS)")

        # Last 5 entries (5-9) should remain
        for i in range(5, 10):
            result = memora.lookup(queries[i], {"agent": "test"})
            assert result.is_hit
            assert result.result == f"result {i}"
            print(f"[DEBUG] Query {i} preserved (HIT)")

    def test_max_entries_none_allows_unlimited(self):
        """Setting max_entries=None should allow unlimited entries."""
        print("\n[DEBUG] Test: max_entries_none_allows_unlimited")

        config = CacheConfig(
            db_uri="memory://",
            max_entries=None,  # Unlimited
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        # Store 100 entries
        for i in range(100):
            memora.store(f"query {i}", {"agent": "test"}, f"result {i}")

        # All should be stored
        assert memora.table.count_rows() == 100
        print(f"[DEBUG] Cache size: {memora.table.count_rows()}")

        # Random checks - all should HIT
        for i in [0, 50, 99]:
            result = memora.lookup(f"query {i}", {"agent": "test"})
            assert result.is_hit
            print(f"[DEBUG] Query {i} still in cache (HIT)")

    def test_eviction_with_different_contexts(self):
        """Eviction should work correctly across different contexts."""
        print("\n[DEBUG] Test: eviction_with_different_contexts")

        config = CacheConfig(
            db_uri="memory://",
            max_entries=3,
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        # Store entries with different contexts
        contexts = [
            {"agent": "agent1"},
            {"agent": "agent2"},
            {"agent": "agent3"},
            {"agent": "agent4"},  # This will trigger eviction
        ]

        for i, ctx in enumerate(contexts):
            memora.store(f"query {i}", ctx, f"result {i}")
            time.sleep(0.01)
            print(
                f"[DEBUG] Stored entry {i} with context {ctx}, size: {memora.table.count_rows()}"
            )

        # Should only have 3 entries
        assert memora.table.count_rows() == 3

        # First entry should be evicted
        result = memora.lookup("query 0", contexts[0])
        assert result.is_miss

        # Last 3 should remain
        for i in range(1, 4):
            result = memora.lookup(f"query {i}", contexts[i])
            assert result.is_hit

    def test_stats_include_size_limits(self):
        """get_stats() should include size limit information."""
        print("\n[DEBUG] Test: stats_include_size_limits")

        config = CacheConfig(
            db_uri="memory://",
            max_entries=1000,
            max_result_size_bytes=5000000,
            enable_metrics=True,
        )
        memora = Memora(config)

        stats = memora.get_stats()

        assert "max_entries" in stats
        assert stats["max_entries"] == 1000

        assert "max_result_size_bytes" in stats
        assert stats["max_result_size_bytes"] == 5000000

        assert "eviction_policy" in stats
        assert stats["eviction_policy"] == "fifo"

        print(
            f"[DEBUG] Stats: max_entries={stats['max_entries']}, "
            f"max_result_size_bytes={stats['max_result_size_bytes']}, "
            f"eviction_policy={stats['eviction_policy']}"
        )

    def test_eviction_edge_case_single_entry(self):
        """Eviction should work with max_entries=1."""
        print("\n[DEBUG] Test: eviction_edge_case_single_entry")

        config = CacheConfig(
            db_uri="memory://",
            max_entries=1,
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        # Store first entry
        memora.store("query 1", {"agent": "test"}, "result 1")
        assert memora.table.count_rows() == 1

        # Store second entry (should evict first)
        time.sleep(0.01)
        memora.store("query 2", {"agent": "test"}, "result 2")
        assert memora.table.count_rows() == 1

        # First should be evicted
        result1 = memora.lookup("query 1", {"agent": "test"})
        assert result1.is_miss

        # Second should remain
        result2 = memora.lookup("query 2", {"agent": "test"})
        assert result2.is_hit
        assert result2.result == "result 2"
