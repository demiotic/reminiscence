"""Tests for eviction policies (FIFO, LRU, LFU)."""

import time
import pytest

from reminiscence import ReminiscenceConfig
from reminiscence.cache import CacheOperations
from reminiscence.embeddings import create_embedder
from reminiscence.storage import create_storage_backend
from reminiscence.eviction import create_eviction_policy
from reminiscence.metrics import CacheMetrics


def create_ops(eviction_policy: str, max_entries: int = 3):
    """Helper to create CacheOperations with specific eviction policy."""
    config = ReminiscenceConfig(
        db_uri="memory://",
        max_entries=max_entries,
        eviction_policy=eviction_policy,
        enable_metrics=True,
        similarity_threshold=0.95,
        log_level="WARNING",
    )

    embedder = create_embedder(config)
    storage = create_storage_backend(config, embedder.embedding_dim)
    eviction = create_eviction_policy(eviction_policy)
    metrics = CacheMetrics()

    return CacheOperations(storage, embedder, eviction, config, metrics)


class TestFIFOPolicy:
    """Test FIFO (First In First Out) eviction."""

    def test_fifo_evicts_oldest(self):
        """FIFO should evict the first inserted entry."""
        ops = create_ops("fifo", max_entries=3)

        # Store 3 entries
        ops.store("query1", {"agent": "test"}, "result1")
        time.sleep(0.01)
        ops.store("query2", {"agent": "test"}, "result2")
        time.sleep(0.01)
        ops.store("query3", {"agent": "test"}, "result3")

        assert ops.storage.count() == 3

        # Store 4th entry - should evict query1 (oldest)
        time.sleep(0.01)
        ops.store("query4", {"agent": "test"}, "result4")

        assert ops.storage.count() == 3

        # query1 should be gone
        result1 = ops.lookup("query1", {"agent": "test"})
        assert result1.is_miss

        # query2, query3, query4 should exist
        result2 = ops.lookup("query2", {"agent": "test"})
        assert result2.is_hit

        result3 = ops.lookup("query3", {"agent": "test"})
        assert result3.is_hit

        result4 = ops.lookup("query4", {"agent": "test"})
        assert result4.is_hit

    def test_fifo_ignores_access_patterns(self):
        """FIFO should not care about access frequency or recency."""
        ops = create_ops("fifo", max_entries=2)

        # Use semantically distinct queries
        ops.store("database optimization techniques", {"agent": "test"}, "result_db")
        time.sleep(0.01)
        ops.store("cloud computing architecture", {"agent": "test"}, "result_cloud")

        # Access "database" query multiple times
        for _ in range(10):
            ops.lookup("database optimization techniques", {"agent": "test"})

        # Store another entry - should still evict "database" (first in)
        time.sleep(0.01)
        ops.store("microservices design patterns", {"agent": "test"}, "result_micro")

        # "database" should be evicted despite being accessed many times
        result = ops.lookup("database optimization techniques", {"agent": "test"})
        assert result.is_miss


class TestLRUPolicy:
    """Test LRU (Least Recently Used) eviction."""

    def test_lru_evicts_least_recently_used(self):
        """LRU should evict the entry that hasn't been accessed recently."""
        ops = create_ops("lru", max_entries=3)

        # Store 3 entries
        ops.store("query1", {"agent": "test"}, "result1")
        time.sleep(0.01)
        ops.store("query2", {"agent": "test"}, "result2")
        time.sleep(0.01)
        ops.store("query3", {"agent": "test"}, "result3")
        time.sleep(0.01)

        # Access query2 and query3 (query1 is least recently used)
        ops.lookup("query2", {"agent": "test"})
        time.sleep(0.01)
        ops.lookup("query3", {"agent": "test"})
        time.sleep(0.01)

        # Store 4th entry - should evict query1 (LRU)
        ops.store("query4", {"agent": "test"}, "result4")

        assert ops.storage.count() == 3

        # query1 should be evicted (least recently used)
        result1 = ops.lookup("query1", {"agent": "test"})
        assert result1.is_miss

        # query2, query3, query4 should exist
        result2 = ops.lookup("query2", {"agent": "test"})
        assert result2.is_hit

        result3 = ops.lookup("query3", {"agent": "test"})
        assert result3.is_hit

        result4 = ops.lookup("query4", {"agent": "test"})
        assert result4.is_hit

    def test_lru_updates_on_access(self):
        """LRU should update access time on cache hits."""
        ops = create_ops("lru", max_entries=2)

        # Use semantically distinct queries to avoid false matches
        ops.store("What is machine learning?", {"agent": "test"}, "result_ml")
        time.sleep(0.01)
        ops.store(
            "Explain quantum computing concepts", {"agent": "test"}, "result_quantum"
        )
        time.sleep(0.01)

        # Access ML query to make it recently used
        ops.lookup("What is machine learning?", {"agent": "test"})
        time.sleep(0.01)

        # Store another - should evict quantum (now least recently used)
        ops.store(
            "How does blockchain technology work?",
            {"agent": "test"},
            "result_blockchain",
        )

        # quantum should be evicted
        result_quantum = ops.lookup(
            "Explain quantum computing concepts", {"agent": "test"}
        )
        assert result_quantum.is_miss

        # ML and blockchain should exist
        result_ml = ops.lookup("What is machine learning?", {"agent": "test"})
        assert result_ml.is_hit

        result_blockchain = ops.lookup(
            "How does blockchain technology work?", {"agent": "test"}
        )
        assert result_blockchain.is_hit


class TestLFUPolicy:
    """Test LFU (Least Frequently Used) eviction."""

    def test_lfu_evicts_least_frequently_used(self):
        """LFU should evict the entry with lowest access count."""
        ops = create_ops("lfu", max_entries=3)

        # Store 3 entries
        ops.store("rarely_used", {"agent": "test"}, "result1")
        ops.store("sometimes_used", {"agent": "test"}, "result2")
        ops.store("frequently_used", {"agent": "test"}, "result3")

        # Access with different frequencies
        # rarely_used: 1 access
        ops.lookup("rarely_used", {"agent": "test"})

        # sometimes_used: 3 accesses
        for _ in range(3):
            ops.lookup("sometimes_used", {"agent": "test"})

        # frequently_used: 10 accesses
        for _ in range(10):
            ops.lookup("frequently_used", {"agent": "test"})

        time.sleep(0.01)

        # Store 4th entry - should evict "rarely_used" (lowest frequency)
        ops.store("new_entry", {"agent": "test"}, "result4")

        assert ops.storage.count() == 3

        # rarely_used should be evicted
        result1 = ops.lookup("rarely_used", {"agent": "test"})
        assert result1.is_miss

        # Others should exist
        result2 = ops.lookup("sometimes_used", {"agent": "test"})
        assert result2.is_hit

        result3 = ops.lookup("frequently_used", {"agent": "test"})
        assert result3.is_hit

        result4 = ops.lookup("new_entry", {"agent": "test"})
        assert result4.is_hit

    def test_lfu_tracks_access_frequency(self):
        """LFU should increment frequency counter on each access."""
        ops = create_ops("lfu", max_entries=2)

        ops.store("low_freq", {"agent": "test"}, "result1")
        ops.store("high_freq", {"agent": "test"}, "result2")

        # Access high_freq many times
        for _ in range(20):
            ops.lookup("high_freq", {"agent": "test"})

        # Access low_freq only once
        ops.lookup("low_freq", {"agent": "test"})

        time.sleep(0.01)

        # Store another - should evict low_freq
        ops.store("new", {"agent": "test"}, "result3")

        # low_freq should be evicted
        result = ops.lookup("low_freq", {"agent": "test"})
        assert result.is_miss

        # high_freq should still exist
        result = ops.lookup("high_freq", {"agent": "test"})
        assert result.is_hit

    def test_lfu_new_entries_start_at_zero(self):
        """New entries should start with frequency 0."""
        ops = create_ops("lfu", max_entries=2)

        # Store entry and access it
        ops.store("accessed", {"agent": "test"}, "result1")
        ops.lookup("accessed", {"agent": "test"})  # frequency = 1

        # Store new entry (frequency = 0)
        ops.store("new", {"agent": "test"}, "result2")

        # Store another - should evict "new" (frequency 0 < 1)
        ops.store("another", {"agent": "test"}, "result3")

        # "new" should be evicted
        result = ops.lookup("new", {"agent": "test"})
        assert result.is_miss

        # "accessed" should still exist
        result = ops.lookup("accessed", {"agent": "test"})
        assert result.is_hit


class TestEvictionPolicyComparison:
    """Compare behavior across policies."""

    def test_same_entries_different_evictions(self):
        """Same access pattern should produce different evictions."""

        # Setup: Store 2 entries, access first one, then add 3rd
        def run_scenario(policy: str):
            ops = create_ops(policy, max_entries=2)

            ops.store("first", {"agent": "test"}, "r1")
            time.sleep(0.01)
            ops.store("second", {"agent": "test"}, "r2")
            time.sleep(0.01)

            # Access "first" multiple times
            for _ in range(5):
                ops.lookup("first", {"agent": "test"})
            time.sleep(0.01)

            # Store 3rd entry
            ops.store("third", {"agent": "test"}, "r3")

            # Check which entries remain
            has_first = ops.lookup("first", {"agent": "test"}).is_hit
            has_second = ops.lookup("second", {"agent": "test"}).is_hit
            has_third = ops.lookup("third", {"agent": "test"}).is_hit

            return has_first, has_second, has_third

        fifo_result = run_scenario("fifo")
        lru_result = run_scenario("lru")
        lfu_result = run_scenario("lfu")

        # FIFO: evicts "first" (oldest), keeps second and third
        assert fifo_result == (False, True, True)

        # LRU: evicts "second" (least recently used), keeps first and third
        assert lru_result == (True, False, True)

        # LFU: evicts "second" (least frequently used), keeps first and third
        assert lfu_result == (True, False, True)


class TestEvictionEdgeCases:
    """Test edge cases for all policies."""

    @pytest.mark.parametrize("policy", ["fifo", "lru", "lfu"])
    def test_eviction_with_single_entry_limit(self, policy):
        """Eviction should work with max_entries=1."""
        ops = create_ops(policy, max_entries=1)

        ops.store("first", {"agent": "test"}, "r1")
        assert ops.storage.count() == 1

        ops.store("second", {"agent": "test"}, "r2")
        assert ops.storage.count() == 1

        # First should be evicted
        result = ops.lookup("first", {"agent": "test"})
        assert result.is_miss

    @pytest.mark.parametrize("policy", ["fifo", "lru", "lfu"])
    def test_no_eviction_below_limit(self, policy):
        """No eviction should happen below max_entries."""
        ops = create_ops(policy, max_entries=10)

        for i in range(5):
            ops.store(f"query{i}", {"agent": "test"}, f"result{i}")

        assert ops.storage.count() == 5

        # All entries should still be accessible
        for i in range(5):
            result = ops.lookup(f"query{i}", {"agent": "test"})
            assert result.is_hit

    @pytest.mark.parametrize("policy", ["fifo", "lru", "lfu"])
    def test_eviction_state_syncs_on_init(self, policy):
        """Eviction policy should sync with existing entries on init."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            max_entries=3,
            eviction_policy=policy,
            log_level="WARNING",
        )

        embedder = create_embedder(config)
        storage = create_storage_backend(config, embedder.embedding_dim)
        eviction = create_eviction_policy(policy)
        metrics = CacheMetrics()

        # Create first ops instance and add entries
        ops1 = CacheOperations(storage, embedder, eviction, config, metrics)
        ops1.store("q1", {"agent": "test"}, "r1")
        ops1.store("q2", {"agent": "test"}, "r2")

        # Create new eviction policy and ops (simulates restart)
        eviction2 = create_eviction_policy(policy)
        ops2 = CacheOperations(storage, embedder, eviction2, config, metrics)

        # Should have synced existing entries
        # Add one more to trigger eviction
        ops2.store("q3", {"agent": "test"}, "r3")
        ops2.store("q4", {"agent": "test"}, "r4")  # Should trigger eviction

        # Should have 3 entries (one was evicted)
        assert ops2.storage.count() == 3


# ============================================================
# TESTS ORIGINALES ADAPTADOS
# ============================================================


class TestCacheOperationsLookup:
    """Test lookup functionality (from original tests)."""

    def test_lookup_empty_cache(self):
        """Lookup on empty cache should miss."""
        ops = create_ops("fifo")
        result = ops.lookup("test", {"agent": "test"})

        assert result.is_miss
        assert ops.metrics.misses == 1

    def test_lookup_after_store(self):
        """Lookup after store should hit."""
        ops = create_ops("fifo")
        ops.store("query", {"agent": "test"}, "result")
        result = ops.lookup("query", {"agent": "test"})

        assert result.is_hit
        assert result.result == "result"
        assert ops.metrics.hits == 1

    def test_lookup_semantic_similarity(self):
        """Should match semantically similar queries."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            eviction_policy="fifo",
            max_entries=10,
            similarity_threshold=0.80,
            log_level="DEBUG",
        )

        embedder = create_embedder(config)
        backend = create_storage_backend(config, embedder.embedding_dim)
        eviction = create_eviction_policy("fifo")

        ops = CacheOperations(
            storage=backend,
            embedder=embedder,
            eviction=eviction,
            config=config,
            metrics=None,
        )

        ops.store(
            "What is machine learning and how does it work?",
            {"agent": "test"},
            "ML explanation",
        )

        # Query similar
        result = ops.lookup(
            "Explain the concept of machine learning", {"agent": "test"}
        )

        assert result.is_hit
        assert result.similarity > 0.80
        assert "ML explanation" in result.result


class TestCacheOperationsStore:
    """Test store functionality (from original tests)."""

    def test_store_basic(self):
        """Basic store should work."""
        ops = create_ops("fifo")
        ops.store("query", {"agent": "test"}, "result")

        assert ops.storage.count() == 1

    def test_store_with_metadata(self):
        """Store with metadata should work."""
        ops = create_ops("fifo")
        metadata = {"tokens": 100, "cost": 0.001}
        ops.store("query", {"agent": "test"}, "result", metadata=metadata)

        assert ops.storage.count() == 1

    @pytest.mark.parametrize("policy", ["fifo", "lru", "lfu"])
    def test_store_triggers_eviction_all_policies(self, policy):
        """Store should evict when max_entries reached (test all policies)."""
        ops = create_ops(policy, max_entries=2)

        # Store 3 entries
        for i in range(3):
            ops.store(f"query {i}", {"agent": "test"}, f"result {i}")
            time.sleep(0.01)

        # Should only have 2
        assert ops.storage.count() == 2

    def test_store_large_data(self):
        """Should handle large data with Arrow IPC serialization."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        ops = create_ops("fifo")

        # Large DataFrame (should use Arrow IPC)
        large_df = pd.DataFrame({"col1": range(5000), "col2": ["text" * 10] * 5000})

        # Should store successfully
        ops.store("large query", {"agent": "test"}, large_df)

        assert ops.storage.count() == 1

        # Should retrieve successfully
        result = ops.lookup("large query", {"agent": "test"})
        assert result.is_hit
        assert isinstance(result.result, pd.DataFrame)
        assert len(result.result) == 5000


class TestCacheOperationsMaintenance:
    """Test maintenance operations (from original tests)."""

    def test_cleanup_expired(self):
        """Cleanup should remove expired entries."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=0.5,
            enable_metrics=True,
            log_level="WARNING",
        )

        embedder = create_embedder(config)
        storage = create_storage_backend(config, embedder.embedding_dim)
        eviction = create_eviction_policy("fifo")
        metrics = CacheMetrics()

        ops = CacheOperations(storage, embedder, eviction, config, metrics)

        # Store entry
        ops.store("query", {"agent": "test"}, "result")

        # Wait for expiration
        time.sleep(0.6)

        # Cleanup
        deleted = ops.cleanup_expired()

        assert deleted == 1
        assert storage.count() == 0

    def test_invalidate_by_context(self):
        """Invalidate by context should work."""
        ops = create_ops("fifo")

        # Store with different contexts
        ops.store("q1", {"agent": "A"}, "r1")
        ops.store("q2", {"agent": "B"}, "r2")

        # Invalidate context A
        deleted = ops.invalidate(context={"agent": "A"})

        assert deleted == 1
        assert ops.storage.count() == 1

    def test_invalidate_by_age(self):
        """Invalidate by age should work."""
        ops = create_ops("fifo")
        ops.store("old", {"agent": "test"}, "result")

        time.sleep(0.1)

        deleted = ops.invalidate(older_than_seconds=0.05)

        assert deleted == 1
        assert ops.storage.count() == 0
