"""Tests for eviction policies (FIFO, LRU, LFU)."""

import time
import pytest

from reminiscence import ReminiscenceConfig
from reminiscence.cache import CacheOperations
from reminiscence.embeddings import create_embedder
from reminiscence.storage import create_storage_backend
from reminiscence.eviction import create_eviction_policy
from reminiscence.metrics import CacheMetrics


class TestFIFOPolicy:
    """Test FIFO (First In First Out) eviction."""

    def test_fifo_evicts_oldest(self, fifo_ops):
        """FIFO should evict the first inserted entry."""
        fifo_ops.store("query1", {"agent": "test"}, "result1")
        time.sleep(0.01)
        fifo_ops.store("query2", {"agent": "test"}, "result2")
        time.sleep(0.01)
        fifo_ops.store("query3", {"agent": "test"}, "result3")

        assert fifo_ops.storage.count() == 3

        time.sleep(0.01)
        fifo_ops.store("query4", {"agent": "test"}, "result4")

        assert fifo_ops.storage.count() == 3

        assert fifo_ops.lookup("query1", {"agent": "test"}).is_miss
        assert fifo_ops.lookup("query2", {"agent": "test"}).is_hit
        assert fifo_ops.lookup("query3", {"agent": "test"}).is_hit
        assert fifo_ops.lookup("query4", {"agent": "test"}).is_hit

    def test_fifo_ignores_access_patterns(self, ops_factory):
        """FIFO should not care about access frequency or recency."""
        ops = ops_factory("fifo", max_entries=2)

        ops.store("database optimization techniques", {"agent": "test"}, "result_db")
        time.sleep(0.01)
        ops.store("cloud computing architecture", {"agent": "test"}, "result_cloud")

        for _ in range(10):
            ops.lookup("database optimization techniques", {"agent": "test"})

        time.sleep(0.01)
        ops.store("microservices design patterns", {"agent": "test"}, "result_micro")

        result = ops.lookup("database optimization techniques", {"agent": "test"})
        assert result.is_miss


class TestLRUPolicy:
    """Test LRU (Least Recently Used) eviction."""

    def test_lru_evicts_least_recently_used(self, lru_ops):
        """LRU should evict the entry that hasn't been accessed recently."""
        lru_ops.store("query1", {"agent": "test"}, "result1")
        time.sleep(0.01)
        lru_ops.store("query2", {"agent": "test"}, "result2")
        time.sleep(0.01)
        lru_ops.store("query3", {"agent": "test"}, "result3")
        time.sleep(0.01)

        lru_ops.lookup("query2", {"agent": "test"})
        time.sleep(0.01)
        lru_ops.lookup("query3", {"agent": "test"})
        time.sleep(0.01)

        lru_ops.store("query4", {"agent": "test"}, "result4")

        assert lru_ops.storage.count() == 3

        assert lru_ops.lookup("query1", {"agent": "test"}).is_miss
        assert lru_ops.lookup("query2", {"agent": "test"}).is_hit
        assert lru_ops.lookup("query3", {"agent": "test"}).is_hit
        assert lru_ops.lookup("query4", {"agent": "test"}).is_hit

    def test_lru_updates_on_access(self, ops_factory):
        """LRU should update access time on cache hits."""
        ops = ops_factory("lru", max_entries=2)

        ops.store("What is machine learning?", {"agent": "test"}, "result_ml")
        time.sleep(0.01)
        ops.store(
            "Explain quantum computing concepts", {"agent": "test"}, "result_quantum"
        )
        time.sleep(0.01)

        ops.lookup("What is machine learning?", {"agent": "test"})
        time.sleep(0.01)

        ops.store(
            "How does blockchain technology work?",
            {"agent": "test"},
            "result_blockchain",
        )

        result_quantum = ops.lookup(
            "Explain quantum computing concepts", {"agent": "test"}
        )
        assert result_quantum.is_miss

        result_ml = ops.lookup("What is machine learning?", {"agent": "test"})
        assert result_ml.is_hit

        result_blockchain = ops.lookup(
            "How does blockchain technology work?", {"agent": "test"}
        )
        assert result_blockchain.is_hit


class TestLFUPolicy:
    """Test LFU (Least Frequently Used) eviction."""

    def test_lfu_evicts_least_frequently_used(self, lfu_ops):
        """LFU should evict the entry with lowest access count."""
        lfu_ops.store("rarely_used", {"agent": "test"}, "result1")
        lfu_ops.store("sometimes_used", {"agent": "test"}, "result2")
        lfu_ops.store("frequently_used", {"agent": "test"}, "result3")

        lfu_ops.lookup("rarely_used", {"agent": "test"})

        for _ in range(3):
            lfu_ops.lookup("sometimes_used", {"agent": "test"})

        for _ in range(10):
            lfu_ops.lookup("frequently_used", {"agent": "test"})

        time.sleep(0.01)

        lfu_ops.store("new_entry", {"agent": "test"}, "result4")

        assert lfu_ops.storage.count() == 3

        assert lfu_ops.lookup("rarely_used", {"agent": "test"}).is_miss
        assert lfu_ops.lookup("sometimes_used", {"agent": "test"}).is_hit
        assert lfu_ops.lookup("frequently_used", {"agent": "test"}).is_hit
        assert lfu_ops.lookup("new_entry", {"agent": "test"}).is_hit

    def test_lfu_tracks_access_frequency(self, ops_factory):
        """LFU should increment frequency counter on each access."""
        ops = ops_factory("lfu", max_entries=2)

        ops.store("low_freq", {"agent": "test"}, "result1")
        ops.store("high_freq", {"agent": "test"}, "result2")

        for _ in range(20):
            ops.lookup("high_freq", {"agent": "test"})

        ops.lookup("low_freq", {"agent": "test"})

        time.sleep(0.01)

        ops.store("new", {"agent": "test"}, "result3")

        result = ops.lookup("low_freq", {"agent": "test"})
        assert result.is_miss

        result = ops.lookup("high_freq", {"agent": "test"})
        assert result.is_hit

    def test_lfu_new_entries_start_at_zero(self, ops_factory):
        """New entries should start with frequency 0."""
        ops = ops_factory("lfu", max_entries=2)

        ops.store("accessed", {"agent": "test"}, "result1")
        ops.lookup("accessed", {"agent": "test"})

        ops.store("new", {"agent": "test"}, "result2")

        ops.store("another", {"agent": "test"}, "result3")

        result = ops.lookup("new", {"agent": "test"})
        assert result.is_miss

        result = ops.lookup("accessed", {"agent": "test"})
        assert result.is_hit


class TestEvictionPolicyComparison:
    """Compare behavior across policies."""

    def test_same_entries_different_evictions(self, ops_factory):
        """Same access pattern should produce different evictions."""

        def run_scenario(policy: str):
            ops = ops_factory(policy, max_entries=2)

            ops.store("first", {"agent": "test"}, "r1")
            time.sleep(0.01)
            ops.store("second", {"agent": "test"}, "r2")
            time.sleep(0.01)

            for _ in range(5):
                ops.lookup("first", {"agent": "test"})
            time.sleep(0.01)

            ops.store("third", {"agent": "test"}, "r3")

            has_first = ops.lookup("first", {"agent": "test"}).is_hit
            has_second = ops.lookup("second", {"agent": "test"}).is_hit
            has_third = ops.lookup("third", {"agent": "test"}).is_hit

            return has_first, has_second, has_third

        fifo_result = run_scenario("fifo")
        lru_result = run_scenario("lru")
        lfu_result = run_scenario("lfu")

        assert fifo_result == (False, True, True)
        assert lru_result == (True, False, True)
        assert lfu_result == (True, False, True)


class TestEvictionEdgeCases:
    """Test edge cases for all policies."""

    @pytest.mark.parametrize("policy", ["fifo", "lru", "lfu"])
    def test_eviction_with_single_entry_limit(self, policy, ops_factory):
        """Eviction should work with max_entries=1."""
        ops = ops_factory(policy, max_entries=1)

        ops.store("first", {"agent": "test"}, "r1")
        assert ops.storage.count() == 1

        ops.store("second", {"agent": "test"}, "r2")
        assert ops.storage.count() == 1

        result = ops.lookup("first", {"agent": "test"})
        assert result.is_miss

    @pytest.mark.parametrize("policy", ["fifo", "lru", "lfu"])
    def test_no_eviction_below_limit(self, policy, ops_factory):
        """No eviction should happen below max_entries."""
        ops = ops_factory(policy, max_entries=10)

        for i in range(5):
            ops.store(f"query{i}", {"agent": "test"}, f"result{i}")

        assert ops.storage.count() == 5

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

        ops1 = CacheOperations(storage, embedder, eviction, config, metrics)
        ops1.store("q1", {"agent": "test"}, "r1")
        ops1.store("q2", {"agent": "test"}, "r2")

        eviction2 = create_eviction_policy(policy)
        ops2 = CacheOperations(storage, embedder, eviction2, config, metrics)

        ops2.store("q3", {"agent": "test"}, "r3")
        ops2.store("q4", {"agent": "test"}, "r4")

        assert ops2.storage.count() == 3


class TestCacheOperationsLookup:
    """Test lookup functionality."""

    def test_lookup_empty_cache(self, cache_ops):
        """Lookup on empty cache should miss."""
        result = cache_ops.lookup("test", {"agent": "test"})

        assert result.is_miss
        assert cache_ops.metrics.misses == 1

    def test_lookup_after_store(self, cache_ops):
        """Lookup after store should hit."""
        cache_ops.store("query", {"agent": "test"}, "result")
        result = cache_ops.lookup("query", {"agent": "test"})

        assert result.is_hit
        assert result.result == "result"
        assert cache_ops.metrics.hits == 1

    def test_lookup_semantic_similarity(self, cache_ops):
        """Should match semantically similar queries."""
        cache_ops.store(
            "What is machine learning and how does it work?",
            {"agent": "test"},
            "ML explanation",
        )

        result = cache_ops.lookup(
            "Explain the concept of machine learning", {"agent": "test"}
        )

        assert result.is_hit
        assert result.similarity > 0.75
        assert "ML explanation" in result.result


class TestCacheOperationsStore:
    """Test store functionality."""

    def test_store_basic(self, cache_ops):
        """Basic store should work."""
        cache_ops.store("query", {"agent": "test"}, "result")

        assert cache_ops.storage.count() == 1

    def test_store_with_metadata(self, cache_ops):
        """Store with metadata should work."""
        metadata = {"tokens": 100, "cost": 0.001}
        cache_ops.store("query", {"agent": "test"}, "result", metadata=metadata)

        assert cache_ops.storage.count() == 1

    @pytest.mark.parametrize("policy", ["fifo", "lru", "lfu"])
    def test_store_triggers_eviction_all_policies(self, policy, ops_factory):
        """Store should evict when max_entries reached."""
        ops = ops_factory(policy, max_entries=2)

        for i in range(3):
            ops.store(f"query {i}", {"agent": "test"}, f"result {i}")
            time.sleep(0.01)

        assert ops.storage.count() == 2

    def test_store_large_data(self, cache_ops):
        """Should handle large data with Arrow IPC serialization."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        large_df = pd.DataFrame({"col1": range(5000), "col2": ["text" * 10] * 5000})

        cache_ops.store("large query", {"agent": "test"}, large_df)

        assert cache_ops.storage.count() == 1

        result = cache_ops.lookup("large query", {"agent": "test"})
        assert result.is_hit
        assert isinstance(result.result, pd.DataFrame)
        assert len(result.result) == 5000


class TestCacheOperationsMaintenance:
    """Test maintenance operations."""

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

        ops.store("query", {"agent": "test"}, "result")

        time.sleep(0.6)

        deleted = ops.cleanup_expired()

        assert deleted == 1
        assert storage.count() == 0

    def test_invalidate_by_context(self, cache_ops):
        """Invalidate by context should work."""
        cache_ops.store("q1", {"agent": "A"}, "r1")
        cache_ops.store("q2", {"agent": "B"}, "r2")

        deleted = cache_ops.invalidate(context={"agent": "A"})

        assert deleted == 1
        assert cache_ops.storage.count() == 1

    def test_invalidate_by_age(self, cache_ops):
        """Invalidate by age should work."""
        cache_ops.store("old", {"agent": "test"}, "result")

        time.sleep(0.1)

        deleted = cache_ops.invalidate(older_than_seconds=0.05)

        assert deleted == 1
        assert cache_ops.storage.count() == 0
