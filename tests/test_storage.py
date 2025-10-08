"""Tests for storage backends."""

import pytest
import tempfile
import time
from pathlib import Path

from reminiscence.storage import create_storage_backend, LanceDBBackend
from reminiscence.types import CacheEntry
from reminiscence import ReminiscenceConfig


class TestStorageFactory:
    """Test storage factory."""

    def test_create_storage_memory(self):
        """Should create memory storage."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = create_storage_backend(config, embedding_dim=384)

        assert isinstance(storage, LanceDBBackend)
        assert storage.count() == 0

    def test_create_storage_disk(self):
        """Should create disk storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReminiscenceConfig(db_uri=str(Path(tmpdir) / "test.db"))
            storage = create_storage_backend(config, embedding_dim=384)

            assert isinstance(storage, LanceDBBackend)
            assert storage.count() == 0


class TestLanceDBBackend:
    """Test LanceDB implementation."""

    def test_storage_singleton_per_uri(self):
        """Test that storage backend is singleton per db_uri."""
        from reminiscence.storage.lancedb import LanceDBBackend
        from reminiscence.config import ReminiscenceConfig

        config1 = ReminiscenceConfig(db_uri="memory://shared")
        config2 = ReminiscenceConfig(db_uri="memory://shared")
        config3 = ReminiscenceConfig(db_uri="memory://other")

        # Same URI → same instance
        backend1 = LanceDBBackend(config1, embedding_dim=384)
        backend2 = LanceDBBackend(config2, embedding_dim=384)

        assert backend1 is backend2

        # Different URI → different instance
        backend3 = LanceDBBackend(config3, embedding_dim=384)

        assert backend1 is not backend3
        assert backend2 is not backend3

    def test_storage_shared_between_caches(self):
        """Test that multiple caches can share the same storage."""
        from reminiscence import Reminiscence, ReminiscenceConfig

        config = ReminiscenceConfig(db_uri="memory://test_shared", log_level="WARNING")

        cache1 = Reminiscence(config)
        cache2 = Reminiscence(config)

        # Same storage backend
        assert cache1.backend is cache2.backend

        # Store in cache1
        cache1.store("query1", {"agent": "test"}, "result1")

        # Should be visible in cache2 (shared storage)
        assert cache2.backend.count() == 1

    def test_count_empty(self):
        """Count on empty storage should be 0."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        assert storage.count() == 0

    def test_add_entry(self):
        """Should add entries."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        entry = CacheEntry(
            query_text="test query",
            context={"agent": "test"},
            embedding=[0.1] * 384,
            result="test result",
            timestamp=time.time(),
            metadata=None,
        )

        storage.add([entry])

        assert storage.count() == 1

    def test_search(self):
        """Should search by embedding and context."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        # Add entry
        embedding = [0.1] * 384
        context = {"agent": "test"}
        entry = CacheEntry(
            query_text="test query",
            context=context,
            embedding=embedding,
            result="test result",
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search with same context
        results = storage.search(
            embedding=embedding, context=context, limit=10, similarity_threshold=0.5
        )

        assert len(results) > 0
        assert results[0].query_text == "test query"
        assert results[0].result == "test result"

    def test_search_different_context_returns_empty(self):
        """Search with different context should return empty."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        # Add entry with context A
        entry = CacheEntry(
            query_text="test",
            context={"agent": "A"},
            embedding=[0.1] * 384,
            result="result A",
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search with context B
        results = storage.search(
            embedding=[0.1] * 384,
            context={"agent": "B"},
            limit=10,
            similarity_threshold=0.5,
        )

        assert len(results) == 0

    def test_to_arrow(self):
        """Should convert to Arrow table."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        entry = CacheEntry(
            query_text="test",
            context={"agent": "test"},
            embedding=[0.1] * 384,
            result="result",
            timestamp=time.time(),
            metadata={"key": "value"},
        )
        storage.add([entry])

        arrow_table = storage.to_arrow()

        assert len(arrow_table) == 1
        assert "query_text" in arrow_table.column_names
        assert "context" in arrow_table.column_names
        assert "context_hash" in arrow_table.column_names
        assert "embedding" in arrow_table.column_names

    def test_add_multiple_entries(self):
        """Should add multiple entries at once."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        entries = [
            CacheEntry(
                query_text=f"query {i}",
                context={"agent": "test"},
                embedding=[0.1 * i] * 384,
                result=f"result {i}",
                timestamp=time.time(),
                metadata=None,
            )
            for i in range(5)
        ]

        storage.add(entries)

        assert storage.count() == 5

    def test_serialization_dataframe(self):
        """Should serialize and deserialize DataFrames."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        entry = CacheEntry(
            query_text="get dataframe",
            context={"agent": "test"},
            embedding=[0.1] * 384,
            result=df,
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search and retrieve
        results = storage.search(
            embedding=[0.1] * 384,
            context={"agent": "test"},
            limit=10,
            similarity_threshold=0.5,
        )

        assert len(results) == 1
        assert isinstance(results[0].result, pd.DataFrame)
        assert results[0].result.equals(df)
