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

    def test_serialization_nested_dict_with_dataframe(self):
        """Should serialize and deserialize nested dict containing DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Nested structure with DataFrame
        nested_result = {
            "data": df,
            "status": "success",
            "rows": 3,
            "metadata": {"source": "test"},
        }

        entry = CacheEntry(
            query_text="get nested result",
            context={"agent": "test"},
            embedding=[0.1] * 384,
            result=nested_result,
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
        assert isinstance(results[0].result, dict)
        assert "data" in results[0].result
        assert isinstance(results[0].result["data"], pd.DataFrame)
        assert results[0].result["data"].equals(df)
        assert results[0].result["status"] == "success"
        assert results[0].result["rows"] == 3
        assert results[0].result["metadata"]["source"] == "test"

    def test_serialization_list_with_dataframes(self):
        """Should serialize and deserialize list containing DataFrames."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})

        # List with DataFrames
        list_result = [df1, df2, "some string", 123]

        entry = CacheEntry(
            query_text="get list of dataframes",
            context={"agent": "test"},
            embedding=[0.2] * 384,
            result=list_result,
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search and retrieve
        results = storage.search(
            embedding=[0.2] * 384,
            context={"agent": "test"},
            limit=10,
            similarity_threshold=0.5,
        )

        assert len(results) == 1
        assert isinstance(results[0].result, list)
        assert len(results[0].result) == 4
        assert isinstance(results[0].result[0], pd.DataFrame)
        assert isinstance(results[0].result[1], pd.DataFrame)
        assert results[0].result[0].equals(df1)
        assert results[0].result[1].equals(df2)
        assert results[0].result[2] == "some string"
        assert results[0].result[3] == 123

    def test_serialization_deeply_nested_structures(self):
        """Should serialize and deserialize deeply nested structures."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        df = pd.DataFrame({"x": [1, 2, 3]})

        # Deeply nested structure
        nested = {
            "level1": {"level2": {"data": df, "count": 3}, "other": "value"},
            "results": [{"df": df, "name": "first"}, {"df": df, "name": "second"}],
        }

        entry = CacheEntry(
            query_text="deeply nested",
            context={"agent": "test"},
            embedding=[0.3] * 384,
            result=nested,
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search and retrieve
        results = storage.search(
            embedding=[0.3] * 384,
            context={"agent": "test"},
            limit=10,
            similarity_threshold=0.5,
        )

        assert len(results) == 1
        result = results[0].result

        # Check nested dict structure
        assert isinstance(result["level1"]["level2"]["data"], pd.DataFrame)
        assert result["level1"]["level2"]["data"].equals(df)
        assert result["level1"]["level2"]["count"] == 3
        assert result["level1"]["other"] == "value"

        # Check nested list structure
        assert len(result["results"]) == 2
        assert isinstance(result["results"][0]["df"], pd.DataFrame)
        assert result["results"][0]["df"].equals(df)
        assert result["results"][0]["name"] == "first"

    def test_serialization_numpy_array(self):
        """Should serialize and deserialize NumPy arrays."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("NumPy not installed")

        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        arr = np.array([[1, 2, 3], [4, 5, 6]])

        entry = CacheEntry(
            query_text="get numpy array",
            context={"agent": "test"},
            embedding=[0.4] * 384,
            result=arr,
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search and retrieve
        results = storage.search(
            embedding=[0.4] * 384,
            context={"agent": "test"},
            limit=10,
            similarity_threshold=0.5,
        )

        assert len(results) == 1
        assert isinstance(results[0].result, np.ndarray)
        assert np.array_equal(results[0].result, arr)
        assert results[0].result.shape == arr.shape
        assert results[0].result.dtype == arr.dtype

    def test_serialization_dict_with_numpy_and_dataframe(self):
        """Should serialize dict containing both NumPy arrays and DataFrames."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("Pandas or NumPy not installed")

        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        df = pd.DataFrame({"col": [1, 2, 3]})
        arr = np.array([10, 20, 30])

        mixed_result = {"dataframe": df, "array": arr, "scalar": 42, "text": "hello"}

        entry = CacheEntry(
            query_text="mixed types",
            context={"agent": "test"},
            embedding=[0.5] * 384,
            result=mixed_result,
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search and retrieve
        results = storage.search(
            embedding=[0.5] * 384,
            context={"agent": "test"},
            limit=10,
            similarity_threshold=0.5,
        )

        assert len(results) == 1
        result = results[0].result

        assert isinstance(result["dataframe"], pd.DataFrame)
        assert result["dataframe"].equals(df)
        assert isinstance(result["array"], np.ndarray)
        assert np.array_equal(result["array"], arr)
        assert result["scalar"] == 42
        assert result["text"] == "hello"

    def test_serialization_polars_dataframe(self):
        """Should serialize and deserialize Polars DataFrames."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        entry = CacheEntry(
            query_text="get polars dataframe",
            context={"agent": "test"},
            embedding=[0.6] * 384,
            result=df,
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search and retrieve
        results = storage.search(
            embedding=[0.6] * 384,
            context={"agent": "test"},
            limit=10,
            similarity_threshold=0.5,
        )

        assert len(results) == 1
        assert isinstance(results[0].result, pl.DataFrame)
        assert results[0].result.equals(df)

    def test_serialization_mixed_list_with_multiple_types(self):
        """Should serialize list with mixed types (primitives, DataFrames, arrays)."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("Pandas or NumPy not installed")

        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        df = pd.DataFrame({"x": [1, 2]})
        arr = np.array([3, 4, 5])

        mixed_list = [df, arr, "string", 123, {"nested": "dict"}, [1, 2, 3], None]

        entry = CacheEntry(
            query_text="mixed list",
            context={"agent": "test"},
            embedding=[0.7] * 384,
            result=mixed_list,
            timestamp=time.time(),
            metadata=None,
        )
        storage.add([entry])

        # Search and retrieve
        results = storage.search(
            embedding=[0.7] * 384,
            context={"agent": "test"},
            limit=10,
            similarity_threshold=0.5,
        )

        assert len(results) == 1
        result = results[0].result

        assert isinstance(result[0], pd.DataFrame)
        assert result[0].equals(df)
        assert isinstance(result[1], np.ndarray)
        assert np.array_equal(result[1], arr)
        assert result[2] == "string"
        assert result[3] == 123
        assert result[4] == {"nested": "dict"}
        assert result[5] == [1, 2, 3]
        assert result[6] is None


class TestDualTableArchitecture:
    """Test dual table architecture (exact + semantic)."""

    def test_dual_tables_created(self):
        """Should create both exact and semantic tables."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        assert hasattr(storage, "exact_table")
        assert hasattr(storage, "semantic_table")
        assert storage.exact_table is not None
        assert storage.semantic_table is not None

    def test_add_to_exact_table(self):
        """Should add to exact table when query_mode=exact."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        entry = CacheEntry(
            query_text="test",
            context={"agent": "test"},
            embedding=None,  # No embedding for exact mode
            result="result",
            timestamp=time.time(),
            metadata=None,
        )

        storage.add([entry], query_mode="exact")

        assert storage.exact_table.count_rows() == 1
        assert storage.semantic_table.count_rows() == 0

    def test_add_to_semantic_table(self):
        """Should add to semantic table when query_mode=semantic."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        entry = CacheEntry(
            query_text="test",
            context={"agent": "test"},
            embedding=[0.1] * 384,
            result="result",
            timestamp=time.time(),
            metadata=None,
        )

        storage.add([entry], query_mode="semantic")

        assert storage.semantic_table.count_rows() == 1
        assert storage.exact_table.count_rows() == 0

    def test_search_exact_mode(self):
        """Should search in exact table."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        entry = CacheEntry(
            query_text="exact query",
            context={"agent": "test"},
            embedding=None,
            result="exact result",
            timestamp=time.time(),
            metadata=None,
        )

        storage.add([entry], query_mode="exact")

        results = storage.search(
            embedding=None,
            context={"agent": "test"},
            limit=10,
            similarity_threshold=0.5,
            query_mode="exact",
            query_text="exact query",
        )

        assert len(results) == 1
        assert results[0].result == "exact result"

    def test_count_includes_both_tables(self):
        """Count should include entries from both tables."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        entry1 = CacheEntry(
            query_text="exact",
            context={"agent": "test"},
            embedding=None,
            result="r1",
            timestamp=time.time(),
            metadata=None,
        )

        entry2 = CacheEntry(
            query_text="semantic",
            context={"agent": "test"},
            embedding=[0.1] * 384,
            result="r2",
            timestamp=time.time(),
            metadata=None,
        )

        storage.add([entry1], query_mode="exact")
        storage.add([entry2], query_mode="semantic")

        assert storage.count() == 2

    def test_clear_clears_both_tables(self):
        """Clear should remove entries from both tables."""
        config = ReminiscenceConfig(db_uri="memory://")
        storage = LanceDBBackend(config, embedding_dim=384)

        entry1 = CacheEntry(
            query_text="exact",
            context={"agent": "test"},
            embedding=None,
            result="r1",
            timestamp=time.time(),
            metadata=None,
        )

        entry2 = CacheEntry(
            query_text="semantic",
            context={"agent": "test"},
            embedding=[0.1] * 384,
            result="r2",
            timestamp=time.time(),
            metadata=None,
        )

        storage.add([entry1], query_mode="exact")
        storage.add([entry2], query_mode="semantic")

        storage.clear()

        assert storage.count() == 0
        assert storage.exact_table.count_rows() == 0
        assert storage.semantic_table.count_rows() == 0
