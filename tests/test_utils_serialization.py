"""Tests for serialization.py - complete version."""

import pytest
from memora.utils.serialization import serialize, deserialize, is_serializable


# Conditional imports
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestSerialization:
    """Basic tests compatible with original version."""

    def test_serialize_string(self):
        """String should serialize to bytes."""
        data = "test string"
        serialized = serialize(data)

        # Now returns bytes, not string
        assert isinstance(serialized, bytes)
        assert deserialize(serialized) == data

    def test_serialize_dict(self):
        """Dict should serialize to JSON (bytes)."""
        data = {"key": "value", "number": 42}
        serialized = serialize(data)

        assert isinstance(serialized, bytes)
        # Verify it contains the key
        assert b"key" in serialized or b'"key"' in serialized

    def test_serialize_deserialize_roundtrip(self):
        """Serialize + deserialize should return original data."""
        original = {"status": "ok", "items": [1, 2, 3], "meta": {"count": 3}}

        serialized = serialize(original)
        deserialized = deserialize(serialized)

        assert deserialized == original

    def test_deserialize_string(self):
        """String should deserialize correctly from bytes."""
        data = "plain string"
        serialized = serialize(data)
        result = deserialize(serialized)

        assert result == data

    def test_serialize_list(self):
        """List should serialize to JSON (bytes)."""
        data = [1, 2, 3, "test"]
        serialized = serialize(data)

        assert isinstance(serialized, bytes)
        assert b"[" in serialized

    def test_serialize_nested(self):
        """Nested structures should serialize."""
        data = {"level1": {"level2": {"level3": [1, 2, 3]}}}

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized == data

    def test_serialize_unicode(self):
        """Unicode should be handled correctly."""
        data = {"text": "Hello 世界 مرحبا 🌍"}

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized == data


class TestAdvancedSerialization:
    """Tests for advanced types."""

    def test_none(self):
        """None should be handled correctly."""
        data = None
        serialized = serialize(data)
        assert deserialize(serialized) is None

    def test_bool(self):
        """Bools should be preserved."""
        assert deserialize(serialize(True)) is True
        assert deserialize(serialize(False)) is False

    def test_floats(self):
        """Floats should preserve precision."""
        data = {"pi": 3.14159, "e": 2.71828}
        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert abs(deserialized["pi"] - 3.14159) < 1e-5
        assert abs(deserialized["e"] - 2.71828) < 1e-5

    def test_mixed_types(self):
        """Mix of types in a structure."""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, "two", 3.0],
            "nested": {"a": 1, "b": [2, 3]},
        }
        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert deserialized == data

    def test_is_serializable(self):
        """Test serializability check."""
        assert is_serializable("string")
        assert is_serializable({"key": "value"})
        assert is_serializable([1, 2, 3])
        assert is_serializable(42)
        assert is_serializable(3.14)
        assert is_serializable(True)
        assert is_serializable(None)


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestNumpySerialization:
    """Tests for numpy arrays."""

    def test_numpy_array_1d(self):
        """1D array should be preserved."""
        data = np.array([1, 2, 3, 4, 5])
        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(deserialized, data)

    def test_numpy_array_2d(self):
        """2D array should preserve shape."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(deserialized, data)
        assert deserialized.shape == (2, 3)

    def test_numpy_float_array(self):
        """Float array should preserve dtype."""
        data = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert deserialized.dtype == np.float32
        assert np.allclose(deserialized, data)

    def test_numpy_scalar(self):
        """Numpy scalars should convert to native Python."""
        data = {"value": np.int64(42)}
        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert deserialized["value"] == 42

    def test_numpy_in_nested_structure(self):
        """Numpy arrays inside nested structures."""
        data = {
            "results": {
                "embeddings": np.array([0.1, 0.2, 0.3]),
                "scores": np.array([95, 87, 92]),
            }
        }
        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert np.array_equal(
            deserialized["results"]["embeddings"], data["results"]["embeddings"]
        )


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestPandasSerialization:
    """Tests for pandas DataFrames."""

    def test_simple_dataframe(self):
        """Simple DataFrame should be preserved."""
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95, 87, 92],
            }
        )

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, pd.DataFrame)
        pd.testing.assert_frame_equal(deserialized, data)

    def test_dataframe_with_index(self):
        """DataFrame with named index."""
        data = pd.DataFrame({"value": [10, 20, 30]}, index=["a", "b", "c"])
        data.index.name = "key"

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized.index.name == "key"
        pd.testing.assert_frame_equal(deserialized, data)

    def test_series(self):
        """Series should be preserved."""
        data = pd.Series([1, 2, 3, 4], name="values")

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, pd.Series)
        pd.testing.assert_series_equal(deserialized, data)

    def test_dataframe_nested_in_dict(self):
        """DataFrame inside dict."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        data = {"status": "ok", "data": df, "metadata": {"rows": 2}}

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized["status"] == "ok"
        pd.testing.assert_frame_equal(deserialized["data"], df)

    def test_dataframe_with_mixed_types(self):
        """DataFrame with mixed types."""
        data = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        pd.testing.assert_frame_equal(deserialized, data)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
class TestPolarsSerialization:
    """Tests for polars DataFrames."""

    def test_simple_dataframe(self):
        """Simple polars DataFrame."""
        data = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95, 87, 92],
            }
        )

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, pl.DataFrame)
        assert deserialized.equals(data)

    def test_polars_nested_in_dict(self):
        """Polars DataFrame inside dict."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        data = {"status": "ok", "data": df, "metadata": {"rows": 2}}

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized["status"] == "ok"
        assert deserialized["data"].equals(df)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dict(self):
        """Empty dict."""
        data = {}
        assert deserialize(serialize(data)) == data

    def test_empty_list(self):
        """Empty list."""
        data = []
        assert deserialize(serialize(data)) == data

    def test_large_nested_structure(self):
        """Deeply nested structure."""
        data = {"level": 1}
        current = data
        for i in range(2, 20):
            current["nested"] = {"level": i}
            current = current["nested"]

        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert deserialized == data

    def test_special_characters_in_keys(self):
        """Keys with special characters."""
        data = {
            "key-with-dash": 1,
            "key.with.dot": 2,
            "key:with:colon": 3,
            "key/with/slash": 4,
        }
        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert deserialized == data
