"""Tests para serialization.py - versión completa."""

import pytest
from memora.utils.serialization import serialize, deserialize, is_serializable

# Imports condicionales
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
    """Tests básicos compatibles con versión original."""

    def test_serialize_string(self):
        """String debe serializarse a bytes."""
        data = "test string"
        serialized = serialize(data)

        # Ahora retorna bytes, no string
        assert isinstance(serialized, bytes)
        assert deserialize(serialized) == data

    def test_serialize_dict(self):
        """Dict debe serializarse a JSON (bytes)."""
        data = {"key": "value", "number": 42}
        serialized = serialize(data)

        assert isinstance(serialized, bytes)
        # Verificar que contiene la clave
        assert b"key" in serialized or b'"key"' in serialized

    def test_serialize_deserialize_roundtrip(self):
        """Serializar + deserializar debe retornar datos originales."""
        original = {"status": "ok", "items": [1, 2, 3], "meta": {"count": 3}}

        serialized = serialize(original)
        deserialized = deserialize(serialized)

        assert deserialized == original

    def test_deserialize_string(self):
        """String debe deserializarse correctamente desde bytes."""
        data = "plain string"
        serialized = serialize(data)
        result = deserialize(serialized)

        assert result == data

    def test_serialize_list(self):
        """List debe serializarse a JSON (bytes)."""
        data = [1, 2, 3, "test"]
        serialized = serialize(data)

        assert isinstance(serialized, bytes)
        assert b"[" in serialized

    def test_serialize_nested(self):
        """Estructuras nested deben serializarse."""
        data = {"level1": {"level2": {"level3": [1, 2, 3]}}}

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized == data

    def test_serialize_unicode(self):
        """Unicode debe manejarse correctamente."""
        data = {"text": "Hello 世界 مرحبا 🌍"}

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized == data


class TestAdvancedSerialization:
    """Tests para tipos avanzados."""

    def test_none(self):
        """None debe manejarse correctamente."""
        data = None
        serialized = serialize(data)
        assert deserialize(serialized) is None

    def test_bool(self):
        """Bools deben preservarse."""
        assert deserialize(serialize(True)) is True
        assert deserialize(serialize(False)) is False

    def test_floats(self):
        """Floats deben preservar precisión."""
        data = {"pi": 3.14159, "e": 2.71828}
        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert abs(deserialized["pi"] - 3.14159) < 1e-5
        assert abs(deserialized["e"] - 2.71828) < 1e-5

    def test_mixed_types(self):
        """Mezcla de tipos en una estructura."""
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
        """Test de verificación de serializabilidad."""
        assert is_serializable("string")
        assert is_serializable({"key": "value"})
        assert is_serializable([1, 2, 3])
        assert is_serializable(42)
        assert is_serializable(3.14)
        assert is_serializable(True)
        assert is_serializable(None)


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestNumpySerialization:
    """Tests para numpy arrays."""

    def test_numpy_array_1d(self):
        """Array 1D debe preservarse."""
        data = np.array([1, 2, 3, 4, 5])
        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(deserialized, data)

    def test_numpy_array_2d(self):
        """Array 2D debe preservar shape."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(deserialized, data)
        assert deserialized.shape == (2, 3)

    def test_numpy_float_array(self):
        """Array de floats debe preservar dtype."""
        data = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert deserialized.dtype == np.float32
        assert np.allclose(deserialized, data)

    def test_numpy_scalar(self):
        """Scalars numpy deben convertirse a Python nativos."""
        data = {"value": np.int64(42)}
        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert deserialized["value"] == 42

    def test_numpy_in_nested_structure(self):
        """Arrays numpy dentro de estructuras nested."""
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
    """Tests para pandas DataFrames."""

    def test_simple_dataframe(self):
        """DataFrame simple debe preservarse."""
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
        """DataFrame con índice nombrado."""
        data = pd.DataFrame({"value": [10, 20, 30]}, index=["a", "b", "c"])
        data.index.name = "key"

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized.index.name == "key"
        pd.testing.assert_frame_equal(deserialized, data)

    def test_series(self):
        """Series debe preservarse."""
        data = pd.Series([1, 2, 3, 4], name="values")

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert isinstance(deserialized, pd.Series)
        pd.testing.assert_series_equal(deserialized, data)

    def test_dataframe_nested_in_dict(self):
        """DataFrame dentro de dict."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        data = {"status": "ok", "data": df, "metadata": {"rows": 2}}

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized["status"] == "ok"
        pd.testing.assert_frame_equal(deserialized["data"], df)

    def test_dataframe_with_mixed_types(self):
        """DataFrame con tipos mixtos."""
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
    """Tests para polars DataFrames."""

    def test_simple_dataframe(self):
        """DataFrame polars simple."""
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
        """DataFrame polars dentro de dict."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        data = {"status": "ok", "data": df, "metadata": {"rows": 2}}

        serialized = serialize(data)
        deserialized = deserialize(serialized)

        assert deserialized["status"] == "ok"
        assert deserialized["data"].equals(df)


class TestEdgeCases:
    """Tests para casos edge."""

    def test_empty_dict(self):
        """Dict vacío."""
        data = {}
        assert deserialize(serialize(data)) == data

    def test_empty_list(self):
        """Lista vacía."""
        data = []
        assert deserialize(serialize(data)) == data

    def test_large_nested_structure(self):
        """Estructura profundamente nested."""
        data = {"level": 1}
        current = data
        for i in range(2, 20):
            current["nested"] = {"level": i}
            current = current["nested"]

        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert deserialized == data

    def test_special_characters_in_keys(self):
        """Keys con caracteres especiales."""
        data = {
            "key-with-dash": 1,
            "key.with.dot": 2,
            "key:with:colon": 3,
            "key/with/slash": 4,
        }
        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert deserialized == data
