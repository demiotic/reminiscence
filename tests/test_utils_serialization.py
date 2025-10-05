import pytest
import pyarrow as pa
from memora.utils import (
    serialize,
    deserialize,
)


class TestSerialization:
    """Tests para utils/serialization.py."""

    def test_serialize_string(self):
        """String debe retornarse sin cambios."""
        data = "test string"
        serialized = serialize(data)

        assert serialized == data

    def test_serialize_dict(self):
        """Dict debe serializarse a JSON."""
        data = {"key": "value", "number": 42}
        serialized = serialize(data)

        assert isinstance(serialized, str)
        assert "key" in serialized

    def test_serialize_deserialize_roundtrip(self):
        """Serializar + deserializar debe retornar datos originales."""
        original = {"status": "ok", "items": [1, 2, 3], "meta": {"count": 3}}

        serialized = serialize(original)
        deserialized = deserialize(serialized)

        assert deserialized == original

    def test_deserialize_string(self):
        """String plano debe retornarse sin cambios."""
        data = "plain string"
        result = deserialize(data)

        assert result == data

    def test_serialize_list(self):
        """List debe serializarse a JSON."""
        data = [1, 2, 3, "test"]
        serialized = serialize(data)

        assert isinstance(serialized, str)
        assert "[" in serialized

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
