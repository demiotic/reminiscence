"""Base serialization interface for Reminiscence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class Serializer(ABC):
    """Abstract base class for serializers.

    Defines the interface for serializing Python objects to bytes
    and deserializing bytes back to Python objects.
    """

    @abstractmethod
    def serialize(self, result: Any) -> Tuple[str, str]:
        """Serialize Python object to base64 string.

        Args:
            result: Python object to serialize.

        Returns:
            Tuple of (base64_encoded_string, type_descriptor).

        Raises:
            TypeError: If object is not serializable.
        """
        pass

    @abstractmethod
    def deserialize(self, data: str, result_type: str) -> Any:
        """Deserialize base64 string to Python object.

        Args:
            data: Base64 encoded string.
            result_type: Type descriptor from serialization.

        Returns:
            Deserialized Python object.

        Raises:
            ValueError: If data cannot be deserialized.
        """
        pass

    def serialize_batch(self, results: List[Any]) -> List[Tuple[str, str]]:
        """Serialize multiple objects (default: sequential).

        Override for batch optimizations (parallel compression/encryption).

        Args:
            results: List of Python objects to serialize.

        Returns:
            List of (base64_string, type_descriptor) tuples.
        """
        return [self.serialize(result) for result in results]

    def deserialize_batch(self, data_list: List[Tuple[str, str]]) -> List[Any]:
        """Deserialize multiple objects (default: sequential).

        Override for batch optimizations (parallel decompression/decryption).

        Args:
            data_list: List of (base64_string, type_descriptor) tuples.

        Returns:
            List of deserialized Python objects.
        """
        return [self.deserialize(data, dtype) for data, dtype in data_list]
