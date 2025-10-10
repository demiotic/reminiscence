"""Serialization/deserialization utilities for cache results."""

import json
import base64
from typing import Any, Tuple, List

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

import pyarrow as pa
from .logging import get_logger

logger = get_logger(__name__)


class ResultSerializer:
    """Handles serialization/deserialization of cache results with optional encryption."""

    def __init__(self, encryptor=None):
        """
        Initialize serializer.

        Args:
            encryptor: Optional encryption backend (e.g., AgeEncryption)
        """
        self.encryptor = encryptor

    def _is_special_type(self, value: Any) -> bool:
        """Check if value is a DataFrame, array, or other special type using explicit type checking."""
        try:
            result_class_name = (
                value.__class__.__module__ + "." + value.__class__.__name__
            )

            return (
                result_class_name == "pyarrow.lib.Table"
                or result_class_name.startswith("polars.")
                or result_class_name == "pandas.core.frame.DataFrame"
                or result_class_name == "numpy.ndarray"
            )
        except AttributeError:
            # Primitive types without __module__ or __class__
            return False

    def _serialize_nested_dict(self, result: dict) -> Tuple[str, str]:
        """Serialize dict that may contain DataFrames/arrays."""
        serialized_dict = {}
        special_types = {}

        for key, value in result.items():
            if self._is_special_type(value):
                serialized_value, value_type = self.serialize(value)
                serialized_dict[key] = f"__special__{key}"
                special_types[key] = {"data": serialized_value, "type": value_type}
            elif isinstance(value, (dict, list, tuple)):
                nested_serialized, nested_type = self.serialize(value)
                if nested_type.startswith("nested_") or nested_type.startswith(
                    "encrypted_"
                ):
                    serialized_dict[key] = f"__special__{key}"
                    special_types[key] = {
                        "data": nested_serialized,
                        "type": nested_type,
                    }
                else:
                    serialized_dict[key] = value
            else:
                serialized_dict[key] = value

        final_data = {"base": serialized_dict, "special": special_types}

        if HAS_ORJSON:
            return orjson.dumps(final_data).decode("utf-8"), "nested_dict"
        else:
            return json.dumps(final_data), "nested_dict"

    def _serialize_nested_list(self, result: list) -> Tuple[str, str]:
        """Serialize list that may contain DataFrames/arrays."""
        serialized_list = []
        special_types = {}

        for idx, value in enumerate(result):
            if self._is_special_type(value):
                serialized_value, value_type = self.serialize(value)
                serialized_list.append(f"__special__{idx}")
                special_types[str(idx)] = {"data": serialized_value, "type": value_type}
            elif isinstance(value, (dict, list, tuple)):
                nested_serialized, nested_type = self.serialize(value)
                if nested_type.startswith("nested_") or nested_type.startswith(
                    "encrypted_"
                ):
                    serialized_list.append(f"__special__{idx}")
                    special_types[str(idx)] = {
                        "data": nested_serialized,
                        "type": nested_type,
                    }
                else:
                    serialized_list.append(value)
            else:
                serialized_list.append(value)

        final_data = {"base": serialized_list, "special": special_types}

        if HAS_ORJSON:
            return orjson.dumps(final_data).decode("utf-8"), "nested_list"
        else:
            return json.dumps(final_data), "nested_list"

    def serialize(self, result: Any) -> Tuple[str, str]:
        """Serialize result with optimal format selection and optional encryption."""
        # Handle nested structures first
        if isinstance(result, dict):
            serialized, result_type = self._serialize_nested_dict(result)
        elif isinstance(result, (list, tuple)):
            serialized, result_type = self._serialize_nested_list(result)

        # Explicit type checking (más robusto que hasattr)
        else:
            result_class_name = (
                result.__class__.__module__ + "." + result.__class__.__name__
            )

            # PyArrow Table
            if result_class_name == "pyarrow.lib.Table":
                try:
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, result.schema) as writer:
                        writer.write_table(result)
                    buffer = sink.getvalue()
                    serialized = base64.b64encode(buffer.to_pybytes()).decode("utf-8")
                    result_type = "arrow_ipc"
                except Exception as e:
                    logger.warning("arrow_serialization_failed", error=str(e))
                    raise

            # Polars DataFrame
            elif result_class_name.startswith("polars."):
                try:
                    arrow_table = result.to_arrow()
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
                        writer.write_table(arrow_table)
                    buffer = sink.getvalue()
                    serialized = base64.b64encode(buffer.to_pybytes()).decode("utf-8")
                    result_type = "polars_arrow"
                except Exception as e:
                    logger.warning("polars_serialization_failed", error=str(e))
                    raise

            # Pandas DataFrame
            elif result_class_name == "pandas.core.frame.DataFrame":
                try:
                    arrow_table = pa.Table.from_pandas(result)
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
                        writer.write_table(arrow_table)
                    buffer = sink.getvalue()
                    serialized = base64.b64encode(buffer.to_pybytes()).decode("utf-8")
                    result_type = "pandas_arrow"
                except Exception as e:
                    logger.warning("pandas_serialization_failed", error=str(e))
                    raise

            # NumPy array
            elif result_class_name == "numpy.ndarray":
                try:
                    arrow_array = pa.array(result.flatten())
                    arrow_table = pa.Table.from_arrays([arrow_array], names=["values"])
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
                        writer.write_table(arrow_table)
                    buffer = sink.getvalue()

                    metadata = {"shape": list(result.shape), "dtype": str(result.dtype)}
                    encoded = base64.b64encode(buffer.to_pybytes()).decode("utf-8")
                    final_data = {"data": encoded, "metadata": metadata}

                    if HAS_ORJSON:
                        serialized = orjson.dumps(final_data).decode("utf-8")
                    else:
                        serialized = json.dumps(final_data)
                    result_type = "numpy_arrow"
                except Exception as e:
                    logger.warning("numpy_serialization_failed", error=str(e))
                    raise

            # JSON-serializable primitives
            else:
                try:
                    if HAS_ORJSON:
                        serialized = orjson.dumps(result).decode("utf-8")
                        result_type = "orjson"
                    else:
                        serialized = json.dumps(result)
                        result_type = "json"
                except (TypeError, ValueError) as e:
                    logger.error(
                        "json_serialization_failed",
                        error=str(e),
                        type=result_class_name,
                    )
                    raise TypeError(
                        f"Type {result_class_name} is not serializable. "
                        f"Supported types: dict, list, str, int, float, bool, None, "
                        f"pandas.DataFrame, polars.DataFrame, pyarrow.Table, numpy.ndarray"
                    )

        # Encrypt if enabled
        if self.encryptor:
            try:
                encrypted_bytes = self.encryptor.encrypt(serialized)
                encrypted_str = base64.b64encode(encrypted_bytes).decode("utf-8")
                return encrypted_str, f"encrypted_{result_type}"
            except Exception as e:
                logger.error("encryption_failed", error=str(e))
                raise

        return serialized, result_type

    def serialize_batch(self, results: List[Any]) -> List[Tuple[str, str]]:
        """Serialize multiple results in parallel."""
        if not results:
            return []

        # For small batches, sequential
        if len(results) <= 3:
            return [self.serialize(result) for result in results]

        # For larger batches with encryption: serialize first, then encrypt in batch
        if self.encryptor:
            from concurrent.futures import ThreadPoolExecutor

            # Step 1: Serialize in parallel (without encryption)
            temp_encryptor = self.encryptor
            self.encryptor = None  # Temporarily disable encryption

            with ThreadPoolExecutor(max_workers=temp_encryptor.max_workers) as executor:
                serialized_list = list(executor.map(self.serialize, results))

            self.encryptor = temp_encryptor  # Restore

            # Step 2: Extract data and types
            serialized_data = [s[0] for s in serialized_list]
            result_types = [s[1] for s in serialized_list]

            # Step 3: Encrypt in batch (parallel internally)
            encrypted_data = self.encryptor.encrypt_batch(serialized_data)

            # Step 4: Encode and add encrypted_ prefix
            final_results = []
            for encrypted_bytes, result_type in zip(encrypted_data, result_types):
                encrypted_str = base64.b64encode(encrypted_bytes).decode("utf-8")
                final_results.append((encrypted_str, f"encrypted_{result_type}"))

            return final_results
        else:
            # Without encryption, sequential is fastest
            return [self.serialize(result) for result in results]

    def deserialize_batch(self, data_list: List[Tuple[str, str]]) -> List[Any]:
        """Deserialize multiple results in parallel."""
        if not data_list:
            return []

        # For small batches, sequential
        if len(data_list) <= 3:
            return [
                self.deserialize(data, result_type) for data, result_type in data_list
            ]

        # For larger batches with encryption: decrypt in batch first, then deserialize
        if self.encryptor:
            from concurrent.futures import ThreadPoolExecutor

            # Check if any are encrypted
            encrypted_indices = []
            _ = []

            for i, (data, result_type) in enumerate(data_list):
                if result_type.startswith("encrypted_"):
                    encrypted_indices.append(i)

            # If we have encrypted data, decrypt in batch
            if encrypted_indices:
                encrypted_bytes_list = []
                for i in encrypted_indices:
                    data, _ = data_list[i]
                    encrypted_bytes = base64.b64decode(data.encode("utf-8"))
                    encrypted_bytes_list.append(encrypted_bytes)

                # Batch decrypt (parallel internally)
                decrypted_strings = self.encryptor.decrypt_batch(encrypted_bytes_list)

                # Build new data_list with decrypted data
                decrypted_data_list = []
                encrypted_idx = 0
                for i, (data, result_type) in enumerate(data_list):
                    if result_type.startswith("encrypted_"):
                        original_type = result_type.replace("encrypted_", "")
                        decrypted_data_list.append(
                            (decrypted_strings[encrypted_idx], original_type)
                        )
                        encrypted_idx += 1
                    else:
                        decrypted_data_list.append((data, result_type))
            else:
                decrypted_data_list = data_list

            # Now deserialize in parallel (no encryption)
            temp_encryptor = self.encryptor
            self.encryptor = None  # Temporarily disable to avoid recursive decrypt

            with ThreadPoolExecutor(max_workers=temp_encryptor.max_workers) as executor:
                deserialized = list(
                    executor.map(
                        lambda x: self.deserialize(x[0], x[1]), decrypted_data_list
                    )
                )

            self.encryptor = temp_encryptor  # Restore
            return deserialized
        else:
            return [
                self.deserialize(data, result_type) for data, result_type in data_list
            ]

    def _deserialize_nested_dict(self, data: str) -> dict:
        """Deserialize nested dict with special types."""
        if HAS_ORJSON:
            parsed = orjson.loads(data.encode("utf-8"))
        else:
            parsed = json.loads(data)

        base_dict = parsed["base"]
        special_types = parsed["special"]

        result = {}
        for key, value in base_dict.items():
            if isinstance(value, str) and value.startswith("__special__"):
                special_key = value.replace("__special__", "")
                special_data = special_types[special_key]["data"]
                special_type = special_types[special_key]["type"]
                result[key] = self.deserialize(special_data, special_type)
            else:
                result[key] = value

        return result

    def _deserialize_nested_list(self, data: str) -> list:
        """Deserialize nested list with special types."""
        if HAS_ORJSON:
            parsed = orjson.loads(data.encode("utf-8"))
        else:
            parsed = json.loads(data)

        base_list = parsed["base"]
        special_types = parsed["special"]

        result = []
        for idx, value in enumerate(base_list):
            if isinstance(value, str) and value.startswith("__special__"):
                special_key = str(idx)
                special_data = special_types[special_key]["data"]
                special_type = special_types[special_key]["type"]
                result.append(self.deserialize(special_data, special_type))
            else:
                result.append(value)

        return result

    def _deserialize_numpy(self, data: str) -> Any:
        """Deserialize numpy array."""
        import numpy as np

        if HAS_ORJSON:
            parsed = orjson.loads(data.encode("utf-8"))
        else:
            parsed = json.loads(data)

        encoded_data = parsed["data"]
        metadata = parsed["metadata"]

        buffer = base64.b64decode(encoded_data.encode("utf-8"))
        reader = pa.ipc.open_stream(buffer)
        arrow_table = reader.read_all()

        flat_array = arrow_table["values"].to_numpy()

        shape = tuple(metadata["shape"])
        dtype = np.dtype(metadata["dtype"])

        return flat_array.reshape(shape).astype(dtype)

    def deserialize(self, data: str, result_type: str) -> Any:
        """
        Deserialize result from string with optional decryption.

        Args:
            data: Serialized data
            result_type: Type indicator

        Returns:
            Deserialized result

        Raises:
            RuntimeError: If data is encrypted but no encryptor configured
        """
        # Check if encrypted
        if result_type.startswith("encrypted_"):
            if not self.encryptor:
                raise RuntimeError(
                    "Cannot decrypt: cache entry is encrypted but no encryptor configured. "
                    "Please set encryption_enabled=True and provide encryption_key in config."
                )

            try:
                # Decrypt
                encrypted_bytes = base64.b64decode(data.encode("utf-8"))
                decrypted_str = self.encryptor.decrypt(encrypted_bytes)

                # Remove 'encrypted_' prefix to get original type
                original_type = result_type.replace("encrypted_", "")
                data = decrypted_str
                result_type = original_type
            except Exception as e:
                logger.error("decryption_failed", error=str(e))
                raise

        # Deserialize based on type
        if result_type == "nested_dict":
            return self._deserialize_nested_dict(data)

        if result_type == "nested_list":
            return self._deserialize_nested_list(data)

        if result_type == "numpy_arrow":
            return self._deserialize_numpy(data)

        if result_type == "arrow_ipc":
            buffer = base64.b64decode(data.encode("utf-8"))
            reader = pa.ipc.open_stream(buffer)
            return reader.read_all()

        elif result_type == "pandas_arrow":
            buffer = base64.b64decode(data.encode("utf-8"))
            reader = pa.ipc.open_stream(buffer)
            arrow_table = reader.read_all()
            return arrow_table.to_pandas()

        elif result_type == "polars_arrow":
            buffer = base64.b64decode(data.encode("utf-8"))
            reader = pa.ipc.open_stream(buffer)
            arrow_table = reader.read_all()
            try:
                import polars as pl

                return pl.from_arrow(arrow_table)
            except ImportError:
                logger.warning("polars_not_available", returning="arrow_table")
                return arrow_table

        elif result_type == "orjson":
            if HAS_ORJSON:
                return orjson.loads(data.encode("utf-8"))
            else:
                return json.loads(data)

        elif result_type == "json":
            return json.loads(data)

        elif result_type == "repr":
            obj = json.loads(data)
            return obj.get("__repr__")

        else:
            logger.error("unknown_result_type", type=result_type)
            return None
