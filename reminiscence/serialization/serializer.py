"""Result serialization with transformation pipeline support."""

import json
import time
from typing import Any, Tuple, List

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

try:
    import pybase64 as base64

    HAS_PYBASE64 = True
except ImportError:
    import base64

    HAS_PYBASE64 = False

import pyarrow as pa
from .pipeline import TransformationPipeline
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResultSerializer:
    """
    Serialize/deserialize cache results with optional transformations.

    Architecture:
    1. Serialization: Python object → bytes
    2. Pipeline: bytes → transformed bytes (compress/encrypt)
    3. Storage: bytes → base64 string

    Optimizations:
    - Uses orjson for JSON (3-5x faster than stdlib)
    - Uses pybase64 if available (2x faster than stdlib)
    - Avoids unnecessary encode/decode cycles
    - Parallel batch operations for compression and encryption
    """

    def __init__(self, encryptor=None, compressor=None):
        """
        Initialize serializer with optional transformers.

        Args:
            encryptor: Optional encryption backend
            compressor: Optional compression backend
        """
        self.encryptor = encryptor
        self.compressor = compressor
        self.pipeline = TransformationPipeline(compressor, encryptor)

        if HAS_PYBASE64:
            logger.debug("serializer_using_pybase64", speedup="~2x")

    def serialize(self, result: Any) -> Tuple[str, str]:
        """
        Serialize result with pipeline transformations.

        Args:
            result: Python object to serialize

        Returns:
            Tuple of (base64_encoded_string, type_descriptor)
        """
        serialized_bytes, result_type = self._serialize_to_bytes(result)

        if self.pipeline.forward_stages:
            transformed_bytes, applied = self.pipeline.transform(serialized_bytes)

            type_parts = []
            if "encrypt" in applied:
                type_parts.append("encrypted")
            if "compress" in applied:
                type_parts.append("compressed")
            type_parts.append(result_type)

            final_type = "_".join(type_parts)
        else:
            transformed_bytes = serialized_bytes
            final_type = result_type

        base64_str = base64.b64encode(transformed_bytes).decode("ascii")

        return base64_str, final_type

    def deserialize(self, data: str, result_type: str) -> Any:
        """
        Deserialize result with pipeline reverse transformations.

        Args:
            data: Base64 encoded string
            result_type: Type descriptor

        Returns:
            Original Python object
        """
        encoded_bytes = base64.b64decode(data.encode("ascii"))

        type_parts = result_type.split("_")
        applied_transformations = []

        if "encrypted" in type_parts:
            applied_transformations.append("encrypt")
            type_parts.remove("encrypted")

        if "compressed" in type_parts:
            applied_transformations.append("compress")
            type_parts.remove("compressed")

        original_type = "_".join(type_parts)

        if applied_transformations:
            decoded_bytes = self.pipeline.reverse(
                encoded_bytes, applied_transformations
            )
        else:
            decoded_bytes = encoded_bytes

        return self._deserialize_from_bytes(decoded_bytes, original_type)

    def serialize_batch(self, results: List[Any]) -> List[Tuple[str, str]]:
        """
        Serialize multiple results with parallel compression/encryption.

        Optimization: Parallelizes compression and encryption for large batches.
        """
        if not results:
            return []

        if len(results) <= 3:
            return [self.serialize(result) for result in results]

        if self.encryptor or self.compressor:
            return self._serialize_batch_optimized(results)
        else:
            return [self.serialize(result) for result in results]

    def _serialize_batch_optimized(self, results: List[Any]) -> List[Tuple[str, str]]:
        """Optimized batch serialization with parallel compression and encryption."""
        from concurrent.futures import ThreadPoolExecutor

        start = time.perf_counter()

        # Step 1: Serialize all objects to bytes (sequential, fast)
        serialized_list = []
        for result in results:
            data_bytes, result_type = self._serialize_to_bytes(result)
            serialized_list.append((data_bytes, result_type))

        data_list = [item[0] for item in serialized_list]
        type_list = [item[1] for item in serialized_list]

        # Step 2: Compress in parallel if enabled
        if self.compressor:
            compress_start = time.perf_counter()
            max_workers = min(4, len(data_list))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                compressed_list = list(
                    executor.map(self.compressor.compress, data_list)
                )

            compress_ms = (time.perf_counter() - compress_start) * 1000
            type_list = [f"compressed_{t}" for t in type_list]

            logger.debug(
                "batch_compress_complete",
                count=len(compressed_list),
                workers=max_workers,
                latency_ms=round(compress_ms, 2),
            )
        else:
            compressed_list = data_list

        # Step 3: Encrypt in batch (uses internal parallelization)
        if self.encryptor:
            encrypted_list = self.encryptor.encrypt_batch(compressed_list)
            type_list = [f"encrypted_{t}" for t in type_list]
        else:
            encrypted_list = compressed_list

        # Step 4: Base64 encode
        results_output = [
            (base64.b64encode(encrypted).decode("ascii"), result_type)
            for encrypted, result_type in zip(encrypted_list, type_list)
        ]

        total_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "batch_serialize_complete",
            count=len(results_output),
            total_ms=round(total_ms, 2),
            per_item_ms=round(total_ms / len(results), 2) if results else 0,
        )

        return results_output

    def _deserialize_batch_optimized(
        self, data_list: List[Tuple[str, str]]
    ) -> List[Any]:
        """Optimized batch deserialization with parallel operations."""
        from concurrent.futures import ThreadPoolExecutor

        start = time.perf_counter()

        encoded_list = [base64.b64decode(data.encode("ascii")) for data, _ in data_list]
        type_list = [result_type for _, result_type in data_list]

        needs_decrypt = any("encrypted" in t for t in type_list)
        needs_decompress = any("compressed" in t for t in type_list)

        if needs_decrypt and self.encryptor:
            decrypted_list = self.encryptor.decrypt_batch(encoded_list)
            type_list = [t.replace("encrypted_", "") for t in type_list]
        else:
            decrypted_list = encoded_list

        if needs_decompress and self.compressor:
            decompress_start = time.perf_counter()
            max_workers = min(4, len(decrypted_list))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                decompressed_list = list(
                    executor.map(self.compressor.decompress, decrypted_list)
                )

            decompress_ms = (time.perf_counter() - decompress_start) * 1000
            type_list = [t.replace("compressed_", "") for t in type_list]

            logger.debug(
                "batch_decompress_complete",
                count=len(decompressed_list),
                workers=max_workers,
                latency_ms=round(decompress_ms, 2),
            )
        else:
            decompressed_list = decrypted_list

        # NEW: Parallelize final deserialization too (big win for large objects)
        deserialize_start = time.perf_counter()

        if len(decompressed_list) > 10:  # Only worth it for larger batches
            max_workers = min(4, len(decompressed_list))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        lambda args: self._deserialize_from_bytes(args[0], args[1]),
                        zip(decompressed_list, type_list),
                    )
                )

            deserialize_ms = (time.perf_counter() - deserialize_start) * 1000
            logger.debug(
                "batch_deserialize_parallel",
                count=len(results),
                workers=max_workers,
                latency_ms=round(deserialize_ms, 2),
            )
        else:
            # Sequential for small batches (less overhead)
            results = [
                self._deserialize_from_bytes(data, result_type)
                for data, result_type in zip(decompressed_list, type_list)
            ]

        total_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "batch_deserialize_complete",
            count=len(results),
            total_ms=round(total_ms, 2),
            per_item_ms=round(total_ms / len(results), 2),
        )

        return results

    def deserialize_batch(self, data_list: List[Tuple[str, str]]) -> List[Any]:
        """Deserialize multiple results with parallel decompression/decryption."""
        if not data_list:
            return []

        if len(data_list) <= 3:
            return [
                self.deserialize(data, result_type) for data, result_type in data_list
            ]

        if self.encryptor or self.compressor:
            return self._deserialize_batch_optimized(data_list)
        else:
            return [
                self.deserialize(data, result_type) for data, result_type in data_list
            ]

    def _is_special_type(self, value: Any) -> bool:
        """Check if value is a DataFrame, array, or other special type."""
        try:
            result_class_name = (
                f"{value.__class__.__module__}.{value.__class__.__name__}"
            )

            return (
                result_class_name == "pyarrow.lib.Table"
                or result_class_name.startswith("polars.")
                or result_class_name == "pandas.core.frame.DataFrame"
                or result_class_name == "numpy.ndarray"
            )
        except AttributeError:
            return False

    def _serialize_to_bytes(self, result: Any) -> Tuple[bytes, str]:
        """
        Serialize Python object to bytes.

        Returns:
            Tuple of (serialized_bytes, type_name)
        """
        if isinstance(result, dict):
            serialized_str, result_type = self._serialize_nested_dict(result)
        elif isinstance(result, (list, tuple)):
            serialized_str, result_type = self._serialize_nested_list(result)
        else:
            result_class_name = (
                f"{result.__class__.__module__}.{result.__class__.__name__}"
            )

            if result_class_name == "pyarrow.lib.Table":
                serialized_str, result_type = (
                    self._serialize_arrow_table(result),
                    "arrow_ipc",
                )
            elif result_class_name.startswith("polars."):
                serialized_str, result_type = (
                    self._serialize_polars(result),
                    "polars_arrow",
                )
            elif result_class_name == "pandas.core.frame.DataFrame":
                serialized_str, result_type = (
                    self._serialize_pandas(result),
                    "pandas_arrow",
                )
            elif result_class_name == "numpy.ndarray":
                serialized_str, result_type = (
                    self._serialize_numpy(result),
                    "numpy_arrow",
                )
            else:
                serialized_str, result_type = self._serialize_json(result)

        if isinstance(serialized_str, str):
            return serialized_str.encode("utf-8"), result_type
        else:
            return serialized_str, result_type

    def _deserialize_from_bytes(self, data: bytes, result_type: str) -> Any:
        """
        Deserialize bytes to Python object.

        Optimization: Uses orjson.loads(bytes) directly when possible.
        """
        if result_type == "nested_dict":
            return self._deserialize_nested_dict_bytes(data)

        if result_type == "nested_list":
            return self._deserialize_nested_list_bytes(data)

        if result_type == "numpy_arrow":
            return self._deserialize_numpy_bytes(data)

        if result_type == "arrow_ipc":
            buffer = base64.b64decode(data)
            reader = pa.ipc.open_stream(buffer)
            return reader.read_all()

        elif result_type == "pandas_arrow":
            buffer = base64.b64decode(data)
            reader = pa.ipc.open_stream(buffer)
            arrow_table = reader.read_all()
            return arrow_table.to_pandas()

        elif result_type == "polars_arrow":
            buffer = base64.b64decode(data)
            reader = pa.ipc.open_stream(buffer)
            arrow_table = reader.read_all()
            try:
                import polars as pl

                return pl.from_arrow(arrow_table)
            except ImportError:
                logger.warning("polars_not_available", returning="arrow_table")
                return arrow_table

        elif result_type in ("orjson", "json"):
            if HAS_ORJSON:
                return orjson.loads(data)
            else:
                return json.loads(data.decode("utf-8"))

        else:
            logger.error("unknown_result_type", type=result_type)
            return None

    def _deserialize_nested_dict_bytes(self, data: bytes) -> dict:
        """Deserialize nested dict directly from bytes."""
        if HAS_ORJSON:
            parsed = orjson.loads(data)
        else:
            parsed = json.loads(data.decode("utf-8"))

        base_dict = parsed["base"]
        special_types = parsed["special"]

        result = {}
        for key, value in base_dict.items():
            if isinstance(value, str) and value.startswith("__special__"):
                special_key = value.replace("__special__", "")
                special_data = special_types[special_key]["data"]
                special_type = special_types[special_key]["type"]

                if isinstance(special_data, str):
                    special_data = special_data.encode("utf-8")

                result[key] = self._deserialize_from_bytes(special_data, special_type)
            else:
                result[key] = value

        return result

    def _deserialize_nested_list_bytes(self, data: bytes) -> list:
        """Deserialize nested list directly from bytes."""
        if HAS_ORJSON:
            parsed = orjson.loads(data)
        else:
            parsed = json.loads(data.decode("utf-8"))

        base_list = parsed["base"]
        special_types = parsed["special"]

        result = []
        for idx, value in enumerate(base_list):
            if isinstance(value, str) and value.startswith("__special__"):
                special_key = str(idx)
                special_data = special_types[special_key]["data"]
                special_type = special_types[special_key]["type"]

                if isinstance(special_data, str):
                    special_data = special_data.encode("utf-8")

                result.append(self._deserialize_from_bytes(special_data, special_type))
            else:
                result.append(value)

        return result

    def _deserialize_numpy_bytes(self, data: bytes) -> Any:
        """Deserialize numpy array from bytes."""
        import numpy as np

        if HAS_ORJSON:
            parsed = orjson.loads(data)
        else:
            parsed = json.loads(data.decode("utf-8"))

        encoded_data = parsed["data"]
        metadata = parsed["metadata"]

        buffer = base64.b64decode(encoded_data.encode("utf-8"))
        reader = pa.ipc.open_stream(buffer)
        arrow_table = reader.read_all()

        flat_array = arrow_table["values"].to_numpy()

        shape = tuple(metadata["shape"])
        dtype = np.dtype(metadata["dtype"])

        return flat_array.reshape(shape).astype(dtype)

    # Los métodos _serialize_* permanecen sin cambios (ya óptimos con orjson/arrow)
    def _serialize_json(self, result: Any) -> Tuple[str, str]:
        """Serialize JSON-compatible objects."""
        try:
            if HAS_ORJSON:
                return orjson.dumps(result).decode("utf-8"), "orjson"
            else:
                return json.dumps(result), "json"
        except (TypeError, ValueError) as e:
            result_class = f"{result.__class__.__module__}.{result.__class__.__name__}"
            logger.error("json_serialization_failed", error=str(e), type=result_class)
            raise TypeError(
                f"Type {result_class} is not serializable. "
                f"Supported: dict, list, str, int, float, bool, None, "
                f"pandas.DataFrame, polars.DataFrame, pyarrow.Table, numpy.ndarray"
            )

    def _serialize_arrow_table(self, table: Any) -> str:
        """Serialize PyArrow table."""
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        buffer = sink.getvalue()
        return base64.b64encode(buffer.to_pybytes()).decode("utf-8")

    def _serialize_pandas(self, df: Any) -> str:
        """Serialize Pandas DataFrame."""
        arrow_table = pa.Table.from_pandas(df)
        return self._serialize_arrow_table(arrow_table)

    def _serialize_polars(self, df: Any) -> str:
        """Serialize Polars DataFrame."""
        arrow_table = df.to_arrow()
        return self._serialize_arrow_table(arrow_table)

    def _serialize_numpy(self, arr: Any) -> str:
        """Serialize NumPy array."""
        arrow_array = pa.array(arr.flatten())
        arrow_table = pa.Table.from_arrays([arrow_array], names=["values"])

        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
            writer.write_table(arrow_table)
        buffer = sink.getvalue()

        metadata = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        encoded = base64.b64encode(buffer.to_pybytes()).decode("utf-8")
        final_data = {"data": encoded, "metadata": metadata}

        if HAS_ORJSON:
            return orjson.dumps(final_data).decode("utf-8")
        else:
            return json.dumps(final_data)

    def _serialize_nested_dict(self, result: dict) -> Tuple[str, str]:
        """Serialize dict that may contain DataFrames/arrays."""
        serialized_dict = {}
        special_types = {}

        for key, value in result.items():
            if self._is_special_type(value):
                serialized_value, value_type = self._serialize_to_bytes(value)
                if isinstance(serialized_value, bytes):
                    serialized_value = serialized_value.decode("utf-8")
                serialized_dict[key] = f"__special__{key}"
                special_types[key] = {"data": serialized_value, "type": value_type}
            elif isinstance(value, (dict, list, tuple)):
                nested_serialized, nested_type = self._serialize_to_bytes(value)
                if isinstance(nested_serialized, bytes):
                    nested_serialized = nested_serialized.decode("utf-8")
                if nested_type.startswith("nested_") or "_" in nested_type:
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
                serialized_value, value_type = self._serialize_to_bytes(value)
                if isinstance(serialized_value, bytes):
                    serialized_value = serialized_value.decode("utf-8")
                serialized_list.append(f"__special__{idx}")
                special_types[str(idx)] = {"data": serialized_value, "type": value_type}
            elif isinstance(value, (dict, list, tuple)):
                nested_serialized, nested_type = self._serialize_to_bytes(value)
                if isinstance(nested_serialized, bytes):
                    nested_serialized = nested_serialized.decode("utf-8")
                if nested_type.startswith("nested_") or "_" in nested_type:
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
