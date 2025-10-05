"""Robust serialization/deserialization for cache.

Strategy:
- orjson as base format (fast, safe, portable)
- Custom handlers for pandas/polars/numpy
- Arrow IPC as fallback for giant DataFrames (>10MB)
- No pickle for security
"""

import logging
from typing import Any
import orjson


logger = logging.getLogger(__name__)


# Optional imports - only if available
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


try:
    import pyarrow as pa

    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False


# Threshold to decide between orjson vs Arrow IPC (bytes)
ARROW_THRESHOLD = 10_000_000  # 10MB


def serialize(data: Any) -> bytes:
    """
    Serialize result for storage.

    Adaptive strategy:
    - Large DataFrames (>10MB) → Arrow IPC
    - Everything else → orjson with handlers

    Args:
        data: Any Python object

    Returns:
        Serialized bytes

    Raises:
        TypeError: If type is not serializable
    """
    # Detect large DataFrames for Arrow IPC
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        estimated_size = data.memory_usage(deep=True).sum()
        if estimated_size > ARROW_THRESHOLD:
            return _serialize_arrow_pandas(data)

    if HAS_POLARS and isinstance(data, pl.DataFrame):
        estimated_size = data.estimated_size()
        if estimated_size > ARROW_THRESHOLD:
            return _serialize_arrow_polars(data)

    # Default: orjson
    try:
        return orjson.dumps(data, default=_orjson_handler)
    except TypeError as e:
        logger.error(f"Serialization failed for type {type(data).__name__}: {e}")
        raise TypeError(
            f"Type {type(data).__name__} is not serializable. "
            "Supported types: dict, list, str, int, float, bool, None, "
            "pandas.DataFrame, pandas.Series, polars.DataFrame, numpy.ndarray"
        ) from e


def deserialize(data: bytes) -> Any:
    """
    Deserialize result from cache.

    Args:
        data: Serialized bytes

    Returns:
        Original Python object
    """
    # Detect Arrow IPC format
    if data.startswith(b"__arrow_pandas__:"):
        return _deserialize_arrow_pandas(data)

    if data.startswith(b"__arrow_polars__:"):
        return _deserialize_arrow_polars(data)

    # Default: orjson
    obj = orjson.loads(data)
    return _reconstruct_nested(obj)


def _orjson_handler(obj: Any) -> Any:
    """Handler for custom types in orjson."""

    # Pandas DataFrame → dict with metadata
    if HAS_PANDAS and isinstance(obj, pd.DataFrame):
        return {
            "__type__": "pandas_df",
            "columns": obj.columns.tolist(),
            "index": obj.index.tolist(),
            "data": obj.values.tolist(),  # List of lists
            "index_name": obj.index.name,
        }

    # Pandas Series
    if HAS_PANDAS and isinstance(obj, pd.Series):
        return {
            "__type__": "pandas_series",
            "values": obj.values.tolist(),
            "index": obj.index.tolist(),
            "name": obj.name,
        }

    # Polars DataFrame
    if HAS_POLARS and isinstance(obj, pl.DataFrame):
        return {
            "__type__": "polars_df",
            "data": obj.to_dicts(),
            "schema": {col: str(dtype) for col, dtype in obj.schema.items()},
        }

    # Numpy array
    if HAS_NUMPY and isinstance(obj, np.ndarray):
        return {
            "__type__": "numpy",
            "data": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }

    # Numpy scalar
    if HAS_NUMPY and isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    raise TypeError(f"Type {type(obj)} not serializable")


def _reconstruct_nested(obj: Any) -> Any:
    """Reconstruct custom objects recursively."""

    # Dictionaries may contain serialized objects
    if isinstance(obj, dict):
        # Detect custom objects by __type__
        if "__type__" in obj:
            obj_type = obj["__type__"]

            if obj_type == "pandas_df" and HAS_PANDAS:
                df = pd.DataFrame(
                    data=obj["data"], columns=obj["columns"], index=obj["index"]
                )
                if obj.get("index_name"):
                    df.index.name = obj["index_name"]
                return df

            elif obj_type == "pandas_series" and HAS_PANDAS:
                return pd.Series(
                    data=obj["values"], index=obj["index"], name=obj.get("name")
                )

            elif obj_type == "polars_df" and HAS_POLARS:
                return pl.DataFrame(obj["data"])

            elif obj_type == "numpy" and HAS_NUMPY:
                arr = np.array(obj["data"], dtype=obj["dtype"])
                return arr.reshape(obj["shape"])

        # Reconstruct recursively
        return {k: _reconstruct_nested(v) for k, v in obj.items()}

    # Lists
    elif isinstance(obj, list):
        return [_reconstruct_nested(item) for item in obj]

    return obj


def _serialize_arrow_pandas(df: "pd.DataFrame") -> bytes:
    """Serialize large pandas DataFrame using Arrow IPC."""
    if not HAS_ARROW:
        raise RuntimeError("pyarrow required for large DataFrame serialization")

    table = pa.Table.from_pandas(df, preserve_index=True)
    sink = pa.BufferOutputStream()

    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)

    payload = sink.getvalue().to_pybytes()
    logger.debug(f"Serialized pandas DataFrame via Arrow IPC: {len(payload)} bytes")

    return b"__arrow_pandas__:" + payload


def _deserialize_arrow_pandas(data: bytes) -> "pd.DataFrame":
    """Deserialize pandas DataFrame from Arrow IPC."""
    if not HAS_ARROW or not HAS_PANDAS:
        raise RuntimeError("pyarrow and pandas required")

    payload = data[len(b"__arrow_pandas__:") :]
    buffer = pa.py_buffer(payload)
    reader = pa.ipc.open_stream(buffer)

    return reader.read_pandas()


def _serialize_arrow_polars(df: "pl.DataFrame") -> bytes:
    """Serialize polars DataFrame using Arrow IPC."""
    if not HAS_ARROW:
        raise RuntimeError("pyarrow required for large DataFrame serialization")

    table = df.to_arrow()
    sink = pa.BufferOutputStream()

    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)

    payload = sink.getvalue().to_pybytes()
    logger.debug(f"Serialized polars DataFrame via Arrow IPC: {len(payload)} bytes")

    return b"__arrow_polars__:" + payload


def _deserialize_arrow_polars(data: bytes) -> "pl.DataFrame":
    """Deserialize polars DataFrame from Arrow IPC."""
    if not HAS_ARROW or not HAS_POLARS:
        raise RuntimeError("pyarrow and polars required")

    payload = data[len(b"__arrow_polars__:") :]
    buffer = pa.py_buffer(payload)
    reader = pa.ipc.open_stream(buffer)

    return pl.from_arrow(reader.read_all())


def is_serializable(data: Any) -> bool:
    """
    Verify if an object is serializable.

    Args:
        data: Object to verify

    Returns:
        True if serializable
    """
    try:
        serialize(data)
        return True
    except (TypeError, ValueError):
        return False
