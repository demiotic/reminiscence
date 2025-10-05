"""Serialización/deserialización robusta para caché.

Estrategia:
- orjson como formato base (rápido, seguro, portable)
- Handlers custom para pandas/polars/numpy
- Arrow IPC como fallback para DataFrames gigantes (>10MB)
- Sin pickle por seguridad
"""

import logging
from typing import Any
import orjson

logger = logging.getLogger(__name__)

# Imports opcionales - solo si están disponibles
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

# Threshold para decidir entre orjson vs Arrow IPC (bytes)
ARROW_THRESHOLD = 10_000_000  # 10MB


def serialize(data: Any) -> bytes:
    """
    Serializa resultado para almacenamiento.

    Estrategia adaptativa:
    - DataFrames grandes (>10MB) → Arrow IPC
    - Todo lo demás → orjson con handlers

    Args:
        data: Cualquier objeto Python

    Returns:
        bytes serializados

    Raises:
        TypeError: Si el tipo no es serializable
    """
    # Detectar DataFrames grandes para Arrow IPC
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
    Deserializa resultado desde caché.

    Args:
        data: bytes serializados

    Returns:
        Objeto Python original
    """
    # Detectar formato Arrow IPC
    if data.startswith(b"__arrow_pandas__:"):
        return _deserialize_arrow_pandas(data)

    if data.startswith(b"__arrow_polars__:"):
        return _deserialize_arrow_polars(data)

    # Default: orjson
    obj = orjson.loads(data)
    return _reconstruct_nested(obj)


def _orjson_handler(obj: Any) -> Any:
    """Handler para tipos custom en orjson."""

    # Pandas DataFrame → dict con metadata
    if HAS_PANDAS and isinstance(obj, pd.DataFrame):
        # FIX: usar to_dict("tight") o manual serialization
        return {
            "__type__": "pandas_df",
            "columns": obj.columns.tolist(),
            "index": obj.index.tolist(),
            "data": obj.values.tolist(),  # Lista de listas
            "index_name": obj.index.name,
        }

    # Pandas Series
    if HAS_PANDAS and isinstance(obj, pd.Series):
        # FIX: convertir índice a string para evitar non-string keys
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
    """Reconstruye objetos custom de forma recursiva."""

    # Diccionarios pueden contener objetos serializados
    if isinstance(obj, dict):
        # Detectar objetos custom por __type__
        if "__type__" in obj:
            obj_type = obj["__type__"]

            if obj_type == "pandas_df" and HAS_PANDAS:
                # FIX: reconstruir desde valores/columnas/índice
                df = pd.DataFrame(
                    data=obj["data"], columns=obj["columns"], index=obj["index"]
                )
                if obj.get("index_name"):
                    df.index.name = obj["index_name"]
                return df

            elif obj_type == "pandas_series" and HAS_PANDAS:
                # FIX: reconstruir desde valores e índice
                return pd.Series(
                    data=obj["values"], index=obj["index"], name=obj.get("name")
                )

            elif obj_type == "polars_df" and HAS_POLARS:
                return pl.DataFrame(obj["data"])

            elif obj_type == "numpy" and HAS_NUMPY:
                arr = np.array(obj["data"], dtype=obj["dtype"])
                return arr.reshape(obj["shape"])

        # Reconstruir recursivamente
        return {k: _reconstruct_nested(v) for k, v in obj.items()}

    # Listas
    elif isinstance(obj, list):
        return [_reconstruct_nested(item) for item in obj]

    return obj


def _serialize_arrow_pandas(df: "pd.DataFrame") -> bytes:
    """Serializa pandas DataFrame grande usando Arrow IPC."""
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
    """Deserializa pandas DataFrame desde Arrow IPC."""
    if not HAS_ARROW or not HAS_PANDAS:
        raise RuntimeError("pyarrow and pandas required")

    payload = data[len(b"__arrow_pandas__:") :]
    buffer = pa.py_buffer(payload)
    reader = pa.ipc.open_stream(buffer)

    return reader.read_pandas()


def _serialize_arrow_polars(df: "pl.DataFrame") -> bytes:
    """Serializa polars DataFrame usando Arrow IPC."""
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
    """Deserializa polars DataFrame desde Arrow IPC."""
    if not HAS_ARROW or not HAS_POLARS:
        raise RuntimeError("pyarrow and polars required")

    payload = data[len(b"__arrow_polars__:") :]
    buffer = pa.py_buffer(payload)
    reader = pa.ipc.open_stream(buffer)

    return pl.from_arrow(reader.read_all())


def is_serializable(data: Any) -> bool:
    """
    Verifica si un objeto es serializable.

    Args:
        data: Objeto a verificar

    Returns:
        True si es serializable
    """
    try:
        serialize(data)
        return True
    except (TypeError, ValueError):
        return False
