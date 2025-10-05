"""Serialización/deserialización de resultados para caché."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def serialize(data: Any) -> str:
    """
    Serializa resultado para almacenamiento en caché.

    Estrategia multi-tier:
    1. String → retornar directo
    2. JSON-serializable → json.dumps()
    3. Fallback → pickle con warning

    Args:
        data: Cualquier objeto Python

    Returns:
        String serializado

    Example:
        >>> serialize({"status": "ok", "value": 42})
        '{"status":"ok","value":42}'
    """
    # String directo
    if isinstance(data, str):
        return data

    # Intentar JSON (preferido)
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        logger.warning(
            f"JSON serialization failed for type {type(data).__name__}: {e}. "
            "Falling back to pickle (less safe)."
        )

        # Fallback a pickle (con prefijo para identificar)
        return _serialize_pickle(data)


def deserialize(data: str) -> Any:
    """
    Deserializa resultado desde caché.

    Args:
        data: String serializado

    Returns:
        Objeto Python original

    Example:
        >>> deserialize('{"status":"ok"}')
        {'status': 'ok'}
    """
    # Detectar pickle por prefijo
    if data.startswith("__pickle__:"):
        return _deserialize_pickle(data)

    # Intentar JSON
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        # Es un string literal
        return data


def _serialize_pickle(data: Any) -> str:
    """Serializa usando pickle con base64 encoding."""
    import pickle
    import base64

    try:
        pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        encoded = base64.b64encode(pickled).decode("ascii")
        return f"__pickle__:{encoded}"
    except Exception as e:
        logger.error(f"Pickle serialization failed: {e}")
        raise ValueError(f"Unable to serialize type {type(data).__name__}") from e


def _deserialize_pickle(data: str) -> Any:
    """Deserializa desde pickle base64."""
    import pickle
    import base64

    try:
        encoded = data.replace("__pickle__:", "")
        pickled = base64.b64decode(encoded)
        return pickle.loads(pickled)
    except Exception as e:
        logger.error(f"Pickle deserialization failed: {e}")
        raise ValueError("Corrupted pickle data") from e


def is_json_serializable(data: Any) -> bool:
    """
    Verifica si un objeto es JSON-serializable.

    Args:
        data: Objeto a verificar

    Returns:
        True si es serializable sin pickle
    """
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False
