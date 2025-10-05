"""Fingerprinting de contexto con normalización robusta."""

import hashlib
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_fingerprint(context: Dict[str, Any]) -> str:
    """
    Genera fingerprint del contexto estructural.

    El fingerprint agrupa queries por contexto COMPLETO,
    mientras que el embedding maneja la similitud semántica.

    Normaliza tipos para evitar colisiones triviales:
    - Floats vs ints: 30.0 == 30
    - Listas desordenadas: [1, 2] == [2, 1] si representan sets
    - Bools: True normalizado a "true"

    Args:
        context: Diccionario con cualquier key/value

    Returns:
        Hash SHA256 (64 caracteres hex)

    Example:
        >>> create_fingerprint({"timeout": 30.0, "tools": ["search"]})
        >>> create_fingerprint({"timeout": 30, "tools": ["search"]})
        # Ambos producen el mismo hash
    """
    # Si el contexto está vacío, usar default
    if not context:
        normalized = {"__default__": True}
    else:
        normalized = _normalize_dict(context)

    # Serializar de forma determinista
    fp_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    hash_result = hashlib.sha256(fp_str.encode("utf-8")).hexdigest()

    logger.debug(f"Fingerprint: {hash_result[:16]}... | context={normalized}")

    return hash_result


def _normalize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza diccionario recursivamente."""
    result = {}

    for key, value in sorted(d.items()):
        result[key] = _normalize_value(value)

    return result


def _normalize_value(value: Any) -> Any:
    """Normaliza un valor individual."""

    # None
    if value is None:
        return None

    # Bools (antes de ints, porque bool es subclase de int)
    if isinstance(value, bool):
        return value

    # Números: normalizar floats que son enteros
    if isinstance(value, (int, float)):
        # Si es float pero representa un entero, convertir
        if isinstance(value, float) and value.is_integer():
            return int(value)
        # Redondear floats a 10 decimales para evitar drift
        if isinstance(value, float):
            return round(value, 10)
        return value

    # Strings
    if isinstance(value, str):
        return value

    # Listas: normalizar elementos (mantener orden)
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]

    # Sets: ordenar para determinismo
    if isinstance(value, set):
        return sorted([_normalize_value(item) for item in value])

    # Dicts: recursión
    if isinstance(value, dict):
        return _normalize_dict(value)

    # Otros tipos: convertir a string
    logger.debug(f"Normalizando tipo no estándar: {type(value).__name__} → str")
    return str(value)


def fingerprint_matches(context1: Dict[str, Any], context2: Dict[str, Any]) -> bool:
    """
    Verifica si dos contextos producen el mismo fingerprint.

    Útil para testing.

    Args:
        context1: Primer contexto
        context2: Segundo contexto

    Returns:
        True si los fingerprints coinciden
    """
    return create_fingerprint(context1) == create_fingerprint(context2)
