"""Fingerprinting de contexto."""

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

    Args:
        context: Diccionario con cualquier key/value

    Returns:
        Hash SHA256

    Example:
        >>> create_fingerprint({"tools": ["search"]})
        'a4d5a016dcc745a9'
    """
    # Si el contexto está vacío, usar default
    if not context:
        fingerprint = {"default": True}
    else:
        # Normalizar el contexto completo
        fingerprint = {}

        for key, value in sorted(context.items()):
            # Normalizar listas ordenándolas
            if isinstance(value, list):
                fingerprint[key] = sorted(str(v) for v in value)
            # Normalizar dicts recursivamente
            elif isinstance(value, dict):
                fingerprint[key] = value  # json.dumps lo manejará
            else:
                fingerprint[key] = value

    fp_str = json.dumps(fingerprint, sort_keys=True)
    hash_result = hashlib.sha256(fp_str.encode()).hexdigest()
    logger.debug(f"Fingerprint: {hash_result[:16]} | context={fingerprint}")
    return hash_result
