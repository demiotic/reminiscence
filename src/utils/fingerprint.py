"""Fingerprinting de contexto."""

import hashlib
import json
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


def create_fingerprint(context: Dict[str, Any]) -> str:
    """
    Genera fingerprint del contexto estructural.

    El fingerprint agrupa queries por contexto (tools, constraints),
    mientras que el embedding maneja la similitud semántica.

    Args:
        context: Diccionario con:
            - tools: List[str] - herramientas disponibles
            - constraints: Dict - restricciones (idioma, formato, etc)
            - entities: List[str] - entidades mencionadas (opcional)

    Returns:
        Hash SHA256 de 16 caracteres

    Example:
        >>> create_fingerprint({"tools": ["search"]})
        'a4d5a016dcc745a9'
    """
    fingerprint = {}

    # Tools: qué puede hacer el agente
    if "tools" in context and context["tools"]:
        fingerprint["tools"] = sorted(context["tools"])

    # Constraints: restricciones de output
    if "constraints" in context and context["constraints"]:
        fingerprint["constraints"] = context["constraints"]

    # Entities: menciones específicas (opcional, experimental)
    if "entities" in context and context["entities"]:
        fingerprint["entities"] = sorted(context["entities"])

    # Default si vacío
    if not fingerprint:
        fingerprint["default"] = True

    fp_str = json.dumps(fingerprint, sort_keys=True)
    hash_result = hashlib.sha256(fp_str.encode()).hexdigest()[:16]

    logger.debug(f"Fingerprint generado: {hash_result} | context={fingerprint}")

    return hash_result
