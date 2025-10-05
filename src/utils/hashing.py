"""Content-addressable hashing para step cache."""

import hashlib
import json
from typing import Any, Dict, List


def content_hash(
    agent_id: str,
    agent_version: str,
    config: Dict[str, Any],
    input_data: Any,
    dependencies: List[str] = None,
) -> str:
    """
    Genera content hash determinista para step cache.

    Similar a Docker layer hashing: hash(agent + version + config + inputs + deps)

    Args:
        agent_id: Identificador del agente
        agent_version: Versión del código (ej: "v1.2.3", git hash)
        config: Configuración del agente (params, model, etc.)
        input_data: Datos de entrada (query, context, etc.)
        dependencies: Lista de hashes de steps upstream (opcional)

    Returns:
        SHA256 hash hexadecimal (64 caracteres)

    Example:
        >>> content_hash(
        ...     agent_id="sql_agent",
        ...     agent_version="v1.2.3",
        ...     config={"db": "prod", "timeout": 30},
        ...     input_data="SELECT * FROM sales"
        ... )
        'a3f5b2c1...'
    """
    components = {
        "agent_id": agent_id,
        "agent_version": agent_version,
        "config": _serialize_deterministic(config),
        "input": _serialize_deterministic(input_data),
    }

    if dependencies:
        # Ordenar deps para determinismo
        components["dependencies"] = sorted(dependencies)

    # Serializar de forma determinista
    serialized = json.dumps(components, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def short_hash(full_hash: str, length: int = 12) -> str:
    """
    Retorna versión corta del hash (estilo Docker/Git).

    Args:
        full_hash: Hash completo (64 chars)
        length: Número de caracteres a retornar

    Returns:
        Hash truncado

    Example:
        >>> short_hash("a3f5b2c1d4e6...", length=8)
        'a3f5b2c1'
    """
    return full_hash[:length]


def verify_content_hash(
    provided_hash: str,
    agent_id: str,
    agent_version: str,
    config: Dict,
    input_data: Any,
    dependencies: List[str] = None,
) -> bool:
    """
    Verifica que un hash coincida con los parámetros dados.

    Útil para debugging o validación de integridad.

    Returns:
        True si el hash es correcto
    """
    computed = content_hash(agent_id, agent_version, config, input_data, dependencies)
    return computed == provided_hash


def _serialize_deterministic(data: Any) -> str:
    """
    Serializa datos de forma determinista para hashing.

    Args:
        data: Cualquier objeto JSON-serializable

    Returns:
        String JSON con claves ordenadas
    """
    if isinstance(data, str):
        return data

    if isinstance(data, (int, float, bool, type(None))):
        return json.dumps(data)

    if isinstance(data, (list, tuple)):
        # Listas: serializar elementos en orden
        return json.dumps([_serialize_deterministic(item) for item in data])

    if isinstance(data, dict):
        # Dicts: ordenar claves
        return json.dumps(
            {k: _serialize_deterministic(v) for k, v in sorted(data.items())},
            sort_keys=True,
            separators=(",", ":"),
        )

    # Para otros tipos, usar repr() como fallback
    return repr(data)


def dependency_chain_hash(step_hashes: List[str]) -> str:
    """
    Genera hash de una cadena de dependencias.

    Útil para invalidación en cascada: si este hash cambia,
    todos los steps downstream deben invalidarse.

    Args:
        step_hashes: Lista ordenada de hashes de steps

    Returns:
        Hash de la cadena completa

    Example:
        >>> deps = ["hash1", "hash2", "hash3"]
        >>> dependency_chain_hash(deps)
        'f4a3b1...'
    """
    combined = "|".join(sorted(step_hashes))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
