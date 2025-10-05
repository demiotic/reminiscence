"""Content-addressable hashing for step cache."""

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
    Generate deterministic content hash for step cache.

    Similar to Docker layer hashing: hash(agent + version + config + inputs + deps)

    Args:
        agent_id: Agent identifier
        agent_version: Code version (e.g., "v1.2.3", git hash)
        config: Agent configuration (params, model, etc.)
        input_data: Input data (query, context, etc.)
        dependencies: List of upstream step hashes (optional)

    Returns:
        SHA256 hexadecimal hash (64 characters)

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
        # Sort deps for determinism
        components["dependencies"] = sorted(dependencies)

    # Serialize deterministically
    serialized = json.dumps(components, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def short_hash(full_hash: str, length: int = 12) -> str:
    """
    Return short version of hash (Docker/Git style).

    Args:
        full_hash: Complete hash (64 chars)
        length: Number of characters to return

    Returns:
        Truncated hash

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
    Verify that a hash matches the given parameters.

    Useful for debugging or integrity validation.

    Returns:
        True if the hash is correct
    """
    computed = content_hash(agent_id, agent_version, config, input_data, dependencies)
    return computed == provided_hash


def _serialize_deterministic(data: Any) -> str:
    """
    Serialize data deterministically for hashing.

    Args:
        data: Any JSON-serializable object

    Returns:
        JSON string with sorted keys
    """
    if isinstance(data, str):
        return data

    if isinstance(data, (int, float, bool, type(None))):
        return json.dumps(data)

    if isinstance(data, (list, tuple)):
        # Lists: serialize elements in order
        return json.dumps([_serialize_deterministic(item) for item in data])

    if isinstance(data, dict):
        # Dicts: sort keys
        return json.dumps(
            {k: _serialize_deterministic(v) for k, v in sorted(data.items())},
            sort_keys=True,
            separators=(",", ":"),
        )

    # For other types, use repr() as fallback
    return repr(data)


def dependency_chain_hash(step_hashes: List[str]) -> str:
    """
    Generate hash of a dependency chain.

    Useful for cascade invalidation: if this hash changes,
    all downstream steps must be invalidated.

    Args:
        step_hashes: Ordered list of step hashes

    Returns:
        Hash of the complete chain

    Example:
        >>> deps = ["hash1", "hash2", "hash3"]
        >>> dependency_chain_hash(deps)
        'f4a3b1...'
    """
    combined = "|".join(sorted(step_hashes))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
