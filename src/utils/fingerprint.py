"""Context fingerprinting with robust normalization."""

import hashlib
import json
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


def create_fingerprint(context: Dict[str, Any]) -> str:
    """
    Generate fingerprint of structural context.

    The fingerprint groups queries by COMPLETE context,
    while the embedding handles semantic similarity.

    Normalizes types to avoid trivial collisions:
    - Floats vs ints: 30.0 == 30
    - Unordered lists: [1, 2] == [2, 1] if they represent sets
    - Bools: True normalized to "true"

    Args:
        context: Dictionary with any key/value pairs

    Returns:
        SHA256 hash (64 hex characters)

    Example:
        >>> create_fingerprint({"timeout": 30.0, "tools": ["search"]})
        >>> create_fingerprint({"timeout": 30, "tools": ["search"]})
        # Both produce the same hash
    """
    # If context is empty, use default
    if not context:
        normalized = {"__default__": True}
    else:
        normalized = _normalize_dict(context)

    # Serialize deterministically
    fp_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    hash_result = hashlib.sha256(fp_str.encode("utf-8")).hexdigest()

    logger.debug(f"Fingerprint: {hash_result[:16]}... | context={normalized}")

    return hash_result


def _normalize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dictionary recursively."""
    result = {}

    for key, value in sorted(d.items()):
        result[key] = _normalize_value(value)

    return result


def _normalize_value(value: Any) -> Any:
    """Normalize an individual value."""

    # None
    if value is None:
        return None

    # Bools (before ints, because bool is a subclass of int)
    if isinstance(value, bool):
        return value

    # Numbers: normalize floats that are integers
    if isinstance(value, (int, float)):
        # If it's a float but represents an integer, convert
        if isinstance(value, float) and value.is_integer():
            return int(value)
        # Round floats to 10 decimals to avoid drift
        if isinstance(value, float):
            return round(value, 10)
        return value

    # Strings
    if isinstance(value, str):
        return value

    # Lists: normalize elements (maintain order)
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]

    # Sets: sort for determinism
    if isinstance(value, set):
        return sorted([_normalize_value(item) for item in value])

    # Dicts: recursion
    if isinstance(value, dict):
        return _normalize_dict(value)

    # Other types: convert to string
    logger.debug(f"Normalizing non-standard type: {type(value).__name__} → str")
    return str(value)


def fingerprint_matches(context1: Dict[str, Any], context2: Dict[str, Any]) -> bool:
    """
    Verify if two contexts produce the same fingerprint.

    Useful for testing.

    Args:
        context1: First context
        context2: Second context

    Returns:
        True if fingerprints match
    """
    return create_fingerprint(context1) == create_fingerprint(context2)
