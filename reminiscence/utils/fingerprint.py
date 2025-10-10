"""Fingerprinting utilities for context hashing."""

import json
import hashlib
from typing import Dict, Any
from functools import lru_cache


@lru_cache(maxsize=2048)
def _compute_fingerprint(context_json: str) -> str:
    """
    Compute SHA256 fingerprint from JSON string (cached).

    LRU cache is safe here because:
    - Context strings are immutable
    - Same context = same hash (deterministic)
    - Common in batch operations (repeated contexts)

    Args:
        context_json: JSON-serialized context string

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(context_json.encode("utf-8")).hexdigest()


def create_fingerprint(context: Dict[str, Any]) -> str:
    """
    Create deterministic fingerprint from context.

    Uses LRU cache for performance when contexts repeat.

    Args:
        context: Context dictionary

    Returns:
        SHA256 fingerprint (hex string)

    Examples:
        >>> ctx = {"agent": "test", "session": "123"}
        >>> fp = create_fingerprint(ctx)
        >>> len(fp)
        64
    """
    # Sort keys for determinism
    context_json = json.dumps(context, sort_keys=True)
    return _compute_fingerprint(context_json)


def clear_fingerprint_cache():
    """
    Clear fingerprint cache (useful for testing).

    In production, you rarely need this since LRU evicts automatically.
    """
    _compute_fingerprint.cache_clear()


def get_fingerprint_cache_info():
    """
    Get cache statistics.

    Returns:
        CacheInfo with hits, misses, maxsize, currsize

    Examples:
        >>> info = get_fingerprint_cache_info()
        >>> print(f"Hit rate: {info.hits / (info.hits + info.misses):.2%}")
    """
    return _compute_fingerprint.cache_info()


def compute_query_hash(query_text: str, context: Dict[str, Any]) -> str:
    """Compute hash of query + context for exact matching."""
    context_json = json.dumps(context, sort_keys=True)
    data = f"{query_text}:{context_json}"
    return hashlib.sha256(data.encode()).hexdigest()
