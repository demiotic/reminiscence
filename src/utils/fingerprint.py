"""Context fingerprinting for fast exact matching."""

import hashlib
import json
from typing import Dict, Any


def create_fingerprint(context: Dict[str, Any]) -> str:
    """
    Generate SHA256 fingerprint of context for fast exact matching.

    Normalizes and sorts keys for deterministic hashing.

    Args:
        context: Context dictionary

    Returns:
        SHA256 hash (64 hex characters)
    """
    if not context:
        context = {}

    context_json = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(context_json.encode("utf-8")).hexdigest()
