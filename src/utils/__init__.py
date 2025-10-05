"""Utilities for Memora."""

from .embeddings import (
    cosine_similarity,
    cosine_similarity_batch,
    euclidean_distance,
    normalize_embedding,
)
from .fingerprint import create_fingerprint, fingerprint_matches
from .serialization import serialize, deserialize, is_serializable
from .hashing import (
    content_hash,
    short_hash,
    verify_content_hash,
    dependency_chain_hash,
)

__all__ = [
    "cosine_similarity",
    "cosine_similarity_batch",
    "euclidean_distance",
    "normalize_embedding",
    "create_fingerprint",
    "fingerprint_matches",
    "serialize",
    "deserialize",
    "is_serializable",
    "content_hash",
    "short_hash",
    "verify_content_hash",
    "dependency_chain_hash",
]
