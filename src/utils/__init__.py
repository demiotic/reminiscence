"""Utilidades para Memora."""

from .fingerprint import create_fingerprint
from .embeddings import (
    cosine_similarity,
    cosine_similarity_batch,
    euclidean_distance,
    normalize_embedding,
)
from .serialization import serialize, deserialize, is_json_serializable
from .hashing import (
    content_hash,
    short_hash,
    verify_content_hash,
    dependency_chain_hash,
)


__all__ = [
    # Fingerprinting
    "create_fingerprint",
    # Embeddings
    "cosine_similarity",
    "cosine_similarity_batch",
    "euclidean_distance",
    "normalize_embedding",
    # Serialization
    "serialize",
    "deserialize",
    "is_json_serializable",
    # Hashing
    "content_hash",
    "short_hash",
    "verify_content_hash",
    "dependency_chain_hash",
]
