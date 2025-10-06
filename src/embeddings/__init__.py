"""Embedding model abstractions."""

from .base import EmbeddingModel
from .sentence_transformers import SentenceTransformerEmbedder


def create_embedder(config) -> EmbeddingModel:
    """Factory to create embedder from config."""
    return SentenceTransformerEmbedder(config)


__all__ = ["EmbeddingModel", "create_embedder"]
