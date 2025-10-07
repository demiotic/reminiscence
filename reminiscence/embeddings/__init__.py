"""Embedding model abstractions."""

from .base import EmbeddingModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_embedder(config) -> EmbeddingModel:
    """
    Factory to create embedder from config.

    Only FastEmbed is supported.
    """
    backend = config.embedding_backend

    if backend in ("fastembed", "auto"):
        return _create_fastembed(config)
    else:
        raise ValueError(
            f"Unknown embedding_backend: {backend}. Only 'fastembed' is supported."
        )


def _create_fastembed(config):
    """Create FastEmbed embedder."""
    try:
        from .fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder(config)
    except ImportError as e:
        raise ImportError(
            "FastEmbed not installed. Install with:\n  pip install reminiscence"
        ) from e


__all__ = ["EmbeddingModel", "create_embedder"]
