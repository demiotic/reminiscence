"""Embedding model abstractions."""

from __future__ import annotations

from .base import EmbeddingModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_embedder(config) -> EmbeddingModel:
    """Factory to create embedder from config.

    Args:
        config: Configuration object with embedding_backend attribute.

    Returns:
        EmbeddingModel instance (currently only FastEmbed).

    Raises:
        ValueError: If embedding_backend is not supported.
        ImportError: If required library not installed.
    """
    backend = config.embedding_backend

    if backend in ("fastembed", "auto"):
        return _create_fastembed(config)
    else:
        raise ValueError(
            f"Unknown embedding_backend: {backend}. Only 'fastembed' is supported."
        )


def _create_fastembed(config) -> EmbeddingModel:
    """Create FastEmbed embedder.

    Args:
        config: Configuration object.

    Returns:
        FastEmbedEmbedder instance.

    Raises:
        ImportError: If fastembed not installed.
    """
    try:
        from .fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder(config)
    except ImportError as e:
        raise ImportError(
            "FastEmbed not installed. Install with:\n  pip install reminiscence"
        ) from e


__all__ = ["EmbeddingModel", "create_embedder"]
