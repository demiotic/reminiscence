"""Embedding model abstractions with auto-detection."""

from .base import EmbeddingModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_embedder(config) -> EmbeddingModel:
    """
    Factory to create embedder from config with auto-detection.

    Priority:
    1. Check config.embedding_backend if set explicitly
    2. Try FastEmbed (if installed)
    3. Fallback to SentenceTransformers (if installed)
    4. Error if neither available
    """
    backend = getattr(config, "embedding_backend", "auto")

    if backend == "fastembed":
        return _create_fastembed(config)
    elif backend == "sentence-transformers":
        return _create_sentence_transformers(config)
    elif backend == "auto":
        # Auto-detect: try fastembed first
        embedder = _try_fastembed(config)
        if embedder:
            return embedder

        # Fallback to sentence-transformers
        embedder = _try_sentence_transformers(config)
        if embedder:
            return embedder

        # Neither available
        raise ImportError(
            "No embedding backend found. Install one of:\n"
            "  pip install reminiscence[fastembed]       (~100MB, recommended)\n"
            "  pip install reminiscence[torch-cpu]       (~500MB)\n"
            "  pip install reminiscence[torch-cuda]      (~4GB)"
        )
    else:
        raise ValueError(f"Unknown embedding_backend: {backend}")


def _try_fastembed(config):
    """Try to create FastEmbed embedder."""
    try:
        from .fastembed import FastEmbedEmbedder

        logger.info("using_embedding_backend", backend="fastembed")
        return FastEmbedEmbedder(config)
    except ImportError:
        logger.debug("fastembed_not_available")
        return None


def _try_sentence_transformers(config):
    """Try to create SentenceTransformers embedder."""
    try:
        from .sentence_transformers import SentenceTransformerEmbedder

        logger.info("using_embedding_backend", backend="sentence-transformers")
        return SentenceTransformerEmbedder(config)
    except ImportError:
        logger.debug("sentence_transformers_not_available")
        return None


def _create_fastembed(config):
    """Force create FastEmbed embedder."""
    try:
        from .fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder(config)
    except ImportError as e:
        raise ImportError(
            "FastEmbed not installed. Install with:\n"
            "  pip install reminiscence[fastembed]"
        ) from e


def _create_sentence_transformers(config):
    """Force create SentenceTransformers embedder."""
    try:
        from .sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(config)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers not installed. Install with:\n"
            "  pip install reminiscence[torch-cpu] or [torch-cuda]"
        ) from e


__all__ = ["EmbeddingModel", "create_embedder"]
