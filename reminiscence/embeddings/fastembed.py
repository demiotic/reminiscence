"""FastEmbed implementation (lightweight, ONNX-optimized)."""

from typing import List
from functools import cached_property

try:
    from fastembed import TextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

from .base import EmbeddingModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FastEmbedEmbedder(EmbeddingModel):
    """
    Embedder using FastEmbed library.

    Lightweight (~100MB) with ONNX Runtime optimization.
    No PyTorch/CUDA dependencies required.
    """

    # Map common model names to FastEmbed equivalents
    MODEL_MAP = {
        "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    }

    def __init__(self, config):
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed not installed. Install with: "
                "pip install reminiscence[fastembed]"
            )

        self.config = config

    @cached_property
    def _model(self) -> TextEmbedding:
        """Lazy-load FastEmbed model."""
        # Map model name if needed
        model_name = self.MODEL_MAP.get(self.config.model_name, self.config.model_name)

        logger.info(
            "loading_model",
            model=model_name,
            backend="fastembed-onnx",
        )

        try:
            model = TextEmbedding(model_name=model_name)
            logger.info("fastembed_model_loaded", model=model_name)
            return model
        except Exception as e:
            logger.error(
                "fastembed_load_failed", error=str(e), model=model_name, exc_info=True
            )
            raise

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        # FastEmbed doesn't expose dimension directly,
        # so we get it from a test embedding
        if not hasattr(self, "_cached_dim"):
            test_emb = list(self._model.embed(["test"]))[0]
            self._cached_dim = len(test_emb)
        return self._cached_dim

    def embed(self, text: str) -> List[float]:
        """Generate normalized embedding."""
        try:
            # FastEmbed returns generator, convert to list
            embeddings = list(self._model.embed([text]))
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(
                "embedding_failed", error=str(e), text_preview=text[:50], exc_info=True
            )
            raise
