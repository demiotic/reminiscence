"""FastEmbed implementation (lightweight, ONNX-optimized)."""

from typing import List
from functools import cached_property

try:
    from fastembed import TextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

from .base import EmbeddingModel
from .model_registry import get_default_model
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FastEmbedEmbedder(EmbeddingModel):
    """Embedder using FastEmbed library."""

    def __init__(self, config):
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed not installed. Install with: "
                "pip install reminiscence[fastembed]"
            )

        self.config = config

        # Use backend default if no model specified
        if config.model_name is None:
            self.model_name = get_default_model("fastembed")
            logger.info(
                "using_default_model", backend="fastembed", model=self.model_name
            )
        else:
            self.model_name = config.model_name
            logger.info(
                "using_custom_model", backend="fastembed", model=self.model_name
            )

    @cached_property
    def _model(self) -> TextEmbedding:
        """Lazy-load FastEmbed model."""
        logger.info(
            "loading_model",
            model=self.model_name,
            backend="fastembed-onnx",
        )

        try:
            model = TextEmbedding(model_name=self.model_name)
            logger.info("fastembed_model_loaded", model=self.model_name)
            return model
        except Exception as e:
            logger.error(
                "fastembed_load_failed",
                error=str(e),
                model=self.model_name,
                exc_info=True,
            )
            raise

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if not hasattr(self, "_cached_dim"):
            # Compute from test embedding
            logger.debug("computing_embedding_dim", model=self.model_name)
            test_emb = list(self._model.embed([""]))[0]
            self._cached_dim = len(test_emb)
            logger.info("embedding_dim_detected", dimension=self._cached_dim)

        return self._cached_dim

    def embed(self, text: str) -> List[float]:
        """Generate normalized embedding."""
        try:
            embeddings = list(self._model.embed([text]))
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(
                "embedding_failed", error=str(e), text_preview=text[:50], exc_info=True
            )
            raise
