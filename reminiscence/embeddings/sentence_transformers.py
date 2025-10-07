"""Sentence Transformers implementation."""

from typing import List
from functools import cached_property
from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedder(EmbeddingModel):
    """Embedder using sentence-transformers library."""

    def __init__(self, config):
        self.config = config

    @cached_property
    def _model(self) -> SentenceTransformer:
        """Lazy-load model."""
        if self.config.use_onnx:
            model_kwargs = {"file_name": self.config.onnx_model_file}

            logger.info(
                "loading_model",
                model=self.config.model_name,
                backend="onnx",
                file=self.config.onnx_model_file,
            )

            model = SentenceTransformer(
                self.config.model_name, backend="onnx", model_kwargs=model_kwargs
            )
        else:
            logger.info(
                "loading_model", model=self.config.model_name, backend="pytorch"
            )
            model = SentenceTransformer(self.config.model_name)

        # Log ONNX info
        self._log_onnx_info(model)

        return model

    def _log_onnx_info(self, model: SentenceTransformer):
        """Log which ONNX model was loaded."""
        try:
            if hasattr(model, "_backend") and hasattr(model._backend, "_model_path"):
                model_path = model._backend._model_path
                logger.info(
                    "onnx_model_loaded",
                    model_path=str(model_path),
                    is_quantized="qint8" in str(model_path)
                    or "quint8" in str(model_path),
                )
        except Exception as e:
            logger.debug("onnx_model_info", error=str(e))

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        """Generate normalized embedding."""
        try:
            embedding = self._model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(
                "embedding_failed", error=str(e), text_preview=text[:50], exc_info=True
            )
            raise
