"""FastEmbed implementation (lightweight, ONNX-optimized)."""

import os
import time
import tempfile
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

    def __init__(self, config, metrics=None):
        """
        Initialize FastEmbed embedder.

        Args:
            config: Configuration object
            metrics: Optional CacheMetrics instance for tracking
        """
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed not installed. Install with: "
                "pip install reminiscence[fastembed]"
            )

        self.config = config
        self.metrics = metrics

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
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or os.path.join(
            tempfile.gettempdir(), "fastembed_cache"
        )

        logger.info(
            "loading_model",
            model=self.model_name,
            backend="fastembed-onnx",
            cache_dir=cache_dir,
        )

        try:
            logger.debug("attempting_local_cache_load", model=self.model_name)

            try:
                model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                logger.info("fastembed_model_loaded_from_cache", model=self.model_name)
                return model

            except (OSError, ValueError) as cache_error:
                logger.info(
                    "model_not_cached_downloading",
                    model=self.model_name,
                    reason=str(cache_error)[:100],
                )

                model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
                logger.info("fastembed_model_downloaded", model=self.model_name)
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
            logger.debug("computing_embedding_dim", model=self.model_name)
            test_emb = list(self._model.embed([""]))[0]
            self._cached_dim = len(test_emb)
            logger.info("embedding_dim_detected", dimension=self._cached_dim)

        return self._cached_dim

    def embed(self, text: str) -> List[float]:
        """Generate normalized embedding."""
        start = time.perf_counter()

        try:
            embeddings = list(self._model.embed([text]))
            embedding = embeddings[0].tolist()

            elapsed_ms = (time.perf_counter() - start) * 1000

            if self.metrics:
                if not hasattr(self.metrics, "embedding_generations"):
                    self.metrics.embedding_generations = 0
                self.metrics.embedding_generations += 1

                if not hasattr(self.metrics, "embedding_latencies_ms"):
                    self.metrics.embedding_latencies_ms = []
                self.metrics.embedding_latencies_ms.append(elapsed_ms)

                if len(self.metrics.embedding_latencies_ms) > 1000:
                    self.metrics.embedding_latencies_ms = (
                        self.metrics.embedding_latencies_ms[-1000:]
                    )

            logger.debug(
                "embedding_generated",
                latency_ms=round(elapsed_ms, 2),
                text_length=len(text),
            )

            return embedding

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if self.metrics:
                if not hasattr(self.metrics, "embedding_errors"):
                    self.metrics.embedding_errors = 0
                self.metrics.embedding_errors += 1

            logger.error(
                "embedding_failed",
                error=str(e),
                text_preview=text[:50],
                latency_ms=round(elapsed_ms, 2),
                exc_info=True,
            )
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start = time.perf_counter()
        batch_size = self.config.embedding_batch_size

        try:
            embeddings = list(self._model.embed(texts, batch_size=batch_size))
            result = [emb.tolist() for emb in embeddings]

            elapsed_ms = (time.perf_counter() - start) * 1000

            if self.metrics:
                if not hasattr(self.metrics, "embedding_generations"):
                    self.metrics.embedding_generations = 0
                self.metrics.embedding_generations += len(texts)

                if not hasattr(self.metrics, "embedding_latencies_ms"):
                    self.metrics.embedding_latencies_ms = []

                avg_latency_per_text = elapsed_ms / len(texts)
                self.metrics.embedding_latencies_ms.extend(
                    [avg_latency_per_text] * len(texts)
                )

                if len(self.metrics.embedding_latencies_ms) > 1000:
                    self.metrics.embedding_latencies_ms = (
                        self.metrics.embedding_latencies_ms[-1000:]
                    )

            logger.debug(
                "batch_embedding_generated",
                batch_size=len(texts),
                latency_ms=round(elapsed_ms, 2),
                latency_per_text_ms=round(elapsed_ms / len(texts), 2),
            )

            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if self.metrics:
                if not hasattr(self.metrics, "embedding_errors"):
                    self.metrics.embedding_errors = 0
                self.metrics.embedding_errors += 1

            logger.error(
                "batch_embedding_failed",
                error=str(e),
                batch_size=len(texts),
                latency_ms=round(elapsed_ms, 2),
                exc_info=True,
            )
            raise

    def get_embedding_stats(self) -> dict:
        """Get embedding-specific statistics."""
        if not self.metrics:
            return {}

        latencies = getattr(self.metrics, "embedding_latencies_ms", [])
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "total_generations": getattr(self.metrics, "embedding_generations", 0),
            "avg_latency_ms": round(avg_latency, 2),
            "errors": getattr(self.metrics, "embedding_errors", 0),
        }
