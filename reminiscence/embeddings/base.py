"""Abstract base for embedding models."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract interface for embedding models."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of embeddings."""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Default implementation: sequential embedding (subclasses should override).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]
