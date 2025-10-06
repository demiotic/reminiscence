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
