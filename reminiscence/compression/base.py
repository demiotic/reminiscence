"""Base class for compression backends."""

from abc import ABC, abstractmethod


class Compressor(ABC):
    """Base class for compression algorithms."""

    @property
    @abstractmethod
    def algorithm(self) -> str:
        """Algorithm name."""
        pass

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass
