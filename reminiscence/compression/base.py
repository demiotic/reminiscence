"""Base class for compression backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Compressor(ABC):
    """Base class for compression algorithms."""

    @property
    @abstractmethod
    def algorithm(self) -> str:
        """Algorithm name.

        Returns:
            Algorithm identifier string.
        """
        pass

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data.

        Args:
            data: Raw bytes to compress.

        Returns:
            Compressed bytes.

        Raises:
            TypeError: If data is not bytes.
        """
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data.

        Args:
            data: Compressed bytes.

        Returns:
            Decompressed bytes.

        Raises:
            TypeError: If data is not bytes.
        """
        pass
