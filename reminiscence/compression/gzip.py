"""Gzip compression backend."""

import gzip
import time
from .base import Compressor
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GzipCompressor(Compressor):
    """
    Gzip compression implementation.

    Gzip is widely supported but slower than zstd.
    Recommended compression levels:
    - 1-3: Fast, lower compression
    - 4-6: Balanced (default: 6)
    - 7-9: Slower, higher compression
    """

    def __init__(self, level: int = 6):
        """
        Initialize Gzip compressor.

        Args:
            level: Compression level (1-9, default: 6)

        Raises:
            ValueError: If level out of range
        """
        if not (1 <= level <= 9):
            raise ValueError(f"Gzip level must be 1-9, got {level}")

        self._algorithm = "gzip"
        self._level = level

        logger.debug(
            "gzip_compressor_initialized",
            level=level,
        )

    @property
    def algorithm(self) -> str:
        """Algorithm name."""
        return self._algorithm

    @property
    def level(self) -> int:
        """Compression level."""
        return self._level

    def compress(self, data: bytes) -> bytes:
        """
        Compress data using Gzip.

        Args:
            data: Raw bytes

        Returns:
            Compressed bytes
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")

        start = time.perf_counter()
        original_size = len(data)

        try:
            compressed = gzip.compress(data, compresslevel=self._level)
            compressed_size = len(compressed)
            elapsed_ms = (time.perf_counter() - start) * 1000

            ratio = compressed_size / original_size if original_size > 0 else 0
            savings_pct = (1 - ratio) * 100

            logger.debug(
                "gzip_compress_complete",
                original_bytes=original_size,
                compressed_bytes=compressed_size,
                ratio=round(ratio, 3),
                savings_pct=round(savings_pct, 1),
                latency_ms=round(elapsed_ms, 2),
            )

            return compressed

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "gzip_compress_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(elapsed_ms, 2),
            )
            raise

    def decompress(self, data: bytes) -> bytes:
        """
        Decompress Gzip data.

        Args:
            data: Compressed bytes

        Returns:
            Original bytes
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")

        start = time.perf_counter()
        compressed_size = len(data)

        try:
            decompressed = gzip.decompress(data)
            decompressed_size = len(decompressed)
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.debug(
                "gzip_decompress_complete",
                compressed_bytes=compressed_size,
                decompressed_bytes=decompressed_size,
                latency_ms=round(elapsed_ms, 2),
            )

            return decompressed

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "gzip_decompress_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(elapsed_ms, 2),
            )
            raise
