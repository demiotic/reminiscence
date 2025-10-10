"""Zstandard compression backend."""

import time
from .base import Compressor
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ZstdCompressor(Compressor):
    """
    Zstandard compression implementation.

    Zstandard offers excellent compression ratios with fast decompression.
    Recommended compression levels:
    - 1-3: Fast, lower compression
    - 3-6: Balanced (default: 3)
    - 7-22: Slower, higher compression
    """

    def __init__(self, level: int = 3):
        """
        Initialize Zstandard compressor.

        Args:
            level: Compression level (1-22, default: 3)

        Raises:
            ImportError: If zstandard library not installed
            ValueError: If level out of range
        """
        if not (1 <= level <= 22):
            raise ValueError(f"Zstandard level must be 1-22, got {level}")

        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError(
                "zstandard package is required for zstd compression. "
                "Install it with: pip install zstandard"
            )

        self._algorithm = "zstd"
        self._level = level
        self._compressor = zstd.ZstdCompressor(level=level)
        self._decompressor = zstd.ZstdDecompressor()

        logger.debug(
            "zstd_compressor_initialized",
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
        Compress data using Zstandard.

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
            compressed = self._compressor.compress(data)
            compressed_size = len(compressed)
            elapsed_ms = (time.perf_counter() - start) * 1000

            ratio = compressed_size / original_size if original_size > 0 else 0
            savings_pct = (1 - ratio) * 100

            logger.debug(
                "zstd_compress_complete",
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
                "zstd_compress_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(elapsed_ms, 2),
            )
            raise

    def decompress(self, data: bytes) -> bytes:
        """
        Decompress Zstandard data.

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
            decompressed = self._decompressor.decompress(data)
            decompressed_size = len(decompressed)
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.debug(
                "zstd_decompress_complete",
                compressed_bytes=compressed_size,
                decompressed_bytes=decompressed_size,
                latency_ms=round(elapsed_ms, 2),
            )

            return decompressed

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "zstd_decompress_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(elapsed_ms, 2),
            )
            raise
