"""Factory for creating compression backends."""

from typing import Optional
from .base import Compressor
from .zstd import ZstdCompressor
from .gzip import GzipCompressor


def create_compressor(
    algorithm: str = "zstd",
    level: Optional[int] = None,
) -> Optional[Compressor]:
    """
    Create compressor backend.

    Args:
        algorithm: Algorithm name ("zstd", "gzip", "none")
        level: Compression level (algorithm-specific, optional)

    Returns:
        Compressor instance or None if algorithm is "none"

    Raises:
        ValueError: If algorithm not supported
    """
    algorithm = algorithm.lower()

    if algorithm == "none":
        return None

    elif algorithm == "zstd":
        return ZstdCompressor(level=level if level is not None else 3)

    elif algorithm == "gzip":
        return GzipCompressor(level=level if level is not None else 6)

    else:
        raise ValueError(
            f"Unsupported compression algorithm: {algorithm}. "
            f"Supported: zstd, gzip, none"
        )
