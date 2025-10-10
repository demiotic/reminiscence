"""Compression backends for Reminiscence."""

from .base import Compressor
from .factory import create_compressor
from .zstd import ZstdCompressor
from .gzip import GzipCompressor

__all__ = [
    "Compressor",
    "create_compressor",
    "ZstdCompressor",
    "GzipCompressor",
]
