"""Compression backends for Reminiscence."""

from __future__ import annotations

from .base import Compressor
from .factory import create_compressor
from .gzip import GzipCompressor
from .zstd import ZstdCompressor

__all__ = [
    "Compressor",
    "create_compressor",
    "ZstdCompressor",
    "GzipCompressor",
]
