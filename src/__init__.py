"""Memora - Caché semántica para sistemas multiagente."""

from .core import Memora
from .config import CacheConfig
from .decorators import MemoraDecorator, create_cached_decorator
from .types import (
    CacheEntry,
    LookupResult,
    AvailabilityCheck,
    StoreRequest,
    LookupRequest,
    InvalidateRequest,
)
from .metrics import CacheMetrics

__version__ = "0.2.0"

__all__ = [
    "Memora",
    "CacheConfig",
    "MemoraDecorator",
    "create_cached_decorator",
    "CacheEntry",
    "LookupResult",
    "AvailabilityCheck",
    "StoreRequest",
    "LookupRequest",
    "InvalidateRequest",
    "CacheMetrics",
]
