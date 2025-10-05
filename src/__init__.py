"""
Memora: Caché semántica para sistemas multiagente.

Example:
    >>> from memora import Memora, CacheConfig
    >>>
    >>> # Modo básico
    >>> cache = Memora(CacheConfig.for_development())
    >>> result = cache.lookup("Analiza ventas Q3", context={"agent": "sql"})
    >>>
    >>> if not result.is_hit:
    ...     data = execute_agent(...)
    ...     cache.store("Analiza ventas Q3", {"agent": "sql"}, data)
"""

__version__ = "0.1.0"

from .core import Memora
from .config import CacheConfig
from .metrics import CacheMetrics
from .types import (
    CacheEntry,
    LookupResult,
    AvailabilityCheck,
    StoreRequest,
    LookupRequest,
    InvalidateRequest,
)
from .decorators import MemoraDecorator, create_cached_decorator

# Shortcuts para imports comunes
from .utils import (
    create_fingerprint,
    cosine_similarity,
    serialize,
    deserialize,
    content_hash,
)

__all__ = [
    # Core
    "Memora",
    "CacheConfig",
    "CacheMetrics",
    # Types
    "CacheEntry",
    "LookupResult",
    "AvailabilityCheck",
    "StoreRequest",
    "LookupRequest",
    "InvalidateRequest",
    # Decorators
    "MemoraDecorator",
    "create_cached_decorator",
    # Utils (re-exported for convenience)
    "create_fingerprint",
    "cosine_similarity",
    "serialize",
    "deserialize",
    "content_hash",
]
