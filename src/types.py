"""Data types for Memora."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pyarrow as pa


@dataclass
class CacheEntry:
    """
    Individual cache entry.

    Represents a stored result with its associated metadata.
    """

    query_text: str
    context_hash: str
    embedding: pa.Array  # FixedSizeListArray of float32
    result: Any
    timestamp: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LookupResult:
    """
    Result of a cache lookup operation.

    Attributes:
        hit: True if valid match was found
        result: Retrieved data (None if miss)
        similarity: Similarity score (0-1)
        matched_query: Original query that matched
        age_seconds: Entry age in seconds
    """

    hit: bool
    result: Optional[Any] = None
    similarity: Optional[float] = None
    matched_query: Optional[str] = None
    age_seconds: Optional[int] = None

    @property
    def is_hit(self) -> bool:
        """Alias for compatibility with different code styles."""
        return self.hit

    @property
    def is_miss(self) -> bool:
        """Inverse of is_hit."""
        return not self.hit


@dataclass
class AvailabilityCheck:
    """
    Availability check result.

    Used by schedulers to know if cache exists without retrieving data.
    """

    available: bool
    age_seconds: Optional[int] = None
    ttl_remaining_seconds: Optional[int] = None
    similarity: Optional[float] = None

    @property
    def is_fresh(self) -> bool:
        """Returns True if entry is recent (< 50% of TTL consumed)."""
        if self.ttl_remaining_seconds is None or self.age_seconds is None:
            return True
        total_ttl = self.age_seconds + self.ttl_remaining_seconds
        return self.age_seconds < (total_ttl * 0.5)


@dataclass
class StoreRequest:
    """
    Request to store in cache (used in remote mode).
    """

    query: str
    context: Dict[str, Any]
    result: Any
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LookupRequest:
    """
    Lookup request (used in remote mode).
    """

    query: str
    context: Optional[Dict[str, Any]] = None
    similarity_threshold: Optional[float] = None


@dataclass
class InvalidateRequest:
    """
    Invalidation request.
    """

    query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    older_than_seconds: Optional[int] = None
