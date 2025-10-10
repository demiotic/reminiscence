"""Data types for Reminiscence."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import pyarrow as pa
import re


@dataclass
class CacheEntry:
    """
    Individual cache entry.

    Represents a stored result with its associated metadata.
    """

    query_text: str
    context: Dict[str, Any]
    embedding: pa.Array
    result: Any
    timestamp: int
    similarity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
    context_threshold: Optional[float] = None

    @property
    def age_seconds(self) -> float:
        """Calculate entry age in seconds."""
        import time

        return time.time() - self.timestamp

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on its specific TTL."""
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds

    @property
    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL in seconds."""
        if self.ttl_seconds is None:
            return None
        remaining = self.ttl_seconds - self.age_seconds
        return max(0.0, remaining)


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
        entry_id: ID of matched entry (for debugging)
        context: Context of matched entry (for debugging)
        ttl_remaining: Remaining TTL for this entry
    """

    hit: bool
    result: Optional[Any] = None
    similarity: Optional[float] = None
    matched_query: Optional[str] = None
    age_seconds: Optional[float] = None
    entry_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    ttl_remaining: Optional[float] = None

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
    age_seconds: Optional[float] = None
    ttl_remaining_seconds: Optional[float] = None
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
    """Request to store in cache (used in remote mode)."""

    query: str
    context: Dict[str, Any]
    result: Any
    metadata: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
    context_threshold: Optional[float] = None


@dataclass
class LookupRequest:
    """Lookup request (used in remote mode)."""

    query: str
    context: Optional[Dict[str, Any]] = None
    similarity_threshold: Optional[float] = None


@dataclass
class InvalidateRequest:
    """Invalidation request with pattern support."""

    query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    older_than_seconds: Optional[float] = None
    query_pattern: Optional[str] = None
    context_pattern: Optional[Dict[str, str]] = None


@dataclass
class BulkInvalidatePattern:
    """
    Pattern-based bulk invalidation specification.

    Examples:
        pattern = BulkInvalidatePattern(query_regex="^SELECT.*")

        pattern = BulkInvalidatePattern(
            context_matches={"model": "gpt-4", "agent_*": "*"}
        )

        pattern = BulkInvalidatePattern(
            query_prefix="translate",
            older_than_seconds=3600
        )
    """

    query_regex: Optional[str] = None
    query_prefix: Optional[str] = None
    query_suffix: Optional[str] = None
    context_matches: Optional[Dict[str, str]] = None
    older_than_seconds: Optional[float] = None
    similarity_below: Optional[float] = None
    entry_ids: Optional[List[str]] = None

    def matches_query(self, query: str) -> bool:
        """Check if query matches pattern."""
        if self.query_regex:
            return bool(re.match(self.query_regex, query))
        if self.query_prefix:
            return query.startswith(self.query_prefix)
        if self.query_suffix:
            return query.endswith(self.query_suffix)
        return True

    def matches_context(self, context: Dict[str, Any]) -> bool:
        """Check if context matches pattern with wildcard support."""
        if not self.context_matches:
            return True

        for key_pattern, value_pattern in self.context_matches.items():
            matched_key = False

            for ctx_key, ctx_value in context.items():
                if self._match_wildcard(key_pattern, ctx_key):
                    matched_key = True
                    if value_pattern == "*":
                        continue
                    if not self._match_wildcard(str(value_pattern), str(ctx_value)):
                        return False

            if not matched_key and "*" not in key_pattern:
                return False

        return True

    def matches_age(self, age_seconds: float) -> bool:
        """Check if entry age matches pattern."""
        if self.older_than_seconds is None:
            return True
        return age_seconds > self.older_than_seconds

    def matches_similarity(self, similarity: Optional[float]) -> bool:
        """Check if similarity matches pattern."""
        if self.similarity_below is None:
            return True
        if similarity is None:
            return False
        return similarity < self.similarity_below

    def matches_entry_id(self, entry_id: str) -> bool:
        """Check if entry ID is in the list."""
        if self.entry_ids is None:
            return True
        return entry_id in self.entry_ids

    @staticmethod
    def _match_wildcard(pattern: str, text: str) -> bool:
        """Match with wildcard support (* = any chars)."""
        regex_pattern = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
        return bool(re.match(regex_pattern, text))
