"""Data types for Reminiscence semantic cache.

This module defines all core types used throughout the Reminiscence API,
including multimodal input handling, cache entries, and request/response types.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pyarrow as pa

# =============================================================================
# Enums for Type Safety
# =============================================================================


class QueryMode(str, Enum):
    """Query matching strategies for cache lookups.

    Attributes:
        SEMANTIC: Semantic similarity search with configurable threshold.
        EXACT: Exact text matching only (no semantic similarity).
        AUTO: Try exact match first, fallback to semantic if no match.
    """

    SEMANTIC = "semantic"
    EXACT = "exact"
    AUTO = "auto"


class EvictionPolicy(str, Enum):
    """Cache eviction strategies.

    Attributes:
        LRU: Least Recently Used - evicts entries not accessed recently.
        LFU: Least Frequently Used - evicts entries accessed least often.
        FIFO: First In First Out - evicts oldest entries.
    """

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"


class CompressionBackend(str, Enum):
    """Compression backends for cache storage.

    Attributes:
        GZIP: Standard gzip compression.
        ZSTD: Zstandard compression (faster, better ratio).
    """

    GZIP = "gzip"
    ZSTD = "zstd"


class EncryptionBackend(str, Enum):
    """Encryption backends for sensitive data.

    Attributes:
        AGE: Age encryption (https://age-encryption.org/).
    """

    AGE = "age"


# =============================================================================
# Multimodal Input
# =============================================================================


@dataclass(frozen=True)
class MultiModalInput:
    """Unified multimodal input for all cache operations.

    Supports text, image, video, and audio inputs either individually
    or in combination. All queries flow through this type - text-only
    is just a special case with only the text field populated.

    This type is immutable (frozen) to ensure it can be safely used
    as a dictionary key or in sets for deduplication.

    Examples:
        Text-only query (most common case):
        >>> input = MultiModalInput(text="What is quantum computing?")

        Image with text prompt:
        >>> input = MultiModalInput(
        ...     text="What's in this image?",
        ...     image=image_bytes
        ... )

        Video analysis:
        >>> input = MultiModalInput(
        ...     text="Summarize this video",
        ...     video="https://example.com/video.mp4"
        ... )

        Multi-modal combination:
        >>> input = MultiModalInput(
        ...     text="Describe the audio and visual content",
        ...     video=video_bytes,
        ...     audio=audio_url
        ... )

    Attributes:
        text: Text content or query string.
        image: Image data as bytes, base64 string, or URL.
        video: Video data as bytes, base64 string, or URL.
        audio: Audio data as bytes, base64 string, or URL.
        metadata: Additional metadata about the input (not used for hashing).
    """

    text: Optional[str] = None
    image: Optional[bytes | str] = None
    video: Optional[bytes | str] = None
    audio: Optional[bytes | str] = None
    metadata: Optional[Dict[str, Any]] = field(default=None, hash=False, compare=False)

    def __post_init__(self) -> None:
        """Validate that at least one modality is present."""
        if not any([self.text, self.image, self.video, self.audio]):
            raise ValueError(
                "MultiModalInput requires at least one modality "
                "(text, image, video, or audio)"
            )

    def __str__(self) -> str:
        """Human-readable representation of the input.

        Returns:
            String describing the modalities present.
        """
        modalities = []
        if self.text:
            preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
            modalities.append(f"text='{preview}'")
        if self.image:
            img_repr = (
                "image=<bytes>"
                if isinstance(self.image, bytes)
                else f"image={str(self.image)[:30]}..."
            )
            modalities.append(img_repr)
        if self.video:
            modalities.append("video=<data>")
        if self.audio:
            modalities.append("audio=<data>")

        return f"MultiModalInput({', '.join(modalities)})"

    def is_text_only(self) -> bool:
        """Check if input contains only text.

        Used for fast path optimization - text-only queries can use
        lightweight embedding models.

        Returns:
            True if only text is present, False otherwise.
        """
        return self.text is not None and not self.has_media()

    def has_media(self) -> bool:
        """Check if input contains any media (image/video/audio).

        Returns:
            True if any media modality is present, False otherwise.
        """
        return any([self.image, self.video, self.audio])

    def primary_modality(self) -> str:
        """Get primary modality for logging and metrics.

        Returns the "heaviest" modality present, in order of priority:
        video > image > audio > text

        Returns:
            String identifier of primary modality.
        """
        if self.video:
            return "video"
        if self.image:
            return "image"
        if self.audio:
            return "audio"
        return "text"

    def modality_count(self) -> int:
        """Count how many modalities are present in this input.

        Returns:
            Number of non-None modalities (1-4).
        """
        return sum(
            [
                self.text is not None,
                self.image is not None,
                self.video is not None,
                self.audio is not None,
            ]
        )


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class CacheEntry:
    """Individual cache entry with multimodal query support.

    Represents a stored result with its associated metadata and
    multimodal query information.

    Attributes:
        query: The multimodal query that generated this result.
        context: Contextual parameters used for matching.
        embedding: Vector embedding of the query for similarity search.
        result: The cached result (can be any Python object).
        timestamp: Unix timestamp in milliseconds when entry was created.
        similarity: Similarity score if retrieved via semantic search.
        metadata: Additional metadata about the entry.
        ttl_seconds: Time-to-live for this specific entry.
        context_threshold: Context matching threshold for this entry.
    """

    query: MultiModalInput
    context: Dict[str, Any]
    embedding: pa.Array
    result: Any
    timestamp: int  # Milliseconds since epoch
    similarity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
    context_threshold: Optional[float] = None

    @property
    def query_text(self) -> str:
        """Get text representation of the query.

        For text-only queries, returns the text content.
        For multimodal queries, returns a descriptive string.

        Returns:
            Text representation of the query.
        """
        if self.query.text:
            return self.query.text
        return f"<{self.query.primary_modality()} query>"

    @property
    def age_seconds(self) -> float:
        """Calculate entry age in seconds since creation.

        Returns:
            Age of the entry in seconds.
        """
        current_ms = int(time.time() * 1000)
        age_ms = current_ms - self.timestamp
        return age_ms / 1000.0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on its TTL.

        Returns:
            True if entry has exceeded its TTL, False otherwise.
        """
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds

    @property
    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL in seconds.

        Returns:
            Remaining seconds until expiration, or None if no TTL set.
        """
        if self.ttl_seconds is None:
            return None
        remaining = self.ttl_seconds - self.age_seconds
        return max(0.0, remaining)

    def __repr__(self) -> str:
        """Developer representation of cache entry.

        Returns:
            Detailed string with all fields.
        """
        return (
            f"CacheEntry("
            f"query={self.query_text!r}, "
            f"age={self.age_seconds:.1f}s, "
            f"similarity={self.similarity}, "
            f"expired={self.is_expired}"
            ")"
        )


@dataclass
class LookupResult:
    """Result of a cache lookup operation.

    Attributes:
        hit: True if a valid match was found, False for cache miss.
        result: Retrieved data (None if miss).
        similarity: Similarity score (0-1) for semantic matches.
        matched_query: Text of the original query that matched.
        age_seconds: Age of the matched entry in seconds.
        entry_id: Unique ID of matched entry (for debugging).
        context: Context of matched entry (for debugging).
        ttl_remaining: Remaining TTL for this entry in seconds.
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
        """Check if lookup was a cache hit.

        Alias for the `hit` attribute for compatibility.

        Returns:
            True if cache hit, False if miss.
        """
        return self.hit

    @property
    def is_miss(self) -> bool:
        """Check if lookup was a cache miss.

        Returns:
            True if cache miss, False if hit.
        """
        return not self.hit

    def __repr__(self) -> str:
        """Developer representation of lookup result.

        Returns:
            String showing hit status and key details.
        """
        if self.hit:
            return (
                f"LookupResult(hit=True, "
                f"similarity={self.similarity:.3f}, "
                f"age={self.age_seconds:.1f}s)"
            )
        return "LookupResult(hit=False)"

    def __str__(self) -> str:
        """Human-readable lookup result.

        Returns:
            Friendly description of the result.
        """
        return "Cache HIT" if self.hit else "Cache MISS"

    def __bool__(self) -> bool:
        """Allow boolean evaluation (if result: ...).

        Returns:
            True if cache hit, False if miss.
        """
        return self.hit


@dataclass
class AvailabilityCheck:
    """Result of a cache availability check.

    Used by schedulers to determine if cache exists without
    retrieving the actual data.

    Attributes:
        available: True if matching cache entry exists.
        age_seconds: Age of the entry in seconds if available.
        ttl_remaining_seconds: Remaining TTL in seconds if available.
        similarity: Similarity score of the match if applicable.
    """

    available: bool
    age_seconds: Optional[float] = None
    ttl_remaining_seconds: Optional[float] = None
    similarity: Optional[float] = None

    @property
    def is_fresh(self) -> bool:
        """Check if cached entry is fresh (< 50% of TTL consumed).

        Returns:
            True if entry is in first half of its lifetime, False otherwise.
        """
        if self.ttl_remaining_seconds is None or self.age_seconds is None:
            return True
        total_ttl = self.age_seconds + self.ttl_remaining_seconds
        return self.age_seconds < (total_ttl * 0.5)


# =============================================================================
# Request/Response Types (for batch operations & remote mode)
# =============================================================================


@dataclass
class StoreRequest:
    """Request to store data in cache.

    Used in batch operations and remote mode for storing cache entries.

    Attributes:
        query: The multimodal query to cache.
        context: Contextual parameters for matching.
        result: The result to cache (can be any Python object).
        metadata: Additional metadata to store with the entry.
        ttl_seconds: Time-to-live for this entry (overrides global TTL).
        context_threshold: Context matching threshold for this entry.
    """

    query: MultiModalInput
    context: Dict[str, Any]
    result: Any
    metadata: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
    context_threshold: Optional[float] = None


@dataclass
class LookupRequest:
    """Request to lookup data in cache.

    Used in batch operations and remote mode for cache lookups.

    Attributes:
        query: The multimodal query to look up.
        context: Contextual parameters for matching.
        similarity_threshold: Minimum similarity score for semantic matches.
        mode: Query matching strategy to use.
    """

    query: MultiModalInput
    context: Optional[Dict[str, Any]] = None
    similarity_threshold: Optional[float] = None
    mode: QueryMode = QueryMode.AUTO


@dataclass
class InvalidateRequest:
    """Request to invalidate cache entries.

    Supports pattern-based invalidation with multiple criteria.

    Attributes:
        query: Specific query to invalidate (exact match).
        context: Context to match for invalidation.
        older_than_seconds: Invalidate entries older than this age.
        query_pattern: Regex pattern for query text matching.
        context_pattern: Pattern dict for context field matching.
    """

    query: Optional[MultiModalInput] = None
    context: Optional[Dict[str, Any]] = None
    older_than_seconds: Optional[float] = None
    query_pattern: Optional[str] = None
    context_pattern: Optional[Dict[str, str]] = None


# =============================================================================
# Pattern Matching for Bulk Operations
# =============================================================================


@dataclass
class BulkInvalidatePattern:
    """Pattern-based bulk invalidation specification.

    Supports complex invalidation patterns including regex matching,
    prefix/suffix matching, context matching with wildcards, and
    age-based invalidation.

    Examples:
        Invalidate all SQL queries:
        >>> pattern = BulkInvalidatePattern(query_regex="^SELECT.*")

        Invalidate by context with wildcard:
        >>> pattern = BulkInvalidatePattern(
        ...     context_matches={"model": "gpt-4", "agent_*": "*"}
        ... )

        Invalidate old translation entries:
        >>> pattern = BulkInvalidatePattern(
        ...     query_prefix="translate",
        ...     older_than_seconds=3600
        ... )

    Attributes:
        query_regex: Regex pattern to match query text.
        query_prefix: Prefix to match in query text.
        query_suffix: Suffix to match in query text.
        context_matches: Dict of context key-value patterns (* wildcard).
        older_than_seconds: Invalidate entries older than this age.
        similarity_below: Invalidate entries with similarity below this.
        entry_ids: Specific entry IDs to invalidate.
    """

    query_regex: Optional[str] = None
    query_prefix: Optional[str] = None
    query_suffix: Optional[str] = None
    context_matches: Optional[Dict[str, str]] = None
    older_than_seconds: Optional[float] = None
    similarity_below: Optional[float] = None
    entry_ids: Optional[List[str]] = None

    def matches_query(self, query: MultiModalInput) -> bool:
        """Check if query matches pattern (uses text representation).

        Args:
            query: The multimodal query to check.

        Returns:
            True if query matches any defined pattern, False otherwise.
        """
        query_text = query.text or f"<{query.primary_modality()}>"

        if self.query_regex:
            return bool(re.match(self.query_regex, query_text))
        if self.query_prefix:
            return query_text.startswith(self.query_prefix)
        if self.query_suffix:
            return query_text.endswith(self.query_suffix)
        return True

    def matches_context(self, context: Dict[str, Any]) -> bool:
        """Check if context matches pattern with wildcard support.

        Supports '*' wildcard in both keys and values.

        Args:
            context: The context dict to check.

        Returns:
            True if context matches pattern, False otherwise.
        """
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
        """Check if entry age matches pattern.

        Args:
            age_seconds: Age of the entry in seconds.

        Returns:
            True if age matches criteria, False otherwise.
        """
        if self.older_than_seconds is None:
            return True
        return age_seconds > self.older_than_seconds

    def matches_similarity(self, similarity: Optional[float]) -> bool:
        """Check if similarity score matches pattern.

        Args:
            similarity: Similarity score to check (0-1).

        Returns:
            True if similarity matches criteria, False otherwise.
        """
        if self.similarity_below is None:
            return True
        if similarity is None:
            return False
        return similarity < self.similarity_below

    def matches_entry_id(self, entry_id: str) -> bool:
        """Check if entry ID is in the explicit list.

        Args:
            entry_id: The entry ID to check.

        Returns:
            True if entry_id is in the list or no list specified.
        """
        if self.entry_ids is None:
            return True
        return entry_id in self.entry_ids

    @staticmethod
    def _match_wildcard(pattern: str, text: str) -> bool:
        """Match text against pattern with wildcard support.

        Converts '*' wildcards to regex patterns.

        Args:
            pattern: Pattern string with optional '*' wildcards.
            text: Text to match against pattern.

        Returns:
            True if text matches pattern, False otherwise.
        """
        regex_pattern = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
        return bool(re.match(regex_pattern, text))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "QueryMode",
    "EvictionPolicy",
    "CompressionBackend",
    "EncryptionBackend",
    # Multimodal
    "MultiModalInput",
    # Core types
    "CacheEntry",
    "LookupResult",
    "AvailabilityCheck",
    # Request/Response
    "StoreRequest",
    "LookupRequest",
    "InvalidateRequest",
    # Patterns
    "BulkInvalidatePattern",
]
