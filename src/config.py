"""Cache configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheConfig:
    """
    Configuration for SemanticCache.

    Attributes:
        model_name: Sentence-transformers model
            - 'paraphrase-multilingual-MiniLM-L12-v2' (default): 384 dims, 50+ languages
            - 'all-MiniLM-L6-v2': 384 dims, English only (faster)
            - 'paraphrase-multilingual-mpnet-base-v2': 768 dims, better quality
        similarity_threshold: Cosine similarity threshold
            - 0.75-0.80 for multilingual models (recommended)
            - 0.85-0.90 for monolingual models
        db_uri: LanceDB URI
            - 'memory://' for in-memory (doesn't persist)
            - './cache.db' for disk persistence
        table_name: Table name in LanceDB
        enable_metrics: If True, collect performance metrics
        ttl_seconds: Time-to-live in seconds (None = no expiration)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        cleanup_threshold: Threshold for automatic cleanup
        auto_create_index: If True, create index automatically
        index_threshold_entries: Minimum entries before creating index
        index_num_partitions: Number of IVF partitions for index
        max_entries: Maximum total entries (triggers eviction)
        max_result_size_bytes: Maximum size per payload (rejects larger)
        eviction_policy: Eviction strategy when max_entries reached
            - 'fifo': First In First Out (remove oldest)
            - 'lru': Least Recently Used (future)
    """

    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.85
    db_uri: str = "memory://"
    table_name: str = "semantic_cache"
    enable_metrics: bool = True
    ttl_seconds: Optional[int] = None
    log_level: str = "INFO"
    cleanup_threshold: float = 0.3

    # Vector index configuration
    auto_create_index: bool = False
    index_threshold_entries: int = 256
    index_num_partitions: int = 256

    # NEW: Size limits and eviction
    max_entries: Optional[int] = 10_000  # None = unlimited
    max_result_size_bytes: int = 10_000_000  # 10MB default
    eviction_policy: str = "fifo"  # "fifo" or "lru" (future)

    @classmethod
    def for_production(cls, db_path: str = "./cache.db") -> "CacheConfig":
        """
        Optimized configuration for production.

        Args:
            db_path: Path for cache persistence

        Returns:
            CacheConfig configured for production
        """
        return cls(
            db_uri=db_path,
            ttl_seconds=86400,  # 24 hours
            enable_metrics=True,
            log_level="INFO",
            auto_create_index=True,
            index_threshold_entries=1000,
            index_num_partitions=512,
            max_entries=50_000,  # 50k entries max in production
            max_result_size_bytes=10_000_000,  # 10MB per result
            eviction_policy="fifo",
        )

    @classmethod
    def for_development(cls) -> "CacheConfig":
        """
        Configuration for development/testing.

        Returns:
            CacheConfig configured for development
        """
        return cls(
            db_uri="memory://",
            enable_metrics=True,
            log_level="DEBUG",
            auto_create_index=False,  # Manual in dev
            max_entries=1_000,  # Lower limit in dev
            max_result_size_bytes=5_000_000,  # 5MB in dev
        )
