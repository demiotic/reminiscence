"""Cache configuration."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheConfig:
    """
    Configuration for Memora semantic cache.

    Attributes:
        model_name: Sentence-transformers model
            - 'paraphrase-multilingual-MiniLM-L12-v2' (default): 384 dims, 50+ languages
            - 'all-MiniLM-L6-v2': 384 dims, English only (faster)
            - 'paraphrase-multilingual-mpnet-base-v2': 768 dims, better quality
        similarity_threshold: Cosine similarity threshold
            - 0.75-0.80 for multilingual models
            - 0.85-0.90 for monolingual models (recommended)
        db_uri: LanceDB URI
            - 'memory://' for in-memory (doesn't persist)
            - './cache.db' for disk persistence
        table_name: Table name in LanceDB
        enable_metrics: If True, collect performance metrics
        ttl_seconds: Time-to-live in seconds (None = no expiration)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Enable JSON structured logging (recommended for production)
        cleanup_threshold: Threshold for automatic cleanup
        auto_create_index: If True, create index automatically
        index_threshold_entries: Minimum entries before creating index
        index_num_partitions: Number of IVF partitions for index
        max_entries: Maximum total entries (triggers eviction when reached)
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
    json_logs: bool = False
    cleanup_threshold: float = 0.3

    # Vector index configuration
    auto_create_index: bool = False
    index_threshold_entries: int = 256
    index_num_partitions: int = 256

    # Size limits and eviction
    max_entries: Optional[int] = 10_000  # None = unlimited
    max_result_size_bytes: int = 10_000_000  # 10MB default
    eviction_policy: str = "fifo"

    @classmethod
    def for_production(cls, db_path: str = "./memora_cache") -> "CacheConfig":
        """
        Production preset - optimized for high load.

        Features:
        - 50k entries capacity
        - 10MB result size limit
        - JSON structured logging (machine-readable)
        - 1 hour TTL
        - Auto-indexing enabled

        Args:
            db_path: Path for cache persistence

        Returns:
            CacheConfig configured for production

        Example:
            >>> config = CacheConfig.for_production()
            >>> memora = Memora(config)
        """
        return cls(
            db_uri=db_path,
            ttl_seconds=3600,
            enable_metrics=True,
            log_level="INFO",
            json_logs=True,
            auto_create_index=True,
            index_threshold_entries=1000,
            index_num_partitions=512,
            max_entries=50_000,
            max_result_size_bytes=10_000_000,
            eviction_policy="fifo",
        )

    @classmethod
    def for_development(cls) -> "CacheConfig":
        """
        Development preset - optimized for testing and debugging.

        Features:
        - 1k entries capacity (smaller for testing)
        - 5MB result size limit
        - Human-readable console logs
        - DEBUG level logging
        - Manual indexing (no auto-indexing)

        Returns:
            CacheConfig configured for development

        Example:
            >>> config = CacheConfig.for_development()
            >>> memora = Memora(config)
        """
        return cls(
            db_uri="memory://",
            ttl_seconds=300,
            enable_metrics=True,
            log_level="DEBUG",
            json_logs=False,
            auto_create_index=False,
            max_entries=1_000,
            max_result_size_bytes=5_000_000,
            eviction_policy="fifo",
        )

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """
        Create configuration from environment variables.

        Supported environment variables:
        - MEMORA_MODEL_NAME: Embedding model name
        - MEMORA_SIMILARITY_THRESHOLD: Similarity threshold (float)
        - MEMORA_DB_URI: Database URI (default: memory://)
        - MEMORA_TABLE_NAME: Table name (default: semantic_cache)
        - MEMORA_ENABLE_METRICS: Enable metrics (true/false, default: true)
        - MEMORA_TTL_SECONDS: TTL in seconds (int, None = no expiration)
        - MEMORA_LOG_LEVEL: Log level (DEBUG/INFO/WARNING/ERROR, default: INFO)
        - MEMORA_JSON_LOGS: Enable JSON logs (true/false, default: false)
        - MEMORA_MAX_ENTRIES: Max cache entries (int, None = unlimited)
        - MEMORA_MAX_RESULT_SIZE_BYTES: Max payload size (int, default: 10MB)
        - MEMORA_EVICTION_POLICY: Eviction policy (fifo/lru, default: fifo)
        - MEMORA_AUTO_CREATE_INDEX: Auto-create index (true/false, default: false)

        Returns:
            CacheConfig with values from environment

        Example:
            # In shell:
            export MEMORA_JSON_LOGS=true
            export MEMORA_LOG_LEVEL=WARNING
            export MEMORA_MAX_ENTRIES=100000

            # In Python:
            >>> config = CacheConfig.from_env()
            >>> memora = Memora(config)
        """

        def get_bool(key: str, default: bool) -> bool:
            """Parse boolean from env var."""
            value = os.getenv(key, str(default)).lower()
            return value in ("true", "1", "yes", "on")

        def get_int_or_none(key: str, default: Optional[int]) -> Optional[int]:
            """Parse optional int from env var."""
            value = os.getenv(key)
            if value is None:
                return default
            return int(value) if value.lower() != "none" else None

        def get_float(key: str, default: float) -> float:
            """Parse float from env var."""
            value = os.getenv(key)
            return float(value) if value else default

        return cls(
            model_name=os.getenv(
                "MEMORA_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            similarity_threshold=get_float("MEMORA_SIMILARITY_THRESHOLD", 0.85),
            db_uri=os.getenv("MEMORA_DB_URI", "memory://"),
            table_name=os.getenv("MEMORA_TABLE_NAME", "semantic_cache"),
            enable_metrics=get_bool("MEMORA_ENABLE_METRICS", True),
            ttl_seconds=get_int_or_none("MEMORA_TTL_SECONDS", None),
            log_level=os.getenv("MEMORA_LOG_LEVEL", "INFO").upper(),
            json_logs=get_bool("MEMORA_JSON_LOGS", False),
            auto_create_index=get_bool("MEMORA_AUTO_CREATE_INDEX", False),
            max_entries=get_int_or_none("MEMORA_MAX_ENTRIES", 10_000),
            max_result_size_bytes=get_int_or_none(
                "MEMORA_MAX_RESULT_SIZE_BYTES", 10_000_000
            ),
            eviction_policy=os.getenv("MEMORA_EVICTION_POLICY", "fifo").lower(),
        )
