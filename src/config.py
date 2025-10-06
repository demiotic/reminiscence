"""Cache configuration."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheConfig:
    """
    Configuration for Memora semantic cache.

    Defaults are optimized for local development/testing.
    Use environment variables (MEMORA_*) for production.

    Environment variables:
    - MEMORA_MODEL_NAME: Embedding model
    - MEMORA_USE_ONNX: Use ONNX backend (true/false)
    - MEMORA_ONNX_MODEL_FILE: ONNX model file
    - MEMORA_SIMILARITY_THRESHOLD: Similarity threshold (0.0-1.0)
    - MEMORA_DB_URI: Database URI
    - MEMORA_TABLE_NAME: Table name
    - MEMORA_ENABLE_METRICS: Enable metrics (true/false)
    - MEMORA_TTL_SECONDS: TTL in seconds (None = no expiration)
    - MEMORA_LOG_LEVEL: Log level (DEBUG/INFO/WARNING/ERROR)
    - MEMORA_JSON_LOGS: JSON logging (true/false)
    - MEMORA_AUTO_CREATE_INDEX: Auto-create index (true/false)
    - MEMORA_INDEX_THRESHOLD_ENTRIES: Min entries for index
    - MEMORA_INDEX_NUM_PARTITIONS: IVF partitions
    - MEMORA_MAX_ENTRIES: Max cache entries
    - MEMORA_EVICTION_POLICY: Eviction policy (fifo/lru/lfu)
    - MEMORA_CLEANUP_INTERVAL_SECONDS: Interval for background cleanup (None = disabled)
    - MEMORA_CLEANUP_INITIAL_DELAY: Initial delay before first cleanup
    """

    # Model - portable defaults for local testing
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    use_onnx: bool = True
    onnx_model_file: str = "model.onnx"

    # Cache - in-memory by default for local dev
    similarity_threshold: float = 0.85
    db_uri: str = "memory://"
    table_name: str = "semantic_cache"
    enable_metrics: bool = True
    ttl_seconds: Optional[int] = None

    # Logging - human-readable for local dev
    log_level: str = "INFO"
    json_logs: bool = False

    # Cleanup
    cleanup_threshold: float = 0.3

    # Index - manual by default (no auto-indexing for small local tests)
    auto_create_index: bool = False
    index_threshold_entries: int = 256
    index_num_partitions: int = 256

    # Limits - small for local testing
    max_entries: Optional[int] = 1_000
    eviction_policy: str = "fifo"

    # Scheduler - disabled by default
    cleanup_interval_seconds: Optional[int] = None  # None = no auto cleanup
    cleanup_initial_delay: int = 60  # seconds

    @classmethod
    def load(cls) -> "CacheConfig":
        """
        Load configuration from environment variables.

        Falls back to development defaults if not set.

        Returns:
            CacheConfig instance
        """

        defaults = cls()

        def parse_bool(value: str) -> bool:
            """Parse boolean from string."""
            return value.lower() in ("true", "1", "yes", "on")

        def parse_int_or_none(value: str) -> Optional[int]:
            """Parse optional int from string."""
            return None if value.lower() == "none" else int(value)

        return cls(
            # Model
            model_name=os.getenv("MEMORA_MODEL_NAME", defaults.model_name),
            use_onnx=parse_bool(
                os.getenv("MEMORA_USE_ONNX", str(defaults.use_onnx).lower())
            ),
            onnx_model_file=os.getenv(
                "MEMORA_ONNX_MODEL_FILE", defaults.onnx_model_file
            ),
            # Cache
            similarity_threshold=float(
                os.getenv(
                    "MEMORA_SIMILARITY_THRESHOLD", str(defaults.similarity_threshold)
                )
            ),
            db_uri=os.getenv("MEMORA_DB_URI", defaults.db_uri),
            table_name=os.getenv("MEMORA_TABLE_NAME", defaults.table_name),
            enable_metrics=parse_bool(
                os.getenv("MEMORA_ENABLE_METRICS", str(defaults.enable_metrics).lower())
            ),
            ttl_seconds=parse_int_or_none(
                os.getenv(
                    "MEMORA_TTL_SECONDS",
                    str(defaults.ttl_seconds) if defaults.ttl_seconds else "none",
                )
            ),
            # Logging
            log_level=os.getenv("MEMORA_LOG_LEVEL", defaults.log_level).upper(),
            json_logs=parse_bool(
                os.getenv("MEMORA_JSON_LOGS", str(defaults.json_logs).lower())
            ),
            # Index
            auto_create_index=parse_bool(
                os.getenv(
                    "MEMORA_AUTO_CREATE_INDEX", str(defaults.auto_create_index).lower()
                )
            ),
            index_threshold_entries=int(
                os.getenv(
                    "MEMORA_INDEX_THRESHOLD_ENTRIES",
                    str(defaults.index_threshold_entries),
                )
            ),
            index_num_partitions=int(
                os.getenv(
                    "MEMORA_INDEX_NUM_PARTITIONS", str(defaults.index_num_partitions)
                )
            ),
            # Limits
            max_entries=parse_int_or_none(
                os.getenv(
                    "MEMORA_MAX_ENTRIES",
                    str(defaults.max_entries) if defaults.max_entries else "none",
                )
            ),
            eviction_policy=os.getenv(
                "MEMORA_EVICTION_POLICY", defaults.eviction_policy
            ).lower(),
            # Scheduler
            cleanup_interval_seconds=parse_int_or_none(
                os.getenv(
                    "MEMORA_CLEANUP_INTERVAL_SECONDS",
                    str(defaults.cleanup_interval_seconds)
                    if defaults.cleanup_interval_seconds
                    else "none",
                )
            ),
            cleanup_initial_delay=int(
                os.getenv(
                    "MEMORA_CLEANUP_INITIAL_DELAY",
                    str(defaults.cleanup_initial_delay),
                )
            ),
        )
