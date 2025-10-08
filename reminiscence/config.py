"""Cache configuration."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReminiscenceConfig:
    """
    Configuration for Reminiscence semantic cache.

    Environment variables:
    - REMINISCENCE_MODEL_NAME: Embedding model (optional, uses backend default)
    - REMINISCENCE_EMBEDDING_BACKEND: Backend (fastembed or auto)
    - REMINISCENCE_SIMILARITY_THRESHOLD: Similarity threshold (0.0-1.0)
    - REMINISCENCE_DB_URI: Database URI
    - REMINISCENCE_TABLE_NAME: Table name
    - REMINISCENCE_ENABLE_METRICS: Enable metrics (true/false)
    - REMINISCENCE_TTL_SECONDS: TTL in seconds (None = no expiration)
    - REMINISCENCE_LOG_LEVEL: Log level (DEBUG/INFO/WARNING/ERROR)
    - REMINISCENCE_JSON_LOGS: JSON logging (true/false)
    - REMINISCENCE_AUTO_CREATE_INDEX: Auto-create index (true/false)
    - REMINISCENCE_INDEX_THRESHOLD_ENTRIES: Min entries for index
    - REMINISCENCE_INDEX_NUM_PARTITIONS: IVF partitions
    - REMINISCENCE_MAX_ENTRIES: Max cache entries
    - REMINISCENCE_EVICTION_POLICY: Eviction policy (fifo/lru/lfu)
    - REMINISCENCE_CLEANUP_INTERVAL_SECONDS: Interval for background cleanup
    - REMINISCENCE_OTEL_ENABLED: Enable OpenTelemetry export (true/false)
    - REMINISCENCE_OTEL_ENDPOINT: OTLP endpoint URL
    - REMINISCENCE_OTEL_HEADERS: OTLP headers (key1=value1,key2=value2)
    - REMINISCENCE_OTEL_SERVICE_NAME: Service name for telemetry
    - REMINISCENCE_OTEL_EXPORT_INTERVAL_MS: Export interval in milliseconds
    """

    # Model
    model_name: Optional[str] = None  # None = use backend default
    embedding_backend: str = "fastembed"  # Only fastembed supported

    # Cache
    similarity_threshold: float = 0.80  # Adjusted for FastEmbed mean pooling
    db_uri: str = "memory://"
    table_name: str = "semantic_cache"
    enable_metrics: bool = True
    ttl_seconds: Optional[int] = None

    # Logging
    log_level: str = "INFO"
    json_logs: bool = False

    # Cleanup
    cleanup_threshold: float = 0.3

    # Index
    auto_create_index: bool = False
    index_threshold_entries: int = 256
    index_num_partitions: int = 256

    # Limits
    max_entries: Optional[int] = 1_000
    eviction_policy: str = "fifo"

    # Scheduler
    cleanup_interval_seconds: Optional[int] = None
    cleanup_initial_delay: int = 60

    # OpenTelemetry
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4318/v1/metrics"
    otel_headers: Optional[str] = None  # Format: "key1=value1,key2=value2"
    otel_service_name: str = "reminiscence"
    otel_export_interval_ms: int = 60000  # Export every 60 seconds by default

    @classmethod
    def load(cls) -> "ReminiscenceConfig":
        """Load configuration from environment variables."""
        defaults = cls()

        def parse_bool(value: str) -> bool:
            """Parse boolean from string."""
            return value.lower() in ("true", "1", "yes", "on")

        def parse_int_or_none(value: str) -> Optional[int]:
            """Parse optional int from string."""
            return None if value.lower() == "none" else int(value)

        def parse_str_or_none(value: str) -> Optional[str]:
            """Parse optional string from string."""
            return None if value.lower() in ("none", "") else value

        return cls(
            # Model
            model_name=parse_str_or_none(os.getenv("REMINISCENCE_MODEL_NAME", "none")),
            embedding_backend=os.getenv(
                "REMINISCENCE_EMBEDDING_BACKEND", defaults.embedding_backend
            ),
            # Cache
            similarity_threshold=float(
                os.getenv(
                    "REMINISCENCE_SIMILARITY_THRESHOLD",
                    str(defaults.similarity_threshold),
                )
            ),
            db_uri=os.getenv("REMINISCENCE_DB_URI", defaults.db_uri),
            table_name=os.getenv("REMINISCENCE_TABLE_NAME", defaults.table_name),
            enable_metrics=parse_bool(
                os.getenv(
                    "REMINISCENCE_ENABLE_METRICS", str(defaults.enable_metrics).lower()
                )
            ),
            ttl_seconds=parse_int_or_none(
                os.getenv(
                    "REMINISCENCE_TTL_SECONDS",
                    str(defaults.ttl_seconds) if defaults.ttl_seconds else "none",
                )
            ),
            # Logging
            log_level=os.getenv("REMINISCENCE_LOG_LEVEL", defaults.log_level).upper(),
            json_logs=parse_bool(
                os.getenv("REMINISCENCE_JSON_LOGS", str(defaults.json_logs).lower())
            ),
            # Index
            auto_create_index=parse_bool(
                os.getenv(
                    "REMINISCENCE_AUTO_CREATE_INDEX",
                    str(defaults.auto_create_index).lower(),
                )
            ),
            index_threshold_entries=int(
                os.getenv(
                    "REMINISCENCE_INDEX_THRESHOLD_ENTRIES",
                    str(defaults.index_threshold_entries),
                )
            ),
            index_num_partitions=int(
                os.getenv(
                    "REMINISCENCE_INDEX_NUM_PARTITIONS",
                    str(defaults.index_num_partitions),
                )
            ),
            # Limits
            max_entries=parse_int_or_none(
                os.getenv(
                    "REMINISCENCE_MAX_ENTRIES",
                    str(defaults.max_entries) if defaults.max_entries else "none",
                )
            ),
            eviction_policy=os.getenv(
                "REMINISCENCE_EVICTION_POLICY", defaults.eviction_policy
            ).lower(),
            # Scheduler
            cleanup_interval_seconds=parse_int_or_none(
                os.getenv(
                    "REMINISCENCE_CLEANUP_INTERVAL_SECONDS",
                    str(defaults.cleanup_interval_seconds)
                    if defaults.cleanup_interval_seconds
                    else "none",
                )
            ),
            cleanup_initial_delay=int(
                os.getenv(
                    "REMINISCENCE_CLEANUP_INITIAL_DELAY",
                    str(defaults.cleanup_initial_delay),
                )
            ),
            # OpenTelemetry
            otel_enabled=parse_bool(
                os.getenv(
                    "REMINISCENCE_OTEL_ENABLED",
                    str(defaults.otel_enabled).lower(),
                )
            ),
            otel_endpoint=os.getenv(
                "REMINISCENCE_OTEL_ENDPOINT",
                defaults.otel_endpoint,
            ),
            otel_headers=parse_str_or_none(
                os.getenv("REMINISCENCE_OTEL_HEADERS", "none")
            ),
            otel_service_name=os.getenv(
                "REMINISCENCE_OTEL_SERVICE_NAME",
                defaults.otel_service_name,
            ),
            otel_export_interval_ms=int(
                os.getenv(
                    "REMINISCENCE_OTEL_EXPORT_INTERVAL_MS",
                    str(defaults.otel_export_interval_ms),
                )
            ),
        )
