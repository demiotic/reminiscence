"""Cache configuration."""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class ReminiscenceConfig:
    """
    Configuration for Reminiscence semantic cache.

    Environment variables:
    - REMINISCENCE_MODEL_NAME: Embedding model (optional, uses backend default)
    - REMINISCENCE_EMBEDDING_BACKEND: Backend (fastembed or auto)
    - REMINISCENCE_EMBEDDING_BATCH_SIZE: Batch size for embedding generation (1-512)
    - REMINISCENCE_SIMILARITY_THRESHOLD: Similarity threshold (0.0-1.0)
    - REMINISCENCE_CONTEXT_THRESHOLDS: JSON dict of context-specific thresholds
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
    - REMINISCENCE_ENCRYPTION_ENABLED: Enable encryption (true/false)
    - REMINISCENCE_ENCRYPTION_KEY: Encryption key string, ARN, URI, or file path
    - REMINISCENCE_ENCRYPTION_BACKEND: Backend (age/aws-kms/gcp-kms/azure-keyvault/vault)
    - REMINISCENCE_ENCRYPTION_MAX_WORKERS: Max threads for batch encryption
    - REMINISCENCE_WARM_UP_EMBEDDER: Pre-load embedder model (true/false, default: true)
    - REMINISCENCE_COMPRESSION_ENABLED: Enable result compression (true/false, default: false)
    - REMINISCENCE_COMPRESSION_ALGORITHM: Algorithm (zstd/gzip/none, default: zstd)
    - REMINISCENCE_COMPRESSION_LEVEL: Compression level (1-22 for zstd, 1-9 for gzip, default: 3)
    """

    model_name: Optional[str] = None
    embedding_backend: str = "fastembed"
    embedding_batch_size: int = 32
    warm_up_embedder: bool = True

    similarity_threshold: float = 0.80
    context_thresholds: Dict[str, float] = field(default_factory=dict)

    db_uri: str = "memory://"
    table_name: str = "semantic_cache"
    enable_metrics: bool = True
    ttl_seconds: Optional[int] = None

    log_level: str = "INFO"
    json_logs: bool = False

    cleanup_threshold: float = 0.3

    auto_create_index: bool = False
    index_threshold_entries: int = 256
    index_num_partitions: int = 256

    max_entries: Optional[int] = 1_000
    eviction_policy: str = "fifo"

    cleanup_interval_seconds: Optional[int] = None
    cleanup_initial_delay: int = 60

    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4318/v1/metrics"
    otel_headers: Optional[str] = None
    otel_service_name: str = "reminiscence"
    otel_export_interval_ms: int = 60000

    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    encryption_backend: Optional[str] = None
    encryption_max_workers: int = 4

    compression_enabled: bool = False
    compression_algorithm: str = "zstd"
    compression_level: int = 3

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.encryption_enabled and not self.encryption_key:
            raise ValueError("encryption_key is required when encryption_enabled=True")

        if self.encryption_enabled and not self.encryption_backend:
            self.encryption_backend = self._detect_encryption_backend()

        if self.compression_enabled:
            valid_algorithms = ["zstd", "gzip", "none"]
            if self.compression_algorithm not in valid_algorithms:
                raise ValueError(
                    f"Invalid compression_algorithm: {self.compression_algorithm}. "
                    f"Must be one of: {', '.join(valid_algorithms)}"
                )

            if self.compression_algorithm == "zstd" and not (
                1 <= self.compression_level <= 22
            ):
                raise ValueError(
                    f"Invalid compression_level for zstd: {self.compression_level}. "
                    "Must be between 1 and 22."
                )

            elif self.compression_algorithm == "gzip" and not (
                1 <= self.compression_level <= 9
            ):
                raise ValueError(
                    f"Invalid compression_level for gzip: {self.compression_level}. "
                    "Must be between 1 and 9."
                )

        for key, threshold in self.context_thresholds.items():
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    f"Invalid threshold for context '{key}': {threshold}. "
                    "Must be between 0.0 and 1.0."
                )

    def _detect_encryption_backend(self) -> str:
        """Auto-detect encryption backend from encryption_key format."""
        if not self.encryption_key:
            raise ValueError("Cannot detect backend: encryption_key is None")

        key = self.encryption_key

        if key.startswith("file://"):
            file_path = key[7:]
            with open(file_path, "r") as f:
                key = f.read().strip()
            self.encryption_key = key

        elif Path(key).is_file():
            with open(key, "r") as f:
                key = f.read().strip()
            self.encryption_key = key

        if key.startswith("arn:aws:kms:"):
            return "aws-kms"

        if key.startswith("projects/") and "/cryptoKeys/" in key:
            return "gcp-kms"

        if key.startswith("https://") and "vault.azure.net" in key:
            return "azure-keyvault"

        if key.startswith("secret/"):
            return "vault"

        if key.startswith("age1") or key.startswith("AGE-SECRET-KEY-"):
            return "age"

        raise ValueError(
            "Cannot auto-detect encryption backend from key format. "
            "Please specify encryption_backend explicitly. "
            "Supported: age, aws-kms, gcp-kms, azure-keyvault, vault"
        )

    def get_threshold_for_context(self, context: Dict[str, Any]) -> float:
        """
        Get similarity threshold for given context.

        Matches context keys against configured thresholds.
        Returns most specific match or default similarity_threshold.
        """
        if not self.context_thresholds:
            return self.similarity_threshold

        for pattern, threshold in self.context_thresholds.items():
            if ":" in pattern:
                key, value = pattern.split(":", 1)
                if key in context and str(context[key]) == value:
                    return threshold
            elif pattern in context:
                return threshold

        return self.similarity_threshold

    @classmethod
    def load(cls) -> "ReminiscenceConfig":
        """Load configuration from environment variables."""
        defaults = cls()

        def parse_bool(value: str) -> bool:
            return value.lower() in ("true", "1", "yes", "on")

        def parse_int_or_none(value: str) -> Optional[int]:
            return None if value.lower() == "none" else int(value)

        def parse_str_or_none(value: str) -> Optional[str]:
            return None if value.lower() in ("none", "") else value

        def parse_context_thresholds(value: str) -> Dict[str, float]:
            if not value or value.lower() == "none":
                return {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}

        return cls(
            model_name=parse_str_or_none(os.getenv("REMINISCENCE_MODEL_NAME", "none")),
            embedding_backend=os.getenv(
                "REMINISCENCE_EMBEDDING_BACKEND", defaults.embedding_backend
            ),
            embedding_batch_size=int(
                os.getenv(
                    "REMINISCENCE_EMBEDDING_BATCH_SIZE",
                    str(defaults.embedding_batch_size),
                )
            ),
            warm_up_embedder=parse_bool(
                os.getenv(
                    "REMINISCENCE_WARM_UP_EMBEDDER",
                    str(defaults.warm_up_embedder).lower(),
                )
            ),
            similarity_threshold=float(
                os.getenv(
                    "REMINISCENCE_SIMILARITY_THRESHOLD",
                    str(defaults.similarity_threshold),
                )
            ),
            context_thresholds=parse_context_thresholds(
                os.getenv("REMINISCENCE_CONTEXT_THRESHOLDS", "{}")
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
            log_level=os.getenv("REMINISCENCE_LOG_LEVEL", defaults.log_level).upper(),
            json_logs=parse_bool(
                os.getenv("REMINISCENCE_JSON_LOGS", str(defaults.json_logs).lower())
            ),
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
            max_entries=parse_int_or_none(
                os.getenv(
                    "REMINISCENCE_MAX_ENTRIES",
                    str(defaults.max_entries) if defaults.max_entries else "none",
                )
            ),
            eviction_policy=os.getenv(
                "REMINISCENCE_EVICTION_POLICY", defaults.eviction_policy
            ).lower(),
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
            encryption_enabled=parse_bool(
                os.getenv(
                    "REMINISCENCE_ENCRYPTION_ENABLED",
                    str(defaults.encryption_enabled).lower(),
                )
            ),
            encryption_key=parse_str_or_none(
                os.getenv("REMINISCENCE_ENCRYPTION_KEY", "none")
            ),
            encryption_backend=parse_str_or_none(
                os.getenv("REMINISCENCE_ENCRYPTION_BACKEND", "none")
            ),
            encryption_max_workers=int(
                os.getenv(
                    "REMINISCENCE_ENCRYPTION_MAX_WORKERS",
                    str(defaults.encryption_max_workers),
                )
            ),
            compression_enabled=parse_bool(
                os.getenv(
                    "REMINISCENCE_COMPRESSION_ENABLED",
                    str(defaults.compression_enabled).lower(),
                )
            ),
            compression_algorithm=os.getenv(
                "REMINISCENCE_COMPRESSION_ALGORITHM",
                defaults.compression_algorithm,
            ).lower(),
            compression_level=int(
                os.getenv(
                    "REMINISCENCE_COMPRESSION_LEVEL",
                    str(defaults.compression_level),
                )
            ),
        )
