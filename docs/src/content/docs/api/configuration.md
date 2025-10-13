---
title: Configuration
description: ReminiscenceConfig for customizing cache behavior
---

The `ReminiscenceConfig` class provides comprehensive configuration options for customizing Reminiscence behavior.

## Constructor

```python
ReminiscenceConfig(
    # Core settings
    similarity_threshold: float = 0.80,
    context_thresholds: Dict[str, float] = {},
    max_entries: Optional[int] = 1000,
    eviction_policy: str = "fifo",
    ttl_seconds: Optional[int] = None,

    # Storage
    db_uri: str = "memory://",
    table_name: str = "semantic_cache",

    # Embeddings
    model_name: Optional[str] = None,
    embedding_backend: str = "fastembed",
    embedding_batch_size: int = 32,
    warm_up_embedder: bool = True,

    # Index
    auto_create_index: bool = False,
    index_threshold_entries: int = 256,
    index_num_partitions: int = 256,

    # Metrics
    enable_metrics: bool = True,

    # OpenTelemetry
    otel_enabled: bool = False,
    otel_endpoint: str = "http://localhost:4318/v1/metrics",
    otel_headers: Optional[str] = None,
    otel_service_name: str = "reminiscence",
    otel_export_interval_ms: int = 60000,

    # Compression
    compression_enabled: bool = False,
    compression_algorithm: str = "zstd",
    compression_level: int = 3,

    # Encryption
    encryption_enabled: bool = False,
    encryption_key: Optional[str] = None,
    encryption_backend: Optional[str] = None,
    encryption_max_workers: int = 4,

    # Logging
    log_level: str = "INFO",
    json_logs: bool = False
)
```

---

## Core Settings

### Similarity Threshold

**`similarity_threshold: float = 0.80`**

Default similarity threshold for semantic matching (0.0-1.0).

```python
config = ReminiscenceConfig(
    similarity_threshold=0.85  # Stricter matching
)
```

**Guidelines:**
- **0.90-0.99**: Very strict (SQL, code, calculations)
- **0.80-0.90**: Balanced (default, recommended)
- **0.60-0.79**: Loose (exploratory, broad matching)

### Context-Specific Thresholds

**`context_thresholds: Dict[str, float] = {}`**

Different thresholds for different contexts.

```python
config = ReminiscenceConfig(
    similarity_threshold=0.80,  # Default
    context_thresholds={
        "agent:sql": 0.95,      # SQL queries need stricter matching
        "model:gpt-4": 0.85,    # GPT-4 slightly stricter
        "agent:translation": 0.75  # Translations can be looser
    }
)
```

### Cache Size & Eviction

**`max_entries: Optional[int] = 1000`**

Maximum number of cache entries. When exceeded, eviction policy kicks in.

```python
config = ReminiscenceConfig(
    max_entries=10000,  # Larger cache
    eviction_policy="lru"  # Evict least recently used
)
```

**Eviction Policies:**
- **`"fifo"`**: First In First Out (default, simple)
- **`"lru"`**: Least Recently Used (time-based)
- **`"lfu"`**: Least Frequently Used (popularity-based)

### TTL (Time-To-Live)

**`ttl_seconds: Optional[int] = None`**

Default expiration time for cache entries. None = no expiration.

```python
config = ReminiscenceConfig(
    ttl_seconds=3600  # Entries expire after 1 hour
)
```

---

## Storage Settings

### Database URI

**`db_uri: str = "memory://"`**

Database location. Use `"memory://"` for in-memory or a file path for persistence.

```python
# In-memory (fastest, data lost on restart)
config = ReminiscenceConfig(db_uri="memory://")

# Persistent SSD (recommended for production)
config = ReminiscenceConfig(db_uri="/mnt/ssd/cache.db")

# Persistent HDD
config = ReminiscenceConfig(db_uri="./cache.db")
```

**Performance:**
- In-memory: 5-10ms lookup
- SSD persistent: 8-15ms lookup
- HDD persistent: 15-50ms lookup

### Table Name

**`table_name: str = "semantic_cache"`**

LanceDB table name (rarely needs changing).

---

## Embedding Settings

### Model Selection

**`model_name: Optional[str] = None`**

Embedding model name. If None, uses backend default.

```python
# Default multilingual model
config = ReminiscenceConfig(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# English-only (faster)
config = ReminiscenceConfig(
    model_name="BAAI/bge-small-en-v1.5"
)

# Code-specific
config = ReminiscenceConfig(
    model_name="jinaai/jina-embeddings-v2-base-code"
)
```

### Embedding Backend

**`embedding_backend: str = "fastembed"`**

Embedding generation backend (currently only FastEmbed supported).

### Batch Size

**`embedding_batch_size: int = 32`**

Number of embeddings to generate in parallel during batch operations.

```python
config = ReminiscenceConfig(
    embedding_batch_size=64  # Larger batches for better throughput
)
```

### Model Warm-Up

**`warm_up_embedder: bool = True`**

Pre-load embedding model on initialization.

```python
# Warm up (recommended)
config = ReminiscenceConfig(warm_up_embedder=True)
# First query: ~5-10ms

# No warm up
config = ReminiscenceConfig(warm_up_embedder=False)
# First query: ~50-100ms (loads model), subsequent: ~5-10ms
```

---

## Indexing Settings

### Auto-Create Index

**`auto_create_index: bool = False`**

Automatically create vector index when threshold reached.

```python
config = ReminiscenceConfig(
    auto_create_index=True,
    index_threshold_entries=256  # Create index at 256 entries
)
```

**Performance impact:**
- Without index: Lookup time grows linearly (10-30ms for 256 entries)
- With index: Lookup time stays constant (5-15ms regardless of size)

### Index Threshold

**`index_threshold_entries: int = 256`**

Number of entries before creating index.

### Index Partitions

**`index_num_partitions: int = 256`**

Number of IVF clusters in vector index.

**Guidelines:**
- Small cache (< 1K): 128 partitions
- Medium cache (1K-10K): 256 partitions (default)
- Large cache (10K-100K): 512 partitions
- Huge cache (100K+): 1024 partitions

**Rule of thumb:** `num_partitions ≈ sqrt(num_entries)`

---

## Observability

### Metrics

**`enable_metrics: bool = True`**

Enable metrics tracking (hits, misses, latencies).

```python
config = ReminiscenceConfig(enable_metrics=True)

# Access metrics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
```

### OpenTelemetry

**`otel_enabled: bool = False`**

Enable OpenTelemetry metrics export.

```python
config = ReminiscenceConfig(
    otel_enabled=True,
    otel_endpoint="http://localhost:4318/v1/metrics",
    otel_service_name="my-cache",
    otel_export_interval_ms=30000  # Export every 30s
)
```

**`otel_headers: Optional[str] = None`**

Custom headers for OTEL endpoint (JSON string).

```python
import json

config = ReminiscenceConfig(
    otel_enabled=True,
    otel_headers=json.dumps({
        "Authorization": "Bearer token123",
        "X-Custom-Header": "value"
    })
)
```

---

## Security

### Compression

**`compression_enabled: bool = False`**

Enable result compression before storage.

```python
config = ReminiscenceConfig(
    compression_enabled=True,
    compression_algorithm="zstd",  # or "gzip"
    compression_level=3  # 1-22 for zstd, 1-9 for gzip
)
```

**Compression algorithms:**
- **`"zstd"`**: Faster, better compression (recommended)
- **`"gzip"`**: More compatible, slightly slower

**Compression levels:**
- **1-3**: Fast, lower compression
- **3-6**: Balanced (default: 3)
- **7-9**: Slower, higher compression

**Typical ratios:**
- DataFrames: 5-10x additional compression (on top of Arrow)
- JSON: 2-3x compression
- Images: 1.1-1.2x (already compressed)

### Encryption

**`encryption_enabled: bool = False`**

Enable encryption for cached data at rest.

```python
import os

config = ReminiscenceConfig(
    encryption_enabled=True,
    encryption_key=os.getenv("CACHE_ENCRYPTION_KEY"),
    encryption_backend="age",  # Age encryption (default)
    encryption_max_workers=4  # Parallel encryption threads
)
```

**Encryption backends:**
- **`"age"`**: Default, fast, secure
- **`"kms"`**: AWS KMS (requires AWS credentials)
- **`"vault"`**: HashiCorp Vault (requires Vault config)

**Security notes:**
- Encryption happens **after** compression
- Key must be 32+ characters for Age encryption
- Store keys securely (environment variables, secrets manager)

---

## Logging

### Log Level

**`log_level: str = "INFO"`**

Logging verbosity.

```python
config = ReminiscenceConfig(log_level="DEBUG")  # More verbose
```

**Levels:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

### JSON Logs

**`json_logs: bool = False`**

Output logs in JSON format (for structured logging).

```python
config = ReminiscenceConfig(json_logs=True)
# Logs: {"level": "info", "message": "Cache hit", "similarity": 0.92}
```

---

## Loading from Environment

### `load()`

Load configuration from environment variables.

```python
@classmethod
load() -> ReminiscenceConfig
```

**Example:**
```python
import os

# Set environment variables
os.environ["REMINISCENCE_MAX_ENTRIES"] = "10000"
os.environ["REMINISCENCE_SIMILARITY_THRESHOLD"] = "0.85"
os.environ["REMINISCENCE_DB_URI"] = "./cache.db"
os.environ["REMINISCENCE_COMPRESSION_ENABLED"] = "true"

# Load config
config = ReminiscenceConfig.load()
cache = Reminiscence(config=config)
```

**Environment Variables:**

**Core:**
- `REMINISCENCE_SIMILARITY_THRESHOLD`: Default threshold (float)
- `REMINISCENCE_MAX_ENTRIES`: Max cache entries (int)
- `REMINISCENCE_EVICTION_POLICY`: fifo/lru/lfu
- `REMINISCENCE_TTL_SECONDS`: Default TTL (int)

**Storage:**
- `REMINISCENCE_DB_URI`: Database location
- `REMINISCENCE_TABLE_NAME`: Table name

**Embeddings:**
- `REMINISCENCE_MODEL_NAME`: Embedding model
- `REMINISCENCE_EMBEDDING_BACKEND`: Backend (fastembed)
- `REMINISCENCE_WARM_UP_EMBEDDER`: true/false

**Indexing:**
- `REMINISCENCE_AUTO_CREATE_INDEX`: true/false
- `REMINISCENCE_INDEX_THRESHOLD_ENTRIES`: Threshold (int)
- `REMINISCENCE_INDEX_NUM_PARTITIONS`: Partitions (int)

**Metrics:**
- `REMINISCENCE_ENABLE_METRICS`: true/false
- `REMINISCENCE_OTEL_ENABLED`: true/false
- `REMINISCENCE_OTEL_ENDPOINT`: OTLP endpoint
- `REMINISCENCE_OTEL_SERVICE_NAME`: Service name

**Security:**
- `REMINISCENCE_COMPRESSION_ENABLED`: true/false
- `REMINISCENCE_COMPRESSION_ALGORITHM`: zstd/gzip
- `REMINISCENCE_ENCRYPTION_ENABLED`: true/false
- `REMINISCENCE_ENCRYPTION_KEY`: Encryption key

**Logging:**
- `REMINISCENCE_LOG_LEVEL`: DEBUG/INFO/WARNING/ERROR
- `REMINISCENCE_JSON_LOGS`: true/false

---

## Complete Example

Production-ready configuration:

```python
import os
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    # Stricter semantic matching
    similarity_threshold=0.85,

    # Context-specific thresholds
    context_thresholds={
        "agent:sql": 0.95,
        "agent:api": 0.90,
    },

    # Large cache with LRU eviction
    max_entries=10000,
    eviction_policy="lru",

    # 1 hour default TTL
    ttl_seconds=3600,

    # Persistent SSD storage
    db_uri="/mnt/ssd/cache.db",

    # Auto-create index
    auto_create_index=True,
    index_threshold_entries=256,

    # Enable compression and encryption
    compression_enabled=True,
    compression_algorithm="zstd",
    encryption_enabled=True,
    encryption_key=os.getenv("CACHE_KEY"),

    # OpenTelemetry for monitoring
    otel_enabled=True,
    otel_endpoint="http://otel-collector:4318/v1/metrics",
    otel_service_name="production-cache",

    # Structured logging
    log_level="INFO",
    json_logs=True
)

cache = Reminiscence(config=config)

# Start background tasks
cache.start_scheduler(
    interval_seconds=1800,  # Cleanup every 30 min
    metrics_export_interval_seconds=30  # Export metrics every 30s
)
```

---

## Next Steps

- **[Core Operations](/reference/api/core-operations/)** - Using configured cache
- **[Data Types](/reference/api/data-types/)** - Request/response models
- **[Examples](/examples/llm-apps/)** - Configuration in practice
