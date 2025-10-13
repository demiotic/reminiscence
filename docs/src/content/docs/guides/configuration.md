---
title: Configuration
description: Complete guide to configuring Reminiscence
---

Reminiscence can be configured via code or environment variables. This guide covers all configuration options.

## Configuration Methods

###

 Via Config Object

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    similarity_threshold=0.85,
    max_entries=10000,
    eviction_policy="lru",
    ttl_seconds=3600,
    db_uri="./cache.db",
    enable_metrics=True
)

cache = Reminiscence(config=config)
```

### Via Environment Variables

```bash
export REMINISCENCE_SIMILARITY_THRESHOLD=0.85
export REMINISCENCE_MAX_ENTRIES=10000
export REMINISCENCE_EVICTION_POLICY=lru
export REMINISCENCE_TTL_SECONDS=3600
export REMINISCENCE_DB_URI=./cache.db
export REMINISCENCE_ENABLE_METRICS=true
```

```python
# Loads from environment
cache = Reminiscence()
```

## Core Settings

### Similarity Threshold

Controls semantic matching strictness (0.0 - 1.0):

```python
# Via config
config = ReminiscenceConfig(similarity_threshold=0.85)

# Via environment
# REMINISCENCE_SIMILARITY_THRESHOLD=0.85
```

**Default:** `0.80`

**Recommendations:**
- `0.90-0.99`: SQL, API calls, code execution
- `0.80-0.90`: General LLM queries, Q&A
- `0.60-0.79`: Broad topic matching

### Context-Specific Thresholds

Different thresholds for different contexts:

```python
import json
import os

os.environ["REMINISCENCE_CONTEXT_THRESHOLDS"] = json.dumps({
    "agent:sql": 0.95,
    "agent:translation": 0.75,
    "model:gpt-4": 0.85
})

cache = Reminiscence()
```

**Format:** JSON dict with `"key:value"` or `"key"` patterns

### Max Entries

Maximum cache size before eviction:

```python
config = ReminiscenceConfig(max_entries=10000)

# Via environment
# REMINISCENCE_MAX_ENTRIES=10000
```

**Default:** `1000`

**Set to `None` for unlimited** (not recommended for production)

### Eviction Policy

Strategy for removing entries when cache is full:

```python
config = ReminiscenceConfig(eviction_policy="lru")

# Via environment
# REMINISCENCE_EVICTION_POLICY=lru
```

**Options:**
- `fifo` (default): First In, First Out
- `lru`: Least Recently Used
- `lfu`: Least Frequently Used

## Storage Settings

### Database URI

Storage backend location:

```python
# In-memory (default)
config = ReminiscenceConfig(db_uri="memory://")

# Persistent file
config = ReminiscenceConfig(db_uri="./cache.db")

# Custom path
config = ReminiscenceConfig(db_uri="/var/lib/reminiscence/cache.db")

# Via environment
# REMINISCENCE_DB_URI=./cache.db
```

**Default:** `memory://`

### Table Name

Custom table name in LanceDB:

```python
config = ReminiscenceConfig(table_name="my_cache")

# Via environment
# REMINISCENCE_TABLE_NAME=my_cache
```

**Default:** `semantic_cache`

## TTL Settings

### Global TTL

Default expiration time for all entries:

```python
config = ReminiscenceConfig(ttl_seconds=3600)  # 1 hour

# Via environment
# REMINISCENCE_TTL_SECONDS=3600
```

**Default:** `None` (no expiration)

### Cleanup Interval

How often to run TTL cleanup:

```python
config = ReminiscenceConfig(
    ttl_seconds=3600,
    cleanup_interval_seconds=1800  # Check every 30 minutes
)

# Via environment
# REMINISCENCE_CLEANUP_INTERVAL_SECONDS=1800
```

**Default:** `3600` (1 hour)

### Cleanup Initial Delay

Delay before first cleanup run:

```python
config = ReminiscenceConfig(
    cleanup_initial_delay=60  # Wait 1 minute before first run
)

# Via environment
# REMINISCENCE_CLEANUP_INITIAL_DELAY=60
```

**Default:** `60` seconds

## Embedding Settings

### Model Selection

Choose embedding model:

```python
# Use default multilingual model
config = ReminiscenceConfig()  # paraphrase-multilingual-MiniLM-L12-v2

# Custom model
config = ReminiscenceConfig(
    model_name="BAAI/bge-small-en-v1.5"  # English-only, faster
)

# Via environment
# REMINISCENCE_MODEL_NAME=BAAI/bge-small-en-v1.5
```

**Default:** `None` (uses backend default: `paraphrase-multilingual-MiniLM-L12-v2`)

**Available models:** See `reminiscence/embeddings/models.yaml`

### Embedding Backend

Embedding provider:

```python
config = ReminiscenceConfig(embedding_backend="fastembed")

# Via environment
# REMINISCENCE_EMBEDDING_BACKEND=fastembed
```

**Default:** `fastembed`

**Options:** `fastembed`, `auto`

### Batch Size

Batch size for embedding generation:

```python
config = ReminiscenceConfig(embedding_batch_size=64)

# Via environment
# REMINISCENCE_EMBEDDING_BATCH_SIZE=64
```

**Default:** `32`

**Range:** `1-512`

### Model Warm-up

Pre-load model on initialization:

```python
config = ReminiscenceConfig(warm_up_embedder=True)  # Default

# Via environment
# REMINISCENCE_WARM_UP_EMBEDDER=true
```

**Default:** `true`

**Trade-off:**
- `true`: 50-100ms initialization, fast first query
- `false`: Fast initialization, slow first query

## Index Settings

### Auto-Create Index

Automatically create vector index:

```python
config = ReminiscenceConfig(auto_create_index=True)

# Via environment
# REMINISCENCE_AUTO_CREATE_INDEX=true
```

**Default:** `false`

**Recommendation:** Enable for caches with 256+ entries

### Index Threshold

Minimum entries required for auto-indexing:

```python
config = ReminiscenceConfig(
    auto_create_index=True,
    index_threshold_entries=256
)

# Via environment
# REMINISCENCE_INDEX_THRESHOLD_ENTRIES=256
```

**Default:** `256`

### Index Partitions

Number of IVF partitions:

```python
config = ReminiscenceConfig(index_num_partitions=512)

# Via environment
# REMINISCENCE_INDEX_NUM_PARTITIONS=512
```

**Default:** `256`

**Guideline:** ~sqrt(num_entries) for optimal performance

## Metrics Settings

### Enable Metrics

Track cache performance:

```python
config = ReminiscenceConfig(enable_metrics=True)  # Default

# Via environment
# REMINISCENCE_ENABLE_METRICS=true
```

**Default:** `true`

**Metrics tracked:**
- Hits/misses, hit rate
- Lookup latencies (p50, p95, p99)
- Storage operations
- Eviction stats
- Embedding generation

## OpenTelemetry Settings

### Enable OTEL

Export metrics to OpenTelemetry collector:

```python
config = ReminiscenceConfig(
    otel_enabled=True,
    otel_endpoint="http://localhost:4318/v1/metrics",
    otel_service_name="my_service"
)

# Via environment
# REMINISCENCE_OTEL_ENABLED=true
# REMINISCENCE_OTEL_ENDPOINT=http://localhost:4318/v1/metrics
# REMINISCENCE_OTEL_SERVICE_NAME=my_service
```

**Default:** `false`

### OTEL Endpoint

OTLP endpoint URL:

```python
config = ReminiscenceConfig(
    otel_endpoint="http://collector:4318/v1/metrics"
)

# Via environment
# REMINISCENCE_OTEL_ENDPOINT=http://collector:4318/v1/metrics
```

**Default:** `http://localhost:4318/v1/metrics`

### OTEL Headers

Custom headers for OTLP requests:

```python
config = ReminiscenceConfig(
    otel_headers="x-api-key=secret,x-tenant=acme"
)

# Via environment
# REMINISCENCE_OTEL_HEADERS=x-api-key=secret,x-tenant=acme
```

**Format:** Comma-separated `key=value` pairs

### Export Interval

How often to export metrics:

```python
config = ReminiscenceConfig(
    otel_export_interval_ms=10000  # 10 seconds
)

# Via environment
# REMINISCENCE_OTEL_EXPORT_INTERVAL_MS=10000
```

**Default:** `60000` (60 seconds)

## Compression Settings

### Enable Compression

Compress cached results:

```python
config = ReminiscenceConfig(
    compression_enabled=True,
    compression_algorithm="zstd",
    compression_level=3
)

# Via environment
# REMINISCENCE_COMPRESSION_ENABLED=true
# REMINISCENCE_COMPRESSION_ALGORITHM=zstd
# REMINISCENCE_COMPRESSION_LEVEL=3
```

**Default:** `false`

### Compression Algorithm

```python
config = ReminiscenceConfig(compression_algorithm="zstd")

# Via environment
# REMINISCENCE_COMPRESSION_ALGORITHM=zstd
```

**Options:**
- `zstd` (default): Fast, high ratio
- `gzip`: Standard compression
- `none`: Disable compression

### Compression Level

```python
config = ReminiscenceConfig(compression_level=5)

# Via environment
# REMINISCENCE_COMPRESSION_LEVEL=5
```

**zstd:** `1-22` (default: `3`)
**gzip:** `1-9` (default: `3`)

**Trade-off:**
- Lower: Faster, larger
- Higher: Slower, smaller

## Encryption Settings

### Enable Encryption

Encrypt cached results:

```python
config = ReminiscenceConfig(
    encryption_enabled=True,
    encryption_key="age1...",  # Age public key
    encryption_backend="age"
)

# Via environment
# REMINISCENCE_ENCRYPTION_ENABLED=true
# REMINISCENCE_ENCRYPTION_KEY=age1...
# REMINISCENCE_ENCRYPTION_BACKEND=age
```

**Default:** `false`

### Encryption Key

Encryption key (format depends on backend):

```python
# Age key
config = ReminiscenceConfig(
    encryption_key="age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8p"
)

# File path
config = ReminiscenceConfig(
    encryption_key="file:///path/to/key.txt"
)

# Environment variable
# REMINISCENCE_ENCRYPTION_KEY=age1...
```

### Encryption Backend

```python
config = ReminiscenceConfig(encryption_backend="age")

# Via environment
# REMINISCENCE_ENCRYPTION_BACKEND=age
```

**Options:**
- `age`: Age encryption (default)
- `aws-kms`: AWS KMS
- `gcp-kms`: Google Cloud KMS
- `azure-keyvault`: Azure Key Vault
- `vault`: HashiCorp Vault

### Encryption Workers

Number of threads for batch encryption:

```python
config = ReminiscenceConfig(encryption_max_workers=8)

# Via environment
# REMINISCENCE_ENCRYPTION_MAX_WORKERS=8
```

**Default:** `4`

## Logging Settings

### Log Level

```python
config = ReminiscenceConfig(log_level="INFO")

# Via environment
# REMINISCENCE_LOG_LEVEL=INFO
```

**Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`

**Default:** `INFO`

### JSON Logs

Enable structured JSON logging:

```python
config = ReminiscenceConfig(json_logs=True)

# Via environment
# REMINISCENCE_JSON_LOGS=true
```

**Default:** `false`

## Production Configuration

Recommended settings for production:

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    # Storage
    db_uri="./cache.db",  # Persistent storage
    max_entries=100000,

    # Eviction
    eviction_policy="lru",  # Most useful entries stay

    # TTL
    ttl_seconds=86400,  # 24 hour expiration
    cleanup_interval_seconds=3600,

    # Performance
    auto_create_index=True,
    index_threshold_entries=256,
    warm_up_embedder=True,

    # Observability
    enable_metrics=True,
    otel_enabled=True,
    otel_endpoint="http://collector:4318/v1/metrics",
    otel_export_interval_ms=30000,  # 30s

    # Security
    compression_enabled=True,
    compression_algorithm="zstd",
    encryption_enabled=True,  # If handling sensitive data
    encryption_key=os.getenv("CACHE_ENCRYPTION_KEY"),

    # Logging
    log_level="INFO",
    json_logs=True
)

cache = Reminiscence(config=config)
```

## Development Configuration

Recommended settings for development:

```python
config = ReminiscenceConfig(
    # Storage
    db_uri="memory://",  # Fast, no persistence

    # Observability
    enable_metrics=True,
    log_level="DEBUG",
    json_logs=False,  # Human-readable logs

    # No security overhead
    compression_enabled=False,
    encryption_enabled=False
)

cache = Reminiscence(config=config)
```

## Environment Variable Reference

Complete list of environment variables:

```bash
# Core
REMINISCENCE_SIMILARITY_THRESHOLD=0.80
REMINISCENCE_CONTEXT_THRESHOLDS={"agent:sql":0.95}
REMINISCENCE_MAX_ENTRIES=1000
REMINISCENCE_EVICTION_POLICY=fifo

# Storage
REMINISCENCE_DB_URI=memory://
REMINISCENCE_TABLE_NAME=semantic_cache

# TTL
REMINISCENCE_TTL_SECONDS=
REMINISCENCE_CLEANUP_INTERVAL_SECONDS=3600
REMINISCENCE_CLEANUP_INITIAL_DELAY=60

# Embeddings
REMINISCENCE_MODEL_NAME=
REMINISCENCE_EMBEDDING_BACKEND=fastembed
REMINISCENCE_EMBEDDING_BATCH_SIZE=32
REMINISCENCE_WARM_UP_EMBEDDER=true

# Index
REMINISCENCE_AUTO_CREATE_INDEX=false
REMINISCENCE_INDEX_THRESHOLD_ENTRIES=256
REMINISCENCE_INDEX_NUM_PARTITIONS=256

# Metrics
REMINISCENCE_ENABLE_METRICS=true

# OpenTelemetry
REMINISCENCE_OTEL_ENABLED=false
REMINISCENCE_OTEL_ENDPOINT=http://localhost:4318/v1/metrics
REMINISCENCE_OTEL_HEADERS=
REMINISCENCE_OTEL_SERVICE_NAME=reminiscence
REMINISCENCE_OTEL_EXPORT_INTERVAL_MS=60000

# Compression
REMINISCENCE_COMPRESSION_ENABLED=false
REMINISCENCE_COMPRESSION_ALGORITHM=zstd
REMINISCENCE_COMPRESSION_LEVEL=3

# Encryption
REMINISCENCE_ENCRYPTION_ENABLED=false
REMINISCENCE_ENCRYPTION_KEY=
REMINISCENCE_ENCRYPTION_BACKEND=
REMINISCENCE_ENCRYPTION_MAX_WORKERS=4

# Logging
REMINISCENCE_LOG_LEVEL=INFO
REMINISCENCE_JSON_LOGS=false
```

## Configuration Validation

Reminiscence validates configuration on initialization:

```python
# Invalid configuration raises ValueError
try:
    config = ReminiscenceConfig(
        encryption_enabled=True
        # Missing encryption_key!
    )
except ValueError as e:
    print(f"Config error: {e}")
    # Config error: encryption_key is required when encryption_enabled=True
```

## Runtime Configuration

Some settings can be overridden at runtime:

```python
cache = Reminiscence(config=config)

# Override similarity threshold per lookup
result = cache.lookup(
    query=query,
    context=context,
    similarity_threshold=0.95  # Overrides config default
)

# Override mode per lookup
result = cache.lookup(
    query=query,
    context=context,
    mode=QueryMode.EXACT
)

# Override TTL per store
cache.store(
    query=query,
    context=context,
    result=data,
    ttl_seconds=300  # 5 minutes (overrides global TTL)
)
```

## Next Steps

- [Background Tasks](/guides/background-tasks/) - Schedulers and maintenance
- [Production](/production/best-practices/) - Production deployment guide
- [OpenTelemetry](/production/opentelemetry/) - Metrics and observability
