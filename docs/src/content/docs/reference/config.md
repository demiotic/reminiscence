---
title: Configuration Options
description: Complete reference for all Reminiscence configuration options
---

Complete reference for all configuration options available in Reminiscence.

## Environment Variables

All configuration can be set via `REMINISCENCE_*` environment variables.

### Core Settings

#### REMINISCENCE_SIMILARITY_THRESHOLD
**Type:** float (0.0-1.0)
**Default:** 0.80
**Description:** Default similarity threshold for semantic matching

```bash
export REMINISCENCE_SIMILARITY_THRESHOLD=0.85
```

#### REMINISCENCE_CONTEXT_THRESHOLDS
**Type:** JSON dict
**Default:** {}
**Description:** Context-specific similarity thresholds

```bash
export REMINISCENCE_CONTEXT_THRESHOLDS='{"agent:sql":0.95,"model:gpt-4":0.85}'
```

#### REMINISCENCE_MAX_ENTRIES
**Type:** int
**Default:** 1000
**Description:** Maximum cache entries before eviction

```bash
export REMINISCENCE_MAX_ENTRIES=10000
```

#### REMINISCENCE_EVICTION_POLICY
**Type:** string
**Default:** fifo
**Options:** fifo, lru, lfu
**Description:** Eviction strategy when cache is full

```bash
export REMINISCENCE_EVICTION_POLICY=lru
```

#### REMINISCENCE_TTL_SECONDS
**Type:** int | none
**Default:** none
**Description:** Global TTL for cache entries (seconds)

```bash
export REMINISCENCE_TTL_SECONDS=3600  # 1 hour
```

### Storage Settings

#### REMINISCENCE_DB_URI
**Type:** string
**Default:** memory://
**Description:** Storage backend URI

```bash
# In-memory
export REMINISCENCE_DB_URI=memory://

# Persistent file
export REMINISCENCE_DB_URI=./cache.db

# Custom path
export REMINISCENCE_DB_URI=/var/lib/reminiscence/cache.db
```

#### REMINISCENCE_TABLE_NAME
**Type:** string
**Default:** semantic_cache
**Description:** LanceDB table name

```bash
export REMINISCENCE_TABLE_NAME=my_cache
```

### Embedding Settings

#### REMINISCENCE_MODEL_NAME
**Type:** string
**Default:** none (uses backend default)
**Description:** Embedding model name

```bash
export REMINISCENCE_MODEL_NAME=BAAI/bge-small-en-v1.5
```

**Available models:**
- `paraphrase-multilingual-MiniLM-L12-v2` (default, 384 dims)
- `BAAI/bge-small-en-v1.5` (English-only, faster)
- `jinaai/jina-embeddings-v2-base-code` (code-specific)

#### REMINISCENCE_EMBEDDING_BACKEND
**Type:** string
**Default:** fastembed
**Options:** fastembed, auto
**Description:** Embedding provider

```bash
export REMINISCENCE_EMBEDDING_BACKEND=fastembed
```

#### REMINISCENCE_EMBEDDING_BATCH_SIZE
**Type:** int (1-512)
**Default:** 32
**Description:** Batch size for embedding generation

```bash
export REMINISCENCE_EMBEDDING_BATCH_SIZE=64
```

#### REMINISCENCE_WARM_UP_EMBEDDER
**Type:** bool
**Default:** true
**Description:** Pre-load embedding model on initialization

```bash
export REMINISCENCE_WARM_UP_EMBEDDER=true
```

### Index Settings

#### REMINISCENCE_AUTO_CREATE_INDEX
**Type:** bool
**Default:** false
**Description:** Automatically create vector index

```bash
export REMINISCENCE_AUTO_CREATE_INDEX=true
```

#### REMINISCENCE_INDEX_THRESHOLD_ENTRIES
**Type:** int
**Default:** 256
**Description:** Minimum entries required for auto-indexing

```bash
export REMINISCENCE_INDEX_THRESHOLD_ENTRIES=256
```

#### REMINISCENCE_INDEX_NUM_PARTITIONS
**Type:** int
**Default:** 256
**Description:** Number of IVF partitions for index

```bash
export REMINISCENCE_INDEX_NUM_PARTITIONS=512
```

**Guidelines:**
- 128: Small caches (< 1K entries)
- 256: Medium caches (1K-10K entries) [default]
- 512: Large caches (10K-100K entries)
- 1024: Very large caches (100K+ entries)

### Metrics Settings

#### REMINISCENCE_ENABLE_METRICS
**Type:** bool
**Default:** true
**Description:** Enable metrics tracking

```bash
export REMINISCENCE_ENABLE_METRICS=true
```

### OpenTelemetry Settings

#### REMINISCENCE_OTEL_ENABLED
**Type:** bool
**Default:** false
**Description:** Enable OpenTelemetry metrics export

```bash
export REMINISCENCE_OTEL_ENABLED=true
```

#### REMINISCENCE_OTEL_ENDPOINT
**Type:** string
**Default:** http://localhost:4318/v1/metrics
**Description:** OTLP endpoint URL

```bash
export REMINISCENCE_OTEL_ENDPOINT=http://otel-collector:4318/v1/metrics
```

#### REMINISCENCE_OTEL_HEADERS
**Type:** string
**Default:** none
**Format:** key1=value1,key2=value2
**Description:** Custom headers for OTLP requests

```bash
export REMINISCENCE_OTEL_HEADERS=x-api-key=secret,x-tenant=acme
```

#### REMINISCENCE_OTEL_SERVICE_NAME
**Type:** string
**Default:** reminiscence
**Description:** Service name for telemetry

```bash
export REMINISCENCE_OTEL_SERVICE_NAME=my-service
```

#### REMINISCENCE_OTEL_EXPORT_INTERVAL_MS
**Type:** int
**Default:** 60000
**Description:** Metrics export interval (milliseconds)

```bash
export REMINISCENCE_OTEL_EXPORT_INTERVAL_MS=30000  # 30 seconds
```

### Compression Settings

#### REMINISCENCE_COMPRESSION_ENABLED
**Type:** bool
**Default:** false
**Description:** Enable result compression

```bash
export REMINISCENCE_COMPRESSION_ENABLED=true
```

#### REMINISCENCE_COMPRESSION_ALGORITHM
**Type:** string
**Default:** zstd
**Options:** zstd, gzip, none
**Description:** Compression algorithm

```bash
export REMINISCENCE_COMPRESSION_ALGORITHM=zstd
```

#### REMINISCENCE_COMPRESSION_LEVEL
**Type:** int
**Default:** 3
**Range:** zstd: 1-22, gzip: 1-9
**Description:** Compression level

```bash
export REMINISCENCE_COMPRESSION_LEVEL=5
```

**Trade-offs:**
- Lower: Faster, larger size
- Higher: Slower, smaller size

### Encryption Settings

#### REMINISCENCE_ENCRYPTION_ENABLED
**Type:** bool
**Default:** false
**Description:** Enable result encryption

```bash
export REMINISCENCE_ENCRYPTION_ENABLED=true
```

#### REMINISCENCE_ENCRYPTION_KEY
**Type:** string
**Default:** none
**Description:** Encryption key

```bash
# Age key
export REMINISCENCE_ENCRYPTION_KEY=age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8p

# File path
export REMINISCENCE_ENCRYPTION_KEY=file:///path/to/key.txt

# From file
export REMINISCENCE_ENCRYPTION_KEY=/etc/secrets/cache_key
```

#### REMINISCENCE_ENCRYPTION_BACKEND
**Type:** string
**Default:** auto-detected
**Options:** age, aws-kms, gcp-kms, azure-keyvault, vault
**Description:** Encryption backend

```bash
export REMINISCENCE_ENCRYPTION_BACKEND=age
```

#### REMINISCENCE_ENCRYPTION_MAX_WORKERS
**Type:** int
**Default:** 4
**Description:** Number of threads for batch encryption

```bash
export REMINISCENCE_ENCRYPTION_MAX_WORKERS=8
```

### Scheduler Settings

#### REMINISCENCE_CLEANUP_INTERVAL_SECONDS
**Type:** int
**Default:** 3600
**Description:** Interval for background cleanup (seconds)

```bash
export REMINISCENCE_CLEANUP_INTERVAL_SECONDS=1800  # 30 minutes
```

#### REMINISCENCE_CLEANUP_INITIAL_DELAY
**Type:** int
**Default:** 60
**Description:** Delay before first cleanup run (seconds)

```bash
export REMINISCENCE_CLEANUP_INITIAL_DELAY=300  # 5 minutes
```

### Logging Settings

#### REMINISCENCE_LOG_LEVEL
**Type:** string
**Default:** INFO
**Options:** DEBUG, INFO, WARNING, ERROR
**Description:** Logging level

```bash
export REMINISCENCE_LOG_LEVEL=INFO
```

#### REMINISCENCE_JSON_LOGS
**Type:** bool
**Default:** false
**Description:** Enable structured JSON logging

```bash
export REMINISCENCE_JSON_LOGS=true
```

## Configuration Presets

### Development

```bash
export REMINISCENCE_DB_URI=memory://
export REMINISCENCE_LOG_LEVEL=DEBUG
export REMINISCENCE_JSON_LOGS=false
export REMINISCENCE_ENABLE_METRICS=true
export REMINISCENCE_COMPRESSION_ENABLED=false
export REMINISCENCE_ENCRYPTION_ENABLED=false
```

### Production

```bash
export REMINISCENCE_DB_URI=./cache.db
export REMINISCENCE_MAX_ENTRIES=100000
export REMINISCENCE_EVICTION_POLICY=lru
export REMINISCENCE_TTL_SECONDS=86400
export REMINISCENCE_AUTO_CREATE_INDEX=true
export REMINISCENCE_WARM_UP_EMBEDDER=true
export REMINISCENCE_ENABLE_METRICS=true
export REMINISCENCE_OTEL_ENABLED=true
export REMINISCENCE_OTEL_ENDPOINT=http://collector:4318/v1/metrics
export REMINISCENCE_OTEL_EXPORT_INTERVAL_MS=30000
export REMINISCENCE_COMPRESSION_ENABLED=true
export REMINISCENCE_COMPRESSION_ALGORITHM=zstd
export REMINISCENCE_COMPRESSION_LEVEL=3
export REMINISCENCE_LOG_LEVEL=INFO
export REMINISCENCE_JSON_LOGS=true
```

### High Performance

```bash
export REMINISCENCE_DB_URI=memory://
export REMINISCENCE_MAX_ENTRIES=1000000
export REMINISCENCE_EVICTION_POLICY=fifo
export REMINISCENCE_AUTO_CREATE_INDEX=true
export REMINISCENCE_INDEX_NUM_PARTITIONS=1024
export REMINISCENCE_WARM_UP_EMBEDDER=true
export REMINISCENCE_EMBEDDING_BATCH_SIZE=64
export REMINISCENCE_COMPRESSION_ENABLED=false
export REMINISCENCE_ENCRYPTION_ENABLED=false
export REMINISCENCE_ENABLE_METRICS=false
```

### Secure

```bash
export REMINISCENCE_DB_URI=./cache.db
export REMINISCENCE_ENCRYPTION_ENABLED=true
export REMINISCENCE_ENCRYPTION_KEY=age1...
export REMINISCENCE_ENCRYPTION_BACKEND=age
export REMINISCENCE_COMPRESSION_ENABLED=true
export REMINISCENCE_LOG_LEVEL=WARNING
export REMINISCENCE_JSON_LOGS=true
```

## Configuration Validation

Reminiscence validates configuration on initialization:

```python
from reminiscence import ReminiscenceConfig

try:
    config = ReminiscenceConfig(
        encryption_enabled=True
        # Missing encryption_key - will raise ValueError
    )
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

**Validation checks:**
- Encryption key required when encryption enabled
- Compression level within valid range
- Context thresholds between 0.0 and 1.0
- Embedding batch size between 1 and 512

## Configuration Methods

### get_threshold_for_context()

Get effective threshold for a context:

```python
config = ReminiscenceConfig(
    similarity_threshold=0.80,
    context_thresholds={"agent:sql": 0.95}
)

# Default threshold
threshold = config.get_threshold_for_context({})  # 0.80

# Context-specific threshold
threshold = config.get_threshold_for_context({"agent": "sql"})  # 0.95
```

## Next Steps

- [API Documentation](/reference/api/) - Complete API reference
- [Metrics Reference](/reference/metrics/) - Available metrics
- [Configuration Guide](/guides/configuration/) - Detailed configuration guide
