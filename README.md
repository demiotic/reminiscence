# Memora
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**Semantic cache for multi-agent systems**

Memora is an **experimental** semantic caching library for AI agents, built on LanceDB and sentence-transformers. **Beta stage** - approaching production readiness.

> [!WARNING]
> **Beta Release (v0.1.0) - Approaching Production**
> 
> Memora is functional with production features, but still missing some critical capabilities:
> - ✅ Health checks
> - ✅ Structured logging
> - ✅ Error handling
> - ✅ Size limits and eviction
> - ❌ Background cleanup scheduler
> - ❌ LRU eviction (only FIFO available)
>
> **Use for:** Development, staging environments, low-concurrency production  
>
> See [Roadmap](#roadmap) for remaining features.

## Why Memora?

Multi-agent systems often execute the same expensive operations repeatedly. Memora caches results based on semantic similarity rather than exact matches:

```python
# These queries hit the same cache entry:
"Analyze Q3 sales data"
"Show me sales analysis for Q3"
"What were the sales in the third quarter?"
```

## Features

### Core Functionality
- **Semantic matching** - Finds similar queries using embeddings (cosine similarity)
- **Context-aware** - Groups cache by execution context (agent, config, tools)
- **Flexible storage** - In-memory, on-disk, or remote LanceDB
- **Type-safe serialization** - Handles pandas/polars DataFrames, numpy arrays, nested dicts
- **TTL and eviction** - Automatic expiration with FIFO eviction policy
- **Metrics tracking** - Hit rate, latency, payload sizes
- **Zero-config decorators** - Cache functions transparently

### ⭐ Monitoring & Observability (v0.1.0-beta)

#### Health Checks
Monitor cache health in production with built-in diagnostics:

```python
from memora import Memora, CacheConfig

memora = Memora(CacheConfig.for_production())

# Check cache health
health = memora.health_check()
print(health["status"])  # "healthy" or "unhealthy"

# Example output:
# {
#   "status": "healthy",
#   "checks": {
#     "embedding": {"ok": true, "error": null},
#     "database": {"ok": true, "error": null},
#     "error_rate": {"ok": true, "details": "Error rate: 0.5%"}
#   },
#   "metrics": {
#     "total_entries": 1234,
#     "recent_errors": {"lookup": 2, "store": 0}
#   }
# }
```

**Use cases:**
- Kubernetes liveness/readiness probes
- Monitoring alerts (Prometheus, Datadog, CloudWatch)
- Production debugging

#### Structured Logging
JSON logs for production observability:

```python
# Development: Human-readable console logs
config = CacheConfig.for_development()  # json_logs=False

# Production: Structured JSON logs
config = CacheConfig.for_production()  # json_logs=True
memora = Memora(config)

# Example JSON log output:
# {"event": "cache_hit", "timestamp": "2025-10-05T14:30:00Z", 
#  "similarity": 0.923, "latency_ms": 12.3, "age_seconds": 120}
```

**Integrations:** ELK Stack, Datadog, Grafana Loki, CloudWatch Logs

#### Environment Variable Configuration
Configure Memora without code changes (Docker/Kubernetes friendly):

```bash
# docker-compose.yml or Kubernetes ConfigMap
environment:
  MEMORA_DB_URI: /var/cache/memora
  MEMORA_JSON_LOGS: true
  MEMORA_LOG_LEVEL: INFO
  MEMORA_MAX_ENTRIES: 100000
  MEMORA_TTL_SECONDS: 3600
  MEMORA_SIMILARITY_THRESHOLD: 0.85
```

```python
from memora import Memora, CacheConfig

# Reads all config from environment variables
memora = Memora(CacheConfig.from_env())
```

**Supported environment variables:**
- `MEMORA_DB_URI` - Database path (default: memory://)
- `MEMORA_JSON_LOGS` - Enable JSON logging (true/false)
- `MEMORA_LOG_LEVEL` - Log level (DEBUG/INFO/WARNING/ERROR)
- `MEMORA_MAX_ENTRIES` - Max cache entries (int or None)
- `MEMORA_TTL_SECONDS` - Time-to-live in seconds
- `MEMORA_SIMILARITY_THRESHOLD` - Match threshold (0.0-1.0)
- `MEMORA_MAX_RESULT_SIZE_BYTES` - Max payload size
- `MEMORA_EVICTION_POLICY` - Eviction strategy (fifo)
- See `CacheConfig.from_env()` docstring for complete list

## Quick Start

```bash
pip install lancedb sentence-transformers orjson pyarrow
```

### Basic Usage

```python
from memora import Memora, CacheConfig

# Initialize
memora = Memora(CacheConfig.for_development())

# Check cache
result = memora.lookup(
    query="Analyze sales for Q3 2024",
    context={"agent": "sql_analyzer", "db": "production"}
)

if result.is_hit:
    print(f"Cache hit! Similarity: {result.similarity:.2f}")
    print(result.result)
else:
    # Execute expensive operation
    data = expensive_sql_query(...)

    # Store in cache
    memora.store(
        query="Analyze sales for Q3 2024",
        context={"agent": "sql_analyzer", "db": "production"},
        result=data
    )
```

### Decorator API

```python
from memora import Memora, CacheConfig, create_cached_decorator

memora = Memora(CacheConfig.for_development())
cached = create_cached_decorator(memora)

@cached(context={"agent": "sql"})
def query_database(query: str, db: str, timeout: int = 30):
    # This function results are automatically cached
    return execute_query(query, db, timeout)

# First call executes, second hits cache
result1 = query_database("SELECT * FROM sales", db="prod")
result2 = query_database("Show all sales data", db="prod")  # Cache hit!
```

## Configuration

### Development

```python
config = CacheConfig.for_development()
# - In-memory storage
# - Debug logging
# - Human-readable logs
# - Metrics enabled
```

### Production

```python
config = CacheConfig.for_production(db_path="./cache.db")
# - Persistent storage
# - 1h TTL
# - JSON structured logging
# - Auto-indexing at 1000 entries
# - 512 IVF partitions
```

### Custom

```python
config = CacheConfig(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold=0.85,
    db_uri="./my_cache.db",
    ttl_seconds=3600,  # 1 hour
    enable_metrics=True,
    json_logs=True,  # Structured logging
    auto_create_index=True,
    index_threshold_entries=500,
    max_entries=50_000,
    max_result_size_bytes=10_000_000,
)
```

## Advanced Features

### Similarity Threshold

Control how strict the matching is:

```python
# Strict matching (0.90+): Only very similar queries match
result = memora.lookup(query, context, similarity_threshold=0.92)

# Relaxed matching (0.70-0.80): More flexible
result = memora.lookup(query, context, similarity_threshold=0.75)
```

### TTL and Invalidation

```python
# Invalidate by context
memora.invalidate(context={"agent": "sql", "db": "staging"})

# Invalidate old entries
memora.invalidate(older_than_seconds=3600)  # Older than 1 hour

# Cleanup expired entries
deleted = memora.cleanup_expired()
```

### Size Limits and Eviction

```python
config = CacheConfig(
    max_entries=10_000,  # Maximum cache entries
    max_result_size_bytes=10_000_000,  # 10MB max per result
    eviction_policy="fifo",  # First In First Out
)

memora = Memora(config)

# When max_entries is reached, oldest entries are automatically evicted
```

### Vector Indexing

For production with thousands of entries:

```python
memora = Memora(CacheConfig.for_production())

# Add many entries...
for i in range(10000):
    memora.store(query=f"query {i}", context={...}, result={...})

# Create IVF-PQ index for fast search
memora.create_index(num_partitions=512)

# Check index stats
stats = memora.get_index_stats()
print(stats)  # {'has_index': True, 'total_entries': 10000}
```

### Metrics

```python
memora = Memora(CacheConfig(enable_metrics=True))

# Use cache...
# ...

# Get performance metrics
stats = memora.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"Latency p95: {stats['lookup_latency_ms']['p95']:.1f}ms")
print(f"Total savings: {stats['total_latency_saved_ms']/1000:.1f}s")
print(f"Errors: {stats['errors']}")
```

## Supported Data Types

Memora handles common Python types and scientific data structures:

- **Basic types**: str, int, float, bool, None, dict, list
- **Pandas**: DataFrame, Series
- **Polars**: DataFrame
- **Numpy**: ndarray (preserves dtype and shape)
- **Nested structures**: Arbitrarily nested combinations

```python
# All of these work out of the box
memora.store(query, context, "simple string")
memora.store(query, context, {"nested": {"data": [1, 2, 3]}})
memora.store(query, context, pd.DataFrame(...))
memora.store(query, context, np.array(...))
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Your Agent                    │
│  ┌─────────────────────────────────────────┐    │
│  │  1. Check cache (lookup)                │    │
│  │  2. Execute if miss                     │    │
│  │  3. Store result                        │    │
│  └─────────────────────────────────────────┘    │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │     Memora     │
         │  Core + Utils  │
         └────────┬───────┘
                  │
        ┏━━━━━━━━━┻━━━━━━━━━┓
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Embeddings   │    │   LanceDB    │
│ (sentence-   │    │  (Vector DB) │
│ transformers)│    │              │
└──────────────┘    └──────────────┘
```

## Examples

See the `examples/` directory:

- `basic_usage.py`: Simple cache operations
- `diagnostics.py`: Monitoring and debugging
- `decorator_example.py`: Function caching with decorators
- `multiagent_pipeline.py`: Real-world multi-agent scenario

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=memora --cov-report=html
```

## Performance

Typical latencies on consumer hardware:

- **Lookup (no index)**: 10-50ms for <1000 entries
- **Lookup (with index)**: 5-15ms for >10000 entries
- **Store**: 5-10ms
- **Embedding generation**: 20-50ms (multilingual model)

For production workloads >1000 entries, enable auto-indexing.

## Requirements

- Python 3.9+
- lancedb >= 0.15.0
- sentence-transformers >= 3.0.0
- orjson >= 3.10.0
- pyarrow >= 18.0.0
- structlog >= 24.0.0 (for structured logging)

Optional:
- pandas (for DataFrame caching)
- polars (for Polars DataFrame caching)
- numpy (for array caching)

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

© 2025 Your Name or Organization

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but **WITHOUT ANY WARRANTY**; without even the implied warranty of
**MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE**.
See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Roadmap

### v0.1.0 (Current)
- [x] Health check method
- [x] Structured logging (JSON)
- [x] Environment variable configuration
- [x] Improved error handling and logging

### v0.2.0 (Planned)
- [ ] Background cleanup scheduler
- [ ] LRU eviction policy
- [ ] Prometheus metrics exporter
- [ ] S3/GCS remote storage backends
- [ ] Multi-tenancy support

### Future
- [ ] Custom embedding models
- [ ] Semantic invalidation (invalidate by query similarity)
- [ ] Distributed caching
- [ ] GraphQL API
