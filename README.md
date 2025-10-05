# Memora
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-155%20passing-brightgreen.svg)]()

**Semantic cache for multi-agent systems and LLM applications**

Memora is a production-ready semantic caching library built on LanceDB and sentence-transformers. It eliminates redundant computations in AI systems by matching queries semantically rather than by exact string comparison.

## Why Memora?

Traditional caching fails for AI systems because users express the same intent differently:

```python
# These queries should hit the same cache:
"Analyze Q3 sales data"
"Show me sales analysis for the third quarter"
"What were Q3 revenues?"
```

Memora solves this with **semantic similarity matching** using embedding vectors, reducing costs and latency in multi-agent systems, RAG pipelines, and LLM applications.

## Features

### ✅ Production-Ready (v0.1.0)

- **Semantic Matching** - Cosine similarity search with configurable thresholds (0.75-0.95)
- **Context-Aware Caching** - Separates cache by execution context (agent, config, database, etc.)
- **Flexible Storage** - In-memory, local disk, or remote LanceDB
- **Type-Safe Serialization** - Handles pandas/polars DataFrames, numpy arrays, nested dicts, large payloads (10MB+)
- **TTL & Eviction** - Time-based expiration + FIFO eviction when limits reached
- **Size Limits** - Configurable max entries and payload sizes
- **Health Checks** - Production monitoring with component diagnostics
- **Structured Logging** - JSON logs for observability (ELK, Datadog, Grafana)
- **Environment Config** - 12-factor app support (Docker/Kubernetes friendly)
- **Metrics Tracking** - Hit rate, latency percentiles (p50/p95/p99), error rates
- **Zero-Config Decorators** - Drop-in function caching with `@cached`
- **Vector Indexing** - IVF-PQ for fast search at scale (>1K entries)

### 🚧 Roadmap (v0.2.0+)

- [ ] Background cleanup scheduler
- [ ] LRU eviction policy (currently FIFO only)
- [ ] Prometheus metrics exporter
- [ ] S3/GCS remote storage
- [ ] Distributed caching (Redis-compatible protocol)
- [ ] Semantic invalidation (invalidate by query similarity)

## Installation

```bash
pip install lancedb sentence-transformers orjson pyarrow structlog

# Optional dependencies
pip install pandas polars numpy  # For DataFrame/array caching
```

**Requirements:**
- Python 3.12+
- lancedb >= 0.25.1
- sentence-transformers >= 5.1.1
- orjson >= 3.11.3
- pyarrow >= 21.0.0
- structlog >= 25.4.0

## Quick Start

### Basic Usage

```python
from memora import Memora, CacheConfig

# Initialize with development defaults
memora = Memora(CacheConfig.for_development())

# Check cache before expensive operation
result = memora.lookup(
    query="Analyze Q3 2024 sales performance",
    context={"agent": "sql_analyzer", "db": "production"}
)

if result.is_hit:
    print(f"✓ Cache hit! Similarity: {result.similarity:.3f}")
    print(f"  Age: {result.age_seconds}s")
    data = result.result
else:
    # Cache miss - execute expensive operation
    data = run_expensive_sql_query(...)
    
    # Store result for future queries
    memora.store(
        query="Analyze Q3 2024 sales performance",
        context={"agent": "sql_analyzer", "db": "production"},
        result=data
    )
```

### Decorator API

Automatic caching for any function:

```python
from memora import Memora, CacheConfig

memora = Memora(CacheConfig.for_production())

@memora.cached(context={"agent": "data_analyzer"})
def analyze_sales(query: str, year: int, region: str = "US"):
    # Expensive computation here
    return perform_analysis(query, year, region)

# First call executes the function
result1 = analyze_sales("Show Q3 revenue trends", year=2024)

# Similar query hits cache (semantic match)
result2 = analyze_sales("Q3 revenue analysis", year=2024)  # ✓ Cache hit!
```

**Context handling:**
- By default, all function parameters (except `query`) are included in cache context
- Use `extract_from_args=False` to ignore function parameters
- Use `exclude_from_context=["param1", "param2"]` to selectively exclude

## Configuration

### Development Preset

```python
config = CacheConfig.for_development()
# - In-memory storage (no persistence)
# - 5-minute TTL
# - 1K max entries
# - DEBUG logging (human-readable)
# - 5MB max payload size
```

### Production Preset

```python
config = CacheConfig.for_production(db_path="./cache.db")
# - Persistent disk storage
# - 1-hour TTL
# - 50K max entries
# - INFO logging (JSON structured)
# - Auto-indexing at 1000 entries
# - 10MB max payload size
```

### Environment Variables

Docker/Kubernetes-friendly configuration:

```bash
# Example .env or docker-compose.yml
MEMORA_DB_URI=/var/cache/memora
MEMORA_JSON_LOGS=true
MEMORA_LOG_LEVEL=INFO
MEMORA_MAX_ENTRIES=100000
MEMORA_TTL_SECONDS=3600
MEMORA_SIMILARITY_THRESHOLD=0.85
MEMORA_MAX_RESULT_SIZE_BYTES=10000000
```

```python
from memora import Memora, CacheConfig

# Reads all config from environment
memora = Memora(CacheConfig.from_env())
```

**Supported variables:**
- `MEMORA_MODEL_NAME` - Embedding model (default: paraphrase-multilingual-MiniLM-L12-v2)
- `MEMORA_SIMILARITY_THRESHOLD` - Match threshold 0.0-1.0 (default: 0.85)
- `MEMORA_DB_URI` - Storage path (default: memory://)
- `MEMORA_TABLE_NAME` - Table name (default: semantic_cache)
- `MEMORA_ENABLE_METRICS` - Track metrics (default: true)
- `MEMORA_TTL_SECONDS` - Expiration time in seconds (default: None)
- `MEMORA_LOG_LEVEL` - DEBUG/INFO/WARNING/ERROR (default: INFO)
- `MEMORA_JSON_LOGS` - Enable JSON logging (default: false)
- `MEMORA_MAX_ENTRIES` - Max cache size (default: 10000)
- `MEMORA_MAX_RESULT_SIZE_BYTES` - Max payload (default: 10MB)
- `MEMORA_EVICTION_POLICY` - fifo (default: fifo)
- `MEMORA_AUTO_CREATE_INDEX` - Auto-index (default: false)

### Custom Configuration

```python
config = CacheConfig(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",  # 384d, 50+ languages
    similarity_threshold=0.85,  # 0.75-0.80 for multilingual, 0.85-0.90 for English
    db_uri="./my_cache.db",  # or "memory://" for testing
    ttl_seconds=7200,  # 2 hours
    enable_metrics=True,
    json_logs=True,  # Structured logging for production
    auto_create_index=True,
    index_threshold_entries=1000,
    max_entries=50_000,
    max_result_size_bytes=10_000_000,  # 10MB
    eviction_policy="fifo",
)

memora = Memora(config)
```

## Advanced Features

### Health Checks for Production

Monitor cache health with built-in diagnostics:

```python
health = memora.health_check()

# Returns:
# {
#   "status": "healthy" | "unhealthy",
#   "checks": {
#     "embedding": {"ok": true, "error": null},
#     "database": {"ok": true, "error": null},
#     "error_rate": {"ok": true, "details": "Error rate: 0.5% (2/400)"}
#   },
#   "metrics": {
#     "total_entries": 1234,
#     "recent_errors": {"lookup": 2, "store": 0}
#   },
#   "timestamp": 1696512000000
# }

if health["status"] == "unhealthy":
    alert_ops_team(health)
```

**Kubernetes Integration:**
```yaml
livenessProbe:
  exec:
    command: ["python", "-c", "from memora import Memora; import sys; sys.exit(0 if Memora().health_check()['status'] == 'healthy' else 1)"]
  initialDelaySeconds: 30
  periodSeconds: 60
```

### Similarity Thresholds

Control cache strictness:

```python
# Strict: Only very similar queries match (recommended for critical operations)
result = memora.lookup(query, context, similarity_threshold=0.92)

# Balanced: Default for most use cases
result = memora.lookup(query, context, similarity_threshold=0.85)

# Relaxed: More flexible matching (use with caution)
result = memora.lookup(query, context, similarity_threshold=0.75)
```

**Guidelines:**
- **0.90-0.95**: Critical operations (financial, medical)
- **0.85-0.90**: General use (recommended)
- **0.75-0.85**: Exploratory/development
- **<0.75**: Too permissive (false positives)

### TTL and Cache Invalidation

```python
# Time-based expiration
config = CacheConfig(ttl_seconds=3600)  # 1 hour

# Manual invalidation by context
deleted = memora.invalidate(context={"agent": "sql", "db": "staging"})
print(f"Invalidated {deleted} entries")

# Invalidate entries older than X seconds
deleted = memora.invalidate(older_than_seconds=7200)  # Older than 2 hours

# Cleanup expired entries (respects TTL)
deleted = memora.cleanup_expired()
```

### Size Limits and Eviction

```python
config = CacheConfig(
    max_entries=10_000,  # Total cache capacity
    max_result_size_bytes=10_000_000,  # 10MB per result
    eviction_policy="fifo",  # First In First Out
)

# When max_entries is reached, oldest entries are automatically evicted
# Oversized payloads (>max_result_size_bytes) are rejected with warning
```

### Vector Indexing for Scale

For production workloads with >1K entries:

```python
memora = Memora(CacheConfig.for_production())

# Add many entries...
for i in range(10_000):
    memora.store(...)

# Create IVF-PQ index for faster search
memora.create_index(num_partitions=512)

# Check index status
stats = memora.get_index_stats()
# {'has_index': True, 'total_entries': 10000}
```

**Performance:**
- **Without index**: 10-50ms lookup (<1K entries)
- **With index**: 5-15ms lookup (>10K entries)

### Metrics and Observability

```python
config = CacheConfig(enable_metrics=True, json_logs=True)
memora = Memora(config)

# ... use cache ...

# Get comprehensive metrics
stats = memora.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"Total requests: {stats['total_requests']}")
print(f"Latency p95: {stats['lookup_latency_ms']['p95']:.1f}ms")
print(f"Latency p99: {stats['lookup_latency_ms']['p99']:.1f}ms")
print(f"Time saved: {stats['total_latency_saved_ms']/1000:.1f}s")
print(f"Errors: lookup={stats['errors']['lookup']}, store={stats['errors']['store']}")
print(f"Payload p95: {stats['result_size_bytes']['p95']} bytes")
```

**Sample output:**
```json
{
  "hits": 847,
  "misses": 153,
  "total_requests": 1000,
  "hit_rate": "84.70%",
  "total_latency_saved_ms": 169400.0,
  "avg_latency_saved_ms": 200.0,
  "lookup_latency_ms": {
    "p50": 8.3,
    "p95": 15.7,
    "p99": 23.1
  },
  "errors": {
    "lookup": 2,
    "store": 0
  },
  "result_size_bytes": {
    "p50": 2048,
    "p95": 524288,
    "p99": 2097152
  }
}
```

## Supported Data Types

Memora handles Python primitives and scientific computing structures:

| Type | Support | Notes |
|------|---------|-------|
| `str`, `int`, `float`, `bool`, `None` | ✅ Native | Direct orjson serialization |
| `dict`, `list`, `tuple` | ✅ Native | Arbitrarily nested |
| `pandas.DataFrame` | ✅ Full | Arrow IPC for large DataFrames (>10MB) |
| `pandas.Series` | ✅ Full | Index and metadata preserved |
| `polars.DataFrame` | ✅ Full | Arrow IPC for large DataFrames |
| `numpy.ndarray` | ✅ Full | dtype and shape preserved |
| Custom objects | ❌ | Must be JSON-serializable or implement `__dict__` |

```python
# All of these work out of the box:
memora.store(query, ctx, "simple string")
memora.store(query, ctx, {"nested": {"data": [1, 2, 3]}})
memora.store(query, ctx, pd.DataFrame(...))
memora.store(query, ctx, pl.DataFrame(...))
memora.store(query, ctx, np.array([1, 2, 3]))
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Your Application                │
│  ┌───────────────────────────────────┐  │
│  │  1. Check cache (lookup)          │  │
│  │  2. Execute if miss               │  │
│  │  3. Store result                  │  │
│  └───────────────────────────────────┘  │
└──────────────┬──────────────────────────┘
               │
               ▼
      ┌────────────────┐
      │     Memora     │
      │   Core Logic   │
      └────────┬───────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌──────────┐    ┌────────────────┐
│Embeddings│    │   LanceDB      │
│sentence- │    │  (Vector DB)   │
│transform │    │  + Arrow IPC   │
└──────────┘    └────────────────┘
```

**Components:**
- **Core**: Cache logic, context fingerprinting, TTL/eviction
- **Embeddings**: sentence-transformers (multilingual support)
- **Storage**: LanceDB (vector database) + Arrow IPC (large payloads)
- **Serialization**: orjson (fast) + custom handlers (DataFrames, numpy)

## Performance

Typical latencies on consumer hardware (M1/M2 Mac, AMD Ryzen):

| Operation | No Index (<1K entries) | With Index (>10K entries) |
|-----------|------------------------|---------------------------|
| Lookup    | 10-50ms               | 5-15ms                    |
| Store     | 5-10ms                | 8-12ms                    |
| Embedding | 20-50ms               | 20-50ms                   |

**Optimization tips:**
- Enable auto-indexing for >1K entries: `auto_create_index=True`
- Use GPU for embeddings (if available): model runs on CUDA automatically
- Increase `similarity_threshold` to reduce false positives
- Use persistent storage (`db_uri="./cache.db"`) to avoid cold starts

## Examples

See the `examples/` directory for complete runnable examples:

- **`basic_usage.py`** - Simple cache operations and patterns
- **`diagnostics.py`** - Health checks, metrics, and monitoring
- **`decorator_example.py`** - Function caching with `@cached`
- **`multiagent_pipeline.py`** - Real-world multi-agent workflow

## Testing

Memora includes comprehensive test coverage (155 tests passing):

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run with coverage report
pytest --cov=memora --cov-report=html

# Run concurrency tests
pytest tests/test_concurrency.py -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## Roadmap

### v0.1.0 (Current Release) ✅
- [x] Health check method with component diagnostics
- [x] Structured logging (JSON) with `structlog`
- [x] Environment variable configuration (`from_env()`)
- [x] Improved error handling and recovery
- [x] Size limits and FIFO eviction
- [x] 155 passing tests

### v0.2.0 (Planned - Q2 2025)
- [ ] Background cleanup scheduler (cron-like)
- [ ] LRU eviction policy
- [ ] Prometheus metrics exporter (`/metrics` endpoint)
- [ ] S3/GCS remote storage backends
- [ ] Multi-tenancy support (namespace isolation)
- [ ] Semantic invalidation (invalidate by query similarity)

### v0.3.0 (Future)
- [ ] Distributed caching (Redis protocol compatibility)
- [ ] GraphQL API for cache operations
- [ ] Custom embedding model support (Instructor, E5)
- [ ] Webhooks for cache events
- [ ] Admin dashboard (web UI)

## Community

- **Issues**: [GitHub Issues](https://github.com/demiotic/memora/issues)
- **Discussions**: [GitHub Discussions](https://github.com/demiotic/memora/discussions)

## Acknowledgments

Built with:
- [LanceDB](https://lancedb.com/) - Modern vector database
- [sentence-transformers](https://www.sbert.net/) - State-of-the-art embeddings
- [Apache Arrow](https://arrow.apache.org/) - Columnar in-memory format
- [structlog](https://www.structlog.org/) - Structured logging

---

**Status**: Production-ready for most use cases. Missing features (LRU, scheduler) are optional for many deployments.