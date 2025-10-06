# Memora
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

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

### ✅ Production-Ready

- **Semantic Matching** - Cosine similarity search with configurable thresholds (0.75-0.95)
- **Hybrid Caching** - Semantic similarity + exact context matching (agent, model, tools, etc.)
- **Type-Safe Serialization** - Handles pandas/polars DataFrames, numpy arrays, nested dicts, large payloads (10MB+)
- **Multiple Eviction Policies** - FIFO, LRU, LFU for intelligent cache management
- **TTL & Expiration** - Time-based expiration with automatic cleanup
- **Size Limits** - Configurable max entries and payload sizes
- **Health Checks** - Production monitoring with component diagnostics
- **Structured Logging** - JSON logs for observability (ELK, Datadog, Grafana)
- **Environment Config** - 12-factor app support (Docker/Kubernetes friendly)
- **Metrics Tracking** - Hit rate, latency percentiles (p50/p95/p99), error rates
- **Zero-Config Decorators** - Drop-in function caching with `@cached`
- **Vector Indexing** - IVF-PQ for fast search at scale (>1K entries)

### 🚧 Roadmap (v0.2.0+)

- [ ] Background cleanup scheduler
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
- Python 3.9+
- lancedb >= 0.25.1
- sentence-transformers >= 5.1.1
- orjson >= 3.11.3
- pyarrow >= 21.0.0
- structlog >= 25.4.0

## Quick Start

### Basic Usage

```python
from memora import Memora, CacheConfig

# Initialize with defaults
cache = Memora()

# Check cache before expensive operation
result = cache.lookup(
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
    cache.store(
        query="Analyze Q3 2024 sales performance",
        context={"agent": "sql_analyzer", "db": "production"},
        result=data
    )
```

### Decorator API

Automatic caching for any function:

```python
from memora import Memora

cache = Memora()

@cache.cached(
    query_param="prompt",
    strict_params=["model", "temperature"]
)
def call_llm(prompt: str, model: str, temperature: float):
    # Expensive LLM call
    return expensive_llm_call(prompt, model, temperature)

# First call executes the function
result1 = call_llm("Explain quantum physics", "gpt-4", 0.7)

# Similar query hits cache (semantic match + exact model/temp)
result2 = call_llm("Can you explain quantum mechanics?", "gpt-4", 0.7)  # ✓ Cache hit!

# Different model = cache miss (context mismatch)
result3 = call_llm("Explain quantum physics", "claude-3", 0.7)  # ✗ Cache miss
```

**Decorator parameters:**
- `query_param`: Parameter name for semantic matching (default: "query")
- `strict_params`: Parameters that must match exactly (model, agent_id, tools, etc.)
- `static_context`: Static context merged with function parameters
- `auto_strict`: Auto-detect non-string params as strict (default: False)

### Auto-Strict Mode

```python
@cache.cached(query_param="prompt", auto_strict=True)
def ask_llm(prompt: str, temperature: float, max_tokens: int):
    # temperature and max_tokens are auto-detected as strict
    # (non-string types)
    return llm_call(prompt, temperature, max_tokens)
```

## Configuration

### Development Setup

```python
config = CacheConfig(
    db_uri="memory://",  # In-memory (no persistence)
    ttl_seconds=300,  # 5 minutes
    max_entries=1000,
    log_level="DEBUG",
    eviction_policy="fifo"
)
cache = Memora(config)
```

### Production Setup

```python
config = CacheConfig(
    db_uri="./cache.lance",  # Persistent storage
    ttl_seconds=3600,  # 1 hour
    max_entries=50_000,
    log_level="INFO",
    json_logs=True,  # Structured logging
    eviction_policy="lru",  # Least Recently Used
    auto_create_index=True,
    index_threshold_entries=1000
)
cache = Memora(config)
```

### Environment Variables

Docker/Kubernetes-friendly configuration:

```bash
# .env or docker-compose.yml
MEMORA_DB_URI=/var/cache/memora
MEMORA_JSON_LOGS=true
MEMORA_LOG_LEVEL=INFO
MEMORA_MAX_ENTRIES=100000
MEMORA_TTL_SECONDS=3600
MEMORA_SIMILARITY_THRESHOLD=0.85
MEMORA_EVICTION_POLICY=lru
MEMORA_AUTO_CREATE_INDEX=true
```

```python
from memora import Memora, CacheConfig

# Reads all config from environment
cache = Memora(CacheConfig.load())
```

**Supported variables:**
- `MEMORA_MODEL_NAME` - Embedding model (default: paraphrase-multilingual-MiniLM-L12-v2)
- `MEMORA_USE_ONNX` - Use ONNX backend (default: true)
- `MEMORA_SIMILARITY_THRESHOLD` - Match threshold 0.0-1.0 (default: 0.85)
- `MEMORA_DB_URI` - Storage path (default: memory://)
- `MEMORA_TABLE_NAME` - Table name (default: semantic_cache)
- `MEMORA_ENABLE_METRICS` - Track metrics (default: true)
- `MEMORA_TTL_SECONDS` - Expiration time in seconds (default: None)
- `MEMORA_LOG_LEVEL` - DEBUG/INFO/WARNING/ERROR (default: INFO)
- `MEMORA_JSON_LOGS` - Enable JSON logging (default: false)
- `MEMORA_MAX_ENTRIES` - Max cache size (default: 1000)
- `MEMORA_EVICTION_POLICY` - fifo/lru/lfu (default: fifo)
- `MEMORA_AUTO_CREATE_INDEX` - Auto-index (default: false)
- `MEMORA_INDEX_THRESHOLD_ENTRIES` - Min entries for index (default: 256)

## Eviction Policies

Memora supports three eviction policies for when `max_entries` is reached:

### FIFO (First In First Out)

```python
config = CacheConfig(eviction_policy="fifo", max_entries=1000)
```

**Behavior:** Evicts the oldest entry regardless of usage patterns.

**Best for:**
- Simple, predictable behavior
- Time-sensitive data where old entries are less valuable
- Low-overhead scenarios

**Tradeoffs:**
- Doesn't consider access patterns
- May evict frequently-used entries

### LRU (Least Recently Used)

```python
config = CacheConfig(eviction_policy="lru", max_entries=1000)
```

**Behavior:** Evicts entries that haven't been accessed recently.

**Best for:**
- Data with temporal locality (recent queries are more likely to be repeated)
- Interactive applications with user sessions
- Workflows with "hot" data that changes over time

**Tradeoffs:**
- Slight memory overhead (tracks access timestamps)
- State lost on restart (resyncs from storage timestamps)

### LFU (Least Frequently Used)

```python
config = CacheConfig(eviction_policy="lfu", max_entries=1000)
```

**Behavior:** Evicts entries with the lowest access count.

**Best for:**
- Data with popularity patterns (some queries used much more than others)
- Long-running services with stable workloads
- Scenarios where "popular" data should stay cached

**Tradeoffs:**
- Tracks frequency counters in memory
- New entries start at frequency 0 (may be evicted quickly)
- State lost on restart

### Choosing a Policy

| Scenario | Recommended Policy | Reason |
|----------|-------------------|--------|
| Development/Testing | FIFO | Simplest, most predictable |
| User-facing apps | LRU | Recent queries are more relevant |
| Batch processing | FIFO | Chronological order matters |
| API caching | LRU or LFU | Depends on traffic patterns |
| Multi-agent systems | LRU | Agents revisit recent contexts |
| Long-running services | LFU | Popular queries stay cached |

**Note:** Eviction metadata (timestamps, counters) is kept in memory and resyncs on restart from storage timestamps where possible.

## Advanced Features

### Hybrid Matching: Semantic + Exact Context

Memora combines semantic similarity with exact context matching:

```python
# These match semantically BUT different contexts
cache.store("Analyze sales", {"agent": "A", "db": "prod"}, result1)
cache.store("Analyze sales", {"agent": "B", "db": "staging"}, result2)

# Query matches first entry (same context)
result = cache.lookup("Show sales analysis", {"agent": "A", "db": "prod"})
# ✓ Returns result1 (semantic match + exact context match)

# Query misses (different context)
result = cache.lookup("Show sales analysis", {"agent": "B", "db": "prod"})
# ✗ Miss (context mismatch even though query is similar)
```

**Use cases:**
- Multi-agent systems (separate cache per agent)
- A/B testing (separate cache per variant)
- Multi-tenancy (separate cache per tenant/database)
- Model versioning (separate cache per model)

### Health Checks for Production

Monitor cache health with built-in diagnostics:

```python
health = cache.health_check()

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
# Strict: Only very similar queries match
result = cache.lookup(query, context, similarity_threshold=0.92)

# Balanced: Default for most use cases
result = cache.lookup(query, context, similarity_threshold=0.85)

# Relaxed: More flexible matching
result = cache.lookup(query, context, similarity_threshold=0.75)
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
deleted = cache.invalidate(context={"agent": "sql", "db": "staging"})
print(f"Invalidated {deleted} entries")

# Invalidate entries older than X seconds
deleted = cache.invalidate(older_than_seconds=7200)

# Cleanup expired entries (respects TTL)
deleted = cache.cleanup_expired()
```

### Availability Checks

Check if data is available without retrieving it:

```python
from memora import AvailabilityCheck

# Lightweight check (doesn't fetch full result)
check = cache.check_availability(query, context)

if check.available:
    print(f"Data available (age: {check.age_seconds}s)")
    print(f"TTL remaining: {check.ttl_remaining_seconds}s")
    print(f"Similarity: {check.similarity}")
else:
    print("Data not in cache")
```

### Vector Indexing for Scale

For production workloads with >1K entries:

```python
cache = Memora(CacheConfig(auto_create_index=True))

# Add many entries...
for i in range(10_000):
    cache.store(...)

# Index is created automatically at threshold

# Or create manually
cache.create_index(num_partitions=512)

# Check index status
stats = cache.get_index_stats()
# {'has_index': True, 'total_entries': 10000}
```

**Performance:**
- **Without index**: 10-50ms lookup (<1K entries)
- **With index**: 5-15ms lookup (>10K entries)

### Background Cleanup Scheduler

Automatically clean expired entries in the background:
```python
from memora import Memora, CacheConfig

config = CacheConfig(
    ttl_seconds=3600,  # 1 hour TTL
    cleanup_interval_seconds=1800  # Cleanup every 30 minutes
)

cache = Memora(config)

# Start automatic cleanup
cache.start_scheduler()

# ... use cache normally ...

# Stop when done
cache.stop_scheduler()

### Metrics and Observability

```python
config = CacheConfig(enable_metrics=True, json_logs=True)
cache = Memora(config)

# ... use cache ...

# Get comprehensive metrics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"Total requests: {stats['total_requests']}")
print(f"Total entries: {stats['total_entries']}")
print(f"Eviction policy: {stats['eviction_policy']}")
print(f"Index created: {stats['index_created']}")
```

**Sample output:**
```json
{
  "total_entries": 1234,
  "max_entries": 10000,
  "eviction_policy": "lru",
  "threshold": 0.85,
  "embedding_dim": 384,
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "ttl_seconds": 3600,
  "storage": "./cache.lance",
  "index_created": true,
  "hits": 847,
  "misses": 153,
  "hit_rate": "84.70%"
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
| Custom objects | Partial | Must be JSON-serializable or implement `__dict__` |

```python
# All of these work out of the box:
cache.store(query, ctx, "simple string")
cache.store(query, ctx, {"nested": {"data": [1, 2, 3]}})
cache.store(query, ctx, pd.DataFrame(...))
cache.store(query, ctx, pl.DataFrame(...))
cache.store(query, ctx, np.array([1, 2, 3]))
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
      │  + Eviction    │
      └────────┬───────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌──────────┐    ┌────────────────┐
│Embeddings│    │   LanceDB      │
│sentence- │    │  (Vector DB)   │
│transform │    │  + Arrow IPC   │
│ + ONNX   │    │                │
└──────────┘    └────────────────┘
               │
               ▼
       ┌───────────────┐
       │   Eviction    │
       │  FIFO/LRU/LFU │
       └───────────────┘
```

**Components:**
- **Core**: Cache logic, hybrid matching, TTL/eviction coordination
- **Embeddings**: sentence-transformers with ONNX optimization
- **Storage**: LanceDB (vector database) + Arrow IPC (large payloads)
- **Eviction**: Pluggable policies (FIFO/LRU/LFU) with in-memory tracking
- **Serialization**: orjson (fast) + custom handlers (DataFrames, numpy)

## Performance

Typical latencies on consumer hardware (M1/M2 Mac, AMD Ryzen):

| Operation | No Index (<1K entries) | With Index (>10K entries) |
|-----------|------------------------|---------------------------|
| Lookup    | 10-50ms               | 5-15ms                    |
| Store     | 5-10ms                | 8-12ms                    |
| Embedding | 20-50ms (ONNX)        | 20-50ms (ONNX)            |

**Optimization tips:**
- Enable auto-indexing for >1K entries: `auto_create_index=True`
- Use ONNX for faster embeddings: `use_onnx=True` (default)
- Choose appropriate eviction policy (LRU for temporal, LFU for popularity)
- Increase `similarity_threshold` to reduce false positives
- Use persistent storage to avoid cold starts

## Examples

Complete runnable examples:

### Multi-Agent System

```python
from memora import Memora, CacheConfig

# Shared cache across agents with context isolation
cache = Memora(CacheConfig(
    eviction_policy="lru",
    max_entries=10_000
))

class SQLAgent:
    def __init__(self, cache):
        self.cache = cache
    
    def query(self, user_query: str, database: str):
        result = self.cache.lookup(
            user_query,
            context={"agent": "sql", "db": database}
        )
        
        if result.is_hit:
            return result.result
        
        # Execute SQL query
        data = self.execute_sql(user_query, database)
        
        # Cache for future
        self.cache.store(
            user_query,
            context={"agent": "sql", "db": database},
            result=data
        )
        
        return data

class AnalysisAgent:
    def __init__(self, cache):
        self.cache = cache
    
    @cache.cached(
        query_param="user_query",
        strict_params=["model", "tools"]
    )
    def analyze(self, user_query: str, model: str, tools: list):
        # Expensive analysis
        return perform_analysis(user_query, model, tools)
```

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_eviction_policies.py -v

# Run with coverage
pytest --cov=memora --cov-report=html
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Roadmap

### v0.1.0 (Current) ✅
- [x] Hybrid matching (semantic + exact context)
- [x] Multiple eviction policies (FIFO/LRU/LFU)
- [x] Health checks and diagnostics
- [x] Structured logging (JSON)
- [x] Environment configuration
- [x] Decorator API with auto-strict mode
- [x] ONNX optimization
- [x] Background cleanup scheduler

### v0.2.0 (Planned)
- [ ] Prometheus metrics exporter
- [ ] S3/GCS remote storage
- [ ] Distributed caching
- [ ] Semantic invalidation

## License

AGPL v3 - See LICENSE file

## Acknowledgments

Built with:
- [LanceDB](https://lancedb.com/) - Vector database
- [sentence-transformers](https://www.sbert.net/) - Embeddings
- [Apache Arrow](https://arrow.apache.org/) - Columnar format
- [structlog](https://www.structlog.org/) - Structured logging

---

**Production-ready for most use cases.** Missing features (scheduler, remote storage) are optional for many deployments.
