# Reminiscence

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**Semantic cache for LLMs and multi-agent systems**

Reminiscence eliminates redundant computations by matching queries semantically instead of exact strings. Perfect for LLM applications, RAG pipelines, and agent workflows.

```python
# These queries hit the same cache entry:
"Analyze Q3 sales data"
"Show me third quarter sales analysis"
"What were Q3 revenues?"
```

## Why semantic caching?

Traditional caches fail for AI systems because users express the same intent differently. Reminiscence uses **FastEmbed** with multilingual sentence transformers to recognize equivalent queries, reducing API costs and latency.

## Quick Start

```bash
pip install reminiscence
```

```python
from reminiscence import Reminiscence

cache = Reminiscence()

result = cache.lookup(
    query="Analyze Q3 2024 sales",
    context={"agent": "analyst", "db": "prod"}
)

if result.is_hit:
    print(f"Cache hit! Similarity: {result.similarity:.2f}")
    data = result.result
else:
    # Execute and cache - repite query y context
    data = "expensive operation"
    cache.store(
        query="Analyze Q3 2024 sales",
        context={"agent": "analyst", "db": "prod"},
        result=data
    )
```

### Decorator API

Automatic caching with hybrid matching (semantic + exact params):

```python
from reminiscence import Reminiscence

cache = Reminiscence()

@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return expensive_llm_call(prompt, model)

# Similar prompts with same model hit cache
call_llm("Explain quantum physics", "gpt-4")
call_llm("Can you explain quantum mechanics?", "gpt-4")  # Cache hit ✓

# Different model = cache miss
call_llm("Explain quantum physics", "claude-3")  # Executes

```

## Key Features

- 🎯 **Semantic matching** - FastEmbed + cosine similarity (multilingual support)
- 🔀 **Hybrid caching** - Semantic similarity + exact context matching
- 🏗️ **Production ready** - LRU/LFU/FIFO eviction, TTL, health checks
- 📊 **OpenTelemetry native** - Metrics, tracing, and spans out of the box
- 🔒 **Type safe** - Handles DataFrames, numpy arrays, nested dicts (10MB+)
- ⚡ **Zero config** - Works instantly, scales to 100K+ entries with auto-indexing
- 🔄 **Background tasks** - Automatic cleanup scheduler and metrics export
- 🌐 **gRPC API** - Remote access for microservices and distributed systems

## Remote Access with gRPC

For distributed systems and microservices, Reminiscence provides a production-ready gRPC API:

```bash
pip install reminiscence[grpc]
```

### Auto-Start Server (Configuration-Driven)

```python
from reminiscence import Reminiscence, ReminiscenceConfig

# Server auto-starts on init
config = ReminiscenceConfig(
    db_uri="./cache.db",
    grpc_enabled=True,       # Enable gRPC server
    grpc_port=50051,         # Default gRPC port
    grpc_max_workers=10,     # Concurrent request handlers
)

cache = Reminiscence(config)  # Server now running on port 50051!

# Or via environment variables:
# REMINISCENCE_GRPC_ENABLED=true
# REMINISCENCE_GRPC_PORT=50051
# REMINISCENCE_GRPC_MAX_WORKERS=10
cache = Reminiscence(ReminiscenceConfig.load())
```

### Manual Server Control

```python
# Start server programmatically
cache = Reminiscence()
cache.start_grpc_server(port=50051, max_workers=10)

# Use cache...

# Stop when done
cache.stop_grpc_server()
```

### Connect from Any Service

```python
from reminiscence.api.client import ReminiscenceClient

# Connect to remote cache
with ReminiscenceClient("localhost:50051") as client:
    # Same API as local cache
    result = client.lookup(query, context)

    if not result.is_hit:
        data = expensive_operation()
        client.store(query, context, data)
```

**Benefits:**
- **Shared cache** across multiple services (Python, Go, Rust, etc.)
- **Centralized caching** reduces duplicate LLM calls
- **Language-agnostic** via Protocol Buffers
- **Production-ready** with health checks and monitoring

See [gRPC API Guide](docs/src/content/docs/guides/grpc-api.md) and [Microservices Example](docs/src/content/docs/examples/grpc-microservices.md) for complete documentation.

## Configuration

### YAML Configuration (Recommended)

For production deployments, use YAML files for clean, version-controlled configuration:

**Create `reminiscence.yaml`:**
```yaml
# Production cache configuration
db_uri: ./cache.db
max_entries: 50000
eviction_policy: lru
ttl_seconds: 3600

similarity_threshold: 0.82
auto_create_index: true

grpc:
  enabled: true
  port: 8080
  max_workers: 50

enable_metrics: true
log_level: INFO

otel:
  enabled: true
  endpoint: http://otel-collector:4318/v1/metrics
  service_name: reminiscence-prod
```

**Load in your application:**
```python
from reminiscence import Reminiscence, ReminiscenceConfig

# Load from YAML (recommended)
config = ReminiscenceConfig.load_from_yaml("reminiscence.yaml")
cache = Reminiscence(config)

# Environment variables override YAML (12-factor app standard)
# REMINISCENCE_GRPC_PORT=9090 overrides yaml's grpc.port
config = ReminiscenceConfig.load_from_yaml("reminiscence.yaml", allow_env_override=True)
cache = Reminiscence(config)
```

See example configurations in `examples/`:
- `reminiscence-dev.yaml` - Development (in-memory, debug logging)
- `reminiscence-prod.yaml` - Production (persistent, OTEL enabled)
- `reminiscence.yaml` - Complete with all options documented

### Alternative Configuration Methods

```python
from reminiscence import Reminiscence, ReminiscenceConfig

# Development (in-memory, defaults)
cache = Reminiscence()

# Environment variables (Docker/Kubernetes)
cache = Reminiscence(ReminiscenceConfig.load())

# Direct instantiation (for testing)
config = ReminiscenceConfig(
    db_uri="./cache.db",
    ttl_seconds=3600,
    eviction_policy="lru",
    max_entries=50_000,
    auto_create_index=True
)
cache = Reminiscence(config)
```

## Background Tasks

Automatic cleanup and metrics export:

```python
cache = Reminiscence(ReminiscenceConfig(
    ttl_seconds=3600,
    otel_enabled=True
))

# Start background tasks
cache.start_scheduler(
    interval_seconds=1800,              # Cleanup every 30 min
    metrics_export_interval_seconds=60  # Export metrics every minute
)

# ... use cache ...

# Stop when done (or use context manager)
cache.stop_scheduler()
```

### Context Manager

```python
with Reminiscence() as cache:
    cache.start_scheduler()
    # ... use cache ...
    # Automatically stops scheduler on exit
```

## Use Cases

- **LLM applications** - Cache similar prompts to reduce API costs (OpenAI, Anthropic, etc.)
- **Multi-agent systems** - Share cache across agents with context isolation
- **RAG pipelines** - Cache retrieved documents, embeddings, and search results
- **Data analysis** - Cache expensive SQL queries, pandas transformations

## Observability

Built-in OpenTelemetry support for production monitoring:

```python
# Automatic metrics collection
config = ReminiscenceConfig(
    enable_metrics=True,
    otel_enabled=True
)
cache = Reminiscence(config)

# Get current stats
stats = cache.get_stats()
print(f"Cache entries: {stats['cache_entries']}")
print(f"Hit rate: {stats['hit_rate']}")
print(f"Schedulers: {stats.get('schedulers', {})}")
```

**Available metrics:**
- Cache hits/misses and hit rate
- Lookup and store latency
- Total entries and evictions
- Error counts by operation
- Scheduler execution stats

Compatible with **Prometheus**, **Grafana**, **Datadog**, **New Relic**, and any OTLP-compatible backend.

## Health Checks

Production-ready health monitoring:

```python
health = cache.health_check()

# Returns comprehensive status
{
    "status": "healthy",  # or "unhealthy"
    "checks": {
        "embedding": {"ok": true, "error": null},
        "database": {"ok": true, "error": null},
        "error_rate": {"ok": true, "details": "..."},
        "schedulers": {"ok": true, "details": "2/2 schedulers running"},
        "opentelemetry": {"ok": true, "details": "Enabled (...)"}
    },
    "metrics": {...},
    "timestamp": 1696512000000
}
```

## Requirements

- Python 3.9+
- Core: `lancedb`, `fastembed`, `orjson`, `pyarrow`, `structlog`
- Optional: `pandas`, `polars`, `numpy` (for DataFrame/array caching)

## Performance

Typical latencies on consumer hardware (M1/M2, AMD Ryzen):

- **Lookup**: 5-15ms (with index), 10-50ms (without)
- **Store**: 5-10ms
- **Embedding**: 20-50ms (cached in-memory after first use)

Scales to **100K+ entries** with automatic vector indexing (IVF-PQ).

## Concurrency Model

Reminiscence is optimized for **single-process** or **low-concurrency** scenarios:

### ✅ Recommended Use Cases

- **CLI tools and scripts** - Single-threaded execution
- **Jupyter notebooks** - Research and prototyping
- **ML inference pipelines** - Typically single-threaded or low concurrency
- **FastAPI/Flask apps** - Low to moderate traffic (< 10 concurrent workers)
- **Background jobs** - Celery, RQ, or similar task queues

### ⚠️ Concurrency Considerations

**Metrics and counters** are thread-safe with minimal performance overhead. **Eviction policy state** uses relaxed consistency for better performance - the `max_entries` limit is a soft limit that may be exceeded by ~5% under high concurrency before cleanup occurs.

**For high-concurrency web services** (>10 concurrent workers), consider:
- Using process-level isolation (one cache instance per worker process)
- Disabling metrics if not needed (`enable_metrics=False`)
- Using a distributed cache (Redis, Memcached) as primary store

### Technical Details

- **LanceDB operations** (add, search, delete) are thread-safe
- **Metrics** use `threading.RLock` for accurate tracking under concurrent access
- **Eviction** uses relaxed consistency - adds entries first, then evicts excess
- **Max entries** is a soft limit (allows ~5% overflow during concurrent writes)

This design prioritizes performance for typical ML/AI workflows over strict consistency guarantees needed for high-throughput web services.

## Development & Architecture

For contributors and those interested in the project's architecture:

- **[CLAUDE.md](CLAUDE.md)** - Development guide with commands, patterns, and testing
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and future roadmap
  - Current Python-based design (v0.6.0)
  - Planned Rust CLI (v0.7.0+) for standalone binary distribution
  - Optional Rust server (v1.0+) for high-throughput production deployments

The architecture document explains the evolution path from Python-first development to a production-ready infrastructure tool following patterns from Redis, PostgreSQL, and modern cloud-native projects.

## License

AGPL v3 - See [LICENSE](LICENSE)

---

### Built with

- **[LanceDB](https://lancedb.com/)** - Vector database for embeddings
- **[FastEmbed](https://github.com/qdrant/fastembed)** - Fast embedding generation (Qdrant)
- **[sentence-transformers](https://www.sbert.net/)** - Multilingual semantic models (paraphrase-multilingual-MiniLM-L12-v2)
- **[Apache Arrow](https://arrow.apache.org/)** - Columnar format for large payloads
- **[OpenTelemetry](https://opentelemetry.io/)** - Observability and distributed tracing
- **[structlog](https://www.structlog.org/)** - Structured logging for production