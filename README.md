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

# Check cache before expensive operation
result = cache.lookup(
    query="Analyze Q3 2024 sales",
    context={"agent": "analyst", "db": "prod"}
)

if result.is_hit:
    print(f"Cache hit! Similarity: {result.similarity:.2f}")
    data = result.result
else:
    # Execute and cache
    data = expensive_operation()
    cache.store(query, context, data)
```

### Decorator API

Automatic caching with hybrid matching (semantic + exact params):

```python
@cache.cached(query_param="prompt", strict_params=["model"])
def call_llm(prompt: str, model: str):
    return expensive_llm_call(prompt, model)

# Similar prompts with same model hit cache
call_llm("Explain quantum physics", "gpt-4")      # Executes
call_llm("Can you explain quantum mechanics?", "gpt-4")  # Cache hit ✓

# Different model = cache miss
call_llm("Explain quantum physics", "claude-3")   # Executes (different context)
```

### Auto-strict mode

Non-string parameters are automatically treated as strict:

```python
@cache.cached(query_param="prompt", auto_strict=True)
def ask_llm(prompt: str, temperature: float, max_tokens: int):
    # temperature and max_tokens auto-detected as strict
    return llm_call(prompt, temperature, max_tokens)
```

## Key Features

- 🎯 **Semantic matching** - FastEmbed + cosine similarity (multilingual support)
- 🔀 **Hybrid caching** - Semantic similarity + exact context matching
- 🏗️ **Production ready** - LRU/LFU/FIFO eviction, TTL, health checks
- 📊 **OpenTelemetry native** - Metrics, tracing, and spans out of the box
- 🔒 **Type safe** - Handles DataFrames, numpy arrays, nested dicts (10MB+)
- ⚡ **Zero config** - Works instantly, scales to 100K+ entries with auto-indexing
- 🔄 **Background tasks** - Automatic cleanup scheduler and metrics export

## Configuration

```python
from reminiscence import Reminiscence, ReminiscenceConfig

# Development (in-memory, defaults)
cache = Reminiscence()

# Production (persistent, optimized)
config = ReminiscenceConfig(
    db_uri="./cache.lance",
    ttl_seconds=3600,
    eviction_policy="lru",
    max_entries=50_000,
    auto_create_index=True
)
cache = Reminiscence(config)

# With OpenTelemetry
config = ReminiscenceConfig(
    otel_enabled=True,
    otel_service_name="my-service",
    otel_endpoint="http://localhost:4317"
)
cache = Reminiscence(config)

# Docker/Kubernetes (environment variables)
cache = Reminiscence(ReminiscenceConfig.load())
```

See [Configuration Guide](https://your-docs-site.github.io/reminiscence/configuration) for all environment variables.

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