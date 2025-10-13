---
title: API Overview
description: Complete API reference for Reminiscence
---

Welcome to the Reminiscence API reference. This documentation is organized into focused sections for easy navigation.

## API Sections

The API documentation is split into the following sections:

### [Core Operations](/reference/api/core-operations/)
The main `Reminiscence` class with all caching operations:
- **Initialization**: Creating cache instances
- **Lookup & Store**: Finding and saving cached results
- **Batch Operations**: Processing multiple queries efficiently (3-5x faster)
- **Invalidation**: Managing cache entries
- **Monitoring**: Stats and health checks
- **Background Tasks**: Schedulers for cleanup and metrics

### [Decorators](/reference/api/decorators/)
Automatic caching with the `@cached` decorator:
- Basic usage with query and context parameters
- Auto-strict mode for automatic context detection
- Query mode selection (SEMANTIC, EXACT, AUTO)
- Advanced patterns and examples

### [Configuration](/reference/api/configuration/)
The `ReminiscenceConfig` class for customizing behavior:
- Core settings (thresholds, eviction, TTL)
- Storage configuration (database, indexing)
- Embeddings setup (models, backends)
- Security (compression, encryption)
- Observability (metrics, OpenTelemetry)

### [Data Types](/reference/api/data-types/)
Request/response models and type definitions:
- **MultiModalInput**: Text, image, video, audio queries
- **LookupResult**: Cache lookup responses
- **LookupRequest/StoreRequest**: Batch operation types
- **QueryMode**: Matching strategies (SEMANTIC, EXACT, AUTO)
- **EvictionPolicy**: Cache eviction strategies (LRU, LFU, FIFO)

### [gRPC API](/dual-plane/grpc-api/)
Remote cache access via gRPC protocol:
- **Server Setup**: Starting gRPC servers for network access
- **Python Client**: ReminiscenceClient for remote operations
- **All Operations**: Full API available over network
- **Production Patterns**: Load balancing, circuit breakers, monitoring

### [Arrow Flight Data Plane](/dual-plane/flight-dataplane/)
High-throughput bulk data streaming:
- **Dual-Plane Architecture**: Control (gRPC) + Data (Flight) planes
- **Zero-Copy Streaming**: 10x faster bulk data access
- **Analytics Integration**: pandas, polars, DuckDB support
- **Use Cases**: Export, migration, analytics dashboards

---

## Quick Reference

### Basic Usage

```python
from reminiscence import Reminiscence
from reminiscence.types import MultiModalInput

# Create cache
cache = Reminiscence()

# Lookup
result = cache.lookup(
    query=MultiModalInput(text="What is AI?"),
    context={"model": "gpt-4"}
)

if result.is_hit:
    data = result.result
else:
    # Compute and store
    data = expensive_operation()
    cache.store(
        query=MultiModalInput(text="What is AI?"),
        context={"model": "gpt-4"},
        result=data
    )
```

### Decorator Usage

```python
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return expensive_llm_call(prompt, model)

# Automatic caching
answer = call_llm("What is AI?", model="gpt-4")
```

### Configuration

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    similarity_threshold=0.85,
    max_entries=10000,
    db_uri="./cache.db",
    auto_create_index=True
)

cache = Reminiscence(config=config)
```

---

## Next Steps

- **[Core Operations](/reference/api/core-operations/)** - Detailed method reference
- **[Decorators](/reference/api/decorators/)** - Automatic caching patterns
- **[Configuration](/reference/api/configuration/)** - All config options
- **[Data Types](/reference/api/data-types/)** - Type definitions
- **[Examples](/examples/llm-apps/)** - Real-world usage patterns
