# Memora

**Semantic cache for multi-agent systems**

Memora is an **experimental** semantic caching library for AI agents, built on LanceDB and sentence-transformers. **Alpha stage** - API may change.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

> [!WARNING]
> **Alpha Release (v0.1.0) - Not Production Ready**
> 
> Memora is functional and tested, but missing critical features for production:
> - No automatic cleanup/compaction
> - No concurrency controls  
> - Limited observability (basic metrics only)
> - API may change in 0.2.x
>
> **Use for:** Prototypes, development, research, experimentation  
> **Avoid for:** Production systems, multi-tenant apps, large-scale deployments (>100k entries)
>
> See [Roadmap](#roadmap) for planned features.

## Why Memora?

Multi-agent systems often execute the same expensive operations repeatedly. Memora caches results based on semantic similarity rather than exact matches:

```python
# These queries hit the same cache entry:
"Analyze Q3 sales data"
"Show me sales analysis for Q3"
"What were the sales in the third quarter?"
```

## Features

- **Semantic matching**: Finds similar queries using embeddings (cosine similarity)
- **Context-aware**: Groups cache by execution context (agent, config, tools)
- **Embedded or remote**: Use in-memory, on-disk, or connect to remote LanceDB
- **Type-safe serialization**: Handles pandas/polars DataFrames, numpy arrays, nested dicts
- **Production-ready**: TTL, metrics, automatic indexing, cleanup
- **Zero-config decorators**: Cache functions transparently

## Quick Start

```bash
pip install lancedb sentence-transformers orjson
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
    """This function's results are automatically cached."""
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
# - Metrics enabled
```

### Production

```python
config = CacheConfig.for_production(db_path="./cache.db")
# - Persistent storage
# - 24h TTL
# - Auto-indexing at 1000 entries
# - 512 IVF partitions
```

### Custom

```python
config = CacheConfig(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold=0.80,
    db_uri="./my_cache.db",
    ttl_seconds=3600,  # 1 hour
    enable_metrics=True,
    auto_create_index=True,
    index_threshold_entries=500,
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
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Latency p95: {stats['lookup_latency_ms']['p95']:.1f}ms")
print(f"Total savings: {stats['total_latency_saved_ms']/1000:.1f}s")
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
pytest tests/test_serialization.py

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
- lancedb >= 0.3
- sentence-transformers >= 2.2
- orjson >= 3.9
- pyarrow >= 12.0

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

- [ ] Remote LanceDB support
- [ ] Custom embedding models
- [ ] Semantic invalidation (invalidate by query similarity)
- [ ] Distributed caching
- [ ] Prometheus metrics exporter