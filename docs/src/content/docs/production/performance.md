---
title: Performance
description: Optimization techniques and performance tuning for Reminiscence
---

This guide covers performance optimization techniques to get the most out of Reminiscence in production.

## Performance Overview

Typical latencies:
- **Lookup (no index)**: 10-50ms
- **Lookup (with index)**: 5-15ms
- **Store**: 5-10ms
- **Embedding generation**: 5-10ms per query
- **Batch operations**: 3-5x faster than loops

## Vector Indexing

The single biggest performance improvement comes from vector indexing.

### Enable Auto-Indexing

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    auto_create_index=True,
    index_threshold_entries=256
)

cache = Reminiscence(config=config)
```

### Manual Index Creation

```python
# Create index manually
cache.create_index(
    num_partitions=256,
    num_subvectors=96  # embedding_dim / 4
)

# Check index status
stats = cache.get_index_stats()
print(f"Index created: {stats['has_index']}")
```

### Index Performance Impact

```python
# Without index (1000 entries)
result = cache.lookup(query, context)  # ~40ms

# With index (1000 entries)
result = cache.lookup(query, context)  # ~8ms

# Speedup: 5x
```

**When to index:**
- Cache size ≥ 256 entries
- Lookup latency > 20ms
- High query volume

**Index tuning:**
```python
# Small cache (< 1K entries)
cache.create_index(num_partitions=128)

# Medium cache (1K-10K entries)
cache.create_index(num_partitions=256)  # Default

# Large cache (10K-100K entries)
cache.create_index(num_partitions=512)

# Very large cache (100K+ entries)
cache.create_index(num_partitions=1024)
```

## Batch Operations

Batch operations are 3-5x faster than loops due to parallelized embeddings.

### Batch Lookup

```python
from reminiscence import LookupRequest
from reminiscence.types import MultiModalInput

# ❌ Slow: Loop (10 queries @ 10ms each = 100ms)
results = []
for query in queries:
    result = cache.lookup(MultiModalInput(text=query), context)
    results.append(result)

# ✓ Fast: Batch (10 queries @ 15ms total = 15ms)
requests = [
    LookupRequest(query=MultiModalInput(text=q), context=context)
    for q in queries
]
results = cache.lookup_batch(requests)

# Speedup: 6.6x
```

### Batch Store

```python
from reminiscence import StoreRequest

# ❌ Slow: Loop
for query, result in zip(queries, results):
    cache.store(
        MultiModalInput(text=query),
        context,
        result
    )

# ✓ Fast: Batch
requests = [
    StoreRequest(
        query=MultiModalInput(text=q),
        context=context,
        result=r
    )
    for q, r in zip(queries, results)
]
cache.store_batch(requests)
```

### Decorator Batch Mode

Decorators use batch mode by default:

```python
@cache.cached(
    query="prompt",
    context=["model"],
    batch_mode=True  # Default - optimized internally
)
def call_llm(prompt: str, model: str):
    return llm(prompt, model)

# Single calls are optimized
result = call_llm("What is AI?", model="gpt-4")
```

## Embedding Optimization

### Model Selection

Choose the right embedding model for your use case:

```python
# Fastest: English-only model
config = ReminiscenceConfig(
    model_name="BAAI/bge-small-en-v1.5"
)
# Latency: ~3-5ms per embedding

# Balanced: Multilingual model (default)
config = ReminiscenceConfig()  # paraphrase-multilingual-MiniLM-L12-v2
# Latency: ~5-10ms per embedding

# Highest quality: Large model
config = ReminiscenceConfig(
    model_name="BAAI/bge-large-en-v1.5"
)
# Latency: ~15-25ms per embedding
```

### Warm-up Embedder

Pre-load the model on initialization:

```python
config = ReminiscenceConfig(
    warm_up_embedder=True  # Default
)

cache = Reminiscence(config=config)
# Initialization: +50-100ms
# First query: ~5ms (model already loaded)

# vs warm_up_embedder=False
# Initialization: fast
# First query: ~50-100ms (loads model on first use)
```

### Batch Size Tuning

```python
config = ReminiscenceConfig(
    embedding_batch_size=64  # Default: 32
)

# Larger batches = better GPU utilization
# But diminishing returns above 64-128
```

## Storage Optimization

### Memory vs Persistent Storage

```python
# In-memory (fastest)
config = ReminiscenceConfig(db_uri="memory://")
# Lookup: ~5-10ms
# Store: ~5ms

# Persistent (slightly slower, but durable)
config = ReminiscenceConfig(db_uri="./cache.db")
# Lookup: ~8-15ms
# Store: ~8-10ms

# SSD vs HDD impact
# SSD: ~10-15ms
# HDD: ~20-50ms
```

### Storage Location

```python
# Fast local SSD
config = ReminiscenceConfig(db_uri="/mnt/ssd/cache.db")

# Network storage (slower)
config = ReminiscenceConfig(db_uri="/mnt/nfs/cache.db")  # Not recommended
```

## Compression

Compression trades CPU for storage:

```python
from reminiscence import Reminiscence, ReminiscenceConfig

# No compression (fastest)
config = ReminiscenceConfig(compression_enabled=False)
# Store: ~5ms
# Size: 100%

# ZSTD compression (balanced)
config = ReminiscenceConfig(
    compression_enabled=True,
    compression_algorithm="zstd",
    compression_level=3  # Default
)
# Store: ~8-10ms
# Size: ~30-50%

# GZIP compression (slower)
config = ReminiscenceConfig(
    compression_enabled=True,
    compression_algorithm="gzip",
    compression_level=6
)
# Store: ~12-15ms
# Size: ~40-60%
```

**When to enable compression:**
- Large results (> 1KB)
- Storage constrained
- Network transfer cost high

**When to skip compression:**
- Small results (< 500 bytes)
- CPU constrained
- Ultra-low latency required

## Eviction Policy Performance

Different policies have different overhead:

```python
# FIFO (fastest)
config = ReminiscenceConfig(eviction_policy="fifo")
# Overhead: ~0.1ms per eviction

# LRU (medium)
config = ReminiscenceConfig(eviction_policy="lru")
# Overhead: ~0.5ms per eviction
# Updates access time on every lookup

# LFU (slowest)
config = ReminiscenceConfig(eviction_policy="lfu")
# Overhead: ~1-2ms per eviction
# Tracks frequency counters
```

**Recommendation:**
- Use **FIFO** for highest throughput
- Use **LRU** for better hit rates (worth the overhead)
- Use **LFU** only if access patterns are very skewed

## Query Mode Performance

```python
# EXACT mode (fastest for exact matches)
result = cache.lookup(query, context, mode=QueryMode.EXACT)
# Checks exact text match first (< 1ms)
# Falls back to vector search if needed

# SEMANTIC mode (consistent latency)
result = cache.lookup(query, context, mode=QueryMode.SEMANTIC)
# Always uses vector search

# AUTO mode (balanced)
result = cache.lookup(query, context, mode=QueryMode.AUTO)  # Default
# Tries exact match first, semantic fallback
```

## Concurrency and Threading

Reminiscence is thread-safe:

```python
import concurrent.futures
import threading

cache = Reminiscence()

def worker(query_id):
    query = f"Query {query_id}"
    result = cache.lookup(MultiModalInput(text=query), {})
    if not result.is_hit:
        cache.store(MultiModalInput(text=query), {}, f"Result {query_id}")
    return result

# Concurrent lookups
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(worker, i) for i in range(100)]
    results = [f.result() for f in futures]

# Metrics are thread-safe
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
```

## Reducing Lookup Latency

### 1. Enable Indexing

```python
config = ReminiscenceConfig(auto_create_index=True)
```

**Impact:** 3-5x speedup for caches with 256+ entries

### 2. Use Batch Operations

```python
results = cache.lookup_batch(requests)  # vs loop
```

**Impact:** 3-5x speedup for multiple queries

### 3. Increase Similarity Threshold

```python
config = ReminiscenceConfig(similarity_threshold=0.85)
```

**Impact:** Faster vector search (fewer candidates to check)

### 4. Reduce Context Size

```python
# ❌ Large context (slower to serialize/compare)
context = {"config": large_json_dict, "metadata": more_data}

# ✓ Minimal context
context = {"config_hash": hash(large_json_dict)}
```

**Impact:** Faster context matching

### 5. Use Faster Embedding Model

```python
config = ReminiscenceConfig(
    model_name="BAAI/bge-small-en-v1.5"  # vs default multilingual
)
```

**Impact:** 2-3x faster embeddings

## Reducing Storage Latency

### 1. Disable Encryption

```python
config = ReminiscenceConfig(encryption_enabled=False)
```

**Impact:** 2-5ms saved per store operation

### 2. Reduce Compression Level

```python
config = ReminiscenceConfig(
    compression_enabled=True,
    compression_level=1  # vs 3 (default)
)
```

**Impact:** ~2-3ms saved per store

### 3. Use In-Memory Storage

```python
config = ReminiscenceConfig(db_uri="memory://")
```

**Impact:** 2-5ms saved per operation

### 4. Disable Metrics

```python
config = ReminiscenceConfig(enable_metrics=False)
```

**Impact:** ~0.1-0.5ms saved (minimal, not recommended)

## Benchmarking

### Measure Lookup Performance

```python
import time

# Warm-up
for i in range(10):
    cache.lookup(MultiModalInput(text=f"warmup {i}"), {})

# Benchmark
latencies = []
for i in range(1000):
    start = time.perf_counter()
    result = cache.lookup(
        MultiModalInput(text=f"query {i % 100}"),
        {}
    )
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

# Stats
import statistics
print(f"Mean: {statistics.mean(latencies):.2f}ms")
print(f"Median: {statistics.median(latencies):.2f}ms")
print(f"P95: {sorted(latencies)[int(len(latencies) * 0.95)]:.2f}ms")
print(f"P99: {sorted(latencies)[int(len(latencies) * 0.99)]:.2f}ms")
```

### Measure Store Performance

```python
latencies = []
for i in range(1000):
    start = time.perf_counter()
    cache.store(
        MultiModalInput(text=f"query {i}"),
        {},
        f"result {i}"
    )
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

print(f"Store latency - Mean: {statistics.mean(latencies):.2f}ms")
```

### Compare Batch vs Loop

```python
queries = [f"query {i}" for i in range(100)]

# Loop
start = time.perf_counter()
for q in queries:
    cache.lookup(MultiModalInput(text=q), {})
loop_ms = (time.perf_counter() - start) * 1000

# Batch
requests = [
    LookupRequest(query=MultiModalInput(text=q), context={})
    for q in queries
]
start = time.perf_counter()
cache.lookup_batch(requests)
batch_ms = (time.perf_counter() - start) * 1000

print(f"Loop: {loop_ms:.1f}ms")
print(f"Batch: {batch_ms:.1f}ms")
print(f"Speedup: {loop_ms / batch_ms:.1f}x")
```

## Production Tuning

Recommended configuration for high-throughput production:

```python
config = ReminiscenceConfig(
    # Storage
    db_uri="./cache.db",  # Persistent storage
    max_entries=100000,   # Large cache

    # Performance
    auto_create_index=True,
    index_threshold_entries=256,
    warm_up_embedder=True,

    # Eviction
    eviction_policy="lru",  # Best hit rate

    # Compression (if results are large)
    compression_enabled=True,
    compression_algorithm="zstd",
    compression_level=3,

    # Encryption (if needed)
    encryption_enabled=False,  # Disable for performance

    # Metrics
    enable_metrics=True,
    otel_enabled=True,

    # Logging
    log_level="INFO",
    json_logs=True
)

cache = Reminiscence(config=config)
cache.start_scheduler()
```

## Performance Monitoring

Track key metrics:

```python
stats = cache.get_stats()

# Lookup latency
print(f"P95 lookup: {stats['lookup_latency_ms']['p95']}ms")

# Hit rate
print(f"Hit rate: {stats['hit_rate']}")

# Cache size
print(f"Entries: {stats['cache_entries']} / {stats['max_entries']}")

# Storage performance
print(f"P95 search: {stats['storage']['search_latency_ms']['p95']}ms")

# Embedding performance
print(f"P95 embedding: {stats['embedding']['latency_ms']['p95']}ms")
```

## Troubleshooting Slow Performance

### High Lookup Latency

**Symptoms:** P95 > 50ms

**Solutions:**
1. Enable vector indexing
2. Use faster embedding model
3. Increase similarity threshold
4. Reduce cache size (evict old entries)

### High Store Latency

**Symptoms:** Store > 20ms

**Solutions:**
1. Disable encryption
2. Reduce compression level
3. Use faster storage (SSD)
4. Reduce result size

### Low Hit Rate

**Symptoms:** Hit rate < 50%

**Solutions:**
1. Lower similarity threshold
2. Increase max_entries
3. Review query patterns
4. Check context granularity

### High Memory Usage

**Symptoms:** Memory growing unbounded

**Solutions:**
1. Set max_entries limit
2. Enable TTL
3. Use persistent storage
4. Enable compression

## Next Steps

- [Best Practices](/production/best-practices/) - Production deployment guide
- [OpenTelemetry](/production/opentelemetry/) - Monitoring and alerts
- [Health Checks](/production/health-checks/) - Health monitoring
