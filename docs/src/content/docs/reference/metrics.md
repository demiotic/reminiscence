---
title: Metrics Reference
description: Complete reference for Reminiscence metrics and monitoring
---

Complete reference for all metrics tracked by Reminiscence.

## Cache Metrics

### Counters

#### hits
**Type:** Counter
**Description:** Total cache hits
**OpenTelemetry:** `reminiscence.cache.hits`

```python
stats = cache.get_stats()
print(f"Hits: {stats['hits']}")
```

#### misses
**Type:** Counter
**Description:** Total cache misses
**OpenTelemetry:** `reminiscence.cache.misses`

```python
print(f"Misses: {stats['misses']}")
```

#### total_requests
**Type:** Counter
**Description:** Total lookup requests (hits + misses)
**OpenTelemetry:** `reminiscence.cache.requests`

```python
print(f"Total requests: {stats['total_requests']}")
```

### Gauges

#### hit_rate
**Type:** Gauge (0.0-1.0)
**Description:** Cache hit rate percentage
**OpenTelemetry:** `reminiscence.cache.hit_rate`

```python
print(f"Hit rate: {stats['hit_rate']}")  # "75.3%"
```

**Thresholds:**
- ✓ Excellent: > 80%
- ✓ Good: 60-80%
- ⚠ Low: 40-60%
- ❌ Poor: < 40%

#### cache_entries
**Type:** Gauge
**Description:** Current number of cache entries
**OpenTelemetry:** `reminiscence.cache.entries`

```python
print(f"Entries: {stats['cache_entries']}/{stats['max_entries']}")
```

#### total_latency_saved_ms
**Type:** Counter
**Description:** Total latency saved by cache hits
**OpenTelemetry:** `reminiscence.cache.latency_saved_ms`

```python
print(f"Latency saved: {stats['total_latency_saved_ms']:.1f}ms")
```

### Histograms

#### lookup_latency_ms
**Type:** Histogram
**Description:** Lookup operation latency distribution
**OpenTelemetry:** `reminiscence.lookup.latency`

```python
latency = stats['lookup_latency_ms']
print(f"P50: {latency['p50']}ms")
print(f"P95: {latency['p95']}ms")
print(f"P99: {latency['p99']}ms")
```

**Targets:**
- ✓ Excellent: P95 < 20ms
- ✓ Good: P95 < 50ms
- ⚠ Acceptable: P95 < 100ms
- ❌ Slow: P95 > 100ms

#### result_size_bytes
**Type:** Histogram
**Description:** Cached result size distribution
**OpenTelemetry:** `reminiscence.result.size`

```python
sizes = stats['result_size_bytes']
print(f"P50 size: {sizes['p50']} bytes")
print(f"P95 size: {sizes['p95']} bytes")
print(f"P99 size: {sizes['p99']} bytes")
```

## Eviction Metrics

### Counters

#### evictions
**Type:** Counter
**Description:** Total evictions
**OpenTelemetry:** `reminiscence.eviction.total`

```python
eviction = stats['eviction']
print(f"Total evictions: {eviction['total_evictions']}")
```

#### eviction_rate
**Type:** Gauge
**Description:** Evictions per request
**OpenTelemetry:** `reminiscence.eviction.rate`

```python
print(f"Eviction rate: {eviction['eviction_rate']}")
```

**Thresholds:**
- ✓ Low: < 1%
- ⚠ Medium: 1-5%
- ❌ High: > 5%

#### evictions_by_policy
**Type:** Counter by policy
**Description:** Evictions grouped by policy
**OpenTelemetry:** `reminiscence.eviction.by_policy{policy="fifo|lru|lfu"}`

```python
by_policy = eviction['by_policy']
print(f"FIFO evictions: {by_policy.get('fifo', 0)}")
print(f"LRU evictions: {by_policy.get('lru', 0)}")
print(f"LFU evictions: {by_policy.get('lfu', 0)}")
```

### Histograms

#### evicted_entry_age_seconds
**Type:** Histogram
**Description:** Age of evicted entries
**OpenTelemetry:** `reminiscence.eviction.entry_age`

```python
age = eviction['evicted_entry_age_seconds']
print(f"P50 evicted age: {age['p50']:.1f}s")
print(f"P95 evicted age: {age['p95']:.1f}s")
```

#### lfu_evicted_frequencies
**Type:** Histogram (LFU only)
**Description:** Access frequency of evicted entries
**OpenTelemetry:** `reminiscence.eviction.lfu.frequency`

```python
if 'lfu_metrics' in eviction:
    freqs = eviction['lfu_metrics']['evicted_frequencies']
    print(f"P50 frequency: {freqs['p50']}")
```

#### lru_evicted_recency_seconds
**Type:** Histogram (LRU only)
**Description:** Time since last access for evicted entries
**OpenTelemetry:** `reminiscence.eviction.lru.recency`

```python
if 'lru_metrics' in eviction:
    recency = eviction['lru_metrics']['evicted_recency_seconds']
    print(f"P50 recency: {recency['p50']:.1f}s")
```

## Storage Metrics

### Counters

#### storage_searches
**Type:** Counter
**Description:** Total search operations
**OpenTelemetry:** `reminiscence.storage.searches`

```python
storage = stats['storage']
print(f"Total searches: {storage['total_searches']}")
```

#### storage_adds
**Type:** Counter
**Description:** Total add operations
**OpenTelemetry:** `reminiscence.storage.adds`

```python
print(f"Total adds: {storage['total_adds']}")
```

#### storage_errors
**Type:** Counter by operation
**Description:** Storage operation errors
**OpenTelemetry:** `reminiscence.storage.errors{operation="search|add"}`

```python
errors = storage['errors']
print(f"Search errors: {errors['search']}")
print(f"Add errors: {errors['add']}")
```

### Histograms

#### storage_search_latency_ms
**Type:** Histogram
**Description:** Storage search latency
**OpenTelemetry:** `reminiscence.storage.search_latency`

```python
search_lat = storage['search_latency_ms']
print(f"P95 search: {search_lat['p95']}ms")
```

**Targets:**
- ✓ Excellent: P95 < 10ms
- ✓ Good: P95 < 30ms
- ⚠ Acceptable: P95 < 50ms

#### storage_add_latency_ms
**Type:** Histogram
**Description:** Storage add latency
**OpenTelemetry:** `reminiscence.storage.add_latency`

```python
add_lat = storage['add_latency_ms']
print(f"P95 add: {add_lat['p95']}ms")
```

## Embedding Metrics

### Counters

#### embedding_generations
**Type:** Counter
**Description:** Total embeddings generated
**OpenTelemetry:** `reminiscence.embedding.generations`

```python
embedding = stats['embedding']
print(f"Total generations: {embedding['total_generations']}")
```

#### embedding_errors
**Type:** Counter
**Description:** Embedding generation errors
**OpenTelemetry:** `reminiscence.embedding.errors`

```python
print(f"Embedding errors: {embedding['errors']}")
```

### Histograms

#### embedding_latency_ms
**Type:** Histogram
**Description:** Embedding generation latency
**OpenTelemetry:** `reminiscence.embedding.latency`

```python
emb_lat = embedding['latency_ms']
print(f"P50: {emb_lat['p50']}ms")
print(f"P95: {emb_lat['p95']}ms")
```

**Targets:**
- ✓ Excellent: P95 < 10ms
- ✓ Good: P95 < 20ms
- ⚠ Acceptable: P95 < 50ms

## Scheduler Metrics

### Counters

#### scheduler_runs
**Type:** Counter
**Description:** Total scheduler runs
**OpenTelemetry:** `reminiscence.scheduler.runs{scheduler="cleanup|metrics"}`

```python
scheduler = stats['scheduler']
print(f"Total runs: {scheduler['total_runs']}")
```

#### scheduler_errors
**Type:** Counter
**Description:** Scheduler errors
**OpenTelemetry:** `reminiscence.scheduler.errors`

```python
print(f"Scheduler errors: {scheduler['errors']}")
```

### Histograms

#### scheduler_cleanup_latency_ms
**Type:** Histogram
**Description:** Cleanup operation latency
**OpenTelemetry:** `reminiscence.scheduler.cleanup_latency`

```python
cleanup_lat = scheduler['cleanup_latency_ms']
print(f"P95 cleanup: {cleanup_lat['p95']}ms")
```

## Error Metrics

### lookup_errors
**Type:** Counter
**Description:** Lookup operation errors

```python
errors = stats['errors']
print(f"Lookup errors: {errors['lookup']}")
```

### store_errors
**Type:** Counter
**Description:** Store operation errors

```python
print(f"Store errors: {errors['store']}")
```

## Metrics Report Example

Complete metrics report:

```python
stats = cache.get_stats()

{
    # Cache metrics
    "hits": 7500,
    "misses": 2500,
    "total_requests": 10000,
    "hit_rate": "75.00%",
    "total_latency_saved_ms": 450000.0,
    "avg_latency_saved_ms": 60.0,

    "lookup_latency_ms": {
        "p50": 8.5,
        "p95": 15.2,
        "p99": 25.8,
        "samples": 10000
    },

    "result_size_bytes": {
        "p50": 1024,
        "p95": 4096,
        "p99": 8192,
        "samples": 7500
    },

    "errors": {
        "lookup": 5,
        "store": 2
    },

    # Eviction metrics
    "eviction": {
        "total_evictions": 250,
        "eviction_rate": "2.50%",
        "by_policy": {
            "lru": 250
        },
        "evicted_entry_age_seconds": {
            "p50": 3600.0,
            "p95": 7200.0,
            "p99": 10800.0,
            "samples": 250
        },
        "lru_metrics": {
            "total_accesses": 10000,
            "evicted_recency_seconds": {
                "p50": 1800.0,
                "p95": 3600.0,
                "p99": 5400.0,
                "samples": 250
            }
        }
    },

    # Storage metrics
    "storage": {
        "total_searches": 10000,
        "total_adds": 2500,
        "search_latency_ms": {
            "p50": 5.2,
            "p95": 12.8,
            "p99": 20.5,
            "samples": 10000
        },
        "add_latency_ms": {
            "p50": 4.8,
            "p95": 10.2,
            "p99": 15.7,
            "samples": 2500
        },
        "errors": {
            "search": 3,
            "add": 1
        }
    },

    # Embedding metrics
    "embedding": {
        "total_generations": 2500,
        "latency_ms": {
            "p50": 6.5,
            "p95": 12.3,
            "p99": 18.9,
            "samples": 2500
        },
        "errors": 0
    },

    # Scheduler metrics
    "scheduler": {
        "total_runs": 24,
        "cleanup_latency_ms": {
            "p50": 150.5,
            "p95": 320.8,
            "p99": 450.2,
            "samples": 24
        },
        "errors": 0
    },

    # Cache metadata
    "cache_entries": 8500,
    "max_entries": 10000,
    "eviction_policy": "lru",
    "threshold": 0.80,
    "embedding_dim": 384,
    "model": "paraphrase-multilingual-MiniLM-L12-v2",
    "ttl_seconds": 3600,
    "storage": "./cache.db",
    "index_created": true
}
```

## Prometheus Queries

Common Prometheus queries for Reminiscence metrics:

### Hit Rate
```promql
reminiscence_cache_hit_rate
```

### Request Rate
```promql
rate(reminiscence_cache_requests[5m])
```

### P95 Lookup Latency
```promql
histogram_quantile(0.95,
  rate(reminiscence_lookup_latency_bucket[5m])
)
```

### Eviction Rate
```promql
rate(reminiscence_eviction_total[5m])
```

### Error Rate
```promql
rate(reminiscence_storage_errors[5m]) +
rate(reminiscence_embedding_errors[5m])
```

### Cache Utilization
```promql
reminiscence_cache_entries / reminiscence_max_entries
```

## Next Steps

- [OpenTelemetry Guide](/production/opentelemetry/) - Metrics export setup
- [Health Checks](/production/health-checks/) - Monitoring cache health
- [API Documentation](/reference/api/) - Complete API reference
