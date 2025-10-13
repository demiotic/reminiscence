---
title: Core Operations
description: Reminiscence class methods for caching operations
---

Complete reference for the `Reminiscence` class and its core caching operations.

## Initialization

### `Reminiscence()`

Create a new cache instance.

```python
Reminiscence(
    config: Optional[ReminiscenceConfig] = None,
    embedder: Optional[EmbeddingModel] = None
)
```

**Parameters:**
- `config` (Optional[ReminiscenceConfig]): Configuration object. If None, loads from environment variables.
- `embedder` (Optional[EmbeddingModel]): Custom embedding model. If None, creates from config.

**Example:**
```python
from reminiscence import Reminiscence, ReminiscenceConfig

# Default configuration
cache = Reminiscence()

# Custom configuration
config = ReminiscenceConfig(
    similarity_threshold=0.85,
    max_entries=10000,
    db_uri="./cache.db"
)
cache = Reminiscence(config=config)

# Use as context manager (auto-cleanup)
with Reminiscence() as cache:
    # Use cache...
    pass  # Scheduler stopped automatically
```

---

## Cache Operations

### `lookup()`

Search for cached result by semantic similarity.

```python
lookup(
    query: MultiModalInput,
    context: Optional[Dict[str, Any]] = None,
    similarity_threshold: Optional[float] = None,
    mode: QueryMode = QueryMode.AUTO,
    track_metrics: bool = True
) -> LookupResult
```

**Parameters:**
- `query` (MultiModalInput): Query to search for
- `context` (Optional[Dict]): Context dict for exact matching
- `similarity_threshold` (Optional[float]): Override default threshold (0.0-1.0)
- `mode` (QueryMode): Matching strategy (AUTO, SEMANTIC, EXACT)
- `track_metrics` (bool): Whether to track metrics

**Returns:** `LookupResult` with hit status and cached data

**Example:**
```python
from reminiscence.types import MultiModalInput
from reminiscence import QueryMode

result = cache.lookup(
    query=MultiModalInput(text="What is AI?"),
    context={"model": "gpt-4"},
    similarity_threshold=0.85,
    mode=QueryMode.SEMANTIC
)

if result.is_hit:
    print(f"Cache hit! Similarity: {result.similarity:.3f}")
    print(f"Age: {result.age_seconds:.1f}s")
    data = result.result
else:
    print("Cache miss - need to compute result")
```

### `store()`

Store result in cache.

```python
store(
    query: MultiModalInput,
    context: Dict[str, Any],
    result: Any,
    metadata: Optional[Dict[str, Any]] = None,
    ttl_seconds: Optional[int] = None,
    context_threshold: Optional[float] = None,
    allow_errors: bool = False,
    mode: QueryMode = QueryMode.AUTO
) -> None
```

**Parameters:**
- `query` (MultiModalInput): Query being cached
- `context` (Dict): Context for exact matching
- `result` (Any): Result to cache (supports DataFrames, arrays, dicts, etc.)
- `metadata` (Optional[Dict]): Additional metadata
- `ttl_seconds` (Optional[int]): Entry-specific TTL (time-to-live)
- `context_threshold` (Optional[float]): Entry-specific similarity threshold
- `allow_errors` (bool): Whether to cache error results (default: False)
- `mode` (QueryMode): Matching strategy

**Example:**
```python
cache.store(
    query=MultiModalInput(text="What is AI?"),
    context={"model": "gpt-4", "temperature": 0.7},
    result="Artificial Intelligence is...",
    ttl_seconds=3600  # Expire after 1 hour
)

# Store DataFrames directly (Arrow format)
import pandas as pd
df = pd.DataFrame({"col": [1, 2, 3]})
cache.store(
    query=MultiModalInput(text="Get sales data"),
    context={"database": "prod"},
    result=df  # Stored as Arrow table - 10-100x compression!
)
```

### `check_availability()`

Check if cache entry is available without retrieving data.

```python
check_availability(
    query: MultiModalInput,
    context: Dict[str, Any],
    similarity_threshold: Optional[float] = None,
    mode: QueryMode = QueryMode.AUTO
) -> AvailabilityCheck
```

**Parameters:**
- `query` (MultiModalInput): Query to check
- `context` (Dict): Context for exact matching
- `similarity_threshold` (Optional[float]): Override default threshold
- `mode` (QueryMode): Matching strategy (AUTO, SEMANTIC, EXACT)

**Returns:** `AvailabilityCheck` with availability status

**Example:**
```python
from reminiscence.types import MultiModalInput

availability = cache.check_availability(
    query=MultiModalInput(text="What is Python?"),
    context={"model": "gpt-4"}
)

if availability.available:
    print("✓ Cache entry exists")
    print(f"  Age: {availability.age_seconds:.1f}s")
    print(f"  Similarity: {availability.similarity:.3f}")
    if availability.ttl_remaining_seconds:
        print(f"  TTL remaining: {availability.ttl_remaining_seconds:.0f}s")
else:
    print("✗ No cache entry found")
```

---

## Batch Operations

Process multiple queries or stores efficiently (3-5x faster than loops).

### `lookup_batch()`

Batch lookup for multiple queries.

```python
lookup_batch(
    requests: List[LookupRequest],
    track_metrics: bool = True
) -> List[LookupResult]
```

**Parameters:**
- `requests` (List[LookupRequest]): List of lookup requests
- `track_metrics` (bool): Whether to track metrics

**Returns:** List of `LookupResult` (same order as requests)

**Example:**
```python
from reminiscence import LookupRequest
from reminiscence.types import MultiModalInput

requests = [
    LookupRequest(
        query=MultiModalInput(text="What is AI?"),
        context={"model": "gpt-4"}
    ),
    LookupRequest(
        query=MultiModalInput(text="What is ML?"),
        context={"model": "gpt-4"}
    ),
]

results = cache.lookup_batch(requests)

for req, res in zip(requests, results):
    if res.is_hit:
        print(f"✓ {req.query.text[:30]} - Hit ({res.similarity:.2f})")
    else:
        print(f"✗ {req.query.text[:30]} - Miss")
```

### `store_batch()`

Batch store for multiple results.

```python
store_batch(
    requests: List[StoreRequest],
    allow_errors: bool = False,
    mode: QueryMode = QueryMode.AUTO
) -> None
```

**Parameters:**
- `requests` (List[StoreRequest]): List of store requests
- `allow_errors` (bool): Whether to cache errors
- `mode` (QueryMode): Matching strategy

**Example:**
```python
from reminiscence import StoreRequest

requests = [
    StoreRequest(
        query=MultiModalInput(text="What is AI?"),
        context={"model": "gpt-4"},
        result="AI is...",
        ttl_seconds=3600
    ),
    StoreRequest(
        query=MultiModalInput(text="What is ML?"),
        context={"model": "gpt-4"},
        result="ML is...",
        ttl_seconds=3600
    )
]

cache.store_batch(requests)
```

---

## Invalidation & Cleanup

### `invalidate()`

Invalidate cache entries by criteria.

```python
invalidate(
    query: Optional[MultiModalInput] = None,
    context: Optional[Dict[str, Any]] = None,
    older_than_seconds: Optional[float] = None
) -> int
```

**Parameters:**
- `query` (Optional[MultiModalInput]): Exact query to invalidate
- `context` (Optional[Dict]): Exact context to match
- `older_than_seconds` (Optional[float]): Age threshold in seconds

**Returns:** Number of entries invalidated

**Example:**
```python
# Invalidate specific entry
count = cache.invalidate(
    query=MultiModalInput(text="old query"),
    context={"model": "gpt-3"}
)
print(f"Invalidated {count} entries")

# Invalidate all entries older than 1 hour
count = cache.invalidate(older_than_seconds=3600)

# Invalidate all entries for a specific context
count = cache.invalidate(context={"database": "old_db"})
```

### `cleanup_expired()`

Remove expired entries based on TTL.

```python
cleanup_expired() -> int
```

**Returns:** Number of entries deleted

**Example:**
```python
deleted = cache.cleanup_expired()
print(f"Removed {deleted} expired entries")
```

### `clear()`

Clear all cache entries.

```python
clear() -> None
```

**Example:**
```python
cache.clear()
print("Cache cleared")
```

---

## Monitoring & Health

### `get_stats()`

Get cache statistics.

```python
get_stats() -> Dict[str, Any]
```

**Returns:** Dictionary with cache statistics:
- `cache_entries`: Total entries in cache
- `exact_entries`: Entries in exact table
- `semantic_entries`: Entries in semantic table
- `hits`: Number of cache hits
- `misses`: Number of cache misses
- `hit_rate`: Hit rate percentage (0.0-1.0)
- `index_created`: Whether vector index exists

**Example:**
```python
stats = cache.get_stats()

print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Total entries: {stats['cache_entries']}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")

if stats['index_created']:
    print("✓ Vector index active")
```

### `health_check()`

Perform health check on cache components.

```python
health_check() -> Dict[str, Any]
```

**Returns:** Dictionary with health status:
- `status`: "healthy" or "unhealthy"
- `components`: Dict of component statuses
- `message`: Human-readable status message

**Example:**
```python
health = cache.health_check()

if health["status"] == "healthy":
    print("✓ Cache is healthy")
    print(f"  Components: {health['components']}")
else:
    print(f"✗ Cache unhealthy: {health['message']}")
```

---

## Background Tasks

### `start_scheduler()`

Start background schedulers for cleanup and metrics export.

```python
start_scheduler(
    interval_seconds: Optional[int] = None,
    initial_delay_seconds: int = 60,
    metrics_export_interval_seconds: Optional[int] = None,
    metrics_initial_delay_seconds: int = 0
) -> None
```

**Parameters:**
- `interval_seconds` (Optional[int]): Cleanup interval in seconds (default: 3600)
- `initial_delay_seconds` (int): Delay before first cleanup (default: 60)
- `metrics_export_interval_seconds` (Optional[int]): Metrics export interval
- `metrics_initial_delay_seconds` (int): Delay before first metrics export

**Example:**
```python
# Start cleanup scheduler (every 30 minutes)
cache.start_scheduler(interval_seconds=1800)

# Start with metrics export (every 30 seconds)
cache.start_scheduler(
    interval_seconds=1800,
    metrics_export_interval_seconds=30
)
```

### `stop_scheduler()`

Stop all background schedulers.

```python
stop_scheduler(timeout: float = 5.0) -> None
```

**Parameters:**
- `timeout` (float): Maximum time to wait for schedulers to stop

**Example:**
```python
# Graceful shutdown
cache.stop_scheduler(timeout=10.0)
```

### `get_scheduler_stats()`

Get statistics for all running schedulers.

```python
get_scheduler_stats() -> Optional[Dict[str, Any]]
```

**Returns:** Dictionary with scheduler statistics or None if no schedulers running

**Example:**
```python
stats = cache.get_scheduler_stats()

if stats:
    print("Scheduler Statistics:")
    for name, sched_stats in stats.items():
        print(f"\n{name}:")
        print(f"  Running: {sched_stats['running']}")
        print(f"  Total runs: {sched_stats['total_runs']}")
        print(f"  Errors: {sched_stats['errors']}")
        if sched_stats.get('last_run_timestamp'):
            print(f"  Last run: {sched_stats['last_run_timestamp']}")
```

---

## Remote Access (gRPC & Flight)

### `start_grpc_server()`

Start gRPC server for remote cache access over the network.

```python
start_grpc_server(
    port: int = 8080,
    max_workers: int = 10,
    blocking: bool = False
) -> None
```

**Parameters:**
- `port` (int): Port to listen on (default: 8080)
- `max_workers` (int): Maximum concurrent request handlers (default: 10)
- `blocking` (bool): If True, blocks until server stops (default: False)

**Example:**
```python
# Start server in background
cache = Reminiscence()
cache.start_grpc_server(port=8080, max_workers=20)
# ... server running in background ...
cache.stop_grpc_server()

# Or blocking mode (for dedicated server processes)
cache.start_grpc_server(port=8080, blocking=True)  # Blocks until Ctrl+C
```

See [gRPC API Guide](/dual-plane/grpc-api/) for complete documentation.

### `stop_grpc_server()`

Stop running gRPC server.

```python
stop_grpc_server(grace: float = 5.0) -> None
```

**Parameters:**
- `grace` (float): Maximum time to wait for graceful shutdown (seconds)

**Example:**
```python
cache.stop_grpc_server(grace=10.0)
```

### `start_flight_server()`

Start Arrow Flight server for high-throughput bulk data operations.

```python
start_flight_server(
    port: int = 8081,
    host: str = "0.0.0.0",
    blocking: bool = False
) -> None
```

**Parameters:**
- `port` (int): Port to listen on (default: 8081)
- `host` (str): Host to bind to (default: "0.0.0.0")
- `blocking` (bool): If True, blocks until server stops (default: False)

**Example:**
```python
# Start Flight server for data plane operations
cache = Reminiscence()
cache.start_flight_server(port=8081)  # Non-blocking
# ... Flight server running in background ...
cache.stop_flight_server()

# Dual-plane architecture (control + data planes)
cache.start_grpc_server(port=8080)   # Control plane
cache.start_flight_server(port=8081)  # Data plane (bulk transfers)
```

See [Arrow Flight Data Plane](/dual-plane/flight-dataplane/) for complete documentation.

### `stop_flight_server()`

Stop running Arrow Flight server.

```python
stop_flight_server() -> None
```

**Example:**
```python
cache.stop_flight_server()
```

---

## Indexing

### `create_index()`

Create IVF-PQ vector index for faster semantic searches.

```python
create_index(
    num_partitions: int = 256,
    num_subvectors: Optional[int] = None
) -> None
```

**Parameters:**
- `num_partitions` (int): Number of IVF partitions (default: 256)
- `num_subvectors` (Optional[int]): Number of PQ subvectors (default: embedding_dim / 4)

**Example:**
```python
# Auto-indexing happens at 256+ entries
# Manual indexing for fine control
cache.create_index(num_partitions=512, num_subvectors=96)
```

### `get_index_stats()`

Get vector index statistics.

```python
get_index_stats() -> Dict[str, Any]
```

**Returns:** Dictionary with index statistics

**Example:**
```python
stats = cache.get_index_stats()
print(f"Has index: {stats['has_index']}")
print(f"Total entries: {stats['total_entries']}")
```

---

## Next Steps

- **[Decorators](/reference/api/decorators/)** - Automatic caching with `@cached`
- **[Configuration](/reference/api/configuration/)** - ReminiscenceConfig options
- **[Data Types](/reference/api/data-types/)** - Request/response models
- **[Examples](/examples/llm-apps/)** - Real-world usage patterns
