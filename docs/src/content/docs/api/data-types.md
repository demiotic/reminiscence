---
title: Data Types
description: Request/response models and type definitions
---

Complete reference for data types used in Reminiscence API operations.

## Input Types

### MultiModalInput

Unified input for text, image, video, and audio queries.

```python
@dataclass(frozen=True)
class MultiModalInput:
    text: Optional[str] = None
    image: Optional[bytes | str] = None
    video: Optional[bytes | str] = None
    audio: Optional[bytes | str] = None
    metadata: Optional[Dict[str, Any]] = None
```

**Fields:**
- `text` (Optional[str]): Text query for semantic matching
- `image` (Optional[bytes | str]): Image data (bytes) or URL (string)
- `video` (Optional[bytes | str]): Video data (bytes) or URL (string)
- `audio` (Optional[bytes | str]): Audio data (bytes) or URL (string)
- `metadata` (Optional[Dict]): Additional metadata

**Examples:**

```python
from reminiscence.types import MultiModalInput

# Text-only query (most common)
query = MultiModalInput(text="What is AI?")

# Image with descriptive text
with open("photo.jpg", "rb") as f:
    image_bytes = f.read()

query = MultiModalInput(
    text="Describe this image",
    image=image_bytes
)

# Video analysis (URL)
query = MultiModalInput(
    text="Summarize this video",
    video="s3://bucket/video.mp4"
)

# Audio transcription
with open("audio.mp3", "rb") as f:
    audio_bytes = f.read()

query = MultiModalInput(
    text="Transcribe audio",
    audio=audio_bytes
)

# With metadata
query = MultiModalInput(
    text="What is quantum computing?",
    metadata={"source": "user_query", "session_id": "abc123"}
)
```

---

## Response Types

### LookupResult

Result of a cache lookup operation.

```python
@dataclass
class LookupResult:
    hit: bool
    result: Optional[Any] = None
    similarity: Optional[float] = None
    matched_query: Optional[str] = None
    age_seconds: Optional[float] = None
    entry_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    ttl_remaining: Optional[float] = None
```

**Fields:**
- `hit` (bool): True if cache hit, False if miss
- `result` (Optional[Any]): Cached result (None on miss)
- `similarity` (Optional[float]): Similarity score (0.0-1.0, None on miss)
- `matched_query` (Optional[str]): Original query that matched (None on miss)
- `age_seconds` (Optional[float]): Age of cached entry in seconds (None on miss)
- `entry_id` (Optional[str]): Unique entry identifier (None on miss)
- `context` (Optional[Dict]): Context of matched entry (None on miss)
- `ttl_remaining` (Optional[float]): Seconds until expiration (None if no TTL)

**Properties:**
- `is_hit`: Alias for `hit` (returns True if cache hit)
- `is_miss`: Opposite of `hit` (returns True if cache miss)

**Examples:**

```python
result = cache.lookup(query, context)

# Check if hit
if result.is_hit:
    print("✓ Cache hit!")
    print(f"  Similarity: {result.similarity:.3f}")
    print(f"  Age: {result.age_seconds:.1f}s")
    print(f"  Matched: {result.matched_query}")

    # Use cached data
    data = result.result

    # Check TTL if set
    if result.ttl_remaining:
        print(f"  Expires in: {result.ttl_remaining:.0f}s")
else:
    print("✗ Cache miss - compute result")
    data = expensive_computation()
    cache.store(query, context, data)
```

---

## Batch Operation Types

### LookupRequest

Request object for batch lookup operations.

```python
@dataclass
class LookupRequest:
    query: MultiModalInput
    context: Optional[Dict[str, Any]] = None
    similarity_threshold: Optional[float] = None
    mode: QueryMode = QueryMode.AUTO
```

**Fields:**
- `query` (MultiModalInput): Query to search for
- `context` (Optional[Dict]): Context for exact matching
- `similarity_threshold` (Optional[float]): Override default threshold
- `mode` (QueryMode): Matching strategy

**Example:**

```python
from reminiscence import LookupRequest
from reminiscence.types import MultiModalInput

requests = [
    LookupRequest(
        query=MultiModalInput(text="What is AI?"),
        context={"model": "gpt-4"},
        similarity_threshold=0.85
    ),
    LookupRequest(
        query=MultiModalInput(text="What is ML?"),
        context={"model": "gpt-4"},
        mode=QueryMode.SEMANTIC
    ),
    LookupRequest(
        query=MultiModalInput(text="SELECT * FROM users"),
        context={"database": "prod"},
        mode=QueryMode.EXACT  # SQL needs exact matching
    )
]

results = cache.lookup_batch(requests)

for req, res in zip(requests, results):
    if res.is_hit:
        print(f"✓ {req.query.text[:30]} - Hit ({res.similarity:.2f})")
    else:
        print(f"✗ {req.query.text[:30]} - Miss")
```

### StoreRequest

Request object for batch store operations.

```python
@dataclass
class StoreRequest:
    query: MultiModalInput
    context: Dict[str, Any]
    result: Any
    metadata: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
    context_threshold: Optional[float] = None
```

**Fields:**
- `query` (MultiModalInput): Query being cached
- `context` (Dict): Context for exact matching
- `result` (Any): Result to cache
- `metadata` (Optional[Dict]): Additional metadata
- `ttl_seconds` (Optional[int]): Entry-specific TTL
- `context_threshold` (Optional[float]): Entry-specific similarity threshold

**Example:**

```python
from reminiscence import StoreRequest

requests = [
    StoreRequest(
        query=MultiModalInput(text="What is AI?"),
        context={"model": "gpt-4"},
        result="AI is...",
        ttl_seconds=3600,  # Expire in 1 hour
        metadata={"source": "openai", "tokens": 150}
    ),
    StoreRequest(
        query=MultiModalInput(text="What is ML?"),
        context={"model": "gpt-4"},
        result="ML is...",
        ttl_seconds=3600
    ),
    StoreRequest(
        query=MultiModalInput(text="Get sales data"),
        context={"database": "prod"},
        result=dataframe,  # DataFrames supported!
        ttl_seconds=300  # 5 minutes for data queries
    )
]

cache.store_batch(requests)
```

---

## Enums

### QueryMode

Query matching strategies.

```python
class QueryMode(str, Enum):
    SEMANTIC = "semantic"  # Semantic similarity search (fuzzy)
    EXACT = "exact"        # Exact text matching (threshold=0.9999)
    AUTO = "auto"          # Try exact first, fallback to semantic
```

**Usage:**

```python
from reminiscence import QueryMode

# Semantic mode - fuzzy matching for natural language
result = cache.lookup(
    query=MultiModalInput(text="What is AI?"),
    context={},
    mode=QueryMode.SEMANTIC
)

# Exact mode - strict matching for SQL, code
result = cache.lookup(
    query=MultiModalInput(text="SELECT * FROM users"),
    context={"database": "prod"},
    mode=QueryMode.EXACT
)

# Auto mode - tries exact first, falls back to semantic
result = cache.lookup(
    query=MultiModalInput(text="What is Python?"),
    context={},
    mode=QueryMode.AUTO  # Default
)
```

**When to use:**

| Mode | Use For | Threshold | Examples |
|------|---------|-----------|----------|
| **SEMANTIC** | Natural language, user queries | 0.80 (default) | "What is AI?", "Explain ML" |
| **EXACT** | SQL, code, deterministic queries | 0.9999 | "SELECT * FROM...", "def foo():" |
| **AUTO** | Mixed workloads, unknown patterns | Tries exact (0.9999) then semantic (0.80) | General-purpose caching |

### EvictionPolicy

Cache eviction strategies.

```python
class EvictionPolicy(str, Enum):
    LRU = "lru"    # Least Recently Used
    LFU = "lfu"    # Least Frequently Used
    FIFO = "fifo"  # First In First Out (default)
```

**Usage:**

```python
from reminiscence import ReminiscenceConfig

# FIFO - simple, predictable (default)
config = ReminiscenceConfig(
    max_entries=1000,
    eviction_policy="fifo"
)

# LRU - evict least recently accessed
config = ReminiscenceConfig(
    max_entries=1000,
    eviction_policy="lru"
)

# LFU - evict least frequently used
config = ReminiscenceConfig(
    max_entries=1000,
    eviction_policy="lfu"
)
```

**Comparison:**

| Policy | Evicts | Best For | Overhead |
|--------|--------|----------|----------|
| **FIFO** | Oldest entries | Simple, stable workloads | Minimal |
| **LRU** | Least recently accessed | Time-sensitive data | Low (tracks access time) |
| **LFU** | Least frequently used | Popularity-based caching | Medium (tracks access count) |

---

## Type Hints

For type checking, import types from `reminiscence`:

```python
from reminiscence import (
    Reminiscence,
    ReminiscenceConfig,
    LookupRequest,
    LookupResult,
    StoreRequest,
    QueryMode,
)
from reminiscence.types import MultiModalInput

def process_query(
    cache: Reminiscence,
    query: MultiModalInput,
    context: dict[str, Any]
) -> LookupResult:
    return cache.lookup(query, context)

def batch_process(
    cache: Reminiscence,
    requests: list[LookupRequest]
) -> list[LookupResult]:
    return cache.lookup_batch(requests)
```

---

## Next Steps

- **[Core Operations](/reference/api/core-operations/)** - Using these types in practice
- **[Decorators](/reference/api/decorators/)** - Automatic type handling
- **[Configuration](/reference/api/configuration/)** - Config options
- **[Data Types Guide](/reference/data-types/)** - Supported result formats (DataFrames, arrays, etc.)
