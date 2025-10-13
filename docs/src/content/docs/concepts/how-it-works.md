---
title: How It Works
description: Understanding Reminiscence's architecture and semantic caching approach
---

Reminiscence solves two fundamental problems that plague AI systems: **semantic matching** and **complex data storage**.

## What Makes Reminiscence Different

Unlike simple semantic caches (gptcache, upstash) that only handle JSON text responses, Reminiscence is built for modern AI applications that produce:

- **LLM outputs** (text, structured JSON, function calls)
- **Tabular data** (pandas/polars DataFrames with millions of rows)
- **NumPy arrays** (scientific computing, ML model outputs)
- **Multimodal content** (images, video, audio alongside text queries)

This is **why we use LanceDB and Apache Arrow** — to efficiently store both vector embeddings AND complex columnar/tabular data formats. Other semantic caches break when you try to cache a 10GB DataFrame or multimodal RAG results.

## The Core Problems

**Problem 1: Semantic Matching**

Traditional caching fails because users ask similar questions in different ways:

- "What's your refund policy?"
- "How do I get my money back?"
- "Can I return this product?"
- "Do you offer refunds?"

String-based caching treats these as four completely different queries, making four expensive LLM calls when one should suffice.

**Problem 2: Complex Data Types**

But semantic matching alone isn't enough. Modern AI applications cache:

- **RAG outputs**: Text answers + source documents + metadata DataFrames
- **Data analysis**: SQL results as pandas DataFrames (100K+ rows)
- **ML predictions**: NumPy arrays, confidence scores, feature vectors
- **Vision models**: Images, bounding boxes, segmentation masks, embeddings

Most semantic caches only handle simple JSON strings. They fail catastrophically when you cache:
```python
# DataFrame with 1M rows
result = pd.DataFrame({"col1": [...], "col2": [...]})

# NumPy array
embeddings = np.array([[0.1, 0.2, ...], ...])  # 10,000 x 384

# Multimodal RAG
{"answer": "...", "sources": df, "images": [bytes, bytes]}
```

**Reminiscence handles all of this natively using Arrow format and LanceDB's columnar storage.**

## How Semantic Caching Works

Instead of matching strings character-by-character, Reminiscence understands what queries *mean*. It converts text into mathematical representations (embeddings) that capture semantic similarity, then uses vector similarity search to find conceptually similar queries.

### The Three-Step Process

**Step 1: Understanding Meaning**

When you cache a query, Reminiscence converts it into a vector—a list of numbers that represents its meaning. Queries with similar meanings produce similar vectors, even if they use completely different words.

Think of it like GPS coordinates for ideas: "refund policy" and "money back guarantee" live in the same conceptual neighborhood, while "refund policy" and "shipping address" are far apart.

**Step 2: Finding Similar Queries**

When a new query arrives, Reminiscence:
1. Converts it to a vector
2. Searches the cache for vectors nearby in meaning-space
3. Returns the cached result if the similarity is high enough

This search happens incredibly fast (5-15ms) thanks to specialized vector indexing.

**Step 3: Context Isolation**

Here's the key innovation: Reminiscence uses **hybrid matching**. Queries match semantically (flexible), but context must match exactly (precise).

This means "analyze sales" will match "show me revenue" *only if* the context also matches—like same database, same date range, same user permissions. You get flexibility without sacrificing correctness.

## Why This Matters

**Cost Savings:** If 30% of your LLM queries are semantically similar to previous ones, you eliminate 30% of your API costs immediately. At scale, this is thousands of dollars saved.

**Speed:** Cache hits return in ~10ms vs 1-3 seconds for LLM API calls. Your application feels instant to users.

**Consistency:** Similar questions get consistent answers instead of LLM variations. This is crucial for production systems where users expect reliable behavior.

## Architecture Overview

Reminiscence is built in layers, each handling a specific responsibility:

### Public API Layer

The `Reminiscence` class is your main entry point. It provides simple methods like `lookup()`, `store()`, and `invalidate()`, plus the `@cached` decorator for automatic caching. This layer handles initialization, configuration, and lifecycle management.

### Operations Layer

Behind the scenes, specialized operation modules handle different concerns:
- **LookupOperations**: Finds semantically similar queries and applies context filtering
- **StorageOperations**: Manages cache writes, batch operations, and eviction
- **InvalidationOperations**: Handles pattern-based deletion and bulk invalidation
- **MaintenanceOperations**: Runs cleanup, collects statistics, and manages imports/exports

This separation means each operation can be optimized independently without affecting others.

### Component Layer

The real work happens here:

**Storage Backend (LanceDB + Arrow)**: This is Reminiscence's secret weapon. LanceDB was specifically chosen because it's one of the few databases that handles BOTH:
- **Vector similarity search** for semantic matching (embeddings)
- **Columnar storage** for tabular data (DataFrames, arrays)

LanceDB uses Apache Arrow under the hood, which means:
- Native pandas/polars DataFrame support (zero-copy reads)
- Efficient NumPy array storage
- Columnar compression (10-100x space savings for tabular data)
- Fast vector indexing (IVF-PQ) for sub-millisecond searches

When you cache a DataFrame with 1M rows, Reminiscence stores it in Arrow format — the same format data tools use internally. No expensive serialization/deserialization. No JSON bloat.

**Embeddings (FastEmbed)**: Converts text queries to vectors using pre-trained multilingual models. The default model supports 100+ languages and generates 384-dimensional embeddings in ~5-10ms.

**Serialization Pipeline**: Handles diverse result types with format-aware serialization:
- **Tabular data**: Stored as Arrow tables (pandas, polars, NumPy) — native format, no conversion overhead
- **JSON**: Serialized with orjson (2-3x faster than standard json)
- **Multimodal**: Images/video/audio stored as bytes or URIs alongside metadata
- **Compression**: Optional zstd/gzip (applied before encryption)
- **Encryption**: Optional Age encryption for sensitive cached data

The pipeline is smart: tabular data goes straight to Arrow format, JSON uses orjson, and media files are stored efficiently as blobs. Each format gets optimal treatment.

**Eviction Policies**: When the cache fills up, eviction policies decide what to remove. FIFO removes oldest entries, LRU removes least recently used, and LFU removes least frequently used. Each policy tracks different metadata to make smart decisions.

**Metrics & Observability**: Tracks hits, misses, latencies, and errors. Optionally exports to OpenTelemetry for production monitoring.

## Deep Dive: How a Lookup Works

Let's trace what happens when you call `cache.lookup()`:

### Phase 1: Embedding Generation

Your query text passes through the embedding model, which loads if it hasn't already (warm-up on first use takes ~50ms, subsequent calls are ~5-10ms). The model converts text to a 384-dimensional vector capturing semantic meaning.

Why 384 dimensions? It's a sweet spot: low enough to be fast, high enough to capture nuanced meanings. Fewer dimensions lose information, more dimensions slow down searches.

### Phase 2: Vector Search

LanceDB performs an approximate nearest neighbor (ANN) search. Instead of comparing your query to every cached entry, it uses an index structure (IVF-PQ) that narrows the search to likely candidates.

Think of it like a library: instead of checking every book, you go to the right section, then the right shelf, then scan a few books. The search returns the top N most similar entries above your similarity threshold (default: 0.80).

### Phase 3: Context Filtering

Here's where hybrid matching shines. The vector search returns semantically similar candidates, but Reminiscence filters them by context. Only entries with *exactly* matching context make it through.

Why exact context matching? Because context represents things that must be identical: database name, user ID, model version, date ranges. Matching these semantically would be dangerous—"database: prod" should never match "database: dev."

### Phase 4: Result Return

If a match survives both filters, you get a cache hit with the original result, similarity score, and age. If nothing matches, you get a miss and need to compute the result yourself.

## Deep Dive: How Storage Works

When you call `cache.store()`:

### Phase 1: Error Detection

First, Reminiscence checks if your result represents an error (exceptions, dicts with "error" keys, None values). By default, errors aren't cached because you don't want to serve cached errors when the real operation might succeed later. You can override this with `allow_errors=True`.

### Phase 2: Query Mode Detection

Depending on your query mode:
- **SEMANTIC**: Generates an embedding for vector search (default for natural language)
- **EXACT**: Uses high threshold (0.9999) for near-exact matching (SQL, API calls)
- **AUTO**: Detects query type automatically (SQL-like queries use exact, others use semantic)

This matters because some queries need precision (financial data, SQL queries) while others benefit from flexibility (natural language questions).

### Phase 3: Serialization Pipeline

Your result passes through a transformation pipeline:

1. **JSON Serialization**: Converts Python objects to JSON. Handles special cases like DataFrames (pandas, polars), NumPy arrays, and nested structures.

2. **Compression** (if enabled): Applies zstd or gzip compression. Compression happens *before* encryption because compressed data is smaller to encrypt and compressed encrypted data wouldn't save much.

3. **Encryption** (if enabled): Encrypts the compressed bytes using Age encryption. This protects sensitive cached data at rest.

The result is stored as bytes alongside the embedding, context hash, and metadata.

### Phase 4: Eviction Check

After storing, Reminiscence checks if the cache exceeds its size limit. If so, it uses the configured eviction policy to select victims. This happens *after* adding (relaxed eviction) to avoid race conditions in multi-threaded environments.

Why relaxed eviction? If two threads check "is cache full?" simultaneously, both might try to add entries, causing subtle bugs. Adding first, then evicting, prevents this.

### Phase 5: Index Management

If auto-indexing is enabled and you've crossed the threshold (default: 256 entries), Reminiscence creates or updates the vector index. This keeps searches fast as your cache grows. Without indexing, search time grows linearly with cache size. With indexing, it stays nearly constant.

## Query Modes Explained

### SEMANTIC Mode (Default)

Best for: Natural language queries, user-generated content, conversational AI

How it works: Pure semantic similarity matching with your configured threshold (default: 0.80). "What is machine learning?" will match "Explain ML" or "Tell me about machine learning."

Trade-off: More cache hits, but occasionally matches queries that are similar but not identical. Monitor similarity scores to tune your threshold.

### EXACT Mode

Best for: SQL queries, API calls with parameters, financial calculations, code execution

How it works: Uses extremely high threshold (0.9999) so only nearly-identical queries match. Typos or punctuation differences might still match, but semantic variations won't.

Trade-off: Fewer false positives, but also fewer cache hits. A small wording change means a miss.

### AUTO Mode

Best for: Mixed workloads, unknown query patterns, general-purpose caching

How it works: Tries exact match first (fast hash lookup), then falls back to semantic if exact fails. Detects SQL-like queries automatically and uses exact mode for them.

Trade-off: Best of both worlds, but slightly more complex behavior to reason about.

## Understanding Context

Context is how you control cache isolation. It's a dictionary of key-value pairs that must match exactly for a cache hit.

### What Belongs in Context?

**DO use context for:**
- Things that change the meaning of results: database name, user ID, model version
- Dimensions you want isolated: region, language, tenant, date range
- Parameters that affect output: temperature, max_tokens, top_k

**DON'T use context for:**
- High-cardinality values that never repeat: timestamps, request IDs, UUIDs
- Data that's part of the query itself: the actual question text
- Things that should match semantically: synonyms, translations

**Example:** Caching database queries

Good context: `{"database": "prod", "user_role": "admin"}`
Bad context: `{"timestamp": 1234567890, "request_id": "abc-def"}`

The good context lets you cache queries per database and role. The bad context means every request is unique, defeating the cache entirely.

## Tabular Data & Multimodal Support

This is where Reminiscence truly shines compared to other semantic caches.

### Caching DataFrames and Arrays

Most semantic caches fail when you try to cache large tabular data. Reminiscence handles it natively:

```python
import pandas as pd
import numpy as np

# Cache a massive DataFrame (this works!)
result = cache.lookup("Get Q3 sales by region", {"database": "analytics"})
if not result.is_hit:
    df = pd.read_sql("SELECT * FROM sales WHERE quarter = 'Q3'", conn)  # 1M rows
    cache.store("Get Q3 sales by region", {"database": "analytics"}, df)
    # Stored as Arrow format - efficient and fast

# Cache NumPy arrays
embeddings = np.random.rand(10000, 384)  # 10K embeddings
cache.store("Generate user embeddings", {"model": "ada-002"}, embeddings)
# Stored as Arrow arrays - no conversion overhead
```

**Why this works:**
- Arrow format is columnar and compressed (10-100x smaller than JSON)
- Zero-copy reads for pandas/polars (no deserialization overhead)
- LanceDB natively understands Arrow tables

**What other caches do:**
```python
# gptcache, upstash, etc. would do this:
df_json = df.to_json()  # MASSIVE string, loses types
cache.store(query, json.dumps(df_json))  # Double serialization!
# Later: json.loads() then pd.read_json() - slow and lossy
```

### Multimodal Queries and Results

Reminiscence supports multimodal inputs for vision/audio models:

```python
from reminiscence.types import MultiModalInput

# Vision model caching
result = cache.lookup(
    MultiModalInput(text="Describe this image", image=image_bytes),
    {"model": "gpt-4o"}
)

# Audio transcription caching
result = cache.lookup(
    MultiModalInput(text="Transcribe", audio=audio_bytes),
    {"model": "whisper-large"}
)

# Video analysis caching
result = cache.lookup(
    MultiModalInput(text="Detect objects", video=video_url),
    {"model": "yolo-v8"}
)
```

The text portion drives semantic matching (embeddings), while media bytes/URIs are stored alongside as Arrow blobs. This enables caching for:
- Vision models (object detection, segmentation, OCR)
- Audio models (transcription, classification, speaker diarization)
- Video analysis (frame extraction, action recognition)
- Multimodal RAG (text + images + tables combined)

## Performance Characteristics

Understanding performance helps you optimize for your use case:

**Lookup Latency:**
- Cache hit: 5-15ms (with index), 10-50ms (without index)
- Embedding generation: 5-10ms (after warm-up)
- Vector search: 1-5ms (with index), grows linearly without

**Storage Latency:**
- Single entry: 5-10ms
- Batch operations: 3-5x faster than loops (parallel embedding generation)

**Memory Usage:**
- ~1KB per cache entry (without compression)
- Embeddings: 1.5KB per entry (384 dimensions × 4 bytes/float)
- Index overhead: ~20% of embedding size

**Scaling Characteristics:**
- Linear growth without index (search time = entries × 0.01ms)
- Logarithmic with index (search time ≈ log(entries) × 0.1ms)
- Recommendation: Enable auto-indexing for caches over 256 entries

## When to Use Reminiscence

**Ideal for:**
- **Data analysis applications** caching SQL results as DataFrames
- **RAG pipelines** with tabular metadata and multimodal content
- **Multi-agent systems** where agents produce complex structured outputs
- **ML model serving** caching predictions as NumPy arrays
- **Vision/audio models** caching multimodal inputs and outputs
- **LLM applications** with repeated similar queries
- **Customer support bots** with common questions

**Choose Reminiscence over gptcache/upstash when you need:**
- DataFrames (pandas, polars) or NumPy arrays in results
- Multimodal caching (images, video, audio)
- Efficient storage of large tabular data (Arrow format)
- Production-grade features (encryption, compression, OTEL)

**Not ideal for:**
- Simple key-value caching (use Redis)
- Truly unique queries every time (no similarity to exploit)
- Real-time data that changes constantly (cache would always be stale)
- Systems with strict sub-5ms latency requirements (semantic search adds ~5-10ms)

## Next Steps

Now that you understand how Reminiscence works, explore specific topics:

- **[Semantic Matching](/concepts/semantic-matching/)** — Deep dive into embeddings, similarity thresholds, and tuning
- **[Hybrid Caching](/concepts/hybrid-caching/)** — Master the semantic + context matching approach
- **[Quick Start](/getting-started/quick-start/)** — Get hands-on with practical examples
- **[Configuration](/guides/configuration/)** — Learn all configuration options and their trade-offs
