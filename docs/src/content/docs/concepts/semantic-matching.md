---
title: Semantic Matching
description: Understanding how Reminiscence matches queries by meaning using embeddings
---

Traditional caching breaks when users rephrase questions. "What is machine learning?" and "Explain ML concepts" are the same question, but string-based caches treat them as different—forcing redundant computation and wasting resources. Semantic matching solves this by understanding **what queries mean**, not just what they say.

## Why Semantic Matching Matters

The fundamental problem with traditional caching is brittleness. Consider a support bot:

**User A**: "What is machine learning?"
**User B**: "Explain machine learning"
**User C**: "Tell me about ML"
**User D**: "How does machine learning work?"

These are all asking the **same fundamental question**, but a traditional cache sees four different strings:
- 4 LLM API calls
- 4x the cost
- 4x the latency
- 4x the carbon footprint

**With semantic matching**, Reminiscence recognizes these queries as semantically similar. After the first query, the other three hit the cache:
- 1 LLM API call
- **75% cost savings**
- **75% latency reduction**
- Consistent answers for all users

This isn't just a nice-to-have—it's the difference between a cache hit rate of 10% (useless) and 40-60% (transformative).

## How Semantic Matching Works

Semantic matching uses **embeddings**—mathematical representations of text that capture meaning. Here's the key insight: similar meanings produce similar numbers, regardless of the exact words used.

### What Are Embeddings?

**Embeddings** are vector representations of text that capture semantic meaning. Similar meanings produce similar vectors.

```python
# Example embeddings (simplified to 3 dimensions for visualization)
embed("What is machine learning?")     # [0.8, 0.6, 0.1]
embed("Explain ML concepts")           # [0.82, 0.58, 0.12]  ← Similar!
embed("How to bake a cake?")           # [0.1, 0.2, 0.9]     ← Different!
```

### Vector Similarity

We measure similarity using **cosine similarity** (0.0 to 1.0):
- **1.0**: Identical meaning
- **0.8-0.95**: Very similar (typical cache hits)
- **0.5-0.8**: Somewhat related
- **< 0.5**: Different topics

```python
from reminiscence import Reminiscence

cache = Reminiscence()

# Store a result
cache.store(
    query="What is quantum computing?",
    context={},
    result="Quantum computing uses quantum mechanics..."
)

# Similar query matches
result = cache.lookup("Explain quantum computers")
print(result.similarity)  # 0.91 - very similar!
print(result.is_hit)      # True
```

## Similarity Thresholds

The **similarity threshold** controls how strictly queries must match.

### Default Threshold: 0.80

Reminiscence uses 0.80 by default, which provides a good balance:

```python
cache = Reminiscence()  # Uses default threshold=0.80

# These queries match (similarity > 0.80)
"What is machine learning?"
"Explain ML"
"Tell me about machine learning"
"How does machine learning work?"

# These don't match (similarity < 0.80)
"What is deep learning?"  # Related but different topic
"What is quantum computing?"  # Completely different
```

### Per-Lookup Threshold

Override the threshold for specific lookups:

```python
# Strict matching (only very similar queries)
result = cache.lookup(
    query="What is Python?",
    context={},
    similarity_threshold=0.95  # Very strict
)

# Loose matching (broader matches)
result = cache.lookup(
    query="What is Python?",
    context={},
    similarity_threshold=0.70  # More lenient
)
```

### Global Threshold Configuration

Set the default threshold via config or environment:

```python
from reminiscence import Reminiscence, ReminiscenceConfig

# Via config object
config = ReminiscenceConfig(similarity_threshold=0.85)
cache = Reminiscence(config=config)

# Or via environment variable
# REMINISCENCE_SIMILARITY_THRESHOLD=0.85
```

### Context-Specific Thresholds

Different thresholds for different contexts:

```python
import json
import os

# Set context-specific thresholds
os.environ["REMINISCENCE_CONTEXT_THRESHOLDS"] = json.dumps({
    "agent:sql": 0.95,      # SQL queries need exact matching
    "model:gpt-4": 0.85,    # Slightly stricter for GPT-4
    "agent:translation": 0.75  # Looser for translations
})

cache = Reminiscence()

# SQL query with strict matching
result = cache.lookup(
    query="SELECT * FROM users",
    context={"agent": "sql"}  # Uses 0.95 threshold
)

# Translation with loose matching
result = cache.lookup(
    query="translate hello to French",
    context={"agent": "translation"}  # Uses 0.75 threshold
)
```

## Choosing the Right Threshold

### High Thresholds (0.90-0.99)

**Use for:**
- SQL queries
- API calls with parameters
- Mathematical calculations
- Code execution

```python
# SQL caching with high threshold
@cache.cached(
    query="sql",
    context=["database"],
    similarity_threshold=0.95
)
def execute_sql(sql: str, database: str):
    return run_query(sql, database)
```

**Characteristics:**
- Very precise matching
- Low false positive rate
- May miss some valid similar queries

### Medium Thresholds (0.80-0.90)

**Use for:**
- General LLM queries
- Q&A systems
- Content generation
- Multi-agent workflows

```python
# LLM caching with default threshold
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
```

**Characteristics:**
- Balanced precision/recall
- Good for natural language
- Recommended starting point

### Low Thresholds (0.60-0.79)

**Use for:**
- Exploratory queries
- Broad topic matching
- Search suggestions
- Recommendation systems

```python
# Broad matching for recommendations
result = cache.lookup(
    query="Movies like Inception",
    context={"type": "recommendation"},
    similarity_threshold=0.70
)
```

**Characteristics:**
- Catches more matches
- Higher false positive rate
- Use with caution

## Embedding Models

Reminiscence uses **FastEmbed** with multiple model options:

### Default: Multilingual Model

```python
# Uses paraphrase-multilingual-MiniLM-L12-v2 by default
cache = Reminiscence()
# - 384 dimensions
# - 100+ languages
# - 5-10ms per embedding
```

### Custom Model Selection

```python
config = ReminiscenceConfig(
    model_name="BAAI/bge-small-en-v1.5"  # English-only, faster
)
cache = Reminiscence(config=config)
```

Available models (from `embeddings/models.yaml`):
- `paraphrase-multilingual-MiniLM-L12-v2` (default)
- `BAAI/bge-small-en-v1.5` (English-only, fast)
- `jinaai/jina-embeddings-v2-base-code` (code-specific)

### Model Warm-up

Pre-load the model on initialization:

```python
config = ReminiscenceConfig(warm_up_embedder=True)  # Default
cache = Reminiscence(config=config)
# First query: ~5ms (model already loaded)

# Without warm-up
config = ReminiscenceConfig(warm_up_embedder=False)
cache = Reminiscence(config=config)
# First query: ~50-100ms (loads model on first use)
```

## Batch Embeddings

Reminiscence optimizes batch operations by generating embeddings in parallel:

```python
from reminiscence import LookupRequest, MultiModalInput

# Batch lookup (embeddings generated together)
requests = [
    LookupRequest(
        query=MultiModalInput(text="What is AI?"),
        context={}
    ),
    LookupRequest(
        query=MultiModalInput(text="What is ML?"),
        context={}
    ),
    LookupRequest(
        query=MultiModalInput(text="What is DL?"),
        context={}
    ),
]

results = cache.lookup_batch(requests)  # 3-5x faster than loop
```

**Performance gain:**
- Single lookup: ~5-10ms per query
- Batch lookup: ~8-15ms for 10 queries (parallel embedding)

## Understanding Similarity Scores

### Score Interpretation

Real-world examples with their typical similarity scores:

```python
# Near-identical queries (0.95-1.0)
"What is machine learning?"
"What is machine learning"  # Missing punctuation
# Similarity: 0.98

# Very similar queries (0.85-0.95)
"What is machine learning?"
"Explain machine learning"
# Similarity: 0.91

"Analyze Q3 sales"
"Show me third quarter revenue"
# Similarity: 0.87

# Somewhat similar (0.70-0.85)
"What is machine learning?"
"What is deep learning?"  # Related concept
# Similarity: 0.78

# Different topics (< 0.70)
"What is machine learning?"
"How to bake a cake?"
# Similarity: 0.12
```

### Monitoring Similarity

Track similarity scores in production:

```python
result = cache.lookup(query, context)

if result.is_hit:
    print(f"Cache hit with similarity: {result.similarity:.3f}")

    # Log low-similarity hits for threshold tuning
    if result.similarity < 0.85:
        logger.warning(
            "Low similarity cache hit",
            query=query,
            matched_query=result.matched_query,
            similarity=result.similarity
        )
```

## Query Mode Strategies

Combine semantic matching with exact matching:

### AUTO Mode (Recommended)

Try exact match first, fallback to semantic:

```python
result = cache.lookup(
    query="What is Python?",
    context={},
    mode=QueryMode.AUTO  # Default
)
# 1. Try exact match (threshold=0.9999)
# 2. If no match, try semantic (threshold=0.80)
```

**Best for:**
- Mixed workloads (exact + fuzzy queries)
- Unknown query patterns
- General-purpose caching

### SEMANTIC Mode

Pure semantic matching:

```python
result = cache.lookup(
    query="What is Python?",
    context={},
    mode=QueryMode.SEMANTIC
)
```

**Best for:**
- Natural language queries
- User-generated content
- Variation in phrasing

### EXACT Mode

High-threshold matching for deterministic queries:

```python
result = cache.lookup(
    query="SELECT * FROM users WHERE id = 123",
    context={"db": "prod"},
    mode=QueryMode.EXACT  # threshold=0.9999
)
```

**Best for:**
- SQL queries
- API calls
- Code execution
- Financial calculations

## Multilingual Matching

The default model supports 100+ languages:

```python
cache = Reminiscence()

# English query
cache.store(
    query="What is artificial intelligence?",
    context={"lang": "en"},
    result="AI is..."
)

# Similar query in French
result = cache.lookup(
    query="Qu'est-ce que l'intelligence artificielle?",
    context={"lang": "en"}  # Won't match due to context
)

# Store French separately
cache.store(
    query="Qu'est-ce que l'intelligence artificielle?",
    context={"lang": "fr"},
    result="L'IA est..."
)
```

**Cross-language matching:**
- Enabled by multilingual model
- Queries in different languages can match
- Use context to enforce language isolation

## Debugging Semantic Matches

### Inspect Matched Query

```python
result = cache.lookup(query, context)

if result.is_hit:
    print(f"Original query: {query}")
    print(f"Matched query: {result.matched_query}")
    print(f"Similarity: {result.similarity:.3f}")
    print(f"Age: {result.age_seconds:.1f}s")
```

### Get All Entries for Analysis

```python
entries = cache.get_all_entries()

for entry in entries:
    print(f"Query: {entry['query']}")
    print(f"Context: {entry['context']}")
    print(f"Age: {entry['age_seconds']:.1f}s")
```

### Export for Analysis

```python
# Export cache to examine similarity patterns
cache.export_to_file("cache_dump.parquet")

# Analyze with pandas
import pandas as pd
df = pd.read_parquet("cache_dump.parquet")
print(df[["query", "context", "timestamp"]].head())
```

## Best Practices

1. **Start with default threshold (0.80)** and tune based on metrics
2. **Use context-specific thresholds** for different query types
3. **Monitor similarity scores** in production to detect issues
4. **Use EXACT mode** for deterministic queries (SQL, API calls)
5. **Use batch operations** for multiple queries (3-5x faster)
6. **Enable metrics** to track hit rates and optimize thresholds

## Next Steps

- [Hybrid Caching](/concepts/hybrid-caching/) - Combining semantic + context matching
- [Configuration Guide](/guides/configuration/) - All configuration options
- [Performance](/production/performance/) - Optimization techniques
