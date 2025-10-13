---
title: Hybrid Caching
description: Understanding how Reminiscence combines semantic similarity with exact context matching
---

The power of Reminiscence comes from **hybrid matching**: queries match semantically (flexible) while context matches exactly (precise). This gives you the flexibility of semantic search with the control of traditional caching.

## The Problem With Extremes

Traditional caching and pure semantic caching each have fatal flaws:

**Traditional Caching (Exact Matching):**
Problem: "What's your refund policy?" and "How do I get my money back?" are treated as completely different queries, even though they mean the same thing. Every slight variation requires a new computation.

**Pure Semantic Caching (No Context):**
Problem: "Analyze sales data" might return results for US sales when you needed EU sales, or production database results when you're working with staging. There's no way to isolate different scenarios.

**Hybrid caching solves both problems.**

## How Hybrid Matching Works

Reminiscence uses a two-stage filtering process:

**Stage 1: Semantic Similarity (Flexible)**
The query is converted to an embedding vector and compared against cached entries using cosine similarity. Queries with similar meanings score highly, regardless of exact wording. This stage casts a wide net to find conceptually similar queries.

**Stage 2: Context Matching (Exact)**
Among the semantically similar candidates, only entries with *exactly* matching context pass through. Context is compared key-by-key: `{"database": "prod", "user_id": 42}` only matches another entry with those exact values.

A cache hit requires both: semantic similarity above threshold AND exact context match.

## Why Exact Context Matching?

Consider what context typically contains:

- **Database identifiers**: `{"database": "prod"}` vs `{"database": "dev"}`
- **User permissions**: `{"role": "admin"}` vs `{"role": "viewer"}`
- **Model versions**: `{"model": "gpt-4"}` vs `{"model": "gpt-3.5"}`
- **Date ranges**: `{"start": "2024-01-01", "end": "2024-12-31"}`

These cannot be fuzzy-matched. Returning production data when someone requests development data would be catastrophic. Serving admin-level results to viewers would be a security breach.

**Exact matching for context ensures correctness while semantic matching for queries ensures flexibility.**

## Context Design Principles

### What Belongs in Context

**Use context for dimensions that must be isolated:**

- System identifiers (database, table, schema, service)
- User attributes (user_id, tenant_id, organization_id, role)
- Model parameters (model_name, version, temperature, max_tokens)
- Time dimensions (date, month, quarter, year)
- Feature flags (enabled_features, experiment_group)
- Environment (production, staging, development)

These define "separate worlds" where the same query means different things.

### What Doesn't Belong in Context

**Avoid high-cardinality unique values:**

- Request IDs, trace IDs, session IDs (every value is unique)
- Timestamps (millisecond precision = every request different)
- IP addresses (unless you genuinely need per-IP caching)
- UUIDs, random tokens, nonces

These guarantee cache misses because no two requests will have matching context.

**Avoid data that should match semantically:**

- Natural language variants (synonyms, translations)
- Formatting differences ("2024-01-01" vs "January 1, 2024")
- Capitalization, punctuation variations

If these belong anywhere, they belong in the query itself where semantic matching can handle them.

## Real-World Example: Multi-Tenant RAG System

Imagine a RAG system serving multiple customers, each with their own document collections:

```python
@cache.cached(
    query="user_question",
    context=["tenant_id", "collection", "model", "top_k"]
)
def rag_answer(
    user_question: str,
    tenant_id: str,
    collection: str,
    model: str,
    top_k: int
):
    # Retrieve documents from tenant's collection
    docs = vector_db.search(user_question, collection=f"tenant_{tenant_id}_{collection}")

    # Generate answer
    answer = llm_generate(docs, user_question, model)

    return answer
```

**What happens:**

User from Tenant A asks: "What are our Q3 results?"
User from Tenant B asks: "What are our Q3 results?"

- **Semantic matching**: Both queries are nearly identical, high similarity score
- **Context matching**: `tenant_id` differs, so they don't match
- **Result**: Each tenant gets their own cached results, no data leakage

Later, Tenant A asks: "Show me third quarter performance"

- **Semantic matching**: Similar to "Q3 results", scores above threshold
- **Context matching**: Same tenant_id, collection, model, top_k
- **Result**: Cache hit! Returns previously computed Q3 results.

## Context Patterns

### Pattern 1: User Isolation

```python
context = {
    "user_id": user_id,
    "tenant_id": tenant_id
}
```

Ensures users never see each other's cached data. Essential for multi-tenant systems.

### Pattern 2: Model Versioning

```python
context = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000
}
```

Different model configurations get separate caches. Important because GPT-4 and GPT-3.5 produce different results for the same query.

### Pattern 3: Time-Based Partitioning

```python
context = {
    "date_range": "2024-Q1",
    "report_type": "sales"
}
```

Cache results per time period. Q1 sales stay separate from Q2 sales, even for similar queries.

### Pattern 4: Environment Separation

```python
context = {
    "environment": os.getenv("ENV", "development"),
    "database": db_name
}
```

Development, staging, and production caches don't interfere. Prevents serving stale dev data in production.

### Pattern 5: A/B Testing

```python
context = {
    "experiment_group": "control" | "treatment",
    "feature_flags": json.dumps(sorted(active_flags))
}
```

Different experiment groups get separate caches, ensuring treatment effects aren't contaminated by control group results.

## Context Best Practices

### DO: Keep Context Minimal

Only include fields that genuinely create different meanings:

```python
# Good - essential context only
context = {"user_id": user_id, "model": model}

# Bad - too many fields = low hit rate
context = {
    "user_id": user_id,
    "session_id": session_id,  # Unique every session
    "request_id": request_id,  # Unique every request
    "timestamp": time.time(),  # Never matches
    "ip_address": request.ip
}
```

### DO: Use Static Context for Constants

Add service-wide context without polluting function signatures:

```python
@cache.cached(
    query="request",
    context=["user_id"],
    static_context={"service": "api_v2", "region": "us-east-1"}
)
def handle_request(request: str, user_id: int):
    return process(request)

# Stored context: {"user_id": 42, "service": "api_v2", "region": "us-east-1"}
```

### DON'T: Put Variable Data in Context

Avoid timestamp-like values that never repeat:

```python
# Bad - timestamp defeats caching
@cache.cached(query="q", context=["timestamp"])
def process(q: str, timestamp: float):
    return work(q)

# Every call has different timestamp = 0% hit rate
```

Use TTL instead:

```python
# Good - TTL expires old results
cache.store(
    query="Get stock price AAPL",
    context={"symbol": "AAPL"},  # No timestamp
    result=price_data,
    ttl_seconds=300  # Expires after 5 minutes
)
```

### DON'T: Overuse Context

Too many context dimensions = combinatorial explosion:

```python
# Bad - 3 databases × 10 users × 2 models × 4 quarters = 240 separate caches
context = {"db": db, "user_id": user, "model": model, "quarter": qtr}

# Good - only essential dimensions
context = {"user_id": user}  # Just 10 separate caches
```

## Context Serialization

Complex values are automatically serialized for exact matching:

**Lists**: `["active", "verified"]` becomes JSON, order matters
**Dicts**: `{"sort": "asc", "limit": 10}` becomes sorted JSON
**Nested structures**: Recursively serialized

Two contexts match only if serialized forms are identical:

```python
context1 = {"filters": ["active", "verified"]}
context2 = {"filters": ["verified", "active"]}  # Different order
# These DON'T match (order differs)

context3 = {"filters": ["active", "verified"]}  # Same order
# context1 and context3 match
```

**Tip**: Use sorted lists or canonicalize structures to avoid order-dependent misses.

## Context Thresholds

Different contexts can have different similarity thresholds:

```python
os.environ["REMINISCENCE_CONTEXT_THRESHOLDS"] = json.dumps({
    "agent:sql": 0.95,      # SQL needs high precision
    "agent:translation": 0.75  # Translations can be looser
})

# SQL queries: only very similar queries match
cache.lookup("SELECT * FROM users", {"agent": "sql"})

# Translations: broader matching acceptable
cache.lookup("translate hello", {"agent": "translation"})
```

This per-context tuning lets you balance precision and recall based on use case.

## Debugging Hybrid Matches

When debugging cache behavior, check both stages:

```python
result = cache.lookup(query, context)

if result.is_hit:
    print(f"✓ Semantic match: {result.similarity:.3f}")
    print(f"✓ Context match: {result.context == context}")
    print(f"  Matched query: {result.matched_query}")
    print(f"  Your query: {query}")
else:
    # Check if semantic matching found anything
    entries = cache.get_all_entries()
    similar = [e for e in entries if semantic_similarity(query, e['query']) > 0.80]

    if similar:
        print(f"✗ Found {len(similar)} semantically similar entries")
        print(f"✗ But none with matching context: {context}")
    else:
        print("✗ No semantically similar entries found")
```

## Performance Implications

**Context cardinality affects hit rate:**

- Low cardinality (few unique context values) = higher hit rate
- High cardinality (many unique context values) = lower hit rate

**Example:**

```python
# 2 databases × 3 models = 6 context combinations
context = {"database": db, "model": model}  # Moderate hit rate

# 1000 users × 2 databases × 3 models = 6000 combinations
context = {"user_id": user, "database": db, "model": model}  # Lower hit rate
```

**Context size affects storage:**

Large context dicts increase storage overhead:

```python
# Small context: ~50 bytes
context = {"model": "gpt-4"}

# Large context: ~1KB+
context = {
    "config": huge_json_blob,
    "metadata": more_data,
    "parameters": even_more
}
```

Prefer small context by hashing large values:

```python
# Better: hash large config
context = {
    "config_hash": hashlib.sha256(json.dumps(config).encode()).hexdigest()[:16]
}
```

## Trade-offs Summary

| Approach | Query Matching | Context Matching | When to Use |
|----------|---------------|------------------|-------------|
| **Traditional** | Exact | Exact | Deterministic queries, exact reproducibility needed |
| **Pure Semantic** | Semantic | None | Single-user systems, no isolation needs |
| **Hybrid (Reminiscence)** | Semantic | Exact | Multi-tenant, multi-model, production systems |

Hybrid caching gives you flexibility where it's safe (queries) and precision where it's critical (context).

## Next Steps

- **[Semantic Matching](/concepts/semantic-matching/)** — Deep dive into similarity thresholds and tuning
- **[How It Works](/concepts/how-it-works/)** — Understand the full architecture
- **[Configuration](/guides/configuration/)** — Configure context thresholds
- **[Examples](/examples/multi-agent/)** — See hybrid caching in multi-agent systems
