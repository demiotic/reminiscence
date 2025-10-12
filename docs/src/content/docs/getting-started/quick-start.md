---
title: Quick Start
description: Get started with Reminiscence in 5 minutes
---

This guide gets you up and running with Reminiscence in under 5 minutes.

## Basic Usage

Create a cache instance and start caching:

```python
from reminiscence import Reminiscence

# Initialize cache
cache = Reminiscence()

# Check cache
result = cache.lookup(
    query="What were Q3 2024 sales?",
    context={"agent": "analyst"}
)

if result.is_hit:
    print(f"Cache hit! Similarity: {result.similarity:.2f}")
    data = result.result
else:
    # Cache miss - execute expensive operation
    data = expensive_llm_call("What were Q3 2024 sales?")

    # Store for future queries
    cache.store(
        query="What were Q3 2024 sales?",
        context={"agent": "analyst"},
        result=data
    )
```

## Using the Decorator

The `@cached` decorator automatically handles lookup and storage:

```python
from reminiscence import Reminiscence

cache = Reminiscence()

@cache.cached(query="prompt", context=["model"])
def ask_llm(prompt: str, model: str) -> str:
    """Expensive LLM call with automatic caching."""
    return openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    ).choices.message.content

# First call executes the function
answer = ask_llm("Explain quantum entanglement", model="gpt-4")

# Similar queries return cached results
answer = ask_llm("What is quantum entanglement?", model="gpt-4")  \# Cache hit!

```

## Query Modes

Reminiscence supports three matching strategies:

```python
# Semantic mode (default) - fuzzy matching
result = cache.lookup(
    "Analyze Q3 sales",
    context={"agent": "analyst"},
    query_mode="semantic"
)

# Exact mode - near-exact string matching (threshold 0.9999)
result = cache.lookup(
    "SELECT * FROM users WHERE id = 123",
    context={"db": "prod"},
    query_mode="exact"
)

# Auto mode - tries exact first, falls back to semantic
result = cache.lookup(
    "What is Python?",
    context={},
    query_mode="auto"
)
```

## Context Isolation

Context provides cache isolation between different scenarios:

```python
# Same query, different contexts = separate cache entries
cache.store(
    query="Analyze sales",
    context={"region": "US", "year": 2024},
    result={"total": 5_000_000}
)

cache.store(
    query="Analyze sales",
    context={"region": "EU", "year": 2024},
    result={"total": 3_500_000}
)

# Exact context matching
us = cache.lookup("Analyze sales", context={"region": "US", "year": 2024})
eu = cache.lookup("Analyze sales", context={"region": "EU", "year": 2024})
```

## Decorator with Context

Use `context` to automatically extract context from function arguments:

```python
@cache.cached(
query="sql_query",
context=["database", "user_id"],
query_mode="exact"
)
def execute_query(sql_query: str, database: str, user_id: int):
    """Cache results per database and user."""
    return run_expensive_query(sql_query, database, user_id)

    # Context automatically built from parameters
    result = execute_query(
        sql_query="SELECT * FROM orders",
        database="prod",
        user_id=42
    )
```

## Auto-Strict Mode

Automatically detect non-string parameters as context:

```python
@cache.cached(
query="prompt",
auto_strict=True  # Detects temperature, max_tokens as context
)
def generate_text(prompt: str, temperature: float, max_tokens: int):
    """Non-string params automatically added to context."""
    return llm_call(prompt, temperature, max_tokens)
```

## Configuration

Customize cache behavior:

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    similarity_threshold=0.85,  # Stricter matching
    max_entries=10000,          # Cache size limit
    eviction_policy="lru",      # LRU, LFU, or FIFO
    ttl_seconds=3600,           # 1 hour expiration
)

cache = Reminiscence(config=config)
```

## Background Tasks

Enable automatic cleanup and metrics export:

```python
cache = Reminiscence()

# Start background schedulers
cache.start_scheduler(
    interval_seconds=1800,  # Cleanup every 30 minutes
    metrics_export_interval_seconds=10  # Export metrics every 10s
)

# Use cache...
# Stop schedulers when done
cache.stop_scheduler()
```

Or use as context manager:

```python
with Reminiscence() as cache:
    cache.start_scheduler()
    # Use cache...
# Automatically stops schedulers on exit
```

## Next Steps

- Learn [How It Works](/concepts/how-it-works/) to understand semantic matching
- Explore [Decorators](/guides/decorators/) for advanced patterns
- See [Configuration](/guides/configuration/) for all options
- Check [API Reference](/reference/api/) for complete documentation