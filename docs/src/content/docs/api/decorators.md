---
title: Decorators
description: Automatic caching with the @cached decorator
---

The `@cached` decorator provides automatic caching for function results, eliminating boilerplate lookup/store code.

## Basic Usage

### `@cached()`

Decorator for automatic function result caching.

```python
cached(
    query: str = "query",
    context: Optional[list] = None,
    static_context: Optional[Dict[str, Any]] = None,
    auto_strict: bool = False,
    mode: QueryMode = QueryMode.AUTO,
    similarity_threshold: Optional[float] = None
)
```

**Parameters:**
- `query` (str): Parameter name containing the query (default: "query")
- `context` (Optional[list]): Parameter names to use as context
- `static_context` (Optional[Dict]): Fixed context values
- `auto_strict` (bool): Auto-detect non-string params as context
- `mode` (QueryMode): Matching strategy
- `similarity_threshold` (Optional[float]): Override default threshold

---

## Examples

### Simple Caching

Cache based on a query parameter:

```python
from reminiscence import Reminiscence

cache = Reminiscence()

@cache.cached(query="prompt")
def call_llm(prompt: str):
    return expensive_llm_call(prompt)

# First call executes function
answer1 = call_llm("What is AI?")

# Similar query hits cache
answer2 = call_llm("Explain AI")  # Cache hit!
```

### With Context

Cache based on query + context for isolation:

```python
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return expensive_llm_call(prompt, model)

# Different models cache separately
gpt4_answer = call_llm("What is AI?", model="gpt-4")
gpt35_answer = call_llm("What is AI?", model="gpt-3.5-turbo")
# Two separate cache entries
```

### Multiple Context Parameters

```python
@cache.cached(
    query="sql_query",
    context=["database", "user_id"]
)
def execute_query(sql_query: str, database: str, user_id: int):
    return run_expensive_query(sql_query, database, user_id)

# Context ensures correct isolation
result1 = execute_query("SELECT * FROM users", "prod", user_id=42)
result2 = execute_query("SELECT * FROM users", "dev", user_id=42)
# Different databases → different cache entries
```

### Static Context

Add fixed context values:

```python
@cache.cached(
    query="sql_query",
    context=["database"],
    static_context={"version": "2.0", "region": "us-east-1"}
)
def execute_sql(sql_query: str, database: str):
    return run_query(sql_query, database)

# Static context automatically included in all lookups
```

### Auto-Strict Mode

Automatically detect non-string parameters as context:

```python
@cache.cached(query="prompt", auto_strict=True)
def generate(prompt: str, temperature: float, max_tokens: int):
    # temperature and max_tokens automatically added to context
    return llm(prompt, temperature=temperature, max_tokens=max_tokens)

# Different parameters = different cache entries
result1 = generate("Write a poem", temperature=0.7, max_tokens=100)
result2 = generate("Write a poem", temperature=0.9, max_tokens=100)
# Different temperature → cache miss
```

**How auto_strict works:**
- String parameters: Used as query (if specified) or ignored
- Non-string parameters (int, float, bool, etc.): Automatically added to context
- Useful when all non-string params affect the result

---

## Query Modes

### Semantic Mode (Default)

Fuzzy matching for natural language:

```python
@cache.cached(
    query="question",
    context=["model"],
    mode=QueryMode.SEMANTIC
)
def ask_llm(question: str, model: str):
    return llm_call(question, model)

# These match semantically
ask_llm("What is ML?", "gpt-4")
ask_llm("Explain machine learning", "gpt-4")  # Cache hit!
```

### Exact Mode

Strict matching for deterministic queries:

```python
@cache.cached(
    query="code",
    context=["language"],
    mode=QueryMode.EXACT
)
def execute_code(code: str, language: str):
    return run_code(code, language)

# Only nearly-identical queries match
execute_code("print('hello')", "python")
execute_code("print('world')", "python")  # Cache miss - different code
```

**Use exact mode for:**
- SQL queries
- Code execution
- Mathematical calculations
- API calls with exact parameters

### Auto Mode

Try exact first, fallback to semantic:

```python
@cache.cached(
    query="prompt",
    context=["model"],
    mode=QueryMode.AUTO  # Default
)
def call_llm(prompt: str, model: str):
    return llm_call(prompt, model)

# Exact match if available, semantic fallback otherwise
```

---

## Advanced Patterns

### Custom Similarity Threshold

Override the default threshold per decorator:

```python
@cache.cached(
    query="prompt",
    context=["model"],
    similarity_threshold=0.90  # Stricter than default 0.80
)
def strict_llm_call(prompt: str, model: str):
    return llm_call(prompt, model)
```

### Async Functions

The decorator works with async functions:

```python
@cache.cached(query="prompt", context=["model"])
async def async_llm_call(prompt: str, model: str):
    return await async_llm(prompt, model)

# Use with await
result = await async_llm_call("What is AI?", "gpt-4")
```

### Combining with Other Decorators

Stack decorators for additional functionality:

```python
from tenacity import retry, stop_after_attempt

@cache.cached(query="prompt", context=["model"])
@retry(stop=stop_after_attempt(3))
def retry_llm_call(prompt: str, model: str):
    return llm_call(prompt, model)  # Retries on failure, caches on success
```

### Dataclass Parameters

Use dataclasses in context:

```python
from dataclasses import dataclass, asdict

@dataclass
class LLMConfig:
    model: str
    temperature: float
    max_tokens: int

@cache.cached(query="prompt", context=["config_dict"])
def call_with_config(prompt: str, config: LLMConfig):
    config_dict = asdict(config)  # Convert to dict for context
    return llm_call(prompt, **config_dict)
```

---

## Best Practices

### DO: Include Result-Affecting Parameters in Context

```python
# ✓ Good: Temperature affects result, so it's in context
@cache.cached(query="prompt", context=["model", "temperature"])
def generate(prompt: str, model: str, temperature: float):
    return llm(prompt, model=model, temperature=temperature)

# ✗ Bad: Temperature not in context - different temps match!
@cache.cached(query="prompt", context=["model"])
def generate(prompt: str, model: str, temperature: float):
    return llm(prompt, model=model, temperature=temperature)
```

### DO: Use Semantic Mode for Natural Language

```python
# ✓ Good: Natural language benefits from fuzzy matching
@cache.cached(query="question", context=["domain"], mode=QueryMode.SEMANTIC)
def answer_question(question: str, domain: str):
    return qa_system(question, domain)
```

### DO: Use Exact Mode for Deterministic Queries

```python
# ✓ Good: SQL needs exact matching
@cache.cached(query="sql", context=["database"], mode=QueryMode.EXACT)
def run_sql(sql: str, database: str):
    return execute(sql, database)
```

### DON'T: Put High-Cardinality Values in Context

```python
# ✗ Bad: Timestamp is unique every call - cache never hits
@cache.cached(query="prompt", context=["timestamp"])
def time_sensitive(prompt: str, timestamp: int):
    return llm(prompt)

# ✓ Good: Only include params that repeat
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str, _timestamp: int = None):
    return llm(prompt, model=model)  # timestamp ignored
```

### DON'T: Duplicate Query Text in Context

```python
# ✗ Bad: Redundant - query already used for matching
@cache.cached(query="prompt", context=["prompt"])
def call_llm(prompt: str):
    return llm(prompt)

# ✓ Good: Context only for additional parameters
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return llm(prompt, model)
```

---

## Error Handling

By default, errors are **not cached**. The decorator only caches successful results:

```python
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    response = api_call(prompt, model)
    if response.error:
        raise Exception(response.error)  # Not cached
    return response.text  # Cached

# To cache errors, use allow_errors in manual store()
```

---

## Comparison: With vs Without Decorator

### Without Decorator (Manual)

```python
def call_llm(prompt: str, model: str):
    # Check cache
    result = cache.lookup(
        query=MultiModalInput(text=prompt),
        context={"model": model}
    )

    if result.is_hit:
        return result.result

    # Execute
    response = expensive_llm_call(prompt, model)

    # Store
    cache.store(
        query=MultiModalInput(text=prompt),
        context={"model": model},
        result=response
    )

    return response
```

### With Decorator (Automatic)

```python
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return expensive_llm_call(prompt, model)
```

**The decorator:**
- Eliminates 10+ lines of boilerplate
- Handles lookup/store automatically
- Works with both sync and async functions
- Respects all caching settings (thresholds, modes, etc.)

---

## Next Steps

- **[Core Operations](/reference/api/core-operations/)** - Manual caching methods
- **[Configuration](/reference/api/configuration/)** - Customize decorator behavior
- **[Examples](/examples/llm-apps/)** - Real-world decorator patterns
