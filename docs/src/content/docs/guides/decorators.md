---
title: Decorators
description: Advanced patterns for using Reminiscence's @cached decorator
---

The `@cached` decorator provides automatic caching for functions. This guide covers advanced patterns and best practices.

## Basic Decorator Usage

```python
from reminiscence import Reminiscence

cache = Reminiscence()

@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str) -> str:
    return expensive_llm_call(prompt, model)

# Automatic caching
answer = call_llm("What is AI?", model="gpt-4")
```

## Query Parameter

Specify which parameter contains the query text:

```python
# Default: query="query"
@cache.cached()
def process(query: str):
    return expensive_operation(query)

# Custom query parameter
@cache.cached(query="question")
def ask(question: str):
    return answer_question(question)

# Works with different parameter names
@cache.cached(query="prompt")
def generate(prompt: str):
    return generate_text(prompt)
```

## Context Parameters

### Single Context Parameter

```python
@cache.cached(query="prompt", context="model")
def call_llm(prompt: str, model: str):
    return llm_api(prompt, model)

# Context: {"model": "gpt-4"}
```

### Multiple Context Parameters

```python
@cache.cached(
    query="prompt",
    context=["model", "temperature", "max_tokens"]
)
def call_llm(prompt: str, model: str, temperature: float, max_tokens: int):
    return llm_api(prompt, model, temperature, max_tokens)

# Context: {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000}
```

### No Context

```python
@cache.cached(query="question")
def general_qa(question: str):
    return answer(question)

# Context: {"__function__": "general_qa"}
# Uses function name as minimal context
```

## Auto-Strict Mode

Automatically detect non-string parameters as context:

```python
@cache.cached(query="prompt", auto_strict=True)
def generate_text(
    prompt: str,        # String → query
    temperature: float, # Non-string → auto-added to context
    max_tokens: int,    # Non-string → auto-added to context
    request_id: str     # String but not query → ignored
):
    return llm(prompt, temperature, max_tokens)

# Equivalent to:
# @cache.cached(query="prompt", context=["temperature", "max_tokens"])
```

**Use when:**
- Clear separation between query (string) and parameters (non-string)
- Rapid prototyping
- Don't want to manually list context fields

## Static Context

Add fixed context to all cache entries:

```python
@cache.cached(
    query="request",
    context=["user_id"],
    static_context={"service": "api_v2", "region": "us-east-1"}
)
def api_handler(request: str, user_id: int):
    return process_request(request, user_id)

# Context: {
#     "user_id": 42,
#     "service": "api_v2",
#     "region": "us-east-1"
# }
```

**Use cases:**
- Service identification
- Version tracking
- Environment labels
- Feature flags

## Query Modes

### AUTO Mode (Default)

Try exact match first, fallback to semantic:

```python
@cache.cached(
    query="prompt",
    context=["model"],
    mode=QueryMode.AUTO  # Default
)
def call_llm(prompt: str, model: str):
    return llm(prompt, model)
```

### SEMANTIC Mode

Pure semantic matching:

```python
from reminiscence import QueryMode

@cache.cached(
    query="question",
    mode=QueryMode.SEMANTIC
)
def qa_system(question: str):
    return answer(question)
```

### EXACT Mode

High-precision matching for deterministic queries:

```python
@cache.cached(
    query="sql",
    context=["database"],
    mode=QueryMode.EXACT
)
def execute_query(sql: str, database: str):
    return db_execute(sql, database)
```

## Similarity Threshold

Override default similarity threshold:

```python
# Strict matching
@cache.cached(
    query="prompt",
    similarity_threshold=0.95
)
def strict_cache(prompt: str):
    return process(prompt)

# Loose matching
@cache.cached(
    query="prompt",
    similarity_threshold=0.70
)
def loose_cache(prompt: str):
    return process(prompt)
```

## Error Handling

Control whether errors are cached:

```python
# Don't cache errors (default)
@cache.cached(query="request", allow_errors=False)
def api_call(request: str):
    response = external_api(request)
    if response.status_code != 200:
        raise APIError(response.text)
    return response.json()

# Cache errors
@cache.cached(query="request", allow_errors=True)
def api_call_cached_errors(request: str):
    try:
        return external_api(request).json()
    except Exception as e:
        return {"error": str(e)}  # This will be cached
```

## Async Functions

Decorators work with async functions:

```python
@cache.cached(query="prompt", context=["model"])
async def async_llm_call(prompt: str, model: str):
    response = await async_openai_client.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Use with await
answer = await async_llm_call("What is AI?", model="gpt-4")
```

## Batch Mode

Decorator supports batch operations internally:

```python
@cache.cached(
    query="prompt",
    context=["model"],
    batch_mode=True  # Default
)
def call_llm(prompt: str, model: str):
    return llm(prompt, model)

# Single query (batch_mode optimizes internally)
answer = call_llm("What is AI?", model="gpt-4")

# Batch queries (if function supports it)
# answers = call_llm(["What is AI?", "What is ML?"], model="gpt-4")
```

**Disable batch mode if needed:**

```python
@cache.cached(batch_mode=False)
def process(query: str):
    return expensive_op(query)
```

## Multimodal Queries

Cache multimodal inputs:

```python
from reminiscence.types import MultiModalInput

@cache.cached(query="query", context=["model"])
def analyze_image(query: MultiModalInput, model: str):
    return vision_model(query, model)

# Text-only
result = analyze_image(
    query=MultiModalInput(text="What is this?"),
    model="gpt-4o"
)

# Image with text
result = analyze_image(
    query=MultiModalInput(text="Describe this", image=img_bytes),
    model="gpt-4o"
)
```

## Method Decorators

Use with class methods:

```python
class LLMService:
    def __init__(self, cache: Reminiscence):
        self.cache = cache

    @cache.cached(query="prompt", context=["model"])
    def call(self, prompt: str, model: str):
        return openai_api(prompt, model)

# Usage
service = LLMService(cache)
answer = service.call("What is AI?", model="gpt-4")
```

## Decorator Chaining

Combine with other decorators:

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

# Cache first, then time
@timing_decorator
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return llm(prompt, model)

# First call: Executes function, times execution
# Second call: Returns from cache (fast), times cache lookup
```

## Real-World Patterns

### Pattern 1: RAG Pipeline

```python
@cache.cached(
    query="user_query",
    context=["collection", "model", "top_k"]
)
def rag_query(user_query: str, collection: str, model: str, top_k: int = 5):
    # Retrieve documents
    docs = vector_db.search(user_query, collection=collection, limit=top_k)

    # Generate answer
    context = "\n\n".join([d.content for d in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}"

    answer = llm_generate(prompt, model=model)

    return {
        "answer": answer,
        "sources": [{"title": d.title, "url": d.url} for d in docs]
    }

# Caches by query + collection + model + top_k
result = rag_query(
    "What is quantum computing?",
    collection="physics",
    model="gpt-4",
    top_k=5
)
```

### Pattern 2: Multi-Model Ensemble

```python
@cache.cached(query="prompt", context=["model"])
def call_single_model(prompt: str, model: str):
    return llm_api(prompt, model)

def ensemble_call(prompt: str, models: list[str]):
    # Each model call is cached independently
    results = [
        call_single_model(prompt, model)
        for model in models
    ]

    # Aggregate results
    return majority_vote(results)

# Models cached separately
answer = ensemble_call(
    "Is this email spam?",
    models=["gpt-4", "claude-3", "gemini-pro"]
)
```

### Pattern 3: Parameterized Code Execution

```python
@cache.cached(
    query="code",
    context=["language", "timeout"],
    mode=QueryMode.EXACT  # Code must match exactly
)
def execute_code(code: str, language: str, timeout: int = 30):
    sandbox = get_sandbox(language)
    result = sandbox.execute(code, timeout=timeout)
    return {
        "output": result.stdout,
        "error": result.stderr,
        "exit_code": result.exit_code
    }

# Cached by exact code + language + timeout
result = execute_code(
    code="print('hello')",
    language="python",
    timeout=30
)
```

### Pattern 4: Translation Service

```python
@cache.cached(
    query="text",
    context=["source_lang", "target_lang", "model"]
)
def translate(text: str, source_lang: str, target_lang: str, model: str = "base"):
    return translation_api(text, source_lang, target_lang, model)

# Each language pair cached separately
french = translate("Hello", "en", "fr", model="base")
spanish = translate("Hello", "en", "es", model="base")

# Semantic matching helps with variations
result = translate("Hi there", "en", "fr", model="base")  # May hit cache
```

### Pattern 5: Data Processing Pipeline

```python
@cache.cached(
    query="dataset_id",
    context=["transform", "aggregation"],
    static_context={"pipeline": "v2"}
)
def process_dataset(dataset_id: str, transform: str, aggregation: str):
    # Load data
    df = load_dataset(dataset_id)

    # Apply transformation
    if transform == "normalize":
        df = normalize(df)
    elif transform == "standardize":
        df = standardize(df)

    # Aggregate
    result = df.groupby("category").agg(aggregation)

    return result.to_dict()

# Cached by dataset + transform + aggregation
result = process_dataset(
    dataset_id="sales_2024",
    transform="normalize",
    aggregation="sum"
)
```

## Performance Tips

### 1. Minimize Context Fields

```python
# ❌ Too many context fields = low hit rate
@cache.cached(
    query="q",
    context=["session_id", "request_id", "timestamp", "ip"]
)
def process(q: str, **kwargs):
    return do_work(q)

# ✓ Only essential context
@cache.cached(query="q", context=["user_id"])
def process(q: str, user_id: int, session_id: str, request_id: str):
    return do_work(q, user_id)
```

### 2. Use Static Context for Constants

```python
# ❌ Redundant context parameter
@cache.cached(query="q", context=["version"])
def process(q: str, version: str = "v2"):
    return do_work(q)

# ✓ Use static context
@cache.cached(query="q", static_context={"version": "v2"})
def process(q: str):
    return do_work(q)
```

### 3. Choose Appropriate Mode

```python
# ❌ SEMANTIC mode for SQL (too loose)
@cache.cached(query="sql", mode=QueryMode.SEMANTIC)
def execute_sql(sql: str):
    return db.execute(sql)

# ✓ EXACT mode for deterministic queries
@cache.cached(query="sql", mode=QueryMode.EXACT)
def execute_sql(sql: str):
    return db.execute(sql)
```

## Debugging Decorators

### Check Cache Behavior

```python
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    print(f"Function executed: {prompt[:50]}...")
    return llm(prompt, model)

# First call: Prints message
call_llm("What is AI?", model="gpt-4")

# Similar query: No print (cache hit)
call_llm("Explain AI", model="gpt-4")

# Different model: Prints message (cache miss due to context)
call_llm("What is AI?", model="gpt-3.5-turbo")
```

### Check Metrics

```python
# Enable metrics
cache = Reminiscence()

@cache.cached(query="prompt")
def process(prompt: str):
    return expensive_op(prompt)

# Make calls
for i in range(100):
    process(f"query {i % 10}")

# Check hit rate
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
```

## Class-Based Decorator Interface

Alternative API using `ReminiscenceDecorator`:

```python
from reminiscence import Reminiscence, ReminiscenceDecorator

cache = Reminiscence()
decorator = ReminiscenceDecorator(cache)

@decorator.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return llm(prompt, model)
```

## Next Steps

- [Background Tasks](/guides/background-tasks/) - Schedulers and cleanup
- [Examples](/examples/llm-apps/) - Real-world decorator examples
- [Performance](/production/performance/) - Optimization techniques
