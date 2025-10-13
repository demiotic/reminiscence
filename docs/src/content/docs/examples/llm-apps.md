---
title: LLM Applications
description: Real-world examples of caching LLM applications with Reminiscence
---

LLM API calls are expensive—both in cost and latency. A single GPT-4 query can cost $0.01-0.10 and take 1-3 seconds. When users ask similar questions repeatedly, you're paying for the same computation over and over. Reminiscence solves this with semantic caching, but the benefits go far beyond simple text caching.

## Why LLM Caching Matters

The business case for LLM caching is compelling:

**Cost Savings**: At 30% cache hit rate (conservative), a system handling 1M queries/month at $0.05/query saves **$15,000/month**. At 50% hit rate, that's **$25,000/month** saved.

**Latency Reduction**: Cache hits return in ~10ms vs 1-3 seconds for LLM API calls. That's a **100-300x speedup**. Your application feels instant to users.

**Consistency**: Users asking the same question get the same answer, reducing confusion from LLM variability. This is crucial for support bots, documentation, and compliance.

**Rate Limit Protection**: Cache hits don't count against your API rate limits. Handle traffic spikes without throttling.

But here's what sets Reminiscence apart: **LLMs don't just return text**. Modern LLM applications produce structured outputs (JSON, function calls), analyze data (DataFrames), and handle multimodal content (images, audio). Traditional semantic caches (gptcache, upstash) only store JSON strings, so they break when you cache complex structured data or DataFrames.

Reminiscence handles all of this natively with Arrow format. Cache a DataFrame result? It's stored in columnar format with 10-100x compression. Cache a NumPy array of embeddings? Zero-copy Arrow storage. This is why production LLM systems need Reminiscence.

## The Semantic Matching Advantage

Traditional caching fails because users ask the same question in different ways:

**Same meaning, different words:**
- "What is machine learning?"
- "Explain machine learning"
- "Tell me about ML"
- "How does machine learning work?"

String-based caching treats these as four different queries, making four LLM calls. **Cost: 4x, latency: 4x.**

With Reminiscence's semantic matching, all four queries match the first cached result. **Cost: 1x, latency: 1x (after first query).** That's **75% cost savings** on just these four queries.

## Basic LLM Caching

Let's start with the fundamentals. Here's how to cache OpenAI API calls with automatic semantic matching:

```python
from reminiscence import Reminiscence
from openai import OpenAI

cache = Reminiscence()
client = OpenAI()

@cache.cached(query="prompt", context=["model", "temperature"])
def call_openai(prompt: str, model: str = "gpt-4", temperature: float = 0.7):
    """Cached OpenAI API calls"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

# First call hits API (~2 seconds, $0.05)
answer = call_openai("What is machine learning?", model="gpt-4")

# Similar query hits cache (~10ms, $0.00)
answer = call_openai("Explain machine learning", model="gpt-4")
```

**What happens here:**
1. The decorator extracts `prompt` as the query and `["model", "temperature"]` as context
2. First call: No cache match → LLM API call → Result stored
3. Second call: Semantic similarity 0.91 → Cache hit → Instant return
4. **Savings**: 2 seconds and $0.05 on the second call

**Why context matters**: Different models or temperatures produce different results, so they cache separately. `model="gpt-4"` never matches `model="gpt-3.5-turbo"` even for identical prompts.

## Multi-Model Comparison

Production systems often compare outputs across multiple models (GPT-4, Claude, Llama). Caching each model separately prevents expensive re-computation:

```python
from reminiscence import Reminiscence
import anthropic
from openai import OpenAI

cache = Reminiscence()
openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

@cache.cached(query="prompt", context=["provider", "model"])
def call_llm(
    prompt: str,
    provider: str,
    model: str,
    **kwargs
):
    """Universal LLM interface with caching"""
    if provider == "openai":
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        response = anthropic_client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

    raise ValueError(f"Unknown provider: {provider}")

# Each provider + model combination cached separately
gpt4 = call_llm("What is AI?", provider="openai", model="gpt-4")
claude = call_llm("What is AI?", provider="anthropic", model="claude-3-opus-20240229")
gpt35 = call_llm("What is AI?", provider="openai", model="gpt-3.5-turbo")

# Similar query for each model hits respective caches
gpt4_cached = call_llm("Explain AI", provider="openai", model="gpt-4")  # Cache hit!
```

**Context isolation in action**: Same semantic query, but different providers/models cache separately. You can compare model outputs without redundant API calls.

**Cost impact**: Model comparison experiments typically run the same prompt across 3-5 models. With caching, only the first run pays full cost—subsequent similar prompts are free.

## Streaming with Caching

Streaming responses provide better UX (users see incremental output), but you still want to cache the complete result:

```python
@cache.cached(query="prompt", context=["model"])
def call_with_streaming(prompt: str, model: str):
    """Cache streaming responses"""
    # Stream from API
    full_response = ""
    for chunk in client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            print(content, end="", flush=True)

    return full_response  # This gets cached

# First call streams (~2 seconds)
response1 = call_with_streaming("Write a poem", model="gpt-4")

# Second similar call returns instantly from cache (~10ms)
response2 = call_with_streaming("Write a poem about nature", model="gpt-4")
```

**Pattern**: Stream on cache miss for UX, return instantly on cache hit. Best of both worlds.

## Conversation History

Multi-turn conversations present a caching challenge: responses depend on both the current message AND the conversation history. The solution is to hash the history into the context:

```python
import hashlib
import json

def hash_messages(messages: list) -> str:
    """Create stable hash of conversation history"""
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode()
    ).hexdigest()[:16]

@cache.cached(
    query="current_message",
    context=["model", "history_hash"]
)
def chat_with_history(
    current_message: str,
    history: list,
    model: str = "gpt-4"
):
    """Cache responses based on full conversation context"""
    messages = history + [{"role": "user", "content": current_message}]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message.content

# Conversation
history = []

# First exchange
response1 = chat_with_history(
    "What is Python?",
    history=history,
    model="gpt-4"
)
history.append({"role": "user", "content": "What is Python?"})
history.append({"role": "assistant", "content": response1})

# Second exchange (depends on history)
response2 = chat_with_history(
    "Show me an example",  # Context-dependent question
    history=history,
    model="gpt-4"
)

# Same conversation path from scratch → cache hits!
history_copy = []
response1_cached = chat_with_history(
    "What is Python?",
    history=history_copy,
    model="gpt-4"
)  # Cache hit!
```

**Why this works**: The `history_hash` in context ensures that same conversation paths cache together, but different conversation histories cache separately. If two users follow the same conversation flow, they benefit from each other's cached results.

**Production tip**: Conversation caching works best for FAQ-style bots where users follow common conversation patterns (support bots, documentation assistants).

## Function Calling / Tools

LLMs with function calling don't just return text—they return structured decisions about which functions to call with what arguments. Caching this is crucial for cost savings:

```python
import json

@cache.cached(query="prompt", context=["model", "available_tools"])
def call_with_tools(prompt: str, model: str, available_tools: str):
    """Cache function calling responses"""
    tools = json.loads(available_tools)  # Parse from string

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="auto"
    )

    return {
        "message": response.choices[0].message.content,
        "tool_calls": [
            {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
            for tc in (response.choices[0].message.tool_calls or [])
        ]
    }

# Define tools
tools_json = json.dumps([
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
])

# Cache function call decisions
result = call_with_tools(
    "What's the weather in Paris?",
    model="gpt-4",
    available_tools=tools_json
)
```

**What gets cached**: The structured result dict with both the message and tool_calls. This is stored efficiently with orjson (2-3x faster than stdlib json).

**Cost savings**: Function calling models (GPT-4 with tools) are 20-30% more expensive than base models. Caching decisions saves significant cost.

## Prompt Templates

Template-based prompting is common in production. Caching at the template level maximizes reuse:

```python
from string import Template

@cache.cached(query="template_filled", context=["template_name", "model"])
def generate_from_template(
    template_name: str,
    variables: dict,
    model: str = "gpt-4"
):
    """Cache template-based generations"""
    templates = {
        "summarize": Template("Summarize this text in $style style:\n\n$text"),
        "translate": Template("Translate '$text' from $source_lang to $target_lang"),
        "code_review": Template("Review this $language code:\n\n$code")
    }

    template = templates[template_name]
    filled = template.substitute(variables)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": filled}]
    )

    return response.choices[0].message.content

# Same template + values = cache hit
summary1 = generate_from_template(
    "summarize",
    {"style": "concise", "text": "Long article..."},
    model="gpt-4"
)

summary2 = generate_from_template(
    "summarize",
    {"style": "concise", "text": "Long article..."},  # Same text
    model="gpt-4"
)  # Cache hit!
```

**Template caching strategy**: By including `template_name` in context, each template caches separately. This prevents cross-contamination while maximizing reuse within each template.

## Embeddings Caching

Embedding generation is cheaper than LLM calls but still adds up at scale. Cache embeddings for repeated text:

```python
@cache.cached(query="text", context=["model"])
def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """Cache OpenAI embeddings"""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

# Cache embeddings
embedding1 = get_embedding("What is AI?", model="text-embedding-3-small")
embedding2 = get_embedding("Explain AI", model="text-embedding-3-small")  # Cache hit!
```

**What gets cached**: The NumPy array of embeddings, stored in Arrow format for efficient retrieval. No JSON conversion overhead.

**Use cases**: Vector search, clustering, similarity detection—anywhere you repeatedly embed the same or similar text.

## Retry Logic with Caching

LLM APIs fail occasionally (rate limits, timeouts, server errors). Retry logic helps, but you want to cache successful results:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@cache.cached(query="prompt", context=["model"])
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_with_retry(prompt: str, model: str = "gpt-4"):
    """Retry failed calls, cache successful ones"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise  # Retry will happen, error not cached

# Retries on failure, caches on success
answer = call_with_retry("What is quantum computing?")
```

**Error handling**: By default, Reminiscence doesn't cache errors (exceptions, error dicts). Only successful results get cached. This prevents serving cached errors when the API might succeed on retry.

## Rate Limiting with Cache

API rate limits are a constant production concern. Caching reduces pressure on rate limits:

```python
from ratelimit import limits, sleep_and_retry

@cache.cached(query="prompt", context=["model"])
@sleep_and_retry
@limits(calls=10, period=60)  # 10 calls per minute
def rate_limited_call(prompt: str, model: str = "gpt-4"):
    """Rate limited API calls with caching"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Cached calls don't count against rate limit
for i in range(100):
    answer = rate_limited_call(f"Question {i}", model="gpt-4")
    # Only unique questions hit API (rate limited)
    # Similar questions hit cache (unlimited)
```

**Rate limit protection**: At 50% hit rate, you effectively double your rate limit capacity. Cache hits are instant and don't consume API quota.

## Cost Tracking

Understanding your LLM costs is crucial. Integrate cost tracking with caching to measure savings:

```python
class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.api_calls = 0

    def track_call(self, model: str, prompt_tokens: int, completion_tokens: int):
        # GPT-4 pricing (example)
        costs = {
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-3.5-turbo": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000}
        }

        cost = (
            prompt_tokens * costs[model]["prompt"] +
            completion_tokens * costs[model]["completion"]
        )

        self.total_cost += cost
        self.api_calls += 1

        return cost

tracker = CostTracker()

@cache.cached(query="prompt", context=["model"])
def call_with_cost_tracking(prompt: str, model: str = "gpt-4"):
    """Track API costs"""
    result = cache.lookup(prompt, {"model": model})

    if result.is_hit:
        logger.info(f"Cache hit - $0.00 (saved ~$0.05)")
        return result.result

    # API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    cost = tracker.track_call(
        model=model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )

    logger.info(f"API call - ${cost:.4f}")

    return response.choices[0].message.content

# Monitor savings
print(f"Total cost: ${tracker.total_cost:.2f}")
print(f"API calls: {tracker.api_calls}")
hit_rate = cache.get_stats()['hit_rate']
print(f"Cache hit rate: {hit_rate}")

# Estimated savings calculation
if hit_rate > 0:
    # If hit rate is 50%, you saved 50% of what you would have spent
    total_without_cache = tracker.total_cost / (1 - hit_rate)
    savings = total_without_cache - tracker.total_cost
    print(f"Estimated savings: ${savings:.2f}")
```

**Production metrics**: Track cost per query, hit rate, and total savings. This data justifies caching infrastructure investment.

## Async LLM Calls

Modern Python applications use async for better concurrency. Caching works with async functions:

```python
from openai import AsyncOpenAI
import asyncio

client = AsyncOpenAI()

@cache.cached(query="prompt", context=["model"])
async def call_openai_async(prompt: str, model: str = "gpt-4"):
    """Cached async OpenAI calls"""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Parallel async calls with caching
async def main():
    tasks = [
        call_openai_async(f"Question {i}", model="gpt-4")
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)
    return results

# Run
results = asyncio.run(main())
```

**Async benefits**: Caching async calls prevents redundant concurrent requests. If 10 async tasks ask similar questions, the cache deduplicates them.

## LangChain Integration

LangChain is popular for LLM application development. Integrate caching at the chain level:

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

@cache.cached(query="formatted_prompt", context=["model"])
def langchain_with_cache(formatted_prompt: str, model: str):
    """Cache LangChain outputs"""
    llm = OpenAI(model=model)
    return llm(formatted_prompt)

# Use in LangChain workflow
template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short explanation of {topic}"
)

def generate_explanation(topic: str, model: str = "gpt-3.5-turbo"):
    formatted = template.format(topic=topic)
    return langchain_with_cache(formatted, model)

# Cached
explanation1 = generate_explanation("quantum computing")
explanation2 = generate_explanation("quantum physics")  # Cache hit for similar topic!
```

**Integration pattern**: Cache at the LLM call level, not the chain level. This provides more granular reuse across different chains.

## Best Practices

1. **Include all result-affecting parameters in context**: model, temperature, system prompt, tools
2. **Use semantic mode for user-generated queries**: Natural language benefits from fuzzy matching
3. **Use exact mode for deterministic prompts**: SQL generation, code execution, calculations
4. **Track costs and hit rates**: Measure ROI of caching infrastructure
5. **Cache structured outputs directly**: DataFrames, JSON, arrays—Reminiscence handles them efficiently
6. **Consider conversation state**: Hash conversation history for multi-turn caching
7. **Monitor similarity scores**: Low-similarity hits might need threshold tuning

## Next Steps

- [RAG Pipelines](/examples/rag/) - Caching RAG systems
- [Multi-Agent Systems](/examples/multi-agent/) - Caching agent workflows
- [Decorators Guide](/guides/decorators/) - Advanced decorator patterns
