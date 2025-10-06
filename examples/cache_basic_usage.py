"""Basic LLM caching example with Memora."""

import time
from memora import Memora, CacheConfig


def call_llm(prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
    """Fake LLM that takes 2 seconds to respond."""
    print(f"  🤖 Calling LLM (model={model}, temp={temperature})...")
    time.sleep(2)  # Simulate API latency

    # Fake responses
    if "python" in prompt.lower():
        return "Python is a high-level programming language known for its simplicity and readability."
    elif "weather" in prompt.lower():
        return "I don't have real-time weather data, but you can check weather.com."
    else:
        return f"This is a simulated response to: {prompt[:50]}..."


def main():
    print("=" * 60)
    print("Basic LLM Caching Example")
    print("=" * 60)

    # Initialize cache with lower threshold for better hit rate
    cache = Memora(
        CacheConfig(
            db_uri="memory://",
            similarity_threshold=0.80,  # Lower threshold for demo
            log_level="ERROR",  # Less verbose
        )
    )

    context = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 100,
    }

    # First call - cache miss
    print("\n1. First query (cache miss):")
    prompt1 = "What is Python?"

    start = time.time()
    result = cache.lookup(prompt1, context)

    if result.is_miss:
        print("  ❌ Cache MISS - calling LLM...")
        response = call_llm(prompt1, model=context["model"])
        cache.store(prompt1, context, response)
    else:
        print(f"  ✅ Cache HIT (similarity: {result.similarity:.3f})")
        response = result.result

    elapsed = time.time() - start
    print(f"  Response: {response}")
    print(f"  ⏱️  Time: {elapsed:.2f}s")

    # Second call - semantically similar, cache hit
    print("\n2. Similar query (should hit cache):")
    prompt2 = "Tell me about the Python programming language"

    start = time.time()
    result = cache.lookup(prompt2, context)

    if result.is_miss:
        print(
            f"  ❌ Cache MISS (similarity was {result.similarity:.3f if result.similarity else 'N/A'})"
        )
        response = call_llm(prompt2, model=context["model"])
        cache.store(prompt2, context, response)
    else:
        print(f"  ✅ Cache HIT! (similarity: {result.similarity:.3f})")
        response = result.result

    elapsed = time.time() - start
    print(f"  Response: {response}")
    print(f"  ⏱️  Time: {elapsed:.2f}s" + (" (saved ~2s!)" if result.is_hit else ""))

    # Third call - exact same query, definite hit
    print("\n3. Exact same query (definitely cache hit):")
    start = time.time()
    result = cache.lookup(prompt1, context)

    if result.is_hit:
        print(f"  ✅ Cache HIT! (similarity: {result.similarity:.3f})")
        response = result.result
    else:
        print("  ❌ Unexpected miss!")
        response = call_llm(prompt1, model=context["model"])

    elapsed = time.time() - start
    print(f"  Response: {response}")
    print(f"  ⏱️  Time: {elapsed:.2f}s (saved ~2s!)")

    # Fourth call - different context (different model), cache miss
    print("\n4. Same query, different model (cache miss):")
    context_different = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100,
    }

    start = time.time()
    result = cache.lookup(prompt1, context_different)

    if result.is_miss:
        print("  ❌ Cache MISS (different context)")
        response = call_llm(prompt1, model=context_different["model"])
        cache.store(prompt1, context_different, response)
    else:
        print(f"  ✅ Cache HIT (similarity: {result.similarity:.3f})")
        response = result.result

    elapsed = time.time() - start
    print(f"  Response: {response}")
    print(f"  ⏱️  Time: {elapsed:.2f}s")

    # Show stats
    print("\n" + "=" * 60)
    print("Cache Statistics")
    print("=" * 60)

    # FIX: get_stats() returns CacheMetrics object, not dict
    metrics = cache.metrics
    if metrics:
        total = metrics.hits + metrics.misses
        hit_rate = metrics.hits / total if total > 0 else 0

        print(f"  Total queries: {total}")
        print(f"  Cache hits: {metrics.hits}")
        print(f"  Cache misses: {metrics.misses}")
        print(f"  Hit rate: {hit_rate:.1%}")
        print(f"  Entries in cache: {cache.table.count_rows()}")

        # Calculate time saved (assuming 2s per LLM call)
        time_saved = metrics.hits * 2.0
        print(f"  Estimated time saved: ~{time_saved:.1f}s")
    else:
        print("  Metrics not enabled")


if __name__ == "__main__":
    main()
