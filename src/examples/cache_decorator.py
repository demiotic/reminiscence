"""LLM caching with decorator pattern."""

import time
from memora import cached  # ← Import explícito


# Wrap LLM function with cache
@cached()  # ← Función, no método del módulo
def ask_llm(query: str, model: str = "gpt-4") -> str:
    """
    Call LLM with automatic caching.

    Semantically similar queries will return cached responses.
    """
    print(f"  🤖 Calling {model} API...")
    time.sleep(2)

    return f"AI response: {query[:50]}..."


def main():
    print("=" * 60)
    print("LLM Caching with Decorator")
    print("=" * 60)

    # First call - miss
    print("\n1. First query:")
    start = time.time()
    response1 = ask_llm("What is machine learning?")
    print(f"  Response: {response1}")
    print(f"  ⏱️  Time: {time.time() - start:.2f}s")

    # Second call - hit (similar)
    print("\n2. Similar query:")
    start = time.time()
    response2 = ask_llm("Explain machine learning to me")
    print(f"  Response: {response2}")
    print(f"  ⏱️  Time: {time.time() - start:.2f}s")

    # Third call - hit (exact)
    print("\n3. Exact same query:")
    start = time.time()
    response3 = ask_llm("What is machine learning?")
    print(f"  Response: {response3}")
    print(f"  ⏱️  Time: {time.time() - start:.2f}s")

    # Show stats
    from memora import get_default_memora

    memora_instance = get_default_memora()
    metrics = memora_instance.metrics

    if metrics:
        total = metrics.hits + metrics.misses
        hit_rate = metrics.hits / total if total > 0 else 0
        print(f"\n📊 Hit rate: {hit_rate:.1%} ({metrics.hits}/{total})")
        print(f"💰 Time saved: ~{metrics.hits * 2.0:.1f}s")


if __name__ == "__main__":
    main()
