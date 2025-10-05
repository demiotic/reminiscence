"""Production-ready LLM caching with monitoring."""

import time
import os
from memora import Memora, CacheConfig


def call_expensive_llm(prompt: str, context: dict) -> str:
    """Simulate expensive LLM call."""
    time.sleep(2)

    # Different responses based on topic
    if "weather" in prompt.lower():
        return "Today's forecast: Sunny, 22°C with light winds."
    elif "quantum" in prompt.lower():
        return "Quantum computing uses quantum mechanics principles like superposition."
    elif "python" in prompt.lower():
        return "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    ..."
    else:
        return f"LLM response to: {prompt[:30]}..."


def main():
    print("=" * 60)
    print("Production LLM Cache Example")
    print("=" * 60)

    # Production config from environment
    config = CacheConfig(
        db_uri=os.getenv("MEMORA_DB_URI", "./prod_cache.db"),
        ttl_seconds=int(os.getenv("MEMORA_TTL_SECONDS", "3600")),
        max_entries=int(os.getenv("MEMORA_MAX_ENTRIES", "10000")),
        similarity_threshold=float(os.getenv("MEMORA_SIMILARITY_THRESHOLD", "0.85")),
        log_level=os.getenv("MEMORA_LOG_LEVEL", "ERROR"),
        enable_metrics=True,
    )

    cache = Memora(config)

    # Check health before starting
    print("\nℹ️  Checking cache health...")
    health = cache.health_check()
    if health["status"] != "healthy":
        print("⚠️  Cache health check failed!")
        print(health)
        return

    print(f"✅ Cache healthy - {health['metrics']['total_entries']} entries")

    # Simulate production workload
    context = {"model": "gpt-4", "user_id": "user_123"}

    queries = [
        ("What is the weather today?", "weather1"),
        ("Tell me the weather forecast", "weather2"),
        ("Explain quantum computing", "quantum1"),
        ("What is quantum computing?", "quantum2"),
        ("Write Python code for sorting", "code1"),
        ("What is the weather like?", "weather3"),
    ]

    print("\n" + "=" * 60)
    print("Processing Queries")
    print("=" * 60)

    for i, (query, label) in enumerate(queries, 1):
        print(f"\n{i}. {label}: {query}")

        start = time.time()
        result = cache.lookup(query, context)

        if result.is_miss:
            print("   ❌ MISS - calling LLM")
            response = call_expensive_llm(query, context)
            cache.store(query, context, response)
        else:
            print(
                f"   ✅ HIT (sim: {result.similarity:.3f}, age: {result.age_seconds:.0f}s)"
            )
            response = result.result

        elapsed = time.time() - start
        print(f"   Response: {response[:50]}...")
        print(f"   ⏱️  {elapsed:.2f}s")

    # Monitor metrics
    print("\n" + "=" * 60)
    print("Performance Metrics")
    print("=" * 60)

    metrics = cache.metrics
    if metrics:
        total = metrics.hits + metrics.misses
        hit_rate = metrics.hits / total if total > 0 else 0
        time_saved = metrics.hits * 2.0  # 2s per LLM call

        print(f"Hit rate: {hit_rate:.1%} ({metrics.hits}/{total})")
        print(f"Time saved: ~{time_saved:.1f}s")
        print(f"Cache entries: {cache.table.count_rows()}")
        print(f"Errors: lookup={metrics.lookup_errors}, store={metrics.store_errors}")

    # Final health check
    print("\n" + "=" * 60)
    print("Final Health Check")
    print("=" * 60)
    health = cache.health_check()
    print(f"Status: {health['status']}")
    for check_name, check_result in health["checks"].items():
        status = "✅" if check_result["ok"] else "❌"
        print(f"  {status} {check_name}: {check_result.get('details', 'OK')}")


if __name__ == "__main__":
    main()
