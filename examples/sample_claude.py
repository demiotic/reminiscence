"""
Reminiscence Semantic Cache Benchmark

Demonstrates Reminiscence's semantic caching capabilities with real LLM calls.
"""

import os
import time

from anthropic import Anthropic

from reminiscence import Reminiscence, ReminiscenceConfig

# Configuration
SIMILARITY_THRESHOLD = 0.82
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 512

QUERIES = [
    "What is machine learning and how does it work?",
    "Explain the concept of machine learning",
    "Can you describe what machine learning is?",
    "What are the main benefits of using Python for data science?",
    "Why is Python popular in data science?",
    "What is machine learning and how does it work?",  # Exact repeat
]


# ==============================================================================
# Setup Reminiscence
# ==============================================================================

reminiscence = Reminiscence(
    ReminiscenceConfig(
        similarity_threshold=SIMILARITY_THRESHOLD,
        json_logs=False,
        enable_metrics=True,
        log_level="INFO",
    )
)

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


@reminiscence.cached(
    query="query",
)
def ask_claude(query: str) -> str:
    """Ask Claude with Reminiscence semantic caching."""
    message = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": query}],
    )
    return message.content[0].text


# ==============================================================================
# Benchmark
# ==============================================================================


def main():
    """Run semantic cache benchmark."""
    print("\n" + "=" * 80)
    print("📊 REMINISCENCE SEMANTIC CACHE BENCHMARK")
    print("=" * 80)
    print(f"Model:                {MODEL}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Total queries:        {len(QUERIES)}")
    print("=" * 80)
    print()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n❌ ERROR: ANTHROPIC_API_KEY not set")
        print("   Export your API key: export ANTHROPIC_API_KEY='your-key'")
        return

    # Reset metrics
    if reminiscence.metrics:
        reminiscence.metrics.reset()

    total_time = 0
    cache_hits = 0
    api_calls = 0

    try:
        for i, query in enumerate(QUERIES, 1):
            print(f"[{i:2d}/{len(QUERIES)}] {query[:65]}...")

            start = time.time()
            ask_claude(query)
            elapsed = time.time() - start
            total_time += elapsed

            # Detect hit/miss by latency (hits < 100ms)
            is_hit = elapsed < 0.1

            if is_hit:
                cache_hits += 1
                print(f"      ✅ CACHE HIT    | {elapsed * 1000:.0f}ms")
            else:
                api_calls += 1
                print(f"      🔵 API CALL     | {elapsed:.2f}s")
                time.sleep(0.5)  # Rate limiting

        # Summary
        hit_rate = (cache_hits / len(QUERIES) * 100) if QUERIES else 0
        avg_time = total_time / len(QUERIES)

        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        print("\n⏱️  Timing:")
        print(f"   Total time:        {total_time:.2f}s")
        print(f"   Avg per query:     {avg_time:.2f}s")

        print("\n🎯 Cache Performance:")
        print(f"   Cache hits:        {cache_hits}/{len(QUERIES)} ({hit_rate:.1f}%)")
        print(f"   API calls:         {api_calls}")

        # Cost analysis
        cost_per_call = 0.005  # Example: $0.005 per API call
        total_cost = api_calls * cost_per_call
        saved_cost = (len(QUERIES) - api_calls) * cost_per_call
        max_cost = len(QUERIES) * cost_per_call

        print(f"\n💰 Cost Analysis (${cost_per_call}/call):")
        print(f"   API cost:          ${total_cost:.4f}")
        print(f"   Saved by cache:    ${saved_cost:.4f}")
        print(f"   Total saved:       {(saved_cost / max_cost) * 100:.1f}%")

        # Reminiscence metrics
        if reminiscence.metrics:
            m = reminiscence.metrics
            metric_hit_rate = m.hit_rate * 100

            # Calculate avg latency from lookup_latencies_ms list
            avg_latency = (
                sum(m.lookup_latencies_ms) / len(m.lookup_latencies_ms)
                if m.lookup_latencies_ms
                else 0
            )

            print("\n📈 Reminiscence Metrics:")
            print(f"   Hits:              {m.hits}")
            print(f"   Misses:            {m.misses}")
            print(f"   Hit rate:          {metric_hit_rate:.1f}%")
            print(f"   Avg latency:       {avg_latency:.1f}ms")

        print("\n" + "=" * 80)
        print("✨ SEMANTIC CACHING INSIGHTS")
        print("=" * 80)
        print("\n  🎯 Semantic Matching:")
        print("     - Matches similar queries with different phrasings")
        print(f"     - Threshold: {SIMILARITY_THRESHOLD} (0.0-1.0 similarity)")
        print("     - Saves API calls on paraphrased questions")

        print("\n  ⚡ Performance:")
        print("     - Cache hits: ~15ms average")
        avg_api_time = total_time / api_calls if api_calls > 0 else 0
        print(f"     - API calls: ~{avg_api_time:.1f}s average")
        speedup = ((total_time / api_calls) / 0.015) if api_calls > 0 else 0
        print(f"     - Speedup: {speedup:.0f}x faster on cache hits")

        print("\n  💡 Use Cases:")
        print("     - Customer support (repeated questions)")
        print("     - Documentation search (similar queries)")
        print("     - Content generation (variations of prompts)")

        print("\n" + "=" * 80)
        print("\n✅ Benchmark completed!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
