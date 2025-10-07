"""
Reminiscence Demo with Fake LLM

Demonstrates semantic caching without external dependencies.
Uses a simulated LLM to show cache behavior.
"""

import time
import random
from reminiscence import Reminiscence, ReminiscenceConfig


# ==============================================================================
# Fake LLM Simulation
# ==============================================================================


class FakeLLM:
    """Simulates an LLM with realistic latency and token generation."""

    def __init__(self, latency_seconds: float = 2.0):
        self.latency = latency_seconds
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Simulate LLM generation with delay."""
        self.call_count += 1

        # Simulate network + generation latency
        time.sleep(self.latency + random.uniform(0, 0.5))

        # Generate fake response based on prompt keywords
        if "machine learning" in prompt.lower():
            return (
                "Machine learning is a subset of artificial intelligence that "
                "enables systems to learn and improve from experience without "
                "being explicitly programmed. It uses algorithms to identify "
                "patterns in data and make predictions."
            )
        elif "python" in prompt.lower():
            return (
                "Python is popular in data science due to its simple syntax, "
                "extensive libraries (NumPy, Pandas, scikit-learn), strong "
                "community support, and versatility across different domains."
            )
        else:
            return f"Generated response for: {prompt[:50]}..."


# ==============================================================================
# Setup Reminiscence
# ==============================================================================

llm = FakeLLM(latency_seconds=2.0)

# Load config from environment variables
config = ReminiscenceConfig.load()
config.similarity_threshold = 0.82  # Demo-specific override
# config.enable_metrics = True
reminiscence = Reminiscence(config)


@reminiscence.cached(query_param="query", static_context={"function": "ask_llm"})
def ask_llm(query: str) -> str:
    """Ask LLM with semantic caching."""
    return llm.generate(query)


# ==============================================================================
# Demo
# ==============================================================================


def demo():
    """Demonstrate semantic cache behavior."""
    print("\n" + "=" * 80)
    print("🚀 REMINISCENCE SEMANTIC CACHE DEMO")
    print("=" * 80)
    print("Simulated LLM latency: 2-2.5 seconds per call")
    print("Cache lookup latency:  ~15ms")
    print("=" * 80)
    print()

    queries = [
        "What is machine learning and how does it work?",
        "Explain the concept of machine learning",  # Similar → HIT
        "Can you describe what machine learning is?",  # Similar → HIT
        "What are the main benefits of using Python for data science?",
        "Why is Python popular in data science?",  # Similar → HIT
        "What is machine learning and how does it work?",  # Exact → HIT
    ]

    print("📝 Running queries:\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query}")

        start = time.time()
        result = ask_llm(query)
        elapsed = time.time() - start

        is_hit = elapsed < 0.1

        if is_hit:
            print(f"    ✅ CACHE HIT | {elapsed * 1000:.0f}ms")
        else:
            print(f"    🔵 API CALL  | {elapsed:.2f}s")

        print(f"    Response: {result[:80]}...")
        print()

    # Summary
    print("=" * 80)
    print("📊 RESULTS")
    print("=" * 80)

    if reminiscence.metrics:
        m = reminiscence.metrics
        hit_rate = m.hit_rate * 100

        # Calculate avg latency
        avg_latency = (
            sum(m.lookup_latencies_ms) / len(m.lookup_latencies_ms)
            if m.lookup_latencies_ms
            else 0
        )

        print("\n🎯 Cache Performance:")
        print(f"   Total queries:     {len(queries)}")
        print(f"   Cache hits:        {m.hits}")
        print(f"   Cache misses:      {m.misses}")
        print(f"   Hit rate:          {hit_rate:.1f}%")
        print(f"   Avg lookup time:   {avg_latency:.1f}ms")

        print("\n💰 LLM Calls:")
        print(f"   Total calls:       {llm.call_count}")
        print(f"   Calls saved:       {len(queries) - llm.call_count}")
        print(
            f"   Savings:           {((len(queries) - llm.call_count) / len(queries) * 100):.1f}%"
        )

        # Time savings
        time_without_cache = len(queries) * 2.0
        time_saved = (len(queries) - llm.call_count) * 2.0
        print("\n⚡ Time Savings:")
        print(f"   Without cache:     ~{time_without_cache:.0f}s")
        print(f"   Time saved:        ~{time_saved:.0f}s")
        print(
            f"   Speedup:           {time_without_cache / (llm.call_count * 2.0):.1f}x"
        )

    print("\n" + "=" * 80)
    print("✨ HOW IT WORKS")
    print("=" * 80)
    print("\n  🎯 Semantic Matching:")
    print("     - Query 1: 'What is machine learning...' → MISS (first time)")
    print("     - Query 2: 'Explain the concept...' → HIT (85%+ similar)")
    print("     - Query 3: 'Can you describe...' → HIT (82%+ similar)")
    print("     - Similar questions reuse cached results instantly")

    print("\n  ⚡ Performance:")
    print("     - Cache hits: ~15ms (150x faster)")
    print("     - Saves expensive LLM calls")
    print("     - No exact string matching required")

    print("\n  💡 Perfect For:")
    print("     - Chatbots (repeated questions)")
    print("     - Search (similar queries)")
    print("     - APIs (duplicate requests)")
    print("=" * 80)
    print()


if __name__ == "__main__":
    demo()
