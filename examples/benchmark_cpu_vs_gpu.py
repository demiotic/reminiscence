"""Benchmark CPU vs GPU performance for embeddings."""

import os
import time
import statistics
from typing import List, Dict
from memora import Memora, CacheConfig


def benchmark_device(device: str, num_queries: int = 50) -> Dict[str, float]:
    """
    Benchmark embedding generation on specified device.

    Args:
        device: 'cpu' or 'cuda'
        num_queries: Number of queries to benchmark

    Returns:
        Dict with timing statistics
    """
    # Force device
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # Create fresh Memora instance
    config = CacheConfig(
        db_uri="memory://",
        log_level="ERROR",
    )
    memora = Memora(config)

    # Warmup
    print(f"\n[{device.upper()}] Warming up...")
    for _ in range(5):
        memora.lookup("warmup query", {"device": device})

    # Benchmark queries
    print(f"[{device.upper()}] Running benchmark with {num_queries} queries...")

    queries = [
        f"Query {i}: What is machine learning and how does it work in practice?"
        for i in range(num_queries)
    ]

    timings: List[float] = []

    start_total = time.time()

    for i, query in enumerate(queries):
        start = time.perf_counter()
        memora.lookup(query, {"device": device, "iteration": i})
        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)  # Convert to ms

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_queries} queries")

    total_time = time.time() - start_total

    # Calculate statistics
    return {
        "device": device,
        "total_time": total_time,
        "avg_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "p95_ms": statistics.quantiles(timings, n=20)[18],
        "queries_per_sec": num_queries / total_time,
    }


def compare_devices(num_queries: int = 50):
    """Compare CPU vs GPU performance."""
    print("=" * 70)
    print("MEMORA CPU vs GPU Benchmark")
    print("=" * 70)
    print(f"\nBenchmarking {num_queries} embedding generations per device...")

    # Benchmark CPU
    cpu_stats = benchmark_device("cpu", num_queries)

    # Benchmark GPU (if available)
    try:
        gpu_stats = benchmark_device("cuda", num_queries)
        has_gpu = True
    except Exception as e:
        print(f"\n⚠️  GPU not available: {e}")
        gpu_stats = None
        has_gpu = False

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n📊 CPU Performance:")
    print(f"  Total time:        {cpu_stats['total_time']:.2f}s")
    print(f"  Avg latency:       {cpu_stats['avg_ms']:.2f}ms")
    print(f"  Median latency:    {cpu_stats['median_ms']:.2f}ms")
    print(f"  Min latency:       {cpu_stats['min_ms']:.2f}ms")
    print(f"  Max latency:       {cpu_stats['max_ms']:.2f}ms")
    print(f"  P95 latency:       {cpu_stats['p95_ms']:.2f}ms")
    print(f"  Throughput:        {cpu_stats['queries_per_sec']:.2f} queries/sec")

    if has_gpu:
        print("\n🚀 GPU Performance:")
        print(f"  Total time:        {gpu_stats['total_time']:.2f}s")
        print(f"  Avg latency:       {gpu_stats['avg_ms']:.2f}ms")
        print(f"  Median latency:    {gpu_stats['median_ms']:.2f}ms")
        print(f"  Min latency:       {gpu_stats['min_ms']:.2f}ms")
        print(f"  Max latency:       {gpu_stats['max_ms']:.2f}ms")
        print(f"  P95 latency:       {gpu_stats['p95_ms']:.2f}ms")
        print(f"  Throughput:        {gpu_stats['queries_per_sec']:.2f} queries/sec")

        # Comparison
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        speedup = cpu_stats["avg_ms"] / gpu_stats["avg_ms"]
        throughput_gain = gpu_stats["queries_per_sec"] / cpu_stats["queries_per_sec"]

        print(f"\n⚡ GPU is {speedup:.2f}x faster than CPU")
        print(f"📈 GPU throughput is {throughput_gain:.2f}x higher than CPU")

        if speedup < 2:
            print("\n💡 Recommendation: CPU is sufficient for this workload")
            print("   GPU overhead not worth it for single/small batch queries")
        elif speedup < 5:
            print("\n💡 Recommendation: GPU beneficial for high-throughput scenarios")
            print("   Use CPU for cost optimization, GPU for performance")
        else:
            print("\n💡 Recommendation: GPU significantly faster")
            print("   Use GPU if available and cost is not a constraint")

    print("\n" + "=" * 70)
    print("CACHE USE CASE ANALYSIS")
    print("=" * 70)

    print("\nFor LLM caching:")
    print("  • CPU latency overhead: ~{cpu_stats['avg_ms']:.0f}ms per lookup")
    print("  • LLM call saved: ~2000ms (typical)")
    print(
        f"  • Cache overhead: {cpu_stats['avg_ms'] / 2000 * 100:.1f}% of LLM call time"
    )
    print("\n✅ Conclusion: CPU overhead is negligible compared to LLM latency")
    print("   CPU is perfectly viable for cache use case")


def main():
    """Run benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Memora CPU vs GPU")
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Number of queries to benchmark (default: 50)",
    )

    args = parser.parse_args()

    compare_devices(num_queries=args.queries)


if __name__ == "__main__":
    main()
