"""Test script to verify OpenTelemetry integration with Reminiscence."""

import time

from dotenv import load_dotenv

from reminiscence import Reminiscence, ReminiscenceConfig

load_dotenv()


def main():
    print("🚀 Starting Reminiscence with OpenTelemetry integration...\n")

    config = ReminiscenceConfig.load()

    print("📊 Configuration:")
    print(f"   OpenTelemetry: {config.otel_enabled}")
    print(f"   Endpoint: {config.otel_endpoint}")
    print(f"   Service: {config.otel_service_name}")
    print(f"   Export Interval: {config.otel_export_interval_ms}ms\n")

    # Use context manager - automatically stops scheduler on exit
    with Reminiscence(config) as cache:
        if cache.otel_exporter:
            print("✅ OpenTelemetry exporter initialized\n")
        else:
            print("❌ OpenTelemetry not enabled\n")
            return

        print("✅ Cache ready")
        print(f"   Model: {config.model_name}")
        print(f"   Embedding dim: {cache.embedder.embedding_dim}\n")

        # Start scheduler for automatic exports every 10s
        print("🔧 Starting scheduler...")
        cache.start_scheduler(metrics_export_interval_seconds=10)
        time.sleep(2)
        print("✅ Scheduler started\n")

        # Test data
        contexts = [
            {"agent": "qa", "model": "gpt-4"},
            {"agent": "summarizer", "model": "claude-3"},
            {"agent": "translator", "model": "gpt-4"},
        ]

        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?",
            "How do transformers work?",
            "What is attention mechanism?",
        ]

        # Store entries
        print("📝 Storing entries...")
        for i, query in enumerate(queries):
            context = contexts[i % len(contexts)]
            cache.store(query, context, f"Answer: {query}")
            print(f"   ✓ {query[:40]}...")

        print()

        # Lookup entries (should hit)
        print("🔍 Looking up entries (should HIT)...")
        for i, query in enumerate(queries):
            context = contexts[i % len(contexts)]
            result = cache.lookup(query, context)
            if result.is_hit:
                print(f"   ✅ HIT: {query[:40]}... (sim: {result.similarity:.3f})")

        print()

        # Stats
        stats = cache.get_stats()
        print("📊 Cache Statistics:")
        print(f"   Entries: {stats['total_entries']}")
        print(f"   Hits: {stats['hits']}")
        print(f"   Misses: {stats['misses']}")
        print(f"   Hit rate: {stats['hit_rate']}\n")

        # Wait for automatic exports
        print("⏳ Waiting 30s for automatic exports...")
        for i in range(3):
            time.sleep(10)
            scheduler_stats = cache.get_scheduler_stats()
            if scheduler_stats and "metrics_export" in scheduler_stats:
                exports = scheduler_stats["metrics_export"]["total_runs"]
                print(f"   [{(i + 1) * 10}s] Exports: {exports}")

        # Force flush before exiting context
        print("\n🔄 Forcing final flush...")
        from opentelemetry import metrics as otel_metrics

        provider = otel_metrics.get_meter_provider()
        if hasattr(provider, "force_flush"):
            result = provider.force_flush(timeout_millis=10000)
            print(f"   Flush result: {result}")

        print("   Waiting for final export...")
        time.sleep(5)

    # Scheduler stops automatically here
    print("\n✅ Done! Check your OpenTelemetry backend for metrics.")
    print("   Service name: {config.otel_service_name}")
    print("   Metrics: cache_hits, cache_misses, cache_hit_rate")


if __name__ == "__main__":
    main()
