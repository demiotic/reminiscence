"""Example: Using background cleanup scheduler."""

from memora import Memora, CacheConfig
import time

# Setup cache with TTL
config = CacheConfig(
    db_uri="./cache_with_scheduler.lance",
    ttl_seconds=300,  # 5 minutes
    max_entries=10_000,
    eviction_policy="lru",
    log_level="INFO",
)

cache = Memora(config)

# Start background cleanup (runs every hour)
cache.start_scheduler(
    interval_seconds=3600,  # 1 hour
    initial_delay_seconds=60,  # Wait 1 minute before first cleanup
)

print("Scheduler started. Adding entries...")

# Simulate application usage
for i in range(100):
    cache.store(
        f"query {i}", {"agent": "demo", "timestamp": time.time()}, f"result {i}"
    )

print(f"Added 100 entries. Total: {cache.backend.count()}")

# Check scheduler stats
stats = cache.get_scheduler_stats()
print(f"\nScheduler stats: {stats}")

# Keep running (in real app, this would be your main loop)
try:
    print("\nPress Ctrl+C to stop...")
    while True:
        time.sleep(10)

        # Periodically check stats
        stats = cache.get_stats()
        print(f"Entries: {stats['total_entries']}, Hits: {stats.get('hits', 0)}")

except KeyboardInterrupt:
    print("\nStopping scheduler...")
    cache.stop_scheduler()
    print("Done!")
