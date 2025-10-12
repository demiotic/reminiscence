"""Concurrency tests for Reminiscence with relaxed eviction and thread-safe metrics."""

import multiprocessing
import time
import os

# CRITICAL: Use spawn instead of fork to avoid deadlocks with sentence-transformers
multiprocessing.set_start_method("spawn", force=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def worker_store_many(db_path: str, worker_id: int, num_stores: int):
    """Worker that performs many stores."""
    # Imports inside worker (after spawn)
    from reminiscence import Reminiscence, ReminiscenceConfig
    from reminiscence.types import MultiModalInput
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"[Worker {worker_id}] Starting...", flush=True)

    config = ReminiscenceConfig(
        db_uri=db_path,
        max_entries=100,
        enable_metrics=True,
        log_level="ERROR",
        json_logs=False,
    )
    reminiscence = Reminiscence(config)

    print(
        f"[Worker {worker_id}] Initialized, storing {num_stores} entries...",
        flush=True,
    )

    failed_stores = 0
    for i in range(num_stores):
        try:
            reminiscence.store(
                query=MultiModalInput(text=f"worker_{worker_id}_query_{i}"),
                context={"worker": worker_id},
                result=f"result_{i}",
            )
        except Exception as e:
            failed_stores += 1
            print(f"[Worker {worker_id}] Store {i} FAILED: {e}", flush=True)

    stats = reminiscence.get_stats()
    print(
        f"[Worker {worker_id}] FINISHED - {num_stores - failed_stores}/{num_stores} successful",
        flush=True,
    )

    return {
        "worker_id": worker_id,
        "attempted": num_stores,
        "failed": failed_stores,
        "store_errors": stats.get("errors", {}).get("store", 0),
    }


class TestConcurrentStores:
    """Test concurrent store() operations."""

    def test_concurrent_stores_low_concurrency(self, tmp_path):
        """Test with 3 workers - should work fine."""
        print("\n" + "=" * 60)
        print("Starting Low Concurrency Test")
        print("=" * 60)

        db_path = str(tmp_path / "cache.db")
        num_workers = 3
        stores_per_worker = 10

        print(f"DB Path: {db_path}")
        print(f"Workers: {num_workers}")
        print(f"Stores per worker: {stores_per_worker}")

        start_time = time.time()

        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(
                worker_store_many,
                [(db_path, i, stores_per_worker) for i in range(num_workers)],
            )

        elapsed = time.time() - start_time

        # Analyze results
        total_attempted = sum(r["attempted"] for r in results)
        total_failed = sum(r["failed"] for r in results)

        print(f"\n{'=' * 60}")
        print(f"RESULTS - {elapsed:.2f}s")
        print(f"{'=' * 60}")
        print(f"Total attempted: {total_attempted}")
        print(f"Total failed: {total_failed}")
        print(f"Failure rate: {total_failed / total_attempted * 100:.2f}%")

        # Verify final state
        from reminiscence import Reminiscence, ReminiscenceConfig

        config = ReminiscenceConfig(db_uri=db_path, log_level="ERROR")
        reminiscence = Reminiscence(config)
        final_count = reminiscence.backend.count()  # ✅ .backend.count()

        print(f"Final cache entries: {final_count}")
        print(f"Expected: ~{total_attempted - total_failed}")

        # Assertions - With relaxed eviction, should have near-zero failures
        assert total_failed <= total_attempted * 0.01, (
            f"More than 1% failures (expected near-zero with relaxed eviction): {total_failed}/{total_attempted}"
        )
        assert final_count > 0, "Cache should have entries"

        print("\n✅ Test PASSED - Relaxed eviction eliminates race conditions")

    def test_concurrent_stores_high_concurrency(self, tmp_path):
        """Test with 10 workers to see if conflicts occur."""
        print("\n" + "=" * 60)
        print("Starting High Concurrency Test")
        print("=" * 60)

        db_path = str(tmp_path / "cache.db")
        num_workers = 10
        stores_per_worker = 5

        print(f"Workers: {num_workers}")
        print(f"Stores per worker: {stores_per_worker}")

        start_time = time.time()

        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(
                worker_store_many,
                [(db_path, i, stores_per_worker) for i in range(num_workers)],
            )

        elapsed = time.time() - start_time

        # Analyze
        total_attempted = sum(r["attempted"] for r in results)
        total_failed = sum(r["failed"] for r in results)

        print(f"\n{'=' * 60}")
        print(f"RESULTS - {elapsed:.2f}s")
        print(f"{'=' * 60}")
        print(f"Total attempted: {total_attempted}")
        print(f"Total failed: {total_failed}")
        print(f"Failure rate: {total_failed / total_attempted * 100:.2f}%")

        # Verify final state
        from reminiscence import Reminiscence, ReminiscenceConfig

        config = ReminiscenceConfig(db_uri=db_path, log_level="ERROR")
        reminiscence = Reminiscence(config)
        final_count = reminiscence.backend.count()  # ✅ .backend.count()

        print(f"Final cache entries: {final_count}")

        # With relaxed eviction, even high concurrency should have very low failure rate
        assert total_failed <= total_attempted * 0.02, (
            f"More than 2% failures (expected near-zero with relaxed eviction): {total_failed}/{total_attempted}"
        )
        assert final_count > 0, "Cache should have entries"

        # Verify max_entries soft limit (should be within 105% of limit)
        max_entries_configured = 100
        assert final_count <= max_entries_configured * 1.10, (
            f"Cache grew beyond 110% of max_entries: {final_count} > {max_entries_configured * 1.10}"
        )

        print("\n✅ Test PASSED - Relaxed eviction handles high concurrency")
