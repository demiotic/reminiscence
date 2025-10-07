"""Concurrency tests for Reminiscence - verify behavior without locking."""

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
                query=f"worker_{worker_id}_query_{i}",
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

        # Assertions
        assert total_failed <= total_attempted * 0.1, (
            f"More than 10% failures: {total_failed}/{total_attempted}"
        )
        assert final_count > 0, "Cache should have entries"

        print("\n✅ Test PASSED - No locking needed for low concurrency")

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

        # Less strict for high concurrency (allow up to 20% failures)
        assert total_failed <= total_attempted * 0.2, (
            f"Too many failures: {total_failed}/{total_attempted}"
        )
        assert final_count > 0, "Cache should have entries"

        print("\n⚠️  Test PASSED - Some conflicts expected at high concurrency")
