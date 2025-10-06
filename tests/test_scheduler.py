"""Tests for background cleanup scheduler."""

import time
import pytest
from memora import Memora, CacheConfig
from memora.scheduler import CleanupScheduler, SchedulerManager


class TestCleanupScheduler:
    """Test CleanupScheduler functionality."""

    def test_scheduler_basic_lifecycle(self):
        """Scheduler should start, run, and stop."""
        call_count = 0

        def dummy_cleanup():
            nonlocal call_count
            call_count += 1
            return call_count

        scheduler = CleanupScheduler(
            cleanup_func=dummy_cleanup,
            interval_seconds=1,
            initial_delay_seconds=0,
        )

        assert not scheduler.is_running()

        scheduler.start()
        assert scheduler.is_running()

        # Wait for at least one cleanup
        time.sleep(1.5)

        assert call_count >= 1

        scheduler.stop()
        assert not scheduler.is_running()

    def test_scheduler_initial_delay(self):
        """Scheduler should respect initial delay."""
        call_count = 0

        def dummy_cleanup():
            nonlocal call_count
            call_count += 1
            return 0

        scheduler = CleanupScheduler(
            cleanup_func=dummy_cleanup,
            interval_seconds=10,
            initial_delay_seconds=1,
        )

        scheduler.start()

        # Should not run immediately
        time.sleep(0.5)
        assert call_count == 0

        # Should run after initial delay
        time.sleep(0.7)
        assert call_count == 1

        scheduler.stop()

    def test_scheduler_tracks_statistics(self):
        """Scheduler should track runs and deletions."""
        deleted_per_run = [5, 3, 7, 0]
        run_index = 0

        def cleanup_with_results():
            nonlocal run_index
            result = (
                deleted_per_run[run_index] if run_index < len(deleted_per_run) else 0
            )
            run_index += 1
            return result

        scheduler = CleanupScheduler(
            cleanup_func=cleanup_with_results,
            interval_seconds=0.5,
            initial_delay_seconds=0,
        )

        scheduler.start()

        # Wait for multiple runs
        time.sleep(2.5)

        scheduler.stop()

        stats = scheduler.get_stats()

        assert stats["running"] == False
        assert stats["total_runs"] >= 3
        assert stats["total_deleted"] == sum(deleted_per_run[: stats["total_runs"]])
        assert stats["last_run_time"] is not None
        assert stats["errors"] == 0

    def test_scheduler_handles_errors(self):
        """Scheduler should continue running after cleanup errors."""
        call_count = 0

        def failing_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated error")
            return 1

        scheduler = CleanupScheduler(
            cleanup_func=failing_cleanup,
            interval_seconds=0.5,
            initial_delay_seconds=0,
        )

        scheduler.start()

        # Wait for multiple runs (including the failing one)
        time.sleep(2)

        scheduler.stop()

        stats = scheduler.get_stats()

        # Should have continued despite error
        assert stats["total_runs"] >= 3
        assert stats["errors"] >= 1

    def test_scheduler_stop_timeout(self):
        """Scheduler stop should timeout if thread doesn't respond."""

        def long_cleanup():
            time.sleep(10)  # Simulate stuck cleanup
            return 0

        scheduler = CleanupScheduler(
            cleanup_func=long_cleanup,
            interval_seconds=0.1,
            initial_delay_seconds=0,
        )

        scheduler.start()
        time.sleep(0.2)  # Let it start cleanup

        # Stop with short timeout
        start = time.time()
        scheduler.stop(timeout=1.0)
        elapsed = time.time() - start

        # Should timeout quickly
        assert elapsed < 2.0

    def test_scheduler_context_manager(self):
        """Scheduler should work as context manager."""
        call_count = 0

        def dummy_cleanup():
            nonlocal call_count
            call_count += 1
            return 0

        with CleanupScheduler(
            cleanup_func=dummy_cleanup,
            interval_seconds=0.5,
            initial_delay_seconds=0,
        ) as scheduler:
            assert scheduler.is_running()
            time.sleep(1.5)

        # Should auto-stop on exit
        assert not scheduler.is_running()
        assert call_count >= 1


class TestSchedulerManager:
    """Test SchedulerManager functionality."""

    def test_manager_multiple_schedulers(self):
        """Manager should handle multiple schedulers."""
        counts = {"fast": 0, "slow": 0}

        def fast_cleanup():
            counts["fast"] += 1
            return 1

        def slow_cleanup():
            counts["slow"] += 1
            return 2

        manager = SchedulerManager()
        manager.add_scheduler(
            "fast", fast_cleanup, interval_seconds=0.5, initial_delay_seconds=0
        )
        manager.add_scheduler(
            "slow", slow_cleanup, interval_seconds=1.5, initial_delay_seconds=0
        )

        manager.start_all()

        time.sleep(2.5)

        manager.stop_all()

        # Fast should have run more times
        assert counts["fast"] >= 3
        assert counts["slow"] >= 1
        assert counts["fast"] > counts["slow"]

    def test_manager_get_stats(self):
        """Manager should return stats for all schedulers."""

        def cleanup1():
            return 5

        def cleanup2():
            return 10

        manager = SchedulerManager()
        manager.add_scheduler("scheduler1", cleanup1, interval_seconds=10)
        manager.add_scheduler("scheduler2", cleanup2, interval_seconds=20)

        stats = manager.get_stats()

        assert "scheduler1" in stats
        assert "scheduler2" in stats
        assert stats["scheduler1"]["interval_seconds"] == 10
        assert stats["scheduler2"]["interval_seconds"] == 20

    def test_manager_context_manager(self):
        """Manager should work as context manager."""
        call_count = 0

        def dummy_cleanup():
            nonlocal call_count
            call_count += 1
            return 0

        with SchedulerManager() as manager:
            manager.add_scheduler(
                "test", dummy_cleanup, interval_seconds=0.5, initial_delay_seconds=0
            )
            manager.start_all()
            time.sleep(1.5)

        # Should auto-stop on exit
        assert call_count >= 1


class TestMemoraSchedulerIntegration:
    """Test scheduler integration with Memora."""

    def test_memora_start_stop_scheduler(self):
        """Memora should start and stop scheduler."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=1,
            log_level="WARNING",
        )
        cache = Memora(config)

        # Store some entries
        cache.store("q1", {"agent": "test"}, "r1")
        cache.store("q2", {"agent": "test"}, "r2")

        assert cache.backend.count() == 2

        # Start scheduler with short interval
        cache.start_scheduler(interval_seconds=2, initial_delay_seconds=1)

        assert cache.scheduler is not None
        assert cache.scheduler.is_running()

        # Wait for entries to expire and cleanup to run
        time.sleep(3)

        # Entries should be cleaned up
        assert cache.backend.count() == 0

        cache.stop_scheduler()
        assert not cache.scheduler.is_running()

    def test_memora_scheduler_stats(self):
        """Memora should expose scheduler stats."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=0.5,
            log_level="WARNING",
        )
        cache = Memora(config)

        # No scheduler yet
        assert cache.get_scheduler_stats() is None

        cache.start_scheduler(interval_seconds=1, initial_delay_seconds=0)

        # Wait for at least one run
        time.sleep(1.5)

        stats = cache.get_scheduler_stats()
        assert stats is not None
        assert stats["running"] == True
        assert stats["total_runs"] >= 1

        cache.stop_scheduler()

    def test_memora_scheduler_in_get_stats(self):
        """Scheduler stats should appear in get_stats()."""
        cache = Memora(CacheConfig(db_uri="memory://", log_level="WARNING"))

        stats = cache.get_stats()
        assert "scheduler" not in stats

        cache.start_scheduler(interval_seconds=10)

        stats = cache.get_stats()
        assert "scheduler" in stats
        assert stats["scheduler"]["running"] == True

        cache.stop_scheduler()

    def test_memora_scheduler_in_health_check(self):
        """Health check should include scheduler status."""
        cache = Memora(CacheConfig(db_uri="memory://", log_level="WARNING"))

        health = cache.health_check()
        assert health["checks"]["scheduler"]["ok"] == True
        assert "Not running" in health["checks"]["scheduler"]["details"]

        cache.start_scheduler(interval_seconds=10)

        health = cache.health_check()
        assert health["checks"]["scheduler"]["ok"] == True
        assert "Running" in health["checks"]["scheduler"]["details"]

        cache.stop_scheduler()

    def test_memora_context_manager_stops_scheduler(self):
        """Context manager should auto-stop scheduler."""
        config = CacheConfig(db_uri="memory://", log_level="WARNING")

        with Memora(config) as cache:
            cache.start_scheduler(interval_seconds=10)
            assert cache.scheduler.is_running()

        # Should auto-stop on exit
        assert not cache.scheduler.is_running()

    def test_memora_scheduler_without_ttl_warning(self):
        """Starting scheduler without TTL should log warning."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=None,  # No TTL
            log_level="WARNING",
        )
        cache = Memora(config)

        # Should start but log warning
        cache.start_scheduler(interval_seconds=10)

        assert cache.scheduler.is_running()

        cache.stop_scheduler()

    def test_memora_scheduler_already_running_warning(self):
        """Starting scheduler twice should log warning."""
        cache = Memora(CacheConfig(db_uri="memory://", log_level="WARNING"))

        cache.start_scheduler(interval_seconds=10)
        cache.start_scheduler(interval_seconds=5)  # Should warn

        assert cache.scheduler.is_running()

        cache.stop_scheduler()

    def test_memora_cleanup_actually_works(self):
        """Scheduler should actually delete expired entries."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=0.5,
            log_level="WARNING",
        )
        cache = Memora(config)

        # Store entries
        for i in range(10):
            cache.store(f"query {i}", {"agent": "test"}, f"result {i}")

        assert cache.backend.count() == 10

        # Wait for expiration
        time.sleep(0.6)

        # Start scheduler
        cache.start_scheduler(interval_seconds=1, initial_delay_seconds=0)

        # Wait for cleanup
        time.sleep(1.5)

        # All should be deleted
        assert cache.backend.count() == 0

        stats = cache.get_scheduler_stats()
        assert stats["total_deleted"] == 10

        cache.stop_scheduler()
