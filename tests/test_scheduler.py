"""Tests for background cleanup scheduler."""

import time
import pytest
from reminiscence import Reminiscence, ReminiscenceConfig
from reminiscence.scheduler import CleanupScheduler, SchedulerManager


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

        # Fixed assertions
        assert not stats["running"]
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


class TestReminiscenceSchedulerIntegration:
    """Test scheduler integration with Reminiscence."""

    def test_reminiscence_start_stop_scheduler(self):
        """Reminiscence should start and stop schedulers."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=1,
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # Store some entries
        cache.store("q1", {"agent": "test"}, "r1")
        cache.store("q2", {"agent": "test"}, "r2")

        assert cache.backend.count() == 2

        # Start scheduler with short interval
        cache.start_scheduler(interval_seconds=2, initial_delay_seconds=1)

        assert cache.scheduler_manager is not None
        assert "cache_cleanup" in cache.scheduler_manager.schedulers
        assert cache.scheduler_manager.schedulers["cache_cleanup"].is_running()

        # Wait for entries to expire and cleanup to run
        time.sleep(3)

        # Entries should be cleaned up
        assert cache.backend.count() == 0

        cache.stop_scheduler()
        assert not cache.scheduler_manager.schedulers["cache_cleanup"].is_running()

    def test_reminiscence_scheduler_stats(self):
        """Reminiscence should expose scheduler stats."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=0.5,
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # No scheduler yet
        assert cache.get_scheduler_stats() is None

        cache.start_scheduler(interval_seconds=1, initial_delay_seconds=0)

        # Wait for at least one run
        time.sleep(1.5)

        stats = cache.get_scheduler_stats()
        assert stats is not None
        assert "cache_cleanup" in stats
        assert stats["cache_cleanup"]["running"]
        assert stats["cache_cleanup"]["total_runs"] >= 1

        cache.stop_scheduler()

    def test_reminiscence_scheduler_in_get_stats(self):
        """Scheduler stats should appear in get_stats()."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=3600,  # Need TTL to create scheduler
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # Start scheduler
        cache.start_scheduler(interval_seconds=10)

        stats = cache.get_stats()
        assert "schedulers" in stats
        assert "cache_cleanup" in stats["schedulers"]
        assert stats["schedulers"]["cache_cleanup"]["total_runs"] >= 0

        cache.stop_scheduler()

    def test_reminiscence_scheduler_in_health_check(self):
        """Health check should include scheduler status."""
        cache = Reminiscence(
            ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        )

        health = cache.health_check()
        assert health["checks"]["schedulers"]["ok"]
        assert "Not running" in health["checks"]["schedulers"]["details"]

        # Need TTL to create cleanup scheduler
        config = ReminiscenceConfig(
            db_uri="memory://", ttl_seconds=3600, log_level="WARNING"
        )
        cache = Reminiscence(config)
        cache.start_scheduler(interval_seconds=10)

        health = cache.health_check()
        assert health["checks"]["schedulers"]["ok"]
        assert "running" in health["checks"]["schedulers"]["details"].lower()

        cache.stop_scheduler()

    def test_reminiscence_context_manager_stops_scheduler(self):
        """Context manager should auto-stop schedulers."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=3600,  # Need TTL to create scheduler
            log_level="WARNING",
        )

        with Reminiscence(config) as cache:
            cache.start_scheduler(interval_seconds=10)
            assert cache.scheduler_manager is not None
            assert cache.scheduler_manager.schedulers["cache_cleanup"].is_running()

        # Should auto-stop on exit
        assert not cache.scheduler_manager.schedulers["cache_cleanup"].is_running()

    def test_reminiscence_scheduler_without_ttl_warning(self):
        """Starting scheduler without TTL should log warning."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=None,  # No TTL
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # Should start but log warning (no cleanup scheduler created)
        cache.start_scheduler(interval_seconds=10)

        # Manager exists but no cleanup scheduler
        assert cache.scheduler_manager is not None
        assert "cache_cleanup" not in cache.scheduler_manager.schedulers

        cache.stop_scheduler()

    def test_reminiscence_scheduler_already_running_warning(self):
        """Starting scheduler twice should log warning."""
        config = ReminiscenceConfig(
            db_uri="memory://", ttl_seconds=3600, log_level="WARNING"
        )
        cache = Reminiscence(config)

        cache.start_scheduler(interval_seconds=10)
        cache.start_scheduler(interval_seconds=5)  # Should warn and skip

        assert cache.scheduler_manager.schedulers["cache_cleanup"].is_running()

        cache.stop_scheduler()

    @pytest.mark.slow
    def test_reminiscence_cleanup_actually_works(self):
        """Scheduler should actually delete expired entries."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            ttl_seconds=0.5,
            log_level="WARNING",
        )

        try:
            cache = Reminiscence(config)
        except ValueError as e:
            if "Could not load model" in str(e):
                pytest.skip("Model download rate limited - skipping test")
            raise

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

        scheduler_stats = cache.get_scheduler_stats()
        assert scheduler_stats["cache_cleanup"]["total_deleted"] == 10

        cache.stop_scheduler()

    def test_metrics_export_with_scheduler(self):
        """Test that metrics are exported correctly by scheduler."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            otel_enabled=True,
            otel_service_name="test-scheduler-metrics",
            log_level="WARNING",
        )

        cache = Reminiscence(config)

        # Perform some operations
        cache.store("q1", {"agent": "test"}, "r1")
        result = cache.lookup("q1", {"agent": "test"})

        assert result.is_hit

        # Start metrics export scheduler
        cache.start_scheduler(metrics_export_interval_seconds=1)

        # Wait for at least one export
        time.sleep(2)

        # Check scheduler stats
        stats = cache.get_scheduler_stats()

        assert "metrics_export" in stats
        assert stats["metrics_export"]["total_runs"] >= 1

        cache.stop_scheduler()
