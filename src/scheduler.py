"""Background cleanup scheduler for automatic cache maintenance."""

import threading
import time
from typing import Optional, Callable
from datetime import datetime

from .utils.logging import get_logger

logger = get_logger(__name__)


class CleanupScheduler:
    """
    Background scheduler for automatic cache cleanup.

    Runs cleanup tasks periodically in a separate thread.
    """

    def __init__(
        self,
        cleanup_func: Callable[[], int],
        interval_seconds: int = 3600,
        initial_delay_seconds: int = 60,
    ):
        """
        Initialize cleanup scheduler.

        Args:
            cleanup_func: Function to call for cleanup (should return count of deleted entries)
            interval_seconds: Time between cleanup runs (default: 1 hour)
            initial_delay_seconds: Delay before first cleanup (default: 1 minute)
        """
        self.cleanup_func = cleanup_func
        self.interval_seconds = interval_seconds
        self.initial_delay_seconds = initial_delay_seconds

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Statistics
        self.total_runs = 0
        self.total_deleted = 0
        self.last_run_time: Optional[float] = None
        self.last_deleted_count = 0
        self.errors = 0

    def start(self):
        """Start the background cleanup scheduler."""
        if self._running:
            logger.warning("scheduler_already_running")
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="memora-cleanup-scheduler", daemon=True
        )
        self._thread.start()

        logger.info(
            "scheduler_started",
            interval_seconds=self.interval_seconds,
            initial_delay_seconds=self.initial_delay_seconds,
        )

    def stop(self, timeout: float = 5.0):
        """
        Stop the background cleanup scheduler.

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)
        """
        if not self._running:
            logger.warning("scheduler_not_running")
            return

        logger.info("scheduler_stopping")
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

            if self._thread.is_alive():
                logger.warning("scheduler_stop_timeout", timeout=timeout)
            else:
                logger.info("scheduler_stopped")

        self._running = False

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        next_run = None
        if self.last_run_time:
            next_run = self.last_run_time + self.interval_seconds
        elif self._running:
            next_run = time.time() + self.initial_delay_seconds

        return {
            "running": self.is_running(),
            "interval_seconds": self.interval_seconds,
            "total_runs": self.total_runs,
            "total_deleted": self.total_deleted,
            "last_run_time": datetime.fromtimestamp(self.last_run_time).isoformat()
            if self.last_run_time
            else None,
            "last_deleted_count": self.last_deleted_count,
            "next_run_time": datetime.fromtimestamp(next_run).isoformat()
            if next_run
            else None,
            "errors": self.errors,
        }

    def _run_loop(self):
        """Main scheduler loop (runs in background thread)."""
        # Initial delay
        if self.initial_delay_seconds > 0:
            logger.debug("scheduler_initial_delay", seconds=self.initial_delay_seconds)
            if self._stop_event.wait(self.initial_delay_seconds):
                return  # Stopped during initial delay

        # Main loop
        while not self._stop_event.is_set():
            try:
                self._run_cleanup()
            except Exception as e:
                self.errors += 1
                logger.error(
                    "scheduler_cleanup_error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    total_errors=self.errors,
                    exc_info=True,
                )

            # Wait for next interval (or until stopped)
            if self._stop_event.wait(self.interval_seconds):
                break  # Stopped during wait

    def _run_cleanup(self):
        """Execute cleanup and track statistics."""
        start_time = time.time()

        logger.debug("scheduler_cleanup_started")

        try:
            deleted_count = self.cleanup_func()

            # Update statistics
            self.total_runs += 1
            self.total_deleted += deleted_count
            self.last_run_time = time.time()
            self.last_deleted_count = deleted_count

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "scheduler_cleanup_completed",
                deleted=deleted_count,
                total_deleted=self.total_deleted,
                total_runs=self.total_runs,
                elapsed_ms=round(elapsed_ms, 1),
            )

        except Exception as e:
            logger.error(
                "scheduler_cleanup_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            raise

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop()
        return False


class SchedulerManager:
    """
    Manager for multiple schedulers with unified lifecycle.

    Useful when you need multiple cleanup tasks with different intervals.
    """

    def __init__(self):
        self.schedulers: dict[str, CleanupScheduler] = {}

    def add_scheduler(
        self,
        name: str,
        cleanup_func: Callable[[], int],
        interval_seconds: int = 3600,
        initial_delay_seconds: int = 60,
    ):
        """Add a scheduler."""
        if name in self.schedulers:
            raise ValueError(f"Scheduler '{name}' already exists")

        scheduler = CleanupScheduler(
            cleanup_func=cleanup_func,
            interval_seconds=interval_seconds,
            initial_delay_seconds=initial_delay_seconds,
        )
        self.schedulers[name] = scheduler

        logger.info("scheduler_added", name=name, interval_seconds=interval_seconds)

    def start_all(self):
        """Start all schedulers."""
        for name, scheduler in self.schedulers.items():
            try:
                scheduler.start()
            except Exception as e:
                logger.error(
                    "scheduler_start_failed", name=name, error=str(e), exc_info=True
                )

    def stop_all(self, timeout: float = 5.0):
        """Stop all schedulers."""
        for name, scheduler in self.schedulers.items():
            try:
                scheduler.stop(timeout=timeout)
            except Exception as e:
                logger.error(
                    "scheduler_stop_failed", name=name, error=str(e), exc_info=True
                )

    def get_stats(self) -> dict:
        """Get statistics for all schedulers."""
        return {
            name: scheduler.get_stats() for name, scheduler in self.schedulers.items()
        }

    def __enter__(self):
        """Context manager support."""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop_all()
        return False
