"""Background cleanup scheduler."""

import threading
import time
from typing import Callable, Dict, Optional, Any

from .utils.logging import get_logger

logger = get_logger(__name__)


class CleanupScheduler:
    """
    Background scheduler for periodic cleanup tasks.

    Runs a cleanup function at regular intervals in a separate thread.
    """

    def __init__(
        self,
        cleanup_func: Callable[[], int],
        interval_seconds: int = 3600,
        initial_delay_seconds: int = 60,
        metrics: Optional[Any] = None,
    ):
        """
        Initialize cleanup scheduler.

        Args:
            cleanup_func: Function to call for cleanup (returns number of deleted items)
            interval_seconds: Time between cleanup runs (default: 3600 = 1 hour)
            initial_delay_seconds: Delay before first cleanup (default: 60)
            metrics: Optional CacheMetrics instance
        """
        self.cleanup_func = cleanup_func
        self.interval_seconds = interval_seconds
        self.initial_delay_seconds = initial_delay_seconds
        self.metrics = metrics

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self.total_runs = 0
        self.total_deleted = 0
        self.errors = 0
        self.last_run_time: Optional[float] = None

    def start(self):
        """Start the cleanup scheduler."""
        if self.is_running():
            logger.warning("scheduler_already_running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CleanupScheduler"
        )
        self._thread.start()

        logger.info(
            "scheduler_started",
            interval_seconds=self.interval_seconds,
            initial_delay_seconds=self.initial_delay_seconds,
        )

    def stop(self, timeout: float = 5.0):
        """
        Stop the cleanup scheduler.

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)
        """
        if not self.is_running():
            return

        logger.info("scheduler_stopping")
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("scheduler_stop_timeout", timeout=timeout)
            else:
                logger.info("scheduler_stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "running": self.is_running(),
            "interval_seconds": self.interval_seconds,
            "total_runs": self.total_runs,
            "total_deleted": self.total_deleted,
            "errors": self.errors,
            "last_run_time": self.last_run_time,
        }

    def _run(self):
        """Main scheduler loop (runs in separate thread)."""
        # Initial delay
        if self.initial_delay_seconds > 0:
            logger.debug("scheduler_initial_delay", seconds=self.initial_delay_seconds)
            if self._stop_event.wait(self.initial_delay_seconds):
                return

        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                deleted = self.cleanup_func()

                self.total_runs += 1
                self.total_deleted += deleted
                self.last_run_time = time.time()

                elapsed = time.time() - start_time

                logger.info(
                    "cleanup_completed",
                    deleted=deleted,
                    elapsed_ms=int(elapsed * 1000),
                    total_runs=self.total_runs,
                )

            except Exception as e:
                self.errors += 1
                logger.error(
                    "cleanup_failed",
                    error=str(e),
                    errors_total=self.errors,
                    exc_info=True,
                )

            # Wait for next interval or stop signal
            if self._stop_event.wait(self.interval_seconds):
                break

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop()
        return False


class SchedulerManager:
    """Manager for multiple cleanup schedulers."""

    def __init__(self, metrics: Optional[Any] = None):
        """
        Initialize scheduler manager.

        Args:
            metrics: Optional CacheMetrics instance to pass to schedulers
        """
        self.schedulers: Dict[str, CleanupScheduler] = {}
        self.metrics = metrics

    def add_scheduler(
        self,
        name: str,
        cleanup_func: Callable[[], int],
        interval_seconds: int = 3600,
        initial_delay_seconds: int = 60,
    ):
        """
        Add a named scheduler.

        Args:
            name: Unique name for the scheduler
            cleanup_func: Function to call for cleanup
            interval_seconds: Time between cleanup runs
            initial_delay_seconds: Delay before first cleanup
        """
        if name in self.schedulers:
            raise ValueError(f"Scheduler '{name}' already exists")

        self.schedulers[name] = CleanupScheduler(
            cleanup_func=cleanup_func,
            interval_seconds=interval_seconds,
            initial_delay_seconds=initial_delay_seconds,
            metrics=self.metrics,
        )

        logger.debug("scheduler_added", name=name, interval_seconds=interval_seconds)

    def start_all(self):
        """Start all schedulers."""
        for name, scheduler in self.schedulers.items():
            logger.debug("starting_scheduler", name=name)
            scheduler.start()

    def stop_all(self, timeout: float = 5.0):
        """
        Stop all schedulers.

        Args:
            timeout: Maximum time to wait for each scheduler to stop
        """
        for name, scheduler in self.schedulers.items():
            logger.debug("stopping_scheduler", name=name)
            scheduler.stop(timeout=timeout)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
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
