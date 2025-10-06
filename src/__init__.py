"""Memora - Semantic cache for LLM results."""

from memora.core import Memora
from memora.config import CacheConfig
from memora.types import LookupResult, AvailabilityCheck
from memora.decorators import create_cached_decorator, MemoraDecorator
from memora.scheduler import CleanupScheduler, SchedulerManager

__version__ = "0.1.0"

__all__ = [
    "Memora",
    "CacheConfig",
    "LookupResult",
    "AvailabilityCheck",
    "create_cached_decorator",
    "MemoraDecorator",
    "CleanupScheduler",
    "SchedulerManager",
]
