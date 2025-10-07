"""Reminiscence - Semantic cache for LLM results."""

from reminiscence.core import Reminiscence
from reminiscence.config import CacheConfig
from reminiscence.types import LookupResult, AvailabilityCheck
from reminiscence.decorators import create_cached_decorator, ReminiscenceDecorator
from reminiscence.scheduler import CleanupScheduler, SchedulerManager

__version__ = "0.1.0"

__all__ = [
    "Reminiscence",
    "CacheConfig",
    "LookupResult",
    "AvailabilityCheck",
    "create_cached_decorator",
    "ReminiscenceDecorator",
    "CleanupScheduler",
    "SchedulerManager",
]
