"""Reminiscence - Semantic cache for LLM results."""

from reminiscence.core import Reminiscence
from reminiscence.config import ReminiscenceConfig
from reminiscence.types import LookupResult, AvailabilityCheck
from reminiscence.decorators import create_cached_decorator, ReminiscenceDecorator
from reminiscence.scheduler import CleanupScheduler, SchedulerManager

__version__ = "0.2.0"

__all__ = [
    "Reminiscence",
    "ReminiscenceConfig",
    "LookupResult",
    "AvailabilityCheck",
    "create_cached_decorator",
    "ReminiscenceDecorator",
    "CleanupScheduler",
    "SchedulerManager",
]
