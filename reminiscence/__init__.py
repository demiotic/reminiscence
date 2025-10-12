"""Reminiscence - Semantic cache for LLM results with multimodal support."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reminiscence")
except PackageNotFoundError:
    # Fallback
    __version__ = "dev"

from reminiscence.config import ReminiscenceConfig
from reminiscence.core import Reminiscence
from reminiscence.decorators import ReminiscenceDecorator, create_cached_decorator
from reminiscence.scheduler import CleanupScheduler, SchedulerManager
from reminiscence.types import (
    AvailabilityCheck,
    LookupRequest,
    LookupResult,
    MultiModalInput,
    QueryMode,
    StoreRequest,
)

__all__ = [
    # Core
    "Reminiscence",
    "ReminiscenceConfig",
    # Types
    "MultiModalInput",
    "QueryMode",
    "LookupResult",
    "LookupRequest",
    "StoreRequest",
    "AvailabilityCheck",
    # Decorators
    "create_cached_decorator",
    "ReminiscenceDecorator",
    # Schedulers
    "CleanupScheduler",
    "SchedulerManager",
]
