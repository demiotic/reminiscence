"""Eviction policy abstractions."""

from __future__ import annotations

from typing import Any, Optional

from ..types import EvictionPolicy as EvictionPolicyEnum
from .base import EvictionPolicy as EvictionPolicyBase
from .fifo import FIFOPolicy
from .lfu import LFUPolicy
from .lru import LRUPolicy


def create_eviction_policy(
    policy: EvictionPolicyEnum, metrics: Optional[Any] = None
) -> EvictionPolicyBase:
    """Factory to create eviction policy from enum.

    Args:
        policy: EvictionPolicy enum value.
        metrics: Optional CacheMetrics instance for tracking.

    Returns:
        EvictionPolicy instance.

    Raises:
        ValueError: If policy not supported.
    """
    policies = {
        EvictionPolicyEnum.FIFO: FIFOPolicy,
        EvictionPolicyEnum.LRU: LRUPolicy,
        EvictionPolicyEnum.LFU: LFUPolicy,
    }

    policy_class = policies.get(policy)
    if not policy_class:
        raise ValueError(
            f"Unknown eviction policy: {policy}. Supported: {list(EvictionPolicyEnum)}"
        )

    # mypy doesn't understand that policy_class is a concrete implementation
    return policy_class(metrics=metrics)  # type: ignore[abstract]


__all__ = [
    "EvictionPolicyBase",
    "create_eviction_policy",
]
