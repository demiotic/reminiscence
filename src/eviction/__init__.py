"""Eviction policy abstractions."""

from .base import EvictionPolicy
from .fifo import FIFOPolicy
from .lru import LRUPolicy
from .lfu import LFUPolicy


def create_eviction_policy(policy_name: str) -> EvictionPolicy:
    """Factory to create eviction policy from name."""
    policies = {
        "fifo": FIFOPolicy,
        "lru": LRUPolicy,
        "lfu": LFUPolicy,
    }

    policy_class = policies.get(policy_name.lower())
    if not policy_class:
        raise ValueError(
            f"Unknown eviction policy: {policy_name}. "
            f"Supported: {list(policies.keys())}"
        )

    return policy_class()


__all__ = ["EvictionPolicy", "create_eviction_policy"]
