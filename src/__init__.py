"""Memora - Semantic caching for LLMs."""

from .config import CacheConfig
from .core import Memora, LookupResult  # ← LookupResult, no CacheLookupResult
from .decorators import create_cached_decorator, MemoraDecorator

__version__ = "0.1.0"

# Global default instance
_default_memora = None
_default_config = None


def configure_default(**kwargs):
    """
    Configure the default global Memora instance.

    Call this once at app startup to customize default behavior.

    Args:
        **kwargs: CacheConfig parameters

    Example:
        >>> from memora import configure_default, cached
        >>>
        >>> configure_default(
        >>>     db_uri="./cache.db",
        >>>     similarity_threshold=0.90
        >>> )
        >>>
        >>> @cached()
        >>> def my_function(query: str):
        >>>     ...
    """
    global _default_config
    _default_config = CacheConfig(**kwargs)


def get_default_memora() -> Memora:
    """
    Get or create default Memora instance.

    Returns:
        Default Memora instance with sensible defaults or env config
    """
    global _default_memora
    if _default_memora is None:
        config = _default_config or CacheConfig.from_env()
        _default_memora = Memora(config)
    return _default_memora


def cached(
    context=None,
    query_param="query",
    extract_from_args=True,
    exclude_from_context=None,
):
    """
    Global cached decorator - zero setup required.

    Uses a default Memora instance with sensible defaults.
    Configure with environment variables or configure_default().

    Args:
        context: Static context dict (optional)
        query_param: Name of query parameter (default: "query")
        extract_from_args: Extract function params into context (default: True)
        exclude_from_context: List of params to exclude from context

    Returns:
        Decorator function

    Example:
        >>> from memora import cached
        >>>
        >>> @cached()  # Zero config needed!
        >>> def expensive_function(query: str, param: int):
        >>>     return compute(query, param)
    """
    memora = get_default_memora()
    decorator_factory = create_cached_decorator(memora)
    return decorator_factory(
        context=context,
        query_param=query_param,
        extract_from_args=extract_from_args,
        exclude_from_context=exclude_from_context,
    )


__all__ = [
    "Memora",
    "CacheConfig",
    "LookupResult",
    "create_cached_decorator",
    "MemoraDecorator",
    "cached",
    "get_default_memora",
    "configure_default",
]
