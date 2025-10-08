"""Decorator utilities for automatic caching with hybrid matching."""

import functools
import inspect
import json
from typing import Any, Callable, Dict, Optional, TypeVar, List

from .core import Reminiscence

F = TypeVar("F", bound=Callable[..., Any])


def _serialize_strict(value: Any) -> Any:
    """
    Serialize value for exact matching in context.

    Converts complex types (lists, dicts, objects) to JSON strings
    for consistent exact matching.

    Args:
        value: Value to serialize

    Returns:
        Serialized value (primitives as-is, complex types as JSON)
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    else:
        try:
            return json.dumps(value, default=str, sort_keys=True)
        except (TypeError, ValueError):
            return repr(value)


def create_cached_decorator(reminiscence: Reminiscence) -> Callable:
    """
    Create a caching decorator bound to a Reminiscence instance.

    Args:
        reminiscence: Reminiscence instance to use for caching

    Returns:
        Decorator function

    Example:
        >>> reminiscence = Reminiscence()
        >>> cached = create_cached_decorator(reminiscence)
        >>>
        >>> @cached(query="prompt", query_mode="semantic", context_params=["model"])
        >>> def call_llm(prompt: str, model: str):
        >>>     return expensive_llm_call(prompt, model)
    """

    def decorator(
        query: str = "query",
        query_mode: str = "semantic",
        context_params: Optional[List[str]] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
        similarity_threshold: Optional[float] = None,
    ) -> Callable[[F], F]:
        """
        Decorator to cache function results with hybrid matching.

        Args:
            query: Name of the query parameter (renamed from query_param)
            query_mode: Query matching strategy ("semantic", "exact", "auto")
            context_params: Parameters for exact context matching
            static_context: Static context dict
            auto_strict: Auto-detect non-string params as context
            similarity_threshold: Minimum similarity score (overrides config)  # ← AÑADIR DOC
        """

        def decorator_func(func: F) -> F:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if query not in params:
                raise ValueError(
                    f"Parameter '{query}' not found in {func.__name__}. "
                    f"Available parameters: {params}"
                )

            if context_params is not None:
                context_list = context_params
            elif auto_strict:
                detected_context = []
                for name, param in sig.parameters.items():
                    if name in {query, "self", "cls"}:
                        continue
                    ann = param.annotation
                    if ann not in {str, "str", inspect.Parameter.empty}:
                        detected_context.append(name)
                context_list = detected_context
            else:
                context_list = []

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                query_value = bound.arguments.get(query)
                if query_value is None:
                    raise ValueError(
                        f"Parameter '{query}' is None. "
                        f"Must provide a value for '{query}'."
                    )

                cache_context = {}

                if static_context is not None:
                    cache_context.update(static_context)

                for param in context_list:
                    value = bound.arguments.get(param)
                    if value is not None:
                        cache_context[param] = _serialize_strict(value)

                if not cache_context:
                    cache_context = {"__function__": func.__name__}

                # Pass similarity_threshold to lookup
                result = reminiscence.lookup(
                    query_value,
                    cache_context,
                    similarity_threshold=similarity_threshold,
                    query_mode=query_mode,
                )

                if result.is_hit:
                    return result.result

                output = func(*args, **kwargs)

                reminiscence.store(
                    query_value, cache_context, output, query_mode=query_mode
                )

                return output

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs) -> Any:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    query_value = bound.arguments.get(query)
                    if query_value is None:
                        raise ValueError(
                            f"Parameter '{query}' is None. "
                            f"Must provide a value for '{query}'."
                        )

                    cache_context = {}

                    if static_context is not None:
                        cache_context.update(static_context)

                    for param in context_list:
                        value = bound.arguments.get(param)
                        if value is not None:
                            cache_context[param] = _serialize_strict(value)

                    if not cache_context:
                        cache_context = {"__function__": func.__name__}

                    # Pass similarity_threshold to lookup
                    result = reminiscence.lookup(
                        query_value,
                        cache_context,
                        similarity_threshold=similarity_threshold,
                        query_mode=query_mode,
                    )

                    if result.is_hit:
                        return result.result

                    output = await func(*args, **kwargs)

                    reminiscence.store(
                        query_value, cache_context, output, query_mode=query_mode
                    )

                    return output

                return async_wrapper

            return wrapper

        return decorator_func

    return decorator


class ReminiscenceDecorator:
    """
    Class-based decorator interface for Reminiscence.

    Provides an alternative API for creating cached decorators.

    Example:
        >>> decorator = ReminiscenceDecorator(reminiscence)
        >>> @decorator.cached(
        ...     query="prompt",
        ...     query_mode="semantic",
        ...     context_params=["model", "agent_id"]
        ... )
        >>> def my_function(prompt: str, model: str, agent_id: str):
        >>>     return expensive_computation(prompt, model, agent_id)
    """

    def __init__(self, reminiscence: Reminiscence):
        """
        Initialize decorator with Reminiscence instance.

        Args:
            reminiscence: Reminiscence instance to use for caching
        """
        self.reminiscence = reminiscence
        self._cached_decorator = create_cached_decorator(reminiscence)

    def cached(
        self,
        query: str = "query",  # ← Renombrado
        query_mode: str = "semantic",  # ← NUEVO
        context_params: Optional[List[str]] = None,  # ← Renombrado
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
    ) -> Callable[[F], F]:
        """
        Create a cached decorator with hybrid matching.

        Args:
            query: Name of the query parameter (renamed from query_param)
            query_mode: Query matching strategy ("semantic", "exact", "auto")
            context_params: Parameters for exact context matching (renamed from strict_params)
            static_context: Static context dict
            auto_strict: Auto-detect non-string params as context

        Returns:
            Decorator function
        """
        return self._cached_decorator(
            query=query,
            query_mode=query_mode,
            context_params=context_params,
            static_context=static_context,
            auto_strict=auto_strict,
        )
