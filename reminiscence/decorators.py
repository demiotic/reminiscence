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
        >>> @cached(query_param="prompt", strict_params=["model"])
        >>> def call_llm(prompt: str, model: str):
        >>>     return expensive_llm_call(prompt, model)
    """

    def decorator(
        query_param: str = "query",
        strict_params: Optional[List[str]] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
    ) -> Callable[[F], F]:
        """
        Decorator to cache function results with hybrid matching.

        Args:
            query_param: Name of the query parameter for semantic search
            strict_params: Parameters that must match exactly (agent_id, tools, model, etc)
            static_context: Static context dict (merged with extracted params)
            auto_strict: If True, auto-detect non-string params as strict

        Returns:
            Decorated function

        Example:
            >>> # Semantic query + exact model/tools matching
            >>> @cached(
            ...     query_param="user_input",
            ...     strict_params=["agent_id", "tools"],
            ...     static_context={"model": "gpt-4"}
            ... )
            >>> def call_agent(user_input: str, agent_id: str, tools: List[Dict]):
            ...     return expensive_call(user_input, agent_id, tools)
            >>>
            >>> # Auto-detect: non-strings become strict
            >>> @cached(query_param="prompt", auto_strict=True)
            >>> def ask_llm(prompt: str, temperature: float, max_tokens: int):
            ...     # temperature and max_tokens auto-detected as strict
            ...     return llm_call(prompt, temperature, max_tokens)
        """

        def decorator_func(func: F) -> F:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # Validate query_param exists
            if query_param not in params:
                raise ValueError(
                    f"Parameter '{query_param}' not found in {func.__name__}. "
                    f"Available parameters: {params}"
                )

            # Determine strict params with correct precedence
            if strict_params is not None:
                # Explicit strict_params takes precedence
                strict_list = strict_params
            elif auto_strict:
                # Auto-detect non-string params as strict
                detected_strict = []
                for name, param in sig.parameters.items():
                    if name in {query_param, "self", "cls"}:
                        continue
                    ann = param.annotation
                    if ann not in {str, "str", inspect.Parameter.empty}:
                        detected_strict.append(name)
                strict_list = detected_strict
            else:
                # No strict params - pure semantic matching
                strict_list = []

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # Extract query value
                query_value = bound.arguments.get(query_param)
                if query_value is None:
                    raise ValueError(
                        f"Parameter '{query_param}' is None. "
                        f"Must provide a value for '{query_param}'."
                    )

                # Build cache context (exact matching)
                cache_context = {}

                # Add static context first
                if static_context is not None:
                    cache_context.update(static_context)

                # Extract strict params
                for param in strict_list:
                    value = bound.arguments.get(param)
                    if value is not None:
                        cache_context[param] = _serialize_strict(value)

                # If no context at all, add function name for disambiguation
                if not cache_context:
                    cache_context = {"__function__": func.__name__}

                # Check cache
                result = reminiscence.lookup(query_value, cache_context)

                if result.is_hit:
                    return result.result

                # Cache miss - execute function
                output = func(*args, **kwargs)

                # Store in cache
                reminiscence.store(query_value, cache_context, output)

                return output

            # Handle async functions
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs) -> Any:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    # Extract query value
                    query_value = bound.arguments.get(query_param)
                    if query_value is None:
                        raise ValueError(
                            f"Parameter '{query_param}' is None. "
                            f"Must provide a value for '{query_param}'."
                        )

                    # Build cache context (same logic as sync)
                    cache_context = {}

                    if static_context is not None:
                        cache_context.update(static_context)

                    for param in strict_list:
                        value = bound.arguments.get(param)
                        if value is not None:
                            cache_context[param] = _serialize_strict(value)

                    if not cache_context:
                        cache_context = {"__function__": func.__name__}

                    # Check cache
                    result = reminiscence.lookup(query_value, cache_context)

                    if result.is_hit:
                        return result.result

                    # Cache miss - execute async function
                    output = await func(*args, **kwargs)

                    # Store in cache
                    reminiscence.store(query_value, cache_context, output)

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
        ...     query_param="prompt",
        ...     strict_params=["model", "agent_id"]
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
        query_param: str = "query",
        strict_params: Optional[List[str]] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
    ) -> Callable[[F], F]:
        """
        Create a cached decorator with hybrid matching.

        Args:
            query_param: Name of the query parameter for semantic search
            strict_params: Parameters that must match exactly
            static_context: Static context dict
            auto_strict: Auto-detect non-string params as strict

        Returns:
            Decorator function
        """
        return self._cached_decorator(
            query_param=query_param,
            strict_params=strict_params,
            static_context=static_context,
            auto_strict=auto_strict,
        )
