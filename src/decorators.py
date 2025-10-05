"""Decorator utilities for automatic caching."""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar

from .core import Memora

F = TypeVar("F", bound=Callable[..., Any])


def create_cached_decorator(memora: Memora) -> Callable:
    """
    Create a caching decorator bound to a Memora instance.

    Args:
        memora: Memora instance to use for caching

    Returns:
        Decorator function

    Example:
        >>> memora = Memora()
        >>> cached = create_cached_decorator(memora)
        >>>
        >>> @cached(context={"model": "gpt-4"})
        >>> def expensive_function(query: str):
        >>>     return expensive_computation(query)
    """

    def decorator(
        context: Optional[Dict[str, Any]] = None,
        query_param: str = "query",
        extract_from_args: bool = True,  # ← DEFAULT: True (más seguro)
        exclude_from_context: Optional[list] = None,
    ) -> Callable[[F], F]:
        """
        Decorator to cache function results.

        Args:
            context: Static context dict (optional). Merged with extracted params
            query_param: Name of the query parameter (default: "query")
            extract_from_args: If True, extract function parameters into context (default: True)
            exclude_from_context: List of param names to exclude from extracted context

        Returns:
            Decorated function

        Example:
            >>> # Default: All params affect cache (safest)
            >>> @cached(context={"agent": "sql"})
            >>> def query_db(query: str, limit: int = 10):
            >>>     return run_query(query, limit)
            >>>
            >>> # Explicit opt-out: Ignore function params
            >>> @cached(context={"agent": "sql"}, extract_from_args=False)
            >>> def query_db(query: str, limit: int):
            >>>     return run_query(query, 100)  # limit always 100
        """

        def decorator_func(func: F) -> F:
            # Get function signature
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # Validate query_param exists
            if query_param not in params:
                raise ValueError(
                    f"Parámetro '{query_param}' no encontrado en {func.__name__}. "
                    f"Parámetros disponibles: {params}"
                )

            # Default exclude list
            if exclude_from_context is None:
                default_exclude = [query_param, "self", "cls"]
            else:
                default_exclude = list(exclude_from_context)
                if query_param not in default_exclude:
                    default_exclude.append(query_param)

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Bind arguments to parameters
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # Extract query value
                query_value = bound.arguments.get(query_param)
                if query_value is None:
                    raise ValueError(
                        f"Parámetro '{query_param}' es None. "
                        f"Debe proporcionar un valor para '{query_param}'."
                    )

                # Build cache context
                cache_context = {}

                # Add static context (if provided)
                if context is not None:
                    cache_context.update(context)

                # Extract from args (if enabled)
                if extract_from_args:
                    extracted = {
                        param: value
                        for param, value in bound.arguments.items()
                        if param not in default_exclude and value is not None
                    }
                    # Static context overrides extracted context
                    for key, value in extracted.items():
                        if key not in cache_context:
                            cache_context[key] = value

                # If no context at all, add function name for disambiguation
                if not cache_context:
                    cache_context = {"__function__": func.__name__}

                # Check cache
                result = memora.lookup(query_value, cache_context)

                if result.is_hit:
                    return result.result

                # Cache miss - execute function
                output = func(*args, **kwargs)

                # Store in cache
                memora.store(query_value, cache_context, output)

                return output

            # Handle async functions
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs) -> Any:
                    # Bind arguments to parameters
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    # Extract query value
                    query_value = bound.arguments.get(query_param)
                    if query_value is None:
                        raise ValueError(
                            f"Parámetro '{query_param}' es None. "
                            f"Debe proporcionar un valor para '{query_param}'."
                        )

                    # Build cache context (same logic as sync)
                    cache_context = {}

                    if context is not None:
                        cache_context.update(context)

                    if extract_from_args:
                        extracted = {
                            param: value
                            for param, value in bound.arguments.items()
                            if param not in default_exclude and value is not None
                        }
                        for key, value in extracted.items():
                            if key not in cache_context:
                                cache_context[key] = value

                    if not cache_context:
                        cache_context = {"__function__": func.__name__}

                    # Check cache
                    result = memora.lookup(query_value, cache_context)

                    if result.is_hit:
                        return result.result

                    # Cache miss - execute async function
                    output = await func(*args, **kwargs)

                    # Store in cache
                    memora.store(query_value, cache_context, output)

                    return output

                return async_wrapper

            return wrapper

        return decorator_func

    return decorator


class MemoraDecorator:
    """
    Class-based decorator interface for Memora.

    Provides an alternative API for creating cached decorators.

    Example:
        >>> decorator = MemoraDecorator(memora)
        >>> @decorator.cached(context={"model": "gpt-4"})
        >>> def my_function(query: str):
        >>>     return expensive_computation(query)
    """

    def __init__(self, memora: Memora):
        """
        Initialize decorator with Memora instance.

        Args:
            memora: Memora instance to use for caching
        """
        self.memora = memora
        self._cached_decorator = create_cached_decorator(memora)

    def cached(
        self,
        context: Optional[Dict[str, Any]] = None,
        query_param: str = "query",
        extract_from_args: bool = False,
        exclude_from_context: Optional[list] = None,
    ) -> Callable[[F], F]:
        """
        Create a cached decorator with specified options.

        Args:
            context: Static context dict (optional)
            query_param: Name of the query parameter (default: "query")
            extract_from_args: If True, extract function parameters into context
            exclude_from_context: List of param names to exclude from extracted context

        Returns:
            Decorator function
        """
        return self._cached_decorator(
            context=context,
            query_param=query_param,
            extract_from_args=extract_from_args,
            exclude_from_context=exclude_from_context,
        )
