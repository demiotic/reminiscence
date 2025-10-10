"""Decorator utilities for automatic caching with hybrid matching."""

import functools
import inspect
import json
from typing import Any, Callable, Dict, Optional, TypeVar, List, Union

from .core import Reminiscence
from .utils.logging import get_logger

logger = get_logger(__name__)

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


def _normalize_to_batch(value: Any) -> tuple[List[Any], bool]:
    """
    Normalize input to batch format.

    Args:
        value: Single item or list of items

    Returns:
        (list_of_values, was_single_item)
    """
    if isinstance(value, list):
        return value, False
    return [value], True


def _normalize_context_params(params: Union[str, List[str], None]) -> List[str]:
    """
    Normalize context_params to list format.

    Args:
        params: Single param name, list of param names, or None

    Returns:
        List of param names (empty if None)

    Examples:
        >>> _normalize_context_params("model")
        ['model']
        >>> _normalize_context_params(["model", "agent_id"])
        ['model', 'agent_id']
        >>> _normalize_context_params(None)
        []
    """
    if params is None:
        return []
    if isinstance(params, str):
        return [params]
    return params


def create_cached_decorator(reminiscence: Reminiscence) -> Callable:
    """
    Create a caching decorator bound to a Reminiscence instance.

    Uses batch operations by default for optimal performance (~4.5% overhead).

    Args:
        reminiscence: Reminiscence instance to use for caching

    Returns:
        Decorator function

    Example:
        >>> reminiscence = Reminiscence()
        >>> cached = create_cached_decorator(reminiscence)
        >>>
        >>> # Single context param (most common)
        >>> @cached(query="prompt", context_params="model")
        >>> def call_llm(prompt: str, model: str):
        ...     return expensive_llm_call(prompt, model)
        >>>
        >>> # Multiple context params
        >>> @cached(query="prompt", context_params=["model", "agent_id"])
        >>> def call_agent(prompt: str, model: str, agent_id: str):
        ...     return agent_call(prompt, model, agent_id)
    """

    def decorator(
        query: str = "query",
        query_mode: str = "semantic",
        context_params: Union[str, List[str], None] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
        similarity_threshold: Optional[float] = None,
        allow_errors: bool = False,
        batch_mode: bool = True,
    ) -> Callable[[F], F]:
        """
        Decorator to cache function results with hybrid matching.

        Args:
            query: Name of the query parameter
            query_mode: Query matching strategy ("semantic", "exact", "auto")
            context_params: Single param name OR list of param names for context matching
            static_context: Static context dict
            auto_strict: Auto-detect non-string params as context
            similarity_threshold: Minimum similarity score (overrides config)
            allow_errors: If False (default), don't cache error results
            batch_mode: Use batch operations internally (default: True, ~4.5% overhead)

        Returns:
            Decorated function
        """

        def decorator_func(func: F) -> F:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if query not in params:
                raise ValueError(
                    f"Parameter '{query}' not found in {func.__name__}. "
                    f"Available parameters: {params}"
                )

            # Normalize context_params to list
            context_list = _normalize_context_params(context_params)

            logger.debug(
                "decorator_configured",
                function=func.__name__,
                query_param=query,
                query_mode=query_mode,
                context_params=context_list,
                batch_mode=batch_mode,
                similarity_threshold=similarity_threshold,
            )

            # Auto-detect context params if enabled
            if not context_list and auto_strict:
                detected_context = []
                for name, param in sig.parameters.items():
                    if name in {query, "self", "cls"}:
                        continue
                    ann = param.annotation
                    if ann not in {str, "str", inspect.Parameter.empty}:
                        detected_context.append(name)
                context_list = detected_context

                logger.debug(
                    "auto_strict_detected",
                    function=func.__name__,
                    detected_params=context_list,
                )

            def build_context(bound_args) -> Dict[str, Any]:
                """Build cache context from bound arguments."""
                cache_context = {}

                if static_context is not None:
                    cache_context.update(static_context)

                for param in context_list:
                    value = bound_args.arguments.get(param)
                    if value is not None:
                        cache_context[param] = _serialize_strict(value)

                if not cache_context:
                    cache_context = {"__function__": func.__name__}

                return cache_context

            # Batch mode implementation
            if batch_mode:

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

                    # Build context (shared for all queries)
                    cache_context = build_context(bound)

                    # Normalize to batch format
                    queries, is_single = _normalize_to_batch(query_value)

                    logger.debug(
                        "decorator_batch_call",
                        function=func.__name__,
                        is_single=is_single,
                        num_queries=len(queries),
                        query_preview=queries[0][:50] if queries else "",
                        context=cache_context,
                        query_mode=query_mode,
                    )

                    # Batch lookup (always use batch internally)
                    lookup_start = __import__("time").time()
                    results = reminiscence.lookup_batch(
                        queries,
                        cache_context,  # Shared context
                        similarity_threshold=similarity_threshold,
                        query_mode=query_mode,
                    )
                    lookup_ms = (__import__("time").time() - lookup_start) * 1000

                    logger.debug(
                        "decorator_lookup_batch_complete",
                        function=func.__name__,
                        num_results=len(results),
                        latency_ms=round(lookup_ms, 2),
                    )

                    # Check if all are cache hits
                    cached_results = {}
                    missing_indices = []

                    for i, result in enumerate(results):
                        if result.is_hit:
                            cached_results[i] = result.result
                            logger.debug(
                                "decorator_cache_hit",
                                function=func.__name__,
                                index=i,
                                query_preview=queries[i][:50],
                                similarity=round(result.similarity, 3)
                                if result.similarity
                                else None,
                            )
                        else:
                            missing_indices.append(i)
                            logger.debug(
                                "decorator_cache_miss",
                                function=func.__name__,
                                index=i,
                                query_preview=queries[i][:50],
                            )

                    # All hits - return immediately
                    if not missing_indices:
                        logger.info(
                            "decorator_all_cache_hits",
                            function=func.__name__,
                            num_queries=len(queries),
                            is_single=is_single,
                        )
                        all_results = [cached_results[i] for i in range(len(queries))]
                        return all_results[0] if is_single else all_results

                    # Execute function for missing items
                    logger.info(
                        "decorator_executing_function",
                        function=func.__name__,
                        missing_count=len(missing_indices),
                        total_queries=len(queries),
                        is_single=is_single,
                    )

                    try:
                        exec_start = __import__("time").time()

                        if is_single:
                            # Original call with single item - ONLY ONE CALL
                            logger.debug(
                                "decorator_executing_single",
                                function=func.__name__,
                                args_preview=str(args)[:100],
                            )
                            output = func(*args, **kwargs)
                            outputs = [output]
                        else:
                            # Call with only missing queries
                            missing_queries = [queries[i] for i in missing_indices]
                            modified_kwargs = kwargs.copy()
                            modified_kwargs[query] = missing_queries

                            logger.debug(
                                "decorator_executing_batch",
                                function=func.__name__,
                                missing_queries=len(missing_queries),
                            )

                            outputs = func(**modified_kwargs)
                            if not isinstance(outputs, list):
                                outputs = [outputs]

                        exec_ms = (__import__("time").time() - exec_start) * 1000
                        logger.debug(
                            "decorator_function_executed",
                            function=func.__name__,
                            num_outputs=len(outputs),
                            latency_ms=round(exec_ms, 2),
                        )

                        # Store batch (always use batch internally)
                        missing_queries = [queries[i] for i in missing_indices]

                        store_start = __import__("time").time()
                        reminiscence.store_batch(
                            missing_queries,
                            cache_context,  # Shared context
                            outputs,
                            query_mode=query_mode,
                            allow_errors=allow_errors,
                        )
                        store_ms = (__import__("time").time() - store_start) * 1000

                        logger.debug(
                            "decorator_store_batch_complete",
                            function=func.__name__,
                            num_stored=len(missing_queries),
                            latency_ms=round(store_ms, 2),
                        )

                        # Merge cached + new results
                        if is_single:
                            logger.info(
                                "decorator_return_single",
                                function=func.__name__,
                                was_cached=0 in cached_results,
                            )
                            return outputs[0]
                        else:
                            final_results = []
                            outputs_iter = iter(outputs)
                            for i in range(len(queries)):
                                if i in cached_results:
                                    final_results.append(cached_results[i])
                                else:
                                    final_results.append(next(outputs_iter))

                            logger.info(
                                "decorator_return_batch",
                                function=func.__name__,
                                total_results=len(final_results),
                                cached_count=len(cached_results),
                                new_count=len(outputs),
                            )

                            return final_results

                    except Exception as e:
                        logger.error(
                            "decorator_function_error",
                            function=func.__name__,
                            error_type=type(e).__name__,
                            error=str(e),
                            exc_info=True,
                        )
                        raise

            # Non-batch mode (original implementation)
            else:

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

                    cache_context = build_context(bound)

                    logger.debug(
                        "decorator_single_call",
                        function=func.__name__,
                        query_preview=query_value[:50]
                        if isinstance(query_value, str)
                        else str(query_value)[:50],
                        context=cache_context,
                        query_mode=query_mode,
                    )

                    # Single lookup
                    lookup_start = __import__("time").time()
                    result = reminiscence.lookup(
                        query_value,
                        cache_context,
                        similarity_threshold=similarity_threshold,
                        query_mode=query_mode,
                    )
                    lookup_ms = (__import__("time").time() - lookup_start) * 1000

                    if result.is_hit:
                        logger.info(
                            "decorator_cache_hit_single",
                            function=func.__name__,
                            similarity=round(result.similarity, 3)
                            if result.similarity
                            else None,
                            latency_ms=round(lookup_ms, 2),
                        )
                        return result.result

                    # Execute function
                    logger.info(
                        "decorator_executing_function_single",
                        function=func.__name__,
                    )

                    try:
                        exec_start = __import__("time").time()
                        output = func(*args, **kwargs)
                        exec_ms = (__import__("time").time() - exec_start) * 1000

                        logger.debug(
                            "decorator_function_executed_single",
                            function=func.__name__,
                            latency_ms=round(exec_ms, 2),
                        )

                        # Store result
                        store_start = __import__("time").time()
                        reminiscence.store(
                            query_value,
                            cache_context,
                            output,
                            query_mode=query_mode,
                            allow_errors=allow_errors,
                        )
                        store_ms = (__import__("time").time() - store_start) * 1000

                        logger.debug(
                            "decorator_store_complete_single",
                            function=func.__name__,
                            latency_ms=round(store_ms, 2),
                        )

                        return output

                    except Exception as e:
                        logger.error(
                            "decorator_function_error_single",
                            function=func.__name__,
                            error_type=type(e).__name__,
                            error=str(e),
                            exc_info=True,
                        )
                        raise

            # Async version
            if inspect.iscoroutinefunction(func):
                if batch_mode:

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

                        cache_context = build_context(bound)
                        queries, is_single = _normalize_to_batch(query_value)

                        logger.debug(
                            "decorator_async_batch_call",
                            function=func.__name__,
                            is_single=is_single,
                            num_queries=len(queries),
                        )

                        # Batch lookup
                        results = reminiscence.lookup_batch(
                            queries,
                            cache_context,
                            similarity_threshold=similarity_threshold,
                            query_mode=query_mode,
                        )

                        cached_results = {}
                        missing_indices = []

                        for i, result in enumerate(results):
                            if result.is_hit:
                                cached_results[i] = result.result
                            else:
                                missing_indices.append(i)

                        if not missing_indices:
                            logger.info(
                                "decorator_async_all_cache_hits",
                                function=func.__name__,
                                num_queries=len(queries),
                            )
                            all_results = [
                                cached_results[i] for i in range(len(queries))
                            ]
                            return all_results[0] if is_single else all_results

                        logger.info(
                            "decorator_async_executing_function",
                            function=func.__name__,
                            missing_count=len(missing_indices),
                        )

                        try:
                            if is_single:
                                output = await func(*args, **kwargs)
                                outputs = [output]
                            else:
                                missing_queries = [queries[i] for i in missing_indices]
                                modified_kwargs = kwargs.copy()
                                modified_kwargs[query] = missing_queries

                                outputs = await func(**modified_kwargs)
                                if not isinstance(outputs, list):
                                    outputs = [outputs]

                            missing_queries = [queries[i] for i in missing_indices]

                            reminiscence.store_batch(
                                missing_queries,
                                cache_context,
                                outputs,
                                query_mode=query_mode,
                                allow_errors=allow_errors,
                            )

                            if is_single:
                                return outputs[0]
                            else:
                                final_results = []
                                outputs_iter = iter(outputs)
                                for i in range(len(queries)):
                                    if i in cached_results:
                                        final_results.append(cached_results[i])
                                    else:
                                        final_results.append(next(outputs_iter))
                                return final_results

                        except Exception as e:
                            logger.error(
                                "decorator_async_function_error",
                                function=func.__name__,
                                error_type=type(e).__name__,
                                error=str(e),
                                exc_info=True,
                            )
                            raise

                    return async_wrapper

                else:
                    # Original async non-batch implementation
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

                        cache_context = build_context(bound)

                        logger.debug(
                            "decorator_async_single_call",
                            function=func.__name__,
                            query_preview=query_value[:50]
                            if isinstance(query_value, str)
                            else str(query_value)[:50],
                        )

                        result = reminiscence.lookup(
                            query_value,
                            cache_context,
                            similarity_threshold=similarity_threshold,
                            query_mode=query_mode,
                        )

                        if result.is_hit:
                            logger.info(
                                "decorator_async_cache_hit",
                                function=func.__name__,
                            )
                            return result.result

                        logger.info(
                            "decorator_async_executing_function_single",
                            function=func.__name__,
                        )

                        try:
                            output = await func(*args, **kwargs)

                            reminiscence.store(
                                query_value,
                                cache_context,
                                output,
                                query_mode=query_mode,
                                allow_errors=allow_errors,
                            )

                            return output

                        except Exception as e:
                            logger.error(
                                "decorator_async_function_error_single",
                                function=func.__name__,
                                error_type=type(e).__name__,
                                error=str(e),
                                exc_info=True,
                            )
                            raise

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
        >>>
        >>> # Single context param (most common)
        >>> @decorator.cached(query="prompt", context_params="model")
        >>> def my_function(prompt: str, model: str):
        >>>     return expensive_computation(prompt, model)
        >>>
        >>> # Multiple context params
        >>> @decorator.cached(query="query", context_params=["model", "agent_id"])
        >>> def another_function(query: str, model: str, agent_id: str):
        >>>     return expensive_computation(query, model, agent_id)
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
        query: str = "query",
        query_mode: str = "semantic",
        context_params: Union[str, List[str], None] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
        similarity_threshold: Optional[float] = None,
        allow_errors: bool = False,
        batch_mode: bool = True,
    ) -> Callable[[F], F]:
        """
        Create a cached decorator with hybrid matching.

        Args:
            query: Name of the query parameter
            query_mode: Query matching strategy ("semantic", "exact", "auto")
            context_params: Single param name OR list of param names for context
            static_context: Static context dict
            auto_strict: Auto-detect non-string params as context
            similarity_threshold: Minimum similarity score (overrides config)
            allow_errors: If False (default), don't cache error results
            batch_mode: Use batch operations internally (default: True)

        Returns:
            Decorator function
        """
        return self._cached_decorator(
            query=query,
            query_mode=query_mode,
            context_params=context_params,
            static_context=static_context,
            auto_strict=auto_strict,
            similarity_threshold=similarity_threshold,
            allow_errors=allow_errors,
            batch_mode=batch_mode,
        )
