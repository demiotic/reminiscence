"""Decorator utilities for automatic caching with hybrid matching."""

from __future__ import annotations

import functools
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .core import Reminiscence
from .types import LookupRequest, MultiModalInput, QueryMode, StoreRequest
from .utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _serialize_strict(value: Any) -> Any:
    """Serialize value for exact matching in context.

    Converts complex types (lists, dicts, objects) to JSON strings
    for consistent exact matching.

    Args:
        value: Value to serialize.

    Returns:
        Serialized value (primitives as-is, complex types as JSON).
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
    """Normalize input to batch format.

    Args:
        value: Single item or list of items.

    Returns:
        (list_of_values, was_single_item).
    """
    if isinstance(value, list):
        return value, False
    return [value], True


def _normalize_context(params: Union[str, List[str], None]) -> List[str]:
    """Normalize context to list format.

    Args:
        params: Single param name, list of param names, or None.

    Returns:
        List of param names (empty if None).

    Examples:
        >>> _normalize_context("model")
        ['model']
        >>> _normalize_context(["model", "agent_id"])
        ['model', 'agent_id']
        >>> _normalize_context(None)
        []
    """
    if params is None:
        return []
    if isinstance(params, str):
        return [params]
    return params


def _to_multimodal_input(value: Any) -> MultiModalInput:
    """Convert value to MultiModalInput.

    Accepts MultiModalInput, str, or dict and normalizes to MultiModalInput.

    Args:
        value: Value to convert (MultiModalInput, str, or dict).

    Returns:
        MultiModalInput instance.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, MultiModalInput):
        return value
    if isinstance(value, str):
        return MultiModalInput(text=value)
    if isinstance(value, dict):
        return MultiModalInput(**value)
    raise ValueError(
        f"Cannot convert {type(value).__name__} to MultiModalInput. "
        "Expected MultiModalInput, str, or dict."
    )


def create_cached_decorator(reminiscence: Reminiscence) -> Callable:
    """Create a caching decorator bound to a Reminiscence instance.

    Uses batch operations by default for optimal performance (~4.5% overhead).

    Args:
        reminiscence: Reminiscence instance to use for caching.

    Returns:
        Decorator function.

    Example:
        >>> from reminiscence import Reminiscence, QueryMode
        >>> reminiscence = Reminiscence()
        >>> cached = create_cached_decorator(reminiscence)
        >>>
        >>> # Single context param (most common)
        >>> @cached(query="prompt", context="model")
        >>> def call_llm(prompt: str, model: str):
        ...     return expensive_llm_call(prompt, model)
        >>>
        >>> # Multimodal query
        >>> @cached(query="query", context="model")
        >>> def analyze_image(query: MultiModalInput, model: str):
        ...     return vision_model(query, model)
    """

    def decorator(
        query: str = "query",
        mode: QueryMode = QueryMode.AUTO,
        context: Union[str, List[str], None] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
        similarity_threshold: Optional[float] = None,
        allow_errors: bool = False,
        batch_mode: bool = True,
    ) -> Callable[[F], F]:
        """Decorator to cache function results with hybrid matching.

        Args:
            query: Name of the query parameter.
            mode: Query matching strategy (default: QueryMode.AUTO).
            context: Single param name OR list of param names for context.
            static_context: Static context dict.
            auto_strict: Auto-detect non-string params as context.
            similarity_threshold: Minimum similarity score (overrides config).
            allow_errors: If False (default), don't cache error results.
            batch_mode: Use batch operations internally (default: True).

        Returns:
            Decorated function.
        """

        def decorator_func(func: F) -> F:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if query not in params:
                raise ValueError(
                    f"Parameter '{query}' not found in {func.__name__}. "
                    f"Available parameters: {params}"
                )

            # Normalize context to list
            context_list = _normalize_context(context)

            logger.debug(
                "decorator_configured",
                function=func.__name__,
                query_param=query,
                mode=mode.value,
                context=context_list,
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

            def build_context(bound_args: Any) -> Dict[str, Any]:
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
                def wrapper(*args: Any, **kwargs: Any) -> Any:
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

                    # Convert all queries to MultiModalInput
                    try:
                        multimodal_queries = [_to_multimodal_input(q) for q in queries]
                    except ValueError as e:
                        logger.warning(
                            f"Failed to convert queries to MultiModalInput: {e}. "
                            "Executing without caching."
                        )
                        return func(*args, **kwargs)

                    logger.debug(
                        "decorator_batch_call",
                        function=func.__name__,
                        is_single=is_single,
                        num_queries=len(queries),
                        query_preview=str(multimodal_queries[0])[:50]
                        if multimodal_queries
                        else "",
                        context=cache_context,
                        mode=mode.value,
                    )

                    # Batch lookup with typed requests
                    lookup_start = __import__("time").time()
                    lookup_requests = [
                        LookupRequest(
                            query=q,
                            context=cache_context,
                            similarity_threshold=similarity_threshold,
                            mode=mode,
                        )
                        for q in multimodal_queries
                    ]
                    results = reminiscence.lookup_batch(lookup_requests)
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
                                query_preview=str(multimodal_queries[i])[:50],
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
                                query_preview=str(multimodal_queries[i])[:50],
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
                            logger.debug(
                                "decorator_executing_single",
                                function=func.__name__,
                                args_preview=str(args)[:100],
                            )
                            output = func(*args, **kwargs)
                            outputs = [output]
                        else:
                            # Call with only missing queries (original format, not MultiModalInput)
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

                        # Store batch with typed requests
                        missing_multimodal = [
                            multimodal_queries[i] for i in missing_indices
                        ]

                        store_start = __import__("time").time()
                        store_requests = [
                            StoreRequest(
                                query=missing_multimodal[i],
                                context=cache_context,
                                result=outputs[i],
                            )
                            for i in range(len(missing_multimodal))
                        ]
                        reminiscence.store_batch(store_requests, allow_errors=allow_errors)
                        store_ms = (__import__("time").time() - store_start) * 1000

                        logger.debug(
                            "decorator_store_batch_complete",
                            function=func.__name__,
                            num_stored=len(missing_multimodal),
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
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    query_value = bound.arguments.get(query)
                    if query_value is None:
                        raise ValueError(
                            f"Parameter '{query}' is None. "
                            f"Must provide a value for '{query}'."
                        )

                    # Convert to MultiModalInput
                    try:
                        multimodal_query = _to_multimodal_input(query_value)
                    except ValueError as e:
                        logger.warning(
                            f"Failed to convert query to MultiModalInput: {e}. "
                            "Executing without caching."
                        )
                        return func(*args, **kwargs)

                    cache_context = build_context(bound)

                    logger.debug(
                        "decorator_single_call",
                        function=func.__name__,
                        query_preview=str(multimodal_query)[:50],
                        context=cache_context,
                        mode=mode.value,
                    )

                    # Single lookup
                    lookup_start = __import__("time").time()
                    result = reminiscence.lookup(
                        multimodal_query,
                        cache_context,
                        similarity_threshold=similarity_threshold,
                        mode=mode,
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
                            multimodal_query,
                            cache_context,
                            output,
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
                    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
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

                        # Convert to MultiModalInput
                        try:
                            multimodal_queries = [
                                _to_multimodal_input(q) for q in queries
                            ]
                        except ValueError as e:
                            logger.warning(
                                f"Failed to convert queries: {e}. "
                                "Executing without caching."
                            )
                            return await func(*args, **kwargs)

                        logger.debug(
                            "decorator_async_batch_call",
                            function=func.__name__,
                            is_single=is_single,
                            num_queries=len(queries),
                        )

                        # Batch lookup with typed requests
                        lookup_requests = [
                            LookupRequest(
                                query=q,
                                context=cache_context,
                                similarity_threshold=similarity_threshold,
                                mode=mode,
                            )
                            for q in multimodal_queries
                        ]
                        results = reminiscence.lookup_batch(lookup_requests)

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

                            missing_multimodal = [
                                multimodal_queries[i] for i in missing_indices
                            ]

                            store_requests = [
                                StoreRequest(
                                    query=missing_multimodal[i],
                                    context=cache_context,
                                    result=outputs[i],
                                )
                                for i in range(len(missing_multimodal))
                            ]
                            reminiscence.store_batch(store_requests, allow_errors=allow_errors)

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
                    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                        bound = sig.bind(*args, **kwargs)
                        bound.apply_defaults()

                        query_value = bound.arguments.get(query)
                        if query_value is None:
                            raise ValueError(
                                f"Parameter '{query}' is None. "
                                f"Must provide a value for '{query}'."
                            )

                        # Convert to MultiModalInput
                        try:
                            multimodal_query = _to_multimodal_input(query_value)
                        except ValueError as e:
                            logger.warning(f"Failed to convert query: {e}")
                            return await func(*args, **kwargs)

                        cache_context = build_context(bound)

                        logger.debug(
                            "decorator_async_single_call",
                            function=func.__name__,
                            query_preview=str(multimodal_query)[:50],
                        )

                        result = reminiscence.lookup(
                            multimodal_query,
                            cache_context,
                            similarity_threshold=similarity_threshold,
                            mode=mode,
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
                                multimodal_query,
                                cache_context,
                                output,
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
    """Class-based decorator interface for Reminiscence.

    Provides an alternative API for creating cached decorators.

    Example:
        >>> from reminiscence import Reminiscence, ReminiscenceDecorator
        >>> reminiscence = Reminiscence()
        >>> decorator = ReminiscenceDecorator(reminiscence)
        >>>
        >>> # Text query
        >>> @decorator.cached(query="prompt", context="model")
        >>> def my_function(prompt: str, model: str):
        ...     return expensive_computation(prompt, model)
        >>>
        >>> # Multimodal query
        >>> @decorator.cached(query="query", context="model")
        >>> def analyze(query: MultiModalInput, model: str):
        ...     return vision_model(query, model)
    """

    def __init__(self, reminiscence: Reminiscence):
        """Initialize decorator with Reminiscence instance.

        Args:
            reminiscence: Reminiscence instance to use for caching.
        """
        self.reminiscence = reminiscence
        self._cached_decorator = create_cached_decorator(reminiscence)

    def cached(
        self,
        query: str = "query",
        mode: QueryMode = QueryMode.AUTO,
        context: Union[str, List[str], None] = None,
        static_context: Optional[Dict[str, Any]] = None,
        auto_strict: bool = False,
        similarity_threshold: Optional[float] = None,
        allow_errors: bool = False,
        batch_mode: bool = True,
    ) -> Callable[[F], F]:
        """Create a cached decorator with hybrid matching.

        Args:
            query: Name of the query parameter.
            mode: Query matching strategy (default: QueryMode.AUTO).
            context: Single param name OR list of param names for context.
            static_context: Static context dict.
            auto_strict: Auto-detect non-string params as context.
            similarity_threshold: Minimum similarity score (overrides config).
            allow_errors: If False (default), don't cache error results.
            batch_mode: Use batch operations internally (default: True).

        Returns:
            Decorator function.
        """
        return self._cached_decorator(
            query=query,
            mode=mode,
            context=context,
            static_context=static_context,
            auto_strict=auto_strict,
            similarity_threshold=similarity_threshold,
            allow_errors=allow_errors,
            batch_mode=batch_mode,
        )
