"""Tests for memora.decorators."""

import pytest
from memora import create_cached_decorator, MemoraDecorator


class TestDecoratorBasics:
    """Basic decorator tests."""

    def test_decorator_factory(self, memora_memory):
        """create_cached_decorator should return functional decorator."""
        cached = create_cached_decorator(memora_memory)
        assert callable(cached)

    def test_decorator_class(self, memora_memory):
        """MemoraDecorator should instantiate correctly."""
        decorator = MemoraDecorator(memora_memory)
        assert decorator.memora is memora_memory
        assert hasattr(decorator, "cached")


class TestSyncFunctions:
    """Tests with synchronous functions."""

    def test_basic_caching(self, memora_memory):
        """Decorator should cache sync function results."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "test"})
        def compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"result: {query} | {param}"

        # First call (executes function)
        result1 = compute("test", param=42)
        assert call_count == 1
        assert result1 == "result: test | 42"

        # Second call (uses cache)
        result2 = compute("test", param=42)
        assert call_count == 1
        assert result2 == result1

    def test_different_params_no_cache(self, memora_memory):
        """Different params should execute function."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "test"})
        def compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"{query}-{param}"

        result1 = compute("test", param=1)
        result2 = compute("test", param=2)

        assert call_count == 2
        assert result1 != result2

    def test_extract_from_args(self, memora_memory):
        """extract_from_args should include params in context."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "test"}, extract_from_args=True)
        def compute(query: str, model: str, temperature: float):
            nonlocal call_count
            call_count += 1
            return f"{query}|{model}|{temperature}"

        # First call with model=gpt-4
        result1 = compute("hello", model="gpt-4", temperature=0.7)
        assert call_count == 1

        # Second call with model=claude (different context)
        result2 = compute("hello", model="claude", temperature=0.7)
        assert call_count == 2  # New call due to different model

        # Should be different because model is part of context
        assert result1 != result2

    def test_custom_query_param(self, memora_memory):
        """Custom query_param should work."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "test"}, query_param="prompt")
        def generate(prompt: str, style: str):
            nonlocal call_count
            call_count += 1
            return f"Generated: {prompt} in {style}"

        result1 = generate("write poem", style="haiku")
        result2 = generate("write poem", style="haiku")

        assert result1 == result2
        assert call_count == 1  # Second call hit cache

    def test_invalid_query_param(self, memora_memory):
        """Invalid query_param should raise error."""
        cached = create_cached_decorator(memora_memory)

        with pytest.raises(ValueError, match="not found"):

            @cached(context={"agent": "test"}, query_param="nonexistent")
            def compute(query: str):
                return "result"


class TestAsyncFunctions:
    """Tests with asynchronous functions."""

    @pytest.mark.asyncio
    async def test_async_basic_caching(self, memora_memory):
        """Decorator should cache async functions."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "async_test"})
        async def async_compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"async result: {query} | {param}"

        # First call
        result1 = await async_compute("test", param=42)
        assert call_count == 1

        # Second call (cache)
        result2 = await async_compute("test", param=42)
        assert call_count == 1
        assert result2 == result1

    @pytest.mark.asyncio
    async def test_async_with_defaults(self, memora_memory):
        """Async with default values should work."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "test"})
        async def fetch_data(query: str, limit: int = 10):
            nonlocal call_count
            call_count += 1
            return f"fetched {limit} items for {query}"

        result1 = await fetch_data("search", limit=10)
        result2 = await fetch_data("search")  # Uses default

        assert result1 == result2
        assert call_count == 1  # Cache hit


class TestContextHandling:
    """Context handling tests."""

    def test_static_context_only(self, memora_memory):
        """Static context only (extract_from_args=False) - explicit opt-out."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "static", "version": "v1"}, extract_from_args=False)
        def compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"{query}-{param}"

        result1 = compute("test", param=1)
        result2 = compute("test", param=2)

        assert result1 == result2
        assert call_count == 1

    def test_static_overrides_runtime(self, memora_memory):
        """Static context should override extracted params."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "test", "version": "v2"}, extract_from_args=True)
        def compute(query: str, version: str):
            nonlocal call_count
            call_count += 1
            return f"{query}-{version}"

        # Although we pass version="v1", static context has "v2"
        result1 = compute("test", version="v1")
        result2 = compute("test", version="v3")  # Different runtime version

        # Same result because version="v2" (static) overrides both
        assert result1 == result2
        assert call_count == 1

    def test_extract_with_no_static(self, memora_memory):
        """extract_from_args without static context should work."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(extract_from_args=True)  # No static context
        def compute(query: str, model: str):
            nonlocal call_count
            call_count += 1
            return f"{query}-{model}"

        result1 = compute("test", model="gpt-4")
        result2 = compute("test", model="gpt-4")
        result3 = compute("test", model="claude")

        assert result1 == result2  # Same context
        assert result1 != result3  # Different context
        assert call_count == 2  # 2 different contexts


class TestComplexResults:
    """Tests with complex result types."""

    def test_dict_result(self, memora_memory):
        """Dict results should be cached."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"})
        def get_data(query: str):
            return {"status": "ok", "data": [1, 2, 3]}

        result1 = get_data("test")
        result2 = get_data("test")

        assert result1 == result2
        assert isinstance(result1, dict)

    def test_list_result(self, memora_memory):
        """List results should be cached."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"})
        def get_items(query: str):
            return [1, 2, 3, 4, 5]

        result1 = get_items("test")
        result2 = get_items("test")

        assert result1 == result2
        assert isinstance(result1, list)

    def test_nested_structures(self, memora_memory):
        """Nested structures should be cached."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"})
        def complex_data(query: str):
            return {
                "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "meta": {"count": 2},
            }

        result1 = complex_data("test")
        result2 = complex_data("test")

        assert result1 == result2


class TestEdgeCases:
    """Edge case tests with decorators."""

    def test_no_context(self, memora_memory):
        """Decorator without context should work."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached()  # No context, no extract_from_args
        def compute(query: str):
            nonlocal call_count
            call_count += 1
            return f"result: {query}"

        result1 = compute("test")
        result2 = compute("test")

        assert result1 == result2
        assert call_count == 1

    def test_function_metadata_preserved(self, memora_memory):
        """Function metadata should be preserved (functools.wraps)."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"})
        def my_function(query: str):
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_none_values_excluded(self, memora_memory):
        """None values should be excluded from extracted context."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(extract_from_args=True)
        def compute(query: str, optional: str = None):
            nonlocal call_count
            call_count += 1
            return query

        result1 = compute("test", optional=None)
        result2 = compute("test")

        # Should hit cache (both have optional=None, excluded)
        assert result1 == result2
        assert call_count == 1
