"""Tests for memora.decorators."""

import pytest
from memora import create_cached_decorator, MemoraDecorator, Memora, CacheConfig


# ============================================================================
# LOCAL SESSION FIXTURES (solo para este archivo)
# ============================================================================


@pytest.fixture(scope="module")
def memora_session_local():
    """Memora instance for this test module only."""
    config = CacheConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
    )
    return Memora(config)


@pytest.fixture
def memora_memory(memora_session_local):
    """Clean Memora for each test."""
    memora_session_local.clear()
    yield memora_session_local


# ============================================================================
# TESTS
# ============================================================================


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

        @cached(static_context={"agent": "test"})
        def compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"result: {query} | {param}"

        # First call (executes function)
        result1 = compute("What is machine learning?", param=42)
        assert call_count == 1
        assert "machine learning" in result1

        # Second call (uses cache)
        result2 = compute("What is machine learning?", param=42)
        assert call_count == 1
        assert result2 == result1

    def test_different_queries_no_cache(self, memora_memory):
        """Different queries should execute function."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(static_context={"agent": "test"})
        def compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"{query}-{param}"

        result1 = compute("How does AI work?", param=1)
        result2 = compute("What is deep learning?", param=1)

        assert call_count == 2
        assert result1 != result2

    def test_strict_params(self, memora_memory):
        """strict_params should enforce exact matching."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(static_context={"agent": "test"}, strict_params=["model"])
        def compute(query: str, model: str, temperature: float):
            nonlocal call_count
            call_count += 1
            return f"{query}|{model}|{temperature}"

        # First call with model=gpt-4
        result1 = compute("hello world", model="gpt-4", temperature=0.7)
        assert call_count == 1

        # Second call with same query and model (cache hit)
        result2 = compute("hello world", model="gpt-4", temperature=0.9)
        assert call_count == 1  # Cache hit (temperature not strict)
        assert result1 == result2

        # Third call with different model (cache miss)
        result3 = compute("hello world", model="claude", temperature=0.7)
        assert call_count == 2  # New call due to different model
        assert result1 != result3

    def test_auto_strict(self, memora_memory):
        """auto_strict should detect non-string params."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(query_param="prompt", auto_strict=True)
        def generate(prompt: str, temperature: float, max_tokens: int):
            nonlocal call_count
            call_count += 1
            return f"{prompt}|{temperature}|{max_tokens}"

        # First call
        result1 = generate("hello world", temperature=0.7, max_tokens=100)
        assert call_count == 1

        # Same prompt, different params (cache miss - params are strict)
        result2 = generate("hello world", temperature=0.8, max_tokens=100)
        assert call_count == 2

        # Same everything (cache hit)
        result3 = generate("hello world", temperature=0.8, max_tokens=100)
        assert call_count == 2
        assert result2 == result3

    def test_custom_query_param(self, memora_memory):
        """Custom query_param should work."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(static_context={"agent": "test"}, query_param="prompt")
        def generate(prompt: str, style: str):
            nonlocal call_count
            call_count += 1
            return f"Generated: {prompt} in {style}"

        result1 = generate("write a poem about nature", style="haiku")
        result2 = generate("write a poem about nature", style="haiku")

        assert result1 == result2
        assert call_count == 1  # Second call hit cache

    def test_invalid_query_param(self, memora_memory):
        """Invalid query_param should raise error."""
        cached = create_cached_decorator(memora_memory)

        with pytest.raises(ValueError, match="not found"):

            @cached(static_context={"agent": "test"}, query_param="nonexistent")
            def compute(query: str):
                return "result"


class TestAsyncFunctions:
    """Tests with asynchronous functions."""

    @pytest.mark.asyncio
    async def test_async_basic_caching(self, memora_memory):
        """Decorator should cache async functions."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(static_context={"agent": "async_test"})
        async def async_compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"async result: {query} | {param}"

        # First call
        result1 = await async_compute("What is Python?", param=42)
        assert call_count == 1

        # Second call (cache)
        result2 = await async_compute("What is Python?", param=42)
        assert call_count == 1
        assert result2 == result1

    @pytest.mark.asyncio
    async def test_async_with_defaults(self, memora_memory):
        """Async with default values should work."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(static_context={"agent": "test"})
        async def fetch_data(query: str, limit: int = 10):
            nonlocal call_count
            call_count += 1
            return f"fetched {limit} items for {query}"

        result1 = await fetch_data("search for articles", limit=10)
        result2 = await fetch_data("search for articles")  # Uses default

        assert result1 == result2
        assert call_count == 1  # Cache hit


class TestContextHandling:
    """Context handling tests."""

    def test_static_context_only(self, memora_memory):
        """Static context with no strict params - query is semantic key."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(static_context={"agent": "static", "version": "v1"})
        def compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"{query}-{call_count}"

        # Same query, different param (cache hit because param not in context)
        result1 = compute("Explain quantum computing", param=1)
        result2 = compute("Explain quantum computing", param=2)

        # Should be same result (cache hit - param doesn't affect caching)
        assert result1 == result2
        assert result1 == "Explain quantum computing-1"
        assert call_count == 1

    def test_complex_strict_param(self, memora_memory):
        """Complex types in strict_params should serialize correctly."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(strict_params=["tools"])
        def call_agent(query: str, tools: list):
            nonlocal call_count
            call_count += 1
            return f"call {call_count}"

        tools1 = [{"name": "search"}, {"name": "calc"}]
        tools2 = [{"name": "search"}, {"name": "calc"}]
        tools3 = [{"name": "calc"}, {"name": "search"}]

        # Use a longer, more semantic query
        query = "What is the weather today in San Francisco?"

        result1 = call_agent(query, tools=tools1)
        assert call_count == 1

        result2 = call_agent(query, tools=tools2)
        assert call_count == 1  # Cache hit - same tools
        assert result1 == result2

        result3 = call_agent(query, tools=tools3)
        assert call_count == 2  # Cache miss - different order
        assert result1 != result3

    def test_no_context_uses_function_name(self, memora_memory):
        """No context should use __function__ as default."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached()  # No context at all
        def compute(query: str):
            nonlocal call_count
            call_count += 1
            return f"result: {query}"

        result1 = compute("How does blockchain work?")
        result2 = compute("How does blockchain work?")

        assert result1 == result2
        assert call_count == 1


class TestComplexResults:
    """Tests with complex result types."""

    def test_dict_result(self, memora_memory):
        """Dict results should be cached."""
        cached = create_cached_decorator(memora_memory)

        @cached(static_context={"agent": "test"})
        def get_data(query: str):
            return {"status": "ok", "data": [1, 2, 3]}

        result1 = get_data("Fetch user data")
        result2 = get_data("Fetch user data")

        assert result1 == result2
        assert isinstance(result1, dict)

    def test_list_result(self, memora_memory):
        """List results should be cached."""
        cached = create_cached_decorator(memora_memory)

        @cached(static_context={"agent": "test"})
        def get_items(query: str):
            return [1, 2, 3, 4, 5]

        result1 = get_items("Get list of items")
        result2 = get_items("Get list of items")

        assert result1 == result2
        assert isinstance(result1, list)

    def test_nested_structures(self, memora_memory):
        """Nested structures should be cached."""
        cached = create_cached_decorator(memora_memory)

        @cached(static_context={"agent": "test"})
        def complex_data(query: str):
            return {
                "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "meta": {"count": 2},
            }

        result1 = complex_data("Get all users")
        result2 = complex_data("Get all users")

        assert result1 == result2


class TestEdgeCases:
    """Edge case tests with decorators."""

    def test_no_context(self, memora_memory):
        """Decorator without context should work."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached()
        def compute(query: str):
            nonlocal call_count
            call_count += 1
            return f"result: {query}"

        result1 = compute("Explain neural networks")
        result2 = compute("Explain neural networks")

        assert result1 == result2
        assert call_count == 1

    def test_function_metadata_preserved(self, memora_memory):
        """Function metadata should be preserved (functools.wraps)."""
        cached = create_cached_decorator(memora_memory)

        @cached(static_context={"agent": "test"})
        def my_function(query: str):
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_none_values_excluded(self, memora_memory):
        """None values should be excluded from strict params."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(strict_params=["optional"])
        def compute(query: str, optional: str = None):
            nonlocal call_count
            call_count += 1
            return query

        result1 = compute("What is AI?", optional=None)
        result2 = compute("What is AI?")

        # Should hit cache (both have optional=None)
        assert result1 == result2
        assert call_count == 1
