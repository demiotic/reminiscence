"""Tests for error result handling in cache."""

import pytest


class TestErrorDetection:
    """Test _is_error_result detection logic."""

    def test_detects_exception_objects(self, cache_ops):
        """Should detect Exception instances as errors."""
        assert cache_ops._is_error_result(ValueError("test error"))
        assert cache_ops._is_error_result(Exception("generic error"))
        assert cache_ops._is_error_result(RuntimeError("runtime"))

    def test_detects_dict_with_error_keys(self, cache_ops):
        """Should detect dicts with error-related keys."""
        assert cache_ops._is_error_result({"error": "something failed"})
        assert cache_ops._is_error_result({"exception": "ValueError"})
        assert cache_ops._is_error_result({"error_message": "timeout"})
        assert cache_ops._is_error_result({"traceback": "..."})
        assert cache_ops._is_error_result({"failed": True})

    def test_detects_error_strings(self, cache_ops):
        """Should detect strings starting with error patterns."""
        assert cache_ops._is_error_result("error: connection failed")
        assert cache_ops._is_error_result("Error: timeout")
        assert cache_ops._is_error_result("exception: ValueError")
        assert cache_ops._is_error_result("traceback: ...")
        assert cache_ops._is_error_result("failed: API unavailable")

    def test_detects_none_as_error(self, cache_ops):
        """Should treat None as potential error."""
        assert cache_ops._is_error_result(None)

    def test_does_not_flag_valid_results(self, cache_ops):
        """Should NOT flag valid results as errors."""
        # Valid results
        assert not cache_ops._is_error_result("valid response")
        assert not cache_ops._is_error_result({"data": "result"})
        assert not cache_ops._is_error_result({"success": True, "data": [1, 2, 3]})
        assert not cache_ops._is_error_result([1, 2, 3])
        assert not cache_ops._is_error_result(42)
        assert not cache_ops._is_error_result(True)

        # Edge case: string containing "error" but not starting with it
        assert not cache_ops._is_error_result("this contains error word")


class TestStoreErrorValidation:
    """Test store() error validation."""

    def test_store_skips_errors_by_default(self, reminiscence):
        """store() should skip error results by default."""
        # Try to store an error
        reminiscence.store("query", {}, {"error": "failed"})

        # Should not be cached - verify via lookup
        result = reminiscence.lookup("query", {})
        assert result.is_miss

        # Metrics should track the skip
        assert reminiscence.metrics.store_errors == 1

    def test_store_skips_none_by_default(self, reminiscence):
        """store() should skip None results by default."""
        reminiscence.store("query", {}, None)

        result = reminiscence.lookup("query", {})
        assert result.is_miss
        assert reminiscence.metrics.store_errors == 1

    def test_store_skips_exceptions_by_default(self, reminiscence):
        """store() should skip Exception objects by default."""
        reminiscence.store("query", {}, ValueError("test"))

        result = reminiscence.lookup("query", {})
        assert result.is_miss
        assert reminiscence.metrics.store_errors == 1

    def test_store_allows_errors_when_explicit(self, reminiscence):
        """store() should cache errors when allow_errors=True."""
        reminiscence.store("query", {}, {"error": "failed"}, allow_errors=True)

        # Can retrieve it
        result = reminiscence.lookup("query", {})
        assert result.is_hit
        assert result.result == {"error": "failed"}

    def test_store_caches_valid_results(self, reminiscence):
        """store() should cache valid results normally."""
        reminiscence.store("query", {}, {"data": "valid"})

        result = reminiscence.lookup("query", {})
        assert result.is_hit
        assert result.result == {"data": "valid"}


class TestStoreBatchErrorValidation:
    """Test store_batch() error filtering."""

    def test_store_batch_filters_errors(self, reminiscence):
        """store_batch() should filter out errors by default."""
        queries = ["q1", "q2", "q3", "q4"]
        contexts = [{}, {}, {}, {}]
        results = [
            {"data": "valid1"},  # OK
            {"error": "failed"},  # Error - skip
            {"data": "valid2"},  # OK
            None,  # Error - skip
        ]

        reminiscence.store_batch(
            queries, contexts, results, query_mode="exact"
        )  # ← EXACT

        # Only 2 valid entries - check via lookups
        assert reminiscence.lookup("q1", {}, query_mode="exact").is_hit
        assert reminiscence.lookup(
            "q2", {}, query_mode="exact"
        ).is_miss  # Error skipped
        assert reminiscence.lookup("q3", {}, query_mode="exact").is_hit
        assert reminiscence.lookup("q4", {}, query_mode="exact").is_miss  # None skipped

        # Errors tracked
        assert reminiscence.metrics.store_errors == 2

    def test_store_batch_with_allow_errors(self, reminiscence):
        """store_batch() should store all when allow_errors=True."""
        queries = ["q1", "q2"]
        contexts = [{}, {}]
        results = [
            {"data": "valid"},
            {"error": "failed"},
        ]

        reminiscence.store_batch(queries, contexts, results, allow_errors=True)

        # Both stored
        assert reminiscence.lookup("q1", {}).is_hit
        assert reminiscence.lookup("q2", {}).is_hit

    def test_store_batch_all_errors_skipped(self, reminiscence):
        """store_batch() should handle all-errors case gracefully."""
        queries = ["q1", "q2", "q3"]
        contexts = [{}, {}, {}]
        results = [
            {"error": "failed1"},
            None,
            {"exception": "timeout"},
        ]

        reminiscence.store_batch(queries, contexts, results)

        # Nothing stored
        assert reminiscence.lookup("q1", {}).is_miss
        assert reminiscence.lookup("q2", {}).is_miss
        assert reminiscence.lookup("q3", {}).is_miss
        assert reminiscence.metrics.store_errors == 3


class TestDecoratorErrorHandling:
    """Test decorator error handling."""

    def test_decorator_does_not_cache_exceptions(self, reminiscence):
        """Decorator should not cache raised exceptions."""
        from reminiscence.decorators import create_cached_decorator

        cached = create_cached_decorator(reminiscence)

        @cached(query="prompt")
        def failing_function(prompt: str):
            raise ValueError("API failed")

        # First call raises
        with pytest.raises(ValueError, match="API failed"):
            failing_function("test prompt")

        # Nothing cached - verify via lookup
        result = reminiscence.lookup(
            "test prompt", {"__function__": "failing_function"}
        )
        assert result.is_miss

        # Second call still raises (not cached)
        with pytest.raises(ValueError, match="API failed"):
            failing_function("test prompt")

    def test_decorator_skips_error_results(self, reminiscence):
        """Decorator should skip caching error results by default."""
        from reminiscence.decorators import create_cached_decorator

        cached = create_cached_decorator(reminiscence)

        @cached(query="prompt")
        def returns_error(prompt: str):
            return {"error": "rate limited"}

        result = returns_error("test prompt")

        # Function executed
        assert result == {"error": "rate limited"}

        # But NOT cached
        lookup = reminiscence.lookup("test prompt", {"__function__": "returns_error"})
        assert lookup.is_miss

    def test_decorator_caches_with_allow_errors(self, reminiscence):
        """Decorator should cache errors when allow_errors=True."""
        from reminiscence.decorators import create_cached_decorator

        cached = create_cached_decorator(reminiscence)

        @cached(query="prompt", allow_errors=True)
        def returns_error(prompt: str):
            return {"error": "rate limited"}

        result = returns_error("test prompt")

        # Verify it's cached
        lookup = reminiscence.lookup("test prompt", {"__function__": "returns_error"})
        assert lookup.is_hit
        assert lookup.result == {"error": "rate limited"}

    def test_decorator_caches_valid_results(self, reminiscence):
        """Decorator should cache valid results normally."""
        from reminiscence.decorators import create_cached_decorator

        cached = create_cached_decorator(reminiscence)

        call_count = 0

        @cached(query="prompt")
        def valid_function(prompt: str):
            nonlocal call_count
            call_count += 1
            return {"data": "success"}

        # First call
        result1 = valid_function("test prompt")
        assert result1 == {"data": "success"}
        assert call_count == 1

        # Second call (cached)
        result2 = valid_function("test prompt")
        assert result2 == {"data": "success"}
        assert call_count == 1  # Not called again


class TestAsyncDecoratorErrorHandling:
    """Test async decorator error handling."""

    @pytest.mark.asyncio
    async def test_async_decorator_does_not_cache_exceptions(self, reminiscence):
        """Async decorator should not cache raised exceptions."""
        from reminiscence.decorators import create_cached_decorator

        cached = create_cached_decorator(reminiscence)

        @cached(query="prompt")
        async def failing_async_function(prompt: str):
            raise ValueError("Async API failed")

        # First call raises
        with pytest.raises(ValueError, match="Async API failed"):
            await failing_async_function("test prompt")

        # Nothing cached
        result = reminiscence.lookup(
            "test prompt", {"__function__": "failing_async_function"}
        )
        assert result.is_miss

    @pytest.mark.asyncio
    async def test_async_decorator_skips_error_results(self, reminiscence):
        """Async decorator should skip caching error results."""
        from reminiscence.decorators import create_cached_decorator

        cached = create_cached_decorator(reminiscence)

        @cached(query="prompt")
        async def async_returns_error(prompt: str):
            return {"error": "timeout"}

        result = await async_returns_error("test prompt")

        assert result == {"error": "timeout"}

        lookup = reminiscence.lookup(
            "test prompt", {"__function__": "async_returns_error"}
        )
        assert lookup.is_miss
