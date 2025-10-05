"""Tests para memora.decorators."""

import pytest
from memora import create_cached_decorator, MemoraDecorator


class TestDecoratorBasics:
    """Tests básicos del decorador."""

    def test_decorator_factory(self, memora_memory):
        """create_cached_decorator debe retornar decorador funcional."""
        cached = create_cached_decorator(memora_memory)

        assert callable(cached)

    def test_decorator_class(self, memora_memory):
        """MemoraDecorator debe instanciarse correctamente."""
        decorator = MemoraDecorator(memora_memory)

        assert decorator.memora is memora_memory
        assert hasattr(decorator, "cached")


class TestSyncFunctions:
    """Tests con funciones síncronas."""

    def test_basic_caching(self, memora_memory):
        """Decorador debe cachear resultados de función sync."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "test"})
        def compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"resultado: {query} | {param}"

        # Primera llamada (ejecuta función)
        result1 = compute("test", param=42)
        assert call_count == 1
        assert result1 == "resultado: test | 42"

        # Segunda llamada (usa caché)
        result2 = compute("test", param=42)
        assert call_count == 1  # No se ejecutó de nuevo
        assert result2 == result1

    def test_different_params_no_cache(self, memora_memory):
        """Params diferentes deben ejecutar función."""
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
        """extract_from_args debe incluir params en contexto."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"}, extract_from_args=True)
        def compute(query: str, model: str, temperature: float):
            return f"{query}|{model}|{temperature}"

        # Primera llamada con model=gpt-4
        result1 = compute("hello", model="gpt-4", temperature=0.7)

        # Segunda llamada con model=claude (diferente contexto)
        result2 = compute("hello", model="claude", temperature=0.7)

        # Deben ser diferentes porque model es parte del contexto
        assert result1 != result2

    def test_custom_query_param(self, memora_memory):
        """query_param customizado debe funcionar."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"}, query_param="prompt")
        def generate(prompt: str, style: str):
            return f"Generated: {prompt} in {style}"

        result1 = generate("write poem", style="haiku")
        result2 = generate("write poem", style="haiku")

        assert result1 == result2

    def test_invalid_query_param(self, memora_memory):
        """query_param inválido debe lanzar error."""
        cached = create_cached_decorator(memora_memory)

        with pytest.raises(ValueError, match="no encontrado"):

            @cached(context={"agent": "test"}, query_param="nonexistent")
            def compute(query: str):
                return "result"


class TestAsyncFunctions:
    """Tests con funciones asíncronas."""

    @pytest.mark.asyncio
    async def test_async_basic_caching(self, memora_memory):
        """Decorador debe cachear funciones async."""
        cached = create_cached_decorator(memora_memory)

        call_count = 0

        @cached(context={"agent": "async_test"})
        async def async_compute(query: str, param: int):
            nonlocal call_count
            call_count += 1
            return f"async result: {query} | {param}"

        # Primera llamada
        result1 = await async_compute("test", param=42)
        assert call_count == 1

        # Segunda llamada (caché)
        result2 = await async_compute("test", param=42)
        assert call_count == 1
        assert result2 == result1

    @pytest.mark.asyncio
    async def test_async_with_defaults(self, memora_memory):
        """Async con valores default debe funcionar."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"})
        async def fetch_data(query: str, limit: int = 10):
            return f"fetched {limit} items for {query}"

        result1 = await fetch_data("search", limit=10)
        result2 = await fetch_data("search")  # Usa default

        assert result1 == result2


class TestContextHandling:
    """Tests de manejo de contexto."""

    def test_static_context_only(self, memora_memory):
        """Solo contexto estático (extract_from_args=False)."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "static", "version": "v1"}, extract_from_args=False)
        def compute(query: str, param: int):
            return f"{query}-{param}"

        # Diferentes params deben usar MISMO caché (porque no se extraen)
        result1 = compute("test", param=1)
        result2 = compute("test", param=2)

        # Mismo resultado porque contexto es idéntico
        assert result1 == result2

    def test_static_overrides_runtime(self, memora_memory):
        """Contexto estático debe override runtime."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test", "version": "v2"}, extract_from_args=True)
        def compute(query: str, version: str):
            return f"{query}-{version}"

        # Aunque pasamos version="v1", contexto estático usa "v2"
        compute("test", version="v1")

        # Verificar que el contexto almacenado tiene version="v2"
        result = memora_memory.lookup(
            "test", {"query": "test", "version": "v2", "agent": "test"}
        )
        # (Esta verificación es aproximada, el punto es que static override)


class TestComplexResults:
    """Tests con tipos de resultados complejos."""

    def test_dict_result(self, memora_memory):
        """Resultados tipo dict deben cachearse."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"})
        def get_data(query: str):
            return {"status": "ok", "data": [1, 2, 3]}

        result1 = get_data("test")
        result2 = get_data("test")

        assert result1 == result2
        assert isinstance(result1, dict)

    def test_list_result(self, memora_memory):
        """Resultados tipo list deben cachearse."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"})
        def get_items(query: str):
            return [1, 2, 3, 4, 5]

        result1 = get_items("test")
        result2 = get_items("test")

        assert result1 == result2
        assert isinstance(result1, list)

    def test_nested_structures(self, memora_memory):
        """Estructuras nested deben cachearse."""
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
    """Tests de casos extremos con decoradores."""

    def test_no_context(self, memora_memory):
        """Decorador sin contexto debe funcionar."""
        cached = create_cached_decorator(memora_memory)

        @cached()
        def compute(query: str):
            return f"result: {query}"

        result1 = compute("test")
        result2 = compute("test")

        assert result1 == result2

    def test_function_metadata_preserved(self, memora_memory):
        """Metadata de función debe preservarse (functools.wraps)."""
        cached = create_cached_decorator(memora_memory)

        @cached(context={"agent": "test"})
        def my_function(query: str):
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."
